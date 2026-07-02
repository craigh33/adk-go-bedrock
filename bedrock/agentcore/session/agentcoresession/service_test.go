package agentcoresession

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime/types"
	"github.com/google/uuid"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/sessiontestsuite"
	"google.golang.org/genai"
)

func TestServiceSuite(t *testing.T) {
	sessiontestsuite.RunServiceTests(t, sessiontestsuite.SuiteOptions{
		SupportsUserProvidedSessionID: false,
	}, func(t *testing.T) session.Service {
		t.Helper()
		svc, err := NewWithAPI(newFakeAPI(), nil)
		if err != nil {
			t.Fatal(err)
		}
		return svc
	})
}

func TestAppendEventStoresFullEventJSON(t *testing.T) {
	ctx := context.Background()
	api := newFakeAPI()
	svc, err := NewWithAPI(api, nil)
	if err != nil {
		t.Fatal(err)
	}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatal(err)
	}

	event := &session.Event{
		ID:           "event-1",
		InvocationID: "not-a-uuid",
		Author:       "assistant",
		Branch:       "root.child",
		Timestamp:    time.Date(2026, 1, 2, 3, 4, 5, 6, time.UTC),
		Actions: session.EventActions{
			StateDelta: map[string]any{
				"k":      "v",
				"temp:x": "skip",
			},
		},
		LLMResponse: model.LLMResponse{
			Content:        genai.NewContentFromText("hello", genai.RoleModel),
			CustomMetadata: map[string]any{"custom": "value"},
		},
	}
	if err := svc.AppendEvent(ctx, created.Session, event); err != nil {
		t.Fatal(err)
	}

	got, err := svc.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatal(err)
	}
	events := sessiontestsuite.Snapshot(got.Session).Events
	if len(events) != 1 {
		t.Fatalf("events len = %d, want 1", len(events))
	}
	if events[0].InvocationID != "not-a-uuid" ||
		events[0].Branch != "root.child" ||
		events[0].Content.Parts[0].Text != "hello" {
		t.Fatalf("event not round-tripped: %+v", events[0])
	}
	if _, ok := events[0].Actions.StateDelta["temp:x"]; ok {
		t.Fatalf("temp state persisted: %+v", events[0].Actions.StateDelta)
	}

	api.mu.Lock()
	defer api.mu.Unlock()
	var invocationIDs []string
	for id := range api.sessions[created.Session.ID()].invocations {
		invocationIDs = append(invocationIDs, id)
	}
	if len(invocationIDs) != 1 {
		t.Fatalf("invocation count = %d, want 1", len(invocationIDs))
	}
	if _, err := uuid.Parse(invocationIDs[0]); err != nil {
		t.Fatalf("invocation id is not a UUID: %q", invocationIDs[0])
	}
}

func TestDeleteEndsBeforeDeleteAndMissingIsOK(t *testing.T) {
	ctx := context.Background()
	api := newFakeAPI()
	svc, err := NewWithAPI(api, nil)
	if err != nil {
		t.Fatal(err)
	}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatal(err)
	}

	err = svc.Delete(ctx, &session.DeleteRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatal(err)
	}
	if got := api.calls; len(got) < 2 || got[len(got)-2] != "EndSession" || got[len(got)-1] != "DeleteSession" {
		t.Fatalf("delete call order = %v", got)
	}

	err = svc.Delete(ctx, &session.DeleteRequest{AppName: "app", UserID: "user", SessionID: created.Session.ID()})
	if err != nil {
		t.Fatalf("missing delete error = %v, want nil", err)
	}
}

func TestWrongUserCannotAppend(t *testing.T) {
	ctx := context.Background()
	api := newFakeAPI()
	svc, err := NewWithAPI(api, nil)
	if err != nil {
		t.Fatal(err)
	}
	created, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user1"})
	if err != nil {
		t.Fatal(err)
	}
	wrong := newLocalSession("app", "user2", created.Session.ID(), nil, nil, nil, time.Now())
	err = svc.AppendEvent(ctx, wrong, &session.Event{ID: "event-1", InvocationID: "inv-1", Author: "user"})
	if err == nil {
		t.Fatal("expected wrong-user append error")
	}
}

func TestAppendEventDoesNotRepublishStaleSharedState(t *testing.T) {
	ctx := context.Background()
	api := newFakeAPI()
	svc, err := NewWithAPI(api, nil)
	if err != nil {
		t.Fatal(err)
	}

	oldOwner, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app",
		UserID:  "user1",
		State:   map[string]any{"app:k": "v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	newOwner, err := svc.Create(ctx, &session.CreateRequest{
		AppName: "app",
		UserID:  "user2",
		State:   map[string]any{"app:k": "v2"},
	})
	if err != nil {
		t.Fatal(err)
	}

	oldOwnerReloaded, err := svc.Get(ctx, &session.GetRequest{
		AppName:   "app",
		UserID:    "user1",
		SessionID: oldOwner.Session.ID(),
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := oldOwnerReloaded.Session.State().Get("app:k")
	if err != nil {
		t.Fatal(err)
	}
	if got != "v2" {
		t.Fatalf("reloaded app state = %v, want v2", got)
	}

	err = svc.AppendEvent(ctx, oldOwnerReloaded.Session, &session.Event{
		ID:           "event-1",
		InvocationID: "inv-1",
		Actions:      session.EventActions{StateDelta: map[string]any{"local": "only"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	err = svc.AppendEvent(ctx, newOwner.Session, &session.Event{
		ID:           "event-2",
		InvocationID: "inv-2",
		Actions:      session.EventActions{StateDelta: map[string]any{"user:k": "uv"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	fresh, err := svc.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user3"})
	if err != nil {
		t.Fatal(err)
	}
	got, err = fresh.Session.State().Get("app:k")
	if err != nil {
		t.Fatal(err)
	}
	if got != "v2" {
		t.Fatalf("shared app state = %v, want v2", got)
	}
}

type fakeAPI struct {
	mu       sync.Mutex
	clock    time.Time
	sessions map[string]*fakeSession
	calls    []string
}

type fakeSession struct {
	id          string
	metadata    map[string]string
	createdAt   time.Time
	updatedAt   time.Time
	ended       bool
	invocations map[string]*fakeInvocation
}

type fakeInvocation struct {
	id        string
	createdAt time.Time
	steps     map[string]*brtypes.InvocationStep
	order     []string
}

func newFakeAPI() *fakeAPI {
	return &fakeAPI{
		clock:    time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		sessions: map[string]*fakeSession{},
	}
}

func (f *fakeAPI) tick() time.Time {
	f.clock = f.clock.Add(time.Second)
	return f.clock
}

func (f *fakeAPI) call(name string) {
	f.calls = append(f.calls, name)
}

func (f *fakeAPI) CreateSession(
	ctx context.Context,
	in *bedrockagentruntime.CreateSessionInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.CreateSessionOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("CreateSession")
	now := f.tick()
	id := uuid.NewString()
	f.sessions[id] = &fakeSession{
		id:          id,
		metadata:    mapsClone(in.SessionMetadata),
		createdAt:   now,
		updatedAt:   now,
		invocations: map[string]*fakeInvocation{},
	}
	return &bedrockagentruntime.CreateSessionOutput{
		CreatedAt:     &now,
		SessionId:     &id,
		SessionStatus: brtypes.SessionStatusActive,
	}, nil
}

func (f *fakeAPI) GetSession(
	ctx context.Context,
	in *bedrockagentruntime.GetSessionInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.GetSessionOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("GetSession")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	return &bedrockagentruntime.GetSessionOutput{
		CreatedAt:       &sess.createdAt,
		LastUpdatedAt:   &sess.updatedAt,
		SessionId:       &sess.id,
		SessionMetadata: mapsClone(sess.metadata),
		SessionStatus:   brtypes.SessionStatusActive,
	}, nil
}

func (f *fakeAPI) UpdateSession(
	ctx context.Context,
	in *bedrockagentruntime.UpdateSessionInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.UpdateSessionOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("UpdateSession")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	now := f.tick()
	sess.metadata = mapsClone(in.SessionMetadata)
	sess.updatedAt = now
	return &bedrockagentruntime.UpdateSessionOutput{
		CreatedAt:     &sess.createdAt,
		LastUpdatedAt: &now,
		SessionId:     &sess.id,
		SessionStatus: brtypes.SessionStatusActive,
	}, nil
}

func (f *fakeAPI) ListSessions(
	ctx context.Context,
	in *bedrockagentruntime.ListSessionsInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.ListSessionsOutput, error) {
	_ = ctx
	_ = in
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("ListSessions")
	var ids []string
	for id := range f.sessions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := &bedrockagentruntime.ListSessionsOutput{}
	for _, id := range ids {
		sess := f.sessions[id]
		out.SessionSummaries = append(out.SessionSummaries, brtypes.SessionSummary{
			CreatedAt:     &sess.createdAt,
			LastUpdatedAt: &sess.updatedAt,
			SessionId:     &sess.id,
			SessionStatus: brtypes.SessionStatusActive,
		})
	}
	return out, nil
}

func (f *fakeAPI) EndSession(
	ctx context.Context,
	in *bedrockagentruntime.EndSessionInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.EndSessionOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("EndSession")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	sess.ended = true
	return &bedrockagentruntime.EndSessionOutput{SessionId: &sess.id, SessionStatus: brtypes.SessionStatusEnded}, nil
}

func (f *fakeAPI) DeleteSession(
	ctx context.Context,
	in *bedrockagentruntime.DeleteSessionInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.DeleteSessionOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("DeleteSession")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	if !sess.ended {
		return nil, conflict("active session")
	}
	delete(f.sessions, sess.id)
	return &bedrockagentruntime.DeleteSessionOutput{}, nil
}

func (f *fakeAPI) CreateInvocation(
	ctx context.Context,
	in *bedrockagentruntime.CreateInvocationInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.CreateInvocationOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("CreateInvocation")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	id := uuid.NewString()
	if in.InvocationId != nil && *in.InvocationId != "" {
		id = *in.InvocationId
	}
	if _, exists := sess.invocations[id]; exists {
		return nil, conflict("invocation exists")
	}
	now := f.tick()
	sess.invocations[id] = &fakeInvocation{
		id:        id,
		createdAt: now,
		steps:     map[string]*brtypes.InvocationStep{},
	}
	return &bedrockagentruntime.CreateInvocationOutput{CreatedAt: &now, InvocationId: &id, SessionId: &sess.id}, nil
}

func (f *fakeAPI) ListInvocations(
	ctx context.Context,
	in *bedrockagentruntime.ListInvocationsInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.ListInvocationsOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("ListInvocations")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	var invs []*fakeInvocation
	for _, inv := range sess.invocations {
		invs = append(invs, inv)
	}
	sort.Slice(invs, func(i, j int) bool { return invs[i].createdAt.Before(invs[j].createdAt) })
	out := &bedrockagentruntime.ListInvocationsOutput{}
	for _, inv := range invs {
		out.InvocationSummaries = append(out.InvocationSummaries, brtypes.InvocationSummary{
			CreatedAt:    &inv.createdAt,
			InvocationId: &inv.id,
			SessionId:    &sess.id,
		})
	}
	return out, nil
}

func (f *fakeAPI) PutInvocationStep(
	ctx context.Context,
	in *bedrockagentruntime.PutInvocationStepInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.PutInvocationStepOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("PutInvocationStep")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	inv, ok := sess.invocations[value(in.InvocationIdentifier)]
	if !ok {
		return nil, notFound("invocation")
	}
	stepID := value(in.InvocationStepId)
	if stepID == "" {
		stepID = uuid.NewString()
	}
	stepTime := time.Time{}
	if in.InvocationStepTime != nil {
		stepTime = *in.InvocationStepTime
	}
	if _, exists := inv.steps[stepID]; !exists {
		inv.order = append(inv.order, stepID)
	}
	inv.steps[stepID] = &brtypes.InvocationStep{
		InvocationId:       &inv.id,
		InvocationStepId:   &stepID,
		InvocationStepTime: &stepTime,
		Payload:            in.Payload,
		SessionId:          &sess.id,
	}
	return &bedrockagentruntime.PutInvocationStepOutput{InvocationStepId: &stepID}, nil
}

func (f *fakeAPI) ListInvocationSteps(
	ctx context.Context,
	in *bedrockagentruntime.ListInvocationStepsInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.ListInvocationStepsOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("ListInvocationSteps")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	inv, ok := sess.invocations[value(in.InvocationIdentifier)]
	if !ok {
		return nil, notFound("invocation")
	}
	out := &bedrockagentruntime.ListInvocationStepsOutput{}
	for _, id := range inv.order {
		step := inv.steps[id]
		out.InvocationStepSummaries = append(out.InvocationStepSummaries, brtypes.InvocationStepSummary{
			InvocationId:       &inv.id,
			InvocationStepId:   &id,
			InvocationStepTime: step.InvocationStepTime,
			SessionId:          &sess.id,
		})
	}
	return out, nil
}

func (f *fakeAPI) GetInvocationStep(
	ctx context.Context,
	in *bedrockagentruntime.GetInvocationStepInput,
	optFns ...func(*bedrockagentruntime.Options),
) (*bedrockagentruntime.GetInvocationStepOutput, error) {
	_ = ctx
	_ = optFns
	f.mu.Lock()
	defer f.mu.Unlock()
	f.call("GetInvocationStep")
	sess, err := f.session(in.SessionIdentifier)
	if err != nil {
		return nil, err
	}
	inv, ok := sess.invocations[value(in.InvocationIdentifier)]
	if !ok {
		return nil, notFound("invocation")
	}
	step, ok := inv.steps[value(in.InvocationStepId)]
	if !ok {
		return nil, notFound("step")
	}
	return &bedrockagentruntime.GetInvocationStepOutput{InvocationStep: step}, nil
}

func (f *fakeAPI) session(id *string) (*fakeSession, error) {
	if id == nil {
		return nil, notFound("session")
	}
	sess, ok := f.sessions[*id]
	if !ok {
		return nil, notFound("session")
	}
	return sess, nil
}

func mapsClone(in map[string]string) map[string]string {
	return maps.Clone(in)
}

func value(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func notFound(msg string) error {
	return &brtypes.ResourceNotFoundException{Message: &msg}
}

func conflict(msg string) error {
	return &brtypes.ConflictException{Message: &msg}
}

func TestErrorHelpersSeeWrappedAWSErrors(t *testing.T) {
	if !isNotFound(fmt.Errorf("wrap: %w", notFound("missing"))) {
		t.Fatal("wrapped not found error was not detected")
	}
	if !isConflict(fmt.Errorf("wrap: %w", conflict("exists"))) {
		t.Fatal("wrapped conflict error was not detected")
	}
	if isNotFound(errors.New("ResourceNotFoundException: text only")) {
		t.Fatal("plain text should not be treated as a typed AWS not-found error")
	}
}
