package agentcorememory

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// fakeAPI is a test double for the AgentCore Memory API.
type fakeAPI struct {
	createInputs []*bedrockagentcore.CreateEventInput
	createErr    error

	retrieveIn  *bedrockagentcore.RetrieveMemoryRecordsInput
	retrieveOut *bedrockagentcore.RetrieveMemoryRecordsOutput
	retrieveErr error
}

func (f *fakeAPI) CreateEvent(
	_ context.Context,
	in *bedrockagentcore.CreateEventInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.CreateEventOutput, error) {
	f.createInputs = append(f.createInputs, in)
	if f.createErr != nil {
		return nil, f.createErr
	}
	return &bedrockagentcore.CreateEventOutput{}, nil
}

func (f *fakeAPI) RetrieveMemoryRecords(
	_ context.Context,
	in *bedrockagentcore.RetrieveMemoryRecordsInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.RetrieveMemoryRecordsOutput, error) {
	f.retrieveIn = in
	if f.retrieveErr != nil {
		return nil, f.retrieveErr
	}
	if f.retrieveOut != nil {
		return f.retrieveOut, nil
	}
	return &bedrockagentcore.RetrieveMemoryRecordsOutput{}, nil
}

type seedEvent struct {
	author      string
	role        string
	text        string // "" => content-less event (should be skipped)
	llmResponse bool
}

// newSession builds an in-memory session populated with the given events. Event
// timestamps are set deterministically so assertions are stable.
func newSession(t *testing.T, appName, userID string, events []seedEvent) session.Session {
	t.Helper()
	ctx := context.Background()
	svc := session.InMemoryService()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	for i, e := range events {
		ev := session.NewEvent(ctx, "inv")
		ev.Author = e.author
		ev.Timestamp = base.Add(time.Duration(i) * time.Minute)
		if e.text != "" {
			content := genai.NewContentFromText(e.text, genai.Role(e.role))
			if e.llmResponse {
				ev.LLMResponse.Content = content
			} else {
				ev.Content = content
			}
		}
		if err := svc.AppendEvent(ctx, resp.Session, ev); err != nil {
			t.Fatalf("append event: %v", err)
		}
	}
	return resp.Session
}

func conversational(t *testing.T, in *bedrockagentcore.CreateEventInput) types.Conversational {
	t.Helper()
	if len(in.Payload) != 1 {
		t.Fatalf("payload len = %d, want 1", len(in.Payload))
	}
	c, ok := in.Payload[0].(*types.PayloadTypeMemberConversational)
	if !ok {
		t.Fatalf("payload type = %T, want conversational", in.Payload[0])
	}
	return c.Value
}

func TestAddSessionToMemory(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{}
	svc, err := NewWithAPI(api, &Config{MemoryID: "mem-1"})
	if err != nil {
		t.Fatal(err)
	}

	sess := newSession(t, "app", "u1", []seedEvent{
		{author: "user", role: genai.RoleUser, text: "hi"},
		{author: "agent", role: genai.RoleModel, text: "hello", llmResponse: true},
		{author: "user", role: genai.RoleUser, text: ""}, // skipped
	})

	if err := svc.AddSessionToMemory(context.Background(), sess); err != nil {
		t.Fatal(err)
	}

	if len(api.createInputs) != 2 {
		t.Fatalf("CreateEvent calls = %d, want 2 (content-less event skipped)", len(api.createInputs))
	}

	// Collect the source events to compare ids/order.
	var srcIDs []string
	for ev := range sess.Events().All() {
		content := ev.Content
		if content == nil {
			content = ev.LLMResponse.Content
		}
		if textFromContent(content) != "" {
			srcIDs = append(srcIDs, ev.ID)
		}
	}

	want := []struct {
		text string
		role types.Role
	}{
		{"hi", types.RoleUser},
		{"hello", types.RoleAssistant},
	}
	for i, in := range api.createInputs {
		assertCreateEvent(t, i, in, sess, srcIDs[i], want[i].text, want[i].role)
	}
}

// assertCreateEvent checks a single CreateEvent input built by AddSessionToMemory.
func assertCreateEvent(
	t *testing.T,
	i int,
	in *bedrockagentcore.CreateEventInput,
	sess session.Session,
	wantToken, wantText string,
	wantRole types.Role,
) {
	t.Helper()
	if got := aws.ToString(in.MemoryId); got != "mem-1" {
		t.Errorf("call %d MemoryId = %q, want mem-1", i, got)
	}
	if got := aws.ToString(in.ActorId); got != "u1" {
		t.Errorf("call %d ActorId = %q, want u1 (default {userId})", i, got)
	}
	if got := aws.ToString(in.SessionId); got != sess.ID() {
		t.Errorf("call %d SessionId = %q, want %q", i, got, sess.ID())
	}
	if in.EventTimestamp == nil {
		t.Errorf("call %d EventTimestamp is nil", i)
	}
	if got := aws.ToString(in.ClientToken); got != wantToken {
		t.Errorf("call %d ClientToken = %q, want event id %q", i, got, wantToken)
	}
	conv := conversational(t, in)
	if conv.Role != wantRole {
		t.Errorf("call %d role = %q, want %q", i, conv.Role, wantRole)
	}
	ct, ok := conv.Content.(*types.ContentMemberText)
	if !ok {
		t.Fatalf("call %d content type = %T, want text", i, conv.Content)
	}
	if ct.Value != wantText {
		t.Errorf("call %d text = %q, want %q", i, ct.Value, wantText)
	}
	assertMetadata(t, in.Metadata, "app_name", "app")
	assertMetadata(t, in.Metadata, "user_id", "u1")
	assertMetadata(t, in.Metadata, "session_id", sess.ID())
}

func assertMetadata(t *testing.T, md map[string]types.MetadataValue, key, want string) {
	t.Helper()
	v, ok := md[key]
	if !ok {
		t.Errorf("metadata missing key %q", key)
		return
	}
	sv, ok := v.(*types.MetadataValueMemberStringValue)
	if !ok {
		t.Errorf("metadata %q type = %T, want string value", key, v)
		return
	}
	if sv.Value != want {
		t.Errorf("metadata %q = %q, want %q", key, sv.Value, want)
	}
}

func TestAddSessionToMemory_propagatesError(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{createErr: errors.New("boom")}
	svc, err := NewWithAPI(api, &Config{MemoryID: "mem-1"})
	if err != nil {
		t.Fatal(err)
	}
	sess := newSession(t, "app", "u1", []seedEvent{{author: "user", role: genai.RoleUser, text: "hi"}})
	if err := svc.AddSessionToMemory(context.Background(), sess); err == nil {
		t.Fatal("want error from CreateEvent, got nil")
	}
}

func TestSearchMemory_buildsRequest(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{}
	svc, err := NewWithAPI(api, &Config{
		MemoryID:   "mem-1",
		Namespace:  "/actors/{actorId}/facts",
		StrategyID: "strat-1",
		TopK:       5,
		MaxResults: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := svc.SearchMemory(context.Background(), &memory.SearchRequest{
		Query:   "what is my name",
		UserID:  "u1",
		AppName: "app",
	}); err != nil {
		t.Fatal(err)
	}

	in := api.retrieveIn
	if in == nil {
		t.Fatal("RetrieveMemoryRecords not called")
	}
	if got := aws.ToString(in.MemoryId); got != "mem-1" {
		t.Errorf("MemoryId = %q, want mem-1", got)
	}
	if got := aws.ToString(in.Namespace); got != "/actors/u1/facts" {
		t.Errorf("Namespace = %q, want /actors/u1/facts", got)
	}
	if in.NamespacePath != nil {
		t.Errorf("NamespacePath = %v, want nil", aws.ToString(in.NamespacePath))
	}
	if got := aws.ToInt32(in.MaxResults); got != 10 {
		t.Errorf("MaxResults = %d, want 10", got)
	}
	if in.SearchCriteria == nil {
		t.Fatal("SearchCriteria is nil")
	}
	if got := aws.ToString(in.SearchCriteria.SearchQuery); got != "what is my name" {
		t.Errorf("SearchQuery = %q", got)
	}
	if got := aws.ToString(in.SearchCriteria.MemoryStrategyId); got != "strat-1" {
		t.Errorf("MemoryStrategyId = %q, want strat-1", got)
	}
	if got := aws.ToInt32(in.SearchCriteria.TopK); got != 5 {
		t.Errorf("TopK = %d, want 5", got)
	}
}

func TestSearchMemory_namespacePath(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{}
	svc, err := NewWithAPI(api, &Config{
		MemoryID:      "mem-1",
		NamespacePath: "/apps/{appName}/users/{userId}",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.SearchMemory(context.Background(), &memory.SearchRequest{
		Query: "q", UserID: "u1", AppName: "app",
	}); err != nil {
		t.Fatal(err)
	}
	if got := aws.ToString(api.retrieveIn.NamespacePath); got != "/apps/app/users/u1" {
		t.Errorf("NamespacePath = %q, want /apps/app/users/u1", got)
	}
	if api.retrieveIn.Namespace != nil {
		t.Errorf("Namespace = %v, want nil", aws.ToString(api.retrieveIn.Namespace))
	}
}

func TestSearchMemory_requiresNamespace(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{}
	svc, err := NewWithAPI(api, &Config{MemoryID: "mem-1"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := svc.SearchMemory(context.Background(), &memory.SearchRequest{Query: "q", UserID: "u1"}); err == nil {
		t.Fatal("want error when no namespace configured, got nil")
	}
}

func TestSearchMemory_mapsRecords(t *testing.T) {
	t.Parallel()
	created := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	api := &fakeAPI{
		retrieveOut: &bedrockagentcore.RetrieveMemoryRecordsOutput{
			MemoryRecordSummaries: []types.MemoryRecordSummary{
				{
					MemoryRecordId:   aws.String("rec-1"),
					Content:          &types.MemoryContentMemberText{Value: "user lives in Berlin"},
					CreatedAt:        aws.Time(created),
					MemoryStrategyId: aws.String("strat-1"),
					Namespaces:       []string{"/actors/u1/facts"},
					Score:            aws.Float64(0.87),
				},
			},
		},
	}
	svc, err := NewWithAPI(api, &Config{MemoryID: "mem-1", Namespace: "/actors/{actorId}/facts"})
	if err != nil {
		t.Fatal(err)
	}

	res, err := svc.SearchMemory(context.Background(), &memory.SearchRequest{Query: "where", UserID: "u1"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Memories) != 1 {
		t.Fatalf("memories = %d, want 1", len(res.Memories))
	}
	e := res.Memories[0]
	if e.ID != "rec-1" {
		t.Errorf("ID = %q, want rec-1", e.ID)
	}
	if e.Content == nil || len(e.Content.Parts) == 0 || e.Content.Parts[0].Text != "user lives in Berlin" {
		t.Errorf("Content = %+v, want text 'user lives in Berlin'", e.Content)
	}
	if !e.Timestamp.Equal(created) {
		t.Errorf("Timestamp = %v, want %v", e.Timestamp, created)
	}
	if got := e.CustomMetadata["score"]; got != 0.87 {
		t.Errorf("score = %v, want 0.87", got)
	}
	if got := e.CustomMetadata["memory_strategy_id"]; got != "strat-1" {
		t.Errorf("memory_strategy_id = %v, want strat-1", got)
	}
}

func TestNewWithAPI_validation(t *testing.T) {
	t.Parallel()
	if _, err := NewWithAPI(nil, &Config{MemoryID: "m"}); err == nil {
		t.Error("want error for nil api")
	}
	if _, err := NewWithAPI(&fakeAPI{}, nil); err == nil {
		t.Error("want error for nil config")
	}
	if _, err := NewWithAPI(&fakeAPI{}, &Config{}); err == nil {
		t.Error("want error for empty MemoryID")
	}
	svc, err := NewWithAPI(&fakeAPI{}, &Config{MemoryID: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if svc.cfg.ActorIDTemplate != defaultActorIDTemplate {
		t.Errorf("ActorIDTemplate default = %q, want %q", svc.cfg.ActorIDTemplate, defaultActorIDTemplate)
	}
}

func TestNewWithAPI_copiesMetadataFilters(t *testing.T) {
	t.Parallel()

	filters := []types.MemoryMetadataFilterExpression{{
		Left: &types.MemoryRecordLeftExpressionMemberMetadataKey{Value: "category"},
	}}
	svc, err := NewWithAPI(&fakeAPI{}, &Config{MemoryID: "mem-1", Namespace: "/facts", MetadataFilters: filters})
	if err != nil {
		t.Fatal(err)
	}

	filters[0] = types.MemoryMetadataFilterExpression{}
	if _, err := svc.SearchMemory(context.Background(), &memory.SearchRequest{Query: "q"}); err != nil {
		t.Fatal(err)
	}
	filter := svc.api.(*fakeAPI).retrieveIn.SearchCriteria.MetadataFilters[0]
	left, ok := filter.Left.(*types.MemoryRecordLeftExpressionMemberMetadataKey)
	if !ok || left.Value != "category" {
		t.Fatalf("copied filter left = %#v, want metadata key category", filter.Left)
	}
}
