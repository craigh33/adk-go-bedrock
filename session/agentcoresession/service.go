package agentcoresession

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime/types"
	"github.com/aws/smithy-go"
	"google.golang.org/adk/session"

	bedrockmappers "github.com/craigh33/adk-go-bedrock/internal/mappers"
)

// Options configures [NewWithAPI].
type Options struct {
	// EncryptionKeyARN is passed to Bedrock CreateSession when set.
	EncryptionKeyARN string
}

// AgentRuntimeAPI is the subset of Bedrock Agent Runtime used by this package.
type AgentRuntimeAPI interface {
	CreateSession(
		context.Context,
		*bedrockagentruntime.CreateSessionInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.CreateSessionOutput, error)
	GetSession(
		context.Context,
		*bedrockagentruntime.GetSessionInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.GetSessionOutput, error)
	UpdateSession(
		context.Context,
		*bedrockagentruntime.UpdateSessionInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.UpdateSessionOutput, error)
	ListSessions(
		context.Context,
		*bedrockagentruntime.ListSessionsInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.ListSessionsOutput, error)
	EndSession(
		context.Context,
		*bedrockagentruntime.EndSessionInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.EndSessionOutput, error)
	DeleteSession(
		context.Context,
		*bedrockagentruntime.DeleteSessionInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.DeleteSessionOutput, error)
	CreateInvocation(
		context.Context,
		*bedrockagentruntime.CreateInvocationInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.CreateInvocationOutput, error)
	ListInvocations(
		context.Context,
		*bedrockagentruntime.ListInvocationsInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.ListInvocationsOutput, error)
	PutInvocationStep(
		context.Context,
		*bedrockagentruntime.PutInvocationStepInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.PutInvocationStepOutput, error)
	ListInvocationSteps(
		context.Context,
		*bedrockagentruntime.ListInvocationStepsInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.ListInvocationStepsOutput, error)
	GetInvocationStep(
		context.Context,
		*bedrockagentruntime.GetInvocationStepInput,
		...func(*bedrockagentruntime.Options),
	) (*bedrockagentruntime.GetInvocationStepOutput, error)
}

// NewWithAPI wires a Bedrock Agent Runtime implementation.
func NewWithAPI(api AgentRuntimeAPI, opts *Options) (session.Service, error) {
	if api == nil {
		return nil, errors.New("nil AgentRuntimeAPI")
	}
	s := &service{api: api}
	if opts != nil {
		s.encryptionKeyARN = strings.TrimSpace(opts.EncryptionKeyARN)
	}
	return s, nil
}

type service struct {
	api              AgentRuntimeAPI
	encryptionKeyARN string
}

func (s *service) Create(ctx context.Context, req *session.CreateRequest) (*session.CreateResponse, error) {
	if req.AppName == "" || req.UserID == "" {
		return nil, fmt.Errorf(
			"app_name and user_id are required, got app_name: %q, user_id: %q",
			req.AppName,
			req.UserID,
		)
	}
	if req.SessionID != "" {
		return nil, fmt.Errorf(
			"user-provided SessionID is not supported for AgentCoreSessionService: %q",
			req.SessionID,
		)
	}

	storedState := map[string]any{}
	maps.Copy(storedState, req.State)

	md, err := bedrockmappers.AgentCoreSessionMetadata(req.AppName, req.UserID, storedState)
	if err != nil {
		return nil, err
	}
	shared, err := s.sharedState(ctx, req.AppName, req.UserID)
	if err != nil {
		return nil, err
	}
	state := map[string]any{}
	maps.Copy(state, shared)
	maps.Copy(state, storedState)

	input := &bedrockagentruntime.CreateSessionInput{
		SessionMetadata: md,
	}
	if s.encryptionKeyARN != "" {
		input.EncryptionKeyArn = &s.encryptionKeyARN
	}
	out, err := s.api.CreateSession(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	if out.SessionId == nil || *out.SessionId == "" {
		return nil, errors.New("create session: empty AgentCore session id")
	}

	updatedAt := time.Now()
	if out.CreatedAt != nil {
		updatedAt = *out.CreatedAt
	}
	return &session.CreateResponse{
		Session: newLocalSession(req.AppName, req.UserID, *out.SessionId, state, storedState, nil, updatedAt),
	}, nil
}

func (s *service) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return nil, fmt.Errorf(
			"app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q",
			req.AppName,
			req.UserID,
			req.SessionID,
		)
	}

	remote, err := s.getOwnedSession(ctx, req.AppName, req.UserID, req.SessionID)
	if err != nil {
		return nil, err
	}
	storedState, err := bedrockmappers.AgentCoreSessionStateFromMetadata(remote.metadata)
	if err != nil {
		return nil, err
	}
	events, err := s.loadEvents(ctx, req.SessionID)
	if err != nil {
		return nil, err
	}
	replayState(storedState, events)

	shared, err := s.sharedState(ctx, req.AppName, req.UserID)
	if err != nil {
		return nil, err
	}
	state := map[string]any{}
	maps.Copy(state, storedState)
	maps.Copy(state, shared)

	events = filterEvents(events, req.After, req.NumRecentEvents)
	return &session.GetResponse{
		Session: newLocalSession(req.AppName, req.UserID, req.SessionID, state, storedState, events, remote.updatedAt),
	}, nil
}

func (s *service) List(ctx context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	if req.AppName == "" {
		return nil, fmt.Errorf("app_name is required, got app_name: %q", req.AppName)
	}

	remotes, err := s.listAllSessions(ctx)
	if err != nil {
		return nil, err
	}
	var sessions []session.Session
	for _, remote := range remotes {
		if !bedrockmappers.AgentCoreSessionMetadataMatchesApp(remote.metadata, req.AppName) {
			continue
		}
		if req.UserID != "" && bedrockmappers.AgentCoreSessionMetadataUserID(remote.metadata) != req.UserID {
			continue
		}
		storedState, err := bedrockmappers.AgentCoreSessionStateFromMetadata(remote.metadata)
		if err != nil {
			return nil, err
		}
		shared, err := sharedStateFromRemotes(remotes, req.AppName, remote.userID)
		if err != nil {
			return nil, err
		}
		state := map[string]any{}
		maps.Copy(state, storedState)
		maps.Copy(state, shared)
		sessions = append(sessions, newLocalSession(
			req.AppName,
			remote.userID,
			remote.id,
			state,
			storedState,
			nil,
			remote.updatedAt,
		))
	}
	return &session.ListResponse{Sessions: sessions}, nil
}

func (s *service) Delete(ctx context.Context, req *session.DeleteRequest) error {
	if req.AppName == "" || req.UserID == "" || req.SessionID == "" {
		return fmt.Errorf(
			"app_name, user_id and session_id are required, got app_name: %q, user_id: %q, session_id: %q",
			req.AppName,
			req.UserID,
			req.SessionID,
		)
	}
	if _, err := s.getOwnedSession(ctx, req.AppName, req.UserID, req.SessionID); err != nil {
		if isNotFound(err) {
			return nil
		}
		return err
	}

	_, err := s.api.EndSession(ctx, &bedrockagentruntime.EndSessionInput{SessionIdentifier: &req.SessionID})
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("end session: %w", err)
	}
	_, err = s.api.DeleteSession(ctx, &bedrockagentruntime.DeleteSessionInput{SessionIdentifier: &req.SessionID})
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("delete session: %w", err)
	}
	return nil
}

func (s *service) AppendEvent(ctx context.Context, sess session.Session, event *session.Event) error {
	if sess == nil {
		return errors.New("session is nil")
	}
	if event == nil {
		return errors.New("event is nil")
	}
	if event.Partial {
		return nil
	}
	local, ok := sess.(*localSession)
	if !ok {
		return fmt.Errorf(
			"AppendEvent for AgentCore session service only supports sessions created by it, got %T",
			sess,
		)
	}

	if _, err := s.getOwnedSession(ctx, local.AppName(), local.UserID(), local.ID()); err != nil {
		return err
	}

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("clone event: %w", err)
	}
	var storedEvent session.Event
	if err := json.Unmarshal(eventJSON, &storedEvent); err != nil {
		return fmt.Errorf("clone event: %w", err)
	}
	trimTempDelta(&storedEvent)

	sessionID := local.ID()
	invocationID := bedrockmappers.AgentCoreSessionInvocationID(sessionID, storedEvent.InvocationID, storedEvent.ID)
	description := "ADK invocation"
	_, err = s.api.CreateInvocation(ctx, &bedrockagentruntime.CreateInvocationInput{
		SessionIdentifier: &sessionID,
		InvocationId:      &invocationID,
		Description:       &description,
	})
	if err != nil && !isConflict(err) {
		return fmt.Errorf("create invocation: %w", err)
	}

	payload, err := bedrockmappers.AgentCoreSessionEncodeEvent(&storedEvent)
	if err != nil {
		return err
	}
	stepID := bedrockmappers.AgentCoreSessionStepID(sessionID, &storedEvent)
	stepTime := storedEvent.Timestamp
	if stepTime.IsZero() {
		stepTime = time.Now()
	}
	_, err = s.api.PutInvocationStep(ctx, &bedrockagentruntime.PutInvocationStepInput{
		SessionIdentifier:    &sessionID,
		InvocationIdentifier: &invocationID,
		InvocationStepId:     &stepID,
		InvocationStepTime:   &stepTime,
		Payload:              bedrockmappers.AgentCoreSessionInvocationStepPayload(payload),
	})
	if err != nil {
		return fmt.Errorf("put invocation step: %w", err)
	}

	shared, err := s.sharedState(ctx, local.AppName(), local.UserID())
	if err != nil {
		return err
	}
	local.mergeSharedState(shared)
	local.appendEvent(&storedEvent)
	md, err := bedrockmappers.AgentCoreSessionMetadata(
		local.AppName(),
		local.UserID(),
		local.snapshotStoredState(),
	)
	if err != nil {
		return err
	}
	_, err = s.api.UpdateSession(ctx, &bedrockagentruntime.UpdateSessionInput{
		SessionIdentifier: &sessionID,
		SessionMetadata:   md,
	})
	if err != nil {
		return fmt.Errorf("update session metadata: %w", err)
	}
	return nil
}

type remoteSession struct {
	id        string
	userID    string
	updatedAt time.Time
	metadata  map[string]string
}

func (s *service) getOwnedSession(ctx context.Context, appName, userID, sessionID string) (*remoteSession, error) {
	out, err := s.api.GetSession(ctx, &bedrockagentruntime.GetSessionInput{SessionIdentifier: &sessionID})
	if err != nil {
		return nil, fmt.Errorf("get session: %w", err)
	}
	if out == nil || out.SessionId == nil || *out.SessionId == "" {
		return nil, fmt.Errorf("session %s not found", sessionID)
	}
	remote := remoteFromGet(out)
	if !bedrockmappers.AgentCoreSessionMetadataMatches(remote.metadata, appName, userID) {
		return nil, fmt.Errorf("session %s does not belong to app %q user %q", sessionID, appName, userID)
	}
	return remote, nil
}

func remoteFromGet(out *bedrockagentruntime.GetSessionOutput) *remoteSession {
	updatedAt := time.Now()
	if out.LastUpdatedAt != nil {
		updatedAt = *out.LastUpdatedAt
	} else if out.CreatedAt != nil {
		updatedAt = *out.CreatedAt
	}
	id := ""
	if out.SessionId != nil {
		id = *out.SessionId
	}
	userID := ""
	if out.SessionMetadata != nil {
		userID = bedrockmappers.AgentCoreSessionMetadataUserID(out.SessionMetadata)
	}
	return &remoteSession{id: id, userID: userID, updatedAt: updatedAt, metadata: maps.Clone(out.SessionMetadata)}
}

func (s *service) listAllSessions(ctx context.Context) ([]*remoteSession, error) {
	var token *string
	var remotes []*remoteSession
	for {
		out, err := s.api.ListSessions(ctx, &bedrockagentruntime.ListSessionsInput{NextToken: token})
		if err != nil {
			return nil, fmt.Errorf("list sessions: %w", err)
		}
		for _, summary := range out.SessionSummaries {
			if summary.SessionId == nil || *summary.SessionId == "" {
				continue
			}
			detail, err := s.api.GetSession(ctx, &bedrockagentruntime.GetSessionInput{
				SessionIdentifier: summary.SessionId,
			})
			if err != nil {
				if isNotFound(err) {
					continue
				}
				return nil, fmt.Errorf("get listed session %s: %w", *summary.SessionId, err)
			}
			remotes = append(remotes, remoteFromGet(detail))
		}
		if out.NextToken == nil || *out.NextToken == "" {
			break
		}
		token = out.NextToken
	}
	sort.SliceStable(remotes, func(i, j int) bool { return remotes[i].updatedAt.Before(remotes[j].updatedAt) })
	return remotes, nil
}

func (s *service) sharedState(ctx context.Context, appName, userID string) (map[string]any, error) {
	remotes, err := s.listAllSessions(ctx)
	if err != nil {
		return nil, err
	}
	return sharedStateFromRemotes(remotes, appName, userID)
}

func sharedStateFromRemotes(remotes []*remoteSession, appName, userID string) (map[string]any, error) {
	state := map[string]any{}
	for _, remote := range remotes {
		if !bedrockmappers.AgentCoreSessionMetadataMatchesApp(remote.metadata, appName) {
			continue
		}
		remoteState, err := bedrockmappers.AgentCoreSessionStateFromMetadata(remote.metadata)
		if err != nil {
			return nil, err
		}
		for k, v := range remoteState {
			switch {
			case strings.HasPrefix(k, session.KeyPrefixApp):
				state[k] = v
			case strings.HasPrefix(k, session.KeyPrefixUser) &&
				bedrockmappers.AgentCoreSessionMetadataUserID(remote.metadata) == userID:
				state[k] = v
			}
		}
	}
	return state, nil
}

func (s *service) loadEvents(ctx context.Context, sessionID string) ([]*session.Event, error) {
	var token *string
	var events []*session.Event
	for {
		out, err := s.api.ListInvocations(ctx, &bedrockagentruntime.ListInvocationsInput{
			SessionIdentifier: &sessionID,
			NextToken:         token,
		})
		if err != nil {
			return nil, fmt.Errorf("list invocations: %w", err)
		}
		for _, inv := range out.InvocationSummaries {
			if inv.InvocationId == nil || *inv.InvocationId == "" {
				continue
			}
			invEvents, err := s.loadInvocationEvents(ctx, sessionID, *inv.InvocationId)
			if err != nil {
				return nil, err
			}
			events = append(events, invEvents...)
		}
		if out.NextToken == nil || *out.NextToken == "" {
			break
		}
		token = out.NextToken
	}
	sort.SliceStable(events, func(i, j int) bool { return events[i].Timestamp.Before(events[j].Timestamp) })
	return events, nil
}

func (s *service) loadInvocationEvents(ctx context.Context, sessionID, invocationID string) ([]*session.Event, error) {
	var token *string
	var events []*session.Event
	for {
		out, err := s.api.ListInvocationSteps(ctx, &bedrockagentruntime.ListInvocationStepsInput{
			SessionIdentifier:    &sessionID,
			InvocationIdentifier: &invocationID,
			NextToken:            token,
		})
		if err != nil {
			return nil, fmt.Errorf("list invocation steps: %w", err)
		}
		for _, summary := range out.InvocationStepSummaries {
			if summary.InvocationStepId == nil || *summary.InvocationStepId == "" {
				continue
			}
			step, err := s.api.GetInvocationStep(ctx, &bedrockagentruntime.GetInvocationStepInput{
				SessionIdentifier:    &sessionID,
				InvocationIdentifier: &invocationID,
				InvocationStepId:     summary.InvocationStepId,
			})
			if err != nil {
				return nil, fmt.Errorf("get invocation step: %w", err)
			}
			if event, ok := bedrockmappers.AgentCoreSessionDecodeInvocationStep(step.InvocationStep); ok {
				events = append(events, event)
			}
		}
		if out.NextToken == nil || *out.NextToken == "" {
			break
		}
		token = out.NextToken
	}
	return events, nil
}

func trimTempDelta(event *session.Event) {
	if len(event.Actions.StateDelta) == 0 {
		return
	}
	for k := range event.Actions.StateDelta {
		if strings.HasPrefix(k, session.KeyPrefixTemp) {
			delete(event.Actions.StateDelta, k)
		}
	}
}

func replayState(state map[string]any, events []*session.Event) {
	for _, event := range events {
		if event == nil {
			continue
		}
		trimTempDelta(event)
		maps.Copy(state, event.Actions.StateDelta)
	}
}

func filterEvents(events []*session.Event, after time.Time, numRecent int) []*session.Event {
	if !after.IsZero() {
		events = events[sort.Search(len(events), func(i int) bool {
			return !events[i].Timestamp.Before(after)
		}):]
	}
	if numRecent > 0 && numRecent < len(events) {
		events = events[len(events)-numRecent:]
	}
	return events
}

func isNotFound(err error) bool {
	var nf *brtypes.ResourceNotFoundException
	if errors.As(err, &nf) {
		return true
	}
	var apiErr smithy.APIError
	return errors.As(err, &apiErr) && apiErr.ErrorCode() == "ResourceNotFoundException"
}

func isConflict(err error) bool {
	var conflict *brtypes.ConflictException
	if errors.As(err, &conflict) {
		return true
	}
	var apiErr smithy.APIError
	return errors.As(err, &apiErr) && apiErr.ErrorCode() == "ConflictException"
}

type localSession struct {
	appName   string
	userID    string
	sessionID string

	mu        sync.RWMutex
	state     map[string]any
	stored    map[string]any
	events    []*session.Event
	updatedAt time.Time
}

func newLocalSession(
	appName, userID, sessionID string,
	state, storedState map[string]any,
	events []*session.Event,
	updatedAt time.Time,
) *localSession {
	stateCopy := map[string]any{}
	maps.Copy(stateCopy, state)
	storedCopy := map[string]any{}
	maps.Copy(storedCopy, storedState)
	return &localSession{
		appName:   appName,
		userID:    userID,
		sessionID: sessionID,
		state:     stateCopy,
		stored:    storedCopy,
		events:    append([]*session.Event(nil), events...),
		updatedAt: updatedAt,
	}
}

func (s *localSession) ID() string      { return s.sessionID }
func (s *localSession) AppName() string { return s.appName }
func (s *localSession) UserID() string  { return s.userID }

func (s *localSession) State() session.State {
	return &state{mu: &s.mu, state: s.state, stored: s.stored}
}

func (s *localSession) Events() session.Events {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return events(append([]*session.Event(nil), s.events...))
}

func (s *localSession) LastUpdateTime() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.updatedAt
}

func (s *localSession) appendEvent(event *session.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == nil {
		s.state = map[string]any{}
	}
	if s.stored == nil {
		s.stored = map[string]any{}
	}
	maps.Copy(s.state, event.Actions.StateDelta)
	maps.Copy(s.stored, event.Actions.StateDelta)
	s.events = append(s.events, event)
	if event.Timestamp.IsZero() {
		s.updatedAt = time.Now()
	} else {
		s.updatedAt = event.Timestamp
	}
}

func (s *localSession) mergeSharedState(shared map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == nil {
		s.state = map[string]any{}
	}
	if s.stored == nil {
		s.stored = map[string]any{}
	}
	for k, v := range shared {
		switch {
		case strings.HasPrefix(k, session.KeyPrefixApp):
			s.state[k] = v
			s.stored[k] = v
		case strings.HasPrefix(k, session.KeyPrefixUser):
			s.state[k] = v
			s.stored[k] = v
		}
	}
}

func (s *localSession) snapshotStoredState() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return maps.Clone(s.stored)
}

type events []*session.Event

func (e events) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, event := range e {
			if !yield(event) {
				return
			}
		}
	}
}

func (e events) Len() int { return len(e) }

func (e events) At(i int) *session.Event {
	if i < 0 || i >= len(e) {
		return nil
	}
	return e[i]
}

type state struct {
	mu     *sync.RWMutex
	state  map[string]any
	stored map[string]any
}

func (s *state) Get(key string) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.state[key]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (s *state) Set(key string, value any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state[key] = value
	s.stored[key] = value
	return nil
}

func (s *state) All() iter.Seq2[string, any] {
	s.mu.RLock()
	stateCopy := maps.Clone(s.state)
	s.mu.RUnlock()
	return func(yield func(string, any) bool) {
		for k, v := range stateCopy {
			if !yield(k, v) {
				return
			}
		}
	}
}

var (
	_ session.Service = (*service)(nil)
	_ session.Session = (*localSession)(nil)
	_ session.Events  = events(nil)
	_ session.State   = (*state)(nil)
)
