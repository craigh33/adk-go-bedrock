// Package agentcore implements the ADK [memory.Service] interface using Amazon
// Bedrock AgentCore Memory.
//
// It writes ADK session events to an AgentCore Memory resource with CreateEvent
// and searches long-term memory with RetrieveMemoryRecords, so agents running on
// Bedrock can persist and recall user-scoped memory across sessions through the
// same [runner.Config.MemoryService] hook used for other providers.
//
// Long-term memory extraction in AgentCore is asynchronous: events written with
// [Service.AddSessionToMemory] are not immediately returned by
// [Service.SearchMemory]. Allow time for extraction to complete before expecting
// freshly written events to be searchable.
//
// The AgentCore Memory resource (and its strategies/namespaces) must be
// provisioned out of band. Callers need IAM permission for
// bedrock-agentcore:CreateEvent and bedrock-agentcore:RetrieveMemoryRecords.
package agentcore

import (
	"context"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/session"
)

var _ memory.Service = (*Service)(nil)

// defaultActorIDTemplate maps ADK identity to an AgentCore actor id. By default
// the actor is the ADK user.
const defaultActorIDTemplate = "{userId}"

// API is the subset of Bedrock AgentCore operations used by this package. The
// concrete [bedrockagentcore.Client] satisfies it directly, and tests can supply
// a fake.
type API interface {
	CreateEvent(
		ctx context.Context,
		params *bedrockagentcore.CreateEventInput,
		optFns ...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.CreateEventOutput, error)
	RetrieveMemoryRecords(
		ctx context.Context,
		params *bedrockagentcore.RetrieveMemoryRecordsInput,
		optFns ...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.RetrieveMemoryRecordsOutput, error)
}

// Config configures a [Service].
type Config struct {
	// MemoryID is the AgentCore Memory resource identifier. Required.
	MemoryID string

	// Region overrides the AWS region. When empty, the default
	// [config.LoadDefaultConfig] resolution is used. Ignored by [NewWithAPI].
	Region string

	// Namespace is the namespace prefix searched by [Service.SearchMemory]. It
	// supports the placeholders {actorId}, {userId} and {appName}, substituted
	// per request. Either Namespace or NamespacePath must be set to search.
	Namespace string

	// NamespacePath, when set, is used instead of Namespace for hierarchical
	// retrieval (all memory records under the same parent hierarchy). It supports
	// the same placeholders as Namespace.
	NamespacePath string

	// StrategyID filters retrieval to a single memory strategy
	// (SearchCriteria.MemoryStrategyId). Optional.
	StrategyID string

	// TopK sets the maximum number of top-scoring records used for semantic
	// ranking (SearchCriteria.TopK). Optional; zero leaves it unset.
	TopK int32

	// MaxResults caps the number of records returned by a single search. Optional;
	// zero uses the AgentCore default (20).
	MaxResults int32

	// MetadataFilters optionally narrows retrieval by memory metadata. Optional.
	MetadataFilters []types.MemoryMetadataFilterExpression

	// ActorIDTemplate maps ADK identity to an AgentCore actor id. It supports the
	// placeholders {userId} and {appName}. Defaults to "{userId}".
	ActorIDTemplate string
}

// Service implements [memory.Service] against an AgentCore Memory resource.
type Service struct {
	api API
	cfg Config
}

// New creates a [Service] using the default AWS configuration chain and a new
// [bedrockagentcore.Client]. cfg.MemoryID is required.
func New(ctx context.Context, cfg *Config) (*Service, error) {
	if cfg == nil {
		return nil, errors.New("agentcore: config is required")
	}
	awsCfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("agentcore: load AWS config: %w", err)
	}
	if cfg.Region != "" {
		awsCfg.Region = cfg.Region
	}
	return NewWithAPI(bedrockagentcore.NewFromConfig(awsCfg), cfg)
}

// NewWithAPI wires a [Service] to an existing AgentCore client (or fake). It is
// the constructor used by tests and callers who build their own client.
func NewWithAPI(api API, cfg *Config) (*Service, error) {
	if api == nil {
		return nil, errors.New("agentcore: API is required")
	}
	if cfg == nil {
		return nil, errors.New("agentcore: config is required")
	}
	c := *cfg
	if c.MemoryID == "" {
		return nil, errors.New("agentcore: MemoryID is required")
	}
	if c.ActorIDTemplate == "" {
		c.ActorIDTemplate = defaultActorIDTemplate
	}
	return &Service{api: api, cfg: c}, nil
}

// AddSessionToMemory writes the session's events to AgentCore Memory. It emits one
// CreateEvent per text-bearing event, preserving each event's role and timestamp;
// events without textual content are skipped. The event id is used as the
// CreateEvent client token so repeated calls on a growing session are idempotent.
//
// AddSessionToMemory implements [memory.Service].
func (s *Service) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if sess == nil {
		return errors.New("agentcore: session is nil")
	}
	actorID := resolveActor(s.cfg.ActorIDTemplate, sess.AppName(), sess.UserID())

	for event := range sess.Events().All() {
		text := textFromContent(event.LLMResponse.Content)
		if text == "" {
			continue
		}
		in := &bedrockagentcore.CreateEventInput{
			MemoryId:       aws.String(s.cfg.MemoryID),
			ActorId:        aws.String(actorID),
			SessionId:      aws.String(sess.ID()),
			EventTimestamp: aws.Time(eventTimestamp(event, sess)),
			Payload: []types.PayloadType{
				&types.PayloadTypeMemberConversational{
					Value: types.Conversational{
						Content: &types.ContentMemberText{Value: text},
						Role:    toAgentCoreRole(event.LLMResponse.Content),
					},
				},
			},
			Metadata: eventMetadata(sess, event),
		}
		if event.ID != "" {
			in.ClientToken = aws.String(event.ID)
		}
		if _, err := s.api.CreateEvent(ctx, in); err != nil {
			return fmt.Errorf("agentcore: CreateEvent: %w", err)
		}
	}
	return nil
}

// SearchMemory retrieves memory records relevant to req.Query and maps them to
// [memory.Entry] values. The configured namespace (with {actorId}, {userId} and
// {appName} substituted from the request) scopes the search; either
// Config.Namespace or Config.NamespacePath must be set.
//
// SearchMemory implements [memory.Service].
func (s *Service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil {
		return nil, errors.New("agentcore: search request is nil")
	}
	actorID := resolveActor(s.cfg.ActorIDTemplate, req.AppName, req.UserID)

	in := &bedrockagentcore.RetrieveMemoryRecordsInput{
		MemoryId: aws.String(s.cfg.MemoryID),
		SearchCriteria: &types.SearchCriteria{
			SearchQuery:     aws.String(req.Query),
			MetadataFilters: s.cfg.MetadataFilters,
		},
	}
	if s.cfg.StrategyID != "" {
		in.SearchCriteria.MemoryStrategyId = aws.String(s.cfg.StrategyID)
	}
	if s.cfg.TopK > 0 {
		in.SearchCriteria.TopK = aws.Int32(s.cfg.TopK)
	}
	if s.cfg.MaxResults > 0 {
		in.MaxResults = aws.Int32(s.cfg.MaxResults)
	}

	switch {
	case s.cfg.NamespacePath != "":
		in.NamespacePath = aws.String(resolveNamespace(s.cfg.NamespacePath, req.AppName, req.UserID, actorID))
	case s.cfg.Namespace != "":
		in.Namespace = aws.String(resolveNamespace(s.cfg.Namespace, req.AppName, req.UserID, actorID))
	default:
		return nil, errors.New("agentcore: Namespace or NamespacePath is required to search")
	}

	out, err := s.api.RetrieveMemoryRecords(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("agentcore: RetrieveMemoryRecords: %w", err)
	}

	res := &memory.SearchResponse{Memories: make([]memory.Entry, 0, len(out.MemoryRecordSummaries))}
	for _, rec := range out.MemoryRecordSummaries {
		res.Memories = append(res.Memories, recordToEntry(rec))
	}
	return res, nil
}
