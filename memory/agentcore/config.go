package agentcorememory

import (
	"context"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
)

const defaultActorIDTemplate = "{userId}"

// API is the subset of Bedrock AgentCore operations used by this package.
// The concrete [bedrockagentcore.Client] satisfies it directly.
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
	// MemoryID is the ARN of the AgentCore Memory resource. Required.
	MemoryID string

	// Region overrides the AWS region used by [New] and [NewWithAWSConfig].
	// Ignored by [NewWithAPI].
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

// New creates a [Service] using the default AWS configuration chain and a new
// [bedrockagentcore.Client]. cfg.MemoryID is required.
func New(ctx context.Context, cfg *Config) (*Service, error) {
	if cfg == nil {
		return nil, errors.New("agentcorememory: config is required")
	}
	awsCfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("agentcorememory: load AWS config: %w", err)
	}
	return NewWithAWSConfig(awsCfg, cfg)
}

// NewWithAWSConfig creates a [Service] using an existing AWS configuration.
// cfg.Region, when set, overrides awsCfg.Region.
func NewWithAWSConfig(awsCfg aws.Config, cfg *Config) (*Service, error) {
	if cfg == nil {
		return nil, errors.New("agentcorememory: config is required")
	}
	if cfg.Region != "" {
		awsCfg.Region = cfg.Region
	}
	return NewWithAPI(bedrockagentcore.NewFromConfig(awsCfg), cfg)
}

// NewWithAPI wires a [Service] to an existing AgentCore client.
func NewWithAPI(api API, cfg *Config) (*Service, error) {
	if api == nil {
		return nil, errors.New("agentcorememory: API is required")
	}
	if cfg == nil {
		return nil, errors.New("agentcorememory: config is required")
	}
	c := *cfg
	c.MetadataFilters = append([]types.MemoryMetadataFilterExpression(nil), cfg.MetadataFilters...)
	if c.MemoryID == "" {
		return nil, errors.New("agentcorememory: MemoryID is required")
	}
	if c.ActorIDTemplate == "" {
		c.ActorIDTemplate = defaultActorIDTemplate
	}
	return &Service{api: api, cfg: c}, nil
}
