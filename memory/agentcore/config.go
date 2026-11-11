package agentcorememory

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
)

const defaultActorIDTemplate = "{userId}"

// MemoryAPI is implemented by [bedrockagentcore.Client].
type MemoryAPI interface {
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

// Option configures a [Service].
type Option func(*serviceConfig)

type serviceConfig struct {
	memoryID        string
	region          string
	namespace       string
	namespacePath   string
	strategyID      string
	topK            int32
	maxResults      int32
	metadataFilters []types.MemoryMetadataFilterExpression
	actorIDTemplate string
}

// WithRegion overrides the AWS region used by [New] and [NewWithAWSConfig].
func WithRegion(region string) Option {
	return func(c *serviceConfig) { c.region = region }
}

// WithNamespace sets the namespace searched by [Service.SearchMemory]. It
// supports the {actorId}, {userId}, and {appName} placeholders. Raw {userId} and
// {appName} values must each fit in one nonempty namespace segment. Wildcards
// must be in separate segments from placeholders. Either a namespace or a
// namespace path is required to search.
func WithNamespace(namespace string) Option {
	return func(c *serviceConfig) { c.namespace = namespace }
}

// WithNamespacePath sets the namespace path for hierarchical searches. It takes
// precedence over [WithNamespace] and supports the same placeholders. A
// trailing slash bounds the final segment unless the path ends with a wildcard.
func WithNamespacePath(path string) Option {
	if path != "" && !strings.HasSuffix(path, "*") && !strings.HasSuffix(path, "/") {
		path += "/"
	}
	return func(c *serviceConfig) { c.namespacePath = path }
}

// WithStrategyID limits searches to one memory strategy.
func WithStrategyID(id string) Option {
	return func(c *serviceConfig) { c.strategyID = id }
}

// WithTopK limits the number of top-scoring records used for semantic ranking.
// Zero uses the AgentCore default.
func WithTopK(topK int32) Option {
	return func(c *serviceConfig) { c.topK = topK }
}

// WithMaxResults caps the number of records returned by a search. Zero uses
// the AgentCore default.
func WithMaxResults(maxResults int32) Option {
	return func(c *serviceConfig) { c.maxResults = maxResults }
}

// WithMetadataFilters narrows searches by memory metadata. The constructor
// copies the filters so subsequent changes to the slice or its values are safe.
func WithMetadataFilters(filters []types.MemoryMetadataFilterExpression) Option {
	return func(c *serviceConfig) {
		c.metadataFilters = nil
		if len(filters) > 0 {
			c.metadataFilters = make([]types.MemoryMetadataFilterExpression, len(filters))
			for i, filter := range filters {
				c.metadataFilters[i] = cloneMetadataFilter(filter)
			}
		}
	}
}

// WithActorIDTemplate sets the template used to derive AgentCore actor IDs. It
// supports {userId} and {appName}. An empty template uses the default {userId}.
func WithActorIDTemplate(template string) Option {
	return func(c *serviceConfig) { c.actorIDTemplate = template }
}

// New creates a [Service] using the default AWS configuration chain.
// memoryID is the required AgentCore Memory resource ARN.
func New(ctx context.Context, memoryID string, opts ...Option) (*Service, error) {
	s, err := newService(memoryID, opts)
	if err != nil {
		return nil, err
	}
	awsCfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("agentcorememory: load AWS config: %w", err)
	}
	s.configureAWSClient(awsCfg)
	return s, nil
}

// NewWithAWSConfig creates a [Service] using an existing AWS configuration.
// memoryID is the required AgentCore Memory resource ARN.
func NewWithAWSConfig(awsCfg aws.Config, memoryID string, opts ...Option) (*Service, error) {
	s, err := newService(memoryID, opts)
	if err != nil {
		return nil, err
	}
	s.configureAWSClient(awsCfg)
	return s, nil
}

// NewWithAPI creates a [Service] using an existing AgentCore client.
// memoryID is the required AgentCore Memory resource ARN.
func NewWithAPI(api MemoryAPI, memoryID string, opts ...Option) (*Service, error) {
	if api == nil {
		return nil, errors.New("agentcorememory: API is required")
	}
	s, err := newService(memoryID, opts)
	if err != nil {
		return nil, err
	}
	s.api = api
	return s, nil
}

func newService(memoryID string, opts []Option) (*Service, error) {
	if memoryID == "" {
		return nil, errors.New("agentcorememory: memory ID is required")
	}
	c := serviceConfig{memoryID: memoryID}
	for _, opt := range opts {
		opt(&c)
	}
	if c.actorIDTemplate == "" {
		c.actorIDTemplate = defaultActorIDTemplate
	}
	return &Service{cfg: c}, nil
}

func (s *Service) configureAWSClient(awsCfg aws.Config) {
	if s.cfg.region != "" {
		awsCfg.Region = s.cfg.region
	}
	s.api = bedrockagentcore.NewFromConfig(awsCfg)
}

func cloneMetadataFilter(filter types.MemoryMetadataFilterExpression) types.MemoryMetadataFilterExpression {
	if left, ok := filter.Left.(*types.MemoryRecordLeftExpressionMemberMetadataKey); ok && left != nil {
		filter.Left = new(*left)
	}
	if right, ok := filter.Right.(*types.MemoryRecordRightExpressionMemberMetadataValue); ok && right != nil {
		copied := *right
		copied.Value = cloneMetadataValue(right.Value)
		filter.Right = &copied
	}
	return filter
}

func cloneMetadataValue(value types.MemoryRecordMetadataValue) types.MemoryRecordMetadataValue {
	switch value := value.(type) {
	case *types.MemoryRecordMetadataValueMemberDateTimeValue:
		if value != nil {
			return new(*value)
		}
	case *types.MemoryRecordMetadataValueMemberNumberValue:
		if value != nil {
			return new(*value)
		}
	case *types.MemoryRecordMetadataValueMemberStringValue:
		if value != nil {
			return new(*value)
		}
	case *types.MemoryRecordMetadataValueMemberStringListValue:
		if value != nil {
			copied := *value
			copied.Value = slices.Clone(value.Value)
			return &copied
		}
	}
	return value
}
