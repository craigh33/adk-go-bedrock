package mantle

import (
	"context"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	anthropicbedrock "github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/craigh33/adk-go-bedrock/bedrock"
)

// MessagesAPI is the subset of the Anthropic Messages API used by the Mantle
// transport. It is satisfied by [*github.com/anthropics/anthropic-sdk-go.MessageService]
// and kept small so tests can substitute a fake without a live endpoint.
type MessagesAPI interface {
	New(
		ctx context.Context,
		params anthropic.MessageNewParams,
		opts ...option.RequestOption,
	) (*anthropic.Message, error)
	NewStreaming(
		ctx context.Context,
		params anthropic.MessageNewParams,
		opts ...option.RequestOption,
	) *ssestream.Stream[anthropic.MessageStreamEventUnion]
}

// Config configures [New]. It is a focused subset of
// [github.com/anthropics/anthropic-sdk-go/bedrock.MantleClientConfig]; any field
// left empty falls back to that client's own resolution (environment variables,
// the default AWS credential chain, and the region-derived base URL).
type Config struct {
	// APIKey authenticates via the x-api-key header. Takes precedence over AWS
	// credentials when set.
	APIKey string
	// AWSRegion selects the Mantle base URL and SigV4 signing region.
	AWSRegion string
	// AWSProfile resolves SigV4 credentials via the named profile.
	AWSProfile string
	// BaseURL overrides the default region-derived Mantle base URL.
	BaseURL string
	// SkipAuth skips Mantle-specific authentication, e.g. when a gateway signs
	// requests on your behalf.
	SkipAuth bool
}

func (c Config) toMantleConfig() anthropicbedrock.MantleClientConfig {
	return anthropicbedrock.MantleClientConfig{
		APIKey:     c.APIKey,
		AWSRegion:  c.AWSRegion,
		AWSProfile: c.AWSProfile,
		BaseURL:    c.BaseURL,
		SkipAuth:   c.SkipAuth,
	}
}

// Client adapts the Anthropic Bedrock Mantle Messages API to the Bedrock
// [github.com/craigh33/adk-go-bedrock/bedrock.RuntimeAPI] interface. Pair it with
// [github.com/craigh33/adk-go-bedrock/bedrock.NewWithAPI] to drive a
// [github.com/craigh33/adk-go-bedrock/bedrock.Model] over Mantle instead of Converse.
type Client struct {
	messages MessagesAPI
}

var _ bedrock.RuntimeAPI = (*Client)(nil)

// New builds a [Client] backed by a live Anthropic Bedrock Mantle client.
func New(ctx context.Context, cfg Config, opts ...option.RequestOption) (*Client, error) {
	mc, err := anthropicbedrock.NewMantleClient(ctx, cfg.toMantleConfig(), opts...)
	if err != nil {
		return nil, fmt.Errorf("create Bedrock Mantle client: %w", err)
	}
	return &Client{messages: &mc.Messages}, nil
}

// NewWithMessages builds a [Client] over an arbitrary [MessagesAPI]
// implementation. It exists primarily for tests and advanced wiring.
func NewWithMessages(messages MessagesAPI) (*Client, error) {
	if messages == nil {
		return nil, errors.New("nil MessagesAPI")
	}
	return &Client{messages: messages}, nil
}

// Converse implements the unary call path of
// [github.com/craigh33/adk-go-bedrock/bedrock.RuntimeAPI] by translating the
// Converse request to the Anthropic Messages API and mapping the reply back to a
// [bedrockruntime.ConverseOutput]. The Bedrock runtime option functions do not
// apply to the Mantle transport and are ignored.
func (c *Client) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	_ ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	if c == nil || c.messages == nil {
		return nil, errors.New("nil Mantle client")
	}
	msgParams, err := MessageParamsFromConverseInput(params)
	if err != nil {
		return nil, err
	}
	msg, err := c.messages.New(ctx, msgParams)
	if err != nil {
		return nil, fmt.Errorf("bedrock Mantle Converse: %w", err)
	}
	return ConverseOutputFromMessage(msg)
}

// ConverseStream implements the streaming call path of
// [github.com/craigh33/adk-go-bedrock/bedrock.RuntimeAPI] by opening an Anthropic
// Messages SSE stream and adapting its events into the Converse stream variants
// that the Model's stream assembly consumes. The Bedrock runtime option functions
// do not apply to the Mantle transport and are ignored.
func (c *Client) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	_ ...func(*bedrockruntime.Options),
) (bedrock.StreamReader, error) {
	if c == nil || c.messages == nil {
		return nil, errors.New("nil Mantle client")
	}
	msgParams, err := MessageParamsFromConverseStreamInput(params)
	if err != nil {
		return nil, err
	}
	return newConverseStream(c.messages.NewStreaming(ctx, msgParams)), nil
}
