// Package client defines the Bedrock Runtime API surface used by the bedrock converse provider.
package client

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// StreamReader is the subset of the Converse stream API used by the model provider.
type StreamReader interface {
	Events() <-chan types.ConverseStreamOutput
	Close() error
	Err() error
}

// RuntimeAPI is the subset of Bedrock Runtime operations used by the converse implementation (mockable in tests).
type RuntimeAPI interface {
	Converse(
		ctx context.Context,
		params *bedrockruntime.ConverseInput,
		optFns ...func(*bedrockruntime.Options),
	) (*bedrockruntime.ConverseOutput, error)
	ConverseStream(
		ctx context.Context,
		params *bedrockruntime.ConverseStreamInput,
		optFns ...func(*bedrockruntime.Options),
	) (StreamReader, error)
}

type adapter struct {
	inner *bedrockruntime.Client
}

// NewFromClient wraps a [bedrockruntime.Client] as [RuntimeAPI].
func NewFromClient(c *bedrockruntime.Client) RuntimeAPI {
	return &adapter{inner: c}
}

func (c *adapter) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return c.inner.Converse(ctx, params, optFns...)
}

func (c *adapter) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (StreamReader, error) {
	out, err := c.inner.ConverseStream(ctx, params, optFns...)
	if err != nil {
		return nil, err
	}
	return out.GetStream(), nil
}

var _ RuntimeAPI = (*adapter)(nil)
