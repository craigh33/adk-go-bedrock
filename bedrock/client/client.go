// Package client defines the Bedrock Runtime API surface used by the bedrock converse provider.
package client

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
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

// Option configures an adapter created with [NewFromClient].
type Option func(*adapter)

// WithTracer attaches an OpenTelemetry [trace.Tracer] used to record spans for
// [RuntimeAPI.Converse] and [RuntimeAPI.ConverseStream].
func WithTracer(tracer trace.Tracer) Option {
	return func(a *adapter) {
		a.tracer = tracer
	}
}

type adapter struct {
	inner  *bedrockruntime.Client
	tracer trace.Tracer
}

// NewFromClient wraps a [bedrockruntime.Client] as [RuntimeAPI].
func NewFromClient(c *bedrockruntime.Client, opts ...Option) RuntimeAPI {
	a := &adapter{inner: c}
	for _, opt := range opts {
		if opt != nil {
			opt(a)
		}
	}
	return a
}

func (c *adapter) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	if c.tracer != nil {
		ctx, span := c.tracer.Start(ctx, "bedrock.Converse",
			trace.WithSpanKind(trace.SpanKindClient),
		)
		defer span.End()
		if params != nil && params.ModelId != nil && *params.ModelId != "" {
			span.SetAttributes(attribute.String("aws.bedrock.model_id", *params.ModelId))
		}
		out, err := c.inner.Converse(ctx, params, optFns...)
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
			return nil, err
		}
		return out, nil
	}
	return c.inner.Converse(ctx, params, optFns...)
}

func (c *adapter) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (StreamReader, error) {
	if c.tracer != nil {
		ctx, span := c.tracer.Start(ctx, "bedrock.ConverseStream",
			trace.WithSpanKind(trace.SpanKindClient),
		)
		defer span.End()
		if params != nil && params.ModelId != nil && *params.ModelId != "" {
			span.SetAttributes(attribute.String("aws.bedrock.model_id", *params.ModelId))
		}
		out, err := c.inner.ConverseStream(ctx, params, optFns...)
		if err != nil {
			span.RecordError(err)
			span.SetStatus(codes.Error, err.Error())
			return nil, err
		}
		return out.GetStream(), nil
	}
	out, err := c.inner.ConverseStream(ctx, params, optFns...)
	if err != nil {
		return nil, err
	}
	return out.GetStream(), nil
}

var _ RuntimeAPI = (*adapter)(nil)
