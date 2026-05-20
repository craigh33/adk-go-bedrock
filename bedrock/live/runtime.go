// Package live implements Amazon Nova Sonic bidirectional ("Live") streaming
// against the adk-go [agent.LiveSession] interface.
//
// The Bedrock backend uses [bedrockruntime.Client.InvokeModelWithBidirectionalStream]
// which exposes opaque [BidirectionalInputPayloadPart] / [BidirectionalOutputPayloadPart]
// byte chunks. The JSON event envelopes documented at
// https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html are
// marshaled and unmarshaled inside this package.
//
// Unlike [bedrock.Model], this package does not plug into adk-go's
// [runner.Runner.RunLive] — that path hardcodes a *genai.Client live connection.
// Callers drive the Bedrock session directly via [Open].
package live

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const otelTracerName = "github.com/craigh33/adk-go-bedrock/bedrock/live"

// BidiStream is the subset of [bedrockruntime.InvokeModelWithBidirectionalStreamEventStream]
// used by [Session]. Tests substitute a fake implementation.
type BidiStream interface {
	Send(ctx context.Context, input types.InvokeModelWithBidirectionalStreamInput) error
	Events() <-chan types.InvokeModelWithBidirectionalStreamOutput
	Close() error
	Err() error
}

// BidiRuntimeAPI is the subset of Bedrock Runtime operations [Session] needs.
type BidiRuntimeAPI interface {
	InvokeModelWithBidirectionalStream(
		ctx context.Context,
		params *bedrockruntime.InvokeModelWithBidirectionalStreamInput,
		optFns ...func(*bedrockruntime.Options),
	) (BidiStream, error)
}

// RuntimeAPIOption configures [NewBidiRuntimeAPI].
type RuntimeAPIOption func(*runtimeAdapter)

// WithTracerProvider sets the OpenTelemetry [trace.TracerProvider] used for
// Bedrock bidi spans. When omitted, [otel.GetTracerProvider] is used.
func WithTracerProvider(tp trace.TracerProvider) RuntimeAPIOption {
	return func(a *runtimeAdapter) {
		a.tracerProvider = tp
	}
}

// NewBidiRuntimeAPI wraps a [bedrockruntime.Client] as [BidiRuntimeAPI].
func NewBidiRuntimeAPI(c *bedrockruntime.Client, opts ...RuntimeAPIOption) BidiRuntimeAPI {
	a := &runtimeAdapter{inner: c}
	for _, opt := range opts {
		opt(a)
	}
	if a.tracerProvider == nil {
		a.tracerProvider = otel.GetTracerProvider()
	}
	return a
}

type runtimeAdapter struct {
	inner          *bedrockruntime.Client
	tracerProvider trace.TracerProvider
}

func (c *runtimeAdapter) tracer() trace.Tracer {
	return c.tracerProvider.Tracer(otelTracerName)
}

func (c *runtimeAdapter) InvokeModelWithBidirectionalStream(
	ctx context.Context,
	params *bedrockruntime.InvokeModelWithBidirectionalStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (BidiStream, error) {
	ctx, span := c.tracer().Start(ctx, "bedrockruntime.InvokeModelWithBidirectionalStream",
		trace.WithSpanKind(trace.SpanKindClient))

	out, err := c.inner.InvokeModelWithBidirectionalStream(ctx, params, optFns...)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.End()
		return nil, err
	}
	return &tracedBidiStream{inner: out.GetStream(), span: span}, nil
}

var _ BidiRuntimeAPI = (*runtimeAdapter)(nil)

// tracedBidiStream wraps the SDK's bidi stream and keeps the OTel span alive
// for the entire session.
type tracedBidiStream struct {
	inner *bedrockruntime.InvokeModelWithBidirectionalStreamEventStream
	span  trace.Span
}

func (t *tracedBidiStream) Send(ctx context.Context, input types.InvokeModelWithBidirectionalStreamInput) error {
	return t.inner.Send(ctx, input)
}

func (t *tracedBidiStream) Events() <-chan types.InvokeModelWithBidirectionalStreamOutput {
	return t.inner.Events()
}

func (t *tracedBidiStream) Err() error {
	return t.inner.Err()
}

func (t *tracedBidiStream) Close() error {
	streamErr := t.inner.Err()
	closeErr := t.inner.Close()
	if streamErr != nil {
		t.span.RecordError(streamErr)
		t.span.SetStatus(codes.Error, streamErr.Error())
	} else {
		t.span.SetStatus(codes.Ok, "")
	}
	t.span.End()
	return closeErr
}
