package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers"
)

var _ model.LLM = (*Model)(nil)

const rawFunctionArgsJSONKey = "rawArgsJson"

// StreamReader is the subset of Bedrock Converse stream API used by this package.
type StreamReader interface {
	Events() <-chan types.ConverseStreamOutput
	Close() error
	Err() error
}

// RuntimeAPI is the subset of Bedrock Runtime operations needed by this package.
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

const otelTracerName = "github.com/craigh33/adk-go-bedrock/bedrock"

// RuntimeAPIOption configures [NewRuntimeAPI].
type RuntimeAPIOption func(*runtimeAdapter)

// WithTracerProvider sets the OpenTelemetry [trace.TracerProvider] used for
// Bedrock runtime spans. When omitted, [otel.GetTracerProvider] is used (the
// global provider, or a no-op implementation if none was registered).
func WithTracerProvider(tp trace.TracerProvider) RuntimeAPIOption {
	return func(a *runtimeAdapter) {
		a.tracerProvider = tp
	}
}

type runtimeAdapter struct {
	inner          *bedrockruntime.Client
	tracerProvider trace.TracerProvider
}

// NewRuntimeAPI wraps a [bedrockruntime.Client] as [RuntimeAPI].
func NewRuntimeAPI(c *bedrockruntime.Client, opts ...RuntimeAPIOption) RuntimeAPI {
	a := &runtimeAdapter{inner: c}
	for _, opt := range opts {
		opt(a)
	}
	if a.tracerProvider == nil {
		a.tracerProvider = otel.GetTracerProvider()
	}
	return a
}

func (c *runtimeAdapter) tracer() trace.Tracer {
	return c.tracerProvider.Tracer(otelTracerName)
}

func (c *runtimeAdapter) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	ctx, span := c.tracer().Start(ctx, "bedrockruntime.Converse",
		trace.WithSpanKind(trace.SpanKindClient))
	defer span.End()

	out, err := c.inner.Converse(ctx, params, optFns...)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return nil, err
	}
	span.SetStatus(codes.Ok, "")
	return out, nil
}

func (c *runtimeAdapter) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (StreamReader, error) {
	ctx, span := c.tracer().Start(ctx, "bedrockruntime.ConverseStream",
		trace.WithSpanKind(trace.SpanKindClient))

	out, err := c.inner.ConverseStream(ctx, params, optFns...)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.End()
		return nil, err
	}
	return &tracedStreamReader{inner: out.GetStream(), span: span}, nil
}

// tracedStreamReader wraps a StreamReader and keeps its OTel span alive for
// the entire duration of stream consumption. The span is ended in Close(),
// where stream.Err() is inspected to determine the final status.
type tracedStreamReader struct {
	inner StreamReader
	span  trace.Span
}

func (t *tracedStreamReader) Events() <-chan types.ConverseStreamOutput {
	return t.inner.Events()
}

func (t *tracedStreamReader) Err() error {
	return t.inner.Err()
}

func (t *tracedStreamReader) Close() error {
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

var _ RuntimeAPI = (*runtimeAdapter)(nil)

// Options configures [New] and [NewWithAPI].
type Options struct {
	// Region overrides AWS region (otherwise [config.LoadDefaultConfig] resolution is used).
	Region string
}

// Model implements [model.LLM] using Amazon Bedrock Runtime Converse / ConverseStream.
type Model struct {
	modelID string
	api     RuntimeAPI
}

// New creates a [Model] using the default AWS configuration chain and a new
// [bedrockruntime.Client]. ModelID is the Bedrock model ID or inference profile ARN.
func New(ctx context.Context, modelID string, opts *Options) (*Model, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelID is required")
	}
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("load AWS config: %w", err)
	}
	if opts != nil && opts.Region != "" {
		cfg.Region = opts.Region
	}
	cli := bedrockruntime.NewFromConfig(cfg)
	return NewWithAPI(modelID, NewRuntimeAPI(cli))
}

// NewWithAPI wires a Bedrock runtime implementation.
func NewWithAPI(modelID string, api RuntimeAPI) (*Model, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelID is required")
	}
	if api == nil {
		return nil, errors.New("nil RuntimeAPI")
	}
	return &Model{modelID: modelID, api: api}, nil
}

// Name returns the configured model identifier (see [New]).
func (m *Model) Name() string {
	if m == nil {
		return ""
	}
	return m.modelID
}

// GenerateContent calls Bedrock Converse or ConverseStream.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m == nil || m.api == nil {
			yield(nil, errors.New("nil bedrock Model"))
			return
		}
		modelID := m.modelID
		if req != nil && req.Model != "" {
			modelID = req.Model
		}
		if stream {
			m.generateStream(ctx, modelID, req)(yield)
			return
		}
		m.generateUnary(ctx, modelID, req)(yield)
	}
}

func (m *Model) generateUnary(
	ctx context.Context,
	modelID string,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		in, err := mappers.ConverseInputFromLLMRequest(modelID, req)
		if err != nil {
			yield(nil, err)
			return
		}
		out, err := m.api.Converse(ctx, in)
		if err != nil {
			yield(nil, err)
			return
		}
		resp, err := mappers.LLMResponseFromConverseOutput(out)
		if !yield(resp, err) {
			return
		}
	}
}
func (m *Model) generateStream(
	ctx context.Context,
	modelID string,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		in, err := mappers.ConverseStreamInputFromLLMRequest(modelID, req)
		if err != nil {
			yield(nil, err)
			return
		}
		stream, err := m.api.ConverseStream(ctx, in)
		if err != nil {
			yield(nil, err)
			return
		}
		defer func() {
			_ = stream.Close()
		}()

		state := newStreamState()
		for ev := range stream.Events() {
			partial, err := state.consumeEvent(ev)
			if err != nil {
				yield(nil, err)
				return
			}
			if partial != nil && !yield(partial, nil) {
				return
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		if !yield(state.finalResponse(), nil) {
			return
		}
	}
}

type streamState struct {
	textBuf           strings.Builder
	lastYieldedLen    int
	textBySlot        map[int32]*streamTextBlock
	lastYieldedBySlot map[int32]int
	toolsBySlot       map[int32]*streamToolCall
	imagesBySlot      map[int32]*streamImageBlock
	reasonBySlot      map[int32]*streamReasoningBlock
	slotOrder         []int32
	lastUsage         *genai.GenerateContentResponseUsageMetadata
	customMetadata    map[string]any
	guardrailTrace    *types.GuardrailTraceAssessment
	stopReason        types.StopReason
}

type streamTextBlock struct {
	Text strings.Builder
}

type streamToolCall struct {
	ID    string
	Name  string
	input strings.Builder
}

type streamImageBlock struct {
	Format types.ImageFormat
	Data   []byte
}

type streamReasoningBlock struct {
	Text      strings.Builder
	Signature string
}

func newStreamState() *streamState {
	return &streamState{
		textBySlot:        make(map[int32]*streamTextBlock),
		lastYieldedBySlot: make(map[int32]int),
		toolsBySlot:       make(map[int32]*streamToolCall),
		imagesBySlot:      make(map[int32]*streamImageBlock),
		reasonBySlot:      make(map[int32]*streamReasoningBlock),
	}
}

func (s *streamState) consumeEvent(ev types.ConverseStreamOutput) (*model.LLMResponse, error) {
	switch v := ev.(type) {
	case *types.ConverseStreamOutputMemberContentBlockStart:
		s.onContentBlockStart(&v.Value)
	case *types.ConverseStreamOutputMemberContentBlockDelta:
		return s.onContentBlockDelta(&v.Value)
	case *types.ConverseStreamOutputMemberMessageStop:
		s.stopReason = v.Value.StopReason
	case *types.ConverseStreamOutputMemberMetadata:
		s.lastUsage = mappers.StreamMetadataToUsage(&v.Value)
		s.mergeCustomMetadata(mappers.StreamMetadataToCustomMetadata(&v.Value))
		if v.Value.Trace != nil {
			s.guardrailTrace = v.Value.Trace.Guardrail
		}
	default:
		// Ignore stream variants we do not map yet; Bedrock may add new event types over time.
	}
	return nil, nil //nolint:nilnil // Stream event type does not emit an intermediate response.
}

func (s *streamState) onContentBlockStart(ev *types.ContentBlockStartEvent) {
	if ev == nil || ev.ContentBlockIndex == nil {
		return
	}
	idx := *ev.ContentBlockIndex
	s.rememberSlot(idx)
	switch st := ev.Start.(type) {
	case *types.ContentBlockStartMemberToolUse:
		call := s.ensureToolCall(idx)
		if st.Value.ToolUseId != nil {
			call.ID = *st.Value.ToolUseId
		}
		if st.Value.Name != nil {
			call.Name = *st.Value.Name
		}
	case *types.ContentBlockStartMemberImage:
		img := s.ensureImageBlock(idx)
		img.Format = st.Value.Format
	}
}

//nolint:gocognit // Streaming unions require branching per delta type.
func (s *streamState) onContentBlockDelta(ev *types.ContentBlockDeltaEvent) (*model.LLMResponse, error) {
	if ev == nil || ev.ContentBlockIndex == nil {
		return nil, nil //nolint:nilnil // Missing delta index means nothing to emit.
	}
	switch d := ev.Delta.(type) {
	case *types.ContentBlockDeltaMemberText:
		idx := *ev.ContentBlockIndex
		s.rememberSlot(idx)
		if d.Value == "" {
			return nil, nil //nolint:nilnil // Empty text delta does not produce output.
		}
		// Track text in the per-slot buffer for final assembly
		textBlock := s.ensureTextBlock(idx)
		if _, err := textBlock.Text.WriteString(d.Value); err != nil {
			return nil, err
		}
		// Track in legacy textBuf for streaming delta calculation
		if _, err := s.textBuf.WriteString(d.Value); err != nil {
			return nil, err
		}
		// Yield only the delta (new text since last yield)
		delta := s.textBuf.String()[s.lastYieldedLen:]
		s.lastYieldedLen = s.textBuf.Len()
		return &model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{{
					Text: delta,
				}},
			},
			Partial: true,
		}, nil
	case *types.ContentBlockDeltaMemberImage:
		img := s.ensureImageBlock(*ev.ContentBlockIndex)
		if d.Value.Source == nil {
			return nil, nil //nolint:nilnil // Empty image delta does not produce output.
		}
		if src, ok := d.Value.Source.(*types.ImageSourceMemberBytes); ok {
			img.Data = append(img.Data, src.Value...)
		}
		return nil, nil //nolint:nilnil // Image deltas are buffered until final response.
	case *types.ContentBlockDeltaMemberReasoningContent:
		reason := s.ensureReasoningBlock(*ev.ContentBlockIndex)
		switch delta := d.Value.(type) {
		case *types.ReasoningContentBlockDeltaMemberText:
			if _, err := reason.Text.WriteString(delta.Value); err != nil {
				return nil, err
			}
		case *types.ReasoningContentBlockDeltaMemberSignature:
			reason.Signature = delta.Value
		}
		return nil, nil //nolint:nilnil // Reasoning deltas are buffered until final response.
	case *types.ContentBlockDeltaMemberToolUse:
		call := s.ensureToolCall(*ev.ContentBlockIndex)
		if d.Value.Input != nil {
			if _, err := call.input.WriteString(*d.Value.Input); err != nil {
				return nil, err
			}
		}
		return nil, nil //nolint:nilnil // Tool input deltas are buffered until final response.
	default:
		return nil, nil //nolint:nilnil // Unsupported delta type is intentionally ignored.
	}
}

func (s *streamState) ensureToolCall(idx int32) *streamToolCall {
	s.rememberSlot(idx)
	if call, ok := s.toolsBySlot[idx]; ok {
		return call
	}
	call := &streamToolCall{}
	s.toolsBySlot[idx] = call
	return call
}

func (s *streamState) ensureTextBlock(idx int32) *streamTextBlock {
	s.rememberSlot(idx)
	if text, ok := s.textBySlot[idx]; ok {
		return text
	}
	text := &streamTextBlock{}
	s.textBySlot[idx] = text
	return text
}

func (s *streamState) ensureImageBlock(idx int32) *streamImageBlock {
	s.rememberSlot(idx)
	if img, ok := s.imagesBySlot[idx]; ok {
		return img
	}
	img := &streamImageBlock{}
	s.imagesBySlot[idx] = img
	return img
}

func (s *streamState) ensureReasoningBlock(idx int32) *streamReasoningBlock {
	s.rememberSlot(idx)
	if reason, ok := s.reasonBySlot[idx]; ok {
		return reason
	}
	reason := &streamReasoningBlock{}
	s.reasonBySlot[idx] = reason
	return reason
}

func (s *streamState) rememberSlot(idx int32) {
	if slices.Contains(s.slotOrder, idx) {
		return
	}
	s.slotOrder = append(s.slotOrder, idx)
}

func (s *streamState) mergeCustomMetadata(md map[string]any) {
	if len(md) == 0 {
		return
	}
	if s.customMetadata == nil {
		s.customMetadata = map[string]any{}
	}
	maps.Copy(s.customMetadata, md)
}

func (s *streamState) finalResponse() *model.LLMResponse {
	parts, unsupportedFormats := s.finalParts()
	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}

	// Store any unsupported image formats in custom metadata so callers can detect them
	if len(unsupportedFormats) > 0 {
		if s.customMetadata == nil {
			s.customMetadata = map[string]any{}
		}
		s.customMetadata["unsupported_image_formats"] = unsupportedFormats
	}

	return &model.LLMResponse{
		Content:        &genai.Content{Role: "model", Parts: parts},
		FinishReason:   mappers.FinishReasonFromStopReasonAndTrace(s.stopReason, s.guardrailTrace),
		UsageMetadata:  s.lastUsage,
		CustomMetadata: s.customMetadata,
		TurnComplete:   true,
	}
}

func (s *streamState) finalParts() ([]*genai.Part, []string) { //nolint:gocognit
	parts := make([]*genai.Part, 0, len(s.slotOrder))
	unsupportedFormats := []string{} // Track unsupported image formats

	// Assemble all parts in strict slot order to preserve Bedrock block ordering
	for _, idx := range s.sortedSlotOrder() {
		// Emit text for this slot
		if text := s.textBySlot[idx]; text != nil && text.Text.Len() > 0 {
			parts = append(parts, &genai.Part{Text: text.Text.String()})
		}

		// Emit reasoning for this slot
		if reason := s.reasonBySlot[idx]; reason != nil && reason.Text.Len() > 0 {
			part := &genai.Part{Text: reason.Text.String(), Thought: true}
			if reason.Signature != "" {
				part.ThoughtSignature = []byte(reason.Signature)
			}
			parts = append(parts, part)
		}

		// Emit image for this slot
		if img := s.imagesBySlot[idx]; img != nil && len(img.Data) > 0 {
			if mime, err := streamImageMIMEFromFormat(img.Format); err == nil {
				parts = append(parts, &genai.Part{InlineData: &genai.Blob{Data: img.Data, MIMEType: mime}})
			} else {
				// Track unsupported format instead of silently dropping the image
				format := fmt.Sprintf("unsupported_format_at_slot_%d: %v", idx, err)
				unsupportedFormats = append(unsupportedFormats, format)
			}
		}

		// Emit tool call for this slot
		if call := s.toolsBySlot[idx]; call != nil {
			id := call.ID
			if id == "" {
				id = fmt.Sprintf("tool_%d", idx)
			}
			name := call.Name
			if name == "" {
				name = id
			}
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{
				ID:   id,
				Name: name,
				Args: functionArgsFromRawJSON(call.input.String()),
			}})
		}
	}
	return parts, unsupportedFormats
}

func (s *streamState) sortedSlotOrder() []int32 {
	if len(s.slotOrder) < 2 {
		return s.slotOrder
	}
	order := append([]int32(nil), s.slotOrder...)
	for i := len(order) - 1; i > 0; i-- {
		for j := range i {
			if order[j] > order[j+1] {
				order[j], order[j+1] = order[j+1], order[j]
			}
		}
	}
	return order
}

func streamImageMIMEFromFormat(f types.ImageFormat) (string, error) {
	switch f {
	case types.ImageFormatJpeg:
		return "image/jpeg", nil
	case types.ImageFormatPng:
		return "image/png", nil
	case types.ImageFormatGif:
		return "image/gif", nil
	case types.ImageFormatWebp:
		return "image/webp", nil
	default:
		return "", fmt.Errorf("unsupported Bedrock stream image format: %q", f)
	}
}

func functionArgsFromRawJSON(raw string) map[string]any {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]any{}
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
		return parsed
	}
	return map[string]any{rawFunctionArgsJSONKey: raw}
}
