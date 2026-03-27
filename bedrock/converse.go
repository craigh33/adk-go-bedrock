package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/mappers"
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

type runtimeAdapter struct {
	inner *bedrockruntime.Client
}

// NewRuntimeAPI wraps a [bedrockruntime.Client] as [RuntimeAPI].
func NewRuntimeAPI(c *bedrockruntime.Client) RuntimeAPI {
	return &runtimeAdapter{inner: c}
}

func (c *runtimeAdapter) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return c.inner.Converse(ctx, params, optFns...)
}

func (c *runtimeAdapter) ConverseStream(
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
	textBuf        strings.Builder
	lastYieldedLen int
	toolsBySlot    map[int32]*streamToolCall
	toolOrder      []int32

	lastUsage  *genai.GenerateContentResponseUsageMetadata
	stopReason types.StopReason
}

type streamToolCall struct {
	ID   string
	Name string

	input strings.Builder
}

func newStreamState() *streamState {
	return &streamState{toolsBySlot: make(map[int32]*streamToolCall)}
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
	default:
	}
	return nil, nil //nolint:nilnil // Stream event type does not emit an intermediate response.
}

func (s *streamState) onContentBlockStart(ev *types.ContentBlockStartEvent) {
	if ev == nil || ev.ContentBlockIndex == nil {
		return
	}
	st, ok := ev.Start.(*types.ContentBlockStartMemberToolUse)
	if !ok {
		return
	}
	idx := *ev.ContentBlockIndex
	call := s.ensureToolCall(idx)
	if st.Value.ToolUseId != nil {
		call.ID = *st.Value.ToolUseId
	}
	if st.Value.Name != nil {
		call.Name = *st.Value.Name
	}
}

func (s *streamState) onContentBlockDelta(ev *types.ContentBlockDeltaEvent) (*model.LLMResponse, error) {
	if ev == nil || ev.ContentBlockIndex == nil {
		return nil, nil //nolint:nilnil // Missing delta index means nothing to emit.
	}
	switch d := ev.Delta.(type) {
	case *types.ContentBlockDeltaMemberText:
		if d.Value == "" {
			return nil, nil //nolint:nilnil // Empty text delta does not produce output.
		}
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
	if call, ok := s.toolsBySlot[idx]; ok {
		return call
	}
	call := &streamToolCall{}
	s.toolsBySlot[idx] = call
	s.toolOrder = append(s.toolOrder, idx)
	return call
}

func (s *streamState) finalResponse() *model.LLMResponse {
	parts := s.finalParts()
	if len(parts) == 0 {
		parts = []*genai.Part{{Text: ""}}
	}
	return &model.LLMResponse{
		Content:       &genai.Content{Role: "model", Parts: parts},
		FinishReason:  mappers.StopReasonToFinishReason(s.stopReason),
		UsageMetadata: s.lastUsage,
		TurnComplete:  true,
	}
}

func (s *streamState) finalParts() []*genai.Part {
	parts := make([]*genai.Part, 0, 1+len(s.toolOrder))
	if s.textBuf.Len() > 0 {
		parts = append(parts, &genai.Part{Text: s.textBuf.String()})
	}
	for _, idx := range s.sortedToolOrder() {
		call := s.toolsBySlot[idx]
		if call == nil {
			continue
		}
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
	return parts
}

func (s *streamState) sortedToolOrder() []int32 {
	if len(s.toolOrder) < 2 {
		return s.toolOrder
	}
	order := append([]int32(nil), s.toolOrder...)
	for i := len(order) - 1; i > 0; i-- {
		for j := range i {
			if order[j] > order[j+1] {
				order[j], order[j+1] = order[j+1], order[j]
			}
		}
	}
	return order
}

func functionArgsFromRawJSON(raw string) map[string]any {
	args := map[string]any{}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return args
	}

	args[rawFunctionArgsJSONKey] = raw

	var parsed map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
		maps.Copy(args, parsed)
	}
	return args
}
