package mantle

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
)

// fakeDecoder replays a fixed list of SSE events for ssestream.NewStream.
type fakeDecoder struct {
	events []ssestream.Event
	idx    int
}

func (d *fakeDecoder) Next() bool {
	if d.idx >= len(d.events) {
		return false
	}
	d.idx++
	return true
}

func (d *fakeDecoder) Event() ssestream.Event { return d.events[d.idx-1] }
func (d *fakeDecoder) Close() error           { return nil }
func (d *fakeDecoder) Err() error             { return nil }

func newTestStream(events []ssestream.Event) *ssestream.Stream[anthropic.MessageStreamEventUnion] {
	return ssestream.NewStream[anthropic.MessageStreamEventUnion](&fakeDecoder{events: events}, nil)
}

func sse(t *testing.T, eventType string, payload map[string]any) ssestream.Event {
	t.Helper()
	payload["type"] = eventType
	data, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal sse payload: %v", err)
	}
	return ssestream.Event{Type: eventType, Data: data}
}

// anthropicStreamEvents is a representative streamed reply: two text deltas,
// then a tool call whose JSON input arrives in two partial chunks, then usage
// and a tool_use stop reason.
func anthropicStreamEvents(t *testing.T) []ssestream.Event {
	t.Helper()
	return []ssestream.Event{
		sse(t, eventMessageStart, map[string]any{
			"message": map[string]any{"usage": map[string]any{"input_tokens": 10}},
		}),
		sse(t, eventContentBlockStart, map[string]any{
			"index": 0, "content_block": map[string]any{"type": "text"},
		}),
		sse(t, eventContentBlockDelta, map[string]any{
			"index": 0, "delta": map[string]any{"type": deltaText, "text": "Hi"},
		}),
		sse(t, eventContentBlockDelta, map[string]any{
			"index": 0, "delta": map[string]any{"type": deltaText, "text": " there"},
		}),
		sse(t, "content_block_stop", map[string]any{"index": 0}),
		sse(t, eventContentBlockStart, map[string]any{
			"index":         1,
			"content_block": map[string]any{"type": "tool_use", "id": "tool_1", "name": "get_weather"},
		}),
		sse(t, eventContentBlockDelta, map[string]any{
			"index": 1, "delta": map[string]any{"type": deltaInputJSON, "partial_json": `{"city":`},
		}),
		sse(t, eventContentBlockDelta, map[string]any{
			"index": 1, "delta": map[string]any{"type": deltaInputJSON, "partial_json": `"NYC"}`},
		}),
		sse(t, "content_block_stop", map[string]any{"index": 1}),
		sse(t, eventMessageDelta, map[string]any{
			"delta": map[string]any{"stop_reason": "tool_use"},
			"usage": map[string]any{"output_tokens": 7},
		}),
		sse(t, "message_stop", map[string]any{}),
	}
}

// equivalentConverseEvents are the native Converse stream events that represent
// the same reply as anthropicStreamEvents.
func equivalentConverseEvents() []types.ConverseStreamOutput {
	return []types.ConverseStreamOutput{
		textDeltaEvent(0, "Hi"),
		textDeltaEvent(0, " there"),
		toolStartEvent(1, "tool_1", "get_weather"),
		toolInputDeltaEvent(1, `{"city":`),
		toolInputDeltaEvent(1, `"NYC"}`),
		metadataEvent(10, 7),
		messageStopEvent(types.StopReasonToolUse),
	}
}

func TestEventTranslator(t *testing.T) {
	var tr eventTranslator

	if got := tr.translate(unionFrom(t, sse(t, eventMessageStart, map[string]any{
		"message": map[string]any{"usage": map[string]any{"input_tokens": 10}},
	}))); got != nil {
		t.Errorf("message_start should emit nothing, got %+v", got)
	}
	if tr.inputTokens != 10 {
		t.Errorf("inputTokens = %d, want 10", tr.inputTokens)
	}

	text := tr.translate(unionFrom(t, sse(t, eventContentBlockDelta, map[string]any{
		"index": 0, "delta": map[string]any{"type": deltaText, "text": "Hi"},
	})))
	assertSingleTextDelta(t, text, "Hi")

	start := tr.translate(unionFrom(t, sse(t, eventContentBlockStart, map[string]any{
		"index":         1,
		"content_block": map[string]any{"type": "tool_use", "id": "tool_1", "name": "get_weather"},
	})))
	if len(start) != 1 {
		t.Fatalf("tool start emitted %d events, want 1", len(start))
	}
	if _, ok := start[0].(*types.ConverseStreamOutputMemberContentBlockStart); !ok {
		t.Errorf("tool start type = %T", start[0])
	}

	msgDelta := tr.translate(unionFrom(t, sse(t, eventMessageDelta, map[string]any{
		"delta": map[string]any{"stop_reason": "end_turn"},
		"usage": map[string]any{"output_tokens": 7},
	})))
	if len(msgDelta) != 2 {
		t.Fatalf("message_delta emitted %d events, want 2 (metadata + stop)", len(msgDelta))
	}
}

func TestConverseStream_Wiring(t *testing.T) {
	stream := newConverseStream(newTestStream(anthropicStreamEvents(t)))
	defer func() { _ = stream.Close() }()

	var got []types.ConverseStreamOutput
	for ev := range stream.Events() {
		got = append(got, ev)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	// 2 text deltas + 1 tool start + 2 tool input deltas + metadata + stop = 7.
	if len(got) != 7 {
		t.Errorf("emitted %d events, want 7: %+v", len(got), got)
	}
}

func TestConverseStream_ErrorPropagation(t *testing.T) {
	events := []ssestream.Event{
		sse(t, eventContentBlockDelta, map[string]any{
			"index": 0, "delta": map[string]any{"type": deltaText, "text": "Hi"},
		}),
		{Type: "error", Data: []byte(`{"error":{"message":"overloaded"}}`)},
	}
	stream := newConverseStream(newTestStream(events))
	for range stream.Events() {
	}
	if stream.Err() == nil {
		t.Error("expected streaming error to propagate via Err()")
	}
}

// TestStreamingParity proves the Mantle streaming adapter yields the same final
// LLMResponse as the native Converse path for an equivalent reply.
func TestStreamingParity(t *testing.T) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("weather in NYC?", genai.RoleUser)},
	}

	events := anthropicStreamEvents(t)
	mantleClient, err := NewWithMessages(&fakeMessages{
		streamFunc: func(anthropic.MessageNewParams) *ssestream.Stream[anthropic.MessageStreamEventUnion] {
			return newTestStream(events)
		},
	})
	if err != nil {
		t.Fatalf("NewWithMessages: %v", err)
	}
	mantleLLM, err := bedrock.NewWithAPI("anthropic.claude-3-haiku", mantleClient)
	if err != nil {
		t.Fatalf("NewWithAPI (mantle): %v", err)
	}
	mantleResp := lastStreamResponse(t, mantleLLM, req)

	converseLLM, err := bedrock.NewWithAPI("anthropic.claude-3-haiku", &fakeConverseAPI{
		events: equivalentConverseEvents(),
	})
	if err != nil {
		t.Fatalf("NewWithAPI (converse): %v", err)
	}
	converseResp := lastStreamResponse(t, converseLLM, req)

	if !reflect.DeepEqual(mantleResp.Content, converseResp.Content) {
		t.Errorf("content mismatch:\n  mantle   = %s\n  converse = %s",
			mustJSON(t, mantleResp.Content), mustJSON(t, converseResp.Content))
	}
	if !reflect.DeepEqual(mantleResp.UsageMetadata, converseResp.UsageMetadata) {
		t.Errorf("usage mismatch: mantle=%+v converse=%+v", mantleResp.UsageMetadata, converseResp.UsageMetadata)
	}
	if mantleResp.FinishReason != converseResp.FinishReason {
		t.Errorf("finish reason mismatch: mantle=%q converse=%q", mantleResp.FinishReason, converseResp.FinishReason)
	}
}

// --- helpers ---

func unionFrom(t *testing.T, ev ssestream.Event) anthropic.MessageStreamEventUnion {
	t.Helper()
	var u anthropic.MessageStreamEventUnion
	if err := json.Unmarshal(ev.Data, &u); err != nil {
		t.Fatalf("unmarshal stream event: %v", err)
	}
	return u
}

func assertSingleTextDelta(t *testing.T, got []types.ConverseStreamOutput, want string) {
	t.Helper()
	if len(got) != 1 {
		t.Fatalf("emitted %d events, want 1", len(got))
	}
	delta, ok := got[0].(*types.ConverseStreamOutputMemberContentBlockDelta)
	if !ok {
		t.Fatalf("event type = %T, want content block delta", got[0])
	}
	text, ok := delta.Value.Delta.(*types.ContentBlockDeltaMemberText)
	if !ok || text.Value != want {
		t.Errorf("delta = %+v, want text %q", delta.Value.Delta, want)
	}
}

func lastStreamResponse(t *testing.T, llm *bedrock.Model, req *model.LLMRequest) *model.LLMResponse {
	t.Helper()
	var last *model.LLMResponse
	for resp, err := range llm.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if resp != nil && resp.TurnComplete {
			last = resp
		}
	}
	if last == nil {
		t.Fatal("no final (TurnComplete) response")
	}
	return last
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(data)
}

func textDeltaEvent(idx int32, text string) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(idx),
		Delta:             &types.ContentBlockDeltaMemberText{Value: text},
	}}
}

func toolStartEvent(idx int32, id, name string) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: aws.Int32(idx),
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
			ToolUseId: aws.String(id),
			Name:      aws.String(name),
		}},
	}}
}

func toolInputDeltaEvent(idx int32, partial string) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: aws.Int32(idx),
		Delta: &types.ContentBlockDeltaMemberToolUse{
			Value: types.ToolUseBlockDelta{Input: aws.String(partial)},
		},
	}}
}

func metadataEvent(in, out int32) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
		Usage: &types.TokenUsage{
			InputTokens:  aws.Int32(in),
			OutputTokens: aws.Int32(out),
			TotalTokens:  aws.Int32(in + out),
		},
	}}
}

func messageStopEvent(sr types.StopReason) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: sr}}
}

// fakeConverseAPI is a bedrock.RuntimeAPI that streams a fixed set of native
// Converse events, used as the parity reference for the Mantle adapter.
type fakeConverseAPI struct {
	events []types.ConverseStreamOutput
}

func (f *fakeConverseAPI) Converse(
	context.Context,
	*bedrockruntime.ConverseInput,
	...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	return nil, errors.New("unused")
}

func (f *fakeConverseAPI) ConverseStream(
	context.Context,
	*bedrockruntime.ConverseStreamInput,
	...func(*bedrockruntime.Options),
) (bedrock.StreamReader, error) {
	ch := make(chan types.ConverseStreamOutput, len(f.events))
	for _, ev := range f.events {
		ch <- ev
	}
	close(ch)
	return &fakeStreamReader{ch: ch}, nil
}

type fakeStreamReader struct {
	ch chan types.ConverseStreamOutput
}

func (f *fakeStreamReader) Events() <-chan types.ConverseStreamOutput { return f.ch }
func (f *fakeStreamReader) Close() error                              { return nil }
func (f *fakeStreamReader) Err() error                                { return nil }
