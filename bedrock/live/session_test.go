package live

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"iter"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers/sonic"
	"github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

// --- fakes -----------------------------------------------------------------

type fakeAPI struct {
	openErr error
	stream  *fakeStream
}

func (a *fakeAPI) InvokeModelWithBidirectionalStream(
	_ context.Context,
	_ *bedrockruntime.InvokeModelWithBidirectionalStreamInput,
	_ ...func(*bedrockruntime.Options),
) (BidiStream, error) {
	if a.openErr != nil {
		return nil, a.openErr
	}
	return a.stream, nil
}

type fakeStream struct {
	mu        sync.Mutex
	sent      [][]byte
	events    chan types.InvokeModelWithBidirectionalStreamOutput
	closed    bool
	closeOnce sync.Once
	err       error
	sendErr   error
}

func newFakeStream() *fakeStream {
	return &fakeStream{
		events: make(chan types.InvokeModelWithBidirectionalStreamOutput, 16),
	}
}

func (s *fakeStream) Send(_ context.Context, in types.InvokeModelWithBidirectionalStreamInput) error {
	if s.sendErr != nil {
		return s.sendErr
	}
	chunk, ok := in.(*types.InvokeModelWithBidirectionalStreamInputMemberChunk)
	if !ok {
		return errors.New("fakeStream.Send: unexpected variant")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sent = append(s.sent, append([]byte(nil), chunk.Value.Bytes...))
	return nil
}

func (s *fakeStream) Events() <-chan types.InvokeModelWithBidirectionalStreamOutput {
	return s.events
}

func (s *fakeStream) Close() error {
	s.closeOnce.Do(func() {
		s.closed = true
		close(s.events)
	})
	return nil
}

func (s *fakeStream) Err() error { return s.err }

// pushEvent feeds a server-side envelope into the fake stream as a Chunk.
func (s *fakeStream) pushEvent(t *testing.T, name string, payload any) {
	t.Helper()
	raw, err := sonic.Wrap(name, payload)
	if err != nil {
		t.Fatalf("wrap %q: %v", name, err)
	}
	s.events <- &types.InvokeModelWithBidirectionalStreamOutputMemberChunk{
		Value: types.BidirectionalOutputPayloadPart{Bytes: raw},
	}
}

func (s *fakeStream) sentEnvelopes(t *testing.T) []string {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]string, 0, len(s.sent))
	for _, b := range s.sent {
		var env sonic.Envelope
		if err := json.Unmarshal(b, &env); err != nil {
			t.Fatalf("sent envelope not valid JSON: %v\n%s", err, b)
		}
		name, _, err := sonic.EventName(env.Event)
		if err != nil {
			t.Fatalf("sent envelope has no event name: %v\n%s", err, b)
		}
		out = append(out, name)
	}
	return out
}

// --- tests -----------------------------------------------------------------

func TestOpenSendsSessionAndPromptStart(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}

	sess, _, err := Open(context.Background(), api, "", &agent.LiveRunConfig{}, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	got := stream.sentEnvelopes(t)
	if len(got) < 2 || got[0] != "sessionStart" || got[1] != "promptStart" {
		t.Fatalf("expected sessionStart, promptStart prefix; got %v", got)
	}
}

func TestOpenSendsSystemInstructionAfterPromptStart(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}

	sess, _, err := Open(context.Background(), api, "", &agent.LiveRunConfig{}, &OpenOptions{
		SystemInstruction: "You are helpful.",
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	got := stream.sentEnvelopes(t)
	want := []string{"sessionStart", "promptStart", "contentStart", "textInput", "contentEnd"}
	if len(got) < len(want) {
		t.Fatalf("got %v, want at least %v", got, want)
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("at %d: got %q want %q (full: %v)", i, got[i], w, got)
		}
	}
}

func TestOpenWrapsAPIError(t *testing.T) {
	api := &fakeAPI{openErr: errors.New("ResourceNotFound")}
	_, _, err := Open(context.Background(), api, "", nil, nil)
	if err == nil {
		t.Fatal("expected error from Open")
	}
	if !strings.Contains(err.Error(), "ResourceNotFound") {
		t.Fatalf("error didn't wrap original: %v", err)
	}
}

func TestSendAudioBlobOpensThenReusesContent(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, _, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	// First audio chunk → contentStart + audioInput.
	if err := sess.Send(agent.LiveRequest{RealtimeInput: &genai.Blob{Data: []byte{1, 2, 3}}}); err != nil {
		t.Fatalf("send 1: %v", err)
	}
	// Second audio chunk → audioInput only.
	if err := sess.Send(agent.LiveRequest{RealtimeInput: &genai.Blob{Data: []byte{4, 5, 6}}}); err != nil {
		t.Fatalf("send 2: %v", err)
	}
	// ActivityEnd → contentEnd.
	if err := sess.Send(agent.LiveRequest{RealtimeInput: &genai.ActivityEnd{}}); err != nil {
		t.Fatalf("send end: %v", err)
	}

	got := stream.sentEnvelopes(t)
	// sessionStart, promptStart, then audio: contentStart, audioInput, audioInput, contentEnd.
	wantAudio := []string{"contentStart", "audioInput", "audioInput", "contentEnd"}
	if len(got) < 2+len(wantAudio) {
		t.Fatalf("got %v", got)
	}
	for i, w := range wantAudio {
		if got[2+i] != w {
			t.Fatalf("at audio[%d]: got %q want %q (full: %v)", i, got[2+i], w, got)
		}
	}
}

func TestSendTextContentRoutesByRole(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, _, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	err = sess.Send(agent.LiveRequest{
		Content: &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{Text: "hello"}},
		},
	})
	if err != nil {
		t.Fatalf("send text: %v", err)
	}

	got := stream.sentEnvelopes(t)
	// Last three should be contentStart, textInput, contentEnd.
	tail := got[len(got)-3:]
	want := []string{"contentStart", "textInput", "contentEnd"}
	for i, w := range want {
		if tail[i] != w {
			t.Fatalf("at tail[%d]: got %q want %q (full: %v)", i, tail[i], w, got)
		}
	}
}

func TestSendToolResultFraming(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, _, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	err = sess.Send(agent.LiveRequest{
		Content: &genai.Content{
			Role: "tool",
			Parts: []*genai.Part{{
				FunctionResponse: &genai.FunctionResponse{
					ID:       "tu-1",
					Name:     "weather",
					Response: map[string]any{"temp": 72},
				},
			}},
		},
	})
	if err != nil {
		t.Fatalf("send tool result: %v", err)
	}

	got := stream.sentEnvelopes(t)
	tail := got[len(got)-3:]
	want := []string{"contentStart", "toolResult", "contentEnd"}
	for i, w := range want {
		if tail[i] != w {
			t.Fatalf("at tail[%d]: got %q want %q (full: %v)", i, tail[i], w, got)
		}
	}
}

func TestIteratorYieldsCompletionEnd(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, iterator, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	pcm := []byte{0x10, 0x20, 0x30}
	stream.pushEvent(t, "contentStart", sonic.ContentStartOutput{ContentID: "c1", Type: "AUDIO", Role: "ASSISTANT"})
	stream.pushEvent(t, "audioOutput", sonic.AudioOutputEvent{
		ContentID: "c1",
		Content:   base64.StdEncoding.EncodeToString(pcm),
	})
	stream.pushEvent(t, "contentEnd", sonic.ContentEndOutput{
		ContentID:  "c1",
		Type:       "AUDIO",
		StopReason: sonic.StopReasonEndTurn,
	})
	stream.pushEvent(t, "completionEnd", sonic.CompletionEndOutput{StopReason: sonic.StopReasonEndTurn})

	events, err := drainUntilTurnComplete(t, iterator, 2*time.Second)
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	_ = sess.Close()

	if len(events) < 2 {
		t.Fatalf("expected at least audio + completion events, got %d", len(events))
	}
	if last := events[len(events)-1]; !last.TurnComplete {
		t.Fatalf("last event should have TurnComplete=true, got %+v", last)
	}
	if !eventsContainAudio(events, len(pcm)) {
		t.Fatal("missing audio blob event")
	}
}

// drainUntilTurnComplete consumes the iterator on a goroutine, returning all
// events seen before TurnComplete or any error. Times out after `timeout`.
func drainUntilTurnComplete(
	t *testing.T,
	it iter.Seq2[*session.Event, error],
	timeout time.Duration,
) ([]*session.Event, error) {
	t.Helper()
	type result struct {
		events []*session.Event
		err    error
	}
	resCh := make(chan result, 1)
	go func() {
		var events []*session.Event
		for ev, err := range it {
			if err != nil {
				resCh <- result{events, err}
				return
			}
			events = append(events, ev)
			if ev != nil && ev.TurnComplete {
				resCh <- result{events, nil}
				return
			}
		}
		resCh <- result{events, nil}
	}()

	select {
	case r := <-resCh:
		return r.events, r.err
	case <-time.After(timeout):
		return nil, errors.New("timed out")
	}
}

func eventsContainAudio(events []*session.Event, wantBytes int) bool {
	for _, e := range events {
		if e == nil || e.Content == nil {
			continue
		}
		for _, p := range e.Content.Parts {
			if p.InlineData != nil && len(p.InlineData.Data) == wantBytes {
				return true
			}
		}
	}
	return false
}

func TestIteratorPropagatesStreamError(t *testing.T) {
	stream := newFakeStream()
	stream.err = errors.New("io: throttled")
	api := &fakeAPI{stream: stream}
	sess, iterator, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	// Closing the stream without events flushes Err() into the iterator.
	go func() { _ = stream.Close() }()

	var sawErr error
	for _, err := range iterator {
		if err != nil {
			sawErr = err
			break
		}
	}
	if sawErr == nil {
		t.Fatal("expected stream Err() to surface to iterator")
	}
	if !strings.Contains(sawErr.Error(), "throttled") {
		t.Fatalf("unexpected error: %v", sawErr)
	}
}

func TestSendOnClosedSession(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, _, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	err = sess.Send(agent.LiveRequest{Content: &genai.Content{Parts: []*genai.Part{{Text: "hi"}}}})
	if !errors.Is(err, ErrSessionClosed) {
		t.Fatalf("expected ErrSessionClosed, got %v", err)
	}
}

func TestOpenAcceptsMCPStyleTool(t *testing.T) {
	// mcptoolset produces tools whose schemas live in ParametersJsonSchema as
	// a raw JSON object (map[string]any) rather than a *genai.Schema. Make
	// sure the live mapper accepts that shape end-to-end.
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	mcpTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "fetch_url",
				Description: "Fetch a URL",
				ParametersJsonSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"url": map[string]any{"type": "string"},
					},
					"required": []any{"url"},
				},
			},
		},
	}
	sess, _, err := Open(context.Background(), api, "", nil, &OpenOptions{
		Tools: map[string]any{"fetch_url": mcpTool},
	})
	if err != nil {
		t.Fatalf("Open with MCP-style tool: %v", err)
	}
	defer sess.Close()

	// The second envelope is promptStart and must carry the MCP tool spec.
	stream.mu.Lock()
	promptStart := stream.sent[1]
	stream.mu.Unlock()
	if !strings.Contains(string(promptStart), "fetch_url") {
		t.Fatalf("promptStart didn't include MCP tool: %s", promptStart)
	}
	if !strings.Contains(string(promptStart), `\"required\":[\"url\"]`) &&
		!strings.Contains(string(promptStart), `"required":["url"]`) {
		t.Fatalf("MCP schema not forwarded verbatim: %s", promptStart)
	}
}

func TestOpenRejectsNovaGrounding(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	_, _, err := Open(context.Background(), api, "", nil, &OpenOptions{
		Tools: map[string]any{"grounding": novagrounding.Tool()},
	})
	if err == nil {
		t.Fatal("expected Open to reject Nova Grounding tool")
	}
	if !errors.Is(err, ErrUnsupportedTool) {
		t.Fatalf("error should wrap ErrUnsupportedTool: %v", err)
	}
}

func TestRunAgentLoopInvokesRegisteredTool(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, iterator, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	// Server scripts: contentStart(TOOL) → toolUse → contentEnd(TOOL_USE)
	// → completionEnd.
	stream.pushEvent(t, "contentStart", sonic.ContentStartOutput{ContentID: "c1", Type: "TOOL", Role: "TOOL"})
	stream.pushEvent(t, "toolUse", sonic.ToolUseOutputEvent{
		ContentID: "c1",
		ToolName:  "weather",
		ToolUseID: "tu-1",
		Content:   `{"location":"Boston"}`,
	})
	stream.pushEvent(t, "contentEnd", sonic.ContentEndOutput{
		ContentID: "c1", Type: "TOOL", StopReason: sonic.StopReasonToolUse,
	})
	stream.pushEvent(t, "completionEnd", sonic.CompletionEndOutput{StopReason: sonic.StopReasonEndTurn})

	var capturedArgs map[string]any
	tools := ToolRegistry{
		"weather": func(_ context.Context, args map[string]any) (map[string]any, error) {
			capturedArgs = args
			return map[string]any{"temp_f": 72, "summary": "clear"}, nil
		},
	}

	done := make(chan error, 1)
	go func() { done <- sess.RunAgentLoop(context.Background(), iterator, tools, nil) }()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("RunAgentLoop: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out")
	}

	if capturedArgs["location"] != "Boston" {
		t.Fatalf("tool didn't receive args: %v", capturedArgs)
	}

	// Verify a toolResult frame went on the wire after the toolUse.
	envelopes := stream.sentEnvelopes(t)
	if !slices.Contains(envelopes, "toolResult") {
		t.Fatalf("expected toolResult on the wire, sent: %v", envelopes)
	}
}

func TestRunAgentLoopMissingToolSendsErrorResult(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, iterator, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	stream.pushEvent(t, "contentStart", sonic.ContentStartOutput{ContentID: "c1", Type: "TOOL", Role: "TOOL"})
	stream.pushEvent(t, "toolUse", sonic.ToolUseOutputEvent{
		ContentID: "c1",
		ToolName:  "unknown_tool",
		ToolUseID: "tu-1",
		Content:   "{}",
	})
	stream.pushEvent(t, "contentEnd", sonic.ContentEndOutput{
		ContentID: "c1", Type: "TOOL", StopReason: sonic.StopReasonToolUse,
	})
	stream.pushEvent(t, "completionEnd", sonic.CompletionEndOutput{StopReason: sonic.StopReasonEndTurn})

	done := make(chan error, 1)
	go func() { done <- sess.RunAgentLoop(context.Background(), iterator, ToolRegistry{}, nil) }()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("RunAgentLoop should swallow missing-tool errors: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out")
	}

	// A toolResult should still have gone back with an {"error": "..."} body.
	envelopes := stream.sentEnvelopes(t)
	if !slices.Contains(envelopes, "toolResult") {
		t.Fatalf("expected toolResult with error body, sent: %v", envelopes)
	}
}

func TestRunAgentLoopHonorsContextCancellation(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, iterator, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer sess.Close()

	loopCtx, cancel := context.WithCancel(context.Background())
	// Cancel before pushing any events, then push one so the loop wakes and
	// sees the cancelled context.
	cancel()
	stream.pushEvent(t, "contentStart", sonic.ContentStartOutput{ContentID: "c1", Type: "TEXT", Role: "ASSISTANT"})

	done := make(chan error, 1)
	go func() { done <- sess.RunAgentLoop(loopCtx, iterator, nil, nil) }()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out")
	}
}

func TestCloseSendsShutdownFraming(t *testing.T) {
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	sess, _, err := Open(context.Background(), api, "", nil, nil)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Open an audio block so Close has to emit contentEnd.
	if err := sess.Send(agent.LiveRequest{RealtimeInput: &genai.Blob{Data: []byte{1}}}); err != nil {
		t.Fatalf("send audio: %v", err)
	}
	if err := sess.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	got := stream.sentEnvelopes(t)
	// Tail: contentEnd (for open audio), promptEnd, sessionEnd.
	tail := got[len(got)-3:]
	want := []string{"contentEnd", "promptEnd", "sessionEnd"}
	for i, w := range want {
		if tail[i] != w {
			t.Fatalf("at tail[%d]: got %q want %q (full: %v)", i, tail[i], w, got)
		}
	}
}
