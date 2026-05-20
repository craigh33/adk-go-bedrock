package webbridge_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/gorilla/websocket"

	"github.com/craigh33/adk-go-bedrock/bedrock/live"
	"github.com/craigh33/adk-go-bedrock/bedrock/live/webbridge"
	"github.com/craigh33/adk-go-bedrock/internal/mappers/sonic"
)

// --- fakes -------------------------------------------------------------------

type fakeAPI struct {
	stream *fakeStream
}

func (a *fakeAPI) InvokeModelWithBidirectionalStream(
	_ context.Context,
	_ *bedrockruntime.InvokeModelWithBidirectionalStreamInput,
	_ ...func(*bedrockruntime.Options),
) (live.BidiStream, error) {
	return a.stream, nil
}

type fakeStream struct {
	mu        sync.Mutex
	sent      [][]byte
	events    chan types.InvokeModelWithBidirectionalStreamOutput
	closeOnce sync.Once
}

func newFakeStream() *fakeStream {
	return &fakeStream{
		events: make(chan types.InvokeModelWithBidirectionalStreamOutput, 16),
	}
}

func (s *fakeStream) Send(_ context.Context, in types.InvokeModelWithBidirectionalStreamInput) error {
	chunk, ok := in.(*types.InvokeModelWithBidirectionalStreamInputMemberChunk)
	if !ok {
		return nil
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
	s.closeOnce.Do(func() { close(s.events) })
	return nil
}

func (s *fakeStream) Err() error { return nil }

func (s *fakeStream) pushEvent(t *testing.T, name string, payload any) {
	t.Helper()
	raw, err := sonic.Wrap(name, payload)
	if err != nil {
		t.Fatalf("Wrap %q: %v", name, err)
	}
	s.events <- &types.InvokeModelWithBidirectionalStreamOutputMemberChunk{
		Value: types.BidirectionalOutputPayloadPart{Bytes: raw},
	}
}

// dialWS starts an httptest server hosting the bridge handler, then opens a
// WebSocket client connection to it.
func dialWS(t *testing.T, api live.BidiRuntimeAPI, opts webbridge.Options) (*websocket.Conn, *httptest.Server) {
	t.Helper()
	handler := webbridge.New(api, opts)
	srv := httptest.NewServer(handler)
	url := "ws" + strings.TrimPrefix(srv.URL, "http") + "?appName=test&userId=u&sessionId=s"
	ws, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		srv.Close()
		t.Fatalf("dial: %v", err)
	}
	return ws, srv
}

// --- tests -------------------------------------------------------------------

func TestHandlerRejectsMissingQueryParams(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(webbridge.New(&fakeAPI{stream: newFakeStream()}, webbridge.Options{}))
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL) // no query string
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", resp.StatusCode)
	}
}

func TestHandlerForwardsBinaryAudio(t *testing.T) {
	t.Parallel()
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	ws, srv := dialWS(t, api, webbridge.Options{})
	defer srv.Close()
	defer ws.Close()

	// Drain anything Sonic might emit in a separate goroutine so the read
	// pump doesn't backpressure our writes.
	go func() {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	}()

	pcm := []byte{1, 2, 3, 4}
	if err := ws.WriteMessage(websocket.BinaryMessage, pcm); err != nil {
		t.Fatalf("write binary: %v", err)
	}
	// Give the read loop a moment to forward.
	waitFor(t, func() bool {
		stream.mu.Lock()
		defer stream.mu.Unlock()
		// sent[0..2] are sessionStart/promptStart/audio_contentStart/audioInput.
		// We look for an audioInput envelope carrying our pcm.
		for _, payload := range stream.sent {
			if strings.Contains(string(payload), `"event":{"audioInput":`) {
				return true
			}
		}
		return false
	})
}

func TestHandlerForwardsTextBlob(t *testing.T) {
	t.Parallel()
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	ws, srv := dialWS(t, api, webbridge.Options{})
	defer srv.Close()
	defer ws.Close()

	go func() {
		for {
			if _, _, err := ws.ReadMessage(); err != nil {
				return
			}
		}
	}()

	pcm := []byte{0xAA, 0xBB, 0xCC, 0xDD}
	req := webbridge.LiveRequestJSON{Blob: &webbridge.LiveBlobJSON{
		MIMEType: "audio/pcm",
		Data:     pcm,
	}}
	body, _ := json.Marshal(req)
	if err := ws.WriteMessage(websocket.TextMessage, body); err != nil {
		t.Fatalf("write text: %v", err)
	}

	waitFor(t, func() bool {
		stream.mu.Lock()
		defer stream.mu.Unlock()
		want := base64.StdEncoding.EncodeToString(pcm)
		for _, payload := range stream.sent {
			if strings.Contains(string(payload), `"event":{"audioInput":`) &&
				strings.Contains(string(payload), want) {
				return true
			}
		}
		return false
	})
}

func TestHandlerWritesAudioOutputAsJSON(t *testing.T) {
	t.Parallel()
	stream := newFakeStream()
	api := &fakeAPI{stream: stream}
	ws, srv := dialWS(t, api, webbridge.Options{})
	defer srv.Close()
	defer ws.Close()

	pcm := []byte{0x10, 0x20, 0x30}
	stream.pushEvent(t, "contentStart", sonic.ContentStartOutput{
		ContentID: "c1", Type: "AUDIO", Role: "ASSISTANT",
	})
	stream.pushEvent(t, "audioOutput", sonic.AudioOutputEvent{
		ContentID: "c1",
		Content:   base64.StdEncoding.EncodeToString(pcm),
	})
	stream.pushEvent(t, "contentEnd", sonic.ContentEndOutput{
		ContentID: "c1", Type: "AUDIO", StopReason: sonic.StopReasonEndTurn,
	})
	stream.pushEvent(t, "completionEnd", sonic.CompletionEndOutput{
		StopReason: sonic.StopReasonEndTurn,
	})

	// Drain WS until we see TurnComplete or audio bytes.
	deadline := time.Now().Add(2 * time.Second)
	saw := false
	for time.Now().Before(deadline) {
		_ = ws.SetReadDeadline(time.Now().Add(200 * time.Millisecond))
		_, body, err := ws.ReadMessage()
		if err != nil {
			break
		}
		var ev webbridge.LiveEventJSON
		if err := json.Unmarshal(body, &ev); err != nil {
			continue
		}
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				if p != nil && p.InlineData != nil && len(p.InlineData.Data) == len(pcm) {
					saw = true
					break
				}
			}
		}
		if saw || ev.TurnComplete {
			break
		}
	}
	if !saw {
		t.Fatal("never received audio bytes on the WebSocket")
	}
}

func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("condition not met within 2s")
}
