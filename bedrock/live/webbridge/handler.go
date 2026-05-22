// Package webbridge exposes a Nova-Sonic-backed WebSocket handler that
// implements the wire format used by adk-go's embedded Angular web UI for
// /run_live.
//
// It exists as a workaround for adk-go v1.3's Gemini-locked Flow.RunLive:
// the upstream RunLiveHandler dials genai.Client.Live.Connect() directly, so
// the mic button fails out-of-the-box for any non-Gemini model. Mount [New]
// (or [NewSublauncher]) before the upstream API handler and the mic button
// will reach Amazon Nova Sonic instead.
//
// Once an upstream LiveBackend interface lands in adk-go, this package should
// be deprecated in favour of Runner.RunLive.
package webbridge

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/live"
)

// Options configures [New]. Sensible defaults are applied for any zero field.
type Options struct {
	// ModelID overrides the Bedrock model. Empty uses [live.DefaultModelID].
	ModelID string

	// SystemInstruction is sent as a SYSTEM/TEXT content block right after the
	// session opens. Empty means no system message.
	SystemInstruction string

	// InputSampleRateHz overrides the audio input sample rate. Sonic accepts
	// 8000, 16000, or 24000. Zero defaults to 16000.
	InputSampleRateHz int32

	// ResponseModalities controls which output modalities Sonic produces.
	// Nil/empty defaults to [Audio, Text].
	ResponseModalities []genai.Modality

	// Tools is the same LLMRequest-style map [live.OpenOptions.Tools] uses.
	// Pass nil if you don't want voice-side tool use. See
	// [examples/bedrock-live-tool] for the standalone pattern.
	Tools map[string]any

	// Logger receives structured diagnostics. nil discards.
	Logger *slog.Logger

	// OnRawSonicEvent, when non-nil, fires for every raw Sonic envelope the
	// server emits — useful for diagnosing protocol issues. Runs on the read
	// goroutine; keep it fast.
	OnRawSonicEvent func(name string, payload []byte)
}

// wsBufferSize is gorilla/websocket's recommended buffer size for short
// audio frames; matches what adkrest.RuntimeAPIController.RunLiveHandler uses.
const wsBufferSize = 1024

// New returns an [http.Handler] that upgrades requests to a WebSocket, opens
// a [live.Session] backed by Nova Sonic, and bridges between the UI's wire
// shape and the Session's Send/iterator surface. The handler enforces the
// query-string contract used by adk-go's UI (?appName=&userId=&sessionId=).
//
// Mount the returned handler at /api/run_live (or wherever your API router
// prefix puts it) BEFORE the upstream adk-go API handler — gorilla mux
// evaluates routes in registration order, so an exact-path match wins over
// the upstream catchall.
func New(api live.BidiRuntimeAPI, opts Options) http.Handler {
	logger := opts.Logger
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	logger = logger.With("component", "bedrock-live-webbridge")
	b := &bridge{
		api:    api,
		opts:   opts,
		logger: logger,
		upgrader: websocket.Upgrader{
			ReadBufferSize:  wsBufferSize,
			WriteBufferSize: wsBufferSize,
			// Permissive origin — matches adkrest's RunLiveHandler default.
			// Tighten this if you serve the UI cross-origin in production.
			CheckOrigin: func(*http.Request) bool { return true },
		},
	}
	return http.HandlerFunc(b.handle)
}

type bridge struct {
	api      live.BidiRuntimeAPI
	opts     Options
	logger   *slog.Logger
	upgrader websocket.Upgrader
}

func (b *bridge) handle(rw http.ResponseWriter, req *http.Request) {
	q := req.URL.Query()
	app := firstNonEmpty(q.Get("appName"), q.Get("app_name"))
	user := firstNonEmpty(q.Get("userId"), q.Get("user_id"))
	sessionID := firstNonEmpty(q.Get("sessionId"), q.Get("session_id"))
	if app == "" || user == "" || sessionID == "" {
		http.Error(rw, "appName, userId, and sessionId query parameters are required", http.StatusBadRequest)
		return
	}

	b.logger.InfoContext(req.Context(), "upgrading websocket",
		"app", app, "user", user, "session", sessionID)

	ws, err := b.upgrader.Upgrade(rw, req, nil)
	if err != nil {
		b.logger.ErrorContext(req.Context(), "upgrade failed", "err", err)
		return
	}
	defer func() { _ = ws.Close() }()

	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()

	sampleRate := b.opts.InputSampleRateHz
	if sampleRate == 0 {
		sampleRate = 16000
	}
	modalities := b.opts.ResponseModalities
	if len(modalities) == 0 {
		modalities = []genai.Modality{genai.ModalityAudio, genai.ModalityText}
	}

	sess, events, err := live.Open(ctx, b.api, b.opts.ModelID, &agent.LiveRunConfig{
		ResponseModalities: modalities,
	}, &live.OpenOptions{
		SystemInstruction: b.opts.SystemInstruction,
		InputSampleRateHz: sampleRate,
		Author:            app,
		Tools:             b.opts.Tools,
		OnRawEvent:        b.opts.OnRawSonicEvent,
	})
	if err != nil {
		b.logger.ErrorContext(req.Context(), "open Nova Sonic session", "err", err)
		_ = ws.WriteMessage(websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseInternalServerErr, err.Error()))
		return
	}
	defer func() { _ = sess.Close() }()
	b.logger.InfoContext(req.Context(), "Sonic session open")

	var once sync.Once
	closeBoth := func(reason string) {
		once.Do(func() {
			_ = ws.WriteMessage(websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseNormalClosure, reason))
			cancel()
		})
	}

	go b.runReadLoop(ws, sess, closeBoth)
	b.runWriteLoop(ws, events, sessionID, closeBoth)
}

func (b *bridge) runReadLoop(
	ws *websocket.Conn,
	sess *live.Session,
	closeBoth func(string),
) {
	defer closeBoth("read loop ended")
	for {
		messageType, payload, err := ws.ReadMessage()
		if err != nil {
			if !websocket.IsCloseError(err,
				websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				b.logger.Warn("ws read", "err", err)
			}
			return
		}
		if err := b.routeClientMessage(sess, messageType, payload); err != nil {
			b.logger.Warn("forward to session", "err", err)
			return
		}
	}
}

func (b *bridge) runWriteLoop(
	ws *websocket.Conn,
	events iter.Seq2[*session.Event, error],
	sessionID string,
	closeBoth func(string),
) {
	for ev, err := range events {
		if err != nil {
			b.logger.Warn("session err", "err", err)
			closeBoth(err.Error())
			return
		}
		if ev == nil {
			continue
		}
		if ev.ID == "" {
			ev.ID = uuid.NewString()
		}
		if ev.InvocationID == "" {
			ev.InvocationID = sessionID
		}
		if err := ws.WriteJSON(FromSessionEvent(ev)); err != nil {
			b.logger.Warn("ws write", "err", err)
			return
		}
	}
}

func (b *bridge) routeClientMessage(
	sess *live.Session,
	messageType int,
	payload []byte,
) error {
	switch messageType {
	case websocket.BinaryMessage:
		return sess.Send(agent.LiveRequest{
			RealtimeInput: &genai.Blob{
				MIMEType: "audio/pcm;rate=16000",
				Data:     payload,
			},
		})
	case websocket.TextMessage:
		var apiReq LiveRequestJSON
		if err := json.Unmarshal(payload, &apiReq); err != nil {
			return fmt.Errorf("decode client text frame: %w", err)
		}
		if apiReq.Close {
			return sess.Close()
		}
		req := agent.LiveRequest{Content: apiReq.Content}
		switch {
		case apiReq.ActivityStart != nil:
			req.RealtimeInput = apiReq.ActivityStart
		case apiReq.ActivityEnd != nil:
			req.RealtimeInput = apiReq.ActivityEnd
		case apiReq.Blob != nil:
			req.RealtimeInput = &genai.Blob{
				MIMEType: apiReq.Blob.MIMEType,
				Data:     apiReq.Blob.Data,
			}
		}
		return sess.Send(req)
	default:
		// Ping/Pong/Close handled by gorilla; ignore other types.
		return nil
	}
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if s := strings.TrimSpace(v); s != "" {
			return s
		}
	}
	return ""
}
