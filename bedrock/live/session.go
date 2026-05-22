package live

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/google/uuid"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers/sonic"
)

// DefaultModelID is the Bedrock model ID used when [Open] is called with an
// empty modelID. Amazon Nova 2 Sonic is the current speech-to-speech model on
// Bedrock; the original Nova Sonic (`amazon.nova-sonic-v1:0`) reaches EOL on
// 2026-09-14.
const DefaultModelID = "amazon.nova-2-sonic-v1:0"

// defaultInputSampleRateHz is the audio sample rate assumed for inbound USER
// audio when [OpenOptions.InputSampleRateHz] is left zero.
const defaultInputSampleRateHz = 16000

// ErrSessionClosed is returned by [Session.Send] when the session is already
// closed (or its context cancelled).
var ErrSessionClosed = errors.New("bedrock/live: session is closed")

// ErrUnsupportedTool is an alias for [sonic.ErrUnsupportedTool], re-exported
// so callers don't need to import the internal sonic package to do
// `errors.Is(err, live.ErrUnsupportedTool)`.
var ErrUnsupportedTool = sonic.ErrUnsupportedTool

// CustomMetadataKeyUnknownEvents is an alias for
// [sonic.CustomMetadataKeyUnknownEvents].
const CustomMetadataKeyUnknownEvents = sonic.CustomMetadataKeyUnknownEvents

// OpenOptions configures [Open] beyond what [agent.LiveRunConfig] expresses.
type OpenOptions struct {
	// Tools is the same map that [model.LLMRequest.Tools] uses — values may be
	// *genai.Tool or *genai.FunctionDeclaration. Only function declarations are
	// forwarded to Nova Sonic.
	Tools map[string]any

	// SystemInstruction is sent as a SYSTEM/TEXT content block right after
	// promptStart. Empty means no system message.
	SystemInstruction string

	// InputSampleRateHz overrides the audio input sample rate (Sonic accepts
	// 8000, 16000, or 24000). Defaults to 16000 when zero.
	InputSampleRateHz int32

	// Author labels emitted session events. Defaults to "bedrock-live".
	Author string

	// GenerateContentConfig provides inference parameters
	// (Temperature, TopP, MaxOutputTokens) the way [llmagent.Config] does. The
	// values are mapped into Sonic's sessionStart.inferenceConfiguration.
	// StopSequences are silently dropped because Sonic's schema doesn't accept
	// them. SystemInstruction on this config is ignored — use
	// [OpenOptions.SystemInstruction] instead so the content block framing is
	// correct.
	GenerateContentConfig *genai.GenerateContentConfig

	// OnRawEvent, when non-nil, is invoked for every raw event envelope the
	// server emits — before any decoding/translation. name is the discriminator
	// key (e.g. "completionStart", "audioOutput"); payload is the raw JSON body
	// under that key. Intended for diagnostics; the callback runs on the read
	// goroutine, so keep it fast.
	OnRawEvent func(name string, payload []byte)
}

// Session is a Bedrock-backed [agent.LiveSession].
type Session struct {
	ctx        context.Context
	cancel     context.CancelFunc
	stream     BidiStream
	promptName string
	author     string

	inputSampleRateHz int32
	onRawEvent        func(name string, payload []byte)

	mu               sync.Mutex
	closed           bool
	audioContentName string // current open USER/AUDIO contentName, "" when none

	// events feeds the read-side iterator returned by Open.
	events chan eventOrErr
	// done closes when the reader goroutine has exited.
	done chan struct{}
}

type eventOrErr struct {
	ev  *session.Event
	err error
}

// Open dials Nova Sonic via [BidiRuntimeAPI.InvokeModelWithBidirectionalStream]
// and writes the sessionStart + promptStart events. It returns the live
// session, an iterator over server events, and any error encountered while
// dialing.
//
// The returned iterator yields a `(nil, err)` pair followed by completion if
// the underlying stream errors mid-session. Callers must call [Session.Close]
// when done to free the HTTP/2 stream.
func Open(
	ctx context.Context,
	api BidiRuntimeAPI,
	modelID string,
	cfg *agent.LiveRunConfig,
	opts *OpenOptions,
) (*Session, iter.Seq2[*session.Event, error], error) {
	if api == nil {
		return nil, nil, errors.New("bedrock/live: nil BidiRuntimeAPI")
	}
	if strings.TrimSpace(modelID) == "" {
		modelID = DefaultModelID
	}
	if opts == nil {
		opts = &OpenOptions{}
	}
	stream, err := api.InvokeModelWithBidirectionalStream(ctx,
		&bedrockruntime.InvokeModelWithBidirectionalStreamInput{ModelId: &modelID})
	if err != nil {
		return nil, nil, fmt.Errorf("InvokeModelWithBidirectionalStream: %w", err)
	}

	sessCtx, cancel := context.WithCancel(ctx)
	author := opts.Author
	if author == "" {
		author = "bedrock-live"
	}
	inputRate := opts.InputSampleRateHz
	if inputRate == 0 {
		inputRate = defaultInputSampleRateHz
	}

	s := &Session{
		ctx:               sessCtx,
		cancel:            cancel,
		stream:            stream,
		promptName:        uuid.NewString(),
		author:            author,
		inputSampleRateHz: inputRate,
		onRawEvent:        opts.OnRawEvent,
		events:            make(chan eventOrErr, 32),
		done:              make(chan struct{}),
	}

	// Write the session/prompt framing synchronously so that any malformed
	// configuration surfaces from Open rather than from the first Send.
	if err := s.sendRaw(sonic.BuildSessionStart(opts.GenerateContentConfig)); err != nil {
		_ = stream.Close()
		cancel()
		return nil, nil, fmt.Errorf("send sessionStart: %w", err)
	}
	if err := s.sendRaw(sonic.BuildPromptStart(s.promptName, promptStartOptionsFrom(cfg, opts))); err != nil {
		_ = stream.Close()
		cancel()
		return nil, nil, fmt.Errorf("send promptStart: %w", err)
	}
	if strings.TrimSpace(opts.SystemInstruction) != "" {
		if err := s.sendTextContent(sonic.RoleSystem, opts.SystemInstruction); err != nil {
			_ = stream.Close()
			cancel()
			return nil, nil, fmt.Errorf("send system instruction: %w", err)
		}
	}

	go s.readLoop()

	return s, s.iterator(), nil
}

// promptStartOptionsFrom flattens the relevant fields from LiveRunConfig +
// OpenOptions into the sonic-package option struct.
func promptStartOptionsFrom(cfg *agent.LiveRunConfig, opts *OpenOptions) sonic.PromptStartOptions {
	out := sonic.PromptStartOptions{Tools: opts.Tools}
	if cfg == nil {
		return out
	}
	out.ResponseModalities = cfg.ResponseModalities
	if cfg.SpeechConfig != nil &&
		cfg.SpeechConfig.VoiceConfig != nil &&
		cfg.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig != nil {
		out.VoiceName = cfg.SpeechConfig.VoiceConfig.PrebuiltVoiceConfig.VoiceName
	}
	return out
}

// Send implements [agent.LiveSession.Send].
//
// Routing:
//   - req.RealtimeInput as *genai.Blob → audio chunk (opens a USER/AUDIO
//     content block on first chunk, then keeps it open across subsequent
//     chunks).
//   - req.RealtimeInput as *genai.ActivityEnd → closes the open audio content
//     block (sends contentEnd). *genai.ActivityStart is accepted as a no-op so
//     callers can use it symmetrically.
//   - req.Content with a FunctionResponse part → toolResult turn.
//   - req.Content with text parts → SYSTEM/USER/ASSISTANT text turn (role taken
//     from req.Content.Role; empty falls back to USER).
func (s *Session) Send(req agent.LiveRequest) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return ErrSessionClosed
	}
	s.mu.Unlock()

	if req.RealtimeInput != nil {
		return s.sendRealtime(req.RealtimeInput)
	}
	if req.Content != nil {
		return s.sendContent(req.Content)
	}
	return errors.New("bedrock/live: LiveRequest has neither RealtimeInput nor Content")
}

// Close implements [agent.LiveSession.Close].
func (s *Session) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	openAudio := s.audioContentName
	s.audioContentName = ""
	s.mu.Unlock()

	// Best-effort shutdown framing: contentEnd if audio is open, promptEnd, sessionEnd.
	if openAudio != "" {
		_ = s.sendRaw(sonic.BuildContentEnd(s.promptName, openAudio))
	}
	_ = s.sendRaw(sonic.BuildPromptEnd(s.promptName))
	_ = s.sendRaw(sonic.BuildSessionEnd())

	s.cancel()
	err := s.stream.Close()
	<-s.done
	return err
}

func (s *Session) sendRealtime(input any) error {
	switch v := input.(type) {
	case *genai.Blob:
		return s.sendAudioBlob(v)
	case *genai.ActivityStart:
		// Sonic opens its audio content block on the first audioInput; nothing
		// to send here.
		return nil
	case *genai.ActivityEnd:
		return s.closeAudioContent()
	default:
		return fmt.Errorf("bedrock/live: unsupported RealtimeInput type %T", input)
	}
}

func (s *Session) sendAudioBlob(blob *genai.Blob) error {
	if blob == nil || len(blob.Data) == 0 {
		return nil
	}
	s.mu.Lock()
	contentName := s.audioContentName
	needsOpen := contentName == ""
	if needsOpen {
		contentName = uuid.NewString()
		s.audioContentName = contentName
	}
	s.mu.Unlock()

	if needsOpen {
		if err := s.sendRaw(sonic.BuildContentStartAudio(s.promptName, contentName, s.inputSampleRateHz)); err != nil {
			return err
		}
	}
	return s.sendRaw(sonic.BuildAudioInput(s.promptName, contentName, blob.Data))
}

func (s *Session) closeAudioContent() error {
	s.mu.Lock()
	contentName := s.audioContentName
	s.audioContentName = ""
	s.mu.Unlock()
	if contentName == "" {
		return nil
	}
	return s.sendRaw(sonic.BuildContentEnd(s.promptName, contentName))
}

func (s *Session) sendContent(c *genai.Content) error {
	if c == nil {
		return errors.New("bedrock/live: nil Content")
	}
	// Tool response framing has priority — Sonic expects a TOOL contentStart
	// wrapper around the result.
	for _, p := range c.Parts {
		if p != nil && p.FunctionResponse != nil {
			return s.sendToolResult(p.FunctionResponse)
		}
	}
	// Otherwise gather all text parts into one turn.
	var sb strings.Builder
	for _, p := range c.Parts {
		if p == nil {
			continue
		}
		if p.Text != "" {
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(p.Text)
		}
	}
	if sb.Len() == 0 {
		return nil
	}
	role := normalizeRole(c.Role)
	return s.sendTextContent(role, sb.String())
}

func (s *Session) sendTextContent(role, text string) error {
	contentName := uuid.NewString()
	if err := s.sendRaw(sonic.BuildContentStartText(s.promptName, contentName, role)); err != nil {
		return err
	}
	if err := s.sendRaw(sonic.BuildTextInput(s.promptName, contentName, text)); err != nil {
		return err
	}
	return s.sendRaw(sonic.BuildContentEnd(s.promptName, contentName))
}

func (s *Session) sendToolResult(fr *genai.FunctionResponse) error {
	if fr == nil || strings.TrimSpace(fr.ID) == "" {
		return errors.New("bedrock/live: FunctionResponse is missing an ID")
	}
	resultJSON, err := toolResultJSON(fr)
	if err != nil {
		return err
	}
	contentName := uuid.NewString()
	if err := s.sendRaw(sonic.BuildContentStartToolResult(s.promptName, contentName, fr.ID)); err != nil {
		return err
	}
	if err := s.sendRaw(sonic.BuildToolResult(s.promptName, contentName, resultJSON)); err != nil {
		return err
	}
	return s.sendRaw(sonic.BuildContentEnd(s.promptName, contentName))
}

func (s *Session) sendRaw(payload []byte, err error) error {
	if err != nil {
		return err
	}
	return s.stream.Send(s.ctx, &types.InvokeModelWithBidirectionalStreamInputMemberChunk{
		Value: types.BidirectionalInputPayloadPart{Bytes: payload},
	})
}

func (s *Session) readLoop() {
	defer close(s.done)
	defer close(s.events)

	state := sonic.NewReadState()
	state.OnRawEvent = s.onRawEvent
	for chunk := range s.stream.Events() {
		select {
		case <-s.ctx.Done():
			return
		default:
		}
		out, ok := chunk.(*types.InvokeModelWithBidirectionalStreamOutputMemberChunk)
		if !ok {
			// Unknown union variants are ignored, matching the lenient stance in
			// bedrock/converse.go for ConverseStream output.
			continue
		}
		resp, err := state.Consume(out.Value.Bytes)
		if err != nil {
			s.tryEmit(eventOrErr{err: err})
			return
		}
		if resp == nil {
			continue
		}
		ev := &session.Event{
			LLMResponse: *resp,
			ID:          uuid.NewString(),
			Timestamp:   time.Now(),
			Author:      s.author,
		}
		if !s.tryEmit(eventOrErr{ev: ev}) {
			return
		}
	}
	if err := s.stream.Err(); err != nil {
		s.tryEmit(eventOrErr{err: err})
	}
}

func (s *Session) tryEmit(e eventOrErr) bool {
	select {
	case <-s.ctx.Done():
		return false
	case s.events <- e:
		return true
	}
}

func (s *Session) iterator() iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		for e := range s.events {
			if !yield(e.ev, e.err) {
				return
			}
		}
	}
}

func normalizeRole(role string) string {
	switch strings.ToUpper(strings.TrimSpace(role)) {
	case sonic.RoleSystem:
		return sonic.RoleSystem
	case sonic.RoleAssistant, "MODEL":
		return sonic.RoleAssistant
	case "":
		return sonic.RoleUser
	default:
		return sonic.RoleUser
	}
}

// toolResultJSON serializes a FunctionResponse's payload into the stringified
// JSON content Nova Sonic expects for a toolResult event.
func toolResultJSON(fr *genai.FunctionResponse) (string, error) {
	if fr == nil {
		return "{}", nil
	}
	payload := fr.Response
	if payload == nil {
		payload = map[string]any{}
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal tool result: %w", err)
	}
	return string(raw), nil
}
