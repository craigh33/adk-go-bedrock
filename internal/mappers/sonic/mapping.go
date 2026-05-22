package sonic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers"
	"github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

// ErrUnsupportedTool is returned by tool conversion when the caller passes a
// tool that Nova Sonic cannot honor. Wrap-detect with [errors.Is].
//
// The current cases are:
//   - The Nova Web Grounding sentinel from [tools/novagrounding] — Grounding is
//     a Bedrock Converse SystemTool and has no equivalent on the bidirectional API.
//   - Any [genai.Tool] variant Sonic doesn't accept (GoogleSearch, Retrieval,
//     ComputerUse, MCP servers, etc.).
//   - Any [OpenOptions.Tools] map value that isn't a *genai.Tool or
//     *genai.FunctionDeclaration.
var ErrUnsupportedTool = errors.New("bedrock/live: unsupported tool")

// CustomMetadataKeyUnknownEvents is the [model.LLMResponse.CustomMetadata] key
// under which Live emits the list of unmodeled server events seen during the
// turn. Each entry is `{"event": <name>, "payload": <raw JSON>}`. Mirrors the
// `bedrock_*` namespace convention used by Converse.
const CustomMetadataKeyUnknownEvents = "bedrock_live_unknown_events"

// Default audio configuration for Nova Sonic. Sonic accepts 8/16/24 kHz mono
// 16-bit LPCM; the input/output defaults are documented at
// https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html.
const (
	defaultInputSampleRate  = 16000
	defaultOutputSampleRate = 24000
	defaultSampleSizeBits   = 16
	defaultChannelCount     = 1
	defaultVoiceID          = "matthew"
	audioMediaType          = "audio/lpcm"
	textMediaType           = "text/plain"
	jsonMediaType           = "application/json"
	audioEncodingBase64     = "base64"
	audioTypeSpeech         = "SPEECH"
)

// Substring Nova Sonic uses on textOutput to flag a barge-in interruption.
// Treated as a marker per docs — there is no dedicated event.
const interruptedMarker = `{ "interrupted" : true }`

// unknownEventsCap bounds the slice to keep a misbehaving server from making
// a session balloon memory.
const unknownEventsCap = 32

// BuildSessionStart encodes the sessionStart envelope from inference params.
// Sonic's sessionStart schema only accepts maxTokens/topP/temperature
// inference parameters; everything else (response modalities, speech config,
// tools) belongs on the subsequent promptStart event.
func BuildSessionStart(cfg *genai.GenerateContentConfig) ([]byte, error) {
	payload := sessionStartInput{
		InferenceConfiguration: sonicInferenceConfigFromGenai(cfg),
	}
	return Wrap("sessionStart", payload)
}

// sonicInferenceConfigFromGenai mirrors the field extraction in
// [mappers.inferenceConfigFromGenai] but emits Sonic's wire-level
// [inferenceConfiguration] struct (which differs from Bedrock Converse's
// types.InferenceConfiguration in shape and the absence of StopSequences).
// Returns nil when no inference parameters are set, so the field marshals as
// omitted.
func sonicInferenceConfigFromGenai(cfg *genai.GenerateContentConfig) *inferenceConfiguration {
	if cfg == nil {
		return nil
	}
	var inf inferenceConfiguration
	anySet := false
	if cfg.Temperature != nil {
		inf.Temperature = cfg.Temperature
		anySet = true
	}
	if cfg.TopP != nil {
		inf.TopP = cfg.TopP
		anySet = true
	}
	if cfg.MaxOutputTokens > 0 {
		mt := cfg.MaxOutputTokens
		inf.MaxTokens = &mt
		anySet = true
	}
	// StopSequences are intentionally not mapped: Sonic's sessionStart
	// inferenceConfiguration only accepts maxTokens/topP/temperature.
	if !anySet {
		return nil
	}
	return &inf
}

// PromptStartOptions configures [BuildPromptStart].
type PromptStartOptions struct {
	// ResponseModalities controls which output configurations are emitted.
	// Nil/empty defaults to [Audio, Text].
	ResponseModalities []genai.Modality
	// VoiceName is the Sonic voice ID for AUDIO output. Empty falls back to
	// the package default ("matthew").
	VoiceName string
	// Tools is the LLMRequest-style map of tools to expose to the model.
	Tools map[string]any
}

// BuildPromptStart encodes promptStart with response modalities, voice,
// and tool catalog.
func BuildPromptStart(promptName string, opts PromptStartOptions) ([]byte, error) {
	if strings.TrimSpace(promptName) == "" {
		return nil, errors.New("BuildPromptStart: empty promptName")
	}
	p := promptStartInput{PromptName: promptName}
	wantsText, wantsAudio := wantedModalities(opts.ResponseModalities)
	if wantsText {
		p.TextOutputConfiguration = &textOutputConfiguration{MediaType: textMediaType}
	}
	if wantsAudio {
		voice := defaultVoiceID
		if v := strings.TrimSpace(opts.VoiceName); v != "" {
			voice = strings.ToLower(v)
		}
		p.AudioOutputConfiguration = &audioOutputConfiguration{
			MediaType:       audioMediaType,
			SampleRateHertz: defaultOutputSampleRate,
			SampleSizeBits:  defaultSampleSizeBits,
			ChannelCount:    defaultChannelCount,
			VoiceID:         voice,
			Encoding:        audioEncodingBase64,
			AudioType:       audioTypeSpeech,
		}
	}
	if len(opts.Tools) > 0 {
		p.ToolUseOutputConfiguration = &toolUseOutputConfiguration{MediaType: jsonMediaType}
		entries, err := toolEntriesFromADK(opts.Tools)
		if err != nil {
			return nil, err
		}
		if len(entries) > 0 {
			p.ToolConfiguration = &toolConfiguration{Tools: entries}
		}
	}
	return Wrap("promptStart", p)
}

// wantedModalities returns (wantsText, wantsAudio) for a modality slice.
// Empty input defaults to both. Sonic only emits TEXT and AUDIO; other
// modalities are silently ignored.
func wantedModalities(mods []genai.Modality) (bool, bool) {
	if len(mods) == 0 {
		return true, true
	}
	var text, audio bool
	for _, m := range mods {
		switch m {
		case genai.ModalityText:
			text = true
		case genai.ModalityAudio:
			audio = true
		case genai.ModalityUnspecified, genai.ModalityImage, genai.ModalityVideo:
			// Ignored.
		}
	}
	return text, audio
}

// toolEntriesFromADK extracts a flat list of Sonic toolSpec entries from the
// LLMRequest-style map[string]any that adk-go uses for tools. The values are
// expected to be either *genai.Tool or *genai.FunctionDeclaration; any other
// type — and any [genai.Tool] variant Sonic doesn't support — returns
// [ErrUnsupportedTool].
func toolEntriesFromADK(tools map[string]any) ([]toolEntry, error) {
	var out []toolEntry
	for name, v := range tools {
		switch tv := v.(type) {
		case *genai.Tool:
			if unsupported := mappers.UnsupportedToolVariantsFromGenai(tv); len(unsupported) > 0 {
				return nil, fmt.Errorf(
					"%w: tool %q has variants Nova Sonic doesn't accept: %s; use FunctionDeclarations only",
					ErrUnsupportedTool, name, strings.Join(unsupported, ", "),
				)
			}
			for _, fd := range tv.FunctionDeclarations {
				entry, err := toolEntryFromFunctionDeclaration(fd)
				if err != nil {
					return nil, err
				}
				out = append(out, entry)
			}
		case *genai.FunctionDeclaration:
			entry, err := toolEntryFromFunctionDeclaration(tv)
			if err != nil {
				return nil, err
			}
			out = append(out, entry)
		default:
			return nil, fmt.Errorf(
				"%w: tool %q has type %T; expected *genai.Tool or *genai.FunctionDeclaration",
				ErrUnsupportedTool, name, v,
			)
		}
	}
	return out, nil
}

func toolEntryFromFunctionDeclaration(fd *genai.FunctionDeclaration) (toolEntry, error) {
	if fd == nil || strings.TrimSpace(fd.Name) == "" {
		return toolEntry{}, errors.New("function declaration is missing a name")
	}
	if fd.Name == novagrounding.SentinelFunctionDeclarationName {
		return toolEntry{}, fmt.Errorf("%w: %s is a Converse SystemTool; Nova Sonic has no equivalent",
			ErrUnsupportedTool, novagrounding.SystemToolName)
	}
	schema, err := mappers.FunctionDeclarationSchema(fd)
	if err != nil {
		return toolEntry{}, fmt.Errorf("tool %q: %w", fd.Name, err)
	}
	raw, err := json.Marshal(schema)
	if err != nil {
		return toolEntry{}, fmt.Errorf("marshal tool %q schema: %w", fd.Name, err)
	}
	return toolEntry{
		ToolSpec: toolSpec{
			Name:        fd.Name,
			Description: fd.Description,
			InputSchema: toolInputSchema{JSON: string(raw)},
		},
	}, nil
}

// ---------- Outbound content framing helpers ----------

// BuildContentStartAudio opens a USER/AUDIO content block. sampleRateHz=0
// falls back to the package default (16 kHz).
func BuildContentStartAudio(promptName, contentName string, sampleRateHz int32) ([]byte, error) {
	if sampleRateHz == 0 {
		sampleRateHz = defaultInputSampleRate
	}
	return Wrap(eventContentStart, contentStartInput{
		PromptName:  promptName,
		ContentName: contentName,
		Type:        contentTypeAudio,
		Interactive: true,
		Role:        RoleUser,
		AudioInputConfiguration: &audioInputConfiguration{
			MediaType:       audioMediaType,
			SampleRateHertz: sampleRateHz,
			SampleSizeBits:  defaultSampleSizeBits,
			ChannelCount:    defaultChannelCount,
			AudioType:       audioTypeSpeech,
			Encoding:        audioEncodingBase64,
		},
	})
}

func BuildContentStartText(promptName, contentName, role string) ([]byte, error) {
	return Wrap(eventContentStart, contentStartInput{
		PromptName:             promptName,
		ContentName:            contentName,
		Type:                   contentTypeText,
		Interactive:            false,
		Role:                   role,
		TextInputConfiguration: &textInputConfiguration{MediaType: textMediaType},
	})
}

func BuildContentStartToolResult(promptName, contentName, toolUseID string) ([]byte, error) {
	return Wrap(eventContentStart, contentStartInput{
		PromptName:  promptName,
		ContentName: contentName,
		Type:        contentTypeTool,
		Interactive: false,
		Role:        RoleTool,
		ToolResultInputConfiguration: &toolResultInputConfiguration{
			ToolUseID:              toolUseID,
			Type:                   contentTypeText,
			TextInputConfiguration: &textInputConfiguration{MediaType: textMediaType},
		},
	})
}

func BuildAudioInput(promptName, contentName string, pcm []byte) ([]byte, error) {
	return Wrap("audioInput", audioInputEvent{
		PromptName:  promptName,
		ContentName: contentName,
		Content:     base64.StdEncoding.EncodeToString(pcm),
	})
}

func BuildTextInput(promptName, contentName, text string) ([]byte, error) {
	return Wrap("textInput", textInputEvent{
		PromptName:  promptName,
		ContentName: contentName,
		Content:     text,
	})
}

func BuildToolResult(promptName, contentName, resultJSON string) ([]byte, error) {
	return Wrap("toolResult", toolResultEvent{
		PromptName:  promptName,
		ContentName: contentName,
		Content:     resultJSON,
	})
}

func BuildContentEnd(promptName, contentName string) ([]byte, error) {
	return Wrap(eventContentEnd, contentEndInput{
		PromptName:  promptName,
		ContentName: contentName,
	})
}

func BuildPromptEnd(promptName string) ([]byte, error) {
	return Wrap("promptEnd", promptEndInput{PromptName: promptName})
}

func BuildSessionEnd() ([]byte, error) {
	return Wrap("sessionEnd", sessionEndInput{})
}

// ---------- Inbound decoder ----------

// ReadState decodes incoming envelopes into adk-go LLMResponse values. It
// tracks per-contentId type/role so that audioOutput / textOutput / toolUse
// events can be routed correctly.
type ReadState struct {
	contentType map[string]string // contentId -> "AUDIO" | "TEXT" | "TOOL"
	contentRole map[string]string // contentId -> "USER" | "ASSISTANT" | "TOOL"
	pendingTool map[string]*pendingTool

	// unknownEvents accumulates any event envelopes whose name we don't model
	// yet. They are forwarded to the final completionEnd response as
	// CustomMetadata[CustomMetadataKeyUnknownEvents] so callers can see when
	// AWS ships new Sonic events.
	unknownEvents []map[string]any

	// OnRawEvent, if set, is invoked for every server event after we extract
	// the discriminator name but before translating to LLMResponse. Used for
	// diagnostics (e.g. the web bridge logs every Sonic event).
	OnRawEvent func(name string, payload []byte)
}

type pendingTool struct {
	ToolName  string
	ToolUseID string
	Args      string
}

// NewReadState constructs an empty ReadState ready to process events.
func NewReadState() *ReadState {
	return &ReadState{
		contentType: make(map[string]string),
		contentRole: make(map[string]string),
		pendingTool: make(map[string]*pendingTool),
	}
}

// Consume decodes one raw envelope and returns the LLMResponse to emit (or
// nil when the event is purely framing). An error halts the session.
func (s *ReadState) Consume(raw []byte) (*model.LLMResponse, error) {
	var env Envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("decode envelope: %w", err)
	}
	name, payload, err := EventName(env.Event)
	if err != nil {
		return nil, err
	}
	if s.OnRawEvent != nil {
		s.OnRawEvent(name, payload)
	}
	switch name {
	case eventCompletionStart:
		return nil, nil //nolint:nilnil // Framing only — no LLMResponse to emit.
	case eventContentStart:
		var ev ContentStartOutput
		if err := json.Unmarshal(payload, &ev); err != nil {
			return nil, fmt.Errorf("decode contentStart: %w", err)
		}
		s.contentType[ev.ContentID] = ev.Type
		s.contentRole[ev.ContentID] = ev.Role
		return nil, nil //nolint:nilnil // contentStart is framing.
	case eventTextOutput:
		return s.onTextOutput(payload)
	case eventAudioOutput:
		return s.onAudioOutput(payload)
	case eventToolUse:
		return s.onToolUse(payload)
	case eventContentEnd:
		return s.onContentEnd(payload)
	case eventCompletionEnd:
		var ev CompletionEndOutput
		if err := json.Unmarshal(payload, &ev); err != nil {
			return nil, fmt.Errorf("decode completionEnd: %w", err)
		}
		resp := &model.LLMResponse{
			Content:      &genai.Content{Role: genaiRoleModel, Parts: []*genai.Part{{Text: ""}}},
			TurnComplete: true,
			FinishReason: mappers.FinishReasonFromSonicStop(ev.StopReason),
		}
		if len(s.unknownEvents) > 0 {
			resp.CustomMetadata = map[string]any{
				CustomMetadataKeyUnknownEvents: s.unknownEvents,
			}
			s.unknownEvents = nil
		}
		return resp, nil
	case eventUsageEvent:
		var ev UsageEventOutput
		if err := json.Unmarshal(payload, &ev); err != nil {
			return nil, fmt.Errorf("decode usageEvent: %w", err)
		}
		return &model.LLMResponse{
			Content:       &genai.Content{Role: genaiRoleModel, Parts: []*genai.Part{{Text: ""}}},
			UsageMetadata: usageFromSonicEvent(&ev),
		}, nil
	default:
		// Buffer the unmodeled event so callers can observe it via
		// CustomMetadata when the turn completes. Cap the slice so a
		// runaway server can't blow up memory.
		if len(s.unknownEvents) < unknownEventsCap {
			var parsed any
			if err := json.Unmarshal(payload, &parsed); err != nil {
				// If the payload isn't valid JSON, stash the raw string.
				parsed = string(payload)
			}
			s.unknownEvents = append(s.unknownEvents, map[string]any{
				"event":   name,
				"payload": parsed,
			})
		}
		return nil, nil //nolint:nilnil // Unknown event surfaced via metadata, not as an LLMResponse.
	}
}

func (s *ReadState) onTextOutput(payload json.RawMessage) (*model.LLMResponse, error) {
	var ev TextOutputEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, fmt.Errorf("decode textOutput: %w", err)
	}
	// Sonic signals barge-in via a substring on textOutput.content.
	if strings.Contains(ev.Content, interruptedMarker) {
		return &model.LLMResponse{
			Content:     &genai.Content{Role: genaiRoleModel, Parts: []*genai.Part{{Text: ""}}},
			Interrupted: true,
		}, nil
	}
	resp := &model.LLMResponse{Partial: true}
	// Prefer the role on the textOutput event itself (Sonic 2 sets it
	// directly per event); fall back to the role we stashed from the
	// preceding contentStart for older payloads.
	role := ev.Role
	if role == "" {
		role = s.contentRole[ev.ContentID]
	}
	switch role {
	case RoleUser:
		// User speech transcript.
		resp.InputTranscription = &genai.Transcription{Text: ev.Content}
	case RoleAssistant:
		// Assistant speech transcript (always treat as output transcription —
		// the assistant audio carries the actual reply).
		resp.OutputTranscription = &genai.Transcription{Text: ev.Content}
	default:
		// Treat as plain text content if role is unknown.
		resp.Content = &genai.Content{Role: genaiRoleModel, Parts: []*genai.Part{{Text: ev.Content}}}
	}
	if resp.Content == nil {
		resp.Content = &genai.Content{Role: genaiRoleModel, Parts: []*genai.Part{{Text: ""}}}
	}
	return resp, nil
}

func (s *ReadState) onAudioOutput(payload json.RawMessage) (*model.LLMResponse, error) {
	var ev AudioOutputEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, fmt.Errorf("decode audioOutput: %w", err)
	}
	data, err := base64.StdEncoding.DecodeString(ev.Content)
	if err != nil {
		return nil, fmt.Errorf("decode audioOutput.content base64: %w", err)
	}
	return &model.LLMResponse{
		Content: &genai.Content{
			Role: genaiRoleModel,
			Parts: []*genai.Part{{
				InlineData: &genai.Blob{
					MIMEType: fmt.Sprintf("audio/pcm;rate=%d", defaultOutputSampleRate),
					Data:     data,
				},
			}},
		},
		Partial: true,
	}, nil
}

func (s *ReadState) onToolUse(payload json.RawMessage) (*model.LLMResponse, error) {
	var ev ToolUseOutputEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, fmt.Errorf("decode toolUse: %w", err)
	}
	// Sonic currently delivers tool args in a single event, but the wire spec
	// frames toolUse inside contentStart/contentEnd — concat defensively in
	// case the protocol starts chunking. Identity (name + toolUseId) is set
	// only on the first chunk; subsequent chunks append to Args.
	pending, ok := s.pendingTool[ev.ContentID]
	if !ok {
		pending = &pendingTool{
			ToolName:  ev.ToolName,
			ToolUseID: ev.ToolUseID,
		}
		s.pendingTool[ev.ContentID] = pending
	}
	if pending.ToolName == "" && ev.ToolName != "" {
		pending.ToolName = ev.ToolName
	}
	if pending.ToolUseID == "" && ev.ToolUseID != "" {
		pending.ToolUseID = ev.ToolUseID
	}
	pending.Args += ev.Content
	// Don't emit the FunctionCall until contentEnd(type=TOOL) confirms it.
	return nil, nil //nolint:nilnil // Tool call is buffered until contentEnd.
}

func (s *ReadState) onContentEnd(payload json.RawMessage) (*model.LLMResponse, error) {
	var ev ContentEndOutput
	if err := json.Unmarshal(payload, &ev); err != nil {
		return nil, fmt.Errorf("decode contentEnd: %w", err)
	}
	defer func() {
		delete(s.contentType, ev.ContentID)
		delete(s.contentRole, ev.ContentID)
	}()
	if ev.Type == contentTypeTool && ev.StopReason == StopReasonToolUse {
		pending, ok := s.pendingTool[ev.ContentID]
		if !ok {
			return nil, nil //nolint:nilnil // contentEnd without buffered toolUse — drop silently.
		}
		delete(s.pendingTool, ev.ContentID)
		args := mappers.FunctionArgsFromRawJSON(pending.Args)
		return &model.LLMResponse{
			Content: &genai.Content{
				Role: genaiRoleModel,
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   pending.ToolUseID,
						Name: pending.ToolName,
						Args: args,
					},
				}},
			},
		}, nil
	}
	if ev.StopReason == StopReasonInterrupted {
		return &model.LLMResponse{
			Content:     &genai.Content{Role: genaiRoleModel, Parts: []*genai.Part{{Text: ""}}},
			Interrupted: true,
		}, nil
	}
	return nil, nil //nolint:nilnil // Other contentEnd variants are framing only.
}

func usageFromSonicEvent(ev *UsageEventOutput) *genai.GenerateContentResponseUsageMetadata {
	if ev == nil {
		return nil
	}
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     clampInt32(ev.TotalInputTokens),
		CandidatesTokenCount: clampInt32(ev.TotalOutputTokens),
		TotalTokenCount:      clampInt32(ev.TotalTokens),
	}
}

// clampInt32 safely narrows an int64 token count into the int32 fields used by
// genai. Sonic's totals are well below 2^31 in practice; we clamp defensively.
func clampInt32(v int64) int32 {
	const maxInt32 = int64(^uint32(0) >> 1) // 2^31 - 1
	switch {
	case v > maxInt32:
		return int32(maxInt32)
	case v < -maxInt32-1:
		return -int32(maxInt32) - 1
	default:
		return int32(v)
	}
}
