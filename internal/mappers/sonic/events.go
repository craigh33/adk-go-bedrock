// Package sonic implements the Amazon Nova Sonic bidirectional-streaming wire
// format used by [github.com/craigh33/adk-go-bedrock/bedrock/live]. It defines
// the JSON event envelopes documented at
// https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html
// and translates them to/from adk-go's [model.LLMResponse] / [agent.LiveRequest]
// types.
//
// This package lives under internal/ so its types are only visible within this
// module. Exported symbols are the minimum the bedrock/live package (and its
// tests) need.
package sonic

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Envelope is the top-level Nova Sonic wire shape: every event is sent as
// {"event":{"<name>":{...}}}.
type Envelope struct {
	Event json.RawMessage `json:"event"`
}

// EventName extracts the discriminator key from a Sonic event envelope.
// Returns the event name (e.g. "audioOutput"), the raw JSON body under that
// key, and an error if the envelope is malformed.
func EventName(raw json.RawMessage) (string, json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return "", nil, fmt.Errorf("decode event discriminator: %w", err)
	}
	if len(m) != 1 {
		return "", nil, fmt.Errorf("expected exactly one event key, got %d", len(m))
	}
	for k, v := range m {
		return k, v, nil
	}
	return "", nil, errors.New("empty event envelope")
}

// Wrap renders a single-key event envelope as the raw bytes Bedrock expects.
// Exposed because tests build fake server-side envelopes with it.
func Wrap(eventName string, payload any) ([]byte, error) {
	inner, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	// Manual stitch keeps the discriminator key first and avoids an extra map
	// allocation per chunk.
	return fmt.Appendf(nil, `{"event":{%q:%s}}`, eventName, inner), nil
}

// Role enum used on contentStart events (input and output sides).
const (
	RoleSystem    = "SYSTEM"
	RoleUser      = "USER"
	RoleAssistant = "ASSISTANT"
	RoleTool      = "TOOL"
)

// Content type enum used on contentStart events.
const (
	contentTypeText  = "TEXT"
	contentTypeAudio = "AUDIO"
	contentTypeTool  = "TOOL"
)

// Stop reason values that appear on contentEnd / completionEnd events.
const (
	StopReasonPartialTurn = "PARTIAL_TURN"
	StopReasonEndTurn     = "END_TURN"
	StopReasonInterrupted = "INTERRUPTED"
	StopReasonToolUse     = "TOOL_USE"
)

// Wire event-name discriminators (internal use only).
const (
	eventContentStart    = "contentStart"
	eventContentEnd      = "contentEnd"
	eventTextOutput      = "textOutput"
	eventAudioOutput     = "audioOutput"
	eventToolUse         = "toolUse"
	eventCompletionStart = "completionStart"
	eventCompletionEnd   = "completionEnd"
	eventUsageEvent      = "usageEvent"
)

// genai role label used on emitted assistant content.
const genaiRoleModel = "model"

// ---------- Input events (client → server) ----------

type sessionStartInput struct {
	InferenceConfiguration *inferenceConfiguration `json:"inferenceConfiguration,omitempty"`
}

type inferenceConfiguration struct {
	MaxTokens   *int32   `json:"maxTokens,omitempty"`
	TopP        *float32 `json:"topP,omitempty"`
	Temperature *float32 `json:"temperature,omitempty"`
}

type promptStartInput struct {
	PromptName                 string                      `json:"promptName"`
	TextOutputConfiguration    *textOutputConfiguration    `json:"textOutputConfiguration,omitempty"`
	AudioOutputConfiguration   *audioOutputConfiguration   `json:"audioOutputConfiguration,omitempty"`
	ToolUseOutputConfiguration *toolUseOutputConfiguration `json:"toolUseOutputConfiguration,omitempty"`
	ToolConfiguration          *toolConfiguration          `json:"toolConfiguration,omitempty"`
}

type textOutputConfiguration struct {
	MediaType string `json:"mediaType"`
}

type audioOutputConfiguration struct {
	MediaType       string `json:"mediaType"`
	SampleRateHertz int32  `json:"sampleRateHertz"`
	SampleSizeBits  int32  `json:"sampleSizeBits"`
	ChannelCount    int32  `json:"channelCount"`
	VoiceID         string `json:"voiceId,omitempty"`
	Encoding        string `json:"encoding,omitempty"`
	AudioType       string `json:"audioType,omitempty"`
}

type toolUseOutputConfiguration struct {
	MediaType string `json:"mediaType"`
}

type toolConfiguration struct {
	Tools []toolEntry `json:"tools"`
}

type toolEntry struct {
	ToolSpec toolSpec `json:"toolSpec"`
}

type toolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema toolInputSchema `json:"inputSchema"`
}

// toolInputSchema.JSON is a *stringified* JSON object, per Nova Sonic docs.
type toolInputSchema struct {
	JSON string `json:"json"`
}

// contentStartInput is the input-side contentStart. It carries a USER/AUDIO,
// SYSTEM/TEXT, USER/TEXT, ASSISTANT/TEXT, or TOOL block configuration.
type contentStartInput struct {
	PromptName                   string                        `json:"promptName"`
	ContentName                  string                        `json:"contentName"`
	Type                         string                        `json:"type"`
	Interactive                  bool                          `json:"interactive"`
	Role                         string                        `json:"role"`
	TextInputConfiguration       *textInputConfiguration       `json:"textInputConfiguration,omitempty"`
	AudioInputConfiguration      *audioInputConfiguration      `json:"audioInputConfiguration,omitempty"`
	ToolResultInputConfiguration *toolResultInputConfiguration `json:"toolResultInputConfiguration,omitempty"`
}

type textInputConfiguration struct {
	MediaType string `json:"mediaType"`
}

type audioInputConfiguration struct {
	MediaType       string `json:"mediaType"`
	SampleRateHertz int32  `json:"sampleRateHertz"`
	SampleSizeBits  int32  `json:"sampleSizeBits"`
	ChannelCount    int32  `json:"channelCount"`
	AudioType       string `json:"audioType,omitempty"`
	Encoding        string `json:"encoding,omitempty"`
}

type toolResultInputConfiguration struct {
	ToolUseID              string                  `json:"toolUseId"`
	Type                   string                  `json:"type"`
	TextInputConfiguration *textInputConfiguration `json:"textInputConfiguration,omitempty"`
}

type textInputEvent struct {
	PromptName  string `json:"promptName"`
	ContentName string `json:"contentName"`
	Content     string `json:"content"`
}

// audioInputEvent.Content is base64-encoded LPCM bytes.
type audioInputEvent struct {
	PromptName  string `json:"promptName"`
	ContentName string `json:"contentName"`
	Content     string `json:"content"`
}

// toolResultEvent.Content is a stringified JSON result.
type toolResultEvent struct {
	PromptName  string `json:"promptName"`
	ContentName string `json:"contentName"`
	Content     string `json:"content"`
}

type contentEndInput struct {
	PromptName  string `json:"promptName"`
	ContentName string `json:"contentName"`
}

type promptEndInput struct {
	PromptName string `json:"promptName"`
}

type sessionEndInput struct{}

// ---------- Output events (server → client) ----------
//
// These are exported so tests in the bedrock/live package can synthesize
// fake server envelopes via [Wrap]. Field sets differ slightly per type but
// we share one struct where the schema is forgiving — JSON's missing-field
// rules keep zero values where the server omits the field.

type CompletionStartOutput struct {
	SessionID    string `json:"sessionId"`
	PromptName   string `json:"promptName"`
	CompletionID string `json:"completionId"`
}

type ContentStartOutput struct {
	SessionID                  string                      `json:"sessionId"`
	PromptName                 string                      `json:"promptName"`
	CompletionID               string                      `json:"completionId"`
	ContentID                  string                      `json:"contentId"`
	Type                       string                      `json:"type"`
	Role                       string                      `json:"role"`
	AdditionalModelFields      string                      `json:"additionalModelFields,omitempty"`
	TextOutputConfiguration    *textOutputConfiguration    `json:"textOutputConfiguration,omitempty"`
	AudioOutputConfiguration   *audioOutputConfiguration   `json:"audioOutputConfiguration,omitempty"`
	ToolUseOutputConfiguration *toolUseOutputConfiguration `json:"toolUseOutputConfiguration,omitempty"`
}

type TextOutputEvent struct {
	SessionID    string `json:"sessionId"`
	PromptName   string `json:"promptName"`
	CompletionID string `json:"completionId"`
	ContentID    string `json:"contentId"`
	Role         string `json:"role"`
	Content      string `json:"content"`
}

// AudioOutputEvent carries audio frames from Sonic to the client. Content is
// base64-encoded 24 kHz LPCM by default.
type AudioOutputEvent struct {
	SessionID    string `json:"sessionId"`
	PromptName   string `json:"promptName"`
	CompletionID string `json:"completionId"`
	ContentID    string `json:"contentId"`
	Content      string `json:"content"`
}

type ToolUseOutputEvent struct {
	SessionID    string `json:"sessionId"`
	PromptName   string `json:"promptName"`
	CompletionID string `json:"completionId"`
	ContentID    string `json:"contentId"`
	ToolName     string `json:"toolName"`
	ToolUseID    string `json:"toolUseId"`
	Content      string `json:"content"`
}

type ContentEndOutput struct {
	SessionID    string `json:"sessionId"`
	PromptName   string `json:"promptName"`
	CompletionID string `json:"completionId"`
	ContentID    string `json:"contentId"`
	Type         string `json:"type"`
	StopReason   string `json:"stopReason"`
}

type CompletionEndOutput struct {
	SessionID    string `json:"sessionId"`
	PromptName   string `json:"promptName"`
	CompletionID string `json:"completionId"`
	StopReason   string `json:"stopReason"`
}

type UsageEventOutput struct {
	SessionID         string       `json:"sessionId"`
	PromptName        string       `json:"promptName"`
	CompletionID      string       `json:"completionId"`
	Details           UsageDetails `json:"details"`
	TotalInputTokens  int64        `json:"totalInputTokens"`
	TotalOutputTokens int64        `json:"totalOutputTokens"`
	TotalTokens       int64        `json:"totalTokens"`
}

type UsageDetails struct {
	Delta UsageBucket `json:"delta"`
	Total UsageBucket `json:"total"`
}

type UsageBucket struct {
	Input  UsageTokens `json:"input"`
	Output UsageTokens `json:"output"`
}

type UsageTokens struct {
	SpeechTokens int64 `json:"speechTokens"`
	TextTokens   int64 `json:"textTokens"`
}
