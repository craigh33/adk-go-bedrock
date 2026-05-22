package webbridge

import (
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

// LiveRequestJSON mirrors server/adkrest/internal/models.LiveRequest — the
// shape the embedded Angular UI sends as text frames over the /run_live
// WebSocket. Exported so users wiring their own UI can stay schema-compatible.
type LiveRequestJSON struct {
	Content       *genai.Content       `json:"content,omitempty"`
	Blob          *LiveBlobJSON        `json:"blob,omitempty"`
	ActivityStart *genai.ActivityStart `json:"activityStart,omitempty"`
	ActivityEnd   *genai.ActivityEnd   `json:"activityEnd,omitempty"`
	Close         bool                 `json:"close,omitempty"`
}

// LiveBlobJSON carries base64-encoded audio bytes inside a text frame.
// `encoding/json` handles the base64 round-trip on []byte automatically.
type LiveBlobJSON struct {
	MIMEType string `json:"mime_type,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// LiveEventJSON mirrors server/adkrest/internal/models.Event — the shape the
// UI's decoder expects on the wire. Audio rides inside
// Content.Parts[].InlineData.Data (base64-encoded by encoding/json).
type LiveEventJSON struct {
	ID                  string                                      `json:"id"`
	InvocationID        string                                      `json:"invocationId"`
	Author              string                                      `json:"author"`
	Partial             bool                                        `json:"partial,omitempty"`
	Content             *genai.Content                              `json:"content"`
	UsageMetadata       *genai.GenerateContentResponseUsageMetadata `json:"usageMetadata"`
	TurnComplete        bool                                        `json:"turnComplete,omitempty"`
	Interrupted         bool                                        `json:"interrupted,omitempty"`
	FinishReason        genai.FinishReason                          `json:"finishReason,omitempty"`
	InputTranscription  *genai.Transcription                        `json:"inputTranscription,omitempty"`
	OutputTranscription *genai.Transcription                        `json:"outputTranscription,omitempty"`
	Actions             LiveEventActionsJSON                        `json:"actions"`
}

// LiveEventActionsJSON is the subset of session.EventActions the UI consumes.
type LiveEventActionsJSON struct {
	StateDelta    map[string]any   `json:"stateDelta,omitempty"`
	ArtifactDelta map[string]int64 `json:"artifactDelta,omitempty"`
}

// FromSessionEvent flattens a session.Event into the JSON shape the UI
// expects. Use directly if you build a different mount than [New].
func FromSessionEvent(ev *session.Event) LiveEventJSON {
	if ev == nil {
		return LiveEventJSON{}
	}
	return LiveEventJSON{
		ID:                  ev.ID,
		InvocationID:        ev.InvocationID,
		Author:              ev.Author,
		Partial:             ev.Partial,
		Content:             ev.Content,
		UsageMetadata:       ev.UsageMetadata,
		TurnComplete:        ev.TurnComplete,
		Interrupted:         ev.Interrupted,
		FinishReason:        ev.FinishReason,
		InputTranscription:  ev.InputTranscription,
		OutputTranscription: ev.OutputTranscription,
		Actions: LiveEventActionsJSON{
			StateDelta:    ev.Actions.StateDelta,
			ArtifactDelta: ev.Actions.ArtifactDelta,
		},
	}
}
