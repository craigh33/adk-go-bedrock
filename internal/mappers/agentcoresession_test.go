package mappers

import (
	"encoding/json"
	"testing"
	"time"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime/types"
	"github.com/google/uuid"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestAgentCoreSessionMetadataRoundTrip(t *testing.T) {
	md, err := AgentCoreSessionMetadata("app", "user", map[string]any{
		"k":      "v",
		"count":  float64(2),
		"app:ak": "av",
	})
	if err != nil {
		t.Fatal(err)
	}

	if !AgentCoreSessionMetadataMatches(md, "app", "user") {
		t.Fatalf("metadata identifiers = %+v", md)
	}
	if !AgentCoreSessionMetadataMatchesApp(md, "app") || AgentCoreSessionMetadataUserID(md) != "user" {
		t.Fatalf("metadata app/user = %+v", md)
	}

	state, err := AgentCoreSessionStateFromMetadata(md)
	if err != nil {
		t.Fatal(err)
	}
	if state["k"] != "v" || state["count"] != float64(2) || state["app:ak"] != "av" {
		t.Fatalf("state round-trip = %+v", state)
	}
}

func TestAgentCoreSessionEncodeEventToInvocationStepPayload(t *testing.T) {
	event := &session.Event{
		ID:           "event-1",
		InvocationID: "inv-1",
		Author:       "assistant",
		Timestamp:    time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
		Actions:      session.EventActions{StateDelta: map[string]any{"k": "v"}},
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("hello", genai.RoleModel),
		},
	}

	text, err := AgentCoreSessionEncodeEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := AgentCoreSessionDecodeInvocationStep(&brtypes.InvocationStep{
		Payload: AgentCoreSessionInvocationStepPayload(text),
	})
	if !ok {
		t.Fatal("decode invocation step ok = false, want true")
	}
	if got.ID != event.ID || got.InvocationID != event.InvocationID || got.Content.Parts[0].Text != "hello" {
		t.Fatalf("decoded event = %+v", got)
	}
}

func TestAgentCoreSessionDecodeInvocationStepIgnoresForeignText(t *testing.T) {
	_, ok := AgentCoreSessionDecodeInvocationStep(&brtypes.InvocationStep{
		Payload: AgentCoreSessionInvocationStepPayload("not json"),
	})
	if ok {
		t.Fatalf("decode invocation step ok=%v, want false", ok)
	}

	raw, _ := json.Marshal(struct {
		Schema string         `json:"schema"`
		Event  *session.Event `json:"event"`
	}{Schema: "other", Event: &session.Event{ID: "e"}})
	_, ok = AgentCoreSessionDecodeInvocationStep(&brtypes.InvocationStep{
		Payload: AgentCoreSessionInvocationStepPayload(string(raw)),
	})
	if ok {
		t.Fatalf("decode invocation step ok=%v, want false", ok)
	}
}

func TestAgentCoreSessionIDMapping(t *testing.T) {
	adkUUID := uuid.NewString()
	if got := AgentCoreSessionInvocationID("session", adkUUID, "event"); got != adkUUID {
		t.Fatalf("uuid invocation id = %q, want %q", got, adkUUID)
	}

	got1 := AgentCoreSessionInvocationID("session", "adk-invocation", "event-1")
	got2 := AgentCoreSessionInvocationID("session", "adk-invocation", "event-2")
	if got1 != got2 {
		t.Fatalf("same ADK invocation mapped to different Bedrock IDs: %q %q", got1, got2)
	}
	if _, err := uuid.Parse(got1); err != nil {
		t.Fatalf("mapped invocation id is not UUID: %q", got1)
	}

	event := &session.Event{ID: "event-1", InvocationID: "inv-1", Timestamp: time.Unix(1, 2)}
	stepID := AgentCoreSessionStepID("session", event)
	if _, err := uuid.Parse(stepID); err != nil {
		t.Fatalf("mapped step id is not UUID: %q", stepID)
	}
}
