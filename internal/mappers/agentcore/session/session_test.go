package session

import (
	"encoding/json"
	"testing"
	"time"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime/types"
	"github.com/google/uuid"
	"google.golang.org/adk/v2/model"
	adksession "google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	sessionmodels "github.com/craigh33/adk-go-bedrock/internal/models/agentcore/session"
)

func TestMetadataRoundTrip(t *testing.T) {
	md, err := Metadata("app", "user", map[string]any{
		"k":      "v",
		"count":  float64(2),
		"app:ak": "av",
	})
	if err != nil {
		t.Fatal(err)
	}

	if !MetadataMatches(md, "app", "user") {
		t.Fatalf("metadata identifiers = %+v", md)
	}
	if !MetadataMatchesApp(md, "app") || MetadataUserID(md) != "user" {
		t.Fatalf("metadata app/user = %+v", md)
	}

	state, err := StateFromMetadata(md)
	if err != nil {
		t.Fatal(err)
	}
	if state["k"] != "v" || state["count"] != float64(2) || state["app:ak"] != "av" {
		t.Fatalf("state round-trip = %+v", state)
	}
}

func TestEncodeEventToInvocationStepPayload(t *testing.T) {
	event := &adksession.Event{
		ID:           "event-1",
		InvocationID: "inv-1",
		Author:       "assistant",
		Timestamp:    time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
		Actions:      adksession.EventActions{StateDelta: map[string]any{"k": "v"}},
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("hello", genai.RoleModel),
		},
	}

	text, err := EncodeEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	got, ok := DecodeInvocationStep(&brtypes.InvocationStep{
		Payload: InvocationStepPayload(text),
	})
	if !ok {
		t.Fatal("decode invocation step ok = false, want true")
	}
	if got.ID != event.ID || got.InvocationID != event.InvocationID || got.Content.Parts[0].Text != "hello" {
		t.Fatalf("decoded event = %+v", got)
	}
}

func TestDecodeInvocationStepIgnoresForeignText(t *testing.T) {
	_, ok := DecodeInvocationStep(&brtypes.InvocationStep{
		Payload: InvocationStepPayload("not json"),
	})
	if ok {
		t.Fatalf("decode invocation step ok=%v, want false", ok)
	}

	raw, _ := json.Marshal(sessionmodels.EventEnvelope{
		Schema: "other",
		Event:  &adksession.Event{ID: "e"},
	})
	_, ok = DecodeInvocationStep(&brtypes.InvocationStep{
		Payload: InvocationStepPayload(string(raw)),
	})
	if ok {
		t.Fatalf("decode invocation step ok=%v, want false", ok)
	}
}

func TestIDMapping(t *testing.T) {
	adkUUID := uuid.NewString()
	if got := InvocationID("session", adkUUID, "event"); got != adkUUID {
		t.Fatalf("uuid invocation id = %q, want %q", got, adkUUID)
	}

	got1 := InvocationID("session", "adk-invocation", "event-1")
	got2 := InvocationID("session", "adk-invocation", "event-2")
	if got1 != got2 {
		t.Fatalf("same ADK invocation mapped to different Bedrock IDs: %q %q", got1, got2)
	}
	if _, err := uuid.Parse(got1); err != nil {
		t.Fatalf("mapped invocation id is not UUID: %q", got1)
	}

	event := &adksession.Event{ID: "event-1", InvocationID: "inv-1", Timestamp: time.Unix(1, 2)}
	stepID := StepID("session", event)
	if _, err := uuid.Parse(stepID); err != nil {
		t.Fatalf("mapped step id is not UUID: %q", stepID)
	}
}
