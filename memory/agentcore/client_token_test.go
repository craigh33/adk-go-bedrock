package agentcorememory

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/google/uuid"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestAddSessionToMemoryCustomEventToken(t *testing.T) {
	t.Parallel()

	const eventID = "event:1"
	ctx := t.Context()
	api := &fakeAPI{}
	mem, err := NewWithAPI(api, "mem-1")
	if err != nil {
		t.Fatal(err)
	}
	sessions := session.InMemoryService()
	for _, sessionID := range []string{"first", "second"} {
		created, err := sessions.Create(ctx, &session.CreateRequest{
			AppName:   "app",
			UserID:    "user",
			SessionID: sessionID,
		})
		if err != nil {
			t.Fatal(err)
		}
		event := session.NewEvent(ctx, "invocation")
		event.ID = eventID
		event.Content = genai.NewContentFromText("Hello", genai.RoleUser)
		if err := sessions.AppendEvent(ctx, created.Session, event); err != nil {
			t.Fatal(err)
		}
		for range 2 {
			if err := mem.AddSessionToMemory(ctx, created.Session); err != nil {
				t.Fatal(err)
			}
		}
	}

	if got := len(api.createInputs); got != 4 {
		t.Fatalf("CreateEvent calls = %d, want 4", got)
	}
	first := aws.ToString(api.createInputs[0].ClientToken)
	second := aws.ToString(api.createInputs[2].ClientToken)
	if first == second {
		t.Errorf("both sessions used client token %q for the same custom event ID", first)
	}
	want := []string{first, first, second, second}
	for i, in := range api.createInputs {
		token := aws.ToString(in.ClientToken)
		if err := uuid.Validate(token); err != nil {
			t.Errorf("call %d client token %q is not a UUID: %v", i, token, err)
		}
		if token != want[i] {
			t.Errorf("call %d client token = %q, want %q", i, token, want[i])
		}
		assertMetadata(t, in.Metadata, "event_id", eventID)
	}
}
