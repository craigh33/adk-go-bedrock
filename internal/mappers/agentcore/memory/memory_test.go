package memory

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

type testSession struct {
	appName   string
	userID    string
	id        string
	updatedAt time.Time
}

func (s testSession) ID() string                { return s.id }
func (s testSession) AppName() string           { return s.appName }
func (s testSession) UserID() string            { return s.userID }
func (testSession) State() session.State        { return nil }
func (testSession) Events() session.Events      { return nil }
func (s testSession) LastUpdateTime() time.Time { return s.updatedAt }

func TestResolveIdentityTemplates(t *testing.T) {
	t.Parallel()

	if got := ResolveActor("{userId}", "app", "u1"); got != "u1" {
		t.Errorf("ResolveActor() = %q, want u1", got)
	}
	if got := ResolveActor("{appName}#{userId}", "app", "u1"); got != "app#u1" {
		t.Errorf("ResolveActor() = %q, want app#u1", got)
	}
	if got := ResolveNamespace("/{appName}/{actorId}/{userId}", "app", "u1", "actor-1"); got != "/app/actor-1/u1" {
		t.Errorf("ResolveNamespace() = %q", got)
	}
}

func TestTextAndRoleFromContent(t *testing.T) {
	t.Parallel()

	if got := TextFromContent(nil); got != "" {
		t.Errorf("TextFromContent(nil) = %q, want empty", got)
	}
	content := &genai.Content{Parts: []*genai.Part{{Text: "a"}, {Text: ""}, {Text: "b"}}}
	if got := TextFromContent(content); got != "a\nb" {
		t.Errorf("TextFromContent() = %q, want a newline-separated value", got)
	}

	for _, tc := range []struct {
		role string
		want types.Role
	}{
		{genai.RoleUser, types.RoleUser},
		{genai.RoleModel, types.RoleAssistant},
		{"function", types.RoleOther},
	} {
		if got := ToAgentCoreRole(genai.NewContentFromText("x", genai.Role(tc.role))); got != tc.want {
			t.Errorf("ToAgentCoreRole(%q) = %q, want %q", tc.role, got, tc.want)
		}
	}
	if got := ToAgentCoreRole(nil); got != types.RoleOther {
		t.Errorf("ToAgentCoreRole(nil) = %q, want OTHER", got)
	}
}

func TestEventTimestampAndMetadata(t *testing.T) {
	t.Parallel()

	updatedAt := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	sess := testSession{appName: "app", userID: "u1", id: "session-1", updatedAt: updatedAt}
	eventTime := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	event := &session.Event{Timestamp: eventTime, Author: "user", ID: "event-1"}
	if got := EventTimestamp(event, sess); !got.Equal(eventTime) {
		t.Errorf("EventTimestamp() = %v, want %v", got, eventTime)
	}
	event.Timestamp = time.Time{}
	if got := EventTimestamp(event, sess); !got.Equal(updatedAt) {
		t.Errorf("EventTimestamp() fallback = %v, want %v", got, updatedAt)
	}

	md := EventMetadata(sess, event)
	assertMetadata(t, md, "app_name", "app")
	assertMetadata(t, md, "user_id", "u1")
	assertMetadata(t, md, "session_id", "session-1")
	assertMetadata(t, md, "author", "user")
	assertMetadata(t, md, "event_id", "event-1")
}

func TestRecordToEntry(t *testing.T) {
	t.Parallel()

	created := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	entry := RecordToEntry(types.MemoryRecordSummary{
		MemoryRecordId:   aws.String("rec-1"),
		Content:          &types.MemoryContentMemberText{Value: "remembered"},
		CreatedAt:        aws.Time(created),
		MemoryStrategyId: aws.String("strategy-1"),
		Namespaces:       []string{"/actors/u1/facts"},
		Score:            aws.Float64(0.75),
	})
	if entry.ID != "rec-1" || entry.Content == nil || len(entry.Content.Parts) == 0 ||
		entry.Content.Parts[0].Text != "remembered" {
		t.Fatalf("entry = %+v, want mapped id and content", entry)
	}
	if !entry.Timestamp.Equal(created) {
		t.Errorf("Timestamp = %v, want %v", entry.Timestamp, created)
	}
	if entry.CustomMetadata["score"] != 0.75 || entry.CustomMetadata["memory_strategy_id"] != "strategy-1" {
		t.Errorf("CustomMetadata = %#v", entry.CustomMetadata)
	}
	if got := entry.CustomMetadata["namespaces"].([]string); len(got) != 1 || got[0] != "/actors/u1/facts" {
		t.Errorf("namespaces = %#v", got)
	}
	if got := memoryContentText(&types.MemoryContentMemberText{Value: "text"}); got != "text" {
		t.Errorf("memoryContentText() = %q, want text", got)
	}
	if got := memoryContentText(nil); got != "" {
		t.Errorf("memoryContentText(nil) = %q, want empty", got)
	}
}

func assertMetadata(t *testing.T, md map[string]types.MetadataValue, key, want string) {
	t.Helper()
	value, ok := md[key].(*types.MetadataValueMemberStringValue)
	if !ok {
		t.Fatalf("metadata %q = %T, want string value", key, md[key])
	}
	if value.Value != want {
		t.Errorf("metadata %q = %q, want %q", key, value.Value, want)
	}
}
