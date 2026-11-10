package agentcorememory

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestResolveIdentityTemplates(t *testing.T) {
	t.Parallel()

	if got := resolveActor("{userId}", "app", "u1"); got != "u1" {
		t.Errorf("resolveActor() = %q, want u1", got)
	}
	if got := resolveActor("{appName}#{userId}", "app", "u1"); got != "app#u1" {
		t.Errorf("resolveActor() = %q, want app#u1", got)
	}
	if got := resolveNamespace("/{appName}/{actorId}/{userId}", "app", "u1", "actor-1"); got != "/app/actor-1/u1" {
		t.Errorf("resolveNamespace() = %q", got)
	}
}

func TestTextAndRoleFromContent(t *testing.T) {
	t.Parallel()

	if got := textFromContent(nil); got != "" {
		t.Errorf("textFromContent(nil) = %q, want empty", got)
	}
	content := &genai.Content{Parts: []*genai.Part{{Text: "a"}, {Text: ""}, {Text: "b"}}}
	if got := textFromContent(content); got != "a\nb" {
		t.Errorf("textFromContent() = %q, want a newline-separated value", got)
	}

	for _, tc := range []struct {
		role string
		want types.Role
	}{
		{genai.RoleUser, types.RoleUser},
		{genai.RoleModel, types.RoleAssistant},
		{"function", types.RoleOther},
	} {
		if got := toAgentCoreRole(genai.NewContentFromText("x", genai.Role(tc.role))); got != tc.want {
			t.Errorf("toAgentCoreRole(%q) = %q, want %q", tc.role, got, tc.want)
		}
	}
	if got := toAgentCoreRole(nil); got != types.RoleOther {
		t.Errorf("toAgentCoreRole(nil) = %q, want OTHER", got)
	}
}

func TestEventTimestampAndMetadata(t *testing.T) {
	t.Parallel()

	sess := newSession(t, "app", "u1", nil)
	wantTime := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	event := &session.Event{Timestamp: wantTime, Author: "user", ID: "event-1"}
	if got := eventTimestamp(event, sess); !got.Equal(wantTime) {
		t.Errorf("eventTimestamp() = %v, want %v", got, wantTime)
	}

	md := eventMetadata(sess, event)
	assertMetadata(t, md, "app_name", "app")
	assertMetadata(t, md, "user_id", "u1")
	assertMetadata(t, md, "session_id", sess.ID())
	assertMetadata(t, md, "author", "user")
	assertMetadata(t, md, "event_id", "event-1")
}

func TestRecordToEntry(t *testing.T) {
	t.Parallel()

	created := time.Date(2026, 5, 6, 7, 8, 9, 0, time.UTC)
	entry := recordToEntry(types.MemoryRecordSummary{
		MemoryRecordId:   aws.String("rec-1"),
		Content:          &types.MemoryContentMemberText{Value: "remembered"},
		CreatedAt:        aws.Time(created),
		MemoryStrategyId: aws.String("strategy-1"),
		Namespaces:       []string{"/actors/u1/facts"},
		Score:            aws.Float64(0.75),
	})
	if entry.ID != "rec-1" || entry.Content.Parts[0].Text != "remembered" {
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
