package memory

import (
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	adkmemory "google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

// ResolveActor maps ADK identity to an AgentCore actor id using template. It
// supports the {userId} and {appName} placeholders.
func ResolveActor(template, appName, userID string) string {
	return strings.NewReplacer(
		"{userId}", userID,
		"{appName}", appName,
	).Replace(template)
}

// ResolveNamespace substitutes the {actorId}, {userId} and {appName} placeholders
// in an AgentCore namespace (or namespace path).
func ResolveNamespace(template, appName, userID, actorID string) string {
	return strings.NewReplacer(
		"{actorId}", actorID,
		"{userId}", userID,
		"{appName}", appName,
	).Replace(template)
}

// TextFromContent concatenates the text of all textual parts of c, separating
// non-empty parts with a newline. It returns "" when c is nil or carries no text
// (for example a bare function call/response), signalling the event should be
// skipped.
func TextFromContent(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p == nil || p.Text == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

// ToAgentCoreRole maps a genai content role to an AgentCore conversational role.
func ToAgentCoreRole(c *genai.Content) types.Role {
	if c == nil {
		return types.RoleOther
	}
	switch c.Role {
	case genai.RoleUser:
		return types.RoleUser
	case genai.RoleModel:
		return types.RoleAssistant
	default:
		return types.RoleOther
	}
}

// EventTimestamp returns the timestamp to record for event, falling back to the
// session's last-update time and finally the current time. AgentCore requires a
// non-nil EventTimestamp.
func EventTimestamp(event *session.Event, sess session.Session) time.Time {
	if !event.Timestamp.IsZero() {
		return event.Timestamp
	}
	if t := sess.LastUpdateTime(); !t.IsZero() {
		return t
	}
	return time.Now()
}

// EventMetadata attaches ADK identifiers to an AgentCore event for traceability.
func EventMetadata(sess session.Session, event *session.Event) map[string]types.MetadataValue {
	md := map[string]types.MetadataValue{
		"app_name":   &types.MetadataValueMemberStringValue{Value: sess.AppName()},
		"user_id":    &types.MetadataValueMemberStringValue{Value: sess.UserID()},
		"session_id": &types.MetadataValueMemberStringValue{Value: sess.ID()},
	}
	if event.Author != "" {
		md["author"] = &types.MetadataValueMemberStringValue{Value: event.Author}
	}
	if event.ID != "" {
		md["event_id"] = &types.MetadataValueMemberStringValue{Value: event.ID}
	}
	return md
}

// RecordToEntry converts an AgentCore memory record summary to an ADK memory
// entry. The record's relevance score, strategy and namespaces are preserved as
// custom metadata.
func RecordToEntry(rec types.MemoryRecordSummary) adkmemory.Entry {
	entry := adkmemory.Entry{
		ID:        aws.ToString(rec.MemoryRecordId),
		Content:   genai.NewContentFromText(memoryContentText(rec.Content), genai.RoleUser),
		Timestamp: aws.ToTime(rec.CreatedAt),
	}

	meta := map[string]any{}
	if rec.Score != nil {
		meta["score"] = *rec.Score
	}
	if rec.MemoryStrategyId != nil {
		meta["memory_strategy_id"] = *rec.MemoryStrategyId
	}
	if len(rec.Namespaces) > 0 {
		meta["namespaces"] = rec.Namespaces
	}
	if len(meta) > 0 {
		entry.CustomMetadata = meta
	}
	return entry
}

// memoryContentText extracts the text from an AgentCore memory content union.
func memoryContentText(c types.MemoryContent) string {
	if t, ok := c.(*types.MemoryContentMemberText); ok {
		return t.Value
	}
	return ""
}
