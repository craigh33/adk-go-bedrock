package memory

import (
	"errors"
	"regexp"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	adkmemory "google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

var namespaceIdentityPattern = regexp.MustCompile(`^[A-Za-z0-9_-]+(:[A-Za-z0-9_-]+)*$`)

// ResolveNamespace substitutes the {actorId}, {userId} and {appName} placeholders
// in an AgentCore namespace or namespace path. Raw ADK identities must fit in
// one nonempty namespace segment. Wildcards must be outside identity segments.
func ResolveNamespace(template, appName, userID, actorID string) (string, error) {
	for segment := range strings.SplitSeq(template, "/") {
		if strings.Contains(segment, "*") && strings.Contains(segment, "{") {
			return "", errors.New("namespace placeholders cannot share a segment with a wildcard")
		}
	}
	if strings.Contains(template, "{userId}") && !namespaceIdentityPattern.MatchString(userID) {
		return "", errors.New("{userId} must be a single, nonempty namespace segment")
	}
	if strings.Contains(template, "{appName}") && !namespaceIdentityPattern.MatchString(appName) {
		return "", errors.New("{appName} must be a single, nonempty namespace segment")
	}
	return strings.NewReplacer(
		"{actorId}", actorID,
		"{userId}", userID,
		"{appName}", appName,
	).Replace(template), nil
}

// TextFromContent joins conversation text with newlines and skips thought parts.
// It returns "" when there is no conversation text to store.
func TextFromContent(c *genai.Content) string {
	if c == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range c.Parts {
		if p == nil || p.Thought || p.Text == "" {
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
	md := make(map[string]types.MetadataValue)
	addMetadata(md, "app_name", sess.AppName())
	addMetadata(md, "user_id", sess.UserID())
	addMetadata(md, "session_id", sess.ID())
	if event.Author != "" {
		addMetadata(md, "author", event.Author)
	}
	if event.ID != "" {
		addMetadata(md, "event_id", event.ID)
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
