package memory

import (
	"crypto/sha256"
	"strconv"

	"github.com/google/uuid"
	"google.golang.org/adk/v2/session"
)

// EventClientToken preserves existing UUID tokens and scopes custom event IDs
// to the memory resource, destination actor, and source ADK session.
func EventClientToken(memoryID, actorID string, sess session.Session, eventID string) string {
	const (
		uuidStringLength = 36
		customUUID       = 8
	)
	if len(eventID) == uuidStringLength && uuid.Validate(eventID) == nil {
		return eventID
	}

	// Quote each field so arbitrary IDs cannot change the boundaries between fields.
	var key []byte
	for _, value := range [...]string{
		"adk-go-bedrock/agentcore/memory/event", memoryID, actorID, sess.AppName(), sess.UserID(), sess.ID(), eventID,
	} {
		key = strconv.AppendQuote(key, value)
	}
	return uuid.NewHash(sha256.New(), uuid.NameSpaceOID, key, customUUID).String()
}
