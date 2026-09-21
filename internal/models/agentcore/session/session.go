package session

import "google.golang.org/adk/v2/session"

// EventEnvelope is the persisted representation of an ADK session event.
type EventEnvelope struct {
	Schema string         `json:"schema"`
	Event  *session.Event `json:"event"`
}
