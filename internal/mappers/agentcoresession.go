package mappers

import (
	"encoding/json"
	"fmt"
	"time"

	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime/types"
	"github.com/google/uuid"
	"google.golang.org/adk/session"
)

const (
	agentCoreSessionMetaSchema = "adk_go_bedrock_schema"
	agentCoreSessionMetaApp    = "adk_app_name"
	agentCoreSessionMetaUser   = "adk_user_id"
	agentCoreSessionMetaState  = "adk_state_json"

	agentCoreSessionSchema      = "agentcoresession.v1"
	agentCoreSessionEventSchema = "agentcoresession.event.v1"
	agentCoreSessionUUIDPrefix  = "adk-go-bedrock/session/agentcoresession"
)

// AgentCoreSessionMetadata maps ADK session identity and state into AgentCore session metadata.
func AgentCoreSessionMetadata(appName, userID string, state map[string]any) (map[string]string, error) {
	stateJSON, err := json.Marshal(state)
	if err != nil {
		return nil, fmt.Errorf("encode session state metadata: %w", err)
	}
	return map[string]string{
		agentCoreSessionMetaSchema: agentCoreSessionSchema,
		agentCoreSessionMetaApp:    appName,
		agentCoreSessionMetaUser:   userID,
		agentCoreSessionMetaState:  string(stateJSON),
	}, nil
}

// AgentCoreSessionStateFromMetadata maps AgentCore session metadata back to ADK session state.
func AgentCoreSessionStateFromMetadata(md map[string]string) (map[string]any, error) {
	stateJSON := md[agentCoreSessionMetaState]
	if stateJSON == "" {
		return map[string]any{}, nil
	}
	var state map[string]any
	if err := json.Unmarshal([]byte(stateJSON), &state); err != nil {
		return nil, fmt.Errorf("decode session state metadata: %w", err)
	}
	if state == nil {
		state = map[string]any{}
	}
	return state, nil
}

// AgentCoreSessionMetadataMatches returns true when metadata belongs to the ADK app/user pair.
func AgentCoreSessionMetadataMatches(md map[string]string, appName, userID string) bool {
	return AgentCoreSessionMetadataMatchesApp(md, appName) && AgentCoreSessionMetadataUserID(md) == userID
}

// AgentCoreSessionMetadataMatchesApp returns true when metadata belongs to the ADK app.
func AgentCoreSessionMetadataMatchesApp(md map[string]string, appName string) bool {
	return md[agentCoreSessionMetaSchema] == agentCoreSessionSchema && md[agentCoreSessionMetaApp] == appName
}

// AgentCoreSessionMetadataUserID returns the ADK user ID stored in AgentCore metadata.
func AgentCoreSessionMetadataUserID(md map[string]string) string {
	return md[agentCoreSessionMetaUser]
}

type agentCoreSessionEventEnvelope struct {
	Schema string         `json:"schema"`
	Event  *session.Event `json:"event"`
}

// AgentCoreSessionEncodeEvent maps an ADK event to an AgentCore invocation step text payload.
func AgentCoreSessionEncodeEvent(event *session.Event) (string, error) {
	b, err := json.Marshal(agentCoreSessionEventEnvelope{Schema: agentCoreSessionEventSchema, Event: event})
	if err != nil {
		return "", fmt.Errorf("encode event: %w", err)
	}
	return string(b), nil
}

// AgentCoreSessionInvocationStepPayload maps text into an AgentCore invocation step payload.
func AgentCoreSessionInvocationStepPayload(text string) brtypes.InvocationStepPayload {
	return &brtypes.InvocationStepPayloadMemberContentBlocks{Value: []brtypes.BedrockSessionContentBlock{
		&brtypes.BedrockSessionContentBlockMemberText{Value: text},
	}}
}

// AgentCoreSessionDecodeInvocationStep maps an adapter-owned AgentCore invocation step to an ADK event.
func AgentCoreSessionDecodeInvocationStep(step *brtypes.InvocationStep) (*session.Event, bool) {
	if step == nil {
		return nil, false
	}
	payload, ok := step.Payload.(*brtypes.InvocationStepPayloadMemberContentBlocks)
	if !ok {
		return nil, false
	}
	for _, block := range payload.Value {
		text, ok := block.(*brtypes.BedrockSessionContentBlockMemberText)
		if !ok || text.Value == "" {
			continue
		}
		var env agentCoreSessionEventEnvelope
		if err := json.Unmarshal([]byte(text.Value), &env); err != nil {
			continue
		}
		if env.Schema != agentCoreSessionEventSchema || env.Event == nil {
			continue
		}
		return env.Event, true
	}
	return nil, false
}

// AgentCoreSessionInvocationID maps an ADK invocation ID to a Bedrock UUID.
func AgentCoreSessionInvocationID(sessionID, adkInvocationID, eventID string) string {
	if id, err := uuid.Parse(adkInvocationID); err == nil {
		return id.String()
	}
	basis := adkInvocationID
	if basis == "" {
		basis = eventID
	}
	return uuid.NewSHA1(
		uuid.NameSpaceURL,
		[]byte(agentCoreSessionUUIDPrefix+":session:"+sessionID+":invocation:"+basis),
	).String()
}

// AgentCoreSessionStepID maps an ADK event ID to a Bedrock invocation step UUID.
func AgentCoreSessionStepID(sessionID string, event *session.Event) string {
	if id, err := uuid.Parse(event.ID); err == nil {
		return id.String()
	}
	name := agentCoreSessionUUIDPrefix +
		":session:" + sessionID +
		":step:" + event.InvocationID +
		":" + event.ID +
		":" + event.Timestamp.Format(time.RFC3339Nano)
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte(name)).String()
}
