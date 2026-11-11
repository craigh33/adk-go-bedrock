package browser

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/agent"

	browsermappers "github.com/craigh33/adk-go-bedrock/internal/agentcore/mappers/browser"
)

type conversationKey struct {
	appName, userID, sessionID string
}

type browserSession struct {
	id        string
	expiresAt time.Time
	busy      bool
}

// acquireSession gives one action exclusive use of its conversation's session.
func (t *browserTool) acquireSession(ctx agent.Context) (*browserSession, func(), error) {
	key := conversationKey{appName: ctx.AppName(), userID: ctx.UserID(), sessionID: ctx.SessionID()}
	if key.sessionID == "" {
		return nil, nil, errors.New("the Browser tool requires an ADK session")
	}
	t.sessionsMu.Lock()
	defer t.sessionsMu.Unlock()

	now := time.Now()
	if !now.Before(t.nextSessionSweep) {
		for key, session := range t.sessions {
			if !session.busy && !now.Before(session.expiresAt) {
				delete(t.sessions, key)
			}
		}
		t.nextSessionSweep = now.Add(time.Minute)
	}
	session := t.sessions[key]
	if session != nil && session.busy {
		return nil, nil, errors.New(
			"another Browser action is already in progress. Wait for it to finish before calling again",
		)
	}
	if session == nil || !now.Before(session.expiresAt) {
		session = &browserSession{}
		t.sessions[key] = session
	}
	session.busy = true
	return session, func() {
		t.sessionsMu.Lock()
		defer t.sessionsMu.Unlock()
		session.busy = false
		if session.id == "" {
			delete(t.sessions, key)
		}
	}, nil
}

func (t *browserTool) runStart(ctx agent.Context, session *browserSession) (map[string]any, error) {
	current, err := t.currentSession(ctx, session)
	if err != nil {
		return nil, err
	}
	if current != nil {
		return sessionResult(actionStart, current), nil
	}
	out, err := t.startSession(ctx, ctx, session)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionStart,
		resultKeyBrowserID: aws.ToString(out.BrowserIdentifier),
		"created_at":       browsermappers.AgentCoreBrowserTimeValue(out.CreatedAt),
		"live_view_url":    browsermappers.AgentCoreBrowserLiveViewEndpoint(out.Streams),
	}, nil
}

func (t *browserTool) runStatus(ctx agent.Context, session *browserSession) (map[string]any, error) {
	out, err := t.currentSession(ctx, session)
	if err != nil {
		return nil, err
	}
	if out == nil {
		return t.noSessionResult(actionStatus), nil
	}
	return sessionResult(actionStatus, out), nil
}

func (t *browserTool) runStop(ctx agent.Context, session *browserSession) (map[string]any, error) {
	if session.id == "" {
		return t.noSessionResult(actionStop), nil
	}
	out, err := t.stopSession(ctx, session, ctx.FunctionCallID())
	if err != nil {
		return nil, err
	}
	if out == nil {
		return t.noSessionResult(actionStop), nil
	}
	return map[string]any{
		resultKeyStatus:        statusSuccess,
		paramAction:            actionStop,
		resultKeyBrowserID:     aws.ToString(out.BrowserIdentifier),
		resultKeySessionStatus: string(types.BrowserSessionStatusTerminated),
		"last_updated_at":      browsermappers.AgentCoreBrowserTimeValue(out.LastUpdatedAt),
	}, nil
}

func (t *browserTool) noSessionResult(action string) map[string]any {
	return map[string]any{
		resultKeyStatus:        statusSuccess,
		paramAction:            action,
		resultKeyBrowserID:     t.browserIdentifier,
		resultKeySessionStatus: statusNoSession,
	}
}

func (t *browserTool) startSession(
	ctx context.Context,
	toolCtx agent.Context,
	session *browserSession,
) (*bedrockagentcore.StartBrowserSessionOutput, error) {
	callID := toolCtx.FunctionCallID()
	if callID != "" {
		// Keep retries idempotent without reusing the token across conversations.
		scope := sha256.Sum256(fmt.Appendf(nil, "%q %q %q %q",
			toolCtx.AppName(), toolCtx.UserID(), toolCtx.SessionID(), t.browserIdentifier))
		callID = fmt.Sprintf("%s-%x", callID, scope)
	}
	in := browsermappers.AgentCoreBrowserStartInput(browsermappers.AgentCoreBrowserStartParams{
		BrowserIdentifier:     t.browserIdentifier,
		FunctionCallID:        callID,
		SessionTimeoutSeconds: t.sessionTimeoutSeconds,
		ViewportWidth:         t.viewportWidth,
		ViewportHeight:        t.viewportHeight,
	})
	out, err := t.api.StartBrowserSession(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("start browser session: %w", err)
	}
	if out == nil || aws.ToString(out.SessionId) == "" {
		return nil, errors.New("AgentCore did not return a Browser session ID")
	}
	session.id = aws.ToString(out.SessionId)
	session.expiresAt = time.Now().Add(time.Duration(t.sessionTimeoutSeconds) * time.Second)
	return out, nil
}

//nolint:nilnil // An absent session is a normal lookup result.
func (t *browserTool) currentSession(
	ctx context.Context,
	session *browserSession,
) (*bedrockagentcore.GetBrowserSessionOutput, error) {
	if session.id == "" {
		return nil, nil
	}
	out, err := t.api.GetBrowserSession(ctx, browsermappers.AgentCoreBrowserGetInput(t.browserIdentifier, session.id))
	switch {
	case isSessionNotFound(err):
	case err != nil:
		return nil, fmt.Errorf("get browser session: %w", err)
	case out == nil:
		return nil, errors.New("AgentCore did not return a Browser session")
	case out.Status != types.BrowserSessionStatusTerminated:
		return out, nil
	}
	session.id = ""
	return nil, nil
}

func (t *browserTool) stopSession(
	ctx context.Context,
	session *browserSession,
	functionCallID string,
) (*bedrockagentcore.StopBrowserSessionOutput, error) {
	out, err := t.api.StopBrowserSession(ctx, browsermappers.AgentCoreBrowserStopInput(
		t.browserIdentifier,
		session.id,
		functionCallID,
	))
	if err != nil && !isSessionNotFound(err) {
		return nil, fmt.Errorf("stop browser session: %w", err)
	}
	session.id = ""
	return out, nil
}

func (t *browserTool) cleanupStartedSession(ctx agent.Context, session *browserSession, cause error) error {
	cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), t.cleanupTimeout)
	defer cancel()
	if _, err := t.stopSession(cleanupCtx, session, ctx.FunctionCallID()); err != nil {
		return errors.Join(cause, fmt.Errorf("cleanup %w", err))
	}
	return cause
}

func sessionResult(action string, out *bedrockagentcore.GetBrowserSessionOutput) map[string]any {
	return map[string]any{
		resultKeyStatus:           statusSuccess,
		resultKeySessionStatus:    string(out.Status),
		paramAction:               action,
		resultKeyBrowserID:        aws.ToString(out.BrowserIdentifier),
		"name":                    aws.ToString(out.Name),
		"created_at":              browsermappers.AgentCoreBrowserTimeValue(out.CreatedAt),
		"last_updated_at":         browsermappers.AgentCoreBrowserTimeValue(out.LastUpdatedAt),
		"session_timeout_seconds": browsermappers.AgentCoreBrowserInt32Value(out.SessionTimeoutSeconds),
		"session_replay_artifact": aws.ToString(out.SessionReplayArtifact),
		"live_view_url":           browsermappers.AgentCoreBrowserLiveViewEndpoint(out.Streams),
	}
}

func (t *browserTool) sessionStreams(
	ctx context.Context,
	session *browserSession,
) (*types.BrowserSessionStream, error) {
	current, err := t.currentSession(ctx, session)
	if err != nil {
		return nil, err
	}
	if current == nil {
		return nil, errors.New("no active Browser session. Call navigate with a URL first")
	}
	return current.Streams, nil
}

func isSessionNotFound(err error) bool {
	var notFound *types.ResourceNotFoundException
	return errors.As(err, &notFound)
}
