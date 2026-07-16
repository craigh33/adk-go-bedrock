package agentcorebrowser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/genai"

	bedrockmappers "github.com/craigh33/adk-go-bedrock/internal/mappers"
)

func (t *browserTool) runStart(ctx agent.Context) (map[string]any, error) {
	out, err := t.startSession(ctx, ctx.FunctionCallID())
	if err != nil {
		return nil, err
	}
	return t.startResult(out), nil
}

func (t *browserTool) runStatus(ctx agent.Context, m map[string]any) (map[string]any, error) {
	sessionID, err := requiredString(m, paramSessionID)
	if err != nil {
		return nil, err
	}
	out, err := t.api.GetBrowserSession(ctx, bedrockmappers.AgentCoreBrowserGetInput(t.browserIdentifier, sessionID))
	if err != nil {
		return nil, fmt.Errorf("get browser session %q: %w", sessionID, err)
	}
	return t.sessionResult(out), nil
}

func (t *browserTool) runStop(ctx agent.Context, m map[string]any) (map[string]any, error) {
	sessionID, err := requiredString(m, paramSessionID)
	if err != nil {
		return nil, err
	}
	out, err := t.stopSession(ctx, sessionID, ctx.FunctionCallID())
	if err != nil {
		return nil, fmt.Errorf("stop browser session %q: %w", sessionID, err)
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionStop,
		resultKeyBrowserID: aws.ToString(out.BrowserIdentifier),
		paramSessionID:     aws.ToString(out.SessionId),
		"last_updated_at":  bedrockmappers.AgentCoreBrowserTimeValue(out.LastUpdatedAt),
	}, nil
}

func (t *browserTool) stopSession(
	ctx context.Context,
	sessionID string,
	functionCallID string,
) (*bedrockagentcore.StopBrowserSessionOutput, error) {
	return t.api.StopBrowserSession(ctx, bedrockmappers.AgentCoreBrowserStopInput(
		t.browserIdentifier,
		sessionID,
		functionCallID,
	))
}

func (t *browserTool) cleanupStartedSession(ctx agent.Context, sessionID string, cause error) error {
	cleanupCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), t.cleanupTimeout)
	defer cancel()
	if _, err := t.stopSession(cleanupCtx, sessionID, ctx.FunctionCallID()); err != nil {
		return errors.Join(cause, fmt.Errorf("cleanup stop browser session %q: %w", sessionID, err))
	}
	return cause
}

//nolint:funlen,gocognit // Navigation keeps ownership-aware cleanup beside each failure point.
func (t *browserTool) runNavigate(ctx agent.Context, m map[string]any) (map[string]any, error) {
	rawURL, err := requiredString(m, paramURL)
	if err != nil {
		return nil, err
	}
	sessionID, err := optionalStringArg(m, paramSessionID)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(ctx, rawURL, URLStageNavigate); err != nil {
		return nil, err
	}
	waitUntil := t.waitUntil
	rawWaitUntil, err := optionalStringArg(m, paramWaitUntil)
	if err != nil {
		return nil, err
	}
	if rawWaitUntil != "" {
		waitUntil, err = normalizeWaitUntil(WaitUntil(rawWaitUntil))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", paramWaitUntil, err)
		}
	}
	waitForSelector, err := optionalStringArg(m, paramWaitForSelector)
	if err != nil {
		return nil, err
	}
	navCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()

	autoStarted := false
	var streams *types.BrowserSessionStream
	if sessionID == "" {
		started, err := t.startSession(navCtx, ctx.FunctionCallID())
		if err != nil {
			return nil, err
		}
		autoStarted = true
		sessionID = aws.ToString(started.SessionId)
		streams = started.Streams
	} else {
		current, err := t.api.GetBrowserSession(
			navCtx,
			bedrockmappers.AgentCoreBrowserGetInput(t.browserIdentifier, sessionID),
		)
		if err != nil {
			return nil, fmt.Errorf("get browser session %q: %w", sessionID, err)
		}
		streams = current.Streams
	}

	cdp, err := t.openCDP(navCtx, bedrockmappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}
	defer cdp.close()

	if err := cdp.navigate(navCtx, rawURL, waitUntil, t.requestHandler, t.authHandler); err != nil {
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}
	if waitForSelector != "" {
		if err := cdp.waitForSelector(navCtx, waitForSelector); err != nil {
			if autoStarted {
				return nil, t.cleanupStartedSession(ctx, sessionID, err)
			}
			return nil, err
		}
	}
	meta, err := cdp.pageMetadata(navCtx)
	if err != nil {
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}
	if err := t.checkURL(navCtx, meta.URL, URLStageFinal); err != nil {
		err = fmt.Errorf("final url: %w", err)
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}

	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionNavigate,
		resultKeyBrowserID: t.browserIdentifier,
		paramSessionID:     sessionID,
		paramURL:           meta.URL,
		resultKeyTitle:     meta.Title,
	}, nil
}

func (t *browserTool) runExtractText(ctx agent.Context, m map[string]any) (map[string]any, error) {
	sessionID, err := requiredString(m, paramSessionID)
	if err != nil {
		return nil, err
	}
	selector, err := optionalStringArg(m, paramSelector)
	if err != nil {
		return nil, err
	}
	waitForSelector, err := optionalStringArg(m, paramWaitForSelector)
	if err != nil {
		return nil, err
	}
	if selector == "" {
		selector = waitForSelector
	}
	actionCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()
	streams, err := t.sessionStreams(actionCtx, sessionID)
	if err != nil {
		return nil, err
	}
	cdp, err := t.openCDP(actionCtx, bedrockmappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()
	if err := cdp.enableRequestInterception(actionCtx, t.requestHandler, t.authHandler); err != nil {
		return nil, err
	}
	if waitForSelector != "" {
		if err := cdp.waitForSelector(actionCtx, waitForSelector); err != nil {
			return nil, err
		}
	}

	result, err := cdp.extractText(actionCtx, selector, t.maxTextBytes)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(actionCtx, result.URL, URLStageCurrent); err != nil {
		return nil, fmt.Errorf("current url: %w", err)
	}
	text, truncated := bedrockmappers.AgentCoreBrowserTruncateUTF8(result.Text, t.maxTextBytes)
	result.Text = text
	result.Truncated = result.Truncated || truncated
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionExtractText,
		resultKeyBrowserID: t.browserIdentifier,
		paramSessionID:     sessionID,
		"text":             result.Text,
		paramURL:           result.URL,
		resultKeyTitle:     result.Title,
		"truncated":        result.Truncated,
	}, nil
}

//nolint:funlen,gocognit // Validate every model-facing option before touching the session or artifact service.
func (t *browserTool) runScreenshot(ctx agent.Context, m map[string]any) (map[string]any, error) {
	sessionID, err := requiredString(m, paramSessionID)
	if err != nil {
		return nil, err
	}
	fileName, err := optionalStringArg(m, paramFileName)
	if err != nil {
		return nil, err
	}
	if strings.ContainsAny(fileName, `/\`) {
		return nil, errors.New("file_name cannot contain path separators")
	}
	requestedFormat, err := optionalStringArg(m, paramFormat)
	if err != nil {
		return nil, err
	}
	format, mimeType, err := bedrockmappers.AgentCoreBrowserScreenshotFormat(requestedFormat, fileName)
	if err != nil {
		return nil, err
	}
	if fileName == "" {
		fileName = "browser_screenshot." + format
	}
	if err := bedrockmappers.AgentCoreBrowserValidateScreenshotFileName(fileName, format); err != nil {
		return nil, err
	}
	fullPage, err := optionalBool(m, paramFullPage, true)
	if err != nil {
		return nil, err
	}
	quality, err := optionalInt(m, paramQuality)
	if err != nil {
		return nil, err
	}
	if quality != nil {
		if *quality < 0 || *quality > 100 {
			return nil, errors.New("quality must be between 0 and 100")
		}
		if format == screenshotFormatPNG {
			return nil, errors.New("quality is only supported for JPEG screenshots")
		}
	}
	artifacts := ctx.Artifacts()
	if artifacts == nil {
		return nil, errors.New("agentcorebrowser: artifact service is unavailable")
	}
	actionCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()
	streams, err := t.sessionStreams(actionCtx, sessionID)
	if err != nil {
		return nil, err
	}
	cdp, err := t.openCDP(actionCtx, bedrockmappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()
	if err := cdp.enableRequestInterception(actionCtx, t.requestHandler, t.authHandler); err != nil {
		return nil, err
	}

	meta, err := cdp.pageMetadata(actionCtx)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(actionCtx, meta.URL, URLStageCurrent); err != nil {
		return nil, fmt.Errorf("current url: %w", err)
	}
	data, err := cdp.screenshot(actionCtx, format, fullPage, quality, t.maxScreenshotBytes)
	if err != nil {
		return nil, err
	}
	meta, err = cdp.pageMetadata(actionCtx)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(actionCtx, meta.URL, URLStageCurrent); err != nil {
		return nil, fmt.Errorf("current url after capture: %w", err)
	}
	if int64(len(data)) > t.maxScreenshotBytes {
		return nil, fmt.Errorf("screenshot exceeds MaxScreenshotBytes (%d)", t.maxScreenshotBytes)
	}
	saveResp, err := artifacts.Save(actionCtx, fileName, genai.NewPartFromBytes(data, mimeType))
	if err != nil {
		return nil, fmt.Errorf("save artifact %q: %w", fileName, err)
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionScreenshot,
		resultKeyBrowserID: t.browserIdentifier,
		paramSessionID:     sessionID,
		paramFileName:      fileName,
		"version":          saveResp.Version,
		"mime_type":        mimeType,
		paramURL:           meta.URL,
		resultKeyTitle:     meta.Title,
	}, nil
}

func (t *browserTool) startSession(
	ctx context.Context,
	functionCallID string,
) (*bedrockagentcore.StartBrowserSessionOutput, error) {
	in := bedrockmappers.AgentCoreBrowserStartInput(bedrockmappers.AgentCoreBrowserStartParams{
		BrowserIdentifier:     t.browserIdentifier,
		FunctionCallID:        functionCallID,
		SessionTimeoutSeconds: t.sessionTimeoutSeconds,
		ViewportWidth:         t.viewportWidth,
		ViewportHeight:        t.viewportHeight,
	})
	out, err := t.api.StartBrowserSession(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("start browser session: %w", err)
	}
	return out, nil
}

func (t *browserTool) startResult(out *bedrockagentcore.StartBrowserSessionOutput) map[string]any {
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionStart,
		resultKeyBrowserID: aws.ToString(out.BrowserIdentifier),
		paramSessionID:     aws.ToString(out.SessionId),
		"created_at":       bedrockmappers.AgentCoreBrowserTimeValue(out.CreatedAt),
		"live_view_url":    bedrockmappers.AgentCoreBrowserLiveViewEndpoint(out.Streams),
	}
}

func (t *browserTool) sessionResult(out *bedrockagentcore.GetBrowserSessionOutput) map[string]any {
	return map[string]any{
		resultKeyStatus:           statusSuccess,
		resultKeySessionStatus:    string(out.Status),
		paramAction:               actionStatus,
		resultKeyBrowserID:        aws.ToString(out.BrowserIdentifier),
		paramSessionID:            aws.ToString(out.SessionId),
		"name":                    aws.ToString(out.Name),
		"created_at":              bedrockmappers.AgentCoreBrowserTimeValue(out.CreatedAt),
		"last_updated_at":         bedrockmappers.AgentCoreBrowserTimeValue(out.LastUpdatedAt),
		"session_timeout_seconds": bedrockmappers.AgentCoreBrowserInt32Value(out.SessionTimeoutSeconds),
		"session_replay_artifact": aws.ToString(out.SessionReplayArtifact),
		"live_view_url":           bedrockmappers.AgentCoreBrowserLiveViewEndpoint(out.Streams),
	}
}

func (t *browserTool) sessionStreams(ctx context.Context, sessionID string) (*types.BrowserSessionStream, error) {
	current, err := t.api.GetBrowserSession(
		ctx,
		bedrockmappers.AgentCoreBrowserGetInput(t.browserIdentifier, sessionID),
	)
	if err != nil {
		return nil, fmt.Errorf("get browser session %q: %w", sessionID, err)
	}
	return current.Streams, nil
}

func (t *browserTool) openCDP(ctx context.Context, endpoint string) (*cdpConn, error) {
	if endpoint == "" {
		return nil, errors.New("agentcorebrowser: browser session has no automation stream endpoint")
	}
	headers, err := t.signedWebSocketHeaders(ctx, endpoint)
	if err != nil {
		return nil, err
	}
	conn, resp, err := t.dialer.DialContext(ctx, endpoint, headers)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("connect automation stream: %w", err)
	}
	if conn == nil {
		return nil, errors.New("connect automation stream: dialer returned a nil connection")
	}
	conn.SetReadLimit(t.automationReadLimit)
	return &cdpConn{conn: conn}, nil
}

func (t *browserTool) signedWebSocketHeaders(ctx context.Context, endpoint string) (http.Header, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("parse automation stream endpoint: %w", err)
	}
	signURL := *u
	switch signURL.Scheme {
	case "wss":
		signURL.Scheme = schemeHTTPS
	case "ws":
		signURL.Scheme = schemeHTTP
	default:
		return nil, fmt.Errorf("automation stream endpoint scheme must be ws or wss, got %q", signURL.Scheme)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, signURL.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("build automation stream request: %w", err)
	}
	creds, err := t.credentials.Retrieve(ctx)
	if err != nil {
		return nil, fmt.Errorf("retrieve AWS credentials: %w", err)
	}
	if err := v4.NewSigner().
		SignHTTP(ctx, creds, req, "UNSIGNED-PAYLOAD", serviceID, t.region, time.Now()); err != nil {
		return nil, fmt.Errorf("sign automation stream request: %w", err)
	}
	return req.Header, nil
}

func (t *browserTool) checkURL(ctx context.Context, raw string, stage URLStage) error {
	check, err := parseURLCheck(raw, stage)
	if err != nil {
		return err
	}
	return t.urlHandler(ctx, check)
}

func parseURLCheck(raw string, stage URLStage) (URLCheck, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return URLCheck{}, fmt.Errorf("url: %w", err)
	}
	check := URLCheck{URL: *u, Stage: stage}
	if err := validateURLStructure(check); err != nil {
		return URLCheck{}, err
	}
	return check, nil
}

func validateURLStructure(check URLCheck) error {
	u := &check.URL
	if u.User != nil {
		return errors.New("url: user info is not allowed")
	}
	switch check.Stage {
	case URLStageNavigate, URLStageCurrent, URLStageFinal:
		if u.Scheme != schemeHTTP && u.Scheme != schemeHTTPS {
			return fmt.Errorf("url: scheme must be http or https, got %q", u.Scheme)
		}
	case URLStageRequest:
		switch u.Scheme {
		case schemeHTTP, schemeHTTPS:
		case schemeData, schemeBlob:
			return nil
		default:
			return fmt.Errorf("url: scheme %q is not allowed", u.Scheme)
		}
	default:
		return fmt.Errorf("url: invalid stage %q", check.Stage)
	}
	if u.Hostname() == "" {
		return errors.New("url: host is required")
	}
	return nil
}

func (t *browserTool) handleURLCheck(_ context.Context, check URLCheck) error {
	if err := validateURLStructure(check); err != nil {
		return err
	}
	u := &check.URL
	if check.Stage == URLStageRequest && (u.Scheme == schemeData || u.Scheme == schemeBlob) {
		return nil
	}
	host := bedrockmappers.AgentCoreBrowserNormalizeHost(u.Hostname())
	if bedrockmappers.AgentCoreBrowserHostMatches(t.deniedHosts, host) {
		return fmt.Errorf("url: host %q is denied", host)
	}
	if len(t.allowedHosts) > 0 && !bedrockmappers.AgentCoreBrowserHostMatches(t.allowedHosts, host) {
		return fmt.Errorf("url: host %q is not allowed", host)
	}
	if len(t.allowedHosts) == 0 && bedrockmappers.AgentCoreBrowserRequiresExplicitAllow(host) {
		return fmt.Errorf("url: host %q requires an explicit allowlist entry", host)
	}
	return nil
}

func (t *browserTool) handleBrowserRequest(
	ctx context.Context,
	request *BrowserRequest,
) (*BrowserResponse, error) {
	if err := t.checkURL(ctx, request.URL, URLStageRequest); err != nil {
		return nil, err
	}
	return nil, nil //nolint:nilnil // A nil response and error means continue unchanged.
}

func normalizeWaitUntil(value WaitUntil) (WaitUntil, error) {
	switch value {
	case "", WaitUntilLoad:
		return WaitUntilLoad, nil
	case WaitUntilDOMContentLoaded, WaitUntilNone:
		return value, nil
	default:
		return "", fmt.Errorf("agentcorebrowser: WaitUntil must be load, dom_content_loaded, or none, got %q", value)
	}
}

func requiredString(m map[string]any, key string) (string, error) {
	value, err := optionalStringArg(m, key)
	if err != nil {
		return "", err
	}
	if value == "" {
		return "", fmt.Errorf("%s is required", key)
	}
	return value, nil
}

func optionalStringArg(m map[string]any, key string) (string, error) {
	value, ok := m[key]
	if !ok {
		return "", nil
	}
	result, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", key)
	}
	return strings.TrimSpace(result), nil
}

func optionalBool(m map[string]any, key string, defaultValue bool) (bool, error) {
	value, ok := m[key]
	if !ok {
		return defaultValue, nil
	}
	result, ok := value.(bool)
	if !ok {
		return false, fmt.Errorf("%s must be a boolean", key)
	}
	return result, nil
}

func optionalInt(m map[string]any, key string) (*int, error) {
	value, ok := m[key]
	if !ok {
		return nil, nil //nolint:nilnil // A nil pointer represents an omitted optional argument.
	}
	var result int64
	switch value := value.(type) {
	case int:
		result = int64(value)
	case int8:
		result = int64(value)
	case int16:
		result = int64(value)
	case int32:
		result = int64(value)
	case int64:
		result = value
	case uint:
		if uint64(value) > math.MaxInt64 {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = int64(value)
	case uint8:
		result = int64(value)
	case uint16:
		result = int64(value)
	case uint32:
		result = int64(value)
	case uint64:
		if value > math.MaxInt64 {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = int64(value)
	case float32:
		parsed, valid := integralInt64(float64(value))
		if !valid {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = parsed
	case float64:
		parsed, valid := integralInt64(value)
		if !valid {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = parsed
	case json.Number:
		parsed, err := value.Int64()
		if err != nil {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = parsed
	default:
		return nil, fmt.Errorf("%s must be an integer", key)
	}
	converted := int(result)
	if int64(converted) != result {
		return nil, fmt.Errorf("%s must be an integer", key)
	}
	return &converted, nil
}

func integralInt64(value float64) (int64, bool) {
	const int64UpperBound = float64(1 << 63)
	if math.IsNaN(value) || math.IsInf(value, 0) || math.Trunc(value) != value ||
		value < -int64UpperBound || value >= int64UpperBound {
		return 0, false
	}
	return int64(value), true
}
