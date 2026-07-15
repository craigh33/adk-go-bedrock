package agentcorebrowser

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"
)

const (
	ToolName = "agentcore_browser"

	defaultBrowserIdentifier = "aws.browser.v1"
	defaultSessionTimeout    = int32(900)
	defaultNavigationTimeout = 30 * time.Second
	defaultMaxTextBytes      = 64 << 10
	minClientTokenLen        = 33
	maxClientTokenLen        = 256

	paramAction    = "action"
	paramSessionID = "session_id"
	paramURL       = "url"
	paramSelector  = "selector"
	paramFileName  = "file_name"
	paramFormat    = "format"

	actionStart       = "start"
	actionNavigate    = "navigate"
	actionExtractText = "extract_text"
	actionScreenshot  = "screenshot"
	actionStatus      = "status"
	actionStop        = "stop"

	schemaTypeString       = "STRING"
	resultKeyStatus        = actionStatus
	resultKeyBrowserID     = "browser_identifier"
	resultKeySessionStatus = "session_status"
	resultKeyTitle         = "title"
	screenshotFormatPNG    = "png"
	screenshotFormatJPEG   = "jpeg"
	screenshotFormatJPG    = "jpg"

	statusSuccess = "success"
	serviceID     = "bedrock-agentcore"
)

// AgentCoreAPI is the Amazon Bedrock AgentCore subset used by the browser tool.
type AgentCoreAPI interface {
	StartBrowserSession(
		context.Context,
		*bedrockagentcore.StartBrowserSessionInput,
		...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.StartBrowserSessionOutput, error)
	GetBrowserSession(
		context.Context,
		*bedrockagentcore.GetBrowserSessionInput,
		...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.GetBrowserSessionOutput, error)
	StopBrowserSession(
		context.Context,
		*bedrockagentcore.StopBrowserSessionInput,
		...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.StopBrowserSessionOutput, error)
}

// Config configures an AgentCore Browser ADK tool.
type Config struct {
	API         AgentCoreAPI
	Region      string
	Credentials aws.CredentialsProvider

	BrowserIdentifier     string
	SessionTimeoutSeconds int32
	ViewportWidth         int32
	ViewportHeight        int32

	AllowedHosts []string
	DeniedHosts  []string

	NavigationTimeout time.Duration
	MaxTextBytes      int
}

type browserTool struct {
	api                   AgentCoreAPI
	region                string
	credentials           aws.CredentialsProvider
	browserIdentifier     string
	sessionTimeoutSeconds int32
	viewportWidth         int32
	viewportHeight        int32
	allowedHosts          []string
	deniedHosts           []string
	navigationTimeout     time.Duration
	maxTextBytes          int
	decl                  *genai.FunctionDeclaration
}

// New creates an ADK-compatible AgentCore Browser tool.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("agentcorebrowser: API is required")
	}
	region := strings.TrimSpace(cfg.Region)
	if region == "" {
		return nil, errors.New("agentcorebrowser: Region is required")
	}
	if cfg.Credentials == nil {
		return nil, errors.New("agentcorebrowser: Credentials is required")
	}
	browserID := strings.TrimSpace(cfg.BrowserIdentifier)
	if browserID == "" {
		browserID = defaultBrowserIdentifier
	}
	sessionTimeout := cfg.SessionTimeoutSeconds
	if sessionTimeout == 0 {
		sessionTimeout = defaultSessionTimeout
	}
	if sessionTimeout < 0 {
		return nil, errors.New("agentcorebrowser: SessionTimeoutSeconds cannot be negative")
	}
	if (cfg.ViewportWidth == 0) != (cfg.ViewportHeight == 0) {
		return nil, errors.New("agentcorebrowser: ViewportWidth and ViewportHeight must be set together")
	}
	if cfg.ViewportWidth < 0 || cfg.ViewportHeight < 0 {
		return nil, errors.New("agentcorebrowser: viewport dimensions cannot be negative")
	}
	navTimeout := cfg.NavigationTimeout
	if navTimeout == 0 {
		navTimeout = defaultNavigationTimeout
	}
	if navTimeout < 0 {
		return nil, errors.New("agentcorebrowser: NavigationTimeout cannot be negative")
	}
	maxText := cfg.MaxTextBytes
	if maxText == 0 {
		maxText = defaultMaxTextBytes
	}
	if maxText < 0 {
		return nil, errors.New("agentcorebrowser: MaxTextBytes cannot be negative")
	}

	return &browserTool{
		api:                   cfg.API,
		region:                region,
		credentials:           cfg.Credentials,
		browserIdentifier:     browserID,
		sessionTimeoutSeconds: sessionTimeout,
		viewportWidth:         cfg.ViewportWidth,
		viewportHeight:        cfg.ViewportHeight,
		allowedHosts:          normalizeHosts(cfg.AllowedHosts),
		deniedHosts:           normalizeHosts(cfg.DeniedHosts),
		navigationTimeout:     navTimeout,
		maxTextBytes:          maxText,
		decl:                  newFunctionDeclaration(),
	}, nil
}

func (t *browserTool) Name() string { return ToolName }

func (t *browserTool) Description() string {
	return "Controls a constrained Amazon Bedrock AgentCore Browser session: start, navigate, extract visible text, capture screenshots, check status, and stop."
}

func (t *browserTool) IsLongRunning() bool { return false }

func (t *browserTool) Declaration() *genai.FunctionDeclaration { return t.decl }

func newFunctionDeclaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        ToolName,
		Description: "Use Amazon Bedrock AgentCore Browser through constrained actions.",
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				paramAction: {
					Type:   schemaTypeString,
					Format: "enum",
					Enum: []string{
						actionStart,
						actionNavigate,
						actionExtractText,
						actionScreenshot,
						actionStatus,
						actionStop,
					},
					Description: "Browser action to perform.",
				},
				paramSessionID: {
					Type:        schemaTypeString,
					Description: "AgentCore Browser session ID. Required for extract_text, screenshot, status, and stop. Optional for navigate.",
				},
				paramURL: {
					Type:        schemaTypeString,
					Description: "HTTP or HTTPS URL to navigate to. Required for navigate.",
				},
				paramSelector: {
					Type:        schemaTypeString,
					Description: "Optional CSS selector for extract_text. Defaults to the document body.",
				},
				paramFileName: {
					Type:        schemaTypeString,
					Description: "Artifact filename for screenshots. Defaults to browser_screenshot.png or browser_screenshot.jpeg.",
				},
				paramFormat: {
					Type:        schemaTypeString,
					Format:      "enum",
					Enum:        []string{screenshotFormatPNG, screenshotFormatJPEG, screenshotFormatJPG},
					Description: "Screenshot format. Defaults to png.",
				},
			},
			Required: []string{paramAction},
		},
	}
}

func (t *browserTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	name := t.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	decl := t.Declaration()
	if decl == nil {
		return nil
	}

	var funcTool *genai.Tool
	for _, gt := range req.Config.Tools {
		if gt != nil && gt.FunctionDeclarations != nil {
			funcTool = gt
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, decl)
	}
	return nil
}

func (t *browserTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}
	action, _ := m[paramAction].(string)
	switch strings.TrimSpace(action) {
	case actionStart:
		return t.runStart(ctx)
	case actionNavigate:
		return t.runNavigate(ctx, m)
	case actionExtractText:
		return t.runExtractText(ctx, m)
	case actionScreenshot:
		return t.runScreenshot(ctx, m)
	case actionStatus:
		return t.runStatus(ctx, m)
	case actionStop:
		return t.runStop(ctx, m)
	case "":
		return nil, errors.New("action is required")
	default:
		return nil, fmt.Errorf("unsupported action %q", action)
	}
}

func (t *browserTool) runStart(ctx agent.Context) (map[string]any, error) {
	out, err := t.startSession(ctx)
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
	out, err := t.api.GetBrowserSession(ctx, &bedrockagentcore.GetBrowserSessionInput{
		BrowserIdentifier: aws.String(t.browserIdentifier),
		SessionId:         aws.String(sessionID),
	})
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
	out, err := t.stopSession(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("stop browser session %q: %w", sessionID, err)
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionStop,
		resultKeyBrowserID: aws.ToString(out.BrowserIdentifier),
		paramSessionID:     aws.ToString(out.SessionId),
		"last_updated_at":  timeValue(out.LastUpdatedAt),
	}, nil
}

func (t *browserTool) stopSession(
	ctx agent.Context,
	sessionID string,
) (*bedrockagentcore.StopBrowserSessionOutput, error) {
	return t.api.StopBrowserSession(ctx, &bedrockagentcore.StopBrowserSessionInput{
		BrowserIdentifier: aws.String(t.browserIdentifier),
		SessionId:         aws.String(sessionID),
		ClientToken:       aws.String(clientToken(ctx.FunctionCallID())),
	})
}

func (t *browserTool) cleanupStartedSession(ctx agent.Context, sessionID string, cause error) error {
	if _, err := t.stopSession(ctx, sessionID); err != nil {
		return errors.Join(cause, fmt.Errorf("cleanup stop browser session %q: %w", sessionID, err))
	}
	return cause
}

func (t *browserTool) runNavigate(ctx agent.Context, m map[string]any) (map[string]any, error) {
	rawURL, err := requiredString(m, paramURL)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(rawURL); err != nil {
		return nil, err
	}

	sessionID := optionalString(m, paramSessionID)
	autoStarted := false
	var streams *types.BrowserSessionStream
	if sessionID == "" {
		started, err := t.startSession(ctx)
		if err != nil {
			return nil, err
		}
		autoStarted = true
		sessionID = aws.ToString(started.SessionId)
		streams = started.Streams
	} else {
		current, err := t.api.GetBrowserSession(ctx, &bedrockagentcore.GetBrowserSessionInput{
			BrowserIdentifier: aws.String(t.browserIdentifier),
			SessionId:         aws.String(sessionID),
		})
		if err != nil {
			return nil, fmt.Errorf("get browser session %q: %w", sessionID, err)
		}
		streams = current.Streams
	}

	cdp, err := t.openCDP(ctx, automationEndpoint(streams))
	if err != nil {
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}
	defer cdp.close()

	navCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()
	if err := cdp.navigate(navCtx, rawURL); err != nil {
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}
	meta, err := cdp.pageMetadata(navCtx)
	if err != nil {
		if autoStarted {
			return nil, t.cleanupStartedSession(ctx, sessionID, err)
		}
		return nil, err
	}
	if err := t.checkURL(meta.URL); err != nil {
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
	selector := optionalString(m, paramSelector)
	streams, err := t.sessionStreams(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	cdp, err := t.openCDP(ctx, automationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()

	result, err := cdp.extractText(ctx, selector)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(result.URL); err != nil {
		return nil, fmt.Errorf("current url: %w", err)
	}
	result.Text, result.Truncated = truncateUTF8(result.Text, t.maxTextBytes)
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

func (t *browserTool) runScreenshot(ctx agent.Context, m map[string]any) (map[string]any, error) {
	sessionID, err := requiredString(m, paramSessionID)
	if err != nil {
		return nil, err
	}
	format, mimeType, err := screenshotFormat(m)
	if err != nil {
		return nil, err
	}
	fileName := optionalString(m, paramFileName)
	if fileName == "" {
		fileName = "browser_screenshot." + format
	}
	streams, err := t.sessionStreams(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	cdp, err := t.openCDP(ctx, automationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()

	meta, err := cdp.pageMetadata(ctx)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(meta.URL); err != nil {
		return nil, fmt.Errorf("current url: %w", err)
	}
	data, err := cdp.screenshot(ctx, format)
	if err != nil {
		return nil, err
	}
	artifacts := ctx.Artifacts()
	if artifacts == nil {
		return nil, errors.New("agentcorebrowser: artifact service is unavailable")
	}
	saveResp, err := artifacts.Save(ctx, fileName, genai.NewPartFromBytes(data, mimeType))
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
	}, nil
}

func (t *browserTool) startSession(ctx agent.Context) (*bedrockagentcore.StartBrowserSessionOutput, error) {
	token := clientToken(ctx.FunctionCallID())
	in := &bedrockagentcore.StartBrowserSessionInput{
		BrowserIdentifier:     aws.String(t.browserIdentifier),
		ClientToken:           aws.String(token),
		Name:                  aws.String("adk-browser"),
		SessionTimeoutSeconds: aws.Int32(t.sessionTimeoutSeconds),
	}
	if t.viewportWidth > 0 {
		in.ViewPort = &types.ViewPort{
			Width:  aws.Int32(t.viewportWidth),
			Height: aws.Int32(t.viewportHeight),
		}
	}
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
		"created_at":       timeValue(out.CreatedAt),
		"live_view_url":    liveViewEndpoint(out.Streams),
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
		"created_at":              timeValue(out.CreatedAt),
		"last_updated_at":         timeValue(out.LastUpdatedAt),
		"session_timeout_seconds": int32Value(out.SessionTimeoutSeconds),
		"session_replay_artifact": aws.ToString(out.SessionReplayArtifact),
		"live_view_url":           liveViewEndpoint(out.Streams),
	}
}

func (t *browserTool) sessionStreams(ctx context.Context, sessionID string) (*types.BrowserSessionStream, error) {
	current, err := t.api.GetBrowserSession(ctx, &bedrockagentcore.GetBrowserSessionInput{
		BrowserIdentifier: aws.String(t.browserIdentifier),
		SessionId:         aws.String(sessionID),
	})
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
	conn, resp, err := websocket.DefaultDialer.DialContext(ctx, endpoint, headers)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("connect automation stream: %w", err)
	}
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
		signURL.Scheme = "https"
	case "ws":
		signURL.Scheme = "http"
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

func (t *browserTool) checkURL(raw string) error {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return fmt.Errorf("url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("url: scheme must be http or https, got %q", u.Scheme)
	}
	if u.User != nil {
		return errors.New("url: user info is not allowed")
	}
	host := normalizeHost(u.Hostname())
	if host == "" {
		return errors.New("url: host is required")
	}
	if hostMatches(t.deniedHosts, host) {
		return fmt.Errorf("url: host %q is denied", host)
	}
	if len(t.allowedHosts) > 0 && !hostMatches(t.allowedHosts, host) {
		return fmt.Errorf("url: host %q is not allowed", host)
	}
	if len(t.allowedHosts) == 0 && requiresExplicitAllow(host) {
		return fmt.Errorf("url: host %q requires an explicit allowlist entry", host)
	}
	return nil
}

func requiredString(m map[string]any, key string) (string, error) {
	value := optionalString(m, key)
	if value == "" {
		return "", fmt.Errorf("%s is required", key)
	}
	return value, nil
}

func optionalString(m map[string]any, key string) string {
	value, _ := m[key].(string)
	return strings.TrimSpace(value)
}

func screenshotFormat(m map[string]any) (string, string, error) {
	format := optionalString(m, paramFormat)
	switch strings.ToLower(format) {
	case "", screenshotFormatPNG:
		return screenshotFormatPNG, "image/png", nil
	case screenshotFormatJPEG, screenshotFormatJPG:
		return screenshotFormatJPEG, "image/jpeg", nil
	default:
		return "", "", fmt.Errorf("format must be png, jpeg, or jpg, got %q", format)
	}
}

func automationEndpoint(streams *types.BrowserSessionStream) string {
	if streams == nil || streams.AutomationStream == nil {
		return ""
	}
	return aws.ToString(streams.AutomationStream.StreamEndpoint)
}

func liveViewEndpoint(streams *types.BrowserSessionStream) string {
	if streams == nil || streams.LiveViewStream == nil {
		return ""
	}
	return aws.ToString(streams.LiveViewStream.StreamEndpoint)
}

func timeValue(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.Format(time.RFC3339Nano)
}

func int32Value(v *int32) int32 {
	if v == nil {
		return 0
	}
	return *v
}

func clientToken(functionCallID string) string {
	base := sanitizeToken(functionCallID)
	if base == "" {
		return uuid.NewString()
	}
	if len(base) >= minClientTokenLen && len(base) <= maxClientTokenLen {
		return base
	}
	sum := sha256.Sum256([]byte(functionCallID))
	hash := hex.EncodeToString(sum[:])
	maxBase := maxClientTokenLen - len(hash) - 1
	if len(base) > maxBase {
		base = strings.TrimRight(base[:maxBase], "-")
	}
	return base + "-" + hash
}

func sanitizeToken(s string) string {
	var b strings.Builder
	lastHyphen := false
	for _, r := range strings.TrimSpace(s) {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastHyphen = false
			continue
		}
		if b.Len() > 0 && !lastHyphen {
			b.WriteByte('-')
			lastHyphen = true
		}
	}
	return strings.TrimRight(b.String(), "-")
}

func normalizeHosts(hosts []string) []string {
	out := make([]string, 0, len(hosts))
	for _, host := range hosts {
		h := strings.TrimPrefix(strings.TrimSpace(host), "*.")
		h = strings.TrimPrefix(h, ".")
		if h = normalizeHost(h); h != "" {
			out = append(out, h)
		}
	}
	return out
}

func normalizeHost(host string) string {
	return strings.TrimSuffix(strings.ToLower(strings.TrimSpace(host)), ".")
}

func hostMatches(patterns []string, host string) bool {
	for _, pattern := range patterns {
		if host == pattern || strings.HasSuffix(host, "."+pattern) {
			return true
		}
	}
	return false
}

func requiresExplicitAllow(host string) bool {
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	host, _, _ = strings.Cut(host, "%")
	addr, err := netip.ParseAddr(host)
	if err != nil {
		return false
	}
	addr = addr.Unmap()
	if !addr.IsGlobalUnicast() {
		return true
	}
	nonPublicIPPrefixes := [...]netip.Prefix{
		netip.MustParsePrefix("0.0.0.0/8"),
		netip.MustParsePrefix("10.0.0.0/8"),
		netip.MustParsePrefix("100.64.0.0/10"),
		netip.MustParsePrefix("127.0.0.0/8"),
		netip.MustParsePrefix("169.254.0.0/16"),
		netip.MustParsePrefix("172.16.0.0/12"),
		netip.MustParsePrefix("192.0.0.0/24"),
		netip.MustParsePrefix("192.0.2.0/24"),
		netip.MustParsePrefix("192.168.0.0/16"),
		netip.MustParsePrefix("198.18.0.0/15"),
		netip.MustParsePrefix("198.51.100.0/24"),
		netip.MustParsePrefix("203.0.113.0/24"),
		netip.MustParsePrefix("224.0.0.0/4"),
		netip.MustParsePrefix("240.0.0.0/4"),
		netip.MustParsePrefix("::/128"),
		netip.MustParsePrefix("::1/128"),
		netip.MustParsePrefix("64:ff9b:1::/48"),
		netip.MustParsePrefix("100::/64"),
		netip.MustParsePrefix("2001:db8::/32"),
		netip.MustParsePrefix("fc00::/7"),
		netip.MustParsePrefix("fe80::/10"),
		netip.MustParsePrefix("ff00::/8"),
	}
	for _, prefix := range nonPublicIPPrefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

func truncateUTF8(s string, maxBytes int) (string, bool) {
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s, false
	}
	b := []byte(s[:maxBytes])
	for !utf8.Valid(b) && len(b) > 0 {
		b = b[:len(b)-1]
	}
	return string(b), true
}

type cdpConn struct {
	conn          *websocket.Conn
	nextID        int64
	pageSessionID string
	events        []cdpMessage
	responses     map[int64]cdpMessage
}

type cdpMessage struct {
	ID        int64           `json:"id,omitempty"`
	SessionID string          `json:"sessionId,omitempty"`
	Method    string          `json:"method,omitempty"`
	Params    json.RawMessage `json:"params,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     *cdpError       `json:"error,omitempty"`
}

type cdpError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type runtimeEvaluateResult struct {
	Result struct {
		Value json.RawMessage `json:"value"`
	} `json:"result"`
	ExceptionDetails *runtimeExceptionDetails `json:"exceptionDetails,omitempty"`
}

type runtimeExceptionDetails struct {
	Text      string `json:"text,omitempty"`
	Exception *struct {
		Description string `json:"description,omitempty"`
	} `json:"exception,omitempty"`
}

type pageMetadata struct {
	URL   string `json:"url"`
	Title string `json:"title"`
}

type textResult struct {
	Text      string `json:"text"`
	URL       string `json:"url"`
	Title     string `json:"title"`
	Truncated bool   `json:"-"`
}

func (c *cdpConn) close() {
	_ = c.conn.Close()
}

func (c *cdpConn) navigate(ctx context.Context, rawURL string) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	if _, err := c.call(ctx, "Page.enable", nil, sessionID); err != nil {
		return err
	}
	if _, err := c.call(ctx, "Page.navigate", map[string]any{paramURL: rawURL}, sessionID); err != nil {
		return err
	}
	return c.waitEvent(ctx, sessionID, "Page.loadEventFired")
}

func (c *cdpConn) extractText(ctx context.Context, selector string) (*textResult, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	selectorJSON := "null"
	if selector != "" {
		b, _ := json.Marshal(selector)
		selectorJSON = string(b)
	}
	expr := `(() => {
const selector = ` + selectorJSON + `;
const node = selector ? document.querySelector(selector) : document.body;
return {text: node ? (node.innerText || node.textContent || "") : "", url: location.href, title: document.title};
})()`
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		"expression":    expr,
		"returnByValue": true,
	}, sessionID)
	if err != nil {
		return nil, err
	}
	var result textResult
	if err := unmarshalRuntimeValue(raw, "extract_text", &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *cdpConn) pageMetadata(ctx context.Context) (*pageMetadata, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		"expression":    `({url: location.href, title: document.title})`,
		"returnByValue": true,
	}, sessionID)
	if err != nil {
		return nil, err
	}
	var result pageMetadata
	if err := unmarshalRuntimeValue(raw, "page metadata", &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *cdpConn) screenshot(ctx context.Context, format string) ([]byte, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	raw, err := c.call(ctx, "Page.captureScreenshot", map[string]any{
		"format":                format,
		"captureBeyondViewport": true,
	}, sessionID)
	if err != nil {
		return nil, err
	}
	var out struct {
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parse screenshot result: %w", err)
	}
	data, err := base64.StdEncoding.DecodeString(out.Data)
	if err != nil {
		return nil, fmt.Errorf("decode screenshot: %w", err)
	}
	return data, nil
}

func (c *cdpConn) pageSession(ctx context.Context) (string, error) {
	if c.pageSessionID != "" {
		return c.pageSessionID, nil
	}
	raw, err := c.call(ctx, "Target.getTargets", nil, "")
	if err != nil {
		return "", err
	}
	var targets struct {
		TargetInfos []struct {
			TargetID string `json:"targetId"`
			Type     string `json:"type"`
		} `json:"targetInfos"`
	}
	if err := json.Unmarshal(raw, &targets); err != nil {
		return "", fmt.Errorf("parse targets: %w", err)
	}
	var targetID string
	for _, info := range targets.TargetInfos {
		if info.Type == "page" {
			targetID = info.TargetID
			break
		}
	}
	if targetID == "" {
		raw, err = c.call(ctx, "Target.createTarget", map[string]any{paramURL: "about:blank"}, "")
		if err != nil {
			return "", err
		}
		var created struct {
			TargetID string `json:"targetId"`
		}
		if err := json.Unmarshal(raw, &created); err != nil {
			return "", fmt.Errorf("parse created target: %w", err)
		}
		targetID = created.TargetID
	}
	raw, err = c.call(ctx, "Target.attachToTarget", map[string]any{
		"targetId": targetID,
		"flatten":  true,
	}, "")
	if err != nil {
		return "", err
	}
	var attached struct {
		SessionID string `json:"sessionId"`
	}
	if err := json.Unmarshal(raw, &attached); err != nil {
		return "", fmt.Errorf("parse attached target: %w", err)
	}
	if attached.SessionID == "" {
		return "", errors.New("attach target returned empty sessionId")
	}
	c.pageSessionID = attached.SessionID
	return attached.SessionID, nil
}

func unmarshalRuntimeValue(raw json.RawMessage, op string, out any) error {
	var eval runtimeEvaluateResult
	if err := json.Unmarshal(raw, &eval); err != nil {
		return fmt.Errorf("parse %s result: %w", op, err)
	}
	if eval.ExceptionDetails != nil {
		return fmt.Errorf("%s JavaScript exception: %s", op, eval.ExceptionDetails.message())
	}
	if err := json.Unmarshal(eval.Result.Value, out); err != nil {
		return fmt.Errorf("parse %s value: %w", op, err)
	}
	return nil
}

func (d *runtimeExceptionDetails) message() string {
	if d == nil {
		return "unknown exception"
	}
	if d.Exception != nil && strings.TrimSpace(d.Exception.Description) != "" {
		return strings.TrimSpace(d.Exception.Description)
	}
	if strings.TrimSpace(d.Text) != "" {
		return strings.TrimSpace(d.Text)
	}
	return "unknown exception"
}

func (c *cdpConn) call(ctx context.Context, method string, params any, sessionID string) (json.RawMessage, error) {
	c.nextID++
	id := c.nextID
	msg := map[string]any{
		"id":     id,
		"method": method,
	}
	if params != nil {
		msg["params"] = params
	}
	if sessionID != "" {
		msg["sessionId"] = sessionID
	}
	if err := c.conn.SetWriteDeadline(deadline(ctx)); err != nil {
		return nil, err
	}
	if err := c.conn.WriteJSON(msg); err != nil {
		return nil, fmt.Errorf("cdp %s: %w", method, err)
	}
	if resp, ok := c.takeResponse(id); ok {
		return c.callResult(method, resp)
	}
	for {
		resp, err := c.readMessage(ctx)
		if err != nil {
			return nil, fmt.Errorf("cdp %s: %w", method, err)
		}
		if resp.ID != id {
			c.bufferMessage(resp)
			continue
		}
		return c.callResult(method, resp)
	}
}

func (c *cdpConn) waitEvent(ctx context.Context, sessionID, method string) error {
	if c.takeEvent(sessionID, method) {
		return nil
	}
	for {
		msg, err := c.readMessage(ctx)
		if err != nil {
			return fmt.Errorf("wait for %s: %w", method, err)
		}
		if msg.Method == method && (sessionID == "" || msg.SessionID == sessionID) {
			return nil
		}
		c.bufferMessage(msg)
	}
}

func (c *cdpConn) callResult(method string, resp cdpMessage) (json.RawMessage, error) {
	if resp.Error != nil {
		return nil, fmt.Errorf("cdp %s: %s", method, resp.Error.Message)
	}
	return resp.Result, nil
}

func (c *cdpConn) readMessage(ctx context.Context) (cdpMessage, error) {
	if err := c.conn.SetReadDeadline(deadline(ctx)); err != nil {
		return cdpMessage{}, err
	}
	var msg cdpMessage
	if err := c.conn.ReadJSON(&msg); err != nil {
		return cdpMessage{}, err
	}
	return msg, nil
}

func (c *cdpConn) bufferMessage(msg cdpMessage) {
	if msg.ID != 0 {
		if c.responses == nil {
			c.responses = map[int64]cdpMessage{}
		}
		c.responses[msg.ID] = msg
		return
	}
	c.events = append(c.events, msg)
}

func (c *cdpConn) takeResponse(id int64) (cdpMessage, bool) {
	if c.responses == nil {
		return cdpMessage{}, false
	}
	msg, ok := c.responses[id]
	if ok {
		delete(c.responses, id)
	}
	return msg, ok
}

func (c *cdpConn) takeEvent(sessionID, method string) bool {
	for i, msg := range c.events {
		if msg.Method == method && (sessionID == "" || msg.SessionID == sessionID) {
			c.events = append(c.events[:i], c.events[i+1:]...)
			return true
		}
	}
	return false
}

func deadline(ctx context.Context) time.Time {
	if d, ok := ctx.Deadline(); ok {
		return d
	}
	return time.Now().Add(defaultNavigationTimeout)
}
