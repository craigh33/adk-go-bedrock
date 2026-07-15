package agentcorebrowser

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"net/http"
	"net/netip"
	"net/url"
	"path"
	"slices"
	"sort"
	"strconv"
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

	defaultBrowserIdentifier  = "aws.browser.v1"
	defaultSessionTimeout     = int32(900)
	maxSessionTimeout         = int32(28_800)
	defaultNavigationTimeout  = 30 * time.Second
	defaultCleanupTimeout     = 10 * time.Second
	defaultMaxTextBytes       = 64 << 10
	defaultMaxScreenshotBytes = int64(16 << 20)
	cdpMessageOverhead        = int64(1 << 20)
	base64DecodedBlockBytes   = int64(3)
	base64EncodedBlockBytes   = int64(4)
	minClientTokenLen         = 33
	maxClientTokenLen         = 256

	paramAction          = "action"
	paramSessionID       = "session_id"
	paramURL             = "url"
	paramSelector        = "selector"
	paramFileName        = "file_name"
	paramFormat          = "format"
	paramWaitUntil       = "wait_until"
	paramWaitForSelector = "wait_for_selector"
	paramFullPage        = "full_page"
	paramQuality         = "quality"

	actionStart       = "start"
	actionNavigate    = "navigate"
	actionExtractText = "extract_text"
	actionScreenshot  = "screenshot"
	actionStatus      = "status"
	actionStop        = "stop"

	schemaTypeString       = "STRING"
	schemaTypeBoolean      = "BOOLEAN"
	schemaTypeInteger      = "INTEGER"
	schemaFormatEnum       = "enum"
	resultKeyStatus        = actionStatus
	resultKeyBrowserID     = "browser_identifier"
	resultKeySessionStatus = "session_status"
	resultKeyTitle         = "title"
	screenshotFormatPNG    = "png"
	screenshotFormatJPEG   = "jpeg"
	screenshotFormatJPG    = "jpg"
	mimeTypeJPEG           = "image/jpeg"
	cdpKeyMethod           = "method"
	cdpKeyRequestID        = "requestId"
	cdpKeyExpression       = "expression"
	cdpKeyReturnByValue    = "returnByValue"
	cdpEventRequestPaused  = "Fetch.requestPaused"
	cdpEventAuthRequired   = "Fetch.authRequired"
	schemeHTTP             = "http"
	schemeHTTPS            = "https"
	schemeData             = "data"
	schemeBlob             = "blob"
	cdpAuthDefault         = "Default"
	cdpAuthCancel          = "CancelAuth"
	cdpAuthCredentials     = "ProvideCredentials"

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

// BrowserRequest is a request paused before it is sent by the browser.
// Middleware may modify URL, Method, Headers, or PostData before continuing.
type BrowserRequest struct {
	URL                 string
	Method              string
	Headers             http.Header
	PostData            []byte
	ResourceType        string
	FrameID             string
	NetworkID           string
	RedirectedRequestID string
}

// BrowserResponse is a synthetic response returned by request middleware.
type BrowserResponse struct {
	StatusCode int
	StatusText string
	Headers    http.Header
	Body       []byte
}

// RequestHandler handles a paused browser request. A nil response continues the request;
// a non-nil response fulfills it. Returning an error blocks the request.
type RequestHandler func(context.Context, *BrowserRequest) (*BrowserResponse, error)

// RequestMiddleware wraps request handling. Calling next applies the remaining middleware
// and built-in host policy; middleware may deliberately omit next to replace that behavior.
type RequestMiddleware func(RequestHandler) RequestHandler

// URLStage identifies where a URL was observed in a browser action.
type URLStage string

const (
	URLStageNavigate URLStage = "navigate"
	URLStageRequest  URLStage = "request"
	URLStageCurrent  URLStage = "current"
	URLStageFinal    URLStage = "final"
)

// URLCheck is passed through URL policy middleware.
type URLCheck struct {
	URL   url.URL
	Stage URLStage
}

// URLHandler accepts or rejects a browser URL.
type URLHandler func(context.Context, URLCheck) error

// URLMiddleware wraps URL policy. Calling next applies the remaining middleware
// and built-in host policy; omitting next replaces the host policy.
type URLMiddleware func(URLHandler) URLHandler

// WebSocketDialer opens an AgentCore Browser automation stream.
type WebSocketDialer interface {
	DialContext(context.Context, string, http.Header) (*websocket.Conn, *http.Response, error)
}

// WaitUntil controls which page lifecycle event navigation waits for.
type WaitUntil string

const (
	WaitUntilLoad             WaitUntil = "load"
	WaitUntilDOMContentLoaded WaitUntil = "dom_content_loaded"
	WaitUntilNone             WaitUntil = "none"
)

// AuthAction controls how an HTTP authentication challenge is answered.
type AuthAction string

const (
	AuthActionDefault            AuthAction = "default"
	AuthActionCancel             AuthAction = "cancel"
	AuthActionProvideCredentials AuthAction = "provide_credentials"
)

// AuthChallenge describes an HTTP authentication challenge from the browser.
type AuthChallenge struct {
	Request BrowserRequest
	Source  string
	Origin  string
	Scheme  string
	Realm   string
}

// AuthResponse answers an HTTP authentication challenge.
type AuthResponse struct {
	Action   AuthAction
	Username string
	Password string
}

// AuthHandler handles HTTP authentication challenges.
type AuthHandler func(context.Context, AuthChallenge) (AuthResponse, error)

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

	RequestMiddlewares []RequestMiddleware
	URLMiddlewares     []URLMiddleware
	Dialer             WebSocketDialer
	AuthHandler        AuthHandler

	NavigationTimeout  time.Duration
	CleanupTimeout     time.Duration
	MaxTextBytes       int
	MaxScreenshotBytes int64
	WaitUntil          WaitUntil
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
	requestHandler        RequestHandler
	urlHandler            URLHandler
	dialer                WebSocketDialer
	authHandler           AuthHandler
	navigationTimeout     time.Duration
	cleanupTimeout        time.Duration
	maxTextBytes          int
	maxScreenshotBytes    int64
	automationReadLimit   int64
	waitUntil             WaitUntil
	decl                  *genai.FunctionDeclaration
}

// New creates an ADK-compatible AgentCore Browser tool.
//
//nolint:funlen,gocognit // Keeping public configuration validation in one constructor makes defaults auditable.
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
	if sessionTimeout > maxSessionTimeout {
		return nil, fmt.Errorf("agentcorebrowser: SessionTimeoutSeconds cannot exceed %d", maxSessionTimeout)
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
	cleanupTimeout := cfg.CleanupTimeout
	if cleanupTimeout == 0 {
		cleanupTimeout = defaultCleanupTimeout
	}
	if cleanupTimeout < 0 {
		return nil, errors.New("agentcorebrowser: CleanupTimeout cannot be negative")
	}
	maxText := cfg.MaxTextBytes
	if maxText == 0 {
		maxText = defaultMaxTextBytes
	}
	if maxText < 0 {
		return nil, errors.New("agentcorebrowser: MaxTextBytes cannot be negative")
	}
	maxScreenshot := cfg.MaxScreenshotBytes
	if maxScreenshot == 0 {
		maxScreenshot = defaultMaxScreenshotBytes
	}
	if maxScreenshot < 0 {
		return nil, errors.New("agentcorebrowser: MaxScreenshotBytes cannot be negative")
	}
	readLimit, err := automationReadLimit(maxScreenshot, maxText)
	if err != nil {
		return nil, err
	}
	waitUntil, err := normalizeWaitUntil(cfg.WaitUntil)
	if err != nil {
		return nil, err
	}
	allowedHosts, err := normalizeHosts("AllowedHosts", cfg.AllowedHosts)
	if err != nil {
		return nil, err
	}
	deniedHosts, err := normalizeHosts("DeniedHosts", cfg.DeniedHosts)
	if err != nil {
		return nil, err
	}

	bt := &browserTool{
		api:                   cfg.API,
		region:                region,
		credentials:           cfg.Credentials,
		browserIdentifier:     browserID,
		sessionTimeoutSeconds: sessionTimeout,
		viewportWidth:         cfg.ViewportWidth,
		viewportHeight:        cfg.ViewportHeight,
		allowedHosts:          allowedHosts,
		deniedHosts:           deniedHosts,
		dialer:                cfg.Dialer,
		authHandler:           cfg.AuthHandler,
		navigationTimeout:     navTimeout,
		cleanupTimeout:        cleanupTimeout,
		maxTextBytes:          maxText,
		maxScreenshotBytes:    maxScreenshot,
		automationReadLimit:   readLimit,
		waitUntil:             waitUntil,
		decl:                  newFunctionDeclaration(),
	}
	if bt.dialer == nil {
		bt.dialer = websocket.DefaultDialer
	}
	handler, err := applyRequestMiddleware(bt.handleBrowserRequest, cfg.RequestMiddlewares)
	if err != nil {
		return nil, err
	}
	bt.requestHandler = handler
	urlHandler, err := applyURLMiddleware(bt.handleURLCheck, cfg.URLMiddlewares)
	if err != nil {
		return nil, err
	}
	bt.urlHandler = urlHandler
	return bt, nil
}

func applyRequestMiddleware(base RequestHandler, middleware []RequestMiddleware) (RequestHandler, error) {
	handler := base
	for i, wrap := range slices.Backward(middleware) {
		if wrap == nil {
			return nil, fmt.Errorf("agentcorebrowser: RequestMiddlewares[%d] is nil", i)
		}
		handler = wrap(handler)
		if handler == nil {
			return nil, fmt.Errorf("agentcorebrowser: RequestMiddlewares[%d] returned a nil handler", i)
		}
	}
	return handler, nil
}

func applyURLMiddleware(base URLHandler, middleware []URLMiddleware) (URLHandler, error) {
	handler := base
	for i, wrap := range slices.Backward(middleware) {
		if wrap == nil {
			return nil, fmt.Errorf("agentcorebrowser: URLMiddlewares[%d] is nil", i)
		}
		handler = wrap(handler)
		if handler == nil {
			return nil, fmt.Errorf("agentcorebrowser: URLMiddlewares[%d] returned a nil handler", i)
		}
	}
	return handler, nil
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
					Format: schemaFormatEnum,
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
					Format:      schemaFormatEnum,
					Enum:        []string{screenshotFormatPNG, screenshotFormatJPEG, screenshotFormatJPG},
					Description: "Screenshot format. Defaults to png.",
				},
				paramWaitUntil: {
					Type:   schemaTypeString,
					Format: schemaFormatEnum,
					Enum: []string{
						string(WaitUntilLoad),
						string(WaitUntilDOMContentLoaded),
						string(WaitUntilNone),
					},
					Description: "Optional navigation completion event. Overrides the configured default for navigate.",
				},
				paramWaitForSelector: {
					Type:        schemaTypeString,
					Description: "Optional CSS selector to wait for during navigate or extract_text.",
				},
				paramFullPage: {
					Type:        schemaTypeBoolean,
					Description: "Capture beyond the viewport for screenshots. Defaults to true.",
				},
				paramQuality: {
					Type:        schemaTypeInteger,
					Description: "Optional JPEG quality from 0 through 100.",
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
	action, err := requiredString(m, paramAction)
	if err != nil {
		return nil, err
	}
	switch action {
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
	default:
		return nil, fmt.Errorf("unsupported action %q", action)
	}
}

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
	out, err := t.stopSession(ctx, sessionID, ctx.FunctionCallID())
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
	ctx context.Context,
	sessionID string,
	functionCallID string,
) (*bedrockagentcore.StopBrowserSessionOutput, error) {
	return t.api.StopBrowserSession(ctx, &bedrockagentcore.StopBrowserSessionInput{
		BrowserIdentifier: aws.String(t.browserIdentifier),
		SessionId:         aws.String(sessionID),
		ClientToken:       aws.String(clientToken(functionCallID)),
	})
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
		current, err := t.api.GetBrowserSession(navCtx, &bedrockagentcore.GetBrowserSessionInput{
			BrowserIdentifier: aws.String(t.browserIdentifier),
			SessionId:         aws.String(sessionID),
		})
		if err != nil {
			return nil, fmt.Errorf("get browser session %q: %w", sessionID, err)
		}
		streams = current.Streams
	}

	cdp, err := t.openCDP(navCtx, automationEndpoint(streams))
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
	cdp, err := t.openCDP(actionCtx, automationEndpoint(streams))
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
	text, truncated := truncateUTF8(result.Text, t.maxTextBytes)
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
	format, mimeType, err := screenshotFormat(requestedFormat, fileName)
	if err != nil {
		return nil, err
	}
	if fileName == "" {
		fileName = "browser_screenshot." + format
	}
	if err := validateScreenshotFileName(fileName, format); err != nil {
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
	cdp, err := t.openCDP(actionCtx, automationEndpoint(streams))
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
	token := clientToken(functionCallID)
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
	host := normalizeHost(u.Hostname())
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

func automationReadLimit(maxScreenshot int64, maxText int) (int64, error) {
	const maxInt64 = int64(^uint64(0) >> 1)
	groups := maxScreenshot / base64DecodedBlockBytes
	if groups > maxInt64/base64EncodedBlockBytes {
		return 0, errors.New("agentcorebrowser: MaxScreenshotBytes is too large")
	}
	base64Bytes := groups * base64EncodedBlockBytes
	if maxScreenshot%base64DecodedBlockBytes != 0 {
		if base64Bytes > maxInt64-base64EncodedBlockBytes {
			return 0, errors.New("agentcorebrowser: MaxScreenshotBytes is too large")
		}
		base64Bytes += base64EncodedBlockBytes
	}
	textBytes := int64(maxText)
	if textBytes > maxInt64/6 {
		return 0, errors.New("agentcorebrowser: MaxTextBytes is too large")
	}
	textBytes *= 6
	limit := max(base64Bytes, textBytes)
	if limit > maxInt64-cdpMessageOverhead {
		return 0, errors.New("agentcorebrowser: configured response limits are too large")
	}
	return limit + cdpMessageOverhead, nil
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

func screenshotFormat(format, fileName string) (string, string, error) {
	if format == "" {
		fileName = strings.ToLower(fileName)
		switch {
		case strings.HasSuffix(fileName, ".jpg"), strings.HasSuffix(fileName, ".jpeg"):
			format = screenshotFormatJPEG
		case strings.HasSuffix(fileName, ".png"):
			format = screenshotFormatPNG
		}
	}
	switch strings.ToLower(format) {
	case "", screenshotFormatPNG:
		return screenshotFormatPNG, "image/png", nil
	case screenshotFormatJPEG, screenshotFormatJPG:
		return screenshotFormatJPEG, mimeTypeJPEG, nil
	default:
		return "", "", fmt.Errorf("format must be png, jpeg, or jpg, got %q", format)
	}
}

func validateScreenshotFileName(fileName, format string) error {
	extension := strings.ToLower(path.Ext(fileName))
	switch extension {
	case "":
		return nil
	case ".png":
		if format == screenshotFormatPNG {
			return nil
		}
	case ".jpg", ".jpeg":
		if format == screenshotFormatJPEG {
			return nil
		}
	default:
		return fmt.Errorf("file_name has unsupported extension %q", extension)
	}
	return errors.New("file_name extension does not match format")
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

func normalizeHosts(name string, hosts []string) ([]string, error) {
	out := make([]string, 0, len(hosts))
	for _, raw := range hosts {
		h := strings.TrimPrefix(strings.TrimSpace(raw), "*.")
		h = strings.TrimPrefix(h, ".")
		h = normalizeHost(h)
		if h == "" {
			continue
		}
		if strings.ContainsAny(h, "/?#@ \t\r\n") || strings.Contains(h, "..") {
			return nil, fmt.Errorf("agentcorebrowser: %s contains invalid host %q", name, raw)
		}
		if strings.Contains(h, ":") {
			h = strings.Trim(h, "[]")
			hostWithoutZone, _, _ := strings.Cut(h, "%")
			_, err := netip.ParseAddr(hostWithoutZone)
			if err != nil {
				return nil, fmt.Errorf("agentcorebrowser: %s contains invalid host %q", name, raw)
			}
		}
		out = append(out, h)
	}
	return out, nil
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
		return looksLikeLegacyIPv4(host)
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

func looksLikeLegacyIPv4(host string) bool {
	for part := range strings.SplitSeq(host, ".") {
		if part == "" {
			return false
		}
		digits := part
		base := 10
		if len(part) > 2 && strings.EqualFold(part[:2], "0x") {
			digits = part[2:]
			base = 16
		}
		if _, err := strconv.ParseUint(digits, base, 32); err != nil {
			return false
		}
	}
	return true
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
	conn           *websocket.Conn
	nextID         int64
	pageSessionID  string
	events         []cdpMessage
	responses      map[int64]cdpMessage
	requestHandler RequestHandler
	authHandler    AuthHandler
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
	Truncated bool   `json:"truncated"`
}

type pausedRequest struct {
	RequestID           string `json:"requestId"`
	ResourceType        string `json:"resourceType"`
	FrameID             string `json:"frameId"`
	NetworkID           string `json:"networkId"`
	RedirectedRequestID string `json:"redirectedRequestId"`
	Request             struct {
		URL      string            `json:"url"`
		Method   string            `json:"method"`
		Headers  map[string]string `json:"headers"`
		PostData *string           `json:"postData"`
	} `json:"request"`
}

type authRequired struct {
	RequestID    string `json:"requestId"`
	ResourceType string `json:"resourceType"`
	FrameID      string `json:"frameId"`
	Request      struct {
		URL      string            `json:"url"`
		Method   string            `json:"method"`
		Headers  map[string]string `json:"headers"`
		PostData *string           `json:"postData"`
	} `json:"request"`
	AuthChallenge struct {
		Source string `json:"source"`
		Origin string `json:"origin"`
		Scheme string `json:"scheme"`
		Realm  string `json:"realm"`
	} `json:"authChallenge"`
}

func (c *cdpConn) close() {
	_ = c.conn.Close()
}

func (c *cdpConn) navigate(
	ctx context.Context,
	rawURL string,
	waitUntil WaitUntil,
	handler RequestHandler,
	authHandler AuthHandler,
) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	if err := c.enableRequestInterception(ctx, handler, authHandler); err != nil {
		return err
	}
	if _, err := c.call(ctx, "Page.enable", nil, sessionID); err != nil {
		return err
	}
	raw, err := c.call(ctx, "Page.navigate", map[string]any{paramURL: rawURL}, sessionID)
	if err != nil {
		return err
	}
	var result struct {
		ErrorText  string `json:"errorText"`
		IsDownload bool   `json:"isDownload"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return fmt.Errorf("parse navigation result: %w", err)
	}
	if result.ErrorText != "" {
		return fmt.Errorf("navigate: %s", result.ErrorText)
	}
	if result.IsDownload {
		return errors.New("navigate: URL started a download")
	}
	event := navigationEvent(waitUntil)
	if event == "" {
		return nil
	}
	return c.waitEvent(ctx, sessionID, event)
}

func navigationEvent(waitUntil WaitUntil) string {
	switch waitUntil {
	case WaitUntilNone:
		return ""
	case WaitUntilDOMContentLoaded:
		return "Page.domContentEventFired"
	case WaitUntilLoad:
		return "Page.loadEventFired"
	default:
		return ""
	}
}

func (c *cdpConn) enableRequestInterception(
	ctx context.Context,
	handler RequestHandler,
	authHandler AuthHandler,
) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	c.requestHandler = handler
	c.authHandler = authHandler
	params := map[string]any{
		"patterns": []map[string]any{{
			"urlPattern":   "*",
			"requestStage": "Request",
		}},
	}
	if authHandler != nil {
		params["handleAuthRequests"] = true
	}
	_, err = c.call(ctx, "Fetch.enable", params, sessionID)
	if err != nil {
		c.requestHandler = nil
		c.authHandler = nil
	}
	return err
}

func (c *cdpConn) waitForSelector(ctx context.Context, selector string) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	selectorJSON, _ := json.Marshal(selector)
	expr := `new Promise((resolve, reject) => {
const selector = ` + string(selectorJSON) + `;
let observer;
const find = () => {
    try {
        if (!document.querySelector(selector)) return false;
        if (observer) observer.disconnect();
        resolve(true);
        return true;
    } catch (error) {
        if (observer) observer.disconnect();
        reject(error);
        return true;
    }
};
if (find()) return;
observer = new MutationObserver(find);
observer.observe(document.documentElement || document, {attributes: true, childList: true, subtree: true});
})`
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		cdpKeyExpression:    expr,
		"awaitPromise":      true,
		cdpKeyReturnByValue: true,
	}, sessionID)
	if err != nil {
		return err
	}
	var found bool
	if err := unmarshalRuntimeValue(raw, "wait_for_selector", &found); err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("wait_for_selector %q did not resolve", selector)
	}
	return nil
}

func (c *cdpConn) extractText(ctx context.Context, selector string, maxBytes int) (*textResult, error) {
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
const maxBytes = ` + strconv.Itoa(maxBytes) + `;
const node = selector ? document.querySelector(selector) : document.body;
let text = node ? (node.innerText || node.textContent || "") : "";
const bytes = new TextEncoder().encode(text);
const truncated = maxBytes > 0 && bytes.length > maxBytes;
if (truncated) {
    let end = maxBytes;
    while (end > 0 && (bytes[end] & 0xc0) === 0x80) {
        end--;
    }
    text = new TextDecoder().decode(bytes.subarray(0, end));
}
return {text, url: location.href, title: document.title, truncated};
})()`
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		cdpKeyExpression:    expr,
		cdpKeyReturnByValue: true,
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
		cdpKeyExpression:    `({url: location.href, title: document.title})`,
		cdpKeyReturnByValue: true,
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

func (c *cdpConn) screenshot(
	ctx context.Context,
	format string,
	fullPage bool,
	quality *int,
	maxBytes int64,
) ([]byte, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	params := map[string]any{
		"format":                format,
		"captureBeyondViewport": fullPage,
	}
	if quality != nil {
		params[paramQuality] = *quality
	}
	raw, err := c.call(ctx, "Page.captureScreenshot", params, sessionID)
	if err != nil {
		return nil, err
	}
	var out struct {
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parse screenshot result: %w", err)
	}
	decodedLen := base64DecodedLen(out.Data)
	if decodedLen > maxBytes {
		return nil, fmt.Errorf("screenshot exceeds MaxScreenshotBytes (%d)", maxBytes)
	}
	data, err := base64.StdEncoding.DecodeString(out.Data)
	if err != nil {
		return nil, fmt.Errorf("decode screenshot: %w", err)
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("screenshot exceeds MaxScreenshotBytes (%d)", maxBytes)
	}
	return data, nil
}

func base64DecodedLen(encoded string) int64 {
	decodedLen := base64.StdEncoding.DecodedLen(len(encoded))
	if strings.HasSuffix(encoded, "=") {
		decodedLen--
	}
	if strings.HasSuffix(encoded, "==") {
		decodedLen--
	}
	return int64(max(decodedLen, 0))
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
		"id":         id,
		cdpKeyMethod: method,
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
	for {
		if resp, ok := c.takeResponse(id); ok {
			return c.callResult(method, resp)
		}
		resp, err := c.readMessage(ctx)
		if err != nil {
			return nil, fmt.Errorf("cdp %s: %w", method, err)
		}
		if handled, err := c.handleInterceptedEvent(ctx, resp); handled {
			if err != nil {
				return nil, err
			}
			continue
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
		if handled, err := c.handleInterceptedEvent(ctx, msg); handled {
			if err != nil {
				return err
			}
			continue
		}
		if msg.Method == method && (sessionID == "" || msg.SessionID == sessionID) {
			return nil
		}
		c.bufferMessage(msg)
	}
}

func (c *cdpConn) handleInterceptedEvent(ctx context.Context, msg cdpMessage) (bool, error) {
	switch msg.Method {
	case cdpEventRequestPaused:
		return c.handlePausedRequest(ctx, msg)
	case cdpEventAuthRequired:
		return c.handleAuthRequired(ctx, msg)
	default:
		return false, nil
	}
}

func (c *cdpConn) handlePausedRequest(ctx context.Context, msg cdpMessage) (bool, error) {
	if c.requestHandler == nil {
		return false, nil
	}
	var paused pausedRequest
	if err := json.Unmarshal(msg.Params, &paused); err != nil {
		return true, fmt.Errorf("parse paused browser request: %w", err)
	}
	sessionID := msg.SessionID
	if sessionID == "" {
		sessionID = c.pageSessionID
	}
	if paused.RequestID == "" {
		return true, errors.New("paused browser request has no requestId")
	}
	request := newBrowserRequest(paused)
	original := cloneBrowserRequest(request)
	response, err := c.requestHandler(ctx, request)
	if err != nil {
		return true, c.failPausedRequest(ctx, sessionID, paused.RequestID, err)
	}
	if response != nil {
		return true, c.fulfillPausedRequest(ctx, sessionID, paused.RequestID, response)
	}
	return true, c.continuePausedRequest(ctx, sessionID, paused.RequestID, original, request)
}

func (c *cdpConn) handleAuthRequired(ctx context.Context, msg cdpMessage) (bool, error) {
	if c.authHandler == nil {
		return false, nil
	}
	var required authRequired
	if err := json.Unmarshal(msg.Params, &required); err != nil {
		return true, fmt.Errorf("parse browser authentication challenge: %w", err)
	}
	if required.RequestID == "" {
		return true, errors.New("browser authentication challenge has no requestId")
	}
	sessionID := msg.SessionID
	if sessionID == "" {
		sessionID = c.pageSessionID
	}
	challenge := AuthChallenge{
		Request: BrowserRequest{
			URL:          required.Request.URL,
			Method:       required.Request.Method,
			Headers:      stringMapHeader(required.Request.Headers),
			ResourceType: required.ResourceType,
			FrameID:      required.FrameID,
		},
		Source: required.AuthChallenge.Source,
		Origin: required.AuthChallenge.Origin,
		Scheme: required.AuthChallenge.Scheme,
		Realm:  required.AuthChallenge.Realm,
	}
	if required.Request.PostData != nil {
		challenge.Request.PostData = []byte(*required.Request.PostData)
	}
	response, err := c.authHandler(ctx, challenge)
	if err != nil {
		return true, c.cancelAuthChallenge(
			ctx,
			sessionID,
			required.RequestID,
			fmt.Errorf("authentication handler: %w", err),
		)
	}
	response, err = normalizeAuthResponse(response)
	if err != nil {
		return true, c.cancelAuthChallenge(ctx, sessionID, required.RequestID, err)
	}
	return true, c.continueWithAuth(ctx, sessionID, required.RequestID, response)
}

func stringMapHeader(values map[string]string) http.Header {
	headers := make(http.Header, len(values))
	for name, value := range values {
		headers.Set(name, value)
	}
	return headers
}

func normalizeAuthResponse(response AuthResponse) (AuthResponse, error) {
	if response.Action == "" {
		response.Action = AuthActionDefault
	}
	switch response.Action {
	case AuthActionDefault, AuthActionCancel:
		if response.Username != "" || response.Password != "" {
			return AuthResponse{}, fmt.Errorf(
				"authentication action %q cannot include credentials",
				response.Action,
			)
		}
	case AuthActionProvideCredentials:
	default:
		return AuthResponse{}, fmt.Errorf("invalid authentication action %q", response.Action)
	}
	return response, nil
}

func (c *cdpConn) cancelAuthChallenge(
	ctx context.Context,
	sessionID string,
	requestID string,
	cause error,
) error {
	cancelErr := c.continueWithAuth(ctx, sessionID, requestID, AuthResponse{Action: AuthActionCancel})
	if cancelErr != nil {
		return errors.Join(cause, cancelErr)
	}
	return cause
}

func (c *cdpConn) continueWithAuth(
	ctx context.Context,
	sessionID string,
	requestID string,
	response AuthResponse,
) error {
	protocolAction := cdpAuthDefault
	switch response.Action {
	case AuthActionDefault:
	case AuthActionCancel:
		protocolAction = cdpAuthCancel
	case AuthActionProvideCredentials:
		protocolAction = cdpAuthCredentials
	default:
		return fmt.Errorf("invalid authentication action %q", response.Action)
	}
	authResponse := map[string]any{"response": protocolAction}
	if response.Action == AuthActionProvideCredentials {
		authResponse["username"] = response.Username
		authResponse["password"] = response.Password
	}
	_, err := c.call(ctx, "Fetch.continueWithAuth", map[string]any{
		cdpKeyRequestID:         requestID,
		"authChallengeResponse": authResponse,
	}, sessionID)
	return err
}

func newBrowserRequest(paused pausedRequest) *BrowserRequest {
	headers := make(http.Header, len(paused.Request.Headers))
	for name, value := range paused.Request.Headers {
		headers.Set(name, value)
	}
	var postData []byte
	if paused.Request.PostData != nil {
		postData = []byte(*paused.Request.PostData)
	}
	return &BrowserRequest{
		URL:                 paused.Request.URL,
		Method:              paused.Request.Method,
		Headers:             headers,
		PostData:            postData,
		ResourceType:        paused.ResourceType,
		FrameID:             paused.FrameID,
		NetworkID:           paused.NetworkID,
		RedirectedRequestID: paused.RedirectedRequestID,
	}
}

func cloneBrowserRequest(request *BrowserRequest) *BrowserRequest {
	cloned := *request
	cloned.Headers = request.Headers.Clone()
	cloned.PostData = slices.Clone(request.PostData)
	return &cloned
}

func (c *cdpConn) continuePausedRequest(
	ctx context.Context,
	sessionID string,
	requestID string,
	original *BrowserRequest,
	request *BrowserRequest,
) error {
	params := map[string]any{cdpKeyRequestID: requestID}
	if request.URL != original.URL {
		params[paramURL] = request.URL
	}
	if request.Method != original.Method {
		params["method"] = request.Method
	}
	if !headersEqual(request.Headers, original.Headers) {
		params["headers"] = headerEntries(request.Headers)
	}
	if (request.PostData == nil) != (original.PostData == nil) ||
		!bytes.Equal(request.PostData, original.PostData) {
		// CDP defines Fetch.continueRequest.postData as base64 when passed over JSON.
		params["postData"] = base64.StdEncoding.EncodeToString(request.PostData)
	}
	_, err := c.call(ctx, "Fetch.continueRequest", params, sessionID)
	return err
}

func (c *cdpConn) fulfillPausedRequest(
	ctx context.Context,
	sessionID string,
	requestID string,
	response *BrowserResponse,
) error {
	statusCode := response.StatusCode
	if statusCode == 0 {
		statusCode = http.StatusOK
	}
	if statusCode < 100 || statusCode > 599 {
		return c.failPausedRequest(ctx, sessionID, requestID, fmt.Errorf(
			"request middleware returned invalid response status %d",
			statusCode,
		))
	}
	params := map[string]any{
		cdpKeyRequestID: requestID,
		"responseCode":  statusCode,
	}
	if response.StatusText != "" {
		params["responsePhrase"] = response.StatusText
	}
	if response.Headers != nil {
		params["responseHeaders"] = headerEntries(response.Headers)
	}
	if response.Body != nil {
		params["body"] = base64.StdEncoding.EncodeToString(response.Body)
	}
	_, err := c.call(ctx, "Fetch.fulfillRequest", params, sessionID)
	return err
}

func (c *cdpConn) failPausedRequest(
	ctx context.Context,
	sessionID string,
	requestID string,
	cause error,
) error {
	_, failErr := c.call(ctx, "Fetch.failRequest", map[string]any{
		cdpKeyRequestID: requestID,
		"errorReason":   "BlockedByClient",
	}, sessionID)
	requestErr := fmt.Errorf("browser request: %w", cause)
	if failErr != nil {
		return errors.Join(requestErr, failErr)
	}
	return requestErr
}

func headersEqual(a, b http.Header) bool {
	return maps.EqualFunc(a, b, slices.Equal)
}

func headerEntries(headers http.Header) []map[string]string {
	names := make([]string, 0, len(headers))
	for name := range headers {
		names = append(names, name)
	}
	sort.Strings(names)
	entries := make([]map[string]string, 0, len(headers))
	for _, name := range names {
		for _, value := range headers[name] {
			entries = append(entries, map[string]string{"name": name, "value": value})
		}
	}
	return entries
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
