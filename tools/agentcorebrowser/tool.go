package agentcorebrowser

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/gorilla/websocket"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	bedrockmappers "github.com/craigh33/adk-go-bedrock/internal/mappers"
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
	readLimit, err := bedrockmappers.AgentCoreBrowserAutomationReadLimit(maxScreenshot, maxText)
	if err != nil {
		return nil, err
	}
	waitUntil, err := normalizeWaitUntil(cfg.WaitUntil)
	if err != nil {
		return nil, err
	}
	allowedHosts, err := bedrockmappers.AgentCoreBrowserNormalizeHosts("AllowedHosts", cfg.AllowedHosts)
	if err != nil {
		return nil, err
	}
	deniedHosts, err := bedrockmappers.AgentCoreBrowserNormalizeHosts("DeniedHosts", cfg.DeniedHosts)
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
