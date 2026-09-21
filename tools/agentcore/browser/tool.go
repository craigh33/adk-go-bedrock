package browser

import (
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/gorilla/websocket"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	browsermappers "github.com/craigh33/adk-go-bedrock/internal/agentcore/mappers/browser"
)

// New creates an ADK-compatible AgentCore Browser tool.
//
//nolint:funlen,gocognit // Keeping public configuration validation in one constructor makes defaults auditable.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("API is required")
	}
	region := strings.TrimSpace(cfg.Region)
	if region == "" {
		return nil, errors.New("region is required")
	}
	if cfg.Credentials == nil {
		return nil, errors.New("credentials provider is required")
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
		return nil, errors.New("SessionTimeoutSeconds cannot be negative")
	}
	if sessionTimeout > maxSessionTimeout {
		return nil, fmt.Errorf("SessionTimeoutSeconds cannot exceed %d", maxSessionTimeout)
	}
	if (cfg.ViewportWidth == 0) != (cfg.ViewportHeight == 0) {
		return nil, errors.New("ViewportWidth and ViewportHeight must be set together")
	}
	if cfg.ViewportWidth < 0 || cfg.ViewportHeight < 0 {
		return nil, errors.New("viewport dimensions cannot be negative")
	}
	navTimeout := cfg.NavigationTimeout
	if navTimeout == 0 {
		navTimeout = defaultNavigationTimeout
	}
	if navTimeout < 0 {
		return nil, errors.New("NavigationTimeout cannot be negative")
	}
	cleanupTimeout := cfg.CleanupTimeout
	if cleanupTimeout == 0 {
		cleanupTimeout = defaultCleanupTimeout
	}
	if cleanupTimeout < 0 {
		return nil, errors.New("CleanupTimeout cannot be negative")
	}
	maxText := cfg.MaxTextBytes
	if maxText == 0 {
		maxText = defaultMaxTextBytes
	}
	if maxText < 0 {
		return nil, errors.New("MaxTextBytes cannot be negative")
	}
	maxScreenshot := cfg.MaxScreenshotBytes
	if maxScreenshot == 0 {
		maxScreenshot = defaultMaxScreenshotBytes
	}
	if maxScreenshot < 0 {
		return nil, errors.New("MaxScreenshotBytes cannot be negative")
	}
	readLimit, err := browsermappers.AgentCoreBrowserAutomationReadLimit(maxScreenshot, maxText)
	if err != nil {
		return nil, err
	}
	waitUntil, err := normalizeWaitUntil(cfg.WaitUntil)
	if err != nil {
		return nil, err
	}
	allowedHosts, err := browsermappers.AgentCoreBrowserNormalizeHosts("AllowedHosts", cfg.AllowedHosts)
	if err != nil {
		return nil, err
	}
	deniedHosts, err := browsermappers.AgentCoreBrowserNormalizeHosts("DeniedHosts", cfg.DeniedHosts)
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
			return nil, fmt.Errorf("RequestMiddlewares[%d] is nil", i)
		}
		handler = wrap(handler)
		if handler == nil {
			return nil, fmt.Errorf("RequestMiddlewares[%d] returned a nil handler", i)
		}
	}
	return handler, nil
}

func applyURLMiddleware(base URLHandler, middleware []URLMiddleware) (URLHandler, error) {
	handler := base
	for i, wrap := range slices.Backward(middleware) {
		if wrap == nil {
			return nil, fmt.Errorf("URLMiddlewares[%d] is nil", i)
		}
		handler = wrap(handler)
		if handler == nil {
			return nil, fmt.Errorf("URLMiddlewares[%d] returned a nil handler", i)
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

// ProcessRequest registers the browser tool and its declaration with an LLM request.
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
