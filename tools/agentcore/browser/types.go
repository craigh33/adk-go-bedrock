package browser

import (
	"context"
	"net/http"
	"net/url"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/coder/websocket"
	"google.golang.org/genai"
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

// Request is a request paused before it is sent by the browser.
// Middleware may modify URL, Method, Headers, or PostData before continuing.
type Request struct {
	URL                 string
	Method              string
	Headers             http.Header
	PostData            []byte
	ResourceType        string
	FrameID             string
	NetworkID           string
	RedirectedRequestID string
}

// Response is a synthetic response returned by request middleware.
type Response struct {
	StatusCode int
	StatusText string
	Headers    http.Header
	Body       []byte
}

// RequestHandler handles a paused browser request. A nil response continues the request;
// a non-nil response fulfills it. Returning an error blocks the request.
type RequestHandler func(context.Context, *Request) (*Response, error)

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

// WebSocketDialer opens an AgentCore Browser automation stream. DialOptions
// contains the SigV4-signed HTTP headers for the handshake.
type WebSocketDialer interface {
	Dial(context.Context, string, *websocket.DialOptions) (*websocket.Conn, *http.Response, error)
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
	Request Request
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
