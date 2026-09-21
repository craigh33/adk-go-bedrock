package agentcorebrowser

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"github.com/gorilla/websocket"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

type fakeAgentCoreAPI struct {
	startOut *bedrockagentcore.StartBrowserSessionOutput
	getOut   *bedrockagentcore.GetBrowserSessionOutput
	stopOut  *bedrockagentcore.StopBrowserSessionOutput

	startErr error
	getErr   error
	stopErr  error

	lastStart *bedrockagentcore.StartBrowserSessionInput
	lastGet   *bedrockagentcore.GetBrowserSessionInput
	lastStop  *bedrockagentcore.StopBrowserSessionInput

	stopContextErr error
	stopDeadline   time.Time
}

func (f *fakeAgentCoreAPI) StartBrowserSession(
	_ context.Context,
	in *bedrockagentcore.StartBrowserSessionInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.StartBrowserSessionOutput, error) {
	f.lastStart = in
	if f.startErr != nil {
		return nil, f.startErr
	}
	return f.startOut, nil
}

func (f *fakeAgentCoreAPI) GetBrowserSession(
	_ context.Context,
	in *bedrockagentcore.GetBrowserSessionInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.GetBrowserSessionOutput, error) {
	f.lastGet = in
	if f.getErr != nil {
		return nil, f.getErr
	}
	return f.getOut, nil
}

func (f *fakeAgentCoreAPI) StopBrowserSession(
	ctx context.Context,
	in *bedrockagentcore.StopBrowserSessionInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.StopBrowserSessionOutput, error) {
	f.lastStop = in
	f.stopContextErr = ctx.Err()
	f.stopDeadline, _ = ctx.Deadline()
	if f.stopErr != nil {
		return nil, f.stopErr
	}
	return f.stopOut, nil
}

type fakeArtifacts struct {
	savedName string
	savedPart *genai.Part
	saveErr   error
	version   int64
}

type dialerFunc func(
	context.Context,
	string,
	http.Header,
) (*websocket.Conn, *http.Response, error)

func (f dialerFunc) DialContext(
	ctx context.Context,
	rawURL string,
	header http.Header,
) (*websocket.Conn, *http.Response, error) {
	return f(ctx, rawURL, header)
}

type trackingBody struct {
	*bytes.Reader

	closed atomic.Bool
}

func (b *trackingBody) Close() error {
	b.closed.Store(true)
	return nil
}

func (f *fakeArtifacts) Save(_ context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	f.savedName = name
	f.savedPart = data
	if f.saveErr != nil {
		return nil, f.saveErr
	}
	return &artifact.SaveResponse{Version: f.version}, nil
}

func (f *fakeArtifacts) List(context.Context) (*artifact.ListResponse, error) {
	return &artifact.ListResponse{}, nil
}

func (f *fakeArtifacts) Load(context.Context, string) (*artifact.LoadResponse, error) {
	return nil, errors.New("not implemented")
}

func (f *fakeArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, errors.New("not implemented")
}

type fakeToolContext struct {
	agent.StrictContextMock

	artifacts      *fakeArtifacts
	functionCallID string
}

func (f *fakeToolContext) Artifacts() agent.Artifacts { return f.artifacts }

func (f *fakeToolContext) FunctionCallID() string {
	if f.functionCallID == "" {
		return "tooluse_test"
	}
	return f.functionCallID
}

var _ agent.Context = (*fakeToolContext)(nil)

func newFakeToolCtx(arts *fakeArtifacts) *fakeToolContext {
	return &fakeToolContext{
		StrictContextMock: agent.StrictContextMock{Ctx: context.Background()},
		artifacts:         arts,
	}
}

func testCreds() aws.CredentialsProvider {
	return aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
		return aws.Credentials{
			AccessKeyID:     "AKID",
			SecretAccessKey: "SECRET",
			Source:          "test",
		}, nil
	})
}

func browserStreams(wsURL string) *types.BrowserSessionStream {
	return &types.BrowserSessionStream{
		AutomationStream: &types.AutomationStream{
			StreamEndpoint: aws.String(wsURL),
			StreamStatus:   types.AutomationStreamStatusEnabled,
		},
		LiveViewStream: &types.LiveViewStream{StreamEndpoint: aws.String("https://live.example")},
	}
}

func fakeCDPServer(t *testing.T, failMethod string) string {
	return fakeCDPServerWithHook(t, failMethod, nil)
}

//nolint:cyclop,gocognit,gocyclo // Keeping the fake CDP request table in one place is clearer for these tests.
func fakeCDPServerWithHook(t *testing.T, failMethod string, hook func(string, map[string]any)) string {
	t.Helper()
	upgrader := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		captured := false
		for {
			var req struct {
				ID        int64          `json:"id"`
				Method    string         `json:"method"`
				SessionID string         `json:"sessionId"`
				Params    map[string]any `json:"params"`
			}
			if err := conn.ReadJSON(&req); err != nil {
				return
			}
			if hook != nil {
				hook(req.Method, req.Params)
			}
			if req.Method == failMethod {
				_ = conn.WriteJSON(map[string]any{
					"id":    req.ID,
					"error": map[string]any{"code": -1, "message": "fake cdp failure"},
				})
				continue
			}
			switch req.Method {
			case "Target.getTargets":
				_ = conn.WriteJSON(map[string]any{
					"id": req.ID,
					"result": map[string]any{
						"targetInfos": []map[string]any{{"targetId": "page-1", "type": "page"}},
					},
				})
			case "Target.attachToTarget":
				_ = conn.WriteJSON(map[string]any{
					"id":     req.ID,
					"result": map[string]any{"sessionId": "session-1"},
				})
			case "Fetch.enable",
				"Fetch.continueRequest",
				"Fetch.failRequest",
				"Fetch.fulfillRequest",
				"Fetch.continueWithAuth",
				"Page.enable":
				_ = conn.WriteJSON(map[string]any{"id": req.ID, "result": map[string]any{}})
			case "Page.navigate":
				requestURL, _ := req.Params[paramURL].(string)
				if failMethod == "Page.navigate.redirect" {
					requestURL = "https://blocked.example.net/redirect"
				}
				_ = conn.WriteJSON(map[string]any{
					"method":    "Fetch.requestPaused",
					"sessionId": "session-1",
					"params": map[string]any{
						"requestId":    "request-1",
						"resourceType": "Document",
						"frameId":      "frame-1",
						"networkId":    "network-1",
						"request": map[string]any{
							"url":      requestURL,
							"method":   "GET",
							"headers":  map[string]string{"Accept": "text/html"},
							"postData": "",
						},
					},
				})
				if failMethod == "Page.navigate.subresource" ||
					failMethod == "Page.navigate.dataSubresource" ||
					failMethod == "Page.navigate.fileSubresource" {
					subresourceURL := "http://169.254.169.254/latest/meta-data"
					if failMethod == "Page.navigate.dataSubresource" {
						subresourceURL = "data:text/plain,hello"
					}
					if failMethod == "Page.navigate.fileSubresource" {
						subresourceURL = "file:///etc/passwd"
					}
					_ = conn.WriteJSON(map[string]any{
						"method":    "Fetch.requestPaused",
						"sessionId": "session-1",
						"params": map[string]any{
							"requestId":    "request-2",
							"resourceType": "Image",
							"request": map[string]any{
								"url":     subresourceURL,
								"method":  "GET",
								"headers": map[string]string{},
							},
						},
					})
				}
				if failMethod == "Page.navigate.errorText" {
					_ = conn.WriteJSON(map[string]any{
						"id": req.ID, "result": map[string]any{"errorText": "net::ERR_NAME_NOT_RESOLVED"},
					})
					continue
				}
				if failMethod == "Page.navigate.auth" || failMethod == "Page.navigate.authError" {
					_ = conn.WriteJSON(map[string]any{
						"method":    "Fetch.authRequired",
						"sessionId": "session-1",
						"params": map[string]any{
							"requestId":    "auth-1",
							"resourceType": "Document",
							"frameId":      "frame-1",
							"request": map[string]any{
								"url":      requestURL,
								"method":   "GET",
								"headers":  map[string]string{"Accept": "text/html"},
								"postData": "",
							},
							"authChallenge": map[string]any{
								"source": "Server",
								"origin": "https://example.com",
								"scheme": "basic",
								"realm":  "test realm",
							},
						},
					})
				}
				if failMethod != "Page.navigate.noEvents" {
					if failMethod != "Page.navigate.loadOnly" {
						_ = conn.WriteJSON(map[string]any{
							"method":    "Page.domContentEventFired",
							"sessionId": "session-1",
						})
					}
					if failMethod != "Page.navigate.domOnly" {
						_ = conn.WriteJSON(map[string]any{"method": "Page.loadEventFired", "sessionId": "session-1"})
					}
				}
				_ = conn.WriteJSON(map[string]any{"id": req.ID, "result": map[string]any{"frameId": "frame-1"}})
			case "Runtime.evaluate":
				expr, _ := req.Params["expression"].(string)
				if failMethod == "Runtime.evaluate.exception" ||
					(failMethod == "Runtime.evaluate.selectorException" && strings.HasPrefix(expr, "new Promise")) {
					_ = conn.WriteJSON(map[string]any{
						"id": req.ID,
						"result": map[string]any{
							"exceptionDetails": map[string]any{
								"text": "Uncaught",
								"exception": map[string]any{
									"description": "SyntaxError: invalid selector",
								},
							},
						},
					})
					continue
				}
				var value any = map[string]any{"url": "https://example.com/after", "title": "Example"}
				if failMethod == "Runtime.evaluate.redirect" ||
					(failMethod == "Runtime.evaluate.redirectAfterCapture" && captured) {
					value.(map[string]any)["url"] = "https://blocked.example.net/after"
				}
				if strings.HasPrefix(expr, "new Promise") {
					value = true
				} else if strings.Contains(expr, "document.querySelector") {
					value.(map[string]any)["text"] = "hello world"
					if strings.Contains(expr, "const maxBytes = 5;") {
						value.(map[string]any)["text"] = "hello"
						value.(map[string]any)["truncated"] = true
					}
				}
				_ = conn.WriteJSON(map[string]any{
					"id": req.ID,
					"result": map[string]any{
						"result": map[string]any{"value": value},
					},
				})
			case "Page.captureScreenshot":
				captured = true
				data := []byte("shot")
				if failMethod == "Page.captureScreenshot.oversized" {
					data = []byte("oversized")
				}
				_ = conn.WriteJSON(map[string]any{
					"id":     req.ID,
					"result": map[string]any{"data": base64.StdEncoding.EncodeToString(data)},
				})
			default:
				_ = conn.WriteJSON(map[string]any{
					"id":    req.ID,
					"error": map[string]any{"code": -32601, "message": "unknown method"},
				})
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

//nolint:gocognit // This table validates the constructor's complete public configuration surface.
func TestNewValidationAndDefaults(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{Region: "us-east-1", Credentials: testCreds()}); err == nil {
		t.Fatal("expected nil API error")
	}
	if _, err := New(Config{API: &fakeAgentCoreAPI{}, Credentials: testCreds()}); err == nil {
		t.Fatal("expected missing region error")
	}
	if _, err := New(Config{API: &fakeAgentCoreAPI{}, Region: "us-east-1"}); err == nil {
		t.Fatal("expected missing credentials error")
	}
	if _, err := New(Config{
		API:                   &fakeAgentCoreAPI{},
		Region:                "us-east-1",
		Credentials:           testCreds(),
		SessionTimeoutSeconds: maxSessionTimeout + 1,
	}); err == nil {
		t.Fatal("expected maximum session timeout error")
	}
	if _, err := New(Config{
		API:          &fakeAgentCoreAPI{},
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"https://example.com"},
	}); err == nil {
		t.Fatal("expected invalid allowed host error")
	}
	if _, err := New(Config{
		API:         &fakeAgentCoreAPI{},
		Region:      "us-east-1",
		Credentials: testCreds(),
		DeniedHosts: []string{"example.com:443"},
	}); err == nil {
		t.Fatal("expected invalid denied host error")
	}
	if _, err := New(Config{
		API:                &fakeAgentCoreAPI{},
		Region:             "us-east-1",
		Credentials:        testCreds(),
		RequestMiddlewares: []RequestMiddleware{nil},
	}); err == nil {
		t.Fatal("expected nil request middleware error")
	}
	if _, err := New(Config{
		API:         &fakeAgentCoreAPI{},
		Region:      "us-east-1",
		Credentials: testCreds(),
		RequestMiddlewares: []RequestMiddleware{
			func(RequestHandler) RequestHandler { return nil },
		},
	}); err == nil {
		t.Fatal("expected nil request handler error")
	}
	if _, err := New(Config{
		API:            &fakeAgentCoreAPI{},
		Region:         "us-east-1",
		Credentials:    testCreds(),
		URLMiddlewares: []URLMiddleware{nil},
	}); err == nil {
		t.Fatal("expected nil URL middleware error")
	}
	if _, err := New(Config{
		API:         &fakeAgentCoreAPI{},
		Region:      "us-east-1",
		Credentials: testCreds(),
		URLMiddlewares: []URLMiddleware{
			func(URLHandler) URLHandler { return nil },
		},
	}); err == nil {
		t.Fatal("expected nil URL handler error")
	}

	for _, tc := range []struct {
		name   string
		change func(*Config)
	}{
		{name: "negative session timeout", change: func(c *Config) { c.SessionTimeoutSeconds = -1 }},
		{name: "negative viewport", change: func(c *Config) { c.ViewportWidth, c.ViewportHeight = -1, -1 }},
		{name: "negative navigation timeout", change: func(c *Config) { c.NavigationTimeout = -1 }},
		{name: "negative cleanup timeout", change: func(c *Config) { c.CleanupTimeout = -1 }},
		{name: "negative max text", change: func(c *Config) { c.MaxTextBytes = -1 }},
		{name: "negative max screenshot", change: func(c *Config) { c.MaxScreenshotBytes = -1 }},
		{name: "invalid wait mode", change: func(c *Config) { c.WaitUntil = "network_idle" }},
		{name: "oversized response limit", change: func(c *Config) { c.MaxScreenshotBytes = int64(^uint64(0) >> 1) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cfg := Config{API: &fakeAgentCoreAPI{}, Region: "us-east-1", Credentials: testCreds()}
			tc.change(&cfg)
			if _, err := New(cfg); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}

	tl, err := New(Config{API: &fakeAgentCoreAPI{}, Region: " us-east-1 ", Credentials: testCreds()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	bt := tl.(*browserTool)
	if bt.browserIdentifier != defaultBrowserIdentifier {
		t.Errorf("browser id = %q", bt.browserIdentifier)
	}
	if bt.sessionTimeoutSeconds != defaultSessionTimeout {
		t.Errorf("timeout = %d", bt.sessionTimeoutSeconds)
	}
	if bt.maxTextBytes != defaultMaxTextBytes {
		t.Errorf("max text = %d", bt.maxTextBytes)
	}
	if bt.cleanupTimeout != defaultCleanupTimeout {
		t.Errorf("cleanup timeout = %s", bt.cleanupTimeout)
	}
	if bt.maxScreenshotBytes != defaultMaxScreenshotBytes {
		t.Errorf("max screenshot = %d", bt.maxScreenshotBytes)
	}
	if bt.waitUntil != WaitUntilLoad {
		t.Errorf("wait until = %q", bt.waitUntil)
	}
	if bt.dialer != websocket.DefaultDialer {
		t.Errorf("dialer = %#v", bt.dialer)
	}
	if bt.automationReadLimit <= defaultMaxScreenshotBytes {
		t.Errorf("automation read limit = %d", bt.automationReadLimit)
	}
}

func TestDeclarationAndProcessRequest(t *testing.T) {
	t.Parallel()
	tl, err := New(Config{API: &fakeAgentCoreAPI{}, Region: "us-east-1", Credentials: testCreds()})
	if err != nil {
		t.Fatal(err)
	}
	bt := tl.(*browserTool)
	decl := bt.Declaration()
	if decl.Name != ToolName {
		t.Errorf("declaration name = %q", decl.Name)
	}
	if got := decl.Parameters.Properties[paramAction].Enum; len(got) != 6 {
		t.Errorf("action enum = %v", got)
	}
	if got := strings.Join(decl.Parameters.Properties[paramFormat].Enum, ","); got != "png,jpeg,jpg" {
		t.Errorf("format enum = %v", got)
	}
	waitUntilEnum := strings.Join(decl.Parameters.Properties[paramWaitUntil].Enum, ",")
	if waitUntilEnum != "load,dom_content_loaded,none" {
		t.Errorf("wait_until enum = %v", waitUntilEnum)
	}
	if got := decl.Parameters.Properties[paramFullPage].Type; got != schemaTypeBoolean {
		t.Errorf("full_page type = %q", got)
	}
	if got := decl.Parameters.Properties[paramQuality].Type; got != schemaTypeInteger {
		t.Errorf("quality type = %q", got)
	}

	req := &model.LLMRequest{}
	if err := bt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if req.Tools[ToolName] == nil {
		t.Fatal("tool not packed into request")
	}
	if len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatal("function declaration not packed")
	}
	if err := bt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err == nil {
		t.Fatal("expected duplicate tool error")
	}
}

func TestHostPolicy(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{
		API:          &fakeAgentCoreAPI{},
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"example.com"},
		DeniedHosts:  []string{"bad.example.com"},
	})
	bt := tl.(*browserTool)

	if err := bt.checkURL(context.Background(), "https://sub.example.com/path", URLStageNavigate); err != nil {
		t.Fatalf("allowed subdomain rejected: %v", err)
	}
	if err := bt.checkURL(context.Background(), "https://bad.example.com", URLStageNavigate); err == nil {
		t.Fatal("expected denied host error")
	}
	if err := bt.checkURL(context.Background(), "https://other.test", URLStageNavigate); err == nil {
		t.Fatal("expected not allowed host error")
	}
	if err := bt.checkURL(context.Background(), "file:///etc/passwd", URLStageNavigate); err == nil {
		t.Fatal("expected scheme error")
	}
}

func TestNormalizeHostsRejectsUnsupportedWildcards(t *testing.T) {
	t.Parallel()
	for _, host := range []string{"*", "example.*", "*.*.example.com"} {
		if _, err := New(Config{
			API:          &fakeAgentCoreAPI{},
			Region:       "us-east-1",
			Credentials:  testCreds(),
			AllowedHosts: []string{host},
		}); err == nil {
			t.Errorf("expected invalid wildcard error for %q", host)
		}
	}
	tl, err := New(Config{
		API:          &fakeAgentCoreAPI{},
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"*.Example.COM."},
	})
	if err != nil {
		t.Fatalf("normalize valid wildcard: %v", err)
	}
	normalized := tl.(*browserTool).allowedHosts
	if len(normalized) != 1 || normalized[0] != "example.com" {
		t.Fatalf("normalized hosts = %v", normalized)
	}
}

func TestURLMiddlewareOrderingAndStages(t *testing.T) {
	t.Parallel()
	var calls []string
	middleware := func(name string) URLMiddleware {
		return func(next URLHandler) URLHandler {
			return func(ctx context.Context, check URLCheck) error {
				calls = append(calls, name+":before:"+string(check.Stage))
				err := next(ctx, check)
				calls = append(calls, name+":after:"+string(check.Stage))
				return err
			}
		}
	}
	tl, err := New(Config{
		API:            &fakeAgentCoreAPI{},
		Region:         "us-east-1",
		Credentials:    testCreds(),
		URLMiddlewares: []URLMiddleware{middleware("first"), middleware("second")},
	})
	if err != nil {
		t.Fatal(err)
	}
	bt := tl.(*browserTool)
	for _, stage := range []URLStage{URLStageNavigate, URLStageRequest, URLStageCurrent, URLStageFinal} {
		if err := bt.checkURL(context.Background(), "https://example.com", stage); err != nil {
			t.Fatalf("stage %q: %v", stage, err)
		}
	}
	want := []string{
		"first:before:navigate", "second:before:navigate", "second:after:navigate", "first:after:navigate",
		"first:before:request", "second:before:request", "second:after:request", "first:after:request",
		"first:before:current", "second:before:current", "second:after:current", "first:after:current",
		"first:before:final", "second:before:final", "second:after:final", "first:after:final",
	}
	if strings.Join(calls, ",") != strings.Join(want, ",") {
		t.Fatalf("middleware calls = %v", calls)
	}
}

func TestURLMiddlewareCanReplaceHostPolicy(t *testing.T) {
	t.Parallel()
	replace := func(URLHandler) URLHandler {
		return func(context.Context, URLCheck) error { return nil }
	}
	tl, err := New(Config{
		API:            &fakeAgentCoreAPI{},
		Region:         "us-east-1",
		Credentials:    testCreds(),
		AllowedHosts:   []string{"example.com"},
		URLMiddlewares: []URLMiddleware{replace},
	})
	if err != nil {
		t.Fatal(err)
	}
	bt := tl.(*browserTool)
	if err := bt.checkURL(context.Background(), "https://other.test", URLStageNavigate); err != nil {
		t.Fatalf("replacement middleware did not replace host policy: %v", err)
	}
	if err := bt.checkURL(context.Background(), "file:///etc/passwd", URLStageNavigate); err == nil {
		t.Fatal("replacement middleware bypassed structural URL validation")
	}
}

func TestURLMiddlewareRewriteIsValidatedByNext(t *testing.T) {
	t.Parallel()
	rewrite := func(next URLHandler) URLHandler {
		return func(ctx context.Context, check URLCheck) error {
			check.URL.Host = "127.0.0.1"
			return next(ctx, check)
		}
	}
	tl, err := New(Config{
		API:            &fakeAgentCoreAPI{},
		Region:         "us-east-1",
		Credentials:    testCreds(),
		URLMiddlewares: []URLMiddleware{rewrite},
	})
	if err != nil {
		t.Fatal(err)
	}
	err = tl.(*browserTool).checkURL(context.Background(), "https://example.com", URLStageRequest)
	if err == nil || !strings.Contains(err.Error(), "explicit allowlist") {
		t.Fatalf("expected rewritten URL policy error, got %v", err)
	}
}

func TestURLStructureRejectsAmbiguousHosts(t *testing.T) {
	t.Parallel()
	for _, host := range []string{
		"example.com@evil.test",
		"example.com/evil.test",
		`example.com\evil.test`,
		"example.com%evil.test",
	} {
		check := URLCheck{URL: url.URL{Scheme: schemeHTTPS, Host: host}, Stage: URLStageNavigate}
		if err := validateURLStructure(check); err == nil {
			t.Errorf("expected invalid host error for %q", host)
		}
	}
	if _, err := parseURLCheck("https://example.com%25evil.test", URLStageNavigate); err == nil {
		t.Fatal("expected encoded percent host error")
	}
	if _, err := parseURLCheck("https://[fe80::1%25en0]", URLStageNavigate); err != nil {
		t.Fatalf("valid IPv6 zone identifier rejected: %v", err)
	}
}

func TestHostPolicyRejectsLocalTargetsByDefault(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeAgentCoreAPI{}, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	for _, rawURL := range []string{
		"https://localhost",
		"https://sub.localhost",
		"https://127.0.0.1",
		"https://[::1]",
		"https://169.254.169.254/latest/meta-data",
		"https://10.0.0.1",
		"https://[fc00::1]",
	} {
		if err := bt.checkURL(context.Background(), rawURL, URLStageNavigate); err == nil {
			t.Fatalf("expected local target rejection for %s", rawURL)
		}
	}
	if err := bt.checkURL(context.Background(), "https://93.184.216.34", URLStageNavigate); err != nil {
		t.Fatalf("public IP rejected: %v", err)
	}

	tl, _ = New(Config{
		API:          &fakeAgentCoreAPI{},
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"localhost", "127.0.0.1"},
	})
	bt = tl.(*browserTool)
	if err := bt.checkURL(context.Background(), "https://localhost", URLStageNavigate); err != nil {
		t.Fatalf("explicit localhost allow rejected: %v", err)
	}
	if err := bt.checkURL(context.Background(), "https://127.0.0.1", URLStageNavigate); err != nil {
		t.Fatalf("explicit loopback allow rejected: %v", err)
	}
}

func TestHostPolicyRejectsLegacyIPv4TargetsByDefault(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeAgentCoreAPI{}, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	for _, rawURL := range []string{
		"https://2130706433",
		"https://127.1",
		"https://0177.0.0.1",
		"https://0x7f000001",
		"https://0x7f.0.0.1",
	} {
		if err := bt.checkURL(context.Background(), rawURL, URLStageNavigate); err == nil {
			t.Fatalf("expected legacy IPv4 target rejection for %s", rawURL)
		}
	}
}
