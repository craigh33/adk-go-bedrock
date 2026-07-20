package agentcorebrowser

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"maps"
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

func TestSignedWebSocketHeadersRejectsUnsupportedScheme(t *testing.T) {
	t.Parallel()
	tl, err := New(Config{API: &fakeAgentCoreAPI{}, Region: "us-east-1", Credentials: testCreds()})
	if err != nil {
		t.Fatal(err)
	}
	bt := tl.(*browserTool)
	if _, err := bt.signedWebSocketHeaders(context.Background(), "https://example.com/stream"); err == nil {
		t.Fatal("expected unsupported scheme error")
	}
}

func TestOpenCDPUsesConfiguredDialerAndSignedHeaders(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "")
	var gotURL string
	var gotHeaders http.Header
	dialer := dialerFunc(func(
		ctx context.Context,
		rawURL string,
		headers http.Header,
	) (*websocket.Conn, *http.Response, error) {
		gotURL = rawURL
		gotHeaders = headers.Clone()
		return websocket.DefaultDialer.DialContext(ctx, rawURL, headers)
	})
	tl, err := New(Config{
		API:         &fakeAgentCoreAPI{},
		Region:      "us-east-1",
		Credentials: testCreds(),
		Dialer:      dialer,
	})
	if err != nil {
		t.Fatal(err)
	}
	cdp, err := tl.(*browserTool).openCDP(context.Background(), wsURL)
	if err != nil {
		t.Fatalf("openCDP: %v", err)
	}
	cdp.close()
	if gotURL != wsURL {
		t.Errorf("dial URL = %q", gotURL)
	}
	if gotHeaders.Get("Authorization") == "" || gotHeaders.Get("X-Amz-Date") == "" {
		t.Errorf("signed headers = %v", gotHeaders)
	}
}

func TestOpenCDPClosesDialResponseOnError(t *testing.T) {
	t.Parallel()
	body := &trackingBody{Reader: bytes.NewReader(nil)}
	dialErr := errors.New("dial failed")
	dialer := dialerFunc(func(
		context.Context,
		string,
		http.Header,
	) (*websocket.Conn, *http.Response, error) {
		return nil, &http.Response{Body: body}, dialErr
	})
	tl, err := New(Config{
		API:         &fakeAgentCoreAPI{},
		Region:      "us-east-1",
		Credentials: testCreds(),
		Dialer:      dialer,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*browserTool).openCDP(context.Background(), "wss://example.com/stream")
	if !errors.Is(err, dialErr) {
		t.Fatalf("openCDP error = %v", err)
	}
	if !body.closed.Load() {
		t.Fatal("dial response body was not closed")
	}
}

func TestOpenCDPRejectsNilConnection(t *testing.T) {
	t.Parallel()
	dialer := dialerFunc(func(
		context.Context,
		string,
		http.Header,
	) (*websocket.Conn, *http.Response, error) {
		return nil, nil, nil
	})
	tl, err := New(Config{
		API:         &fakeAgentCoreAPI{},
		Region:      "us-east-1",
		Credentials: testCreds(),
		Dialer:      dialer,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*browserTool).openCDP(context.Background(), "wss://example.com/stream")
	if err == nil || !strings.Contains(err.Error(), "nil connection") {
		t.Fatalf("expected nil connection error, got %v", err)
	}
}

func TestOpenCDPRejectsOversizedMessage(t *testing.T) {
	t.Parallel()
	upgrader := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()
		var req map[string]any
		if err := conn.ReadJSON(&req); err != nil {
			return
		}
		_ = conn.WriteJSON(map[string]any{
			"id":     req["id"],
			"result": map[string]any{"padding": strings.Repeat("x", 2<<20)},
		})
	}))
	t.Cleanup(srv.Close)
	wsURL := "ws" + strings.TrimPrefix(srv.URL, "http")
	tl, err := New(Config{
		API:                &fakeAgentCoreAPI{},
		Region:             "us-east-1",
		Credentials:        testCreds(),
		MaxTextBytes:       1,
		MaxScreenshotBytes: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	cdp, err := tl.(*browserTool).openCDP(context.Background(), wsURL)
	if err != nil {
		t.Fatal(err)
	}
	defer cdp.close()
	_, err = cdp.pageSession(context.Background())
	if err == nil || !strings.Contains(err.Error(), "read limit") {
		t.Fatalf("expected WebSocket read limit error, got %v", err)
	}
}

func TestStartStatusStopUseAgentCoreAPI(t *testing.T) {
	t.Parallel()
	now := time.Now()
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			CreatedAt:         &now,
			Streams:           browserStreams("wss://agentcore.example/automation"),
		},
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier:     aws.String("aws.browser.v1"),
			SessionId:             aws.String("session-1"),
			Status:                types.BrowserSessionStatusReady,
			SessionTimeoutSeconds: aws.Int32(900),
			Streams:               browserStreams("wss://agentcore.example/automation"),
		},
		stopOut: &bedrockagentcore.StopBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			LastUpdatedAt:     &now,
		},
	}
	tl, _ := New(Config{
		API:                   api,
		Region:                "us-east-1",
		Credentials:           testCreds(),
		ViewportWidth:         1280,
		ViewportHeight:        720,
		SessionTimeoutSeconds: 60,
	})
	bt := tl.(*browserTool)
	ctx := newFakeToolCtx(&fakeArtifacts{})
	ctx.functionCallID = "tooluse_abc"

	startOut, err := bt.Run(ctx, map[string]any{paramAction: actionStart})
	if err != nil {
		t.Fatalf("start: %v", err)
	}
	if _, ok := startOut["automation_stream_url"]; ok {
		t.Fatal("start result leaked automation stream URL")
	}
	if got := aws.ToString(
		api.lastStart.ClientToken,
	); !strings.HasPrefix(got, "tooluse-abc-") ||
		len(got) < 33 {
		t.Errorf("client token = %q", got)
	}
	if got := aws.ToInt32(api.lastStart.SessionTimeoutSeconds); got != 60 {
		t.Errorf("session timeout = %d", got)
	}
	if api.lastStart.ViewPort == nil || aws.ToInt32(api.lastStart.ViewPort.Width) != 1280 {
		t.Fatalf("viewport not set: %#v", api.lastStart.ViewPort)
	}

	statusOut, err := bt.Run(ctx, map[string]any{paramAction: actionStatus, paramSessionID: "session-1"})
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if statusOut[resultKeyStatus] != statusSuccess || statusOut[resultKeySessionStatus] != "READY" {
		t.Errorf(
			"status result = %v session_status = %v",
			statusOut[resultKeyStatus],
			statusOut[resultKeySessionStatus],
		)
	}
	if _, ok := statusOut["automation_stream_url"]; ok {
		t.Fatal("status result leaked automation stream URL")
	}
	if _, ok := statusOut["automation_stream_status"]; ok {
		t.Fatal("status result leaked automation stream status")
	}
	if aws.ToString(api.lastGet.SessionId) != "session-1" {
		t.Errorf("get session id = %q", aws.ToString(api.lastGet.SessionId))
	}

	if _, err := bt.Run(ctx, map[string]any{paramAction: actionStop, paramSessionID: "session-1"}); err != nil {
		t.Fatalf("stop: %v", err)
	}
	if aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Errorf("stop session id = %q", aws.ToString(api.lastStop.SessionId))
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

func TestNavigateStartsSessionAndUsesCDP(t *testing.T) {
	t.Parallel()
	var attachCount atomic.Int32
	var continueCount atomic.Int32
	wsURL := fakeCDPServerWithHook(t, "", func(method string, _ map[string]any) {
		if method == "Target.attachToTarget" {
			attachCount.Add(1)
		}
		if method == "Fetch.continueRequest" {
			continueCount.Add(1)
		}
	})
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}})
	bt := tl.(*browserTool)

	out, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
	if out[paramSessionID] != "session-1" {
		t.Errorf("session_id = %v", out[paramSessionID])
	}
	if out["url"] != "https://example.com/after" || out["title"] != "Example" {
		t.Errorf("metadata = url %v title %v", out["url"], out["title"])
	}
	if attachCount.Load() != 1 {
		t.Errorf("attach count = %d", attachCount.Load())
	}
	if continueCount.Load() != 1 {
		t.Errorf("continued document requests = %d", continueCount.Load())
	}
}

func TestNavigateBlocksDeniedRedirectBeforeRequest(t *testing.T) {
	t.Parallel()
	var failCount atomic.Int32
	wsURL := fakeCDPServerWithHook(t, "Page.navigate.redirect", func(method string, _ map[string]any) {
		if method == "Fetch.failRequest" {
			failCount.Add(1)
		}
	})
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:          api,
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"example.com"},
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "browser request") {
		t.Fatalf("expected redirect policy error, got %v", err)
	}
	if failCount.Load() != 1 {
		t.Fatalf("failed document requests = %d", failCount.Load())
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
}

func TestNavigateAppliesHostPolicyToSubresources(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name       string
		serverMode string
		wantError  bool
	}{
		{name: "blocks HTTP subresource", serverMode: "Page.navigate.subresource", wantError: true},
		{name: "allows browser-local scheme", serverMode: "Page.navigate.dataSubresource"},
		{name: "blocks local file scheme", serverMode: "Page.navigate.fileSubresource", wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			wsURL := fakeCDPServer(t, tc.serverMode)
			api := &fakeAgentCoreAPI{
				startOut: &bedrockagentcore.StartBrowserSessionOutput{
					BrowserIdentifier: aws.String("aws.browser.v1"),
					SessionId:         aws.String("session-1"),
					Streams:           browserStreams(wsURL),
				},
			}
			tl, _ := New(Config{
				API:          api,
				Region:       "us-east-1",
				Credentials:  testCreds(),
				AllowedHosts: []string{"example.com"},
			})

			_, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
				paramAction: actionNavigate,
				paramURL:    "https://example.com",
			})
			if tc.wantError && err == nil {
				t.Fatalf("expected subresource policy error, got %v", err)
			}
			if !tc.wantError && err != nil {
				t.Fatalf("navigate: %v", err)
			}
		})
	}
}

func TestRequestMiddlewareRewritesRequest(t *testing.T) {
	t.Parallel()
	continued := make(chan map[string]any, 1)
	wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
		if method == "Fetch.continueRequest" {
			continued <- params
		}
	})
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	middleware := func(next RequestHandler) RequestHandler {
		return func(ctx context.Context, request *BrowserRequest) (*BrowserResponse, error) {
			if request.ResourceType != "Document" || request.FrameID != "frame-1" ||
				request.NetworkID != "network-1" {
				t.Errorf("request metadata = %#v", request)
			}
			if request.Headers.Get("Accept") != "text/html" {
				t.Errorf("accept header = %q", request.Headers.Get("Accept"))
			}
			request.URL = "https://example.com/rewritten"
			request.Method = http.MethodPost
			request.Headers.Set("X-Test", "yes")
			request.PostData = []byte("payload")
			return next(ctx, request)
		}
	}
	tl, _ := New(Config{
		API:                api,
		Region:             "us-east-1",
		Credentials:        testCreds(),
		AllowedHosts:       []string{"example.com"},
		RequestMiddlewares: []RequestMiddleware{middleware},
	})

	_, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
	params := <-continued
	encoded, _ := json.Marshal(params)
	for _, want := range []string{
		`"url":"https://example.com/rewritten"`,
		`"method":"POST"`,
		`"postData":"cGF5bG9hZA=="`,
		`"name":"X-Test"`,
		`"value":"yes"`,
	} {
		if !bytes.Contains(encoded, []byte(want)) {
			t.Errorf("continued request %s does not contain %s", encoded, want)
		}
	}
}

func TestRequestMiddlewareFulfillsRequest(t *testing.T) {
	t.Parallel()
	fulfilled := make(chan map[string]any, 1)
	wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
		if method == "Fetch.fulfillRequest" {
			fulfilled <- params
		}
	})
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	middleware := func(RequestHandler) RequestHandler {
		return func(context.Context, *BrowserRequest) (*BrowserResponse, error) {
			return &BrowserResponse{
				StatusCode: http.StatusCreated,
				StatusText: "Created by middleware",
				Headers:    http.Header{"Content-Type": []string{"text/plain"}},
				Body:       []byte("synthetic"),
			}, nil
		}
	}
	tl, _ := New(Config{
		API:                api,
		Region:             "us-east-1",
		Credentials:        testCreds(),
		RequestMiddlewares: []RequestMiddleware{middleware},
	})

	_, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
	encoded, _ := json.Marshal(<-fulfilled)
	for _, want := range []string{
		`"responseCode":201`,
		`"responsePhrase":"Created by middleware"`,
		`"body":"c3ludGhldGlj"`,
		`"name":"Content-Type"`,
	} {
		if !bytes.Contains(encoded, []byte(want)) {
			t.Errorf("fulfilled request %s does not contain %s", encoded, want)
		}
	}
}

func TestRequestMiddlewareCanReplaceDefaultPolicy(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Page.navigate.subresource")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	middleware := func(next RequestHandler) RequestHandler {
		return func(ctx context.Context, request *BrowserRequest) (*BrowserResponse, error) {
			if request.ResourceType == "Image" {
				return nil, nil
			}
			return next(ctx, request)
		}
	}
	tl, _ := New(Config{
		API:                api,
		Region:             "us-east-1",
		Credentials:        testCreds(),
		AllowedHosts:       []string{"example.com"},
		RequestMiddlewares: []RequestMiddleware{middleware},
	})

	_, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
}

func TestNavigateWaitModesAndOverride(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name       string
		serverMode string
		configured WaitUntil
		override   string
	}{
		{name: "load default", serverMode: "Page.navigate.loadOnly"},
		{name: "DOMContentLoaded", serverMode: "Page.navigate.domOnly", configured: WaitUntilDOMContentLoaded},
		{name: "none", serverMode: "Page.navigate.noEvents", configured: WaitUntilNone},
		{name: "per action override", serverMode: "Page.navigate.noEvents", override: string(WaitUntilNone)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			wsURL := fakeCDPServer(t, tc.serverMode)
			api := &fakeAgentCoreAPI{startOut: &bedrockagentcore.StartBrowserSessionOutput{
				BrowserIdentifier: aws.String("aws.browser.v1"),
				SessionId:         aws.String("session-1"),
				Streams:           browserStreams(wsURL),
			}}
			tl, err := New(Config{
				API:         api,
				Region:      "us-east-1",
				Credentials: testCreds(),
				WaitUntil:   tc.configured,
			})
			if err != nil {
				t.Fatal(err)
			}
			args := map[string]any{paramAction: actionNavigate, paramURL: "https://example.com"}
			if tc.override != "" {
				args[paramWaitUntil] = tc.override
			}
			if _, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), args); err != nil {
				t.Fatalf("navigate: %v", err)
			}
		})
	}
}

func TestNavigateRejectsInvalidWaitOverrideBeforeStartingSession(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionNavigate,
		paramURL:       "https://example.com",
		paramWaitUntil: "network_idle",
	})
	if err == nil || !strings.Contains(err.Error(), paramWaitUntil) {
		t.Fatalf("expected wait_until validation error, got %v", err)
	}
	if api.lastStart != nil {
		t.Fatal("session started for invalid wait_until")
	}
}

func TestNavigateOptionalStringArgumentsRejectInvalidTypes(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		key  string
	}{
		{name: "wait_until", key: paramWaitUntil},
		{name: "wait_for_selector", key: paramWaitForSelector},
		{name: "session_id", key: paramSessionID},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			api := &fakeAgentCoreAPI{}
			tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
			if err != nil {
				t.Fatal(err)
			}
			args := map[string]any{
				paramAction: actionNavigate,
				paramURL:    "https://example.com",
				tc.key:      true,
			}
			_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), args)
			if err == nil || !strings.Contains(err.Error(), tc.key+" must be a string") {
				t.Fatalf("expected argument type error, got %v", err)
			}
			if api.lastStart != nil {
				t.Fatal("session started before argument validation")
			}
		})
	}
}

func TestRequiredStringArgumentsRejectInvalidTypes(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		action string
		key    string
	}{
		{name: "navigate URL", action: actionNavigate, key: paramURL},
		{name: "status session ID", action: actionStatus, key: paramSessionID},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			api := &fakeAgentCoreAPI{}
			tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
			if err != nil {
				t.Fatal(err)
			}
			_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
				paramAction: tc.action,
				tc.key:      true,
			})
			if err == nil || !strings.Contains(err.Error(), tc.key+" must be a string") {
				t.Fatalf("expected argument type error, got %v", err)
			}
			if api.lastStart != nil || api.lastGet != nil {
				t.Fatal("AgentCore API called before argument validation")
			}
		})
	}
}

func TestModelFacingStringArgumentsRejectInvalidTypes(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		key  string
		args map[string]any
	}{
		{
			name: "action",
			key:  paramAction,
			args: map[string]any{paramAction: true},
		},
		{
			name: "extract selector",
			key:  paramSelector,
			args: map[string]any{
				paramAction:    actionExtractText,
				paramSessionID: "session-1",
				paramSelector:  true,
			},
		},
		{
			name: "screenshot file name",
			key:  paramFileName,
			args: map[string]any{
				paramAction:    actionScreenshot,
				paramSessionID: "session-1",
				paramFileName:  true,
			},
		},
		{
			name: "screenshot format",
			key:  paramFormat,
			args: map[string]any{
				paramAction:    actionScreenshot,
				paramSessionID: "session-1",
				paramFormat:    true,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			api := &fakeAgentCoreAPI{}
			tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
			if err != nil {
				t.Fatal(err)
			}
			_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), tc.args)
			if err == nil || !strings.Contains(err.Error(), tc.key+" must be a string") {
				t.Fatalf("expected argument type error, got %v", err)
			}
			if api.lastStart != nil || api.lastGet != nil || api.lastStop != nil {
				t.Fatal("AgentCore API called before argument validation")
			}
		})
	}
}

func TestNavigateWaitsForSelector(t *testing.T) {
	t.Parallel()
	evaluations := make(chan map[string]any, 4)
	wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
		if method == "Runtime.evaluate" {
			evaluations <- params
		}
	})
	api := &fakeAgentCoreAPI{startOut: &bedrockagentcore.StartBrowserSessionOutput{
		BrowserIdentifier: aws.String("aws.browser.v1"),
		SessionId:         aws.String("session-1"),
		Streams:           browserStreams(wsURL),
	}}
	tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:          actionNavigate,
		paramURL:             "https://example.com",
		paramWaitForSelector: "main[data-ready]",
	})
	if err != nil {
		t.Fatalf("navigate: %v", err)
	}
	waitParams := <-evaluations
	if waitParams["awaitPromise"] != true || waitParams["returnByValue"] != true {
		t.Errorf("selector evaluation params = %v", waitParams)
	}
	if expr, _ := waitParams["expression"].(string); !strings.Contains(expr, `main[data-ready]`) ||
		!strings.Contains(expr, "MutationObserver") || !strings.Contains(expr, "attributes: true") {
		t.Errorf("selector expression = %q", expr)
	}
}

func TestExtractTextUsesWaitSelectorAsExtractionSelector(t *testing.T) {
	t.Parallel()
	evaluations := make(chan string, 4)
	wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
		if method == "Runtime.evaluate" {
			expr, _ := params["expression"].(string)
			evaluations <- expr
		}
	})
	api := &fakeAgentCoreAPI{getOut: &bedrockagentcore.GetBrowserSessionOutput{
		BrowserIdentifier: aws.String("aws.browser.v1"),
		SessionId:         aws.String("session-1"),
		Streams:           browserStreams(wsURL),
	}}
	tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:          actionExtractText,
		paramSessionID:       "session-1",
		paramWaitForSelector: "article.ready",
	})
	if err != nil {
		t.Fatalf("extract_text: %v", err)
	}
	waitExpr := <-evaluations
	extractExpr := <-evaluations
	if !strings.HasPrefix(waitExpr, "new Promise") {
		t.Errorf("wait expression = %q", waitExpr)
	}
	if !strings.Contains(extractExpr, `const selector = "article.ready";`) {
		t.Errorf("extract expression = %q", extractExpr)
	}
}

//nolint:gocognit // The cases share one CDP setup and differ only in selector timing behavior.
func TestSelectorWaitErrorsAndTimeout(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name       string
		serverMode string
		timeout    time.Duration
		delay      time.Duration
		want       string
	}{
		{name: "invalid selector", serverMode: "Runtime.evaluate.selectorException", timeout: time.Second, want: "SyntaxError: invalid selector"},
		{name: "timeout", timeout: 50 * time.Millisecond, delay: 200 * time.Millisecond, want: "i/o timeout"},
		{name: "delayed resolution", timeout: time.Second, delay: 20 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			wsURL := fakeCDPServerWithHook(t, tc.serverMode, func(method string, params map[string]any) {
				if method == "Runtime.evaluate" && tc.delay > 0 {
					expr, _ := params["expression"].(string)
					if strings.HasPrefix(expr, "new Promise") {
						time.Sleep(tc.delay)
					}
				}
			})
			api := &fakeAgentCoreAPI{startOut: &bedrockagentcore.StartBrowserSessionOutput{
				BrowserIdentifier: aws.String("aws.browser.v1"),
				SessionId:         aws.String("session-1"),
				Streams:           browserStreams(wsURL),
			}}
			tl, err := New(Config{
				API:               api,
				Region:            "us-east-1",
				Credentials:       testCreds(),
				NavigationTimeout: tc.timeout,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
				paramAction:          actionNavigate,
				paramURL:             "https://example.com",
				paramWaitForSelector: "main",
			})
			if tc.want == "" && err != nil {
				t.Fatalf("navigate: %v", err)
			}
			if tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("expected error containing %q, got %v", tc.want, err)
			}
		})
	}
}

//nolint:gocognit // A table keeps every public authentication response on the same protocol path.
func TestAuthenticationFlows(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name         string
		response     AuthResponse
		handlerError error
		wantProtocol string
		wantError    string
	}{
		{name: "default", response: AuthResponse{}, wantProtocol: "Default"},
		{name: "cancel", response: AuthResponse{Action: AuthActionCancel}, wantProtocol: "CancelAuth"},
		{name: "credentials", response: AuthResponse{Action: AuthActionProvideCredentials, Username: "user", Password: "pass"}, wantProtocol: "ProvideCredentials"},
		{name: "handler error", handlerError: errors.New("no credentials"), wantProtocol: "CancelAuth", wantError: "no credentials"},
		{name: "invalid action", response: AuthResponse{Action: "retry"}, wantProtocol: "CancelAuth", wantError: "invalid authentication action"},
		{name: "credentials with default", response: AuthResponse{Username: "user"}, wantProtocol: "CancelAuth", wantError: "cannot include credentials"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			authParams := make(chan map[string]any, 1)
			enableParams := make(chan map[string]any, 1)
			wsURL := fakeCDPServerWithHook(t, "Page.navigate.auth", func(method string, params map[string]any) {
				switch method {
				case "Fetch.enable":
					enableParams <- params
				case "Fetch.continueWithAuth":
					authParams <- params
				}
			})
			api := &fakeAgentCoreAPI{startOut: &bedrockagentcore.StartBrowserSessionOutput{
				BrowserIdentifier: aws.String("aws.browser.v1"),
				SessionId:         aws.String("session-1"),
				Streams:           browserStreams(wsURL),
			}}
			handler := func(_ context.Context, challenge AuthChallenge) (AuthResponse, error) {
				if challenge.Source != "Server" || challenge.Origin != "https://example.com" ||
					challenge.Scheme != "basic" || challenge.Realm != "test realm" ||
					challenge.Request.URL != "https://example.com" {
					t.Errorf("challenge = %#v", challenge)
				}
				return tc.response, tc.handlerError
			}
			tl, err := New(Config{
				API:         api,
				Region:      "us-east-1",
				Credentials: testCreds(),
				AuthHandler: handler,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
				paramAction: actionNavigate,
				paramURL:    "https://example.com",
			})
			if tc.wantError == "" && err != nil {
				t.Fatalf("navigate: %v", err)
			}
			if tc.wantError != "" && (err == nil || !strings.Contains(err.Error(), tc.wantError)) {
				t.Fatalf("expected error containing %q, got %v", tc.wantError, err)
			}
			if (<-enableParams)["handleAuthRequests"] != true {
				t.Error("Fetch.enable did not enable auth handling")
			}
			response, _ := (<-authParams)["authChallengeResponse"].(map[string]any)
			if response["response"] != tc.wantProtocol {
				t.Errorf("auth response = %v", response)
			}
			if tc.wantProtocol == "ProvideCredentials" &&
				(response["username"] != "user" || response["password"] != "pass") {
				t.Errorf("auth credentials = %v", response)
			}
		})
	}
}

func TestNilAuthHandlerDoesNotEnableAuthInterception(t *testing.T) {
	t.Parallel()
	enableParams := make(chan map[string]any, 1)
	wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
		if method == "Fetch.enable" {
			enableParams <- params
		}
	})
	api := &fakeAgentCoreAPI{startOut: &bedrockagentcore.StartBrowserSessionOutput{
		BrowserIdentifier: aws.String("aws.browser.v1"),
		SessionId:         aws.String("session-1"),
		Streams:           browserStreams(wsURL),
	}}
	tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	}); err != nil {
		t.Fatalf("navigate: %v", err)
	}
	if _, ok := (<-enableParams)["handleAuthRequests"]; ok {
		t.Fatal("Fetch.enable unexpectedly enabled auth handling")
	}
}

func TestExtractTextGetsSessionAndTruncates(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds(), MaxTextBytes: 5})
	bt := tl.(*browserTool)

	out, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionExtractText,
		paramSessionID: "session-1",
		paramSelector:  "main",
	})
	if err != nil {
		t.Fatalf("extract_text: %v", err)
	}
	if out["text"] != "hello" || out["truncated"] != true {
		t.Errorf("text result = %q truncated %v", out["text"], out["truncated"])
	}
	if aws.ToString(api.lastGet.SessionId) != "session-1" {
		t.Errorf("get session id = %q", aws.ToString(api.lastGet.SessionId))
	}
}

func TestExtractTextReturnsRuntimeEvaluateException(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Runtime.evaluate.exception")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionExtractText,
		paramSessionID: "session-1",
		paramSelector:  "[",
	})
	if err == nil || !strings.Contains(err.Error(), "SyntaxError: invalid selector") {
		t.Fatalf("expected runtime exception, got %v", err)
	}
	if api.lastStop != nil {
		t.Fatal("extract_text should not stop caller-owned session")
	}
}

func TestExtractTextRejectsDisallowedCurrentURL(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Runtime.evaluate.redirect")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:          api,
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"example.com"},
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionExtractText,
		paramSessionID: "session-1",
		paramSelector:  "main",
	})
	if err == nil || !strings.Contains(err.Error(), "current url") {
		t.Fatalf("expected current URL policy error, got %v", err)
	}
	if api.lastStop != nil {
		t.Fatal("extract_text should not stop caller-owned session")
	}
}

func TestExtractTextAppliesTimeout(t *testing.T) {
	t.Parallel()
	var evaluateCalls atomic.Int32
	wsURL := fakeCDPServerWithHook(t, "", func(method string, _ map[string]any) {
		if method == "Runtime.evaluate" {
			evaluateCalls.Add(1)
			time.Sleep(500 * time.Millisecond)
		}
	})
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:               api,
		Region:            "us-east-1",
		Credentials:       testCreds(),
		NavigationTimeout: 100 * time.Millisecond,
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionExtractText,
		paramSessionID: "session-1",
	})
	if err == nil || !strings.Contains(err.Error(), "i/o timeout") {
		t.Fatalf("expected extract_text timeout, got %v", err)
	}
	if evaluateCalls.Load() != 1 {
		t.Fatalf("runtime evaluate calls = %d", evaluateCalls.Load())
	}
}

func TestNavigateStopsAutoStartedSessionOnMetadataError(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Runtime.evaluate.exception")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "SyntaxError: invalid selector") {
		t.Fatalf("expected metadata exception, got %v", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
}

func TestCleanupStartedSessionIgnoresCallerCancellation(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	const cleanupTimeout = 2 * time.Second
	tl, _ := New(Config{
		API:            api,
		Region:         "us-east-1",
		Credentials:    testCreds(),
		CleanupTimeout: cleanupTimeout,
	})
	bt := tl.(*browserTool)
	parent, cancel := context.WithCancel(context.Background())
	cancel()
	ctx := &fakeToolContext{
		StrictContextMock: agent.StrictContextMock{Ctx: parent},
		functionCallID:    "tooluse_cleanup",
	}
	cause := errors.New("navigation failed")

	err := bt.cleanupStartedSession(ctx, "session-1", cause)
	if !errors.Is(err, cause) {
		t.Fatalf("cleanup error = %v", err)
	}
	if api.stopContextErr != nil {
		t.Fatalf("cleanup inherited caller cancellation: %v", api.stopContextErr)
	}
	if api.stopDeadline.IsZero() {
		t.Fatal("cleanup context has no deadline")
	}
	if remaining := time.Until(api.stopDeadline); remaining <= 0 || remaining > cleanupTimeout {
		t.Fatalf("cleanup deadline remaining = %v", remaining)
	}
	if aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("stop session id = %q", aws.ToString(api.lastStop.SessionId))
	}
}

func TestNavigateAppliesTimeoutToMetadata(t *testing.T) {
	t.Parallel()
	var metadataCalls atomic.Int32
	wsURL := fakeCDPServerWithHook(t, "", func(method string, _ map[string]any) {
		if method == "Runtime.evaluate" {
			metadataCalls.Add(1)
			time.Sleep(500 * time.Millisecond)
		}
	})
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:               api,
		Region:            "us-east-1",
		Credentials:       testCreds(),
		NavigationTimeout: 100 * time.Millisecond,
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "i/o timeout") {
		t.Fatalf("expected metadata timeout, got %v", err)
	}
	if metadataCalls.Load() != 1 {
		t.Fatalf("metadata calls = %d", metadataCalls.Load())
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
}

func TestNavigateRejectsDisallowedFinalURL(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Runtime.evaluate.redirect")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:          api,
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"example.com"},
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "final url") {
		t.Fatalf("expected final URL policy error, got %v", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
}

func TestScreenshotRejectsDisallowedCurrentURLBeforeSaving(t *testing.T) {
	t.Parallel()
	var captureCount atomic.Int32
	wsURL := fakeCDPServerWithHook(t, "Runtime.evaluate.redirect", func(method string, _ map[string]any) {
		if method == "Page.captureScreenshot" {
			captureCount.Add(1)
		}
	})
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	arts := &fakeArtifacts{}
	tl, _ := New(Config{
		API:          api,
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"example.com"},
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(arts), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
	})
	if err == nil || !strings.Contains(err.Error(), "current url") {
		t.Fatalf("expected current URL policy error, got %v", err)
	}
	if captureCount.Load() != 0 {
		t.Fatalf("screenshot captured before URL policy check")
	}
	if arts.savedName != "" {
		t.Fatalf("artifact saved despite URL policy error: %q", arts.savedName)
	}
}

func TestScreenshotRejectsRedirectAfterCapture(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Runtime.evaluate.redirectAfterCapture")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	arts := &fakeArtifacts{}
	tl, _ := New(Config{
		API:          api,
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"example.com"},
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(arts), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
	})
	if err == nil || !strings.Contains(err.Error(), "current url after capture") {
		t.Fatalf("expected post-capture URL policy error, got %v", err)
	}
	if arts.savedName != "" {
		t.Fatalf("artifact saved after redirect: %q", arts.savedName)
	}
}

func TestScreenshotAppliesTimeout(t *testing.T) {
	t.Parallel()
	var evaluateCalls atomic.Int32
	wsURL := fakeCDPServerWithHook(t, "", func(method string, _ map[string]any) {
		if method == "Runtime.evaluate" {
			evaluateCalls.Add(1)
			time.Sleep(500 * time.Millisecond)
		}
	})
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:               api,
		Region:            "us-east-1",
		Credentials:       testCreds(),
		NavigationTimeout: 100 * time.Millisecond,
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
	})
	if err == nil || !strings.Contains(err.Error(), "i/o timeout") {
		t.Fatalf("expected screenshot timeout, got %v", err)
	}
	if evaluateCalls.Load() != 1 {
		t.Fatalf("runtime evaluate calls = %d", evaluateCalls.Load())
	}
}

func TestScreenshotSavesArtifact(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	arts := &fakeArtifacts{version: 2}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	out, err := bt.Run(newFakeToolCtx(arts), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
		paramFileName:  "page.jpg",
		paramFormat:    "jpg",
	})
	if err != nil {
		t.Fatalf("screenshot: %v", err)
	}
	if out["version"] != int64(2) {
		t.Errorf("version = %v", out["version"])
	}
	if arts.savedName != "page.jpg" {
		t.Errorf("artifact name = %q", arts.savedName)
	}
	if string(arts.savedPart.InlineData.Data) != "shot" {
		t.Errorf("artifact bytes = %q", arts.savedPart.InlineData.Data)
	}
	if arts.savedPart.InlineData.MIMEType != mimeTypeJPEG {
		t.Errorf("mime = %q", arts.savedPart.InlineData.MIMEType)
	}
	if out[paramURL] != "https://example.com/after" || out[resultKeyTitle] != "Example" {
		t.Errorf("metadata = url %v title %v", out[paramURL], out[resultKeyTitle])
	}
}

//nolint:gocognit // The table verifies option boundaries and their exact CDP parameter mapping.
func TestScreenshotOptions(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name     string
		args     map[string]any
		wantFull bool
		quality  *float64
	}{
		{
			name:     "defaults to full page",
			args:     map[string]any{paramFormat: screenshotFormatPNG},
			wantFull: true,
		},
		{
			name:     "viewport JPEG at minimum quality",
			args:     map[string]any{paramFormat: screenshotFormatJPEG, paramFullPage: false, paramQuality: float64(0)},
			wantFull: false,
			quality:  func() *float64 { value := float64(0); return &value }(),
		},
		{
			name:     "JPEG maximum quality",
			args:     map[string]any{paramFormat: screenshotFormatJPEG, paramQuality: float64(100)},
			wantFull: true,
			quality:  func() *float64 { value := float64(100); return &value }(),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			captured := make(chan map[string]any, 1)
			wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
				if method == "Page.captureScreenshot" {
					captured <- params
				}
			})
			api := &fakeAgentCoreAPI{getOut: &bedrockagentcore.GetBrowserSessionOutput{
				BrowserIdentifier: aws.String("aws.browser.v1"),
				SessionId:         aws.String("session-1"),
				Streams:           browserStreams(wsURL),
			}}
			tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
			if err != nil {
				t.Fatal(err)
			}
			args := map[string]any{paramAction: actionScreenshot, paramSessionID: "session-1"}
			maps.Copy(args, tc.args)
			if _, err := tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), args); err != nil {
				t.Fatalf("screenshot: %v", err)
			}
			params := <-captured
			if params["captureBeyondViewport"] != tc.wantFull {
				t.Errorf("captureBeyondViewport = %v", params["captureBeyondViewport"])
			}
			gotQuality, qualitySet := params[paramQuality].(float64)
			if tc.quality == nil && qualitySet {
				t.Errorf("unexpected quality = %v", gotQuality)
			}
			if tc.quality != nil && (!qualitySet || gotQuality != *tc.quality) {
				t.Errorf("quality = %v, set %v", gotQuality, qualitySet)
			}
		})
	}
}

func TestScreenshotOptionValidation(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		args map[string]any
		want string
	}{
		{name: "PNG quality", args: map[string]any{paramQuality: 80}, want: "only supported for JPEG"},
		{name: "negative quality", args: map[string]any{paramFormat: screenshotFormatJPEG, paramQuality: -1}, want: "between 0 and 100"},
		{name: "quality above maximum", args: map[string]any{paramFormat: screenshotFormatJPEG, paramQuality: 101}, want: "between 0 and 100"},
		{name: "fractional quality", args: map[string]any{paramFormat: screenshotFormatJPEG, paramQuality: 80.5}, want: "must be an integer"},
		{name: "string quality", args: map[string]any{paramFormat: screenshotFormatJPEG, paramQuality: "80"}, want: "must be an integer"},
		{name: "non boolean full page", args: map[string]any{paramFullPage: "false"}, want: "must be a boolean"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			api := &fakeAgentCoreAPI{}
			tl, err := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
			if err != nil {
				t.Fatal(err)
			}
			args := map[string]any{paramAction: actionScreenshot, paramSessionID: "session-1"}
			maps.Copy(args, tc.args)
			_, err = tl.(*browserTool).Run(newFakeToolCtx(&fakeArtifacts{}), args)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected error containing %q, got %v", tc.want, err)
			}
			if api.lastGet != nil {
				t.Fatal("browser session fetched before screenshot option validation")
			}
		})
	}
}

//nolint:gocognit // Both size-limit paths intentionally share the artifact persistence assertions.
func TestScreenshotSizeLimit(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name       string
		serverMode string
		limit      int64
		wantError  bool
	}{
		{name: "exact limit", limit: 4},
		{name: "oversized", serverMode: "Page.captureScreenshot.oversized", limit: 4, wantError: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			wsURL := fakeCDPServer(t, tc.serverMode)
			api := &fakeAgentCoreAPI{getOut: &bedrockagentcore.GetBrowserSessionOutput{
				BrowserIdentifier: aws.String("aws.browser.v1"),
				SessionId:         aws.String("session-1"),
				Streams:           browserStreams(wsURL),
			}}
			arts := &fakeArtifacts{}
			tl, err := New(Config{
				API:                api,
				Region:             "us-east-1",
				Credentials:        testCreds(),
				MaxScreenshotBytes: tc.limit,
			})
			if err != nil {
				t.Fatal(err)
			}
			_, err = tl.(*browserTool).Run(newFakeToolCtx(arts), map[string]any{
				paramAction:    actionScreenshot,
				paramSessionID: "session-1",
			})
			if tc.wantError {
				if err == nil || !strings.Contains(err.Error(), "MaxScreenshotBytes") {
					t.Fatalf("expected screenshot size error, got %v", err)
				}
				if arts.savedName != "" {
					t.Fatalf("oversized screenshot was saved as %q", arts.savedName)
				}
				return
			}
			if err != nil {
				t.Fatalf("screenshot: %v", err)
			}
			if arts.savedName == "" {
				t.Fatal("exact-limit screenshot was not saved")
			}
		})
	}
}

func TestScreenshotInfersFormatAndRejectsArtifactPaths(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)
	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
		paramFileName:  "screenshots/page.png",
	})
	if err == nil || !strings.Contains(err.Error(), "path separators") {
		t.Fatalf("expected artifact path error, got %v", err)
	}
	if api.lastGet != nil {
		t.Fatal("browser session fetched before file_name validation")
	}
	_, err = bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
		paramFileName:  "page.png",
		paramFormat:    screenshotFormatJPEG,
	})
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("expected artifact extension error, got %v", err)
	}
	_, err = bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:    actionScreenshot,
		paramSessionID: "session-1",
		paramFileName:  "page.gif",
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported extension") {
		t.Fatalf("expected unsupported artifact extension error, got %v", err)
	}
}

func TestCDPErrorIsReturned(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Page.navigate")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "fake cdp failure") {
		t.Fatalf("expected CDP failure, got %v", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
}

func TestNavigateReturnsPageErrorText(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Page.navigate.errorText")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "ERR_NAME_NOT_RESOLVED") {
		t.Fatalf("expected navigation errorText, got %v", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
}
