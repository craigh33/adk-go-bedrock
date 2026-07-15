package agentcorebrowser

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
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

//nolint:gocognit // Keeping the fake CDP request table in one place is clearer for these tests.
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
			case "Fetch.enable", "Fetch.continueRequest", "Fetch.failRequest", "Fetch.fulfillRequest", "Page.enable":
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
				_ = conn.WriteJSON(map[string]any{"method": "Page.loadEventFired", "sessionId": "session-1"})
				_ = conn.WriteJSON(map[string]any{"id": req.ID, "result": map[string]any{"frameId": "frame-1"}})
			case "Runtime.evaluate":
				if failMethod == "Runtime.evaluate.exception" {
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
				value := map[string]any{"url": "https://example.com/after", "title": "Example"}
				if failMethod == "Runtime.evaluate.redirect" ||
					(failMethod == "Runtime.evaluate.redirectAfterCapture" && captured) {
					value["url"] = "https://blocked.example.net/after"
				}
				if expr, _ := req.Params["expression"].(string); strings.Contains(expr, "document.querySelector") {
					value["text"] = "hello world"
					if strings.Contains(expr, "const maxBytes = 5;") {
						value["text"] = "hello"
						value["truncated"] = true
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
				_ = conn.WriteJSON(map[string]any{
					"id":     req.ID,
					"result": map[string]any{"data": base64.StdEncoding.EncodeToString([]byte("shot"))},
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
		len(got) < minClientTokenLen {
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

	if err := bt.checkURL("https://sub.example.com/path"); err != nil {
		t.Fatalf("allowed subdomain rejected: %v", err)
	}
	if err := bt.checkURL("https://bad.example.com"); err == nil {
		t.Fatal("expected denied host error")
	}
	if err := bt.checkURL("https://other.test"); err == nil {
		t.Fatal("expected not allowed host error")
	}
	if err := bt.checkURL("file:///etc/passwd"); err == nil {
		t.Fatal("expected scheme error")
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
		if err := bt.checkURL(rawURL); err == nil {
			t.Fatalf("expected local target rejection for %s", rawURL)
		}
	}
	if err := bt.checkURL("https://93.184.216.34"); err != nil {
		t.Fatalf("public IP rejected: %v", err)
	}

	tl, _ = New(Config{
		API:          &fakeAgentCoreAPI{},
		Region:       "us-east-1",
		Credentials:  testCreds(),
		AllowedHosts: []string{"localhost", "127.0.0.1"},
	})
	bt = tl.(*browserTool)
	if err := bt.checkURL("https://localhost"); err != nil {
		t.Fatalf("explicit localhost allow rejected: %v", err)
	}
	if err := bt.checkURL("https://127.0.0.1"); err != nil {
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
		if err := bt.checkURL(rawURL); err == nil {
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
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
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
	if remaining := time.Until(api.stopDeadline); remaining <= 0 || remaining > defaultCleanupTimeout {
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

func TestScreenshotInfersFormatAndRejectsArtifactPaths(t *testing.T) {
	t.Parallel()
	format, mimeType, err := screenshotFormat(map[string]any{paramFileName: "page.JPG"})
	if err != nil || format != screenshotFormatJPEG || mimeType != mimeTypeJPEG {
		t.Fatalf("inferred screenshot format = %q %q, err %v", format, mimeType, err)
	}
	if err := validateScreenshotFileName("page", screenshotFormatPNG); err != nil {
		t.Fatalf("extensionless artifact name: %v", err)
	}
	api := &fakeAgentCoreAPI{}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)
	_, err = bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
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

func TestTruncateUTF8KeepsValidString(t *testing.T) {
	t.Parallel()
	got, truncated := truncateUTF8("héllo", 2)
	if got != "h" || !truncated {
		t.Fatalf("truncateUTF8 = %q %v", got, truncated)
	}
}

func TestClientTokenMeetsAgentCoreShape(t *testing.T) {
	t.Parallel()
	got := clientToken("tooluse_abc")
	if !strings.HasPrefix(got, "tooluse-abc-") {
		t.Fatalf("clientToken prefix = %q", got)
	}
	if len(got) < minClientTokenLen || len(got) > maxClientTokenLen {
		t.Fatalf("clientToken length = %d", len(got))
	}
}
