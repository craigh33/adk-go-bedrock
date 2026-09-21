package agentcorebrowser

import (
	"bytes"
	"context"
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
	"github.com/gorilla/websocket"
)

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
