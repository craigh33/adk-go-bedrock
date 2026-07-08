package agentcorebrowser

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
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
	_ context.Context,
	in *bedrockagentcore.StopBrowserSessionInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.StopBrowserSessionOutput, error) {
	f.lastStop = in
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
	t.Helper()
	upgrader := websocket.Upgrader{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("upgrade: %v", err)
			return
		}
		defer conn.Close()
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
			case "Page.enable":
				_ = conn.WriteJSON(map[string]any{"id": req.ID, "result": map[string]any{}})
			case "Page.navigate":
				_ = conn.WriteJSON(map[string]any{"method": "Page.loadEventFired", "sessionId": "session-1"})
				_ = conn.WriteJSON(map[string]any{"id": req.ID, "result": map[string]any{"frameId": "frame-1"}})
			case "Runtime.evaluate":
				value := map[string]any{"url": "https://example.com/after", "title": "Example"}
				if expr, _ := req.Params["expression"].(string); strings.Contains(expr, "document.querySelector") {
					value["text"] = "hello world"
				}
				_ = conn.WriteJSON(map[string]any{
					"id": req.ID,
					"result": map[string]any{
						"result": map[string]any{"value": value},
					},
				})
			case "Page.captureScreenshot":
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

func TestStartStatusStopUseAgentCoreAPI(t *testing.T) {
	t.Parallel()
	now := time.Now()
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			CreatedAt:         &now,
		},
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier:     aws.String("aws.browser.v1"),
			SessionId:             aws.String("session-1"),
			Status:                types.BrowserSessionStatusReady,
			SessionTimeoutSeconds: aws.Int32(900),
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

	if _, err := bt.Run(ctx, map[string]any{paramAction: actionStart}); err != nil {
		t.Fatalf("start: %v", err)
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

	if out, err := bt.Run(ctx, map[string]any{paramAction: actionStatus, paramSessionID: "session-1"}); err != nil {
		t.Fatalf("status: %v", err)
	} else if out[resultKeyStatus] != statusSuccess || out[resultKeySessionStatus] != "READY" {
		t.Errorf("status result = %v session_status = %v", out[resultKeyStatus], out[resultKeySessionStatus])
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

func TestNavigateStartsSessionAndUsesCDP(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "")
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
	if arts.savedPart.InlineData.MIMEType != "image/jpeg" {
		t.Errorf("mime = %q", arts.savedPart.InlineData.MIMEType)
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
