package browser

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/coder/websocket"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/genai"
)

type browsePageTestContext struct {
	*fakeToolContext
}

func (c browsePageTestContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }

type browsePageTestTool interface {
	Name() string
	Declaration() *genai.FunctionDeclaration
	Run(agent.Context, any) (map[string]any, error)
}

func newBrowsePageTestTool(t *testing.T, api *fakeAgentCoreAPI, dialer WebSocketDialer) browsePageTestTool {
	t.Helper()
	tool, err := NewPage(Config{
		API:            api,
		Region:         "us-east-1",
		Credentials:    testCreds(),
		AllowedHosts:   []string{"example.com"},
		CleanupTimeout: time.Minute,
		Dialer:         dialer,
	})
	if err != nil {
		t.Fatal(err)
	}
	return tool.(browsePageTestTool)
}

func TestBrowsePageDeclaration(t *testing.T) {
	t.Parallel()
	tool := newBrowsePageTestTool(t, &fakeAgentCoreAPI{}, nil)
	if tool.Name() != "browse_page" {
		t.Fatalf("tool name = %q", tool.Name())
	}
	encoded, err := json.Marshal(tool.Declaration().ParametersJsonSchema)
	if err != nil {
		t.Fatal(err)
	}
	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required"`
	}
	if err := json.Unmarshal(encoded, &schema); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(schema.Required, []string{"url"}) {
		t.Errorf("required arguments = %v", schema.Required)
	}
	for _, param := range []string{"url", "selector", "screenshot"} {
		if _, ok := schema.Properties[param]; !ok {
			t.Errorf("schema missing %q", param)
		}
	}
	if len(schema.Properties) != 3 {
		t.Errorf("unexpected argument properties: %s", encoded)
	}
}

func TestBrowsePageReadsAndCapturesInOneSession(t *testing.T) {
	t.Parallel()
	var attaches atomic.Int32
	var selected atomic.Bool
	wsURL := fakeCDPServerWithHook(t, "", func(method string, params map[string]any) {
		if method == "Target.attachToTarget" {
			attaches.Add(1)
		}
		if expr, ok := params[cdpKeyExpression].(string); ok &&
			strings.Contains(expr, `const selector = "#article";`) &&
			strings.Contains(expr, "document.body") {
			selected.Store(true)
		}
	})
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			SessionId: aws.String("session-1"),
			Streams:   browserStreams(wsURL),
		},
	}
	const version = 2
	arts := &fakeArtifacts{version: version}
	tool := newBrowsePageTestTool(t, api, nil)
	out, err := tool.Run(browsePageTestContext{newFakeToolCtx(arts)}, map[string]any{
		"url":        "https://example.com",
		"selector":   "#article",
		"screenshot": "page.jpg",
	})
	if err != nil {
		t.Fatalf("browse_page: %v", err)
	}
	if out["status"] != "success" || out["url"] != "https://example.com/after" || out["title"] != "Example" ||
		out["text"] != "hello world" || out["truncated"] != false || !selected.Load() {
		t.Errorf("unexpected page result: %v (selected = %v)", out, selected.Load())
	}
	wantScreenshot := map[string]any{"file_name": "page.jpg", "version": float64(version), "mime_type": mimeTypeJPEG}
	if !reflect.DeepEqual(out["screenshot"], wantScreenshot) {
		t.Errorf("screenshot = %v, want %v", out["screenshot"], wantScreenshot)
	}
	if arts.savedName != "page.jpg" || arts.savedPart == nil || arts.savedPart.InlineData == nil ||
		string(arts.savedPart.InlineData.Data) != "shot" || arts.savedPart.InlineData.MIMEType != mimeTypeJPEG {
		t.Errorf("saved artifact = %q, %+v", arts.savedName, arts.savedPart)
	}
	if _, hasSessionID := out["session_id"]; hasSessionID {
		t.Errorf("result exposes a session ID: %v", out)
	}
	if api.lastStart == nil || api.lastGet != nil || api.lastStop == nil ||
		aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Errorf("unexpected session calls: start=%v get=%v stop=%v", api.lastStart, api.lastGet, api.lastStop)
	}
	if got := attaches.Load(); got != 1 {
		t.Errorf("CDP attachments = %d, want 1", got)
	}
}

func TestBrowsePageCleansUpAfterCancellation(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stopErr := errors.New("stop failed")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			SessionId: aws.String("session-1"),
			Streams:   browserStreams("wss://browser.example.com"),
		},
		stopErr: stopErr,
	}
	dialer := dialerFunc(
		func(context.Context, string, *websocket.DialOptions) (*websocket.Conn, *http.Response, error) {
			cancel()
			return nil, nil, context.Canceled
		},
	)
	tool := newBrowsePageTestTool(t, api, dialer)
	toolCtx := newFakeToolCtx(nil)
	toolCtx.StrictContextMock.Ctx = ctx
	_, err := tool.Run(browsePageTestContext{toolCtx}, map[string]any{"url": "https://example.com"})
	if !errors.Is(err, context.Canceled) || !errors.Is(err, stopErr) {
		t.Fatalf("browse_page error = %v, want both cancellation and cleanup failure", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("stop did not receive the session started by browse_page: %v", api.lastStop)
	}
	if api.stopContextErr != nil || api.stopDeadline.IsZero() || time.Until(api.stopDeadline) > time.Minute {
		t.Errorf("invalid cleanup context: error=%v deadline=%v", api.stopContextErr, api.stopDeadline)
	}
}

func TestBrowsePageRejectsDisallowedURLBeforeStarting(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tool := newBrowsePageTestTool(t, api, nil)
	_, err := tool.Run(browsePageTestContext{newFakeToolCtx(nil)}, map[string]any{"url": "https://blocked.example.org"})
	if err == nil || !strings.Contains(err.Error(), "is not allowed") {
		t.Fatalf("browse_page error = %v, want host policy rejection", err)
	}
	if api.lastStart != nil || api.lastStop != nil {
		t.Errorf("invalid URL touched Browser sessions: start=%v stop=%v", api.lastStart, api.lastStop)
	}
}
