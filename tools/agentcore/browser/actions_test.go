package browser

import (
	"context"
	"errors"
	"maps"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/agent"
)

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
	if _, exists := startOut["session_id"]; exists {
		t.Error("start result leaked the AWS Browser session ID")
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

	statusOut, err := bt.Run(ctx, map[string]any{paramAction: actionStatus})
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
	if _, exists := statusOut["session_id"]; exists {
		t.Error("status result leaked the AWS Browser session ID")
	}
	if aws.ToString(api.lastGet.SessionId) != "session-1" {
		t.Errorf("get session id = %q", aws.ToString(api.lastGet.SessionId))
	}

	stopOut, err := bt.Run(ctx, map[string]any{paramAction: actionStop})
	if err != nil {
		t.Fatalf("stop: %v", err)
	}
	if _, exists := stopOut["session_id"]; exists {
		t.Error("stop result leaked the AWS Browser session ID")
	}
	if aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Errorf("stop session id = %q", aws.ToString(api.lastStop.SessionId))
	}
}

func TestBrowserSessionIsScopedToADKConversationAndTool(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{SessionId: aws.String("session-1")},
		getOut:   &bedrockagentcore.GetBrowserSessionOutput{Status: types.BrowserSessionStatusReady},
	}
	cfg := Config{API: api, Region: "us-east-1", Credentials: testCreds()}
	tl, _ := New(cfg)
	bt := tl.(*browserTool)
	otherTool, _ := New(cfg)
	owner := newFakeToolCtx(&fakeArtifacts{})
	if _, err := bt.Run(owner, map[string]any{paramAction: actionStart}); err != nil {
		t.Fatalf("start: %v", err)
	}
	otherApp, otherUser, otherConversation := *owner, *owner, *owner
	otherApp.appName = "other-app"
	otherUser.userID = "other-user"
	otherConversation.sessionID = "other-conversation"
	for _, other := range []*fakeToolContext{&otherApp, &otherUser, &otherConversation} {
		out, err := bt.Run(other, map[string]any{paramAction: actionStatus})
		if err != nil || out[resultKeySessionStatus] != "NO_SESSION" {
			t.Errorf("status for another ADK conversation = %v, %v", out, err)
		}
	}
	out, err := otherTool.(*browserTool).Run(owner, map[string]any{paramAction: actionStop})
	if err != nil || out[resultKeySessionStatus] != "NO_SESSION" {
		t.Errorf("stop on another tool instance = %v, %v", out, err)
	}
	if api.lastGet != nil || api.lastStop != nil {
		t.Fatal("an unrelated context used the AWS Browser session")
	}
}

func TestRunRejectsSuppliedBrowserSessionID(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)
	for _, id := range []any{"session-1", "", true, nil} {
		_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
			paramAction:  actionStop,
			"session_id": id,
		})
		if err == nil || err.Error() != "session_id is managed by the Browser tool" {
			t.Errorf("supplied session ID %v: %v", id, err)
		}
	}
	if api.lastStart != nil || api.lastGet != nil || api.lastStop != nil {
		t.Fatal("a supplied session ID caused an AgentCore API call")
	}
}

func TestFollowUpRequiresManagedBrowserSession(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)
	ctx := newFakeToolCtx(&fakeArtifacts{})
	for _, action := range []string{actionExtractText, actionScreenshot} {
		_, err := bt.Run(ctx, map[string]any{paramAction: action})
		if err == nil || !strings.Contains(err.Error(), "navigate") {
			t.Errorf("%s without an active session = %v", action, err)
		}
	}
	if api.lastStart != nil || api.lastGet != nil {
		t.Fatal("a follow-up action made an AgentCore API call without an active session")
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
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			SessionId: aws.String("session-1"),
			Status:    types.BrowserSessionStatusReady,
			Streams:   browserStreams(wsURL),
		},
		stopOut: &bedrockagentcore.StopBrowserSessionOutput{},
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
	if _, exists := out["session_id"]; exists {
		t.Error("navigate result leaked the AWS Browser session ID")
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
	api.lastStart = nil
	arts := &fakeArtifacts{}
	for _, action := range []string{actionExtractText, actionScreenshot, actionStop} {
		ctx := newFakeToolCtx(arts)
		out, err := bt.Run(ctx, map[string]any{paramAction: action})
		if err != nil {
			t.Fatalf("%s with a new ADK tool context: %v", action, err)
		}
		if _, exists := out["session_id"]; exists {
			t.Errorf("%s result leaked the AWS Browser session ID", action)
		}
	}
	if api.lastStart != nil || api.lastGet == nil || api.lastStop == nil ||
		aws.ToString(api.lastGet.SessionId) != "session-1" ||
		aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatal("follow-up actions did not reuse the navigation session")
	}
	if arts.savedName == "" {
		t.Fatal("the screenshot follow-up did not save an artifact")
	}
	out, err = bt.Run(newFakeToolCtx(arts), map[string]any{paramAction: actionStatus})
	if err != nil || out[resultKeySessionStatus] != "NO_SESSION" {
		t.Errorf("status after stop = %v, %v", out, err)
	}
}

func TestNavigateRejectsInvalidWaitOverrideBeforeStartingSession(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, err := New(
		Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}},
	)
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
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			api := &fakeAgentCoreAPI{}
			tl, err := New(
				Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}},
			)
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
			name: "navigate URL",
			key:  paramURL,
			args: map[string]any{paramAction: actionNavigate, paramURL: true},
		},
		{
			name: "extract selector",
			key:  paramSelector,
			args: map[string]any{
				paramAction:   actionExtractText,
				paramSelector: true,
			},
		},
		{
			name: "screenshot file name",
			key:  paramFileName,
			args: map[string]any{
				paramAction:   actionScreenshot,
				paramFileName: true,
			},
		},
		{
			name: "screenshot format",
			key:  paramFormat,
			args: map[string]any{
				paramAction: actionScreenshot,
				paramFormat: true,
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
		Status:            types.BrowserSessionStatusReady,
		Streams:           browserStreams(wsURL),
	}}
	tl, err := New(
		Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}},
	)
	if err != nil {
		t.Fatal(err)
	}
	bt := tl.(*browserTool)
	_, err = bt.Run(fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{}), map[string]any{
		paramAction:          actionExtractText,
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

func TestExtractTextGetsSessionAndTruncates(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Status:            types.BrowserSessionStatusReady,
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(
		Config{
			API:          api,
			Region:       "us-east-1",
			Credentials:  testCreds(),
			MaxTextBytes: 5,
			AllowedHosts: []string{"example.com"},
		},
	)
	bt := tl.(*browserTool)

	out, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{}), map[string]any{
		paramAction:   actionExtractText,
		paramSelector: "main",
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

func TestExtractTextRejectsDisallowedCurrentURL(t *testing.T) {
	t.Parallel()
	wsURL := fakeCDPServer(t, "Runtime.evaluate.redirect")
	api := &fakeAgentCoreAPI{
		getOut: &bedrockagentcore.GetBrowserSessionOutput{
			BrowserIdentifier: aws.String("aws.browser.v1"),
			SessionId:         aws.String("session-1"),
			Status:            types.BrowserSessionStatusReady,
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

	_, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{}), map[string]any{
		paramAction:   actionExtractText,
		paramSelector: "main",
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
			Status:            types.BrowserSessionStatusReady,
			Streams:           browserStreams(wsURL),
		},
	}
	tl, _ := New(Config{
		API:               api,
		Region:            "us-east-1",
		Credentials:       testCreds(),
		AllowedHosts:      []string{"example.com"},
		NavigationTimeout: 100 * time.Millisecond,
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{}), map[string]any{
		paramAction: actionExtractText,
	})
	if !errors.Is(err, context.DeadlineExceeded) {
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
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}})
	bt := tl.(*browserTool)
	ctx := newFakeToolCtx(&fakeArtifacts{})
	ctx.functionCallID = "tooluse_cleanup"

	_, err := bt.Run(ctx, map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if err == nil || !strings.Contains(err.Error(), "SyntaxError: invalid selector") {
		t.Fatalf("expected metadata exception, got %v", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("auto-started session was not stopped: %#v", api.lastStop)
	}
	if aws.ToString(api.lastStop.ClientToken) == aws.ToString(api.lastStart.ClientToken) {
		t.Fatal("cleanup reused the start client token")
	}
	out, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramAction: actionStatus})
	if err != nil || out[resultKeySessionStatus] != "NO_SESSION" || api.lastGet != nil {
		t.Errorf("status after cleanup = %v, %v", out, err)
	}
}

func TestStopRetriesFailedNavigationCleanup(t *testing.T) {
	t.Parallel()
	cleanupErr := errors.New("temporary stop failure")
	wsURL := fakeCDPServer(t, "Runtime.evaluate.exception")
	api := &fakeAgentCoreAPI{
		startOut: &bedrockagentcore.StartBrowserSessionOutput{
			SessionId: aws.String("session-1"),
			Streams:   browserStreams(wsURL),
		},
		stopOut: &bedrockagentcore.StopBrowserSessionOutput{},
		stopErr: cleanupErr,
	}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}})
	bt := tl.(*browserTool)
	ctx := newFakeToolCtx(&fakeArtifacts{})
	_, err := bt.Run(ctx, map[string]any{paramAction: actionNavigate, paramURL: "https://example.com"})
	if !errors.Is(err, cleanupErr) {
		t.Fatalf("navigate error = %v", err)
	}
	api.stopErr = nil
	api.lastStop = nil
	if _, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramAction: actionStop}); err != nil {
		t.Fatalf("retry stop: %v", err)
	}
	if api.lastStop == nil || aws.ToString(api.lastStop.SessionId) != "session-1" {
		t.Fatalf("retry did not stop the tracked session: %v", api.lastStop)
	}
}

func TestNavigateReplacesStaleBrowserSession(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name   string
		getErr error
	}{
		{name: "terminated"},
		{name: "resource not found", getErr: &types.ResourceNotFoundException{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			wsURL := fakeCDPServer(t, "")
			api := &fakeAgentCoreAPI{
				getOut: &bedrockagentcore.GetBrowserSessionOutput{Status: types.BrowserSessionStatusTerminated},
				getErr: tc.getErr,
				startOut: &bedrockagentcore.StartBrowserSessionOutput{
					SessionId: aws.String("session-2"),
					Streams:   browserStreams(wsURL),
				},
			}
			tl, _ := New(Config{
				API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"},
			})
			bt := tl.(*browserTool)
			ctx := fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{})
			out, err := bt.Run(ctx, map[string]any{paramAction: actionNavigate, paramURL: "https://example.com"})
			if err != nil || out[resultKeyStatus] != statusSuccess {
				t.Fatalf("navigate with stale session = %v, %v", out, err)
			}
			if api.lastStart == nil || api.lastGet == nil || aws.ToString(api.lastGet.SessionId) != "session-1" {
				t.Fatal("navigation did not replace the stale session")
			}
		})
	}
}

func TestRunRejectsOverlappingActionsForConversation(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds()})
	bt := tl.(*browserTool)
	_, release, err := bt.acquireSession(newFakeToolCtx(&fakeArtifacts{}))
	if err != nil {
		t.Fatal(err)
	}
	defer release()

	_, err = bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramAction: actionStart})
	if err == nil || !strings.Contains(err.Error(), "Browser action is already in progress") {
		t.Fatalf("overlapping start = %v", err)
	}
	if api.lastStart != nil {
		t.Fatal("overlapping action created another Browser session")
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

	err := bt.cleanupStartedSession(ctx, &browserSession{id: "session-1"}, cause)
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
		AllowedHosts:      []string{"example.com"},
		NavigationTimeout: 100 * time.Millisecond,
	})
	bt := tl.(*browserTool)

	_, err := bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction: actionNavigate,
		paramURL:    "https://example.com",
	})
	if !errors.Is(err, context.DeadlineExceeded) {
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
			Status:            types.BrowserSessionStatusReady,
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

	_, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, arts), map[string]any{
		paramAction: actionScreenshot,
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
			Status:            types.BrowserSessionStatusReady,
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

	_, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{}), map[string]any{
		paramAction: actionScreenshot,
	})
	if !errors.Is(err, context.DeadlineExceeded) {
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
			Status:            types.BrowserSessionStatusReady,
			Streams:           browserStreams(wsURL),
		},
	}
	arts := &fakeArtifacts{version: 2}
	tl, _ := New(Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}})
	bt := tl.(*browserTool)

	out, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, arts), map[string]any{
		paramAction:   actionScreenshot,
		paramFileName: "page.jpg",
		paramFormat:   "jpg",
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
				Status:            types.BrowserSessionStatusReady,
				Streams:           browserStreams(wsURL),
			}}
			tl, err := New(
				Config{API: api, Region: "us-east-1", Credentials: testCreds(), AllowedHosts: []string{"example.com"}},
			)
			if err != nil {
				t.Fatal(err)
			}
			args := map[string]any{paramAction: actionScreenshot}
			maps.Copy(args, tc.args)
			bt := tl.(*browserTool)
			if _, err := bt.Run(fakeToolCtxWithBrowserSession(t, bt, &fakeArtifacts{}), args); err != nil {
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
			args := map[string]any{paramAction: actionScreenshot}
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
				Status:            types.BrowserSessionStatusReady,
				Streams:           browserStreams(wsURL),
			}}
			arts := &fakeArtifacts{}
			tl, err := New(Config{
				API:                api,
				Region:             "us-east-1",
				Credentials:        testCreds(),
				AllowedHosts:       []string{"example.com"},
				MaxScreenshotBytes: tc.limit,
			})
			if err != nil {
				t.Fatal(err)
			}
			bt := tl.(*browserTool)
			_, err = bt.Run(fakeToolCtxWithBrowserSession(t, bt, arts), map[string]any{
				paramAction: actionScreenshot,
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
		paramAction:   actionScreenshot,
		paramFileName: "screenshots/page.png",
	})
	if err == nil || !strings.Contains(err.Error(), "path separators") {
		t.Fatalf("expected artifact path error, got %v", err)
	}
	if api.lastGet != nil {
		t.Fatal("browser session fetched before file_name validation")
	}
	_, err = bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:   actionScreenshot,
		paramFileName: "page.png",
		paramFormat:   screenshotFormatJPEG,
	})
	if err == nil || !strings.Contains(err.Error(), "does not match") {
		t.Fatalf("expected artifact extension error, got %v", err)
	}
	_, err = bt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramAction:   actionScreenshot,
		paramFileName: "page.gif",
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported extension") {
		t.Fatalf("expected unsupported artifact extension error, got %v", err)
	}
}
