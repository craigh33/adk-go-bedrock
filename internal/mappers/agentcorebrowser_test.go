package mappers

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
)

func TestAgentCoreBrowserSessionInputs(t *testing.T) {
	t.Parallel()
	start := AgentCoreBrowserStartInput(AgentCoreBrowserStartParams{
		BrowserIdentifier:     "browser-id",
		FunctionCallID:        "tooluse_abc",
		SessionTimeoutSeconds: 60,
		ViewportWidth:         1280,
		ViewportHeight:        720,
	})
	if aws.ToString(start.BrowserIdentifier) != "browser-id" || aws.ToString(start.Name) != "adk-browser" {
		t.Fatalf("start input identity = %#v", start)
	}
	if token := aws.ToString(start.ClientToken); !strings.HasPrefix(token, "tooluse-abc-") || len(token) < 33 {
		t.Fatalf("start client token = %q", token)
	}
	if aws.ToInt32(start.SessionTimeoutSeconds) != 60 || start.ViewPort == nil ||
		aws.ToInt32(start.ViewPort.Width) != 1280 || aws.ToInt32(start.ViewPort.Height) != 720 {
		t.Fatalf("start input settings = %#v", start)
	}

	get := AgentCoreBrowserGetInput("browser-id", "session-id")
	if aws.ToString(get.BrowserIdentifier) != "browser-id" || aws.ToString(get.SessionId) != "session-id" {
		t.Fatalf("get input = %#v", get)
	}
	stop := AgentCoreBrowserStopInput("browser-id", "session-id", "tooluse_abc")
	if aws.ToString(stop.BrowserIdentifier) != "browser-id" || aws.ToString(stop.SessionId) != "session-id" ||
		aws.ToString(stop.ClientToken) != aws.ToString(start.ClientToken) {
		t.Fatalf("stop input = %#v", stop)
	}
}

func TestAgentCoreBrowserStreamValues(t *testing.T) {
	t.Parallel()
	streams := &types.BrowserSessionStream{
		AutomationStream: &types.AutomationStream{StreamEndpoint: aws.String("wss://automation.example")},
		LiveViewStream:   &types.LiveViewStream{StreamEndpoint: aws.String("https://live.example")},
	}
	if got := AgentCoreBrowserAutomationEndpoint(streams); got != "wss://automation.example" {
		t.Fatalf("automation endpoint = %q", got)
	}
	if got := AgentCoreBrowserLiveViewEndpoint(streams); got != "https://live.example" {
		t.Fatalf("live view endpoint = %q", got)
	}
	if AgentCoreBrowserAutomationEndpoint(nil) != "" || AgentCoreBrowserLiveViewEndpoint(nil) != "" {
		t.Fatal("nil streams should map to empty endpoints")
	}
	now := time.Date(2026, time.July, 16, 12, 30, 0, 123, time.UTC)
	if got := AgentCoreBrowserTimeValue(&now); got != "2026-07-16T12:30:00.000000123Z" {
		t.Fatalf("time value = %q", got)
	}
	value := int32(42)
	if AgentCoreBrowserInt32Value(&value) != 42 || AgentCoreBrowserInt32Value(nil) != 0 {
		t.Fatal("unexpected optional int mapping")
	}
}

func TestAgentCoreBrowserScreenshotMapping(t *testing.T) {
	t.Parallel()
	format, mimeType, err := AgentCoreBrowserScreenshotFormat("", "page.JPG")
	if err != nil || format != "jpeg" || mimeType != "image/jpeg" {
		t.Fatalf("inferred screenshot format = %q %q, err %v", format, mimeType, err)
	}
	if err := AgentCoreBrowserValidateScreenshotFileName("page", "png"); err != nil {
		t.Fatalf("extensionless artifact name: %v", err)
	}
	if err := AgentCoreBrowserValidateScreenshotFileName("page.png", "jpeg"); err == nil {
		t.Fatal("expected mismatched extension error")
	}
	if _, _, err := AgentCoreBrowserScreenshotFormat("webp", ""); err == nil {
		t.Fatal("expected unsupported format error")
	}
}

func TestAgentCoreBrowserHostMapping(t *testing.T) {
	t.Parallel()
	for _, host := range []string{"*", "example.*", "*.*.example.com"} {
		if _, err := AgentCoreBrowserNormalizeHosts("AllowedHosts", []string{host}); err == nil {
			t.Errorf("expected invalid wildcard error for %q", host)
		}
	}
	normalized, err := AgentCoreBrowserNormalizeHosts("AllowedHosts", []string{"*.Example.COM."})
	if err != nil {
		t.Fatalf("normalize valid wildcard: %v", err)
	}
	if len(normalized) != 1 || normalized[0] != "example.com" {
		t.Fatalf("normalized hosts = %v", normalized)
	}
	if !AgentCoreBrowserHostMatches(normalized, "sub.example.com") {
		t.Fatal("expected subdomain match")
	}
	for _, host := range []string{"localhost", "127.0.0.1", "0x7f.0.0.1", "2001:db8::1"} {
		if !AgentCoreBrowserRequiresExplicitAllow(host) {
			t.Errorf("expected %q to require an explicit allow", host)
		}
	}
	if AgentCoreBrowserRequiresExplicitAllow("8.8.8.8") {
		t.Fatal("public IP unexpectedly requires an explicit allow")
	}
}

func TestAgentCoreBrowserResponseLimits(t *testing.T) {
	t.Parallel()
	limit, err := AgentCoreBrowserAutomationReadLimit(16<<20, 64<<10)
	if err != nil || limit <= 16<<20 {
		t.Fatalf("automation read limit = %d, err %v", limit, err)
	}
	if _, err := AgentCoreBrowserAutomationReadLimit(math.MaxInt64, 1); err == nil {
		t.Fatal("expected oversized screenshot limit error")
	}
	got, truncated := AgentCoreBrowserTruncateUTF8("héllo", 2)
	if got != "h" || !truncated {
		t.Fatalf("truncate UTF-8 = %q %v", got, truncated)
	}
}
