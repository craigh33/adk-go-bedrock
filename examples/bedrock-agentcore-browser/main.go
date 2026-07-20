// Bedrock AgentCore Browser example for adk-go. Set AWS_REGION and credentials.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
	"github.com/craigh33/adk-go-bedrock/tools/agentcorebrowser"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return fmt.Errorf("load AWS config: %w", err)
	}
	if awsCfg.Region == "" {
		return errors.New("AWS region is unset: set AWS_REGION or add a region to your AWS profile")
	}

	modelID := strings.TrimSpace(os.Getenv("BEDROCK_MODEL_ID"))
	if modelID == "" {
		modelID = "us.amazon.nova-2-lite-v1:0"
	}
	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := converse.NewWithAPI(modelID, converse.NewRuntimeAPI(br))
	if err != nil {
		return fmt.Errorf("bedrock model: %w", err)
	}

	browserCfg, err := browserConfigFromEnv(awsCfg)
	if err != nil {
		return err
	}
	browserTool, err := agentcorebrowser.New(browserCfg)
	if err != nil {
		return fmt.Errorf("browser tool: %w", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "agentcore-browser-assistant",
		Description: "An assistant that can browse constrained websites with Bedrock AgentCore Browser",
		Model:       llm,
		Instruction: `Use agentcore_browser when the user asks to browse a website, get visible page text, or capture a screenshot.
Navigate first, keep the returned session_id for follow-up extract_text or screenshot calls, and stop the session when finished.`,
		Tools: []tool.Tool{browserTool},
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		return fmt.Errorf("agent: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-agentcore-browser-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		ArtifactService:   artifact.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return fmt.Errorf("runner: %w", err)
	}

	userMsg := "Open https://example.com, extract the visible text, save a screenshot as example.png, then stop the browser session."
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)
	return printRunEvents(ctx, r, a, userMsg)
}

func browserConfigFromEnv(awsCfg aws.Config) (agentcorebrowser.Config, error) {
	sessionTimeout, err := int32Env("AGENTCORE_BROWSER_SESSION_TIMEOUT_SECONDS")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	viewportWidth, err := int32Env("AGENTCORE_BROWSER_VIEWPORT_WIDTH")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	viewportHeight, err := int32Env("AGENTCORE_BROWSER_VIEWPORT_HEIGHT")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	maxTextBytes, err := intEnv("AGENTCORE_BROWSER_MAX_TEXT_BYTES")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	maxScreenshotBytes, err := int64Env("AGENTCORE_BROWSER_MAX_SCREENSHOT_BYTES")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	navigationTimeout, err := durationEnv("AGENTCORE_BROWSER_NAVIGATION_TIMEOUT")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	cleanupTimeout, err := durationEnv("AGENTCORE_BROWSER_CLEANUP_TIMEOUT")
	if err != nil {
		return agentcorebrowser.Config{}, err
	}
	return agentcorebrowser.Config{
		API:                   bedrockagentcore.NewFromConfig(awsCfg),
		Region:                awsCfg.Region,
		Credentials:           awsCfg.Credentials,
		BrowserIdentifier:     strings.TrimSpace(os.Getenv("AGENTCORE_BROWSER_ID")),
		SessionTimeoutSeconds: sessionTimeout,
		ViewportWidth:         viewportWidth,
		ViewportHeight:        viewportHeight,
		AllowedHosts:          csvEnv("AGENTCORE_BROWSER_ALLOWED_HOSTS"),
		DeniedHosts:           csvEnv("AGENTCORE_BROWSER_DENIED_HOSTS"),
		NavigationTimeout:     navigationTimeout,
		CleanupTimeout:        cleanupTimeout,
		MaxTextBytes:          maxTextBytes,
		MaxScreenshotBytes:    maxScreenshotBytes,
		WaitUntil: agentcorebrowser.WaitUntil(
			strings.TrimSpace(os.Getenv("AGENTCORE_BROWSER_WAIT_UNTIL")),
		),
	}, nil
}

func int32Env(name string) (int32, error) {
	value, err := int64EnvWithSize(name, 32)
	if err != nil {
		return 0, err
	}
	return int32(value), nil //nolint:gosec // ParseInt enforces the signed 32-bit range.
}

func intEnv(name string) (int, error) {
	value, err := int64EnvWithSize(name, strconv.IntSize)
	if err != nil {
		return 0, err
	}
	return int(value), nil
}

func int64Env(name string) (int64, error) {
	return int64EnvWithSize(name, 64)
}

func int64EnvWithSize(name string, bitSize int) (int64, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, nil
	}
	value, err := strconv.ParseInt(raw, 10, bitSize)
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer: %w", name, err)
	}
	return value, nil
}

func durationEnv(name string) (time.Duration, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, nil
	}
	value, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("%s must be a duration: %w", name, err)
	}
	return value, nil
}

func printRunEvents(ctx context.Context, r *runner.Runner, a agent.Agent, userMsg string) error {
	for ev, err := range r.Run(ctx, "local-user", "demo-session", genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
		if ev.Author != a.Name() || ev.LLMResponse.Partial || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				fmt.Printf("Tool call: %s(%v)\n", p.FunctionCall.Name, p.FunctionCall.Args)
			}
			if p.FunctionResponse != nil {
				fmt.Printf("Tool result: %v\n", p.FunctionResponse.Response)
			}
			if p.Text != "" {
				fmt.Print(p.Text)
			}
		}
		fmt.Println()
	}
	return nil
}

func csvEnv(name string) []string {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return nil
	}
	var out []string
	for part := range strings.SplitSeq(raw, ",") {
		if s := strings.TrimSpace(part); s != "" {
			out = append(out, s)
		}
	}
	return out
}
