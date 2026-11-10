// Bedrock AgentCore session example for adk-go: set BEDROCK_MODEL_ID and
// authenticate with AWS using the default credential chain. ADK_USER_ID sets
// the ADK session user. Optionally set AGENTCORE_SESSION_ID to resume a session.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/agentcore/session/agentcoresession"
	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

const appName = "bedrock-agentcore-session-example"

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	awsCfg, err := loadAWSConfig(ctx)
	if err != nil {
		return err
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		return fmt.Errorf("tracer provider: %w", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	llm, err := newModel(awsCfg, tp)
	if err != nil {
		return err
	}
	sessionService, err := agentcoresession.NewWithAPI(
		bedrockagentruntime.NewFromConfig(awsCfg),
		&agentcoresession.Options{EncryptionKeyARN: strings.TrimSpace(os.Getenv("AGENTCORE_SESSION_KMS_KEY_ARN"))},
	)
	if err != nil {
		return fmt.Errorf("bedrock session service: %w", err)
	}

	userID := envOrDefault("ADK_USER_ID", "local-user")
	sessionID, err := ensureSession(ctx, sessionService, userID)
	if err != nil {
		return err
	}

	a, err := newAgent(llm)
	if err != nil {
		return err
	}
	r, err := runner.New(runner.Config{AppName: appName, Agent: a, SessionService: sessionService})
	if err != nil {
		return fmt.Errorf("runner: %w", err)
	}

	if err := runPrompt(ctx, r, a, userID, sessionID); err != nil {
		return err
	}
	return printStoredSession(ctx, sessionService, userID, sessionID)
}

func loadAWSConfig(ctx context.Context) (aws.Config, error) {
	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return aws.Config{}, fmt.Errorf("load AWS config: %w", err)
	}
	if awsCfg.Region == "" {
		return aws.Config{}, errors.New("AWS region is unset: set AWS_REGION or configure a profile region")
	}
	return awsCfg, nil
}

func newModel(awsCfg aws.Config, tp trace.TracerProvider) (model.LLM, error) {
	modelID := envOrDefault("BEDROCK_MODEL_ID", "eu.amazon.nova-2-lite-v1:0")
	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := converse.NewWithAPI(modelID, converse.NewRuntimeAPI(br, converse.WithTracerProvider(tp)))
	if err != nil {
		return nil, fmt.Errorf("bedrock model: %w", err)
	}
	return llm, nil
}

func ensureSession(ctx context.Context, sessionService session.Service, userID string) (string, error) {
	sessionID := strings.TrimSpace(os.Getenv("AGENTCORE_SESSION_ID"))
	if sessionID == "" {
		created, err := sessionService.Create(ctx, &session.CreateRequest{
			AppName: appName,
			UserID:  userID,
			State: map[string]any{
				"app:session_backend": "agentcore",
			},
		})
		if err != nil {
			return "", fmt.Errorf("create session: %w", err)
		}
		sessionID = created.Session.ID()
		fmt.Printf("Created AgentCore session: %s\n", sessionID)
		return sessionID, nil
	}

	req := &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID}
	if _, err := sessionService.Get(ctx, req); err != nil {
		return "", fmt.Errorf("resume session: %w", err)
	}
	fmt.Printf("Resuming AgentCore session: %s\n", sessionID)
	return sessionID, nil
}

func newAgent(llm model.LLM) (agent.Agent, error) {
	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "A helpful assistant",
		Model:       llm,
		Instruction: "You reply briefly and clearly.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("agent: %w", err)
	}
	return a, nil
}

func runPrompt(ctx context.Context, r *runner.Runner, a agent.Agent, userID, sessionID string) error {
	userMsg := "Remember that this session is backed by Bedrock, then answer: what is 2+2?"
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	for ev, err := range r.Run(
		ctx,
		userID,
		sessionID,
		genai.NewContentFromText(userMsg, genai.RoleUser),
		agent.RunConfig{},
		runner.WithStateDelta(map[string]any{"last_prompt": userMsg}),
	) {
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
		if ev.Author != a.Name() || ev.LLMResponse.Partial || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text != "" {
				fmt.Print(p.Text)
			}
		}
		fmt.Println()
	}
	return nil
}

func printStoredSession(ctx context.Context, sessionService session.Service, userID, sessionID string) error {
	req := &session.GetRequest{AppName: appName, UserID: userID, SessionID: sessionID}
	got, err := sessionService.Get(ctx, req)
	if err != nil {
		return fmt.Errorf("get persisted session: %w", err)
	}
	lastPrompt, _ := got.Session.State().Get("last_prompt")
	fmt.Printf(
		"Session ID: %s\nStored events: %d\nlast_prompt: %v\n",
		sessionID,
		got.Session.Events().Len(),
		lastPrompt,
	)
	return nil
}

func envOrDefault(name, defaultValue string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	return value
}
