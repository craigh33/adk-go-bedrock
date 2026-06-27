// Bedrock AgentCore Gateway example for adk-go: set AGENTCORE_GATEWAY_ENDPOINT,
// AGENTCORE_GATEWAY_ACCESS_TOKEN, BEDROCK_MODEL_ID, and authenticate with AWS
// using the default credential chain. Optionally set AWS_REGION.
package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/full"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

func main() {
	os.Exit(runMain())
}

func runMain() int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	endpoint := strings.TrimSpace(os.Getenv("AGENTCORE_GATEWAY_ENDPOINT"))
	if endpoint == "" {
		log.Print("AGENTCORE_GATEWAY_ENDPOINT is required")
		return 1
	}
	accessToken := strings.TrimSpace(os.Getenv("AGENTCORE_GATEWAY_ACCESS_TOKEN"))
	if accessToken == "" {
		log.Print("AGENTCORE_GATEWAY_ACCESS_TOKEN is required")
		return 1
	}

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		log.Printf("load AWS config (check credentials and AWS_PROFILE): %v", err)
		return 1
	}
	if awsCfg.Region == "" {
		log.Print("AWS region is unset: set AWS_REGION or add region to your ~/.aws/config profile")
		return 1
	}

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID not set; defaulting to eu.amazon.nova-2-lite-v1:0")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		log.Printf("tracer provider: %v", err)
		return 1
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br, bedrock.WithTracerProvider(tp)))
	if err != nil {
		log.Printf("bedrock model: %v", err)
		return 1
	}

	gatewayTools, err := mcptoolset.New(mcptoolset.Config{
		Transport: &mcp.StreamableClientTransport{
			Endpoint: endpoint,
			HTTPClient: oauth2.NewClient(ctx, oauth2.StaticTokenSource(&oauth2.Token{
				AccessToken: accessToken,
			})),
		},
	})
	if err != nil {
		log.Printf("create AgentCore Gateway toolset: %v", err)
		return 1
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "gateway_agent",
		Description: "A helpful assistant with AgentCore Gateway tools.",
		Model:       llm,
		Instruction: "You are a helpful assistant. Use the available Gateway tools when they help answer the user.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
		Toolsets: []tool.Toolset{gatewayTools},
	})
	if err != nil {
		log.Printf("agent: %v", err)
		return 1
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(a),
	}
	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.Printf("Run failed: %v\n\n%s", err, l.CommandLineSyntax())
		return 1
	}
	return 0
}
