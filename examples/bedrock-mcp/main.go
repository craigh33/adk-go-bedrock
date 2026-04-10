// Bedrock MCP example for adk-go: demonstrates how to use ADK's MCP toolset with the
// Bedrock Converse provider. It starts an in-memory MCP server that exposes a weather
// tool, then runs a simple CLI chat loop through an ADK runner.
//
// Set BEDROCK_MODEL_ID and authenticate with AWS using the default credential chain.
// Optionally set AWS_REGION. Run:
//
//	go run ./examples/bedrock-mcp
package main

import (
	"context"
	"fmt"
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

type weatherInput struct {
	City string `json:"city" jsonschema:"city name"`
}

type weatherOutput struct {
	WeatherSummary string `json:"weather_summary" jsonschema:"weather summary in the given city"`
}

func getWeather(
	_ context.Context,
	_ *mcp.CallToolRequest,
	input weatherInput,
) (*mcp.CallToolResult, weatherOutput, error) {
	return nil, weatherOutput{
		WeatherSummary: fmt.Sprintf("Today in %q is sunny and 72°F.", input.City),
	}, nil
}

func localMCPTransport(ctx context.Context) mcp.Transport {
	clientTransport, serverTransport := mcp.NewInMemoryTransports()

	server := mcp.NewServer(&mcp.Implementation{Name: "weather_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{
		Name:        "get_weather",
		Description: "Returns the weather in the given city",
	}, getWeather)
	if _, err := server.Connect(ctx, serverTransport, nil); err != nil {
		log.Fatalf("connect in-memory MCP server: %v", err)
	}

	return clientTransport
}

func githubMCPTransport(ctx context.Context) mcp.Transport {
	ts := oauth2.StaticTokenSource(
		&oauth2.Token{AccessToken: os.Getenv("GITHUB_PAT")},
	)
	return &mcp.StreamableClientTransport{
		Endpoint:   "https://api.githubcopilot.com/mcp/",
		HTTPClient: oauth2.NewClient(ctx, ts),
	}
}

func main() {
	os.Exit(runMain())
}

//nolint:funlen //example main function with setup and run loop
func runMain() int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

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

	var transport mcp.Transport
	if strings.ToLower(os.Getenv("AGENT_MODE")) == "github" {
		transport = githubMCPTransport(ctx)
	} else {
		transport = localMCPTransport(ctx)
	}

	mcpToolSet, err := mcptoolset.New(mcptoolset.Config{
		Transport: transport,
	})
	if err != nil {
		log.Printf("create MCP tool set: %v", err)
		return 1
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "helper_agent",
		Description: "A helpful assistant with MCP tools.",
		Model:       llm,
		Instruction: "You are a helpful assistant. Use the available MCP tools when they help answer the user.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
		Toolsets: []tool.Toolset{mcpToolSet},
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
