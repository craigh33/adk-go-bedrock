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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/mcptoolset"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
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

func userMessageFromArgs(args []string) string {
	userMsg := "What is the weather in Seattle? Use your MCP tool if needed."
	if len(args) > 1 {
		userMsg = strings.Join(args[1:], " ")
	}

	return userMsg
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		log.Printf("load AWS config (check credentials and AWS_PROFILE): %v", err)
		return
	}
	if awsCfg.Region == "" {
		log.Print("AWS region is unset: set AWS_REGION or add region to your ~/.aws/config profile")
		return
	}

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID not set; defaulting to eu.amazon.nova-2-lite-v1:0")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br))
	if err != nil {
		log.Printf("bedrock model: %v", err)
		return
	}

	mcpToolSet, err := mcptoolset.New(mcptoolset.Config{
		Transport: localMCPTransport(ctx),
	})
	if err != nil {
		log.Printf("create MCP tool set: %v", err)
		return
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
		return
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-mcp-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Printf("runner: %v", err)
		return
	}

	userMsg := userMessageFromArgs(os.Args)

	fmt.Printf("User: %s\n\n", userMsg)

	for ev, err := range r.Run(ctx, "local-user", "demo-session", genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Printf("run: %v", err)
			return
		}
		if ev.Author != a.Name() {
			continue
		}
		if ev.LLMResponse.Partial {
			continue
		}
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text != "" {
				fmt.Print(p.Text)
			}
		}
		fmt.Println()
	}
}
