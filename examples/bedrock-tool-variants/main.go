// Bedrock tool variants example for adk-go: demonstrates how to use non-function tool variants
// including Google Search, Code Execution, Retrieval, MCP Servers, and combinations with function
// declarations. Set BEDROCK_MODEL_ID to a Bedrock model that supports these tools and authenticate
// with AWS using the default credential chain. Run:
//
//	go run ./examples/bedrock-tool-variants
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
)

// demonstrateGoogleSearch shows how to enable Google Search as a tool.
func demonstrateGoogleSearch(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Google Search Tool ===")
	fmt.Println("Request: What are the latest AI developments in 2026?")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What are the latest AI developments in 2026? Search for recent news."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			// Enable Google Search as a tool variant
			Tools: []*genai.Tool{{
				GoogleSearch: &genai.GoogleSearch{},
			}},
			MaxOutputTokens: 1024,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateCodeExecution shows how to enable code execution as a tool.
func demonstrateCodeExecution(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Code Execution Tool ===")
	fmt.Println("Request: Calculate factorial of 10")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Calculate the factorial of 10 and verify the result."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			// Enable code execution
			Tools: []*genai.Tool{{
				CodeExecution: &genai.ToolCodeExecution{},
			}},
			MaxOutputTokens: 512,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateRetrieval shows how to enable retrieval as a tool.
func demonstrateRetrieval(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Retrieval Tool ===")
	fmt.Println("Request: Find information from knowledge base")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What are the main components of the system architecture?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			// Enable retrieval from knowledge base
			Tools: []*genai.Tool{{
				Retrieval: &genai.Retrieval{},
			}},
			MaxOutputTokens: 512,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateURLContext shows how to enable URL context as a tool.
func demonstrateURLContext(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== URL Context Tool ===")
	fmt.Println("Request: Summarize content from a URL")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Please analyze this document: https://example.com/whitepaper.html"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			// Enable URL context retrieval
			Tools: []*genai.Tool{{
				URLContext: &genai.URLContext{},
			}},
			MaxOutputTokens: 512,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateMCPServers shows how to enable MCP servers as tools.
func demonstrateMCPServers(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== MCP Servers Tool ===")
	fmt.Println("Request: Use MCP server to fetch data")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Connect to the documentation server and find information about the API."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			// Enable MCP servers
			Tools: []*genai.Tool{{
				MCPServers: []*genai.MCPServer{
					{
						Name: "docs",
					},
				},
			}},
			MaxOutputTokens: 512,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateFunctionDeclarations shows how to use traditional function declarations.
func demonstrateFunctionDeclarations(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Function Declarations ===")
	fmt.Println("Request: Get weather information")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What is the weather in San Francisco right now?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get current weather for a location",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"location": {
								Type:        genai.TypeString,
								Description: "City name",
							},
						},
						Required: []string{"location"},
					},
				}},
			}},
			MaxOutputTokens: 512,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
				if part.FunctionCall != nil {
					fmt.Printf("Tool Called: %s\n", part.FunctionCall.Name)
				}
			}
		}
	}

	return nil
}

// demonstrateMixedTools shows how to combine function declarations with tool variants.
func demonstrateMixedTools(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Mixed Tools (Functions + Variants) ===")
	fmt.Println("Request: Search the web and execute code to analyze results")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Search for 'Go programming 2026' and write code to analyze the results."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				// Include both function declarations and tool variants
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "save_results",
					Description: "Save analysis results to storage",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"results": {
								Type:        genai.TypeString,
								Description: "Analysis results",
							},
						},
						Required: []string{"results"},
					},
				}},
				// Add tool variants alongside function declarations
				GoogleSearch:  &genai.GoogleSearch{},
				CodeExecution: &genai.ToolCodeExecution{},
			}},
			MaxOutputTokens: 1024,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
				if part.FunctionCall != nil {
					fmt.Printf("Tool Called: %s\n", part.FunctionCall.Name)
				}
			}
		}
	}

	return nil
}

// demonstrateMultipleVariants shows using multiple tool variants together.
func demonstrateMultipleVariants(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Multiple Tool Variants ===")
	fmt.Println("Request: Research, analyze, and retrieve from knowledge base")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						Text: "Search for recent developments in quantum computing, analyze them with code, and cross-reference with our knowledge base.",
					},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				// Multiple tool variants in one tool entry
				GoogleSearch:  &genai.GoogleSearch{},
				CodeExecution: &genai.ToolCodeExecution{},
				Retrieval:     &genai.Retrieval{},
			}},
			MaxOutputTokens: 1024,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateToolVariantsWithSystemInstruction shows using tools with system instructions.
func demonstrateToolVariantsWithSystemInstruction(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Tool Variants with System Instructions ===")
	fmt.Println("Request: Use tools while following specific instructions")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What are the top 5 developments in AI this month? Provide a structured analysis."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{
					Text: `You are an AI research analyst. When using tools:
1. Search for recent information
2. Analyze findings thoroughly
3. Provide a structured summary with:
   - Key developments
   - Impact assessment
   - Future implications
4. Use code execution if needed for analysis`,
				}},
			},
			Tools: []*genai.Tool{{
				GoogleSearch:  &genai.GoogleSearch{},
				CodeExecution: &genai.ToolCodeExecution{},
			}},
			MaxOutputTokens: 1024,
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

func main() {
	ctx := context.Background()

	// Load AWS configuration
	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		log.Fatalf("load AWS config: %v", err)
	}
	if awsCfg.Region == "" {
		log.Fatal("AWS region is unset: set AWS_REGION or add region to ~/.aws/config")
	}

	// Get model ID
	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID is required (e.g., us.anthropic.claude-3-5-sonnet-20241022-v2:0)")
		modelID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
	}

	// Create Bedrock LLM
	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br))
	if err != nil {
		log.Fatalf("create bedrock model: %v", err)
	}

	// Run all demonstrations
	demonstrations := []struct {
		name string
		fn   func(context.Context, model.LLM) error
	}{
		{"Google Search", demonstrateGoogleSearch},
		{"Code Execution", demonstrateCodeExecution},
		{"Retrieval", demonstrateRetrieval},
		{"URL Context", demonstrateURLContext},
		{"MCP Servers", demonstrateMCPServers},
		{"Function Declarations", demonstrateFunctionDeclarations},
		{"Mixed Tools", demonstrateMixedTools},
		{"Multiple Variants", demonstrateMultipleVariants},
		{"Tool Variants with System Instructions", demonstrateToolVariantsWithSystemInstruction},
	}

	for _, demo := range demonstrations {
		if err := demo.fn(ctx, llm); err != nil {
			log.Printf("ERROR in %s: %v\n", demo.name, err)
		}
	}

	fmt.Println("\n=== Tool Variants Examples Complete ===")
	fmt.Println("\nKey Points:")
	fmt.Println("- Google Search, Code Execution, Retrieval, and other variants are now supported")
	fmt.Println("- Function declarations and tool variants can be used together")
	fmt.Println("- Multiple variants can be combined in a single tool entry")
	fmt.Println("- Tool variants work seamlessly with system instructions")
	fmt.Println("- All supported tool types are sent to Bedrock for model use")
}
