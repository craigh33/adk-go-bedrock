// Bedrock tool-calling agent example for adk-go: demonstrates how to define
// and use tools with the Bedrock model. The agent can call a weather tool to answer
// questions about weather. Set BEDROCK_MODEL_ID and authenticate with AWS using the
// default credential chain. Optionally set AWS_REGION. Run:
//
//	go run ./examples/bedrock-tool-calling
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

// getWeather simulates a weather API call and returns fake weather data.
func getWeather(city string) string {
	weathers := []string{"sunny", "rainy", "cloudy", "snowy"}
	temps := []int{55, 65, 72, 45, 80}
	hash := 0
	for _, r := range strings.ToLower(city) {
		hash += int(r)
	}
	weather := weathers[hash%len(weathers)]
	temp := temps[hash%len(temps)]
	return fmt.Sprintf("Weather in %s: %s, %d°F", city, weather, temp)
}

// weatherToolHandler processes tool calls for the weather tool.
func weatherToolHandler(toolCall *genai.FunctionCall) (map[string]any, error) {
	city, ok := toolCall.Args["city"].(string)
	if !ok {
		return nil, errors.New("city parameter is required and must be a string")
	}

	result := getWeather(city)
	return map[string]any{"result": result}, nil
}

func firstResponse(ctx context.Context, llm model.LLM, req *model.LLMRequest) (*model.LLMResponse, error) {
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, err
		}
		if resp != nil {
			return resp, nil
		}
	}
	return nil, errors.New("empty model response")
}

//nolint:funlen // Example keeps the end-to-end tool-calling flow in one function for readability.
func main() {
	ctx := context.Background()

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		log.Fatalf("load AWS config (check credentials and AWS_PROFILE): %v", err)
	}
	if awsCfg.Region == "" {
		log.Fatal("AWS region is unset: set AWS_REGION or add region to your ~/.aws/config profile")
	}

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID is required (e.g. eu.amazon.nova-2-lite-v1:0) using default model")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		log.Fatalf("tracer provider: %v", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br, bedrock.WithTracerProvider(tp)))
	if err != nil {
		log.Fatalf("bedrock model: %v", err)
	}

	// Define the weather tool
	weatherTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "get_weather",
			Description: "Gets the current weather for a specified city",
			Parameters: &genai.Schema{
				Type: "object",
				Properties: map[string]*genai.Schema{
					"city": {
						Type:        "string",
						Description: "The city name to get weather for",
					},
				},
				Required: []string{"city"},
			},
		}},
	}

	userMsg := "What's the weather like in Seattle?"
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)

	userContent := genai.NewContentFromText(userMsg, genai.RoleUser)
	initialReq := &model.LLMRequest{
		Contents: []*genai.Content{userContent},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
			Tools:           []*genai.Tool{weatherTool},
		},
	}

	resp, err := firstResponse(ctx, llm, initialReq)
	if err != nil {
		log.Fatalf("initial model call: %v", err)
	}

	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		log.Fatal("model returned no content")
	}

	var modelToolParts []*genai.Part
	var toolResultParts []*genai.Part

	for _, part := range resp.Content.Parts {
		if part.FunctionCall == nil {
			continue
		}
		fmt.Printf("Tool call: %s args=%v\n", part.FunctionCall.Name, part.FunctionCall.Args)
		result, werr := weatherToolHandler(part.FunctionCall)
		if werr != nil {
			log.Fatalf("weather tool: %v", werr)
		}
		fmt.Printf("Tool result: %v\n\n", result["result"])
		modelToolParts = append(modelToolParts, part)
		toolResultParts = append(toolResultParts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
			ID:       part.FunctionCall.ID,
			Name:     part.FunctionCall.Name,
			Response: result,
		}})
	}

	if len(toolResultParts) == 0 {
		fmt.Println("Assistant response:")
		for _, p := range resp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
		}
		fmt.Printf("\nFinish reason: %s\n", resp.FinishReason)
		return
	}

	followupReq := &model.LLMRequest{
		Contents: []*genai.Content{
			userContent,
			{Role: genai.RoleModel, Parts: modelToolParts},
			{Role: genai.RoleUser, Parts: toolResultParts},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
			Tools:           []*genai.Tool{weatherTool},
		},
	}

	finalResp, err := firstResponse(ctx, llm, followupReq)
	if err != nil {
		log.Fatalf("final model call: %v", err)
	}

	fmt.Println("Assistant's final response:")
	if finalResp.Content != nil {
		for _, p := range finalResp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
		}
	}
	fmt.Printf("\nFinish reason: %s\n", finalResp.FinishReason)
}
