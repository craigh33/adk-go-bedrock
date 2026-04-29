// Bedrock function-tool example for adk-go: uses ADK functiontool with a typed handler
// (ParametersJsonSchema) and the Bedrock Converse provider. Set BEDROCK_MODEL_ID and
// authenticate with AWS using the default credential chain. Optionally set AWS_REGION. Run:
//
//	go run ./examples/bedrock-function-tool
package main

import (
	"context"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

// WeatherArgs is the JSON input the model supplies for get_weather.
type WeatherArgs struct {
	Location string `json:"location"`
}

// WeatherResult is returned to the model after the tool runs.
type WeatherResult struct {
	Location    string    `json:"location"`
	Temperature int       `json:"temperature"`
	Condition   string    `json:"condition"`
	Humidity    int       `json:"humidity"`
	Timestamp   time.Time `json:"timestamp"`
}

// GetWeather simulates a weather API and returns structured data for the model.
//
//nolint:gosec // G404: math/rand for non-cryptographic fake weather only.
func GetWeather(_ tool.Context, args WeatherArgs) (WeatherResult, error) {
	temperatures := []int{-10, -5, 0, 5, 10, 15, 20, 25, 30, 35}
	conditions := []string{"sunny", "cloudy", "rainy", "snowy", "windy"}

	return WeatherResult{
		Location:    args.Location,
		Temperature: temperatures[rand.IntN(len(temperatures))],
		Condition:   conditions[rand.IntN(len(conditions))],
		Humidity:    rand.IntN(61) + 30, // 30–90
		Timestamp:   time.Now(),
	}, nil
}

func main() {
	ctx := context.Background()

	// Default AWS authentication: same resolution order as the AWS CLI — env vars
	// (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN), shared credentials
	// file, config file (including profile and region), SSO token provider, IMDS on EC2, etc.
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
		log.Panicf("bedrock model: %v", err)
	}

	getWeatherTool, err := functiontool.New(functiontool.Config{
		Name:        "get_weather",
		Description: "Get weather information for a location (temperature, condition, humidity, timestamp).",
	}, GetWeather)
	if err != nil {
		//nolint:gocritic // exitAfterDefer: example skips tracer shutdown on tool setup failure
		log.Fatalf("function tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "weather_agent",
		Model:       llm,
		Description: "Weather agent.",
		Instruction: `You are a helpful assistant that provides weather information for a given location. Use the get_weather tool to fetch the current weather data, including temperature, condition, humidity, and timestamp. Always provide the most accurate and up-to-date information based on the user's request.`,
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
		Tools: []tool.Tool{getWeatherTool},
	})
	if err != nil {
		log.Panicf("agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-function-tool-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Panicf("runner: %v", err)
	}

	userMsg := "What is the weather like in Seattle? Summarize the tool output briefly."
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	for ev, err := range r.Run(ctx, "local-user", "demo-session", genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Panicf("run: %v", err)
		}
		if ev.Author != a.Name() {
			continue
		}
		if ev.LLMResponse.Partial {
			continue
		}
		if ev.LLMResponse.Content != nil {
			for _, p := range ev.LLMResponse.Content.Parts {
				if p.Text != "" {
					fmt.Print(p.Text)
				}
			}
			fmt.Println()
		}
	}
}
