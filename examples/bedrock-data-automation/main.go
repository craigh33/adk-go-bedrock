// Bedrock Data Automation example for adk-go. Set BEDROCK_MODEL_ID,
// BDA_PROFILE_ARN, BDA_OUTPUT_S3_URI, AWS_REGION, and AWS credentials.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	bdaruntime "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/tools/bedrockdataautomation"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	profileARN := strings.TrimSpace(os.Getenv("BDA_PROFILE_ARN"))
	if profileARN == "" {
		return errors.New("BDA_PROFILE_ARN is required")
	}
	outputS3URI := strings.TrimSpace(os.Getenv("BDA_OUTPUT_S3_URI"))
	if outputS3URI == "" {
		return errors.New("BDA_OUTPUT_S3_URI is required")
	}

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return fmt.Errorf("load AWS config: %w", err)
	}
	if awsCfg.Region == "" {
		return errors.New("AWS region is unset: set AWS_REGION or configure a profile region")
	}

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(
		envOrDefault("BEDROCK_MODEL_ID", "eu.amazon.nova-2-lite-v1:0"),
		bedrock.NewRuntimeAPI(br),
	)
	if err != nil {
		return fmt.Errorf("bedrock model: %w", err)
	}

	bdaTool, err := bedrockdataautomation.New(bedrockdataautomation.Config{
		API:                      bdaruntime.NewFromConfig(awsCfg),
		S3:                       s3.NewFromConfig(awsCfg),
		DataAutomationProfileARN: profileARN,
		DataAutomationProjectARN: strings.TrimSpace(os.Getenv("BDA_PROJECT_ARN")),
		OutputS3URI:              outputS3URI,
		InputS3URI:               strings.TrimSpace(os.Getenv("BDA_INPUT_S3_URI")),
	})
	if err != nil {
		return fmt.Errorf("data automation tool: %w", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "data-automation-assistant",
		Description: "An assistant that analyzes documents and media with Bedrock Data Automation",
		Model:       llm,
		Instruction: `When asked to analyze a document, image, audio, or video in S3, call analyze_data with s3_uri.
If the user asks to save JSON output as an artifact, set result_artifact_name to the requested filename.`,
		Tools: []tool.Tool{bdaTool},
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		return fmt.Errorf("agent: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-data-automation-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		ArtifactService:   artifact.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return fmt.Errorf("runner: %w", err)
	}

	userMsg := "Analyze s3://your-bucket/path/to/document.pdf and save the JSON as bda-result.json"
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)
	return printRunEvents(ctx, r, a, userMsg)
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

func envOrDefault(name, defaultValue string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	return value
}
