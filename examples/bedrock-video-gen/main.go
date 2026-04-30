// Bedrock video generation example for adk-go: demonstrates the videogenerator tool with the ADK
// runner and Amazon Nova Reel async inference. Requires an S3 URI where Bedrock may write outputs
// (same bucket/prefix your role allows for async invoke). Set NOVA_REEL_OUTPUT_S3_URI (e.g.
// s3://my-bucket/reel-output). Optionally set AWS_REGION (Nova Reel is available in select regions;
// see AWS docs). Authenticate with the default credential chain.
//
//	go run ./examples/bedrock-video-gen
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
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/tools/videogenerator"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	outputURI := strings.TrimSpace(os.Getenv("NOVA_REEL_OUTPUT_S3_URI"))
	if outputURI == "" {
		return errors.New("NOVA_REEL_OUTPUT_S3_URI is required (e.g. s3://your-bucket/prefix)")
	}

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return fmt.Errorf("load AWS config (check credentials and AWS_PROFILE): %w", err)
	}
	if awsCfg.Region == "" {
		return errors.New("AWS region is unset: set AWS_REGION or add region to your ~/.aws/config profile")
	}

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID not set, using default: eu.amazon.nova-2-lite-v1:0")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	br := bedrockruntime.NewFromConfig(awsCfg)
	s3Client := s3.NewFromConfig(awsCfg)

	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br))
	if err != nil {
		return fmt.Errorf("bedrock model: %w", err)
	}

	videoTool, err := videogenerator.New(videogenerator.Config{
		API:         br,
		S3OutputURI: outputURI,
		S3:          s3Client,
		Provider:    videogenerator.NewReelProvider("", 0),
	})
	if err != nil {
		return fmt.Errorf("video generator tool: %w", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "video-assistant",
		Description: "An assistant that can generate short videos from text prompts using Nova Reel",
		Model:       llm,
		Instruction: `You are a video generation assistant. When the user asks you to create or generate a video,
use the generate_video tool with a descriptive English prompt (under 512 characters for single-shot)
and a sensible .mp4 file name. Generation is asynchronous and may take minutes. After success, confirm the artifact file name and S3 location if relevant.`,
		Tools: []tool.Tool{videoTool},
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		return fmt.Errorf("agent: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-video-gen-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		ArtifactService:   artifact.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		return fmt.Errorf("runner: %w", err)
	}

	userMsg := "Generate a 6-second video of golden sunlight through forest leaves"
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
		if ev.Author != a.Name() {
			continue
		}
		if ev.LLMResponse.Partial {
			continue
		}
		if ev.Content == nil {
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
