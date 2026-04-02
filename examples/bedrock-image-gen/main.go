// Bedrock image generation example for adk-go: demonstrates how to use the imagegenerator tool
// with the ADK runner to generate images via Amazon Nova Canvas. The agent receives a
// prompt, invokes the generate_image tool, and the resulting image is saved as an artifact.
// Set BEDROCK_MODEL_ID for the conversational model and authenticate with AWS using the default
// credential chain. Optionally set AWS_REGION. Run:
//
//	go run ./examples/bedrock-image-gen
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/tools/imagegenerator"
)

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
		log.Println("BEDROCK_MODEL_ID not set, using default: eu.amazon.nova-2-lite-v1:0")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	br := bedrockruntime.NewFromConfig(awsCfg)

	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br))
	if err != nil {
		log.Fatalf("bedrock model: %v", err)
	}

	imgTool, err := imagegenerator.New(imagegenerator.Config{
		API:      br,
		Provider: imagegenerator.NewCanvasProvider(""),
	})
	if err != nil {
		log.Fatalf("image generator tool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "image-assistant",
		Description: "An assistant that can generate images from text prompts",
		Model:       llm,
		Instruction: `You are an image generation assistant. When the user asks you to create
or generate an image, use the generate_image tool with a descriptive prompt and a sensible
file name. After the image is generated, confirm what was created and the file name.`,
		Tools: []tool.Tool{imgTool},
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		log.Fatalf("agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-image-gen-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		ArtifactService:   artifact.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	userMsg := "Generate an image of a cute steampunk robot in a garden"
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)

	for ev, err := range r.Run(ctx, "local-user", "demo-session", genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run: %v", err)
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
}
