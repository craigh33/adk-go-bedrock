// Bedrock chat example for adk-go: set BEDROCK_MODEL_ID and authenticate with AWS using the
// default credential chain (environment variables, shared config ~/.aws/credentials and
// ~/.aws/config, SSO / AWS_PROFILE, web identity, EC2/ECS/Lambda role, etc.). Optionally set
// AWS_REGION or configure a region on your profile. Run:
//
//	go run ./examples/bedrock-chat
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
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/craig-hutcheon/adk-go-bedrock/bedrock"
	"github.com/craig-hutcheon/adk-go-bedrock/bedrock/client"
)

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

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, client.NewFromClient(br))
	if err != nil {
		log.Fatalf("bedrock model: %v", err)
	}

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
		log.Fatalf("agent: %v", err)
	}

	r, err := runner.New(runner.Config{
		AppName:           "bedrock-chat-example",
		Agent:             a,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	userMsg := "What is 2+2? Reply with just the number."
	if len(os.Args) > 1 {
		userMsg = os.Args[1]
	}

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
