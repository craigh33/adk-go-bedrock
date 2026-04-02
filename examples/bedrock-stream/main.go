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
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
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
		log.Println("BEDROCK_MODEL_ID is required (e.g. eu.amazon.nova-2-lite-v1:0) using default model")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	prompt := "Explain what streaming model output means in one short paragraph."
	if len(os.Args) > 1 {
		prompt = strings.Join(os.Args[1:], " ")
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

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	fmt.Printf("Prompt: %s\n\n", prompt)
	var accumulatedText strings.Builder
	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			continue
		}
		if resp.Partial {
			if txt := resp.Content.Parts[0].Text; txt != "" {
				accumulatedText.WriteString(txt)
				fmt.Printf("[partial] %s\n", txt)
			}
			continue
		}

		fmt.Println("\n[final]")
		for _, p := range resp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
			if p.FunctionCall != nil {
				fmt.Printf("function_call id=%s name=%s args=%v\n",
					p.FunctionCall.ID, p.FunctionCall.Name, p.FunctionCall.Args)
			}
		}
		fmt.Printf("finish_reason=%s turn_complete=%t\n",
			resp.FinishReason, resp.TurnComplete)
	}
}
