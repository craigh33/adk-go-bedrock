// Bedrock request guardrail example for adk-go: attaches a preconfigured
// Bedrock guardrail to one Converse request.
//
// Set BEDROCK_MODEL_ID, BEDROCK_GUARDRAIL_ID, BEDROCK_GUARDRAIL_VERSION, and
// authenticate with AWS using the default credential chain. Run:
//
//	go run ./examples/bedrock-request-guardrail
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
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

	modelID := strings.TrimSpace(os.Getenv("BEDROCK_MODEL_ID"))
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID not set, using default")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}
	guardrailID := strings.TrimSpace(os.Getenv("BEDROCK_GUARDRAIL_ID"))
	guardrailVersion := strings.TrimSpace(os.Getenv("BEDROCK_GUARDRAIL_VERSION"))
	if guardrailID == "" || guardrailVersion == "" {
		log.Fatal("set BEDROCK_GUARDRAIL_ID and BEDROCK_GUARDRAIL_VERSION")
	}

	prompt := strings.TrimSpace(os.Getenv("PROMPT"))
	if prompt == "" {
		prompt = "Explain why clear password policies matter in one short paragraph."
	}
	if len(os.Args) > 1 {
		prompt = strings.TrimSpace(strings.Join(os.Args[1:], " "))
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		log.Fatalf("tracer provider: %v", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	runtimeAPI := converse.NewRuntimeAPI(br, converse.WithTracerProvider(tp))
	llm, err := converse.NewWithAPI(
		modelID,
		runtimeAPI,
		converse.WithGuardrail(guardrailID, guardrailVersion, types.GuardrailTraceEnabled),
	)
	if err != nil {
		//nolint:gocritic // exitAfterDefer: example skips tracer shutdown on model setup failure
		log.Fatalf("create bedrock model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	fmt.Printf("Model: %s\nGuardrail: %s:%s\nPrompt: %s\n\n", modelID, guardrailID, guardrailVersion, prompt)
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Fatalf("generate: %v", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Println(part.Text)
				}
			}
		}
		fmt.Printf("\nfinish_reason=%s\n", resp.FinishReason)
		printGuardrailMetadata(resp)
	}
}

func printGuardrailMetadata(resp *model.LLMResponse) {
	if resp.CustomMetadata == nil {
		fmt.Println("guardrail_trace=false")
		return
	}
	if ratings, ok := resp.CustomMetadata["safety_ratings"].([]*genai.SafetyRating); ok {
		for _, rating := range ratings {
			fmt.Printf("safety_rating category=%s blocked=%t probability=%s\n",
				rating.Category, rating.Blocked, rating.Probability)
		}
	}
	_, hasTrace := resp.CustomMetadata["bedrock_guardrail_trace"]
	fmt.Printf("guardrail_trace=%t\n", hasTrace)
}
