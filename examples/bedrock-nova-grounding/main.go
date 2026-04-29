// Nova Web Grounding example: set BEDROCK_MODEL_ID to a US inference profile that supports
// Web Grounding (see AWS docs), AWS_REGION to a US Bedrock region, and authenticate with AWS.
//
//	go run ./examples/bedrock-nova-grounding
package main

import (
	"context"
	"encoding/json"
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
	"github.com/craigh33/adk-go-bedrock/internal/mappers"
	"github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

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
		log.Println(
			"BEDROCK_MODEL_ID unset; using us.amazon.nova-2-lite-v1:0",
		)
		modelID = "us.amazon.nova-2-lite-v1:0"
	}

	prompt := "What is one timely news headline about renewable energy? Cite sources briefly."
	if len(os.Args) > 1 {
		prompt = strings.Join(os.Args[1:], " ")
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		return fmt.Errorf("tracer provider: %w", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br, bedrock.WithTracerProvider(tp)))
	if err != nil {
		return fmt.Errorf("bedrock model: %w", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText(prompt, genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			Tools:           []*genai.Tool{novagrounding.Tool()},
			MaxOutputTokens: 1024,
		},
	}

	fmt.Printf("Prompt: %s\n\n", prompt)

	var last *model.LLMResponse
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		last = resp
	}
	if last == nil || last.Content == nil {
		return errors.New("empty response")
	}

	fmt.Println("Assistant:")
	for _, p := range last.Content.Parts {
		if p.Text != "" {
			fmt.Println(p.Text)
		}
		printCitationParts(p)
	}
	fmt.Printf("\nFinish reason: %s\n", last.FinishReason)
	return nil
}

func printCitationParts(p *genai.Part) {
	if p == nil || p.PartMetadata == nil {
		return
	}
	raw, ok := p.PartMetadata[mappers.PartMetadataKeyBedrockCitations]
	if !ok {
		return
	}
	rows, ok := raw.([]any)
	if !ok || len(rows) == 0 {
		return
	}
	fmt.Println("\nCitations (retain and display with grounded answers per AWS guidance):")
	for i, row := range rows {
		b, err := json.MarshalIndent(row, "  ", "  ")
		if err != nil {
			fmt.Printf("  [%d] %+v\n", i, row)
			continue
		}
		fmt.Printf("  [%d] %s\n", i, string(b))
	}
}
