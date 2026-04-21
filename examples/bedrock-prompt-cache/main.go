// Bedrock prompt caching example for adk-go: demonstrates WithCacheSystemPrompt,
// which appends a Bedrock CachePoint after the system prompt so that repeated
// requests reuse the cached prompt tokens instead of re-processing them.
//
// Set BEDROCK_MODEL_ID and authenticate with AWS using the default credential
// chain. Run:
//
//	go run ./examples/bedrock-prompt-cache
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

// largeSystemPrompt simulates a detailed, token-heavy system instruction that
// is worth caching — a long persona, policy document, or knowledge base excerpt.
// It is repeated to hit Bedrock's minimum for prompt caching.
//
//nolint:gochecknoglobals // This is just an example
var largeSystemPrompt = strings.Repeat(`You are an expert software engineer and technical writer.

Your responsibilities:
- Explain complex technical concepts clearly and concisely
- Write production-quality code with best practices
- Review code for bugs, security issues, and performance problems
- Suggest architectural improvements backed by reasoning
- Cite relevant standards, RFCs, or documentation when appropriate

Guidelines:
- Default to Go, Python, or TypeScript unless the user specifies otherwise
- Prefer readability over cleverness
- Always highlight security implications when they exist
- When giving code, include brief inline comments for non-obvious logic
- Structure long answers with headings so they are easy to scan
`, 20)

func ask(ctx context.Context, llm model.LLM, question string) (*genai.GenerateContentResponseUsageMetadata, error) {
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{{Text: question}},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: largeSystemPrompt}},
			},
		},
	}

	var usage *genai.GenerateContentResponseUsageMetadata
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("%s\n", part.Text)
				}
			}
		}
		if resp.UsageMetadata != nil {
			usage = resp.UsageMetadata
		}
	}
	return usage, nil
}

func printUsage(label string, u *genai.GenerateContentResponseUsageMetadata) {
	if u == nil {
		return
	}
	fmt.Printf("[%s] prompt=%d  candidates=%d  fromCache=%d total=%d tokens\n",
		label, u.PromptTokenCount, u.CandidatesTokenCount, u.CachedContentTokenCount, u.TotalTokenCount)
}

func main() {
	ctx := context.Background()

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		log.Fatalf("load AWS config: %v", err)
	}
	if awsCfg.Region == "" {
		log.Fatal("AWS region is unset: set AWS_REGION or configure ~/.aws/config")
	}

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID not set, using default")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		log.Fatalf("tracer provider: %v", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	runtimeAPI := bedrock.NewRuntimeAPI(br, bedrock.WithTracerProvider(tp))

	// WithCacheSystemPrompt appends a CachePoint block after the system prompt
	// on every request. Bedrock caches everything up to that marker, so
	// subsequent requests pay (90%) less for those tokens.
	llm, err := bedrock.NewWithAPI(modelID, runtimeAPI, bedrock.WithCacheSystemPrompt())
	if err != nil {
		log.Fatalf("create bedrock model: %v", err) //nolint:gocritic // This is just an example
	}

	questions := []string{
		"What are the key differences between a mutex and a semaphore?",
		"How does Go's garbage collector work at a high level?",
		"What is the difference between authentication and authorisation?",
	}

	fmt.Printf("Model: %s\n", modelID)
	fmt.Printf("System prompt: %d chars (cached after first request)\n\n", len(largeSystemPrompt))

	for i, q := range questions {
		fmt.Printf("=== Question %d: %s ===\n", i+1, q)
		usage, err := ask(ctx, llm, q)
		if err != nil {
			log.Printf("question %d: %v", i+1, err)
			continue
		}
		printUsage(fmt.Sprintf("Q%d", i+1), usage)
		// On the second and subsequent requests the system prompt tokens are
		// served from cache. Bedrock reports this via CacheReadInputTokens in
		// the raw TokenUsage — reflected as CachedContentTokenCount in the
		// genai UsageMetadata when the field is populated by the mapper.
		fmt.Println()
	}
}
