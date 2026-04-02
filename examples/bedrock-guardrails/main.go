// Bedrock guardrails example for adk-go: demonstrates how to work with Bedrock guardrails,
// including safety assessments, content filtering, and custom metadata. This example shows
// how guardrail metadata is exposed through ADK-friendly CustomMetadata and FinishReason.
// Set BEDROCK_MODEL_ID and authenticate with AWS using the default credential chain.
// Run:
//
//	go run ./examples/bedrock-guardrails
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

// demonstrateSafetyRatings shows how to extract and interpret safety ratings from responses.
func demonstrateSafetyRatings(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Safety Ratings Example ===")
	fmt.Println("Request: Explain climate change science")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Explain climate change science in a balanced way."},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	}

	// Generate response and capture safety metadata
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					displayLen := minInt(len(part.Text), 200)
					fmt.Printf("Response: %s\n", part.Text[:displayLen])
					if len(part.Text) > 200 {
						fmt.Println("...")
					}
				}
			}
		}

		// Extract and display safety ratings from custom metadata
		if resp.CustomMetadata != nil {
			if ratings, ok := resp.CustomMetadata["safety_ratings"].([]*genai.SafetyRating); ok {
				fmt.Println("\n--- Safety Assessment ---")
				for _, rating := range ratings {
					fmt.Printf("Category: %s\n", rating.Category)
					fmt.Printf("  Blocked: %v\n", rating.Blocked)
					fmt.Printf("  Probability: %s\n", rating.Probability)
					fmt.Printf("  Probability Score: %.2f\n", rating.ProbabilityScore)
					if rating.Severity != genai.HarmSeverityUnspecified {
						fmt.Printf("  Severity: %s\n", rating.Severity)
						fmt.Printf("  Severity Score: %.2f\n", rating.SeverityScore)
					}
				}
			}

			// Display guardrail trace if available
			if trace, ok := resp.CustomMetadata["bedrock_guardrail_trace"]; ok {
				fmt.Printf("\n--- Guardrail Trace ---\n%v\n", trace)
			}
		}

		// Display usage metadata
		if resp.UsageMetadata != nil {
			fmt.Printf("Usage - Prompt: %d, Candidates: %d, Total: %d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}

	return nil
}

// demonstrateGuardrailIntervention shows how to detect when guardrails block content.
func demonstrateGuardrailIntervention(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Guardrail Intervention Detection ===")
	fmt.Println("Demonstrating how to detect guardrail blocks")

	// This is a benign example - in production, you'd handle actual guardrail blocks
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What is the capital of France?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 256,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		// Check the finish reason to detect guardrail interventions
		if resp.Content != nil {
			fmt.Printf("Response content received\n")
		}

		if resp.UsageMetadata != nil {
			fmt.Printf("Prompt tokens: %d, Candidates: %d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount)

			// Bedrock-specific finish reasons for guardrails:
			// - FinishReasonSafety: Guardrail intervention
			// - FinishReasonSPII: Sensitive personally identifiable information detected
			// - FinishReasonBlocklist: Word filter or blocklist detected
			// - FinishReasonProhibitedContent: Topic-based prohibition triggered
		}
	}

	return nil
}

// demonstrateGuardrailMetadataExtraction shows how to extract detailed guardrail metadata.
func demonstrateGuardrailMetadataExtraction(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Guardrail Metadata Extraction ===")
	fmt.Println("Extracting and analyzing guardrail assessment data")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Tell me about cybersecurity best practices"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					displayLen := minInt(len(part.Text), 150)
					fmt.Printf("Response preview: %s\n", part.Text[:displayLen])
					if len(part.Text) > 150 {
						fmt.Println("...")
					}
				}
			}
		}

		if resp.CustomMetadata != nil {
			fmt.Println("\n--- Complete Metadata ---")
			for key, value := range resp.CustomMetadata {
				fmt.Printf("%s: %v\n", key, value)
			}

			// Example: Check for specific assessment types
			if trace, ok := resp.CustomMetadata["bedrock_guardrail_trace"]; ok {
				fmt.Printf("\nGuardrail assessment present: %v\n", trace != nil)
			}

			// Example: Access additional model response fields (if provided by the model)
			if additionalFields, ok := resp.CustomMetadata["bedrock_additional_model_response_fields"]; ok {
				fmt.Printf("Additional model response fields: %v\n", additionalFields)
			}
		}

		if resp.UsageMetadata != nil {
			fmt.Printf("\nTokens used - Prompt: %d, Candidates: %d, Total: %d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}

	return nil
}

// demonstrateMultipleSafetyCategories shows how different harm categories are assessed.
func demonstrateMultipleSafetyCategories(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Multiple Safety Categories ===")
	fmt.Println("Demonstrating assessment across different harm categories")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What are the rules of chess?"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					displayLen := minInt(len(part.Text), 200)
					fmt.Printf("Response: %s\n", part.Text[:displayLen])
					if len(part.Text) > 200 {
						fmt.Println("...")
					}
				}
			}
		}

		// Display categorized safety assessment
		if resp.CustomMetadata != nil { //nolint:nestif // necessary nested checks for safety ratings
			if ratings, ok := resp.CustomMetadata["safety_ratings"].([]*genai.SafetyRating); ok {
				if len(ratings) > 0 {
					fmt.Println("\n--- Safety Assessment by Category ---")
					for _, rating := range ratings {
						status := "Allowed"
						if rating.Blocked {
							status = "BLOCKED"
						}
						fmt.Printf("[%s] %s - Prob: %s (%.2f), Severity: %s (%.2f)\n",
							status,
							rating.Category,
							rating.Probability,
							rating.ProbabilityScore,
							rating.Severity,
							rating.SeverityScore)
					}
				} else {
					fmt.Println("No safety concerns detected")
				}
			}
		}
	}

	return nil
}

// demonstrateStreamingWithGuardrails shows how guardrail metadata is available in streaming.
func demonstrateStreamingWithGuardrails(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Streaming with Guardrails ===")
	fmt.Println("Demonstrating guardrail metadata in streaming responses")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Write a brief poem about spring"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 256,
		},
	}

	partialCount := 0
	for resp, err := range llm.GenerateContent(ctx, req, true) { // true = streaming
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil && resp.Content.Parts != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Print(part.Text)
					if resp.Partial {
						partialCount++
					}
				}
			}
		}

		// Note: In streaming, guardrail metadata is accumulated and delivered with metadata events
		if resp.CustomMetadata != nil && !resp.Partial {
			fmt.Println("\n\n--- Final Guardrail Assessment ---")
			if ratings, ok := resp.CustomMetadata["safety_ratings"].([]*genai.SafetyRating); ok {
				for _, rating := range ratings {
					fmt.Printf("%s: %v\n", rating.Category, rating.Probability)
				}
			}
		}
	}
	fmt.Printf("\nStreamed %d partial chunks\n", partialCount)

	return nil
}

// minInt returns the minimum of two integers.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	ctx := context.Background()

	// Load AWS config
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
		log.Println(
			"BEDROCK_MODEL_ID is required (e.g. eu.amazon.nova-2-lite-v1:0) using default model",
		)
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

	// Run all demonstrations
	demonstrations := []struct {
		name string
		fn   func(context.Context, model.LLM) error
	}{
		{"Safety Ratings", demonstrateSafetyRatings},
		{"Guardrail Intervention Detection", demonstrateGuardrailIntervention},
		{"Guardrail Metadata Extraction", demonstrateGuardrailMetadataExtraction},
		{"Multiple Safety Categories", demonstrateMultipleSafetyCategories},
		{"Streaming with Guardrails", demonstrateStreamingWithGuardrails},
	}

	for _, demo := range demonstrations {
		if err := demo.fn(ctx, llm); err != nil {
			log.Printf("ERROR in %s: %v\n", demo.name, err)
		}
	}

	fmt.Println("\n=== Guardrails Examples Complete ===")
	fmt.Println("\nKey Points:")
	fmt.Println("- Safety ratings are available in CustomMetadata[\"safety_ratings\"]")
	fmt.Println("- Guardrail trace data is available in CustomMetadata[\"bedrock_guardrail_trace\"]")
	fmt.Println("- FinishReason indicates if guardrails blocked the response")
	fmt.Println("- Streaming accumulates guardrail metadata into final response")
	fmt.Println("- Note: To use Bedrock guardrails, pre-configure a guardrail in AWS")
	fmt.Println("  and pass its ID and version to the model via custom Bedrock APIs")
}
