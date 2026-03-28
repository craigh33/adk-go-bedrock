// Bedrock system instruction example for adk-go: demonstrates how to use system
// instructions for role definition, output formatting, and behavioral control.
// Set BEDROCK_MODEL_ID and authenticate with AWS using the default credential chain.
// Run:
//
//	go run ./examples/bedrock-system-instruction
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
)

// demonstrateTechnicalAssistant shows system instruction for a technical expert role.
func demonstrateTechnicalAssistant(
	ctx context.Context,
	llm model.LLM,
) error { //nolint:unparam // consistent function signature
	fmt.Println("\n=== Technical Assistant ===")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Explain how OAuth 2.0 works"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: `You are an expert software architect with 20 years of experience.
Your expertise includes:
- Security protocols and authentication
- System design and architecture
- Cloud infrastructure

When explaining technical concepts:
1. Start with the core principle
2. Provide a step-by-step breakdown
3. Include practical examples
4. Mention common pitfalls and best practices
5. Suggest when to use alternatives

Use technical terminology accurately. Assume the user has intermediate knowledge.`},
				},
			},
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response:\n%s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateJSONOutputFormatter shows how to format structured output as JSON.
func demonstrateJSONOutputFormatter(
	ctx context.Context,
	llm model.LLM,
) error { //nolint:unparam // consistent function signature
	fmt.Println("\n=== JSON Output Formatter ===")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Analyze the sentiment of: 'I love this product! It's amazing!'"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: `You are a sentiment analysis system. Respond ONLY with valid JSON in this format:
{
  "text": "the input text",
  "sentiment": "positive|negative|neutral",
  "score": 0.0-1.0,
  "keywords": ["key", "words"],
  "reasoning": "brief explanation"
}

Do not include any text outside the JSON object.`},
				},
			},
		},
	}

	var responseText strings.Builder
	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					responseText.WriteString(part.Text)
				}
			}
		}
	}

	// Try to parse and display the JSON
	responseTextValue := responseText.String()
	var result map[string]any
	if err := json.Unmarshal([]byte(responseTextValue), &result); err != nil {
		fmt.Printf("Raw response: %s\n", responseTextValue)
	} else {
		fmt.Printf("Parsed response:\n%s\n", formatJSON(result))
	}

	return nil
}

// demonstrateCreativeWriter shows how system instruction shapes creative output.
func demonstrateCreativeWriter(
	ctx context.Context,
	llm model.LLM,
) error { //nolint:unparam // consistent function signature
	fmt.Println("\n=== Creative Writer ===")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Write a haiku about rain"},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: `You are a celebrated poet and writer.
Your style is:
- Evocative and sensory-rich
- Uses literary devices like metaphor and alliteration
- Creates emotional resonance
- Maintains perfect form and structure

For haikus specifically:
- Exactly 5-7-5 syllables
- Focus on nature and seasons
- Include a seasonal reference (kigo)
- Create a moment of realization (aha moment)`},
				},
			},
		},
	}

	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Poem:\n%s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateConversationWithConsistentContext shows multi-turn with system instruction.
func demonstrateConversationWithConsistentContext(
	ctx context.Context,
	llm model.LLM,
) error { //nolint:unparam // consistent function signature
	fmt.Println("\n=== Multi-turn Conversation with Consistent System Context ===")

	systemInstruction := &genai.Content{
		Parts: []*genai.Part{
			{Text: `You are a friendly Python tutor with 15 years of teaching experience.
Your teaching style:
- Break down complex concepts into simple parts
- Use real-world examples
- Provide code examples when helpful
- Encourage questions
- Correct misconceptions gently`},
		},
	}

	// First turn
	contents := []*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{Text: "What is a list comprehension?"},
			},
		},
	}

	req := &model.LLMRequest{
		Contents: contents,
		Config: &genai.GenerateContentConfig{
			SystemInstruction: systemInstruction,
		},
	}

	var firstResponse strings.Builder
	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					firstResponse.WriteString(part.Text)
				}
			}
		}
	}

	firstResponseText := firstResponse.String()
	fmt.Printf("Tutor: %s\n\n", firstResponseText)

	// Add to conversation history
	contents = append(contents, &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{
			{Text: firstResponseText},
		},
	})

	// Second turn
	contents = append(contents, &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{Text: "Can you show me an example?"},
		},
	})

	req = &model.LLMRequest{
		Contents: contents,
		Config: &genai.GenerateContentConfig{
			SystemInstruction: systemInstruction,
		},
	}

	var secondResponse strings.Builder
	for resp := range llm.GenerateContent(ctx, req, false) {
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					secondResponse.WriteString(part.Text)
				}
			}
		}
	}

	fmt.Printf("Student: Your question was answered!\n")
	return nil
}

// formatJSON returns a pretty-printed JSON string.
func formatJSON(data any) string {
	b, _ := json.MarshalIndent(data, "", "  ")
	return string(b)
}

func main() {
	ctx := context.Background()

	// Load AWS configuration
	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		log.Fatalf("load AWS config: %v", err)
	}
	if awsCfg.Region == "" {
		log.Fatal("AWS region is unset: set AWS_REGION or add region to ~/.aws/config")
	}

	// Get model ID
	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID is required (e.g., us.anthropic.claude-3-5-sonnet-20241022-v2:0)")
		modelID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
	}

	// Create Bedrock LLM
	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br))
	if err != nil {
		log.Fatalf("create bedrock model: %v", err)
	}

	// Run demonstrations
	if err := demonstrateTechnicalAssistant(ctx, llm); err != nil {
		log.Printf("technical assistant: %v", err)
	}

	if err := demonstrateJSONOutputFormatter(ctx, llm); err != nil {
		log.Printf("JSON formatter: %v", err)
	}

	if err := demonstrateCreativeWriter(ctx, llm); err != nil {
		log.Printf("creative writer: %v", err)
	}

	if err := demonstrateConversationWithConsistentContext(ctx, llm); err != nil {
		log.Printf("conversation with context: %v", err)
	}
}
