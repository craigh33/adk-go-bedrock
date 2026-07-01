// Bedrock Mantle chat example for adk-go. It drives an ADK runner through the
// Anthropic-compatible Bedrock Mantle endpoint instead of the native Converse
// API, reusing the same converse.Model via a Mantle RuntimeAPI implementation.
//
// Authenticate with either:
//   - an Anthropic/Bedrock API key: set AWS_BEARER_TOKEN_BEDROCK (or
//     ANTHROPIC_API_KEY), or
//   - the default AWS credential chain for SigV4 (env vars, shared config,
//     SSO / AWS_PROFILE, instance role, etc.).
//
// Set AWS_REGION (used for the Mantle base URL and SigV4 signing) and,
// optionally, BEDROCK_MANTLE_MODEL_ID. Run:
//
//	go run ./examples/bedrock-mantle-chat
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
	"github.com/craigh33/adk-go-bedrock/bedrock/mantle"
)

const defaultModelID = "anthropic.claude-sonnet-4-5-20250929-v1:0"

func main() {
	ctx := context.Background()

	region := strings.TrimSpace(os.Getenv("AWS_REGION"))
	if region == "" {
		log.Fatal("AWS_REGION is required (used for the Mantle base URL and SigV4 signing)")
	}

	modelID := strings.TrimSpace(os.Getenv("BEDROCK_MANTLE_MODEL_ID"))
	if modelID == "" {
		log.Printf("BEDROCK_MANTLE_MODEL_ID is unset; using default %q", defaultModelID)
		modelID = defaultModelID
	}

	// APIKey is optional: when empty, the Mantle client falls back to the
	// AWS_BEARER_TOKEN_BEDROCK / ANTHROPIC_AWS_API_KEY env vars and then to the
	// default AWS credential chain (SigV4).
	mantleClient, err := mantle.New(ctx, mantle.Config{
		AWSRegion: region,
		APIKey:    strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")),
	})
	if err != nil {
		log.Fatalf("bedrock mantle client: %v", err)
	}

	llm, err := converse.NewWithAPI(modelID, mantleClient)
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
		AppName:           "bedrock-mantle-chat-example",
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

	msg := genai.NewContentFromText(userMsg, genai.RoleUser)
	for ev, err := range r.Run(ctx, "local-user", "demo-session", msg, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run: %v", err)
		}
		if ev.Author != a.Name() || ev.LLMResponse.Partial {
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
