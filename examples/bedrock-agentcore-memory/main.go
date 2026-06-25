// Bedrock AgentCore Memory example for adk-go: persist a prior conversation to an
// Amazon Bedrock AgentCore Memory resource and let an agent recall it in a new
// session via the ADK load_memory / preload_memory tools.
//
// Authenticate with AWS using the default credential chain (environment variables,
// shared config, SSO / AWS_PROFILE, EC2/ECS/Lambda role, etc.) and set:
//
//	BEDROCK_MODEL_ID        Bedrock model ID or inference profile ARN (LLM)
//	AGENTCORE_MEMORY_ID     AgentCore Memory resource ID (required)
//	AGENTCORE_NAMESPACE     namespace prefix to search, e.g. /actors/{actorId}/facts (required)
//	AGENTCORE_STRATEGY_ID   memory strategy ID to filter retrieval (optional)
//	AWS_REGION              region of the Memory resource (optional if set on profile)
//
// Long-term memory extraction in AgentCore is asynchronous, so a memory written in
// this run is usually NOT searchable immediately — re-run the recall query after
// extraction completes.
//
//	go run ./examples/bedrock-agentcore-memory
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/loadmemorytool"
	"google.golang.org/adk/tool/preloadmemorytool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/memory/agentcore"
)

func main() {
	ctx := context.Background()

	region := strings.TrimSpace(os.Getenv("AWS_REGION"))

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Println("BEDROCK_MODEL_ID is required (e.g. eu.amazon.nova-2-lite-v1:0) using default model")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	memoryID := strings.TrimSpace(os.Getenv("AGENTCORE_MEMORY_ID"))
	if memoryID == "" {
		log.Fatal("AGENTCORE_MEMORY_ID is required (the AgentCore Memory resource ID)")
	}
	namespace := strings.TrimSpace(os.Getenv("AGENTCORE_NAMESPACE"))
	if namespace == "" {
		log.Fatal("AGENTCORE_NAMESPACE is required (e.g. /actors/{actorId}/facts); it must match your memory strategy")
	}

	llm, err := bedrock.New(ctx, modelID, &bedrock.Options{Region: region})
	if err != nil {
		log.Fatalf("bedrock model: %v", err)
	}

	memSvc, err := agentcore.New(ctx, &agentcore.Config{
		MemoryID:   memoryID,
		Region:     region,
		Namespace:  namespace,
		StrategyID: strings.TrimSpace(os.Getenv("AGENTCORE_STRATEGY_ID")),
		TopK:       5,
	})
	if err != nil {
		log.Fatalf("agentcore memory service: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "memory_assistant",
		Model:       llm,
		Description: "Agent that can recall information from AgentCore Memory.",
		Instruction: "You are a helpful assistant with access to long-term memory. " +
			"Relevant memory may be preloaded automatically. If it is not enough, " +
			"use the load_memory tool to search for more. Use what you recall to answer.",
		Tools: []tool.Tool{
			preloadmemorytool.New(),
			loadmemorytool.New(),
		},
		GenerateContentConfig: &genai.GenerateContentConfig{MaxOutputTokens: 512},
	})
	if err != nil {
		log.Fatalf("agent: %v", err)
	}

	userID, appName := "demo-user", "bedrock-agentcore-memory-example"
	sessionService := session.InMemoryService()

	// Seed a prior conversation and persist it to AgentCore Memory.
	prior, err := seedPriorSession(ctx, sessionService, appName, userID)
	if err != nil {
		log.Fatalf("seed prior session: %v", err)
	}
	if err := memSvc.AddSessionToMemory(ctx, prior); err != nil {
		log.Fatalf("add session to memory: %v", err)
	}
	fmt.Println("Wrote a prior conversation about a Tokyo trip to AgentCore Memory.")
	fmt.Println("Note: long-term extraction is asynchronous — recall may be empty until it completes.")

	// New session that should be able to recall the prior conversation.
	resp, err := sessionService.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		log.Fatalf("create session: %v", err)
	}
	current := resp.Session

	r, err := runner.New(runner.Config{
		AppName:        appName,
		Agent:          a,
		SessionService: sessionService,
		MemoryService:  memSvc,
	})
	if err != nil {
		log.Fatalf("runner: %v", err)
	}

	userMsg := "What do you remember about my trip to Tokyo?"
	if len(os.Args) > 1 {
		userMsg = os.Args[1]
	}
	printRecall(ctx, r, a.Name(), userID, current.ID(), userMsg)
}

// printRecall runs the agent once and prints its (non-partial) text response.
func printRecall(ctx context.Context, r *runner.Runner, agentName, userID, sessionID, prompt string) {
	fmt.Printf("\nUser  -> %s\nAgent -> ", prompt)
	for ev, err := range r.Run(ctx, userID, sessionID, genai.NewContentFromText(prompt, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("run: %v", err)
		}
		if ev.Author != agentName || ev.LLMResponse.Partial || ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			if p.Text != "" {
				fmt.Print(p.Text)
			}
		}
	}
	fmt.Println()
}

// seedPriorSession creates an in-memory session with a short conversation that is
// later written to AgentCore Memory.
func seedPriorSession(ctx context.Context, svc session.Service, appName, userID string) (session.Session, error) {
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		return nil, err
	}
	s := resp.Session

	turns := []struct{ author, content string }{
		{"user", "I just got back from an amazing trip to Tokyo!"},
		{"model", "That sounds wonderful! What were the highlights?"},
		{"user", "I visited Senso-ji temple in Asakusa and had ramen in Shinjuku."},
		{"model", "Great choices! Did you see any other sights?"},
		{"user", "I went up Tokyo Skytree and saw Mount Fuji in the distance."},
	}
	for _, turn := range turns {
		ev := session.NewEvent("prior-session")
		ev.Author = turn.author
		ev.Content = genai.NewContentFromText(turn.content, genai.Role(turn.author))
		if err := svc.AppendEvent(ctx, s, ev); err != nil {
			return nil, err
		}
	}
	return s, nil
}
