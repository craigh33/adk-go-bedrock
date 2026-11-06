package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/agentcore/decider"
	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
	decidertool "github.com/craigh33/adk-go-bedrock/tools/agentcore/decider"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	useTool := flag.Bool("tool", false, "Run the evaluator as an ADK tool with a Bedrock LLM")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	awsCfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return fmt.Errorf("load AWS config: %w", err)
	}
	client, err := decider.New(decider.Config{
		API:              bedrockagentcore.NewFromConfig(awsCfg),
		AgentRuntimeARN:  os.Getenv("AGENTCORE_RUNTIME_ARN"),
		Qualifier:        os.Getenv("AGENTCORE_RUNTIME_QUALIFIER"),
		RuntimeSessionID: os.Getenv("AGENTCORE_RUNTIME_SESSION_ID"),
	})
	if err != nil {
		return err
	}
	var question decider.Question
	if err = question.FromChoiceQuestion(decider.ChoiceQuestion{
		Type: "choice", Instructions: "Which team should handle the request?",
		Criteria: map[string]any{"billing": "Invoices and payments", "technical": "Bugs and integrations"},
	}); err != nil {
		return fmt.Errorf("define question: %w", err)
	}
	questions := map[string]decider.Question{"department": question}
	text := "I was charged twice. Please help."
	if flag.NArg() > 0 {
		text = strings.Join(flag.Args(), " ")
	}
	if !*useTool {
		response, evaluateErr := client.Evaluate(ctx, &decider.Request{State: text, Questions: questions})
		if evaluateErr != nil {
			return evaluateErr
		}
		if err = json.NewEncoder(os.Stdout).Encode(response); err != nil {
			return fmt.Errorf("print assessment: %w", err)
		}
		return nil
	}
	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		return errors.New("set BEDROCK_MODEL_ID for -tool")
	}
	evaluate, err := decidertool.New(decidertool.Config{
		API: client, Questions: questions, Name: "classify_request",
		Description: "Choose the team responsible for a support request",
	})
	if err != nil {
		return err
	}
	llm, err := converse.New(ctx, modelID, &converse.Options{Region: awsCfg.Region})
	if err != nil {
		return err
	}
	a, err := llmagent.New(llmagent.Config{
		Name: "support_assistant", Model: llm, Tools: []tool.Tool{evaluate},
		Instruction: "Use classify_request to assess the user's request, then explain the selected team.",
	})
	if err != nil {
		return fmt.Errorf("create agent: %w", err)
	}
	r, err := runner.New(runner.Config{AppName: "decider-example", Agent: a,
		SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		return fmt.Errorf("create runner: %w", err)
	}
	for event, runErr := range r.Run(ctx, "example-user", "example-session",
		genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{}) {
		if runErr != nil {
			return fmt.Errorf("run agent: %w", runErr)
		}
		if event.Content == nil || event.Partial {
			continue
		}
		for _, part := range event.Content.Parts {
			if part.Text != "" {
				fmt.Println(part.Text)
			}
		}
	}
	return nil
}
