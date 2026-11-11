// Package main demonstrates an ADK agent that routes support requests with AgentCore Decider.
package main

import (
	"context"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/full"
	"google.golang.org/adk/v2/tool"

	"github.com/craigh33/adk-go-bedrock/bedrock/agentcore/decider"
	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
	decidertool "github.com/craigh33/adk-go-bedrock/tools/agentcore/decider"
)

func main() {
	ctx := context.Background()

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	runtimeARN := os.Getenv("AGENTCORE_RUNTIME_ARN")
	if modelID == "" || runtimeARN == "" {
		log.Fatal("set BEDROCK_MODEL_ID and AGENTCORE_RUNTIME_ARN")
	}

	awsCfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatalf("load AWS config: %v", err)
	}
	if awsCfg.Region == "" {
		log.Fatal("set AWS_REGION or configure a region in your AWS profile")
	}

	runtime := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := converse.NewWithAPI(modelID, converse.NewRuntimeAPI(runtime))
	if err != nil {
		log.Fatalf("create Bedrock model: %v", err)
	}

	client, err := decider.New(decider.Config{
		API:              bedrockagentcore.NewFromConfig(awsCfg),
		AgentRuntimeARN:  runtimeARN,
		Qualifier:        os.Getenv("AGENTCORE_RUNTIME_QUALIFIER"),
		RuntimeSessionID: os.Getenv("AGENTCORE_RUNTIME_SESSION_ID"),
	})
	if err != nil {
		log.Fatalf("create Decider client: %v", err)
	}

	var department decider.Question
	if err := department.FromChoiceQuestion(decider.ChoiceQuestion{
		Type:         "choice",
		Instructions: "Which team should handle this support request?",
		Criteria: map[string]any{
			"billing":   "Invoices, charges, and payments",
			"technical": "Bugs and integration failures",
		},
	}); err != nil {
		log.Fatalf("define the routing question: %v", err)
	}

	classifyRequest, err := decidertool.New(decidertool.Config{
		API:         client,
		Name:        "classify_request",
		Description: "Choose the team responsible for a support request.",
		Questions: map[string]decider.Question{
			"department": department,
		},
	})
	if err != nil {
		log.Fatalf("create Decider tool: %v", err)
	}

	supportAgent, err := llmagent.New(llmagent.Config{
		Name:        "support_agent",
		Model:       llm,
		Description: "Routes support requests to billing or technical support.",
		Instruction: "Pass each support request to classify_request. " +
			"Report the department and confidence from the tool result.",
		Tools: []tool.Tool{
			classifyRequest,
		},
	})
	if err != nil {
		log.Fatalf("create support agent: %v", err)
	}

	launcherConfig := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(supportAgent),
	}
	l := full.NewLauncher()
	if err := l.Execute(ctx, launcherConfig, os.Args[1:]); err != nil {
		log.Fatalf("run agent: %v\n\n%s", err, l.CommandLineSyntax())
	}
}
