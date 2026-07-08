// Bedrock AgentCore Code Interpreter example for adk-go. Set BEDROCK_MODEL_ID,
// AWS_REGION, optional AGENTCORE_REGION, and AWS credentials.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/converse"
	"github.com/craigh33/adk-go-bedrock/tools/agentcorecodeinterpreter"
)

const (
	appName                  = "bedrock-agentcore-code-interpreter-example"
	defaultCodeInterpreterID = "aws.codeinterpreter.v1"
	userID                   = "local-user"
	sessionID                = "demo-session"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context) error {
	codeInterpreterID := envOrDefault("AGENTCORE_CODE_INTERPRETER_ID", defaultCodeInterpreterID)

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return fmt.Errorf("load AWS config: %w", err)
	}
	if awsCfg.Region == "" {
		return errors.New("AWS region is unset: set AWS_REGION or configure a profile region")
	}

	modelID := envOrDefault("BEDROCK_MODEL_ID", "global.amazon.nova-2-lite-v1:0")
	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := converse.NewWithAPI(modelID, converse.NewRuntimeAPI(br))
	if err != nil {
		return fmt.Errorf("bedrock model: %w", err)
	}

	agentCoreCfg := awsCfg
	if r := strings.TrimSpace(os.Getenv("AGENTCORE_REGION")); r != "" {
		agentCoreCfg.Region = r
	}
	codeTool, err := agentcorecodeinterpreter.New(agentcorecodeinterpreter.Config{
		API:                       bedrockagentcore.NewFromConfig(agentCoreCfg),
		CodeInterpreterIdentifier: codeInterpreterID,
	})
	if err != nil {
		return fmt.Errorf("code interpreter tool: %w", err)
	}

	sessionService := session.InMemoryService()
	artifactService := artifact.InMemoryService()
	if _, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	}); err != nil {
		return fmt.Errorf("create session: %w", err)
	}
	if err := seedArtifacts(ctx, artifactService); err != nil {
		return err
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "agentcore-code-interpreter-assistant",
		Description: "An assistant that analyzes artifacts with AgentCore Code Interpreter",
		Model:       llm,
		Instruction: `Use execute_code for data analysis. For the seeded sales.csv artifact, load it with input_artifacts [{"artifact_name":"sales.csv","path":"sales.csv"}]. AgentCore artifact paths must be relative, so code should open "sales.csv" and write "summary.txt"; do not use absolute paths like /tmp. The CSV columns are region, product, and revenue. Use Python standard library code only, especially csv; do not use pandas or other third-party packages. When writing a summary file, save it through output_artifacts [{"path":"summary.txt","artifact_name":"summary.txt"}].`,
		Tools:       []tool.Tool{codeTool},
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		return fmt.Errorf("agent: %w", err)
	}

	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           a,
		SessionService:  sessionService,
		ArtifactService: artifactService,
	})
	if err != nil {
		return fmt.Errorf("runner: %w", err)
	}

	userMsg := `Analyze sales.csv by region using Python standard library only. The CSV columns are region, product, and revenue. Print the revenue totals and save a plain-text summary to summary.txt. Use relative sandbox paths only.`
	if len(os.Args) > 1 {
		userMsg = strings.Join(os.Args[1:], " ")
	}

	fmt.Printf("User: %s\n\n", userMsg)
	if err := printRunEvents(ctx, r, a, userMsg); err != nil {
		return err
	}
	printSummaryArtifact(ctx, artifactService)
	return nil
}

func seedArtifacts(ctx context.Context, artifactService artifact.Service) error {
	const csv = `region,product,revenue
EMEA,widgets,1200
AMER,widgets,1500
EMEA,gadgets,800
APAC,gadgets,950
AMER,gadgets,650
`
	_, err := artifactService.Save(ctx, &artifact.SaveRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  "sales.csv",
		Part:      genai.NewPartFromText(csv),
	})
	if err != nil {
		return fmt.Errorf("seed sales.csv artifact: %w", err)
	}
	return nil
}

func printRunEvents(ctx context.Context, r *runner.Runner, a agent.Agent, userMsg string) error {
	for ev, err := range r.Run(ctx, userID, sessionID, genai.NewContentFromText(userMsg, genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			return fmt.Errorf("run: %w", err)
		}
		if ev.Author != a.Name() || ev.LLMResponse.Partial || ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil {
				fmt.Printf("Tool call: %s(%v)\n", p.FunctionCall.Name, p.FunctionCall.Args)
			}
			if p.FunctionResponse != nil {
				fmt.Printf("Tool result: %v\n", p.FunctionResponse.Response)
			}
			if p.Text != "" {
				fmt.Print(p.Text)
			}
		}
		fmt.Println()
	}
	return nil
}

func printSummaryArtifact(ctx context.Context, artifactService artifact.Service) {
	resp, err := artifactService.Load(ctx, &artifact.LoadRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
		FileName:  "summary.txt",
	})
	if err == nil {
		if resp.Part != nil && resp.Part.Text != "" {
			fmt.Printf("\nSaved summary.txt:\n%s\n", resp.Part.Text)
		}
		return
	}
	fmt.Println("summary.txt artifact was not saved")
}

func envOrDefault(name, defaultValue string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	return value
}
