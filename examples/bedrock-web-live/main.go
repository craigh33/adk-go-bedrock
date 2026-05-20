// Command bedrock-web-live launches the ADK local web UI with a
// Nova-Sonic-backed bidirectional voice flow on the mic button. Text chat
// goes through Bedrock Converse (any Bedrock chat model); the mic button
// goes through bedrock/live.Session via the bedrock/live/webbridge package.
//
// Because Nova 2 Sonic is only available in us-east-1, us-west-2, and
// ap-northeast-1, this example defaults the text model to
// amazon.nova-2-lite-v1:0 — available in every Sonic region — so a single
// AWS_REGION serves both flows.
package main

import (
	"context"
	"log"
	"log/slog"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/console"
	"google.golang.org/adk/cmd/launcher/universal"
	"google.golang.org/adk/cmd/launcher/web"
	"google.golang.org/adk/cmd/launcher/web/a2a"
	"google.golang.org/adk/cmd/launcher/web/api"
	"google.golang.org/adk/cmd/launcher/web/triggers/eventarc"
	"google.golang.org/adk/cmd/launcher/web/triggers/pubsub"
	"google.golang.org/adk/cmd/launcher/web/webui"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/bedrock/live"
	"github.com/craigh33/adk-go-bedrock/bedrock/live/webbridge"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

// defaultTextModel is available in every Nova 2 Sonic region
// (us-east-1, us-west-2, ap-northeast-1), so text and voice both work from
// a single AWS_REGION. Override with BEDROCK_MODEL_ID if you have access to
// a different text model in your Sonic region.
const defaultTextModel = "us.amazon.nova-2-lite-v1:0"

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

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		log.Printf("BEDROCK_MODEL_ID not set; defaulting text model to %s", defaultTextModel)
		modelID = defaultTextModel
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		log.Fatalf("tracer provider: %v", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br, bedrock.WithTracerProvider(tp)))
	if err != nil {
		log.Panicf("bedrock model: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "assistant",
		Description: "A helpful assistant. Use the mic button for voice.",
		Model:       llm,
		Instruction: "You reply briefly and clearly. For voice replies, prefer one or two sentences.",
		GenerateContentConfig: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	})
	if err != nil {
		log.Panicf("agent: %v", err)
	}

	launcherCfg := &launcher.Config{
		AgentLoader:     agent.NewSingleLoader(a),
		ArtifactService: artifact.InMemoryService(),
	}

	// Build the Nova-Sonic-backed /run_live bridge as a reusable sublauncher.
	// It must be mounted BEFORE api.NewLauncher() so its exact-path match
	// wins over the upstream catchall.
	bridgeHandler := webbridge.New(
		live.NewBidiRuntimeAPI(br, live.WithTracerProvider(tp)),
		webbridge.Options{
			SystemInstruction: "You are a friendly voice assistant. Reply briefly and clearly.",
			Logger:            slog.Default(),
		},
	)
	bridgeSub := webbridge.NewSublauncher(bridgeHandler, webbridge.SublauncherOptions{})

	l := universal.NewLauncher(
		console.NewLauncher(),
		web.NewLauncher(
			webui.NewLauncher(),
			bridgeSub,
			a2a.NewLauncher(),
			pubsub.NewLauncher(),
			eventarc.NewLauncher(),
			api.NewLauncher(),
		),
	)
	if err = l.Execute(ctx, launcherCfg, os.Args[1:]); err != nil {
		// G706 fires on the launcher's compiled help text but CommandLineSyntax
		// is built from our own usage descriptions, not user input.
		log.Panicf("Run failed: %v\n\n%s", err, l.CommandLineSyntax()) //nolint:gosec // G706 false positive
	}
}
