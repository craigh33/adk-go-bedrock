// Command bedrock-live-tool demonstrates the bedrock/live package's
// auto-execute tool loop ([live.Session.RunAgentLoop]) against Amazon Nova 2
// Sonic. Reads 16 kHz LPCM audio from a file, streams it to Sonic, invokes a
// fake weather tool when the model asks for it, and writes the model's
// 24 kHz LPCM reply to disk.
//
// Play the response with ffplay (or sox):
//
//	ffplay -f s16le -ar 24000 -ac 1 /tmp/sonic-tool-out.pcm
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock/live"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

const (
	weatherToolName = "get_weather"
	locationArg     = "location"
)

func weatherTool() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        weatherToolName,
		Description: "Get current weather for a location (temperature in Fahrenheit, condition).",
		ParametersJsonSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				locationArg: map[string]any{
					"type":        "string",
					"description": "City name, e.g. \"Boston\" or \"Tokyo\".",
				},
			},
			"required": []any{locationArg},
		},
	}
}

//nolint:gosec // G404: fake weather only.
func runWeather(_ context.Context, args map[string]any) (map[string]any, error) {
	location, _ := args[locationArg].(string)
	if location == "" {
		return nil, errors.New("location is required")
	}
	conditions := []string{"sunny", "cloudy", "rainy", "windy"}
	return map[string]any{
		locationArg: location,
		"temp_f":    rand.IntN(60) + 40, // 40–100°F
		"condition": conditions[rand.IntN(len(conditions))],
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}, nil
}

const defaultSystemPrompt = "You are a friendly voice assistant. When the user asks about the weather, " +
	"call the get_weather tool with their location and then describe the result."

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	var (
		audioIn  = flag.String("audio-in", "", "path to 16 kHz mono LE16 LPCM input file")
		audioOut = flag.String("audio-out", "/tmp/sonic-tool-out.pcm", "path to write 24 kHz mono LE16 LPCM output")
		system   = flag.String("system", defaultSystemPrompt, "system instruction")
		chunkMs  = flag.Int("chunk-ms", 32, "audio input chunk size in milliseconds")
	)
	flag.Parse()

	if *audioIn == "" {
		return errors.New("--audio-in is required (16 kHz mono LE16 LPCM file)")
	}

	var loadOpts []func(*config.LoadOptions) error
	if r := strings.TrimSpace(os.Getenv("AWS_REGION")); r != "" {
		loadOpts = append(loadOpts, config.WithRegion(r))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return fmt.Errorf("load AWS config: %w", err)
	}
	if awsCfg.Region == "" {
		return errors.New("AWS region is unset: set AWS_REGION")
	}

	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		return fmt.Errorf("tracer provider: %w", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	api := live.NewBidiRuntimeAPI(bedrockruntime.NewFromConfig(awsCfg), live.WithTracerProvider(tp))

	modelID := os.Getenv("BEDROCK_MODEL_ID")
	if modelID == "" {
		modelID = live.DefaultModelID
	}
	fmt.Printf("Opening session: model=%s region=%s\n", modelID, awsCfg.Region)

	sess, events, err := live.Open(ctx, api, modelID, &agent.LiveRunConfig{
		ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityAudio},
	}, &live.OpenOptions{
		SystemInstruction: *system,
		InputSampleRateHz: 16000,
		Tools: map[string]any{
			weatherToolName: weatherTool(),
		},
	})
	if err != nil {
		return fmt.Errorf("open Nova Sonic session: %w", err)
	}
	defer func() {
		if cerr := sess.Close(); cerr != nil {
			log.Printf("close session: %v", cerr)
		}
	}()

	outFile, err := os.Create(*audioOut)
	if err != nil {
		return fmt.Errorf("create %s: %w", *audioOut, err)
	}
	defer outFile.Close()

	tools := live.ToolRegistry{
		weatherToolName: runWeather,
	}

	// Stream the audio asynchronously; RunAgentLoop owns the read side.
	streamErr := make(chan error, 1)
	go func() {
		if err := streamPCMFile(ctx, sess, *audioIn, *chunkMs); err != nil {
			streamErr <- err
			return
		}
		streamErr <- sess.Send(agent.LiveRequest{RealtimeInput: &genai.ActivityEnd{}})
	}()

	emit := func(ev *session.Event) { renderEvent(ev, outFile) }
	if err := sess.RunAgentLoop(ctx, events, tools, emit); err != nil {
		return fmt.Errorf("run agent loop: %w", err)
	}
	if err := <-streamErr; err != nil {
		log.Printf("audio sender finished with: %v", err)
	}
	fmt.Printf("\nWrote %s — play with: ffplay -f s16le -ar 24000 -ac 1 %s\n", *audioOut, *audioOut)
	return nil
}

func streamPCMFile(ctx context.Context, sess *live.Session, path string, chunkMs int) error {
	if chunkMs <= 0 {
		chunkMs = 32
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	bytesPerChunk := chunkMs * 32
	if bytesPerChunk%2 != 0 {
		bytesPerChunk++
	}
	pace := time.Duration(chunkMs) * time.Millisecond
	for off := 0; off < len(data); off += bytesPerChunk {
		end := min(off+bytesPerChunk, len(data))
		if err := sess.Send(agent.LiveRequest{
			RealtimeInput: &genai.Blob{
				MIMEType: "audio/pcm;rate=16000",
				Data:     data[off:end],
			},
		}); err != nil {
			return err
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pace):
		}
	}
	return nil
}

func renderEvent(ev *session.Event, out *os.File) {
	if ev == nil {
		return
	}
	if ev.Interrupted {
		fmt.Println("[interrupted]")
	}
	if ev.InputTranscription != nil && ev.InputTranscription.Text != "" {
		fmt.Printf("[user] %s\n", ev.InputTranscription.Text)
	}
	if ev.OutputTranscription != nil && ev.OutputTranscription.Text != "" {
		fmt.Printf("[model] %s\n", ev.OutputTranscription.Text)
	}
	if ev.Content == nil {
		return
	}
	for _, p := range ev.Content.Parts {
		switch {
		case p == nil:
		case p.FunctionCall != nil:
			fmt.Printf("[tool-call] %s(%v)\n", p.FunctionCall.Name, p.FunctionCall.Args)
		case p.FunctionResponse != nil:
			fmt.Printf("[tool-result] %s -> %v\n", p.FunctionResponse.Name, p.FunctionResponse.Response)
		case p.InlineData != nil && len(p.InlineData.Data) > 0:
			if _, err := out.Write(p.InlineData.Data); err != nil {
				log.Printf("write audio: %v", err)
			}
		case p.Text != "":
			fmt.Printf("[text] %s\n", p.Text)
		}
	}
}
