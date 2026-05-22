// Command bedrock-live demonstrates the bedrock/live package against Amazon
// Nova Sonic. It reads 16 kHz LPCM audio from a file, streams it to Sonic,
// and writes the model's 24 kHz LPCM reply to an output file while printing
// interleaved transcripts.
//
// Play the response with ffplay (or sox):
//
//	ffplay -f s16le -ar 24000 -ac 1 /tmp/sonic-out.pcm
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"iter"
	"log"
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

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	var (
		audioIn  = flag.String("audio-in", "", "path to 16 kHz mono LE16 LPCM input file")
		audioOut = flag.String("audio-out", "/tmp/sonic-out.pcm", "path to write 24 kHz mono LE16 LPCM output")
		system   = flag.String("system", "You are a friendly voice assistant. Reply briefly.", "system instruction")
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
		return fmt.Errorf("load AWS config (check credentials and AWS_PROFILE): %w", err)
	}
	if awsCfg.Region == "" {
		return errors.New("AWS region is unset: set AWS_REGION or add region to your ~/.aws/config profile")
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

	// Read side runs in a goroutine so we can stream input concurrently.
	done := make(chan error, 1)
	go func() {
		done <- consumeEvents(events, outFile)
	}()

	if err := streamPCMFile(ctx, sess, *audioIn, *chunkMs); err != nil {
		return fmt.Errorf("stream audio: %w", err)
	}
	// End-of-utterance: tells Sonic to flush its VAD and respond.
	if err := sess.Send(agent.LiveRequest{RealtimeInput: &genai.ActivityEnd{}}); err != nil {
		log.Printf("send ActivityEnd: %v", err)
	}

	if err := <-done; err != nil {
		return fmt.Errorf("consume events: %w", err)
	}
	fmt.Printf("\nWrote %s — play with: ffplay -f s16le -ar 24000 -ac 1 %s\n", *audioOut, *audioOut)
	return nil
}

// streamPCMFile sends the file in chunkMs-sized slices, sleeping between
// chunks to roughly match real-time pacing so Sonic's VAD behaves as it would
// for a live microphone.
func streamPCMFile(ctx context.Context, sess *live.Session, path string, chunkMs int) error {
	if chunkMs <= 0 {
		chunkMs = 32
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	// 16 kHz mono 16-bit PCM → 32 bytes per ms.
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

// consumeEvents prints transcripts and writes audio bytes to out, stopping
// when the iterator yields a TurnComplete event or an error.
func consumeEvents(events iter.Seq2[*session.Event, error], out *os.File) error {
	var audioBytes int
	for ev, err := range events {
		if err != nil {
			return err
		}
		if ev == nil {
			continue
		}
		n, werr := handleEvent(ev, out)
		if werr != nil {
			return werr
		}
		audioBytes += n
		if ev.TurnComplete {
			fmt.Printf("[turn-complete] wrote %d audio bytes\n", audioBytes)
			return nil
		}
	}
	return nil
}

func handleEvent(ev *session.Event, out *os.File) (int, error) {
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
		return 0, nil
	}
	var audioBytes int
	for _, p := range ev.Content.Parts {
		n, err := handlePart(p, out)
		if err != nil {
			return audioBytes, err
		}
		audioBytes += n
	}
	return audioBytes, nil
}

func handlePart(p *genai.Part, out *os.File) (int, error) {
	if p == nil {
		return 0, nil
	}
	if p.FunctionCall != nil {
		fmt.Printf("[tool] %s(%v)\n", p.FunctionCall.Name, p.FunctionCall.Args)
	}
	if p.Text != "" {
		fmt.Printf("[text] %s\n", p.Text)
	}
	if p.InlineData == nil || len(p.InlineData.Data) == 0 {
		return 0, nil
	}
	return out.Write(p.InlineData.Data)
}
