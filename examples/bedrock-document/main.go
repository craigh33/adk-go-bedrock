// Bedrock document upload example for debugging genai → Converse mapping and model calls.
//
// Use -dry-run to verify that your file maps to Bedrock document blocks without calling AWS.
// Use -combined to mimic a Web UI that puts the prompt and file in a single Part.
//
//	go run ./examples/bedrock-document -path ./memo.docx
//	go run ./examples/bedrock-document -dry-run -path ./report.pdf
//	go run ./examples/bedrock-document -stream -combined -path ./notes.docx
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/bedrock/mappers"
)

func main() {
	dryRun := flag.Bool("dry-run", false, "only run request mapping (no Bedrock API call)")
	stream := flag.Bool("stream", false, "use ConverseStream via GenerateContent(..., true)")
	combined := flag.Bool("combined", false, "put prompt text and file bytes in the same Part (Web UI–style)")
	prompt := flag.String(
		"prompt",
		"Briefly describe what this document contains in one or two sentences.",
		"user text sent with the document",
	)
	pathFlag := flag.String("path", "", "path to a document file (pdf, docx, etc.), or set DOCUMENT_PATH")
	flag.Parse()

	docPath := strings.TrimSpace(*pathFlag)
	if docPath == "" {
		docPath = strings.TrimSpace(os.Getenv("DOCUMENT_PATH"))
	}
	if docPath == "" {
		log.Fatal("usage: go run . -path /path/to/file.docx   (or set DOCUMENT_PATH)")
	}

	docPath = filepath.Clean(docPath)
	data, err := os.ReadFile(docPath) // #nosec G304 -- path comes from -path / DOCUMENT_PATH (operator-chosen file)
	if err != nil {
		log.Fatalf("read file: %v", err)
	}
	mime := mappers.MIMETypeFromExtension(docPath)
	base := filepath.Base(docPath)

	modelID := strings.TrimSpace(os.Getenv("BEDROCK_MODEL_ID"))
	if modelID == "" {
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	req := buildRequest(*prompt, data, mime, base, *combined)

	fmt.Printf(
		"Document debug\n  path:     %s\n  size:     %d bytes\n  mime:     %s\n  model:    %s\n  combined: %v\n  stream:   %v\n\n",
		docPath,
		len(data),
		mime,
		modelID,
		*combined,
		*stream,
	)

	if *dryRun {
		runDryRun(modelID, req)
		return
	}

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

	if strings.TrimSpace(os.Getenv("BEDROCK_MODEL_ID")) == "" {
		log.Println("BEDROCK_MODEL_ID unset; using default model id for this example")
	}

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br))
	if err != nil {
		log.Fatalf("bedrock model: %v", err)
	}

	if *stream {
		runStream(ctx, llm, req)
		return
	}
	runUnary(ctx, llm, req)
}

func buildRequest(prompt string, data []byte, mime, displayName string, combined bool) *model.LLMRequest {
	var parts []*genai.Part
	if combined {
		parts = []*genai.Part{{
			Text: prompt,
			InlineData: &genai.Blob{
				Data:        data,
				MIMEType:    mime,
				DisplayName: displayName,
			},
		}}
	} else {
		parts = []*genai.Part{
			genai.NewPartFromText(prompt),
			{
				InlineData: &genai.Blob{
					Data:        data,
					MIMEType:    mime,
					DisplayName: displayName,
				},
			},
		}
	}
	return &model.LLMRequest{
		Contents: []*genai.Content{{
			Role:  genai.RoleUser,
			Parts: parts,
		}},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 1024,
		},
	}
}

func runDryRun(modelID string, req *model.LLMRequest) {
	in, err := mappers.ConverseInputFromLLMRequest(modelID, req)
	if err != nil {
		log.Fatalf("mapping failed (fix mapper or request): %v", err)
	}
	fmt.Println("dry-run: mapping OK — would send to Bedrock Converse:")
	if in.ModelId != nil {
		fmt.Printf("  ModelId: %s\n", *in.ModelId)
	}
	for i, msg := range in.Messages {
		fmt.Printf("  message[%d] role=%s blocks=%d\n", i, msg.Role, len(msg.Content))
		for j, blk := range msg.Content {
			fmt.Printf("    block[%d] %s\n", j, contentBlockSummary(blk))
		}
	}
	fmt.Println("\nNext: run without -dry-run to call the model (uses AWS credentials).")
}

func contentBlockSummary(b types.ContentBlock) string {
	switch b.(type) {
	case *types.ContentBlockMemberText:
		return "text"
	case *types.ContentBlockMemberDocument:
		return "document"
	case *types.ContentBlockMemberImage:
		return "image"
	case *types.ContentBlockMemberAudio:
		return "audio"
	case *types.ContentBlockMemberVideo:
		return "video"
	case *types.ContentBlockMemberToolUse:
		return "tool_use"
	case *types.ContentBlockMemberToolResult:
		return "tool_result"
	case *types.ContentBlockMemberReasoningContent:
		return "reasoning"
	default:
		return fmt.Sprintf("%T", b)
	}
}

func runUnary(ctx context.Context, llm model.LLM, req *model.LLMRequest) {
	fmt.Println("--- response (unary) ---")
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			log.Fatalf("GenerateContent: %v", err)
		}
		if resp == nil || resp.Content == nil {
			continue
		}
		for _, part := range resp.Content.Parts {
			if part.Text != "" {
				fmt.Println(part.Text)
			}
		}
		if resp.UsageMetadata != nil {
			fmt.Printf("\n(tokens) prompt=%d candidates=%d total=%d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}
}

func runStream(ctx context.Context, llm model.LLM, req *model.LLMRequest) {
	fmt.Println("--- response (stream) ---")
	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			log.Fatalf("stream: %v", err)
		}
		if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			continue
		}
		if resp.Partial {
			if t := resp.Content.Parts[0].Text; t != "" {
				fmt.Print(t)
			}
			continue
		}
		fmt.Println("\n--- final chunk ---")
		for _, p := range resp.Content.Parts {
			if p.Text != "" {
				fmt.Println(p.Text)
			}
		}
		if resp.UsageMetadata != nil {
			fmt.Printf("(tokens) prompt=%d candidates=%d total=%d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}
}
