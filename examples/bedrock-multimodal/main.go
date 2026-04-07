// Bedrock multimodal example for adk-go: demonstrates image analysis, document processing,
// vision-based reasoning, tool calling with rich media, and other multimodal content with
// the Bedrock Converse provider. Set BEDROCK_MODEL_ID to a multimodal-capable model and
// authenticate with AWS using the default credential chain. Run:
//
//	go run ./examples/bedrock-multimodal
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
	"github.com/craigh33/adk-go-bedrock/examples/internal/exampletrace"
)

// downloadImage fetches an image from a URL and returns its raw binary content.
func downloadImage(url string) ([]byte, error) {
	resp, err := http.Get(url) //nolint:gosec,noctx // demonstration utility accepts arbitrary URLs
	if err != nil {
		return nil, fmt.Errorf("download image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download image: status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read image data: %w", err)
	}

	return data, nil
}

// mimeTypeFromURL infers the image MIME type from the URL file extension, defaulting to JPEG.
func mimeTypeFromURL(rawURL string) string {
	lower := strings.ToLower(rawURL)
	// Strip any query string before checking the extension.
	if i := strings.IndexByte(lower, '?'); i >= 0 {
		lower = lower[:i]
	}
	switch {
	case strings.HasSuffix(lower, ".png"):
		return "image/png"
	case strings.HasSuffix(lower, ".gif"):
		return "image/gif"
	case strings.HasSuffix(lower, ".webp"):
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

// analyzeImageWithVision demonstrates image analysis with the Bedrock model.
func analyzeImageWithVision(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Image Analysis Example ===")
	fmt.Printf("Downloading image from: %s\n", imageURL)

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "What do you see in this image? Please describe it in detail."},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "You are a helpful vision assistant. Provide detailed, accurate descriptions of images."},
				},
			},
		},
	}

	// Generate response
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Response: %s\n", part.Text)
				}
			}
		}
		if resp.UsageMetadata != nil {
			fmt.Printf("Usage - Prompt: %d, Candidates: %d, Total: %d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}

	return nil
}

// analyzeImageWithSystemInstruction demonstrates using system instructions for vision tasks.
func analyzeImageWithSystemInstruction(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Image Analysis with System Instruction ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	// Create a request with detailed system instruction for image analysis
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Analyze this image"},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: `You are an expert image analyst. When analyzing images:
1. Identify all objects and their relationships
2. Describe colors, lighting, and composition
3. Infer context and possible purposes
4. Note any text or symbols visible
5. Provide a concise but comprehensive summary`},
				},
			},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Analysis: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateConversationHistory shows how to maintain conversation history with images.
func demonstrateConversationHistory(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Conversation History with Images ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	// First turn: user provides image
	contents := []*genai.Content{
		{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{Text: "Here's an image. What do you see?"},
				{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
			},
		},
	}

	req := &model.LLMRequest{
		Contents: contents,
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: "Be concise but informative."},
				},
			},
		},
	}

	var modelResponse strings.Builder
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					modelResponse.WriteString(part.Text)
				}
			}
		}
	}

	modelResponseText := modelResponse.String()
	fmt.Printf("Model: %s\n", modelResponseText)

	// Second turn: follow-up question without re-sending the image
	contents = append(contents, &genai.Content{
		Role: genai.RoleModel,
		Parts: []*genai.Part{
			{Text: modelResponseText},
		},
	})

	contents = append(contents, &genai.Content{
		Role: genai.RoleUser,
		Parts: []*genai.Part{
			{Text: "What colors are most prominent in the image?"},
		},
	})

	req = &model.LLMRequest{
		Contents: contents,
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Follow-up: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateVisionWithReasoning shows how extended thinking works with multimodal content.
func demonstrateVisionWithReasoning(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Vision with Extended Reasoning ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	// Create a request that asks for reasoning about the image
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						Text: "Carefully analyze this image and identify all the key visual elements, their relationships, and what story or message they convey. Think step by step.",
					},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{Text: `You are an expert visual analyst with expertise in:
- Semiotics and visual communication
- Composition and design principles
- Cultural and contextual interpretation
- Detailed observation and reporting

Provide thorough, insightful analysis with specific observations.`},
				},
			},
			MaxOutputTokens: 1024,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Analysis: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateToolCallingWithImages demonstrates tool calling with image inputs.
func demonstrateToolCallingWithImages(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Tool Calling with Images ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	// Define a tool for tagging images
	imageTaggingTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "tag_image",
				Description: "Tag an image with relevant keywords and metadata",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"primary_subject": {
							Type:        genai.TypeString,
							Description: "The main subject or focus of the image",
						},
						"tags_csv": {
							Type:        genai.TypeString,
							Description: "Comma-separated relevant tags for the image",
						},
						"colors_csv": {
							Type:        genai.TypeString,
							Description: "Comma-separated dominant colors in the image",
						},
						"style": {
							Type:        genai.TypeString,
							Description: "Visual style or aesthetic (e.g., modern, vintage, abstract)",
						},
					},
					Required: []string{"primary_subject", "tags_csv"},
				},
			},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						Text: "Please analyze this image and tag it using the tag_image tool. Return comma-separated strings for tags_csv and colors_csv.",
					},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools:           []*genai.Tool{imageTaggingTool},
			MaxOutputTokens: 512,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					fmt.Printf("Tool Called: %s\n", part.FunctionCall.Name)
					if part.FunctionCall.Args != nil {
						argsJSON, _ := json.MarshalIndent(part.FunctionCall.Args, "", "  ")
						fmt.Printf("Arguments:\n%s\n", argsJSON)
					}
				}
			}
		}
	}

	return nil
}

// demonstrateMultipleImages shows how to analyze multiple images in sequence.
func demonstrateMultipleImages(ctx context.Context, llm model.LLM) error {
	fmt.Println("\n=== Comparing Multiple Images ===")

	// Use two different public images
	imageURLs := []string{
		"https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png",
		"https://www.google.com/images/branding/googleg/1x/googleg_standard_color_128dp.png",
	}

	type imageBlob struct {
		data     []byte
		mimeType string
	}
	blobs := make([]imageBlob, 0, len(imageURLs))
	for _, url := range imageURLs {
		data, err := downloadImage(url)
		if err != nil {
			fmt.Printf("Note: Could not download image from %s: %v\n", url, err)
			continue
		}
		blobs = append(blobs, imageBlob{data: data, mimeType: mimeTypeFromURL(url)})
	}

	if len(blobs) < 2 {
		fmt.Println("Note: Could not download enough images for comparison")
		return nil
	}

	// Create comparison request
	parts := []*genai.Part{
		{Text: "Compare these two images. What are the similarities and differences?"},
	}
	for _, b := range blobs {
		parts = append(parts, &genai.Part{
			InlineData: &genai.Blob{MIMEType: b.mimeType, Data: b.data},
		})
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role:  genai.RoleUser,
				Parts: parts,
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					{
						Text: "You are a visual comparison expert. Provide detailed analysis of similarities and differences.",
					},
				},
			},
			MaxOutputTokens: 512,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Comparison: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateStreamingWithImages shows streaming responses with image content.
func demonstrateStreamingWithImages(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Streaming Analysis of Images ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Describe this image in detail, streaming your analysis as you go:"},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 512,
		},
	}

	chunkCount := 0
	for resp, err := range llm.GenerateContent(ctx, req, true) { // true = streaming
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Print(part.Text)
					if resp.Partial {
						chunkCount++
					}
				}
			}
		}
	}
	fmt.Printf("\n(Streamed in %d chunks)\n", chunkCount)

	return nil
}

// demonstrateMediaInToolResponses shows how to use rich media in function responses.
func demonstrateMediaInToolResponses(ctx context.Context, llm model.LLM, imageURL string) error { //nolint:funlen
	fmt.Println("\n=== Media in Tool Responses ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	// First request: ask to analyze and call a tool
	imageLabelTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "analyze_image_regions",
				Description: "Analyze specific regions of an image",
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"regions_csv": {
							Type:        genai.TypeString,
							Description: "Comma-separated regions to analyze (for example: top,bottom,center)",
						},
					},
				},
			},
		},
	}

	mimeType := mimeTypeFromURL(imageURL)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						Text: "Analyze the regions of this image. If you call the tool, pass the regions as a comma-separated string in regions_csv, then use the tool result to provide an overall assessment.",
					},
					{InlineData: &genai.Blob{MIMEType: mimeType, Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools:           []*genai.Tool{imageLabelTool},
			MaxOutputTokens: 512,
		},
	}

	var modelToolParts []*genai.Part
	var toolResultParts []*genai.Part
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					fmt.Printf("Tool call: %s\n", part.FunctionCall.Name)
					modelToolParts = append(modelToolParts, part)
					toolResultParts = append(toolResultParts, &genai.Part{FunctionResponse: &genai.FunctionResponse{
						ID:   part.FunctionCall.ID,
						Name: part.FunctionCall.Name,
						Response: map[string]any{
							"results": "Analysis complete",
						},
						Parts: []*genai.FunctionResponsePart{{
							InlineData: &genai.FunctionResponseBlob{MIMEType: mimeType, Data: imageData},
						}},
					}})
				}
			}
		}
	}

	if len(toolResultParts) == 0 {
		fmt.Println("No tool call was returned; skipping tool-response follow-up.")
		return nil
	}

	followupReq := &model.LLMRequest{
		Contents: []*genai.Content{
			req.Contents[0],
			{Role: genai.RoleModel, Parts: modelToolParts},
			{Role: genai.RoleUser, Parts: toolResultParts},
		},
		Config: req.Config,
	}

	for resp, err := range llm.GenerateContent(ctx, followupReq, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Assessment: %s\n", part.Text)
				}
			}
		}
	}

	return nil
}

// demonstrateImageConstraints shows handling of Bedrock image constraints.
func demonstrateImageConstraints(ctx context.Context, llm model.LLM, imageURL string) error {
	fmt.Println("\n=== Understanding Image Constraints ===")

	imageData, err := downloadImage(imageURL)
	if err != nil {
		return fmt.Errorf("download image: %w", err)
	}

	fmt.Println("Bedrock Converse Image Constraints:")
	fmt.Println("- Maximum image size: Check AWS documentation")
	fmt.Println("- Supported formats: JPEG, PNG, GIF, WebP")
	fmt.Println("- Images only in user turns (not in model turns)")
	fmt.Println("- Image quality affects token usage")

	// Example: image analysis with constraints in mind
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{Text: "Analyze this image for key elements:"},
					{InlineData: &genai.Blob{MIMEType: mimeTypeFromURL(imageURL), Data: imageData}},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: 256,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if resp == nil {
			continue
		}
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Analysis: %s\n", part.Text[:min(len(part.Text), 200)])
				}
			}
		}
		if resp.UsageMetadata != nil {
			fmt.Printf("Usage - Prompt: %d, Candidates: %d, Total: %d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount,
				resp.UsageMetadata.TotalTokenCount)
		}
	}

	return nil
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
		log.Println("BEDROCK_MODEL_ID is required (e.g. eu.amazon.nova-2-lite-v1:0) using default model")
		modelID = "eu.amazon.nova-2-lite-v1:0"
	}

	// Get image URL from environment or use a default
	imageURL := os.Getenv("IMAGE_URL")
	if imageURL == "" {
		// Use a sample public image for testing
		imageURL = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"
	}

	// Create Bedrock LLM
	tp, shutdownTP, err := exampletrace.TracerProvider(ctx)
	if err != nil {
		log.Fatalf("tracer provider: %v", err)
	}
	defer func() { _ = shutdownTP(context.Background()) }()

	br := bedrockruntime.NewFromConfig(awsCfg)
	llm, err := bedrock.NewWithAPI(modelID, bedrock.NewRuntimeAPI(br, bedrock.WithTracerProvider(tp)))
	if err != nil {
		log.Panicf("create bedrock model: %v", err)
	}

	// Run all examples
	examples := []struct {
		name string
		fn   func(context.Context, model.LLM, string) error
	}{
		{"Image Analysis with Vision", analyzeImageWithVision},
		{"Image Analysis with System Instruction", analyzeImageWithSystemInstruction},
		{"Conversation History with Images", demonstrateConversationHistory},
		{"Vision with Extended Reasoning", demonstrateVisionWithReasoning},
		{"Tool Calling with Images", demonstrateToolCallingWithImages},
		{"Streaming with Images", demonstrateStreamingWithImages},
		{"Media in Tool Responses", demonstrateMediaInToolResponses},
		{"Image Constraints", demonstrateImageConstraints},
	}

	for _, example := range examples {
		if err := example.fn(ctx, llm, imageURL); err != nil {
			log.Printf("ERROR in %s: %v\n", example.name, err)
		}
	}

	// Run example that doesn't need imageURL
	if err := demonstrateMultipleImages(ctx, llm); err != nil {
		log.Printf("ERROR in Comparing Multiple Images: %v\n", err)
	}

	fmt.Println("\n=== Multimodal Examples Complete ===")
}
