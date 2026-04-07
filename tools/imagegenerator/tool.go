package imagegenerator

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// DefaultCanvasModelID is the default Bedrock model for [CanvasProvider] when modelID is empty.
const DefaultCanvasModelID = "amazon.nova-canvas-v1:0"

const (
	defaultCanvasWidth    = 512
	defaultCanvasHeight   = 512
	defaultCanvasCfgScale = 8.0
)

// InvokeModelAPI is the subset of the Bedrock Runtime API needed for image generation.
type InvokeModelAPI interface {
	InvokeModel(
		ctx context.Context,
		params *bedrockruntime.InvokeModelInput,
		optFns ...func(*bedrockruntime.Options),
	) (*bedrockruntime.InvokeModelOutput, error)
}

// ModelProvider abstracts model-specific request/response formats for image generation.
type ModelProvider interface {
	ModelID() string
	MarshalRequest(prompt string) ([]byte, error)
	UnmarshalResponse(body []byte) (imageData []byte, mimeType string, err error)
}

// Config configures the image generator tool.
type Config struct {
	API      InvokeModelAPI
	Provider ModelProvider
}

// imageGenTool implements the ADK tool interfaces for image generation.
type imageGenTool struct {
	api      InvokeModelAPI
	provider ModelProvider
}

// New creates a new ADK-compatible image generator tool.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("imagegenerator: API is required")
	}
	if cfg.Provider == nil {
		return nil, errors.New("imagegenerator: Provider is required")
	}
	return &imageGenTool{api: cfg.API, provider: cfg.Provider}, nil
}

func (t *imageGenTool) Name() string {
	return "generate_image"
}

func (t *imageGenTool) Description() string {
	return "Generates an image from a text prompt using Amazon Bedrock and saves it as an artifact."
}

func (t *imageGenTool) IsLongRunning() bool {
	return false
}

func (t *imageGenTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"prompt": {
					Type:        "STRING",
					Description: "The text prompt describing the image to generate",
				},
				"file_name": {
					Type:        "STRING",
					Description: "The filename to save the generated image as (e.g. 'landscape.png'). If not provided, the image will be saved as 'generated_image.png'.",
				},
			},
			Required: []string{"prompt"},
		},
	}
}

// ProcessRequest packs the tool declaration into the LLM request so the model
// can discover and invoke it. This replicates the behaviour of the internal
// toolutils.PackTool helper which external modules cannot import.
func (t *imageGenTool) ProcessRequest(_ tool.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	name := t.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	decl := t.Declaration()
	if decl == nil {
		return nil
	}

	var funcTool *genai.Tool
	for _, gt := range req.Config.Tools {
		if gt != nil && gt.FunctionDeclarations != nil {
			funcTool = gt
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, decl)
	}
	return nil
}

// Run executes the image generation tool: calls Bedrock InvokeModel, then
// persists the resulting image as an artifact via Artifacts().Save.
func (t *imageGenTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}

	prompt, _ := m["prompt"].(string)
	if prompt == "" {
		return nil, errors.New("prompt is required")
	}
	fileName, _ := m["file_name"].(string)
	if fileName == "" {
		fileName = "generated_image.png"
	}

	body, err := t.provider.MarshalRequest(prompt)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	modelID := t.provider.ModelID()
	output, err := t.api.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
		Body:        body,
	})
	if err != nil {
		return nil, fmt.Errorf("invoke model %s: %w", modelID, err)
	}

	imageData, mimeType, err := t.provider.UnmarshalResponse(output.Body)
	if err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	part := &genai.Part{
		InlineData: &genai.Blob{
			Data:     imageData,
			MIMEType: mimeType,
		},
	}
	saveResp, err := ctx.Artifacts().Save(ctx, fileName, part)
	if err != nil {
		return nil, fmt.Errorf("save artifact %q: %w", fileName, err)
	}

	return map[string]any{
		"file_name": fileName,
		"version":   saveResp.Version,
		"status":    "success",
	}, nil
}

// ---------------------------------------------------------------------------
// Amazon Nova Canvas provider
// ---------------------------------------------------------------------------

// CanvasProvider generates images using Amazon Nova Canvas via InvokeModel (TEXT_IMAGE).
// See https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
type CanvasProvider struct {
	modelID        string
	Width          int
	Height         int
	CfgScale       float64
	NumberOfImages int
	Quality        string
	Seed           int64
}

// NewCanvasProvider returns a provider configured for Nova Canvas text-to-image.
// Pass an empty modelID to use [DefaultCanvasModelID].
func NewCanvasProvider(
	modelID string,
	width int,
	height int,
	cfgScale float64,
	numberOfImages int,
	quality string,
	seed int64,
) *CanvasProvider {
	if modelID == "" {
		modelID = DefaultCanvasModelID
	}
	if width == 0 {
		width = defaultCanvasWidth
	}
	if height == 0 {
		height = defaultCanvasHeight
	}
	if cfgScale == 0 {
		cfgScale = defaultCanvasCfgScale
	}
	if numberOfImages == 0 {
		numberOfImages = 1
	}
	if quality == "" {
		quality = "standard"
	}
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	return &CanvasProvider{
		modelID:        modelID,
		Width:          width,
		Height:         height,
		CfgScale:       cfgScale,
		NumberOfImages: numberOfImages,
		Quality:        quality,
		Seed:           seed,
	}
}

func (p *CanvasProvider) ModelID() string { return p.modelID }

func (p *CanvasProvider) MarshalRequest(prompt string) ([]byte, error) {
	req := canvasRequest{
		TaskType: "TEXT_IMAGE",
		TextToImageParams: canvasTextToImageParams{
			Text: prompt,
		},
		ImageGenerationConfig: canvasImageGenerationConfig{
			NumberOfImages: p.NumberOfImages,
			Quality:        p.Quality,
			CfgScale:       p.CfgScale,
			Height:         p.Height,
			Width:          p.Width,
			Seed:           p.Seed,
		},
	}
	return json.Marshal(req)
}

func (p *CanvasProvider) UnmarshalResponse(body []byte) ([]byte, string, error) {
	var resp canvasResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, "", fmt.Errorf("decode image generation response: %w", err)
	}
	if len(resp.Images) == 0 {
		return nil, "", errors.New("model returned no images")
	}
	data, err := base64.StdEncoding.DecodeString(resp.Images[0])
	if err != nil {
		return nil, "", fmt.Errorf("decode base64 image: %w", err)
	}
	return data, "image/png", nil
}

type canvasRequest struct {
	TaskType              string                      `json:"taskType"`
	TextToImageParams     canvasTextToImageParams     `json:"textToImageParams"`
	ImageGenerationConfig canvasImageGenerationConfig `json:"imageGenerationConfig"`
}

type canvasTextToImageParams struct {
	Text string `json:"text"`
}

type canvasImageGenerationConfig struct {
	NumberOfImages int     `json:"numberOfImages"`
	Quality        string  `json:"quality"`
	CfgScale       float64 `json:"cfgScale"`
	Height         int     `json:"height"`
	Width          int     `json:"width"`
	Seed           int64   `json:"seed"`
}

type canvasResponse struct {
	Images []string `json:"images"`
}
