package imagegenerator

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/genai"
)

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

type fakeInvokeAPI struct {
	output *bedrockruntime.InvokeModelOutput
	err    error

	capturedInput *bedrockruntime.InvokeModelInput
}

func (f *fakeInvokeAPI) InvokeModel(
	_ context.Context,
	params *bedrockruntime.InvokeModelInput,
	_ ...func(*bedrockruntime.Options),
) (*bedrockruntime.InvokeModelOutput, error) {
	f.capturedInput = params
	if f.err != nil {
		return nil, f.err
	}
	return f.output, nil
}

type fakeArtifacts struct {
	savedName string
	savedPart *genai.Part
	saveErr   error
	version   int64
}

func (f *fakeArtifacts) Save(_ context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	f.savedName = name
	f.savedPart = data
	if f.saveErr != nil {
		return nil, f.saveErr
	}
	return &artifact.SaveResponse{Version: f.version}, nil
}

func (f *fakeArtifacts) List(context.Context) (*artifact.ListResponse, error) {
	return &artifact.ListResponse{}, nil
}

func (f *fakeArtifacts) Load(context.Context, string) (*artifact.LoadResponse, error) {
	return nil, errors.New("not implemented")
}

func (f *fakeArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, errors.New("not implemented")
}

type fakeToolContext struct {
	context.Context

	artifacts *fakeArtifacts
}

func (f *fakeToolContext) FunctionCallID() string         { return "test-call-id" }
func (f *fakeToolContext) Actions() *session.EventActions { return &session.EventActions{} }
func (f *fakeToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, errors.New("not implemented")
}
func (f *fakeToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (f *fakeToolContext) RequestConfirmation(string, any) error                { return nil }
func (f *fakeToolContext) Artifacts() agent.Artifacts                           { return f.artifacts }
func (f *fakeToolContext) State() session.State                                 { return nil }
func (f *fakeToolContext) UserContent() *genai.Content                          { return nil }
func (f *fakeToolContext) InvocationID() string                                 { return "inv-1" }
func (f *fakeToolContext) AgentName() string                                    { return "test-agent" }
func (f *fakeToolContext) ReadonlyState() session.ReadonlyState                 { return nil }
func (f *fakeToolContext) UserID() string                                       { return "user-1" }
func (f *fakeToolContext) AppName() string                                      { return "test-app" }
func (f *fakeToolContext) SessionID() string                                    { return "session-1" }
func (f *fakeToolContext) Branch() string                                       { return "" }

var _ tool.Context = (*fakeToolContext)(nil)

func newFakeToolCtx(arts *fakeArtifacts) *fakeToolContext {
	return &fakeToolContext{Context: context.Background(), artifacts: arts}
}

// testCanvasProvider builds a provider with default dimensions/quality; zero modelID uses [DefaultCanvasModelID].
func testCanvasProvider(modelID string) *CanvasProvider {
	return NewCanvasProvider(modelID, 0, 0, 0, 0, "", 0)
}

// ---------------------------------------------------------------------------
// New()
// ---------------------------------------------------------------------------

func TestNew_NilAPI(t *testing.T) {
	t.Parallel()
	_, err := New(Config{Provider: testCanvasProvider("")})
	if err == nil {
		t.Fatal("expected error for nil API")
	}
}

func TestNew_NilProvider(t *testing.T) {
	t.Parallel()
	_, err := New(Config{API: &fakeInvokeAPI{}})
	if err == nil {
		t.Fatal("expected error for nil Provider")
	}
}

func TestNew_OK(t *testing.T) {
	t.Parallel()
	tl, err := New(Config{API: &fakeInvokeAPI{}, Provider: testCanvasProvider("")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tl.Name() != "generate_image" {
		t.Errorf("Name() = %q, want %q", tl.Name(), "generate_image")
	}
	if tl.IsLongRunning() {
		t.Error("IsLongRunning should be false")
	}
}

// ---------------------------------------------------------------------------
// Declaration
// ---------------------------------------------------------------------------

func TestDeclaration(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeInvokeAPI{}, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)
	decl := gt.Declaration()
	if decl == nil {
		t.Fatal("Declaration() returned nil")
	}
	if decl.Name != "generate_image" {
		t.Errorf("declaration name = %q", decl.Name)
	}
	if decl.Parameters == nil || decl.Parameters.Properties["prompt"] == nil {
		t.Error("missing prompt parameter")
	}
	if decl.Parameters.Properties["file_name"] == nil {
		t.Error("missing file_name parameter")
	}
	if len(decl.Parameters.Required) != 1 || decl.Parameters.Required[0] != "prompt" {
		t.Errorf("Required = %v, want [prompt]", decl.Parameters.Required)
	}
}

// ---------------------------------------------------------------------------
// ProcessRequest
// ---------------------------------------------------------------------------

func TestProcessRequest_PacksDeclaration(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeInvokeAPI{}, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)
	req := &model.LLMRequest{}
	if err := gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if req.Config == nil || len(req.Config.Tools) == 0 {
		t.Fatal("expected at least one tool in Config")
	}
	if len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatal("expected one function declaration")
	}
	if req.Config.Tools[0].FunctionDeclarations[0].Name != "generate_image" {
		t.Error("wrong declaration name")
	}
}

func TestProcessRequest_DuplicateError(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeInvokeAPI{}, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)
	req := &model.LLMRequest{}
	_ = gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req)
	err := gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req)
	if err == nil {
		t.Fatal("expected duplicate tool error")
	}
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

func canvasSuccessBody() []byte {
	imgData := base64.StdEncoding.EncodeToString([]byte("fake-png-bytes"))
	resp := canvasResponse{Images: []string{imgData}}
	b, _ := json.Marshal(resp)
	return b
}

func TestRun_Success(t *testing.T) {
	t.Parallel()
	api := &fakeInvokeAPI{output: &bedrockruntime.InvokeModelOutput{Body: canvasSuccessBody()}}
	arts := &fakeArtifacts{version: 1}
	tl, _ := New(Config{API: api, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)

	result, err := gt.Run(newFakeToolCtx(arts), map[string]any{
		"prompt":    "a sunset over mountains",
		"file_name": "sunset.png",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result["status"] != "success" {
		t.Errorf("status = %v", result["status"])
	}
	if result["file_name"] != "sunset.png" {
		t.Errorf("file_name = %v", result["file_name"])
	}
	if arts.savedName != "sunset.png" {
		t.Errorf("artifact saved as %q", arts.savedName)
	}
	if arts.savedPart == nil || arts.savedPart.InlineData == nil {
		t.Fatal("artifact part missing inline data")
	}
	if arts.savedPart.InlineData.MIMEType != "image/png" {
		t.Errorf("MIME = %q", arts.savedPart.InlineData.MIMEType)
	}
}

func TestRun_EmptyPrompt(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeInvokeAPI{}, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)

	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": ""})
	if err == nil {
		t.Fatal("expected error for empty prompt")
	}
}

func TestRun_DefaultFileName(t *testing.T) {
	t.Parallel()
	api := &fakeInvokeAPI{output: &bedrockruntime.InvokeModelOutput{Body: canvasSuccessBody()}}
	arts := &fakeArtifacts{version: 1}
	tl, _ := New(Config{API: api, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)

	result, err := gt.Run(newFakeToolCtx(arts), map[string]any{"prompt": "hello"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result["file_name"] != "generated_image.png" {
		t.Errorf("expected default file name, got %v", result["file_name"])
	}
}

func TestRun_InvokeModelError(t *testing.T) {
	t.Parallel()
	api := &fakeInvokeAPI{err: errors.New("throttled")}
	tl, _ := New(Config{API: api, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)

	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": "hi"})
	if err == nil {
		t.Fatal("expected invoke model error")
	}
}

func TestRun_ArtifactSaveError(t *testing.T) {
	t.Parallel()
	api := &fakeInvokeAPI{output: &bedrockruntime.InvokeModelOutput{Body: canvasSuccessBody()}}
	arts := &fakeArtifacts{saveErr: errors.New("storage full")}
	tl, _ := New(Config{API: api, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)

	_, err := gt.Run(newFakeToolCtx(arts), map[string]any{"prompt": "hi"})
	if err == nil {
		t.Fatal("expected artifact save error")
	}
}

func TestRun_BadArgsType(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeInvokeAPI{}, Provider: testCanvasProvider("")})
	gt := tl.(*imageGenTool)

	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), "not-a-map")
	if err == nil {
		t.Fatal("expected error for bad args type")
	}
}

// ---------------------------------------------------------------------------
// CanvasProvider unit tests
// ---------------------------------------------------------------------------

func TestCanvasProvider_MarshalRequest(t *testing.T) {
	t.Parallel()
	p := testCanvasProvider("")
	body, err := p.MarshalRequest("a dog")
	if err != nil {
		t.Fatalf("MarshalRequest: %v", err)
	}
	var req canvasRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.TaskType != "TEXT_IMAGE" {
		t.Errorf("TaskType = %q", req.TaskType)
	}
	if req.TextToImageParams.Text != "a dog" {
		t.Errorf("Text = %q", req.TextToImageParams.Text)
	}
}

func TestCanvasProvider_UnmarshalResponse_NoImages(t *testing.T) {
	t.Parallel()
	p := testCanvasProvider("")
	body, _ := json.Marshal(canvasResponse{Images: []string{}})
	_, _, err := p.UnmarshalResponse(body)
	if err == nil {
		t.Fatal("expected error for empty images")
	}
}

func TestCanvasProvider_ModelID_Default(t *testing.T) {
	t.Parallel()
	p := testCanvasProvider("")
	if p.ModelID() != DefaultCanvasModelID {
		t.Errorf("ModelID = %q, want %q", p.ModelID(), DefaultCanvasModelID)
	}
}

func TestCanvasProvider_ModelID_Custom(t *testing.T) {
	t.Parallel()
	custom := "eu.amazon.nova-canvas-v1:0"
	p := testCanvasProvider(custom)
	if p.ModelID() != custom {
		t.Errorf("ModelID = %q", p.ModelID())
	}
}

func TestClampNovaCanvasSeed(t *testing.T) {
	t.Parallel()
	if clampNovaCanvasSeed(100) != 100 {
		t.Errorf("expected 100, got %d", clampNovaCanvasSeed(100))
	}
	if clampNovaCanvasSeed(maxNovaCanvasSeed) != maxNovaCanvasSeed {
		t.Errorf("expected max, got %d", clampNovaCanvasSeed(maxNovaCanvasSeed))
	}
	if clampNovaCanvasSeed(maxNovaCanvasSeed+1) != 0 {
		t.Errorf("expected 0 for overflow mod, got %d", clampNovaCanvasSeed(maxNovaCanvasSeed+1))
	}
	if clampNovaCanvasSeed(-1) != 0 {
		t.Errorf("expected 0 for negative, got %d", clampNovaCanvasSeed(-1))
	}
}

func TestCanvasProvider_MarshalRequest_SeedWithinAPIRange(t *testing.T) {
	t.Parallel()
	p := NewCanvasProvider(DefaultCanvasModelID, 0, 0, 0, 0, "", 0)
	body, err := p.MarshalRequest("x")
	if err != nil {
		t.Fatal(err)
	}
	var req canvasRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatal(err)
	}
	if req.ImageGenerationConfig.Seed < 0 || req.ImageGenerationConfig.Seed > maxNovaCanvasSeed {
		t.Errorf("seed %d out of API range [0,%d]", req.ImageGenerationConfig.Seed, maxNovaCanvasSeed)
	}
}
