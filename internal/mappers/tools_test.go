package mappers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestToolConfigurationFromGenai(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "get_weather",
				Description: "weather",
				Parameters: &genai.Schema{
					Type: "object",
					Properties: map[string]*genai.Schema{
						"city": {Type: "string"},
					},
				},
			}},
		}},
	}
	in, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) < 1 {
		t.Fatalf("expected at least one tool (function declaration), got: %+v", in.ToolConfig)
	}
	if got := len(in.ToolConfig.Tools); got != 1 {
		t.Fatalf("expected exactly one function declaration tool, got %d: %+v", got, in.ToolConfig.Tools)
	}
	if _, ok := in.ToolConfig.Tools[0].(*types.ToolMemberToolSpec); !ok {
		t.Fatalf("expected function declaration to map to ToolSpec only, got %T", in.ToolConfig.Tools[0])
	}
}

func TestToolConfigurationFromGenai_FunctionDeclarationDoesNotMapTypedNilVariants(t *testing.T) {
	t.Parallel()
	tool := &genai.Tool{}
	tool.FunctionDeclarations = []*genai.FunctionDeclaration{{
		Name:        "tag_image",
		Description: "Tag an image",
		Parameters: &genai.Schema{
			Type: "object",
			Properties: map[string]*genai.Schema{
				"tags_csv": {Type: "string"},
			},
		},
	}}
	in, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   &genai.GenerateContentConfig{Tools: []*genai.Tool{tool}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	if in.ToolConfig == nil {
		t.Fatal("expected tool config")
	}
	if got := len(in.ToolConfig.Tools); got != 1 {
		t.Fatalf("expected exactly one mapped tool, got %d: %+v", got, in.ToolConfig.Tools)
	}
	if _, ok := in.ToolConfig.Tools[0].(*types.ToolMemberSystemTool); ok {
		t.Fatalf("unexpected system tool mapping for function-only tool: %+v", in.ToolConfig.Tools[0])
	}
}

func TestToolConfigurationFromGenai_GoogleSearchReturnsUnsupportedError(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
	}
	_, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	}, false)
	if err == nil || !strings.Contains(err.Error(), "GoogleSearch") {
		t.Fatalf("expected unsupported GoogleSearch error, got: %v", err)
	}
}

func TestToolConfigurationFromGenai_MixedToolList_MapsOnlyFunctionDeclarations(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{
			{}, // Simulates a non-function tool variant at mapping time.
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "search_docs",
					Description: "Search docs",
					Parameters: &genai.Schema{
						Type: "object",
						Properties: map[string]*genai.Schema{
							"query": {Type: "string"},
						},
					},
				}},
			},
		},
	}
	in, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) < 1 {
		t.Fatalf("expected at least one mapped function declaration tool, got: %+v", in.ToolConfig)
	}
}

func TestToolConfigurationFromGenai_MixedFunctionAndUnsupportedVariantReturnsError(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        "save_results",
				Description: "Save results",
				Parameters: &genai.Schema{
					Type: "object",
					Properties: map[string]*genai.Schema{
						"results": {Type: "string"},
					},
				},
			}},
			GoogleSearch: &genai.GoogleSearch{},
		}},
	}
	_, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	}, false)
	if err == nil || !strings.Contains(err.Error(), "GoogleSearch") {
		t.Fatalf("expected mixed unsupported variant error mentioning GoogleSearch, got: %v", err)
	}
}

func TestToolConfigurationFromGenai_MCPServersReturnUnsupportedError(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			MCPServers: []*genai.MCPServer{{
				Name: "docs",
			}},
		}},
	}
	_, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	}, false)
	if err == nil || !strings.Contains(err.Error(), "MCPServers") {
		t.Fatalf("expected unsupported MCPServers error, got: %v", err)
	}
}

func TestToolConfigurationFromGenai_MultipleToolVariantsReturnUnsupportedError(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			GoogleSearch:  &genai.GoogleSearch{},
			CodeExecution: &genai.ToolCodeExecution{},
			Retrieval:     &genai.Retrieval{},
		}},
	}
	_, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	}, false)
	if err == nil {
		t.Fatal("expected unsupported tool variants error")
	}
	for _, name := range []string{"GoogleSearch", "CodeExecution", "Retrieval"} {
		if !strings.Contains(err.Error(), name) {
			t.Fatalf("expected error to mention %s, got: %v", name, err)
		}
	}
}

func TestNormalizeSchema_MapPassthrough(t *testing.T) {
	t.Parallel()
	in := map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}}
	got, err := normalizeSchema(in)
	if err != nil {
		t.Fatal(err)
	}
	got["_probe"] = 1
	if in["_probe"] != 1 {
		t.Fatal("expected same map backing store for map[string]any input")
	}
	delete(in, "_probe")
}

func TestNormalizeSchema_StructRoundTripsToMap(t *testing.T) {
	t.Parallel()
	// Simulates ADK FunctionTool / genai where parametersJsonSchema is not map[string]any.
	type nested struct {
		Min float64 `json:"minimum"`
	}
	type propSchema struct {
		Type    string `json:"type"`
		Nested  nested `json:"meta"`
		Ignored string `json:"-"`
	}
	schema := struct {
		Type       string                `json:"type"`
		Properties map[string]propSchema `json:"properties"`
	}{
		Type: "object",
		Properties: map[string]propSchema{
			"n": {Type: "NUMBER", Nested: nested{Min: 1}, Ignored: "skip"},
		},
	}
	got, err := normalizeSchema(schema)
	if err != nil {
		t.Fatal(err)
	}
	if got["type"] != "object" {
		t.Fatalf("type: got %v", got["type"])
	}
	props, ok := got["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties: got %T", got["properties"])
	}
	n, ok := props["n"].(map[string]any)
	if !ok {
		t.Fatalf("properties.n: got %T", props["n"])
	}
	if n["type"] != "NUMBER" {
		t.Fatalf("properties.n.type: got %v", n["type"])
	}
	meta, ok := n["meta"].(map[string]any)
	if !ok {
		t.Fatalf("meta: got %T", n["meta"])
	}
	if meta["minimum"] != float64(1) {
		t.Fatalf("meta.minimum: got %v", meta["minimum"])
	}
}

func TestNormalizeSchema_UnmarshalableJSONReturnsError(t *testing.T) {
	t.Parallel()
	ch := make(chan int)
	_, err := normalizeSchema(ch)
	if err == nil || !strings.Contains(err.Error(), "marshal") {
		t.Fatalf("expected marshal error, got: %v", err)
	}
}

func TestFunctionParametersToToolInputSchema_ParametersJsonSchemaStruct(t *testing.T) {
	t.Parallel()
	type item struct {
		Type string `json:"type"`
	}
	fd := &genai.FunctionDeclaration{
		Name:        "fn",
		Description: "d",
		ParametersJsonSchema: struct {
			Items item `json:"items"`
		}{Items: item{Type: "STRING"}},
	}
	schema, err := functionParametersToToolInputSchema(fd)
	if err != nil {
		t.Fatal(err)
	}
	jsonMember, ok := schema.(*types.ToolInputSchemaMemberJson)
	if !ok {
		t.Fatalf("got %T", schema)
	}
	raw, err := jsonMember.Value.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	items, ok := decoded["items"].(map[string]any)
	if !ok {
		t.Fatalf("items: got %T", decoded["items"])
	}
	if items["type"] != "STRING" {
		t.Fatalf("items.type: got %v", items["type"])
	}
}
