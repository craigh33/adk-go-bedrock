package mappers

import (
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
	})
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
	})
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
	})
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
	})
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
	})
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
	})
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
	})
	if err == nil {
		t.Fatal("expected unsupported tool variants error")
	}
	for _, name := range []string{"GoogleSearch", "CodeExecution", "Retrieval"} {
		if !strings.Contains(err.Error(), name) {
			t.Fatalf("expected error to mention %s, got: %v", name, err)
		}
	}
}
