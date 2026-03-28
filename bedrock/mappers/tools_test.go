package mappers

import (
	"testing"

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
}

func TestToolConfigurationFromGenai_GoogleSearchMapsToSystemTool(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
	}
	in, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) < 1 {
		t.Fatalf("expected at least one system tool for GoogleSearch, got: %+v", in.ToolConfig)
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

func TestToolConfigurationFromGenai_MCPServersMapsToSystemTools(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			MCPServers: []*genai.MCPServer{{
				Name: "docs",
			}},
		}},
	}
	in, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) < 1 {
		t.Fatalf("expected at least one system tool for MCP server, got: %+v", in.ToolConfig)
	}
}

func TestToolConfigurationFromGenai_MultipleToolVariantsMapped(t *testing.T) {
	t.Parallel()
	cfg := &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{
			GoogleSearch:  &genai.GoogleSearch{},
			CodeExecution: &genai.ToolCodeExecution{},
			Retrieval:     &genai.Retrieval{},
		}},
	}
	in, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("x", "user")},
		Config:   cfg,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) < 3 {
		t.Fatalf("expected at least three system tools, got: %+v", in.ToolConfig)
	}
}
