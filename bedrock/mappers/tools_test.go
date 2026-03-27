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
	if in.ToolConfig == nil || len(in.ToolConfig.Tools) != 1 {
		t.Fatalf("tools: %+v", in.ToolConfig)
	}
}
