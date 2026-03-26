package mappers

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestConverseInputFromLLMRequest_basicUserMessage(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Model: "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		Contents: []*genai.Content{
			genai.NewContentFromText("Hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are concise.", "system"),
			Temperature:       ptrFloat32(0.2),
			MaxOutputTokens:   100,
		},
	}
	in, err := ConverseInputFromLLMRequest("model-id", req)
	if err != nil {
		t.Fatal(err)
	}
	if aws.ToString(in.ModelId) != "model-id" {
		t.Fatalf("ModelId: got %q", aws.ToString(in.ModelId))
	}
	if len(in.Messages) < 1 {
		t.Fatalf("expected messages")
	}
	if in.InferenceConfig == nil || in.InferenceConfig.Temperature == nil || *in.InferenceConfig.Temperature != 0.2 {
		t.Fatalf("inference config temperature: %+v", in.InferenceConfig)
	}
}

func TestMaybeAppendUserContent_empty(t *testing.T) {
	t.Parallel()
	out := MaybeAppendUserContent(nil)
	if len(out) != 1 || out[0].Role != "user" {
		t.Fatalf("got %+v", out)
	}
}

func TestLLMResponseFromConverseOutput_text(t *testing.T) {
	t.Parallel()
	out := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{
					&types.ContentBlockMemberText{Value: "hi"},
				},
			},
		},
		StopReason: types.StopReasonEndTurn,
		Usage: &types.TokenUsage{
			InputTokens:  aws.Int32(3),
			OutputTokens: aws.Int32(1),
			TotalTokens:  aws.Int32(4),
		},
	}
	resp, err := LLMResponseFromConverseOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content == nil || len(resp.Content.Parts) != 1 || resp.Content.Parts[0].Text != "hi" {
		t.Fatalf("content: %+v", resp.Content)
	}
	if resp.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish: %v", resp.FinishReason)
	}
	if resp.UsageMetadata == nil || resp.UsageMetadata.TotalTokenCount != 4 {
		t.Fatalf("usage: %+v", resp.UsageMetadata)
	}
}

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

func TestStopReasonMapping(t *testing.T) {
	t.Parallel()
	cases := []struct {
		in   types.StopReason
		want genai.FinishReason
	}{
		{types.StopReasonMaxTokens, genai.FinishReasonMaxTokens},
		{types.StopReasonEndTurn, genai.FinishReasonStop},
	}
	for _, c := range cases {
		if got := StopReasonToFinishReason(c.in); got != c.want {
			t.Errorf("%v: got %v want %v", c.in, got, c.want)
		}
	}
}

func TestMessageToGenaiContent_roundTrip(t *testing.T) {
	t.Parallel()
	msg := &types.Message{
		Role: types.ConversationRoleAssistant,
		Content: []types.ContentBlock{
			&types.ContentBlockMemberText{Value: "answer"},
		},
	}
	c, err := MessageToGenaiContent(msg)
	if err != nil {
		t.Fatal(err)
	}
	if c.Role != "model" || c.Parts[0].Text != "answer" {
		t.Fatalf("got %+v", c)
	}
}

func TestPartsToContentBlocks_functionCall(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		FunctionCall: &genai.FunctionCall{
			ID:   "toolu_1",
			Name: "fn",
			Args: map[string]any{"x": 1},
		},
	}}, types.ConversationRoleAssistant)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	tu := blocks[0].(*types.ContentBlockMemberToolUse)
	if aws.ToString(tu.Value.Name) != "fn" {
		t.Fatalf("name: %+v", tu.Value)
	}
}

func ptrFloat32(f float32) *float32 { return &f }
