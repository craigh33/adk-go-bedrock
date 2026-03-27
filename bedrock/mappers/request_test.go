package mappers

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
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
