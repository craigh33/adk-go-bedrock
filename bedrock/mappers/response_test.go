package mappers

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"
)

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
