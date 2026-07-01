package mantle

import (
	"context"
	"errors"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/bedrock"
)

// fakeMessages is a test double for the Anthropic Messages API.
type fakeMessages struct {
	newFunc    func(anthropic.MessageNewParams) (*anthropic.Message, error)
	streamFunc func(anthropic.MessageNewParams) *ssestream.Stream[anthropic.MessageStreamEventUnion]
	gotParams  anthropic.MessageNewParams
}

func (f *fakeMessages) New(
	_ context.Context,
	params anthropic.MessageNewParams,
	_ ...option.RequestOption,
) (*anthropic.Message, error) {
	f.gotParams = params
	return f.newFunc(params)
}

func (f *fakeMessages) NewStreaming(
	_ context.Context,
	params anthropic.MessageNewParams,
	_ ...option.RequestOption,
) *ssestream.Stream[anthropic.MessageStreamEventUnion] {
	f.gotParams = params
	return f.streamFunc(params)
}

func TestNewWithMessages_NilError(t *testing.T) {
	if _, err := NewWithMessages(nil); err == nil {
		t.Error("NewWithMessages(nil): expected error")
	}
}

func TestClientConverse(t *testing.T) {
	msg := mustUnmarshalMessage(t, `{
		"id": "msg_1", "type": "message", "role": "assistant",
		"model": "anthropic.claude-3-haiku", "stop_reason": "end_turn",
		"content": [{"type": "text", "text": "4"}],
		"usage": {"input_tokens": 5, "output_tokens": 1}
	}`)
	fake := &fakeMessages{newFunc: func(anthropic.MessageNewParams) (*anthropic.Message, error) {
		return msg, nil
	}}
	client, err := NewWithMessages(fake)
	if err != nil {
		t.Fatalf("NewWithMessages: %v", err)
	}

	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("us.anthropic.claude-3-haiku"),
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "2+2?"}},
		}},
	}
	out, err := client.Converse(context.Background(), in)
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	if fake.gotParams.Model != "anthropic.claude-3-haiku" {
		t.Errorf("model passed to Messages API = %q, want normalized", fake.gotParams.Model)
	}
	msgMember, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		t.Fatalf("output type = %T", out.Output)
	}
	text, ok := msgMember.Value.Content[0].(*types.ContentBlockMemberText)
	if !ok || text.Value != "4" {
		t.Errorf("content = %+v, want text '4'", msgMember.Value.Content)
	}
}

func TestClientConverse_Errors(t *testing.T) {
	var nilClient *Client
	if _, err := nilClient.Converse(context.Background(), &bedrockruntime.ConverseInput{}); err == nil {
		t.Error("nil client: expected error")
	}

	fake := &fakeMessages{newFunc: func(anthropic.MessageNewParams) (*anthropic.Message, error) {
		return nil, errors.New("boom")
	}}
	client, err := NewWithMessages(fake)
	if err != nil {
		t.Fatalf("NewWithMessages: %v", err)
	}
	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("anthropic.claude-3-haiku"),
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}},
		}},
	}
	if _, err := client.Converse(context.Background(), in); err == nil {
		t.Error("upstream error: expected error")
	}
}

// TestClientConverse_ThroughModel drives the unary path end to end through a
// bedrock.Model: genai request -> Converse -> Anthropic -> Converse -> genai.
func TestClientConverse_ThroughModel(t *testing.T) {
	msg := mustUnmarshalMessage(t, `{
		"id": "msg_1", "type": "message", "role": "assistant",
		"model": "anthropic.claude-3-haiku", "stop_reason": "end_turn",
		"content": [{"type": "text", "text": "4"}],
		"usage": {"input_tokens": 5, "output_tokens": 1}
	}`)
	fake := &fakeMessages{newFunc: func(anthropic.MessageNewParams) (*anthropic.Message, error) {
		return msg, nil
	}}
	client, err := NewWithMessages(fake)
	if err != nil {
		t.Fatalf("NewWithMessages: %v", err)
	}
	llm, err := bedrock.NewWithAPI("anthropic.claude-3-haiku", client)
	if err != nil {
		t.Fatalf("NewWithAPI: %v", err)
	}

	req := &model.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("2+2?", genai.RoleUser)}}
	var last *model.LLMResponse
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		last = resp
	}
	if last == nil || last.Content == nil || len(last.Content.Parts) == 0 {
		t.Fatalf("no content in final response: %+v", last)
	}
	if last.Content.Parts[0].Text != "4" {
		t.Errorf("text = %q, want '4'", last.Content.Parts[0].Text)
	}
}
