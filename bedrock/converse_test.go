package bedrock

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craig-hutcheon/adk-go-bedrock/bedrock/client"
)

type fakeAPI struct {
	converseOut *bedrockruntime.ConverseOutput
	converseErr error

	stream    client.StreamReader
	streamErr error
}

func (f *fakeAPI) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	if f.converseErr != nil {
		return nil, f.converseErr
	}
	return f.converseOut, nil
}

func (f *fakeAPI) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (client.StreamReader, error) {
	if f.streamErr != nil {
		return nil, f.streamErr
	}
	return f.stream, nil
}

type fakeStream struct {
	ch  chan types.ConverseStreamOutput
	err error
}

func (f *fakeStream) Events() <-chan types.ConverseStreamOutput { return f.ch }
func (f *fakeStream) Close() error                              { return nil }
func (f *fakeStream) Err() error                                { return f.err }

func TestConverse_GenerateContent_unary(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{
		converseOut: &bedrockruntime.ConverseOutput{
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Role: types.ConversationRoleAssistant,
					Content: []types.ContentBlock{
						&types.ContentBlockMemberText{Value: "ok"},
					},
				},
			},
			StopReason: types.StopReasonEndTurn,
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(1),
				OutputTokens: aws.Int32(1),
				TotalTokens:  aws.Int32(2),
			},
		},
	}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var got int
	for r, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatal(err)
		}
		got++
		if r.Content.Parts[0].Text != "ok" {
			t.Fatalf("text %q", r.Content.Parts[0].Text)
		}
	}
	if got != 1 {
		t.Fatalf("responses: %d", got)
	}
}

func TestConverse_GenerateContent_stream(t *testing.T) {
	t.Parallel()
	ch := make(chan types.ConverseStreamOutput, 4)
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			Delta: &types.ContentBlockDeltaMemberText{Value: "hel"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			Delta: &types.ContentBlockDeltaMemberText{Value: "lo"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberMetadata{
		Value: types.ConverseStreamMetadataEvent{
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(2),
				OutputTokens: aws.Int32(2),
				TotalTokens:  aws.Int32(4),
			},
		},
	}
	close(ch)

	api := &fakeAPI{stream: &fakeStream{ch: ch}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var partial, final int
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if r.Partial {
			partial++
		} else {
			final++
		}
	}
	if partial < 1 || final != 1 {
		t.Fatalf("partial=%d final=%d", partial, final)
	}
}
