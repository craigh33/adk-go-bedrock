package bedrock

import (
	"context"
	"errors"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type fakeAPI struct {
	converseOut *bedrockruntime.ConverseOutput
	converseErr error

	stream    StreamReader
	streamErr error
}

func (f *fakeAPI) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	_ = ctx
	_ = params
	_ = optFns
	if f.converseErr != nil {
		return nil, f.converseErr
	}
	return f.converseOut, nil
}

func (f *fakeAPI) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (StreamReader, error) {
	_ = ctx
	_ = params
	_ = optFns
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
	idx := int32(0)
	ch := make(chan types.ConverseStreamOutput, 4)
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta:             &types.ContentBlockDeltaMemberText{Value: "hel"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta:             &types.ContentBlockDeltaMemberText{Value: "lo"},
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

func TestConverse_GenerateContent_streamToolCalls_parseAndRawArgs(t *testing.T) {
	t.Parallel()

	idx0 := int32(0)
	idx1 := int32(1)
	toolAID := "tool-a"
	toolAName := "get_weather"
	toolBID := "tool-b"
	toolBName := "get_time"

	ch := make(chan types.ConverseStreamOutput, 8)
	ch <- &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: &idx0,
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
			ToolUseId: &toolAID,
			Name:      &toolAName,
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &idx0,
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
			Input: aws.String("{\"city\":\"Dub"),
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: &idx1,
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
			ToolUseId: &toolBID,
			Name:      &toolBName,
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &idx0,
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
			Input: aws.String("lin\"}"),
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &idx1,
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
			Input: aws.String("{not-json}"),
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}}
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

	var final *model.LLMResponse
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if !r.Partial {
			final = r
		}
	}
	if final == nil {
		t.Fatal("missing final response")
	}
	if final.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish reason: got %v", final.FinishReason)
	}
	if final.Content == nil || len(final.Content.Parts) != 2 {
		t.Fatalf("parts: %+v", final.Content)
	}
	callA := final.Content.Parts[0].FunctionCall
	callB := final.Content.Parts[1].FunctionCall
	if callA == nil || callB == nil {
		t.Fatalf("expected function calls, got %+v", final.Content.Parts)
	}
	if callA.Name != toolAName || callA.ID != toolAID {
		t.Fatalf("callA: %+v", callA)
	}
	if callA.Args[rawFunctionArgsJSONKey] != "{\"city\":\"Dublin\"}" || callA.Args["city"] != "Dublin" {
		t.Fatalf("callA args: %+v", callA.Args)
	}
	if callB.Name != toolBName || callB.ID != toolBID {
		t.Fatalf("callB: %+v", callB)
	}
	if callB.Args[rawFunctionArgsJSONKey] != "{not-json}" {
		t.Fatalf("callB args: %+v", callB.Args)
	}
}

func TestConverse_GenerateContent_streamError(t *testing.T) {
	t.Parallel()

	ch := make(chan types.ConverseStreamOutput)
	close(ch)
	api := &fakeAPI{stream: &fakeStream{ch: ch, err: errors.New("stream failed")}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil || gotErr.Error() != "stream failed" {
		t.Fatalf("error: %v", gotErr)
	}
}
