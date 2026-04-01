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

func TestPartsToContentBlocks_multimodalInlineAndFile(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{
		{InlineData: &genai.Blob{Data: []byte{0x01, 0x02}, MIMEType: "image/png"}},
		{InlineData: &genai.Blob{Data: []byte{0x03, 0x04}, MIMEType: "audio/wav"}},
		{FileData: &genai.FileData{FileURI: "s3://bucket/video.mp4", MIMEType: "video/mp4"}},
		{
			FileData: &genai.FileData{
				FileURI:     "s3://bucket/report.pdf",
				MIMEType:    "application/pdf",
				DisplayName: "report.pdf",
			},
		},
	}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 4 {
		t.Fatalf("blocks: %+v", blocks)
	}
	if _, ok := blocks[0].(*types.ContentBlockMemberImage); !ok {
		t.Fatalf("block[0] type: %T", blocks[0])
	}
	if _, ok := blocks[1].(*types.ContentBlockMemberAudio); !ok {
		t.Fatalf("block[1] type: %T", blocks[1])
	}
	if _, ok := blocks[2].(*types.ContentBlockMemberVideo); !ok {
		t.Fatalf("block[2] type: %T", blocks[2])
	}
	doc, ok := blocks[3].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("block[3] type: %T", blocks[3])
	}
	if got := aws.ToString(doc.Value.Name); got != "report.pdf" {
		t.Fatalf("document name: %q", got)
	}
}

func TestPartsToContentBlocks_thoughtReasoning(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		Text:             "reasoning text",
		Thought:          true,
		ThoughtSignature: []byte("sig-1"),
	}}, types.ConversationRoleAssistant)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	rb, ok := blocks[0].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("block type: %T", blocks[0])
	}
	rt, ok := rb.Value.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok {
		t.Fatalf("reasoning type: %T", rb.Value)
	}
	if aws.ToString(rt.Value.Text) != "reasoning text" || aws.ToString(rt.Value.Signature) != "sig-1" {
		t.Fatalf("reasoning value: %+v", rt.Value)
	}
}

func TestPartsToContentBlocks_functionResponseMedia(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		FunctionResponse: &genai.FunctionResponse{
			ID:   "call_1",
			Name: "fn",
			Response: map[string]any{
				"ok": true,
			},
			Parts: []*genai.FunctionResponsePart{
				{InlineData: &genai.FunctionResponseBlob{Data: []byte{0x01}, MIMEType: "image/png"}},
				{FileData: &genai.FunctionResponseFileData{FileURI: "s3://bucket/demo.mp4", MIMEType: "video/mp4"}},
			},
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	tr, ok := blocks[0].(*types.ContentBlockMemberToolResult)
	if !ok {
		t.Fatalf("block type: %T", blocks[0])
	}
	if len(tr.Value.Content) != 3 {
		t.Fatalf("tool result content: %+v", tr.Value.Content)
	}
}

func TestConverseInputFromLLMRequest_safetySettingsFailFast(t *testing.T) {
	t.Parallel()
	_, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config: &genai.GenerateContentConfig{
			SafetySettings: []*genai.SafetySetting{{
				Category:  genai.HarmCategoryHarassment,
				Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
			}},
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func ptrFloat32(f float32) *float32 { return &f }
