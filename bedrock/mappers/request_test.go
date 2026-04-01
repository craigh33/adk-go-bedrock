package mappers

import (
	"errors"
	"strings"
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
	if got := aws.ToString(doc.Value.Name); got != "report-pdf" {
		t.Fatalf("document name: %q (Bedrock rejects dots in names; expect sanitized form)", got)
	}
}

func TestPartsToContentBlocks_pdfMIMEWithParams(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte("%PDF-1.4 fake"),
			MIMEType:    "application/pdf; charset=binary",
			DisplayName: "memo.pdf",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatPdf {
		t.Fatalf("format: %v", doc.Value.Format)
	}
}

func TestPartsToContentBlocks_pdfOctetStreamWithFilename(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte("%PDF-1.4 fake"),
			MIMEType:    "application/octet-stream",
			DisplayName: "report.pdf",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatPdf {
		t.Fatalf("format: %v", doc.Value.Format)
	}
}

func TestPartsToContentBlocks_docxMIMEWithParams(t *testing.T) {
	t.Parallel()
	const docxMIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte("PK\x03\x04"),
			MIMEType:    docxMIME + "; charset=binary",
			DisplayName: "memo.docx",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatDocx {
		t.Fatalf("format: %v", doc.Value.Format)
	}
}

func TestPartsToContentBlocks_docxAsApplicationZip(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte("PK\x03\x04 fake docx zip"),
			MIMEType:    "application/zip",
			DisplayName: "memo.docx",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks: %+v", blocks)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatDocx {
		t.Fatalf("format: %v", doc.Value.Format)
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

func TestPartsToContentBlocks_textAndInlineDataSamePart(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		Text: "Summarize this for me",
		InlineData: &genai.Blob{
			Data:        []byte("PK\x03\x04"),
			MIMEType:    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
			DisplayName: "N-able Internal Document Template.docx",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 2 {
		t.Fatalf("want document then text (2 blocks), got %d: %#v", len(blocks), blocks)
	}
	doc0, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("block[0] want document, got %T", blocks[0])
	}
	if got := aws.ToString(doc0.Value.Name); got != "N-able Internal Document Template-docx" {
		t.Fatalf("document name: %q", got)
	}
	if txt, ok := blocks[1].(*types.ContentBlockMemberText); !ok || txt.Value != "Summarize this for me" {
		t.Fatalf("block[1] want user text, got %#v", blocks[1])
	}
}

func TestConverseInputFromLLMRequest_emptyUserParts(t *testing.T) {
	t.Parallel()
	_, err := ConverseInputFromLLMRequest("mid", &model.LLMRequest{
		Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{}}},
		Config:   &genai.GenerateContentConfig{},
	})
	if err == nil {
		t.Fatal("expected error for empty user message with no mappable parts")
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

func TestPartsToContentBlocks_powerPointRejected(t *testing.T) {
	t.Parallel()
	_, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte("PK\x03\x04"),
			MIMEType:    "application/octet-stream",
			DisplayName: "slides.pptx",
		},
	}}, types.ConversationRoleUser)
	if err == nil {
		t.Fatal("expected error for PowerPoint")
	}
	if !errors.Is(err, ErrBedrockPowerPointNotSupported) {
		t.Fatalf("want ErrBedrockPowerPointNotSupported, got %v", err)
	}
}

func TestPartsToContentBlocks_toolCallUnsupported(t *testing.T) {
	t.Parallel()
	_, err := PartsToContentBlocks([]*genai.Part{{
		ToolCall: &genai.ToolCall{ID: "server-tool-1"},
	}}, types.ConversationRoleUser)
	if err == nil {
		t.Fatal("expected error for ToolCall")
	}
	if !strings.Contains(err.Error(), "toolCall") {
		t.Fatalf("expected error mentioning toolCall, got %v", err)
	}
}

func TestPartsToContentBlocks_toolResponseUnsupported(t *testing.T) {
	t.Parallel()
	_, err := PartsToContentBlocks([]*genai.Part{{
		ToolResponse: &genai.ToolResponse{ID: "server-tool-1"},
	}}, types.ConversationRoleUser)
	if err == nil {
		t.Fatal("expected error for ToolResponse")
	}
	if !strings.Contains(err.Error(), "toolResponse") {
		t.Fatalf("expected error mentioning toolResponse, got %v", err)
	}
}

func TestSanitizeDocumentNameForBedrock(t *testing.T) {
	t.Parallel()
	tests := []struct {
		in, want string
	}{
		{"test.pdf", "test-pdf"},
		{"report.pdf", "report-pdf"},
		{"N-able Internal Document Template.docx", "N-able Internal Document Template-docx"},
		{"a  b  c", "a b c"},
		{"file..name", "file-name"},
		{"a--b", "a-b"},
		{"--leading", "leading"},
		{"", "document"},
		{"...", "document"},
	}
	for _, tt := range tests {
		if got := sanitizeDocumentNameForBedrock(tt.in); got != tt.want {
			t.Errorf("sanitizeDocumentNameForBedrock(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func ptrFloat32(f float32) *float32 { return &f }
