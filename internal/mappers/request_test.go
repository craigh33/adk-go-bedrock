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
	in, err := ConverseInputFromLLMRequest("model-id", req, false)
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

func TestPartsToContentBlocks_applicationJSONInline(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte(`{"ok":true}`),
			MIMEType:    "application/json",
			DisplayName: "payload.json",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatTxt {
		t.Fatalf("format: %v", doc.Value.Format)
	}
}

func TestPartsToContentBlocks_octetStreamJSONHARLogByFilename(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name string
		want types.DocumentFormat
	}{
		{"report.json", types.DocumentFormatTxt},
		{"trace.har", types.DocumentFormatTxt},
		{"app.log", types.DocumentFormatTxt},
	} {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			blocks, err := PartsToContentBlocks([]*genai.Part{{
				InlineData: &genai.Blob{
					Data:        []byte("content"),
					MIMEType:    "application/octet-stream",
					DisplayName: tt.name,
				},
			}}, types.ConversationRoleUser)
			if err != nil {
				t.Fatal(err)
			}
			doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
			if !ok {
				t.Fatalf("want document block, got %T", blocks[0])
			}
			if doc.Value.Format != tt.want {
				t.Fatalf("format: %v", doc.Value.Format)
			}
		})
	}
}

func TestPartsToContentBlocks_textJavaScriptToTxt(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		InlineData: &genai.Blob{
			Data:        []byte("console.log(1)"),
			MIMEType:    "text/javascript",
			DisplayName: "main.js",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatTxt {
		t.Fatalf("format: %v", doc.Value.Format)
	}
}

func TestPartsToContentBlocks_fileDataOctetStreamInferredFromS3Key(t *testing.T) {
	t.Parallel()
	blocks, err := PartsToContentBlocks([]*genai.Part{{
		FileData: &genai.FileData{
			FileURI:  "s3://bucket/prefix/payload.json",
			MIMEType: "application/octet-stream",
		},
	}}, types.ConversationRoleUser)
	if err != nil {
		t.Fatal(err)
	}
	doc, ok := blocks[0].(*types.ContentBlockMemberDocument)
	if !ok {
		t.Fatalf("want document block, got %T", blocks[0])
	}
	if doc.Value.Format != types.DocumentFormatTxt {
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
	}, false)
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
	}, false)
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

func TestConverseInputFromLLMRequest_cachePointAfterAllSystemBlocks(t *testing.T) {
	t.Parallel()
	// System prompt from SystemInstruction + system-role content in Contents.
	// CachePoint must appear as the very last system block, after both sources.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "system", Parts: []*genai.Part{{Text: "extra system"}}},
			genai.NewContentFromText("hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("base instruction", "system"),
		},
	}
	in, err := ConverseInputFromLLMRequest("mid", req, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(in.System) != 3 {
		t.Fatalf("want 3 system blocks (instruction + extra + cachepoint), got %d: %#v", len(in.System), in.System)
	}
	if _, ok := in.System[0].(*types.SystemContentBlockMemberText); !ok {
		t.Fatalf("system[0] want text, got %T", in.System[0])
	}
	if _, ok := in.System[1].(*types.SystemContentBlockMemberText); !ok {
		t.Fatalf("system[1] want text, got %T", in.System[1])
	}
	if _, ok := in.System[2].(*types.SystemContentBlockMemberCachePoint); !ok {
		t.Fatalf("system[2] want CachePoint, got %T", in.System[2])
	}
}

func TestConverseInputFromLLMRequest_cachePointNotAddedWithoutSystemBlocks(t *testing.T) {
	t.Parallel()
	// No system prompt at all — CachePoint must not be added even when cacheSystemPrompt=true.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", "user"),
		},
		Config: &genai.GenerateContentConfig{},
	}
	in, err := ConverseInputFromLLMRequest("mid", req, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(in.System) != 0 {
		t.Fatalf("want no system blocks, got %d: %#v", len(in.System), in.System)
	}
}

func TestConverseInputFromLLMRequest_cachePointNotAddedWhenDisabled(t *testing.T) {
	t.Parallel()
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("hello", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("be helpful", "system"),
		},
	}
	in, err := ConverseInputFromLLMRequest("mid", req, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(in.System) != 1 {
		t.Fatalf("want 1 system block, got %d", len(in.System))
	}
	if _, ok := in.System[0].(*types.SystemContentBlockMemberCachePoint); ok {
		t.Fatal("CachePoint must not be added when cacheSystemPrompt=false")
	}
}

func TestConverseInputFromLLMRequest_cachePointOnlyFromContentsSystem(t *testing.T) {
	t.Parallel()
	// System prompt provided only via Contents (no SystemInstruction).
	// CachePoint must still appear after the system block.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Role: "system", Parts: []*genai.Part{{Text: "only from contents"}}},
			genai.NewContentFromText("hello", "user"),
		},
		Config: &genai.GenerateContentConfig{},
	}
	in, err := ConverseInputFromLLMRequest("mid", req, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(in.System) != 2 {
		t.Fatalf("want 2 system blocks (text + cachepoint), got %d: %#v", len(in.System), in.System)
	}
	if _, ok := in.System[1].(*types.SystemContentBlockMemberCachePoint); !ok {
		t.Fatalf("system[1] want CachePoint, got %T", in.System[1])
	}
}

func ptrFloat32(f float32) *float32 {
	p := f
	return &p
}
