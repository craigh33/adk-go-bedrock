package mappers

import (
	"encoding/json"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
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

func TestTokenUsageToGenai_cachedTokens(t *testing.T) {
	t.Parallel()
	u := &types.TokenUsage{
		InputTokens:          aws.Int32(100),
		OutputTokens:         aws.Int32(50),
		TotalTokens:          aws.Int32(150),
		CacheReadInputTokens: aws.Int32(40),
	}
	meta := TokenUsageToGenai(u)
	if meta == nil {
		t.Fatal("expected non-nil metadata")
	}
	if meta.PromptTokenCount != 100 {
		t.Errorf("PromptTokenCount: got %d, want 100", meta.PromptTokenCount)
	}
	if meta.CandidatesTokenCount != 50 {
		t.Errorf("CandidatesTokenCount: got %d, want 50", meta.CandidatesTokenCount)
	}
	if meta.TotalTokenCount != 150 {
		t.Errorf("TotalTokenCount: got %d, want 150", meta.TotalTokenCount)
	}
	if meta.CachedContentTokenCount != 40 {
		t.Errorf("CachedContentTokenCount: got %d, want 40", meta.CachedContentTokenCount)
	}
}

func TestTokenUsageToGenai_noCachedTokens(t *testing.T) {
	t.Parallel()
	u := &types.TokenUsage{
		InputTokens:  aws.Int32(10),
		OutputTokens: aws.Int32(5),
		TotalTokens:  aws.Int32(15),
	}
	meta := TokenUsageToGenai(u)
	if meta == nil {
		t.Fatal("expected non-nil metadata")
	}
	if meta.CachedContentTokenCount != 0 {
		t.Errorf("CachedContentTokenCount: got %d, want 0 when no cache read", meta.CachedContentTokenCount)
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

func TestMessageToGenaiContent_multimodalAndReasoning(t *testing.T) {
	t.Parallel()
	reasonText := "thinking"
	reasonSig := "sig-1"
	msg := &types.Message{
		Role: types.ConversationRoleAssistant,
		Content: []types.ContentBlock{
			&types.ContentBlockMemberImage{Value: types.ImageBlock{
				Format: types.ImageFormatPng,
				Source: &types.ImageSourceMemberBytes{Value: []byte{0x01, 0x02}},
			}},
			&types.ContentBlockMemberReasoningContent{
				Value: &types.ReasoningContentBlockMemberReasoningText{Value: types.ReasoningTextBlock{
					Text:      &reasonText,
					Signature: &reasonSig,
				}},
			},
			&types.ContentBlockMemberVideo{Value: types.VideoBlock{
				Format: types.VideoFormatMp4,
				Source: &types.VideoSourceMemberS3Location{
					Value: types.S3Location{Uri: aws.String("s3://bucket/video.mp4")},
				},
			}},
		},
	}
	c, err := MessageToGenaiContent(msg)
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Parts) != 3 {
		t.Fatalf("parts: %+v", c.Parts)
	}
	if c.Parts[0].InlineData == nil || c.Parts[0].InlineData.MIMEType != "image/png" {
		t.Fatalf("image part: %+v", c.Parts[0])
	}
	if !c.Parts[1].Thought || c.Parts[1].Text != "thinking" || string(c.Parts[1].ThoughtSignature) != "sig-1" {
		t.Fatalf("reasoning part: %+v", c.Parts[1])
	}
	if c.Parts[2].FileData == nil || c.Parts[2].FileData.FileURI != "s3://bucket/video.mp4" {
		t.Fatalf("video part: %+v", c.Parts[2])
	}
}

func TestToolResultToFunctionResponse_media(t *testing.T) {
	t.Parallel()
	tr := &types.ToolResultBlock{
		ToolUseId: aws.String("call_1"),
		Content: []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberJson{Value: brdoc.NewLazyDocument(map[string]any{"ok": true})},
			&types.ToolResultContentBlockMemberImage{Value: types.ImageBlock{
				Format: types.ImageFormatPng,
				Source: &types.ImageSourceMemberBytes{Value: []byte{0x01}},
			}},
			&types.ToolResultContentBlockMemberDocument{Value: types.DocumentBlock{
				Name:   aws.String("report.pdf"),
				Format: types.DocumentFormatPdf,
				Source: &types.DocumentSourceMemberS3Location{
					Value: types.S3Location{Uri: aws.String("s3://bucket/report.pdf")},
				},
			}},
		},
	}
	fr, err := toolResultToFunctionResponse(tr)
	if err != nil {
		t.Fatal(err)
	}
	if fr.ID != "call_1" || fr.Response["ok"] != true {
		t.Fatalf("response: %+v", fr)
	}
	if len(fr.Parts) != 2 {
		t.Fatalf("parts: %+v", fr.Parts)
	}
	if fr.Parts[0].InlineData == nil || fr.Parts[0].InlineData.MIMEType != "image/png" {
		t.Fatalf("image part: %+v", fr.Parts[0])
	}
	if fr.Parts[1].FileData == nil || fr.Parts[1].FileData.FileURI != "s3://bucket/report.pdf" {
		t.Fatalf("document part: %+v", fr.Parts[1])
	}
}

func TestDocumentToMap_numericToolResultAndToolUse(t *testing.T) {
	t.Parallel()
	tr := &types.ToolResultBlock{
		ToolUseId: aws.String("call_num"),
		Content: []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberJson{Value: brdoc.NewLazyDocument(map[string]any{
				"count": float64(42),
				"ratio": 0.25,
			})},
		},
	}
	fr, err := toolResultToFunctionResponse(tr)
	if err != nil {
		t.Fatal(err)
	}
	c, ok := fr.Response["count"].(float64)
	if !ok || c != 42 {
		t.Fatalf("count: got %T %v want float64 42", fr.Response["count"], fr.Response["count"])
	}
	r, ok := fr.Response["ratio"].(float64)
	if !ok || r != 0.25 {
		t.Fatalf("ratio: got %T %v want float64 0.25", fr.Response["ratio"], fr.Response["ratio"])
	}
	b, err := json.Marshal(fr.Response)
	if err != nil {
		t.Fatal(err)
	}
	var roundtrip map[string]any
	if err := json.Unmarshal(b, &roundtrip); err != nil {
		t.Fatal(err)
	}
	if _, ok := roundtrip["count"].(float64); !ok {
		t.Fatalf("JSON round-trip count not numeric: %v", roundtrip["count"])
	}

	msg := &types.Message{
		Role: types.ConversationRoleAssistant,
		Content: []types.ContentBlock{
			&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String("tool_1"),
				Name:      aws.String("demo"),
				Input:     brdoc.NewLazyDocument(map[string]any{"limit": float64(10), "offset": 0.0}),
			}},
		},
	}
	content, err := MessageToGenaiContent(msg)
	if err != nil {
		t.Fatal(err)
	}
	if len(content.Parts) != 1 || content.Parts[0].FunctionCall == nil {
		t.Fatalf("parts: %+v", content.Parts)
	}
	args := content.Parts[0].FunctionCall.Args
	if lim, ok := args["limit"].(float64); !ok || lim != 10 {
		t.Fatalf("limit: got %T %v", args["limit"], args["limit"])
	}
	if off, ok := args["offset"].(float64); !ok || off != 0 {
		t.Fatalf("offset: got %T %v", args["offset"], args["offset"])
	}
}

func TestLLMResponseFromConverseOutput_guardrailMetadata(t *testing.T) {
	t.Parallel()
	out := &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{Value: types.Message{
			Role: types.ConversationRoleAssistant,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "blocked"},
			},
		}},
		StopReason: types.StopReasonGuardrailIntervened,
		Trace: &types.ConverseTrace{Guardrail: &types.GuardrailTraceAssessment{
			OutputAssessments: map[string][]types.GuardrailAssessment{
				"0": {{
					SensitiveInformationPolicy: &types.GuardrailSensitiveInformationPolicyAssessment{
						PiiEntities: []types.GuardrailPiiEntityFilter{
							{Action: types.GuardrailSensitiveInformationPolicyActionBlocked},
						},
					},
					ContentPolicy: &types.GuardrailContentPolicyAssessment{
						Filters: []types.GuardrailContentFilter{{
							Type:           types.GuardrailContentFilterTypeHate,
							Confidence:     types.GuardrailContentFilterConfidenceHigh,
							FilterStrength: types.GuardrailContentFilterStrengthMedium,
							Action:         types.GuardrailContentPolicyActionBlocked,
						}},
					},
				}},
			},
		}},
	}
	resp, err := LLMResponseFromConverseOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != genai.FinishReasonSPII {
		t.Fatalf("finish reason: %v", resp.FinishReason)
	}
	ratings, ok := resp.CustomMetadata[customMetadataKeySafetyRatings].([]*genai.SafetyRating)
	if !ok || len(ratings) != 1 {
		t.Fatalf("safety ratings: %+v", resp.CustomMetadata)
	}
	if ratings[0].Category != genai.HarmCategoryHateSpeech || !ratings[0].Blocked {
		t.Fatalf("rating: %+v", ratings[0])
	}
}
