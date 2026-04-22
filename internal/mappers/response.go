package mappers

import (
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// errSkipPart signals contentBlockToPart should omit this block (not an API error).
var errSkipPart = errors.New("skip part")

const (
	customMetadataKeyAdditionalModelResponseFields = "bedrock_additional_model_response_fields"
	customMetadataKeyGuardrailTrace                = "bedrock_guardrail_trace"
	customMetadataKeyPromptRouter                  = "bedrock_prompt_router"
	customMetadataKeySafetyRatings                 = "safety_ratings"

	rankLow    = 1
	rankMedium = 2
	rankHigh   = 3
)

// LLMResponseFromConverseOutput maps a Bedrock [bedrockruntime.ConverseOutput] to [model.LLMResponse].
func LLMResponseFromConverseOutput(out *bedrockruntime.ConverseOutput) (*model.LLMResponse, error) {
	if out == nil {
		return nil, errors.New("nil ConverseOutput")
	}
	msg, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok || msg == nil {
		return nil, fmt.Errorf("unexpected Converse output type %T", out.Output)
	}
	content, err := MessageToGenaiContent(&msg.Value)
	if err != nil {
		return nil, err
	}
	return &model.LLMResponse{
		Content:        content,
		FinishReason:   FinishReasonFromStopReasonAndTrace(out.StopReason, guardrailTraceFromConverse(out.Trace)),
		UsageMetadata:  TokenUsageToGenai(out.Usage),
		CustomMetadata: customMetadataFromConverseOutput(out),
	}, nil
}

// MessageToGenaiContent converts a Bedrock assistant/user message to genai content.
func MessageToGenaiContent(m *types.Message) (*genai.Content, error) {
	if m == nil {
		return nil, errors.New("nil message")
	}
	role := "model"
	if m.Role == types.ConversationRoleUser {
		role = "user"
	}
	var parts []*genai.Part
	for _, b := range m.Content {
		p, err := contentBlockToPart(b)
		if err != nil {
			if errors.Is(err, errSkipPart) {
				continue
			}
			return nil, err
		}
		if p != nil {
			parts = append(parts, p)
		}
	}
	return &genai.Content{Role: role, Parts: parts}, nil
}

func contentBlockToPart(b types.ContentBlock) (*genai.Part, error) { //nolint:gocognit
	switch v := b.(type) {
	case *types.ContentBlockMemberAudio:
		if v == nil {
			return nil, errSkipPart
		}
		return audioBlockToPart(&v.Value)
	case *types.ContentBlockMemberDocument:
		if v == nil {
			return nil, errSkipPart
		}
		return documentBlockToPart(&v.Value)
	case *types.ContentBlockMemberImage:
		if v == nil {
			return nil, errSkipPart
		}
		return imageBlockToPart(&v.Value)
	case *types.ContentBlockMemberReasoningContent:
		if v == nil {
			return nil, errSkipPart
		}
		return reasoningContentBlockToPart(v.Value)
	case *types.ContentBlockMemberText:
		if v == nil || v.Value == "" {
			return nil, errSkipPart
		}
		return &genai.Part{Text: v.Value}, nil
	case *types.ContentBlockMemberToolUse:
		if v == nil {
			return nil, errSkipPart
		}
		args, err := documentToMap(v.Value.Input)
		if err != nil {
			return nil, fmt.Errorf("tool use input: %w", err)
		}
		id := ""
		if v.Value.ToolUseId != nil {
			id = *v.Value.ToolUseId
		}
		name := ""
		if v.Value.Name != nil {
			name = *v.Value.Name
		}
		return &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   id,
				Name: name,
				Args: args,
			},
		}, nil
	case *types.ContentBlockMemberToolResult:
		if v == nil {
			return nil, errSkipPart
		}
		resp, err := toolResultToFunctionResponse(&v.Value)
		if err != nil {
			return nil, err
		}
		return &genai.Part{FunctionResponse: resp}, nil
	case *types.ContentBlockMemberVideo:
		if v == nil {
			return nil, errSkipPart
		}
		return videoBlockToPart(&v.Value)
	default:
		return nil, errSkipPart
	}
}

func audioBlockToPart(b *types.AudioBlock) (*genai.Part, error) {
	if b == nil || b.Source == nil {
		return nil, errSkipPart
	}
	mime, err := audioMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	switch src := b.Source.(type) {
	case *types.AudioSourceMemberBytes:
		return &genai.Part{InlineData: &genai.Blob{Data: src.Value, MIMEType: mime}}, nil
	case *types.AudioSourceMemberS3Location:
		return &genai.Part{FileData: &genai.FileData{FileURI: awsToString(src.Value.Uri), MIMEType: mime}}, nil
	default:
		return nil, errSkipPart
	}
}

// documentBlockToPart converts a Bedrock document block to a genai Part.
//
//nolint:dupl // similar logic for different return types.
func documentBlockToPart(b *types.DocumentBlock) (*genai.Part, error) {
	if b == nil || b.Source == nil {
		return nil, errSkipPart
	}
	mime, err := documentMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	name := awsToString(b.Name)
	switch src := b.Source.(type) {
	case *types.DocumentSourceMemberBytes:
		return &genai.Part{InlineData: &genai.Blob{Data: src.Value, MIMEType: mime, DisplayName: name}}, nil
	case *types.DocumentSourceMemberContent:
		text, err := documentContentBlocksToText(src.Value)
		if err != nil {
			return nil, err
		}
		return &genai.Part{InlineData: &genai.Blob{Data: []byte(text), MIMEType: mime, DisplayName: name}}, nil
	case *types.DocumentSourceMemberS3Location:
		return &genai.Part{
			FileData: &genai.FileData{FileURI: awsToString(src.Value.Uri), MIMEType: mime, DisplayName: name},
		}, nil
	case *types.DocumentSourceMemberText:
		return &genai.Part{InlineData: &genai.Blob{Data: []byte(src.Value), MIMEType: mime, DisplayName: name}}, nil
	default:
		return nil, errSkipPart
	}
}

func imageBlockToPart(b *types.ImageBlock) (*genai.Part, error) {
	if b == nil || b.Source == nil {
		return nil, errSkipPart
	}
	mime, err := imageMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	switch src := b.Source.(type) {
	case *types.ImageSourceMemberBytes:
		return &genai.Part{InlineData: &genai.Blob{Data: src.Value, MIMEType: mime}}, nil
	case *types.ImageSourceMemberS3Location:
		return &genai.Part{FileData: &genai.FileData{FileURI: awsToString(src.Value.Uri), MIMEType: mime}}, nil
	default:
		return nil, errSkipPart
	}
}

func reasoningContentBlockToPart(b types.ReasoningContentBlock) (*genai.Part, error) {
	switch v := b.(type) {
	case *types.ReasoningContentBlockMemberReasoningText:
		if v == nil || v.Value.Text == nil || *v.Value.Text == "" {
			return nil, errSkipPart
		}
		part := &genai.Part{Text: *v.Value.Text, Thought: true}
		if v.Value.Signature != nil {
			part.ThoughtSignature = []byte(*v.Value.Signature)
		}
		return part, nil
	default:
		return nil, errSkipPart
	}
}

func videoBlockToPart(b *types.VideoBlock) (*genai.Part, error) {
	if b == nil || b.Source == nil {
		return nil, errSkipPart
	}
	mime, err := videoMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	switch src := b.Source.(type) {
	case *types.VideoSourceMemberBytes:
		return &genai.Part{InlineData: &genai.Blob{Data: src.Value, MIMEType: mime}}, nil
	case *types.VideoSourceMemberS3Location:
		return &genai.Part{FileData: &genai.FileData{FileURI: awsToString(src.Value.Uri), MIMEType: mime}}, nil
	default:
		return nil, errSkipPart
	}
}

func toolResultToFunctionResponse(b *types.ToolResultBlock) (*genai.FunctionResponse, error) { //nolint:gocognit
	id := ""
	if b.ToolUseId != nil {
		id = *b.ToolUseId
	}
	m := map[string]any{}
	var parts []*genai.FunctionResponsePart
	for _, c := range b.Content {
		switch t := c.(type) {
		case *types.ToolResultContentBlockMemberJson:
			if t == nil {
				continue
			}
			mm, err := documentToMap(t.Value)
			if err != nil {
				return nil, err
			}
			maps.Copy(m, mm)
		case *types.ToolResultContentBlockMemberText:
			m["text"] = t.Value
		case *types.ToolResultContentBlockMemberDocument:
			part, err := documentBlockToFunctionResponsePart(&t.Value)
			if err != nil {
				return nil, err
			}
			if part != nil {
				parts = append(parts, part)
			}
		case *types.ToolResultContentBlockMemberImage:
			part, err := imageBlockToFunctionResponsePart(&t.Value)
			if err != nil {
				return nil, err
			}
			if part != nil {
				parts = append(parts, part)
			}
		case *types.ToolResultContentBlockMemberVideo:
			part, err := videoBlockToFunctionResponsePart(&t.Value)
			if err != nil {
				return nil, err
			}
			if part != nil {
				parts = append(parts, part)
			}
		}
	}
	return &genai.FunctionResponse{
		ID:       id,
		Name:     "",
		Parts:    parts,
		Response: m,
	}, nil
}

//nolint:dupl // similar logic for different return types.
func documentBlockToFunctionResponsePart(b *types.DocumentBlock) (*genai.FunctionResponsePart, error) {
	if b == nil || b.Source == nil {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	mime, err := documentMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	name := awsToString(b.Name)
	switch src := b.Source.(type) {
	case *types.DocumentSourceMemberBytes:
		return &genai.FunctionResponsePart{
			InlineData: &genai.FunctionResponseBlob{Data: src.Value, MIMEType: mime, DisplayName: name},
		}, nil
	case *types.DocumentSourceMemberContent:
		text, err := documentContentBlocksToText(src.Value)
		if err != nil {
			return nil, err
		}
		return &genai.FunctionResponsePart{
			InlineData: &genai.FunctionResponseBlob{Data: []byte(text), MIMEType: mime, DisplayName: name},
		}, nil
	case *types.DocumentSourceMemberS3Location:
		return &genai.FunctionResponsePart{
			FileData: &genai.FunctionResponseFileData{
				FileURI:     awsToString(src.Value.Uri),
				MIMEType:    mime,
				DisplayName: name,
			},
		}, nil
	case *types.DocumentSourceMemberText:
		return &genai.FunctionResponsePart{
			InlineData: &genai.FunctionResponseBlob{Data: []byte(src.Value), MIMEType: mime, DisplayName: name},
		}, nil
	default:
		return nil, nil //nolint:nilnil // Unknown union member is intentionally skipped.
	}
}

func imageBlockToFunctionResponsePart(b *types.ImageBlock) (*genai.FunctionResponsePart, error) {
	if b == nil || b.Source == nil {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	mime, err := imageMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	switch src := b.Source.(type) {
	case *types.ImageSourceMemberBytes:
		return &genai.FunctionResponsePart{
			InlineData: &genai.FunctionResponseBlob{Data: src.Value, MIMEType: mime},
		}, nil
	case *types.ImageSourceMemberS3Location:
		return &genai.FunctionResponsePart{
			FileData: &genai.FunctionResponseFileData{FileURI: awsToString(src.Value.Uri), MIMEType: mime},
		}, nil
	default:
		return nil, nil //nolint:nilnil // Unknown union member is intentionally skipped.
	}
}

func videoBlockToFunctionResponsePart(b *types.VideoBlock) (*genai.FunctionResponsePart, error) {
	if b == nil || b.Source == nil {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	mime, err := videoMIMEFromFormat(b.Format)
	if err != nil {
		return nil, err
	}
	switch src := b.Source.(type) {
	case *types.VideoSourceMemberBytes:
		return &genai.FunctionResponsePart{
			InlineData: &genai.FunctionResponseBlob{Data: src.Value, MIMEType: mime},
		}, nil
	case *types.VideoSourceMemberS3Location:
		return &genai.FunctionResponsePart{
			FileData: &genai.FunctionResponseFileData{FileURI: awsToString(src.Value.Uri), MIMEType: mime},
		}, nil
	default:
		return nil, nil //nolint:nilnil // Unknown union member is intentionally skipped.
	}
}

func documentToMap(d brdoc.Interface) (map[string]any, error) {
	if d == nil {
		return map[string]any{}, nil
	}
	b, err := d.MarshalSmithyDocument()
	if err != nil {
		return nil, err
	}
	var raw any
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	raw = dereferenceJSONValue(raw)
	switch out := raw.(type) {
	case nil:
		return map[string]any{}, nil
	case map[string]any:
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported json type, %T", raw)
	}
}

func dereferenceJSONValue(v any) any {
	for v != nil {
		rv := reflect.ValueOf(v)
		switch rv.Kind() { //nolint:exhaustive // pointer/interface types are only case of interest
		case reflect.Pointer, reflect.Interface:
			if rv.IsNil() {
				return nil
			}
			v = rv.Elem().Interface()
		default:
			return v
		}
	}
	return v
}

// TokenUsageToGenai maps Bedrock token usage to genai usage metadata.
func TokenUsageToGenai(u *types.TokenUsage) *genai.GenerateContentResponseUsageMetadata {
	if u == nil {
		return nil
	}
	meta := &genai.GenerateContentResponseUsageMetadata{}
	if u.InputTokens != nil {
		meta.PromptTokenCount = *u.InputTokens
	}
	if u.OutputTokens != nil {
		meta.CandidatesTokenCount = *u.OutputTokens
	}
	if u.TotalTokens != nil {
		meta.TotalTokenCount = *u.TotalTokens
	}

	if u.CacheReadInputTokens != nil {
		// The ADK only supports returning how many cached tokens have been used (i.e. read)
		// not how many have been written.
		meta.CachedContentTokenCount = *u.CacheReadInputTokens
	}
	return meta
}

// StopReasonToFinishReason maps Bedrock stop reasons to genai finish reasons.
func StopReasonToFinishReason(sr types.StopReason) genai.FinishReason {
	return FinishReasonFromStopReasonAndTrace(sr, nil)
}

func FinishReasonFromStopReasonAndTrace(sr types.StopReason, trace *types.GuardrailTraceAssessment) genai.FinishReason {
	switch sr {
	case types.StopReasonEndTurn, types.StopReasonStopSequence, types.StopReasonToolUse:
		return genai.FinishReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return genai.FinishReasonMaxTokens
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return finishReasonFromGuardrailTrace(trace)
	case types.StopReasonMalformedToolUse, types.StopReasonMalformedModelOutput:
		return genai.FinishReasonMalformedFunctionCall
	default:
		return genai.FinishReasonOther
	}
}

// StreamMetadataToUsage extracts usage from a Converse stream metadata event.
func StreamMetadataToUsage(meta *types.ConverseStreamMetadataEvent) *genai.GenerateContentResponseUsageMetadata {
	if meta == nil {
		return nil
	}
	return TokenUsageToGenai(meta.Usage)
}

func StreamMetadataToCustomMetadata(meta *types.ConverseStreamMetadataEvent) map[string]any {
	if meta == nil {
		return nil
	}
	md := map[string]any{}
	if meta.Trace != nil && meta.Trace.Guardrail != nil {
		md[customMetadataKeyGuardrailTrace] = meta.Trace.Guardrail
		if ratings := SafetyRatingsFromGuardrailTrace(meta.Trace.Guardrail); len(ratings) > 0 {
			md[customMetadataKeySafetyRatings] = ratings
		}
	}
	if meta.ServiceTier != nil {
		md["bedrock_service_tier"] = *meta.ServiceTier
	}
	if meta.PerformanceConfig != nil {
		md["bedrock_performance_config"] = meta.PerformanceConfig
	}
	if len(md) == 0 {
		return nil
	}
	return md
}

func customMetadataFromConverseOutput(out *bedrockruntime.ConverseOutput) map[string]any {
	if out == nil {
		return nil
	}
	md := map[string]any{}
	if out.AdditionalModelResponseFields != nil {
		md[customMetadataKeyAdditionalModelResponseFields] = out.AdditionalModelResponseFields
	}
	if out.Trace != nil {
		if out.Trace.Guardrail != nil {
			md[customMetadataKeyGuardrailTrace] = out.Trace.Guardrail
			if ratings := SafetyRatingsFromGuardrailTrace(out.Trace.Guardrail); len(ratings) > 0 {
				md[customMetadataKeySafetyRatings] = ratings
			}
		}
		if out.Trace.PromptRouter != nil {
			md[customMetadataKeyPromptRouter] = out.Trace.PromptRouter
		}
	}
	if len(md) == 0 {
		return nil
	}
	return md
}

func guardrailTraceFromConverse(trace *types.ConverseTrace) *types.GuardrailTraceAssessment {
	if trace == nil {
		return nil
	}
	return trace.Guardrail
}

func SafetyRatingsFromGuardrailTrace(trace *types.GuardrailTraceAssessment) []*genai.SafetyRating {
	if trace == nil {
		return nil
	}
	ratingsByCategory := map[genai.HarmCategory]*genai.SafetyRating{}
	for _, assessment := range trace.InputAssessment {
		mergeSafetyRatings(ratingsByCategory, safetyRatingsFromAssessment(&assessment))
	}
	for _, assessments := range trace.OutputAssessments {
		for _, assessment := range assessments {
			mergeSafetyRatings(ratingsByCategory, safetyRatingsFromAssessment(&assessment))
		}
	}
	out := make([]*genai.SafetyRating, 0, len(ratingsByCategory))
	for _, rating := range ratingsByCategory {
		out = append(out, rating)
	}
	return out
}

func safetyRatingsFromAssessment(assessment *types.GuardrailAssessment) []*genai.SafetyRating {
	if assessment == nil || assessment.ContentPolicy == nil {
		return nil
	}
	out := make([]*genai.SafetyRating, 0, len(assessment.ContentPolicy.Filters))
	for _, filter := range assessment.ContentPolicy.Filters {
		category, ok := harmCategoryFromGuardrailFilterType(filter.Type)
		if !ok {
			continue
		}
		out = append(out, &genai.SafetyRating{
			Blocked:          filter.Action == types.GuardrailContentPolicyActionBlocked,
			Category:         category,
			Probability:      harmProbabilityFromGuardrailConfidence(filter.Confidence),
			ProbabilityScore: guardrailConfidenceScore(filter.Confidence),
			Severity:         harmSeverityFromGuardrailStrength(filter.FilterStrength),
			SeverityScore:    guardrailStrengthScore(filter.FilterStrength),
		})
	}
	return out
}

func mergeSafetyRatings(dst map[genai.HarmCategory]*genai.SafetyRating, ratings []*genai.SafetyRating) {
	for _, rating := range ratings {
		if rating == nil {
			continue
		}
		if existing, ok := dst[rating.Category]; ok {
			if !existing.Blocked && rating.Blocked {
				existing.Blocked = true
			}
			if probabilityRank(rating.Probability) > probabilityRank(existing.Probability) {
				existing.Probability = rating.Probability
				existing.ProbabilityScore = rating.ProbabilityScore
			}
			if severityRank(rating.Severity) > severityRank(existing.Severity) {
				existing.Severity = rating.Severity
				existing.SeverityScore = rating.SeverityScore
			}
			continue
		}
		dst[rating.Category] = rating
	}
}

func finishReasonFromGuardrailTrace(trace *types.GuardrailTraceAssessment) genai.FinishReason {
	if trace == nil {
		return genai.FinishReasonSafety
	}
	if guardrailTraceHasSensitiveInfoBlock(trace) {
		return genai.FinishReasonSPII
	}
	if guardrailTraceHasWordBlock(trace) {
		return genai.FinishReasonBlocklist
	}
	if guardrailTraceHasTopicBlock(trace) {
		return genai.FinishReasonProhibitedContent
	}
	return genai.FinishReasonSafety
}

func guardrailTraceHasSensitiveInfoBlock(trace *types.GuardrailTraceAssessment) bool {
	return guardrailTraceAny(trace, func(assessment types.GuardrailAssessment) bool {
		if assessment.SensitiveInformationPolicy == nil {
			return false
		}
		for _, entity := range assessment.SensitiveInformationPolicy.PiiEntities {
			if entity.Action == types.GuardrailSensitiveInformationPolicyActionBlocked ||
				entity.Action == types.GuardrailSensitiveInformationPolicyActionAnonymized {
				return true
			}
		}
		for _, regex := range assessment.SensitiveInformationPolicy.Regexes {
			if regex.Action == types.GuardrailSensitiveInformationPolicyActionBlocked ||
				regex.Action == types.GuardrailSensitiveInformationPolicyActionAnonymized {
				return true
			}
		}
		return false
	})
}

func guardrailTraceHasTopicBlock(trace *types.GuardrailTraceAssessment) bool {
	return guardrailTraceAny(trace, func(assessment types.GuardrailAssessment) bool {
		if assessment.TopicPolicy == nil {
			return false
		}
		for _, topic := range assessment.TopicPolicy.Topics {
			if topic.Action == types.GuardrailTopicPolicyActionBlocked {
				return true
			}
		}
		return false
	})
}

func guardrailTraceHasWordBlock(trace *types.GuardrailTraceAssessment) bool {
	return guardrailTraceAny(trace, func(assessment types.GuardrailAssessment) bool {
		if assessment.WordPolicy == nil {
			return false
		}
		for _, word := range assessment.WordPolicy.CustomWords {
			if word.Action == types.GuardrailWordPolicyActionBlocked {
				return true
			}
		}
		for _, word := range assessment.WordPolicy.ManagedWordLists {
			if word.Action == types.GuardrailWordPolicyActionBlocked {
				return true
			}
		}
		return false
	})
}

func guardrailTraceAny(trace *types.GuardrailTraceAssessment, pred func(types.GuardrailAssessment) bool) bool {
	if trace == nil {
		return false
	}
	for _, assessment := range trace.InputAssessment {
		if pred(assessment) {
			return true
		}
	}
	for _, assessments := range trace.OutputAssessments {
		if slices.ContainsFunc(assessments, pred) {
			return true
		}
	}
	return false
}

func harmCategoryFromGuardrailFilterType(t types.GuardrailContentFilterType) (genai.HarmCategory, bool) {
	switch t {
	case types.GuardrailContentFilterTypeInsults:
		return genai.HarmCategoryHarassment, true
	case types.GuardrailContentFilterTypeHate:
		return genai.HarmCategoryHateSpeech, true
	case types.GuardrailContentFilterTypeSexual:
		return genai.HarmCategorySexuallyExplicit, true
	case types.GuardrailContentFilterTypeViolence, types.GuardrailContentFilterTypeMisconduct:
		return genai.HarmCategoryDangerousContent, true
	case types.GuardrailContentFilterTypePromptAttack:
		return genai.HarmCategoryJailbreak, true
	default:
		return "", false
	}
}

func harmProbabilityFromGuardrailConfidence(c types.GuardrailContentFilterConfidence) genai.HarmProbability {
	switch c { //nolint:exhaustive // None case handled by default
	case types.GuardrailContentFilterConfidenceLow:
		return genai.HarmProbabilityLow
	case types.GuardrailContentFilterConfidenceMedium:
		return genai.HarmProbabilityMedium
	case types.GuardrailContentFilterConfidenceHigh:
		return genai.HarmProbabilityHigh
	default:
		return genai.HarmProbabilityUnspecified
	}
}

func harmSeverityFromGuardrailStrength(s types.GuardrailContentFilterStrength) genai.HarmSeverity {
	switch s { //nolint:exhaustive // None case handled by default
	case types.GuardrailContentFilterStrengthLow:
		return genai.HarmSeverityLow
	case types.GuardrailContentFilterStrengthMedium:
		return genai.HarmSeverityMedium
	case types.GuardrailContentFilterStrengthHigh:
		return genai.HarmSeverityHigh
	default:
		return genai.HarmSeverityUnspecified
	}
}

func guardrailConfidenceScore(c types.GuardrailContentFilterConfidence) float32 {
	switch c { //nolint:exhaustive // None case handled by default
	case types.GuardrailContentFilterConfidenceLow:
		return 0.33 //nolint:mnd // Score mapping for low confidence
	case types.GuardrailContentFilterConfidenceMedium:
		return 0.66 //nolint:mnd // Score mapping for medium confidence
	case types.GuardrailContentFilterConfidenceHigh:
		return 1.0
	default:
		return 0
	}
}

func guardrailStrengthScore(s types.GuardrailContentFilterStrength) float32 {
	switch s { //nolint:exhaustive // None case handled by default
	case types.GuardrailContentFilterStrengthLow:
		return 0.33 //nolint:mnd // Score mapping for low strength
	case types.GuardrailContentFilterStrengthMedium:
		return 0.66 //nolint:mnd // Score mapping for medium strength
	case types.GuardrailContentFilterStrengthHigh:
		return 1.0
	default:
		return 0
	}
}

func probabilityRank(p genai.HarmProbability) int {
	switch p {
	case genai.HarmProbabilityUnspecified:
		return 0
	case genai.HarmProbabilityNegligible, genai.HarmProbabilityLow:
		return rankLow
	case genai.HarmProbabilityMedium:
		return rankMedium
	case genai.HarmProbabilityHigh:
		return rankHigh
	default:
		return 0
	}
}

func severityRank(s genai.HarmSeverity) int {
	switch s {
	case genai.HarmSeverityUnspecified:
		return 0
	case genai.HarmSeverityNegligible, genai.HarmSeverityLow:
		return rankLow
	case genai.HarmSeverityMedium:
		return rankMedium
	case genai.HarmSeverityHigh:
		return rankHigh
	default:
		return 0
	}
}

func documentContentBlocksToText(blocks []types.DocumentContentBlock) (string, error) {
	var sb []string
	for _, block := range blocks {
		switch v := block.(type) {
		case *types.DocumentContentBlockMemberText:
			if v != nil && v.Value != "" {
				sb = append(sb, v.Value)
			}
		default:
			return "", fmt.Errorf("unsupported Bedrock document content block type %T", block)
		}
	}
	return joinTextBlocks(sb), nil
}

func joinTextBlocks(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	totalLen := 0
	for _, part := range parts {
		totalLen += len(part)
	}
	var out strings.Builder
	out.Grow(totalLen)
	for _, part := range parts {
		out.WriteString(part)
	}
	return out.String()
}

func awsToString(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func imageMIMEFromFormat(f types.ImageFormat) (string, error) {
	switch f {
	case types.ImageFormatJpeg:
		return "image/jpeg", nil
	case types.ImageFormatPng:
		return mimeImagePNG, nil
	case types.ImageFormatGif:
		return "image/gif", nil
	case types.ImageFormatWebp:
		return "image/webp", nil
	default:
		return "", fmt.Errorf("unsupported Bedrock image format: %q", f)
	}
}

func audioMIMEFromFormat(f types.AudioFormat) (string, error) {
	switch f {
	case types.AudioFormatMp3, types.AudioFormatMpeg:
		return "audio/mpeg", nil
	case types.AudioFormatOpus:
		return "audio/opus", nil
	case types.AudioFormatWav:
		return "audio/wav", nil
	case types.AudioFormatAac:
		return "audio/aac", nil
	case types.AudioFormatFlac:
		return "audio/flac", nil
	case types.AudioFormatMp4:
		return "audio/mp4", nil
	case types.AudioFormatOgg:
		return "audio/ogg", nil
	case types.AudioFormatMkv, types.AudioFormatMka:
		return "audio/x-matroska", nil
	case types.AudioFormatXAac:
		return "audio/x-aac", nil
	case types.AudioFormatM4a:
		return "audio/m4a", nil
	case types.AudioFormatMpga:
		return "audio/mpga", nil
	case types.AudioFormatPcm:
		return "audio/pcm", nil
	case types.AudioFormatWebm:
		return "audio/webm", nil
	default:
		return "", fmt.Errorf("unsupported Bedrock audio format: %q", f)
	}
}

func videoMIMEFromFormat(f types.VideoFormat) (string, error) {
	switch f {
	case types.VideoFormatMkv:
		return "video/x-matroska", nil
	case types.VideoFormatMov:
		return "video/quicktime", nil
	case types.VideoFormatMp4:
		return "video/mp4", nil
	case types.VideoFormatWebm:
		return "video/webm", nil
	case types.VideoFormatFlv:
		return "video/x-flv", nil
	case types.VideoFormatMpeg:
		return "video/mpeg", nil
	case types.VideoFormatMpg:
		return "video/mpg", nil
	case types.VideoFormatWmv:
		return "video/x-ms-wmv", nil
	case types.VideoFormatThreeGp:
		return "video/3gpp", nil
	default:
		return "", fmt.Errorf("unsupported Bedrock video format: %q", f)
	}
}

func documentMIMEFromFormat(f types.DocumentFormat) (string, error) {
	switch f {
	case types.DocumentFormatPdf:
		return "application/pdf", nil
	case types.DocumentFormatCsv:
		return "text/csv", nil
	case types.DocumentFormatDoc:
		return "application/msword", nil
	case types.DocumentFormatDocx:
		return "application/vnd.openxmlformats-officedocument.wordprocessingml.document", nil
	case types.DocumentFormatXls:
		return "application/vnd.ms-excel", nil
	case types.DocumentFormatXlsx:
		return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", nil
	case types.DocumentFormatHtml:
		return "text/html", nil
	case types.DocumentFormatTxt:
		return "text/plain", nil
	case types.DocumentFormatMd:
		return "text/markdown", nil
	default:
		return "", fmt.Errorf("unsupported Bedrock document format: %q", f)
	}
}
