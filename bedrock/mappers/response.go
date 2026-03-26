package mappers

import (
	"errors"
	"fmt"
	"maps"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// errSkipPart signals contentBlockToPart should omit this block (not an API error).
var errSkipPart = errors.New("skip part")

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
		Content:       content,
		FinishReason:  StopReasonToFinishReason(out.StopReason),
		UsageMetadata: TokenUsageToGenai(out.Usage),
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

func contentBlockToPart(b types.ContentBlock) (*genai.Part, error) {
	switch v := b.(type) {
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
	default:
		return nil, errSkipPart
	}
}

func toolResultToFunctionResponse(b *types.ToolResultBlock) (*genai.FunctionResponse, error) {
	id := ""
	if b.ToolUseId != nil {
		id = *b.ToolUseId
	}
	m := map[string]any{}
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
		}
	}
	return &genai.FunctionResponse{
		ID:       id,
		Name:     "",
		Response: m,
	}, nil
}

func documentToMap(d brdoc.Interface) (map[string]any, error) {
	if d == nil {
		return map[string]any{}, nil
	}
	var out map[string]any
	if err := d.UnmarshalSmithyDocument(&out); err != nil {
		return nil, err
	}
	return out, nil
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
	return meta
}

// StopReasonToFinishReason maps Bedrock stop reasons to genai finish reasons.
func StopReasonToFinishReason(sr types.StopReason) genai.FinishReason {
	switch sr {
	case types.StopReasonEndTurn, types.StopReasonStopSequence, types.StopReasonToolUse:
		return genai.FinishReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return genai.FinishReasonMaxTokens
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return genai.FinishReasonSafety
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
