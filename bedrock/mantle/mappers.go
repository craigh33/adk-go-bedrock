package mantle

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// defaultMaxTokens is used when a Converse request carries no MaxTokens. The
// Anthropic Messages API requires max_tokens, whereas Bedrock Converse treats it
// as optional, so a request that omits it still needs a concrete ceiling here.
const defaultMaxTokens = 4096

// MessageParamsFromConverseInput maps a Bedrock [bedrockruntime.ConverseInput] to
// Anthropic [anthropic.MessageNewParams]. It mirrors the Converse-side
// ConverseInputFromLLMRequest so the Mantle transport can satisfy the same
// RuntimeAPI contract without changes to the caller.
func MessageParamsFromConverseInput(in *bedrockruntime.ConverseInput) (anthropic.MessageNewParams, error) {
	if in == nil {
		return anthropic.MessageNewParams{}, errors.New("nil ConverseInput")
	}
	if err := checkUnsupportedCapabilities(in); err != nil {
		return anthropic.MessageNewParams{}, err
	}

	modelID := aws.ToString(in.ModelId)
	if err := ValidateModelID(modelID); err != nil {
		return anthropic.MessageNewParams{}, err
	}

	messages, err := messageParamsFromMessages(in.Messages)
	if err != nil {
		return anthropic.MessageNewParams{}, err
	}

	params := anthropic.MessageNewParams{
		Model:     NormalizeModelID(modelID),
		MaxTokens: defaultMaxTokens,
		Messages:  messages,
		System:    systemTextBlocksFromConverse(in.System),
	}

	applyInferenceConfig(&params, in.InferenceConfig)

	if err := applyToolConfig(&params, in.ToolConfig); err != nil {
		return anthropic.MessageNewParams{}, err
	}

	return params, nil
}

func systemTextBlocksFromConverse(system []types.SystemContentBlock) []anthropic.TextBlockParam {
	var blocks []anthropic.TextBlockParam
	for _, b := range system {
		if text, ok := b.(*types.SystemContentBlockMemberText); ok && text.Value != "" {
			blocks = append(blocks, anthropic.TextBlockParam{Text: text.Value})
		}
	}
	return blocks
}

func messageParamsFromMessages(msgs []types.Message) ([]anthropic.MessageParam, error) {
	out := make([]anthropic.MessageParam, 0, len(msgs))
	for i := range msgs {
		blocks, err := contentBlockParamsFromConverse(msgs[i].Content)
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		switch msgs[i].Role {
		case types.ConversationRoleUser:
			out = append(out, anthropic.NewUserMessage(blocks...))
		case types.ConversationRoleAssistant:
			out = append(out, anthropic.NewAssistantMessage(blocks...))
		case types.ConversationRoleSystem:
			return nil, errors.New("system messages must be provided via ConverseInput.System, not as a message role")
		default:
			return nil, fmt.Errorf("unsupported conversation role for bedrock Mantle: %q", msgs[i].Role)
		}
	}
	return out, nil
}

func contentBlockParamsFromConverse(blocks []types.ContentBlock) ([]anthropic.ContentBlockParamUnion, error) {
	var out []anthropic.ContentBlockParamUnion
	for _, b := range blocks {
		param, err := contentBlockParamFromConverse(b)
		if err != nil {
			return nil, err
		}
		if param != nil {
			out = append(out, *param)
		}
	}
	return out, nil
}

func contentBlockParamFromConverse(b types.ContentBlock) (*anthropic.ContentBlockParamUnion, error) {
	switch v := b.(type) {
	case *types.ContentBlockMemberText:
		if v.Value == "" {
			return nil, nil //nolint:nilnil // Empty text blocks are dropped.
		}
		block := anthropic.NewTextBlock(v.Value)
		return &block, nil
	case *types.ContentBlockMemberImage:
		return imageBlockParamFromConverse(&v.Value)
	case *types.ContentBlockMemberToolUse:
		return toolUseBlockParamFromConverse(&v.Value)
	case *types.ContentBlockMemberToolResult:
		return toolResultBlockParamFromConverse(&v.Value)
	case *types.ContentBlockMemberReasoningContent:
		return reasoningBlockParamFromConverse(v.Value), nil
	default:
		return nil, fmt.Errorf(
			"bedrock Mantle (Anthropic Messages API) does not support %T content blocks; "+
				"only text, image, reasoning, tool use, and tool result blocks are supported",
			b,
		)
	}
}

func imageBlockParamFromConverse(img *types.ImageBlock) (*anthropic.ContentBlockParamUnion, error) {
	src, ok := img.Source.(*types.ImageSourceMemberBytes)
	if !ok {
		return nil, errors.New("bedrock Mantle image blocks require inline bytes; S3 image sources are not supported")
	}
	mediaType, err := imageMediaTypeFromFormat(img.Format)
	if err != nil {
		return nil, err
	}
	block := anthropic.NewImageBlockBase64(mediaType, base64.StdEncoding.EncodeToString(src.Value))
	return &block, nil
}

func toolUseBlockParamFromConverse(tu *types.ToolUseBlock) (*anthropic.ContentBlockParamUnion, error) {
	input, err := documentToMap(tu.Input)
	if err != nil {
		return nil, fmt.Errorf("tool use %q input: %w", aws.ToString(tu.Name), err)
	}
	block := anthropic.NewToolUseBlock(aws.ToString(tu.ToolUseId), input, aws.ToString(tu.Name))
	return &block, nil
}

func toolResultBlockParamFromConverse(tr *types.ToolResultBlock) (*anthropic.ContentBlockParamUnion, error) {
	content, err := toolResultContentFromConverse(tr.Content)
	if err != nil {
		return nil, err
	}
	block := anthropic.ToolResultBlockParam{
		ToolUseID: aws.ToString(tr.ToolUseId),
		Content:   content,
	}
	if tr.Status == types.ToolResultStatusError {
		block.IsError = anthropic.Bool(true)
	}
	return &anthropic.ContentBlockParamUnion{OfToolResult: &block}, nil
}

func toolResultContentFromConverse(
	blocks []types.ToolResultContentBlock,
) ([]anthropic.ToolResultBlockParamContentUnion, error) {
	var out []anthropic.ToolResultBlockParamContentUnion
	for _, c := range blocks {
		switch t := c.(type) {
		case *types.ToolResultContentBlockMemberText:
			out = append(out, textResultContent(t.Value))
		case *types.ToolResultContentBlockMemberJson:
			m, err := documentToMap(t.Value)
			if err != nil {
				return nil, fmt.Errorf("tool result json: %w", err)
			}
			encoded, err := json.Marshal(m)
			if err != nil {
				return nil, fmt.Errorf("tool result json: %w", err)
			}
			out = append(out, textResultContent(string(encoded)))
		default:
			return nil, fmt.Errorf(
				"bedrock Mantle tool results support only text and JSON content, got %T",
				c,
			)
		}
	}
	return out, nil
}

func textResultContent(text string) anthropic.ToolResultBlockParamContentUnion {
	return anthropic.ToolResultBlockParamContentUnion{OfText: &anthropic.TextBlockParam{Text: text}}
}

func reasoningBlockParamFromConverse(rc types.ReasoningContentBlock) *anthropic.ContentBlockParamUnion {
	reasoning, ok := rc.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok {
		return nil
	}
	text := aws.ToString(reasoning.Value.Text)
	if text == "" {
		return nil
	}
	block := anthropic.NewThinkingBlock(aws.ToString(reasoning.Value.Signature), text)
	return &block
}

func applyInferenceConfig(params *anthropic.MessageNewParams, cfg *types.InferenceConfiguration) {
	if cfg == nil {
		return
	}
	if cfg.MaxTokens != nil {
		params.MaxTokens = int64(*cfg.MaxTokens)
	}
	if cfg.Temperature != nil {
		params.Temperature = anthropic.Float(float64(*cfg.Temperature))
	}
	if cfg.TopP != nil {
		params.TopP = anthropic.Float(float64(*cfg.TopP))
	}
	if len(cfg.StopSequences) > 0 {
		params.StopSequences = append([]string(nil), cfg.StopSequences...)
	}
}

func applyToolConfig(params *anthropic.MessageNewParams, cfg *types.ToolConfiguration) error {
	if cfg == nil {
		return nil
	}
	tools, err := toolParamsFromConverse(cfg.Tools)
	if err != nil {
		return err
	}
	params.Tools = tools

	choice, ok, err := toolChoiceFromConverse(cfg.ToolChoice)
	if err != nil {
		return err
	}
	if ok {
		params.ToolChoice = choice
	}
	return nil
}

func toolParamsFromConverse(tools []types.Tool) ([]anthropic.ToolUnionParam, error) {
	var out []anthropic.ToolUnionParam
	for _, t := range tools {
		spec, ok := t.(*types.ToolMemberToolSpec)
		if !ok {
			return nil, fmt.Errorf("bedrock Mantle supports only tool specifications, got %T", t)
		}
		schema, err := toolInputSchemaFromConverse(spec.Value.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("tool %q: %w", aws.ToString(spec.Value.Name), err)
		}
		tool := anthropic.ToolParam{
			Name:        aws.ToString(spec.Value.Name),
			InputSchema: schema,
		}
		if desc := aws.ToString(spec.Value.Description); desc != "" {
			tool.Description = anthropic.String(desc)
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return out, nil
}

func toolInputSchemaFromConverse(schema types.ToolInputSchema) (anthropic.ToolInputSchemaParam, error) {
	jsonSchema, ok := schema.(*types.ToolInputSchemaMemberJson)
	if !ok {
		return anthropic.ToolInputSchemaParam{}, fmt.Errorf("unsupported tool input schema type %T", schema)
	}
	m, err := documentToMap(jsonSchema.Value)
	if err != nil {
		return anthropic.ToolInputSchemaParam{}, err
	}
	out := anthropic.ToolInputSchemaParam{}
	if props, ok := m["properties"]; ok {
		out.Properties = props
	}
	if required, ok := m["required"].([]any); ok {
		out.Required = stringSlice(required)
	}
	out.ExtraFields = extraSchemaFields(m)
	return out, nil
}

// extraSchemaFields carries every schema key other than the ones with dedicated
// ToolInputSchemaParam fields, so constraints such as "$defs" or "description"
// survive the round trip to the Messages API.
func extraSchemaFields(schema map[string]any) map[string]any {
	extra := map[string]any{}
	for k, v := range schema {
		switch k {
		case "type", "properties", "required":
			continue
		default:
			extra[k] = v
		}
	}
	if len(extra) == 0 {
		return nil
	}
	return extra
}

func toolChoiceFromConverse(tc types.ToolChoice) (anthropic.ToolChoiceUnionParam, bool, error) {
	switch v := tc.(type) {
	case nil:
		return anthropic.ToolChoiceUnionParam{}, false, nil
	case *types.ToolChoiceMemberAuto:
		return anthropic.ToolChoiceUnionParam{OfAuto: &anthropic.ToolChoiceAutoParam{}}, true, nil
	case *types.ToolChoiceMemberAny:
		return anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}, true, nil
	case *types.ToolChoiceMemberTool:
		return anthropic.ToolChoiceParamOfTool(aws.ToString(v.Value.Name)), true, nil
	default:
		return anthropic.ToolChoiceUnionParam{}, false, fmt.Errorf("unsupported tool choice type %T", tc)
	}
}

// ConverseOutputFromMessage maps an Anthropic [anthropic.Message] back to a
// Bedrock [bedrockruntime.ConverseOutput]. It mirrors the Converse-side
// LLMResponseFromConverseOutput so the existing response translation and stream
// assembly continue to work unmodified.
func ConverseOutputFromMessage(msg *anthropic.Message) (*bedrockruntime.ConverseOutput, error) {
	if msg == nil {
		return nil, errors.New("nil Mantle message")
	}
	blocks, err := converseContentBlocksFromMessage(msg.Content)
	if err != nil {
		return nil, err
	}
	return &bedrockruntime.ConverseOutput{
		Output: &types.ConverseOutputMemberMessage{
			Value: types.Message{Role: types.ConversationRoleAssistant, Content: blocks},
		},
		StopReason: stopReasonFromAnthropic(msg.StopReason),
		Usage:      tokenUsageFromAnthropic(msg.Usage),
	}, nil
}

func converseContentBlocksFromMessage(blocks []anthropic.ContentBlockUnion) ([]types.ContentBlock, error) {
	out := make([]types.ContentBlock, 0, len(blocks))
	for i := range blocks {
		block, err := converseContentBlockFromMessage(blocks[i])
		if err != nil {
			return nil, err
		}
		if block != nil {
			out = append(out, block)
		}
	}
	return out, nil
}

func converseContentBlockFromMessage(b anthropic.ContentBlockUnion) (types.ContentBlock, error) {
	switch b.Type {
	case "text":
		if b.Text == "" {
			return nil, nil //nolint:nilnil // Empty text blocks are dropped.
		}
		return &types.ContentBlockMemberText{Value: b.Text}, nil
	case "thinking":
		return reasoningContentBlockFromMessage(b), nil
	case "tool_use":
		return toolUseContentBlockFromMessage(b)
	default:
		// Skip response variants without a Converse equivalent (redacted thinking,
		// server tool use, web search results). Bedrock may add block types over time.
		return nil, nil //nolint:nilnil // Unmapped block types are intentionally skipped.
	}
}

func reasoningContentBlockFromMessage(b anthropic.ContentBlockUnion) types.ContentBlock {
	thinking := b.AsThinking()
	if thinking.Thinking == "" {
		return nil
	}
	value := types.ReasoningTextBlock{Text: aws.String(thinking.Thinking)}
	if thinking.Signature != "" {
		value.Signature = aws.String(thinking.Signature)
	}
	return &types.ContentBlockMemberReasoningContent{
		Value: &types.ReasoningContentBlockMemberReasoningText{Value: value},
	}
}

func toolUseContentBlockFromMessage(b anthropic.ContentBlockUnion) (types.ContentBlock, error) {
	toolUse := b.AsToolUse()
	input, err := rawJSONToMap(toolUse.Input)
	if err != nil {
		return nil, fmt.Errorf("tool use %q input: %w", toolUse.Name, err)
	}
	return &types.ContentBlockMemberToolUse{
		Value: types.ToolUseBlock{
			ToolUseId: aws.String(toolUse.ID),
			Name:      aws.String(toolUse.Name),
			Input:     brdoc.NewLazyDocument(input),
		},
	}, nil
}

func stopReasonFromAnthropic(sr anthropic.StopReason) types.StopReason {
	switch sr {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonPauseTurn:
		return types.StopReasonEndTurn
	case anthropic.StopReasonMaxTokens:
		return types.StopReasonMaxTokens
	case anthropic.StopReasonStopSequence:
		return types.StopReasonStopSequence
	case anthropic.StopReasonToolUse:
		return types.StopReasonToolUse
	case anthropic.StopReasonRefusal:
		return types.StopReasonContentFiltered
	default:
		return types.StopReasonEndTurn
	}
}

func tokenUsageFromAnthropic(u anthropic.Usage) *types.TokenUsage {
	input := int32FromInt64(u.InputTokens)
	output := int32FromInt64(u.OutputTokens)
	return &types.TokenUsage{
		InputTokens:  aws.Int32(input),
		OutputTokens: aws.Int32(output),
		TotalTokens:  aws.Int32(int32FromInt64(u.InputTokens + u.OutputTokens)),
	}
}

func imageMediaTypeFromFormat(f types.ImageFormat) (string, error) {
	switch f {
	case types.ImageFormatJpeg:
		return "image/jpeg", nil
	case types.ImageFormatPng:
		return "image/png", nil
	case types.ImageFormatGif:
		return "image/gif", nil
	case types.ImageFormatWebp:
		return "image/webp", nil
	default:
		return "", fmt.Errorf("unsupported Bedrock Mantle image format: %q", f)
	}
}

func documentToMap(d brdoc.Interface) (map[string]any, error) {
	if d == nil {
		return map[string]any{}, nil
	}
	var m map[string]any
	if err := d.UnmarshalSmithyDocument(&m); err != nil {
		return nil, err
	}
	if m == nil {
		return map[string]any{}, nil
	}
	return m, nil
}

func rawJSONToMap(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, err
	}
	if m == nil {
		return map[string]any{}, nil
	}
	return m, nil
}

func stringSlice(values []any) []string {
	out := make([]string, 0, len(values))
	for _, v := range values {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func int32FromInt64(v int64) int32 {
	switch {
	case v > math.MaxInt32:
		return math.MaxInt32
	case v < math.MinInt32:
		return math.MinInt32
	default:
		return int32(v)
	}
}
