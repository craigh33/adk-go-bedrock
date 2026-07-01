package mantle

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func marshalToMap(t *testing.T, v any) map[string]any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return m
}

func TestMessageParamsFromConverseInput_Basic(t *testing.T) {
	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
		System:  []types.SystemContentBlock{&types.SystemContentBlockMemberText{Value: "be brief"}},
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}},
		}},
		InferenceConfig: &types.InferenceConfiguration{
			Temperature:   aws.Float32(0.5),
			TopP:          aws.Float32(0.25),
			MaxTokens:     aws.Int32(256),
			StopSequences: []string{"STOP"},
		},
	}

	params, err := MessageParamsFromConverseInput(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.Model != "anthropic.claude-sonnet-4-5-20250929-v1:0" {
		t.Errorf("model = %q, want normalized anthropic id", params.Model)
	}
	if params.MaxTokens != 256 {
		t.Errorf("max tokens = %d, want 256", params.MaxTokens)
	}
	if len(params.System) != 1 || params.System[0].Text != "be brief" {
		t.Errorf("system = %+v, want single 'be brief' block", params.System)
	}

	m := marshalToMap(t, params)
	if got := m["temperature"]; got != 0.5 {
		t.Errorf("temperature = %v, want 0.5", got)
	}
	if got := m["top_p"]; got != 0.25 {
		t.Errorf("top_p = %v, want 0.25", got)
	}
}

func TestMessageParamsFromConverseInput_DefaultMaxTokens(t *testing.T) {
	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("anthropic.claude-3-haiku"),
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}},
		}},
	}
	params, err := MessageParamsFromConverseInput(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if params.MaxTokens != defaultMaxTokens {
		t.Errorf("max tokens = %d, want default %d", params.MaxTokens, defaultMaxTokens)
	}
}

func TestMessageParamsFromConverseInput_Image(t *testing.T) {
	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("anthropic.claude-3-haiku"),
		Messages: []types.Message{{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberImage{Value: types.ImageBlock{
				Format: types.ImageFormatPng,
				Source: &types.ImageSourceMemberBytes{Value: []byte{0x1, 0x2, 0x3}},
			}}},
		}},
	}
	params, err := MessageParamsFromConverseInput(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	m := marshalToMap(t, params)
	block := firstContentBlock(t, m)
	if block["type"] != "image" {
		t.Fatalf("block type = %v, want image", block["type"])
	}
	source, ok := block["source"].(map[string]any)
	if !ok {
		t.Fatalf("image source missing: %v", block["source"])
	}
	if source["media_type"] != "image/png" {
		t.Errorf("media_type = %v, want image/png", source["media_type"])
	}
	if source["data"] != "AQID" { // base64 of 0x01 0x02 0x03
		t.Errorf("data = %v, want base64 AQID", source["data"])
	}
}

func TestMessageParamsFromConverseInput_ToolUseAndResult(t *testing.T) {
	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("anthropic.claude-3-haiku"),
		Messages: []types.Message{
			{
				Role: types.ConversationRoleAssistant,
				Content: []types.ContentBlock{&types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
					ToolUseId: aws.String("tool_1"),
					Name:      aws.String("get_weather"),
					Input:     brdoc.NewLazyDocument(map[string]any{"city": "NYC"}),
				}}},
			},
			{
				Role: types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{
					ToolUseId: aws.String("tool_1"),
					Content: []types.ToolResultContentBlock{&types.ToolResultContentBlockMemberJson{
						Value: brdoc.NewLazyDocument(map[string]any{"temp": 72}),
					}},
				}}},
			},
		},
	}
	params, err := MessageParamsFromConverseInput(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	m := marshalToMap(t, params)
	messages, _ := m["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(messages))
	}

	toolUse := nthContentBlock(t, messages[0], 0)
	if toolUse["type"] != "tool_use" || toolUse["name"] != "get_weather" || toolUse["id"] != "tool_1" {
		t.Errorf("tool_use block = %+v", toolUse)
	}

	toolResult := nthContentBlock(t, messages[1], 0)
	if toolResult["type"] != "tool_result" || toolResult["tool_use_id"] != "tool_1" {
		t.Errorf("tool_result block = %+v", toolResult)
	}
}

func TestMessageParamsFromConverseInput_ToolsAndChoice(t *testing.T) {
	in := &bedrockruntime.ConverseInput{
		ModelId: aws.String("anthropic.claude-3-haiku"),
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "weather?"}},
		}},
		ToolConfig: &types.ToolConfiguration{
			Tools: []types.Tool{&types.ToolMemberToolSpec{Value: types.ToolSpecification{
				Name:        aws.String("get_weather"),
				Description: aws.String("look up weather"),
				InputSchema: &types.ToolInputSchemaMemberJson{Value: brdoc.NewLazyDocument(map[string]any{
					"type":       "object",
					"properties": map[string]any{"city": map[string]any{"type": "string"}},
					"required":   []any{"city"},
				})},
			}}},
			ToolChoice: &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}},
		},
	}
	params, err := MessageParamsFromConverseInput(in)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	m := marshalToMap(t, params)
	tools, _ := m["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1", len(tools))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["name"] != "get_weather" || tool["description"] != "look up weather" {
		t.Errorf("tool = %+v", tool)
	}
	schema, _ := tool["input_schema"].(map[string]any)
	if _, ok := schema["properties"].(map[string]any); !ok {
		t.Errorf("input_schema.properties missing: %+v", schema)
	}
	choice, _ := m["tool_choice"].(map[string]any)
	if choice["type"] != "auto" {
		t.Errorf("tool_choice = %+v, want auto", choice)
	}
}

func TestMessageParamsFromConverseInput_UnsupportedCapabilities(t *testing.T) {
	base := func() *bedrockruntime.ConverseInput {
		return &bedrockruntime.ConverseInput{
			ModelId: aws.String("anthropic.claude-3-haiku"),
			Messages: []types.Message{{
				Role:    types.ConversationRoleUser,
				Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}},
			}},
		}
	}

	guardrails := base()
	guardrails.GuardrailConfig = &types.GuardrailConfiguration{
		GuardrailIdentifier: aws.String("g"),
		GuardrailVersion:    aws.String("1"),
	}
	if _, err := MessageParamsFromConverseInput(guardrails); err == nil {
		t.Error("guardrail config: expected error")
	}

	output := base()
	output.OutputConfig = &types.OutputConfig{}
	if _, err := MessageParamsFromConverseInput(output); err == nil {
		t.Error("output config: expected error")
	}

	additional := base()
	additional.AdditionalModelRequestFields = brdoc.NewLazyDocument(map[string]any{"x": 1})
	if _, err := MessageParamsFromConverseInput(additional); err == nil {
		t.Error("additional model request fields: expected error")
	}
}

func TestMessageParamsFromConverseInput_Errors(t *testing.T) {
	if _, err := MessageParamsFromConverseInput(nil); err == nil {
		t.Error("nil input: expected error")
	}

	badModel := &bedrockruntime.ConverseInput{
		ModelId: aws.String("amazon.nova-2-lite-v1:0"),
		Messages: []types.Message{{
			Role:    types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: "hi"}},
		}},
	}
	if _, err := MessageParamsFromConverseInput(badModel); err == nil {
		t.Error("non-anthropic model: expected error")
	}

	unsupportedBlock := &bedrockruntime.ConverseInput{
		ModelId: aws.String("anthropic.claude-3-haiku"),
		Messages: []types.Message{{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{&types.ContentBlockMemberDocument{Value: types.DocumentBlock{
				Format: types.DocumentFormatPdf,
				Source: &types.DocumentSourceMemberBytes{Value: []byte{0x1}},
			}}},
		}},
	}
	if _, err := MessageParamsFromConverseInput(unsupportedBlock); err == nil {
		t.Error("document block: expected error")
	}
}

func TestConverseOutputFromMessage(t *testing.T) {
	msg := mustUnmarshalMessage(t, `{
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "anthropic.claude-3-haiku",
		"stop_reason": "tool_use",
		"content": [
			{"type": "text", "text": "Let me check."},
			{"type": "thinking", "thinking": "the user wants weather", "signature": "sig-abc"},
			{"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"city": "NYC"}}
		],
		"usage": {"input_tokens": 12, "output_tokens": 8}
	}`)

	out, err := ConverseOutputFromMessage(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out.StopReason != types.StopReasonToolUse {
		t.Errorf("stop reason = %q, want tool_use", out.StopReason)
	}
	if out.Usage == nil || aws.ToInt32(out.Usage.InputTokens) != 12 ||
		aws.ToInt32(out.Usage.OutputTokens) != 8 || aws.ToInt32(out.Usage.TotalTokens) != 20 {
		t.Errorf("usage = %+v, want 12/8/20", out.Usage)
	}

	msgMember, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		t.Fatalf("output type = %T, want message member", out.Output)
	}
	blocks := msgMember.Value.Content
	if len(blocks) != 3 {
		t.Fatalf("blocks = %d, want 3", len(blocks))
	}
	if text, ok := blocks[0].(*types.ContentBlockMemberText); !ok || text.Value != "Let me check." {
		t.Errorf("block 0 = %+v, want text", blocks[0])
	}
	reasoning, ok := blocks[1].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("block 1 = %T, want reasoning", blocks[1])
	}
	reasoningText, ok := reasoning.Value.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok || aws.ToString(reasoningText.Value.Text) != "the user wants weather" {
		t.Errorf("reasoning = %+v", reasoning.Value)
	}
	toolUse, ok := blocks[2].(*types.ContentBlockMemberToolUse)
	if !ok || aws.ToString(toolUse.Value.Name) != "get_weather" {
		t.Errorf("block 2 = %+v, want tool_use get_weather", blocks[2])
	}
}

func TestConverseOutputFromMessage_NilAndStopReasons(t *testing.T) {
	if _, err := ConverseOutputFromMessage(nil); err == nil {
		t.Error("nil message: expected error")
	}

	cases := map[anthropic.StopReason]types.StopReason{
		anthropic.StopReasonEndTurn:      types.StopReasonEndTurn,
		anthropic.StopReasonMaxTokens:    types.StopReasonMaxTokens,
		anthropic.StopReasonStopSequence: types.StopReasonStopSequence,
		anthropic.StopReasonToolUse:      types.StopReasonToolUse,
		anthropic.StopReasonPauseTurn:    types.StopReasonEndTurn,
		anthropic.StopReasonRefusal:      types.StopReasonContentFiltered,
	}
	for in, want := range cases {
		if got := stopReasonFromAnthropic(in); got != want {
			t.Errorf("stopReasonFromAnthropic(%q) = %q, want %q", in, got, want)
		}
	}
}

func mustUnmarshalMessage(t *testing.T, jsonStr string) *anthropic.Message {
	t.Helper()
	var m anthropic.Message
	if err := json.Unmarshal([]byte(jsonStr), &m); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return &m
}

func firstContentBlock(t *testing.T, params map[string]any) map[string]any {
	t.Helper()
	messages, _ := params["messages"].([]any)
	if len(messages) == 0 {
		t.Fatalf("no messages in %+v", params)
	}
	return nthContentBlock(t, messages[0], 0)
}

func nthContentBlock(t *testing.T, message any, n int) map[string]any {
	t.Helper()
	msg, ok := message.(map[string]any)
	if !ok {
		t.Fatalf("message not an object: %v", message)
	}
	content, _ := msg["content"].([]any)
	if len(content) <= n {
		t.Fatalf("content block %d missing in %+v", n, msg)
	}
	block, ok := content[n].(map[string]any)
	if !ok {
		t.Fatalf("content block %d not an object: %v", n, content[n])
	}
	return block
}
