// Package mappers converts between ADK/genai types and Amazon Bedrock Converse API types.
package mappers

import (
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// MaybeAppendUserContent mirrors the Gemini provider behavior so empty histories or
// assistant-terminated turns still receive a valid user message.
func MaybeAppendUserContent(contents []*genai.Content) []*genai.Content {
	if len(contents) == 0 {
		return append(contents, genai.NewContentFromText(
			"Handle the requests as specified in the System Instruction.", "user"))
	}
	if last := contents[len(contents)-1]; last != nil && last.Role != "user" {
		return append(contents, genai.NewContentFromText(
			"Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.",
			"user",
		))
	}
	return contents
}

// ConverseInputFromLLMRequest builds a Bedrock [bedrockruntime.ConverseInput] from an ADK request.
func ConverseInputFromLLMRequest(modelID string, req *model.LLMRequest) (*bedrockruntime.ConverseInput, error) {
	if req == nil {
		return nil, errors.New("nil LLMRequest")
	}
	cfg := req.Config
	if cfg == nil {
		cfg = &genai.GenerateContentConfig{}
	}

	contents := MaybeAppendUserContent(append([]*genai.Content(nil), req.Contents...))

	system := buildSystemBlocks(cfg)
	sysFromContents, msgsFromContents := splitContents(contents)
	system = append(system, sysFromContents...)

	messages, err := contentsToMessages(msgsFromContents)
	if err != nil {
		return nil, err
	}

	in := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(modelID),
		Messages: messages,
		System:   system,
	}

	if inf := inferenceConfigFromGenai(cfg); inf != nil {
		in.InferenceConfig = inf
	}

	tools, err := toolConfigurationFromGenai(cfg)
	if err != nil {
		return nil, err
	}
	if tools != nil {
		in.ToolConfig = tools
	}

	return in, nil
}

// ConverseStreamInputFromLLMRequest builds a [bedrockruntime.ConverseStreamInput] from an ADK request.
func ConverseStreamInputFromLLMRequest(
	modelID string,
	req *model.LLMRequest,
) (*bedrockruntime.ConverseStreamInput, error) {
	conv, err := ConverseInputFromLLMRequest(modelID, req)
	if err != nil {
		return nil, err
	}
	return &bedrockruntime.ConverseStreamInput{
		ModelId:                           conv.ModelId,
		Messages:                          conv.Messages,
		System:                            conv.System,
		InferenceConfig:                   conv.InferenceConfig,
		ToolConfig:                        conv.ToolConfig,
		AdditionalModelRequestFields:      conv.AdditionalModelRequestFields,
		AdditionalModelResponseFieldPaths: conv.AdditionalModelResponseFieldPaths,
		OutputConfig:                      conv.OutputConfig,
		PerformanceConfig:                 conv.PerformanceConfig,
		PromptVariables:                   conv.PromptVariables,
		RequestMetadata:                   conv.RequestMetadata,
		ServiceTier:                       conv.ServiceTier,
	}, nil
}

func buildSystemBlocks(cfg *genai.GenerateContentConfig) []types.SystemContentBlock {
	if cfg == nil || cfg.SystemInstruction == nil {
		return nil
	}
	var blocks []types.SystemContentBlock
	for _, part := range cfg.SystemInstruction.Parts {
		if part == nil {
			continue
		}
		if part.Text != "" {
			blocks = append(blocks, &types.SystemContentBlockMemberText{Value: part.Text})
		}
	}
	return blocks
}

func splitContents(contents []*genai.Content) ([]types.SystemContentBlock, []*genai.Content) {
	var system []types.SystemContentBlock
	var rest []*genai.Content
	for _, c := range contents {
		if c == nil {
			continue
		}
		if c.Role == "system" {
			system = append(system, contentToSystemBlocks(c)...)
			continue
		}
		rest = append(rest, c)
	}
	return system, rest
}

func contentToSystemBlocks(c *genai.Content) []types.SystemContentBlock {
	var blocks []types.SystemContentBlock
	for _, p := range c.Parts {
		if p == nil {
			continue
		}
		if p.Text != "" {
			blocks = append(blocks, &types.SystemContentBlockMemberText{Value: p.Text})
		}
	}
	return blocks
}

func mapConversationRole(genaiRole string) (types.ConversationRole, error) {
	switch genaiRole {
	case "user":
		return types.ConversationRoleUser, nil
	case "model":
		return types.ConversationRoleAssistant, nil
	default:
		return "", fmt.Errorf("unsupported content role for Bedrock Converse: %q (expected user or model)", genaiRole)
	}
}

func contentsToMessages(contents []*genai.Content) ([]types.Message, error) {
	var out []types.Message
	for _, c := range contents {
		if c == nil {
			continue
		}
		role, err := mapConversationRole(c.Role)
		if err != nil {
			return nil, err
		}
		blocks, err := PartsToContentBlocks(c.Parts, role)
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		out = append(out, types.Message{
			Role:    role,
			Content: blocks,
		})
	}
	return out, nil
}

// PartsToContentBlocks maps genai parts to Bedrock content blocks for the given conversation role.
//
//nolint:gocognit // Role checks and per-part kind switching are clearer in one function than split helpers.
func PartsToContentBlocks(parts []*genai.Part, role types.ConversationRole) ([]types.ContentBlock, error) {
	var blocks []types.ContentBlock
	for _, p := range parts {
		if p == nil {
			continue
		}
		if p.Thought {
			continue
		}
		switch {
		case p.Text != "":
			blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
		case p.InlineData != nil && len(p.InlineData.Data) > 0:
			if role != types.ConversationRoleUser {
				return nil, errors.New("inline media is only supported for user messages on Bedrock Converse")
			}
			imgFmt, err := imageFormatFromMIME(p.InlineData.MIMEType)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, &types.ContentBlockMemberImage{
				Value: types.ImageBlock{
					Format: imgFmt,
					Source: &types.ImageSourceMemberBytes{Value: p.InlineData.Data},
				},
			})
		case p.FunctionCall != nil:
			if role != types.ConversationRoleAssistant {
				return nil, errors.New("functionCall parts must be in a model-role content")
			}
			tu, err := functionCallToToolUse(p.FunctionCall)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, tu)
		case p.FunctionResponse != nil:
			if role != types.ConversationRoleUser {
				return nil, errors.New("functionResponse parts must be in a user-role content")
			}
			tr, err := functionResponseToToolResult(p.FunctionResponse)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, tr)
		default:
			continue
		}
	}
	return blocks, nil
}

func imageFormatFromMIME(mime string) (types.ImageFormat, error) {
	switch strings.ToLower(mime) {
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg, nil
	case "image/png":
		return types.ImageFormatPng, nil
	case "image/gif":
		return types.ImageFormatGif, nil
	case "image/webp":
		return types.ImageFormatWebp, nil
	default:
		return "", fmt.Errorf("unsupported image mime type for Bedrock: %q", mime)
	}
}

func functionCallToToolUse(fc *genai.FunctionCall) (types.ContentBlock, error) {
	if fc == nil {
		return nil, errors.New("nil FunctionCall")
	}
	id := fc.ID
	if id == "" {
		id = "call_" + fc.Name
	}
	var input brdoc.Interface
	if fc.Args == nil {
		input = brdoc.NewLazyDocument(map[string]any{})
	} else {
		input = brdoc.NewLazyDocument(fc.Args)
	}
	return &types.ContentBlockMemberToolUse{
		Value: types.ToolUseBlock{
			ToolUseId: aws.String(id),
			Name:      aws.String(fc.Name),
			Input:     input,
		},
	}, nil
}

func functionResponseToToolResult(fr *genai.FunctionResponse) (types.ContentBlock, error) {
	if fr == nil {
		return nil, errors.New("nil FunctionResponse")
	}
	id := fr.ID
	if id == "" {
		id = "call_" + fr.Name
	}
	var jsonContent []types.ToolResultContentBlock
	if fr.Response != nil {
		jsonContent = append(jsonContent, &types.ToolResultContentBlockMemberJson{
			Value: brdoc.NewLazyDocument(fr.Response),
		})
	}
	return &types.ContentBlockMemberToolResult{
		Value: types.ToolResultBlock{
			ToolUseId: aws.String(id),
			Content:   jsonContent,
		},
	}, nil
}

func inferenceConfigFromGenai(cfg *genai.GenerateContentConfig) *types.InferenceConfiguration {
	if cfg == nil {
		return nil
	}
	var inf types.InferenceConfiguration
	anySet := false
	if cfg.Temperature != nil {
		inf.Temperature = cfg.Temperature
		anySet = true
	}
	if cfg.TopP != nil {
		inf.TopP = cfg.TopP
		anySet = true
	}
	if cfg.MaxOutputTokens > 0 {
		inf.MaxTokens = aws.Int32(cfg.MaxOutputTokens)
		anySet = true
	}
	if len(cfg.StopSequences) > 0 {
		inf.StopSequences = append([]string(nil), cfg.StopSequences...)
		anySet = true
	}
	if !anySet {
		return nil
	}
	return &inf
}
