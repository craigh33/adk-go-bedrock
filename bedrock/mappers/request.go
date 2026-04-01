// Package mappers converts between ADK/genai types and Amazon Bedrock Converse API types.
package mappers

import (
	"errors"
	"fmt"
	"path"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	genaiRoleUser   = "user"
	genaiRoleModel  = "model"
	genaiRoleSystem = "system"
)

// MaybeAppendUserContent mirrors the Gemini provider behavior so empty histories or
// assistant-terminated turns still receive a valid user message.
func MaybeAppendUserContent(contents []*genai.Content) []*genai.Content {
	if len(contents) == 0 {
		return append(contents, genai.NewContentFromText(
			"Handle the requests as specified in the System Instruction.", genaiRoleUser))
	}
	if last := contents[len(contents)-1]; last != nil && last.Role != genaiRoleUser {
		return append(contents, genai.NewContentFromText(
			"Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.",
			genaiRoleUser,
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

	guardrailCfg, err := guardrailConfigFromGenai(cfg)
	if err != nil {
		return nil, err
	}
	if guardrailCfg != nil {
		in.GuardrailConfig = guardrailCfg
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
		GuardrailConfig:                   guardrailStreamConfigFromConverse(conv.GuardrailConfig),
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
		if c.Role == genaiRoleSystem {
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
	case genaiRoleUser:
		return types.ConversationRoleUser, nil
	case genaiRoleModel:
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
		switch {
		case p.Thought:
			rb, err := thoughtPartToReasoningContentBlock(p, role)
			if err != nil {
				return nil, err
			}
			if rb != nil {
				blocks = append(blocks, rb)
			}
		case p.Text != "":
			blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
		case p.InlineData != nil && len(p.InlineData.Data) > 0:
			block, err := inlineDataToContentBlock(p, role)
			if err != nil {
				return nil, err
			}
			if block != nil {
				blocks = append(blocks, block)
			}
		case p.FileData != nil && p.FileData.FileURI != "":
			block, err := fileDataToContentBlock(p, role)
			if err != nil {
				return nil, err
			}
			if block != nil {
				blocks = append(blocks, block)
			}
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

func thoughtPartToReasoningContentBlock(p *genai.Part, role types.ConversationRole) (types.ContentBlock, error) {
	if p == nil || !p.Thought {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	if role != types.ConversationRoleAssistant {
		return nil, errors.New("thought parts must be in a model-role content")
	}
	if p.Text == "" {
		return nil, nil //nolint:nilnil // No reasoning text to send back.
	}
	val := types.ReasoningTextBlock{Text: aws.String(p.Text)}
	if len(p.ThoughtSignature) > 0 {
		sig := string(p.ThoughtSignature)
		val.Signature = aws.String(sig)
	}
	return &types.ContentBlockMemberReasoningContent{
		Value: &types.ReasoningContentBlockMemberReasoningText{Value: val},
	}, nil
}

func inlineDataToContentBlock(p *genai.Part, role types.ConversationRole) (types.ContentBlock, error) {
	if p == nil || p.InlineData == nil || len(p.InlineData.Data) == 0 {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	if role != types.ConversationRoleUser {
		return nil, errors.New("inline media is only supported for user messages on Bedrock Converse")
	}
	switch classifyMIME(p.InlineData.MIMEType) { //nolint:exhaustive // mediaKindUnknown handled by inference logic
	case mediaKindImage:
		imgFmt, err := imageFormatFromMIME(p.InlineData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberImage{Value: types.ImageBlock{
			Format: imgFmt,
			Source: &types.ImageSourceMemberBytes{Value: p.InlineData.Data},
		}}, nil
	case mediaKindAudio:
		audioFmt, err := audioFormatFromMIME(p.InlineData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberAudio{Value: types.AudioBlock{
			Format: audioFmt,
			Source: &types.AudioSourceMemberBytes{Value: p.InlineData.Data},
		}}, nil
	case mediaKindVideo:
		videoFmt, err := videoFormatFromMIME(p.InlineData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberVideo{Value: types.VideoBlock{
			Format: videoFmt,
			Source: &types.VideoSourceMemberBytes{Value: p.InlineData.Data},
		}}, nil
	case mediaKindDocument:
		docFmt, err := documentFormatFromMIME(p.InlineData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberDocument{Value: types.DocumentBlock{
			Name:   aws.String(documentName(p.InlineData.DisplayName, "", p.InlineData.MIMEType)),
			Format: docFmt,
			Source: &types.DocumentSourceMemberBytes{Value: p.InlineData.Data},
		}}, nil
	default:
		return nil, fmt.Errorf("unsupported inline mime type for Bedrock: %q", p.InlineData.MIMEType)
	}
}

func fileDataToContentBlock(p *genai.Part, role types.ConversationRole) (types.ContentBlock, error) {
	if p == nil || p.FileData == nil || p.FileData.FileURI == "" {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	if role != types.ConversationRoleUser {
		return nil, errors.New("file media is only supported for user messages on Bedrock Converse")
	}
	s3, err := s3LocationFromURI(p.FileData.FileURI)
	if err != nil {
		return nil, err
	}
	switch classifyMIME(p.FileData.MIMEType) { //nolint:exhaustive // mediaKindUnknown handled by inference logic
	case mediaKindImage:
		imgFmt, err := imageFormatFromMIME(p.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberImage{Value: types.ImageBlock{
			Format: imgFmt,
			Source: &types.ImageSourceMemberS3Location{Value: s3},
		}}, nil
	case mediaKindAudio:
		audioFmt, err := audioFormatFromMIME(p.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberAudio{Value: types.AudioBlock{
			Format: audioFmt,
			Source: &types.AudioSourceMemberS3Location{Value: s3},
		}}, nil
	case mediaKindVideo:
		videoFmt, err := videoFormatFromMIME(p.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberVideo{Value: types.VideoBlock{
			Format: videoFmt,
			Source: &types.VideoSourceMemberS3Location{Value: s3},
		}}, nil
	case mediaKindDocument:
		docFmt, err := documentFormatFromMIME(p.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberDocument{Value: types.DocumentBlock{
			Name:   aws.String(documentName(p.FileData.DisplayName, p.FileData.FileURI, p.FileData.MIMEType)),
			Format: docFmt,
			Source: &types.DocumentSourceMemberS3Location{Value: s3},
		}}, nil
	default:
		return nil, fmt.Errorf("unsupported file mime type for Bedrock: %q", p.FileData.MIMEType)
	}
}

type mediaKind string

const (
	mediaKindUnknown  mediaKind = "unknown"
	mediaKindImage    mediaKind = "image"
	mediaKindAudio    mediaKind = "audio"
	mediaKindVideo    mediaKind = "video"
	mediaKindDocument mediaKind = "document"

	mimeApplicationPDF  = "application/pdf"
	mimeApplicationDOC  = "application/msword"
	mimeApplicationDOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
	mimeApplicationXLS  = "application/vnd.ms-excel"
	mimeApplicationXLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
	mimeImagePNG        = "image/png"
)

func classifyMIME(mime string) mediaKind {
	mime = strings.ToLower(strings.TrimSpace(mime))
	switch {
	case strings.HasPrefix(mime, "image/"):
		return mediaKindImage
	case strings.HasPrefix(mime, "audio/"):
		return mediaKindAudio
	case strings.HasPrefix(mime, "video/"):
		return mediaKindVideo
	case strings.HasPrefix(mime, "text/"):
		return mediaKindDocument
	case strings.HasPrefix(mime, "application/"):
		switch mime {
		case mimeApplicationPDF,
			mimeApplicationDOC,
			mimeApplicationDOCX,
			mimeApplicationXLS,
			mimeApplicationXLSX:
			return mediaKindDocument
		}
	}
	return mediaKindUnknown
}

func imageFormatFromMIME(mime string) (types.ImageFormat, error) {
	switch strings.ToLower(mime) {
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg, nil
	case mimeImagePNG:
		return types.ImageFormatPng, nil
	case "image/gif":
		return types.ImageFormatGif, nil
	case "image/webp":
		return types.ImageFormatWebp, nil
	default:
		return "", fmt.Errorf("unsupported image mime type for Bedrock: %q", mime)
	}
}

func audioFormatFromMIME(mime string) (types.AudioFormat, error) {
	switch strings.ToLower(mime) {
	case "audio/mpeg", "audio/mp3":
		return types.AudioFormatMp3, nil
	case "audio/opus":
		return types.AudioFormatOpus, nil
	case "audio/wav", "audio/x-wav", "audio/wave":
		return types.AudioFormatWav, nil
	case "audio/aac":
		return types.AudioFormatAac, nil
	case "audio/flac", "audio/x-flac":
		return types.AudioFormatFlac, nil
	case "audio/mp4":
		return types.AudioFormatMp4, nil
	case "audio/ogg":
		return types.AudioFormatOgg, nil
	case "audio/x-matroska":
		return types.AudioFormatMka, nil
	case "audio/x-aac":
		return types.AudioFormatXAac, nil
	case "audio/m4a":
		return types.AudioFormatM4a, nil
	case "audio/mpga":
		return types.AudioFormatMpga, nil
	case "audio/pcm", "audio/l16":
		return types.AudioFormatPcm, nil
	case "audio/webm":
		return types.AudioFormatWebm, nil
	default:
		return "", fmt.Errorf("unsupported audio mime type for Bedrock: %q", mime)
	}
}

func videoFormatFromMIME(mime string) (types.VideoFormat, error) {
	switch strings.ToLower(mime) {
	case "video/x-matroska":
		return types.VideoFormatMkv, nil
	case "video/quicktime":
		return types.VideoFormatMov, nil
	case "video/mp4":
		return types.VideoFormatMp4, nil
	case "video/webm":
		return types.VideoFormatWebm, nil
	case "video/x-flv":
		return types.VideoFormatFlv, nil
	case "video/mpeg":
		return types.VideoFormatMpeg, nil
	case "video/mpg":
		return types.VideoFormatMpg, nil
	case "video/x-ms-wmv":
		return types.VideoFormatWmv, nil
	case "video/3gpp":
		return types.VideoFormatThreeGp, nil
	default:
		return "", fmt.Errorf("unsupported video mime type for Bedrock: %q", mime)
	}
}

func documentFormatFromMIME(mime string) (types.DocumentFormat, error) {
	switch strings.ToLower(mime) {
	case mimeApplicationPDF:
		return types.DocumentFormatPdf, nil
	case "text/csv":
		return types.DocumentFormatCsv, nil
	case mimeApplicationDOC:
		return types.DocumentFormatDoc, nil
	case mimeApplicationDOCX:
		return types.DocumentFormatDocx, nil
	case mimeApplicationXLS:
		return types.DocumentFormatXls, nil
	case mimeApplicationXLSX:
		return types.DocumentFormatXlsx, nil
	case "text/html":
		return types.DocumentFormatHtml, nil
	case "text/plain":
		return types.DocumentFormatTxt, nil
	case "text/markdown", "text/x-markdown":
		return types.DocumentFormatMd, nil
	default:
		return "", fmt.Errorf("unsupported document mime type for Bedrock: %q", mime)
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
	for _, part := range fr.Parts {
		block, err := functionResponsePartToToolResultContentBlock(part)
		if err != nil {
			return nil, err
		}
		if block != nil {
			jsonContent = append(jsonContent, block)
		}
	}
	return &types.ContentBlockMemberToolResult{
		Value: types.ToolResultBlock{
			ToolUseId: aws.String(id),
			Content:   jsonContent,
		},
	}, nil
}

//nolint:gocognit // multimodal union mapping requires several media-type branches.
func functionResponsePartToToolResultContentBlock(
	part *genai.FunctionResponsePart,
) (types.ToolResultContentBlock, error) {
	if part == nil {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	if part.InlineData != nil && len(part.InlineData.Data) > 0 {
		switch classifyMIME(part.InlineData.MIMEType) { //nolint:exhaustive // mediaKindUnknown handled by inference
		case mediaKindImage:
			imgFmt, err := imageFormatFromMIME(part.InlineData.MIMEType)
			if err != nil {
				return nil, err
			}
			return &types.ToolResultContentBlockMemberImage{Value: types.ImageBlock{
				Format: imgFmt,
				Source: &types.ImageSourceMemberBytes{Value: part.InlineData.Data},
			}}, nil
		case mediaKindVideo:
			videoFmt, err := videoFormatFromMIME(part.InlineData.MIMEType)
			if err != nil {
				return nil, err
			}
			return &types.ToolResultContentBlockMemberVideo{Value: types.VideoBlock{
				Format: videoFmt,
				Source: &types.VideoSourceMemberBytes{Value: part.InlineData.Data},
			}}, nil
		case mediaKindDocument:
			docFmt, err := documentFormatFromMIME(part.InlineData.MIMEType)
			if err != nil {
				return nil, err
			}
			return &types.ToolResultContentBlockMemberDocument{Value: types.DocumentBlock{
				Name:   aws.String(documentName(part.InlineData.DisplayName, "", part.InlineData.MIMEType)),
				Format: docFmt,
				Source: &types.DocumentSourceMemberBytes{Value: part.InlineData.Data},
			}}, nil
		case mediaKindAudio:
			return nil, fmt.Errorf(
				"audio function response parts are not supported by Bedrock tool results: %q",
				part.InlineData.MIMEType,
			)
		default:
			return nil, fmt.Errorf(
				"unsupported function response inline mime type for Bedrock: %q",
				part.InlineData.MIMEType,
			)
		}
	}
	if part.FileData == nil || part.FileData.FileURI == "" {
		return nil, nil //nolint:nilnil // Optional conversion.
	}
	s3, err := s3LocationFromURI(part.FileData.FileURI)
	if err != nil {
		return nil, err
	}
	switch classifyMIME(part.FileData.MIMEType) { //nolint:exhaustive // mediaKindUnknown handled by inference
	case mediaKindImage:
		imgFmt, err := imageFormatFromMIME(part.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ToolResultContentBlockMemberImage{Value: types.ImageBlock{
			Format: imgFmt,
			Source: &types.ImageSourceMemberS3Location{Value: s3},
		}}, nil
	case mediaKindVideo:
		videoFmt, err := videoFormatFromMIME(part.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ToolResultContentBlockMemberVideo{Value: types.VideoBlock{
			Format: videoFmt,
			Source: &types.VideoSourceMemberS3Location{Value: s3},
		}}, nil
	case mediaKindDocument:
		docFmt, err := documentFormatFromMIME(part.FileData.MIMEType)
		if err != nil {
			return nil, err
		}
		return &types.ToolResultContentBlockMemberDocument{Value: types.DocumentBlock{
			Name:   aws.String(documentName(part.FileData.DisplayName, part.FileData.FileURI, part.FileData.MIMEType)),
			Format: docFmt,
			Source: &types.DocumentSourceMemberS3Location{Value: s3},
		}}, nil
	case mediaKindAudio:
		return nil, fmt.Errorf(
			"audio function response file parts are not supported by Bedrock tool results: %q",
			part.FileData.MIMEType,
		)
	default:
		return nil, fmt.Errorf("unsupported function response file mime type for Bedrock: %q", part.FileData.MIMEType)
	}
}

func s3LocationFromURI(uri string) (types.S3Location, error) {
	if !strings.HasPrefix(strings.ToLower(uri), "s3://") {
		return types.S3Location{}, fmt.Errorf("bedrock Converse file URIs must use s3://, got %q", uri)
	}
	return types.S3Location{Uri: aws.String(uri)}, nil
}

func documentName(displayName, fileURI, mime string) string {
	if name := strings.TrimSpace(displayName); name != "" {
		return name
	}
	if fileURI != "" {
		if base := strings.TrimSpace(path.Base(fileURI)); base != "" && base != "." && base != "/" {
			return base
		}
	}
	switch classifyMIME(mime) { //nolint:exhaustive // mediaKindUnknown returns empty string
	case mediaKindDocument:
		return "document"
	case mediaKindImage:
		return "image"
	case mediaKindVideo:
		return "video"
	case mediaKindAudio:
		return "audio"
	default:
		return "attachment"
	}
}

//nolint:unparam // signature mirrors other config mappers and leaves room for future Bedrock guardrail mapping.
func guardrailConfigFromGenai(
	cfg *genai.GenerateContentConfig,
) (*types.GuardrailConfiguration, error) {
	if cfg == nil {
		return nil, nil //nolint:nilnil // Optional config.
	}
	if len(cfg.SafetySettings) > 0 || cfg.ModelArmorConfig != nil {
		return nil, errors.New(
			"genai safety settings and model armor config cannot be mapped to Bedrock Converse automatically: Bedrock requires a preconfigured guardrail identifier and version",
		)
	}
	return nil, nil //nolint:nilnil // No Bedrock-native guardrail config available from generic ADK config.
}

func guardrailStreamConfigFromConverse(cfg *types.GuardrailConfiguration) *types.GuardrailStreamConfiguration {
	if cfg == nil {
		return nil
	}
	return &types.GuardrailStreamConfiguration{
		GuardrailIdentifier: cfg.GuardrailIdentifier,
		GuardrailVersion:    cfg.GuardrailVersion,
		Trace:               cfg.Trace,
	}
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
