// Package novagrounding enables Amazon Nova Web Grounding via the Bedrock Converse
// provider: the sentinel function declaration is mapped to Bedrock's system tool
// "nova_grounding". See https://docs.aws.amazon.com/nova/latest/userguide/grounding.html
package novagrounding

import "google.golang.org/genai"

// SentinelFunctionDeclarationName is the reserved function name recognized by the
// Bedrock mapper. It is never sent as a custom ToolSpecification (AWS forbids a
// toolSpec named nova_grounding); it becomes types.SystemTool{Name: "nova_grounding"}.
const SentinelFunctionDeclarationName = "__adk_bedrock_nova_grounding"

// SystemToolName is the Bedrock system tool id for Nova Web Grounding.
const SystemToolName = "nova_grounding"

// Tool returns a genai.Tool placeholder that triggers Nova Web Grounding when used
// with this project's Bedrock model implementation. Combine with other tools by
// appending additional entries to GenerateContentConfig.Tools.
func Tool() *genai.Tool {
	return &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        SentinelFunctionDeclarationName,
				Description: "Enables Amazon Nova Web Grounding (real-time web search via Bedrock system tool nova_grounding).",
				Parameters: &genai.Schema{
					Type:       "object",
					Properties: map[string]*genai.Schema{},
				},
			},
		},
	}
}
