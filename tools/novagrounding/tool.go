// Package novagrounding enables Amazon Nova Web Grounding for the Bedrock Converse
// provider only: the sentinel function declaration is mapped to Bedrock's
// SystemTool name "nova_grounding". This package is AWS/Bedrock-specific and is
// not a generic cross-provider web-search tool.
// See https://docs.aws.amazon.com/nova/latest/userguide/grounding.html
package novagrounding

import "google.golang.org/genai"

// SentinelFunctionDeclarationName is the reserved function name recognized by the
// Bedrock mapper. It is never sent as a custom ToolSpecification (AWS forbids a
// toolSpec named nova_grounding); it becomes types.SystemTool{Name: "nova_grounding"}.
const SentinelFunctionDeclarationName = "__adk_bedrock_nova_grounding"

// SystemToolName is the Bedrock Converse API SystemTool name value for Nova Web
// Grounding (types.SystemTool{Name: "nova_grounding"}). This is distinct from the
// IAM resource identifier often used with bedrock:InvokeTool
// ("amazon.nova_grounding").
const SystemToolName = "nova_grounding"

// Tool returns a genai.Tool placeholder that triggers Nova Web Grounding when used
// with this project's Bedrock model implementation. Combine with other tools by
// appending additional entries to GenerateContentConfig.Tools.
func Tool() *genai.Tool {
	return &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        SentinelFunctionDeclarationName,
				Description: "Enables Amazon Nova Web Grounding for Bedrock Converse only (SystemTool name: nova_grounding; IAM resource identifier may be amazon.nova_grounding).",
				Parameters: &genai.Schema{
					Type:       "object",
					Properties: map[string]*genai.Schema{},
				},
			},
		},
	}
}
