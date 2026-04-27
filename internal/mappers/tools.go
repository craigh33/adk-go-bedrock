package mappers

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

const unsupportedToolVariantCount = 11

func toolConfigurationFromGenai(cfg *genai.GenerateContentConfig) (*types.ToolConfiguration, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil, nil //nolint:nilnil // optional ToolConfiguration: nil means no tools
	}
	var specs []types.Tool
	var novaGroundingAdded bool
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		var err error
		specs, novaGroundingAdded, err = appendFunctionDeclarationSpecs(specs, t, novaGroundingAdded)
		if err != nil {
			return nil, err
		}
		if unsupported := unsupportedToolVariantsFromGenai(t); len(unsupported) > 0 {
			return nil, fmt.Errorf(
				"bedrock Converse does not support these genai tool variants: %s; use FunctionDeclarations instead",
				strings.Join(unsupported, ", "),
			)
		}
	}
	if len(specs) == 0 {
		return nil, nil //nolint:nilnil // optional ToolConfiguration: nil means no tools
	}
	return &types.ToolConfiguration{Tools: specs}, nil
}

func appendFunctionDeclarationSpecs(
	specs []types.Tool,
	t *genai.Tool,
	novaGroundingAdded bool,
) ([]types.Tool, bool, error) {
	for _, fd := range t.FunctionDeclarations {
		if fd == nil || fd.Name == "" {
			continue
		}
		if fd.Name == novagrounding.SentinelFunctionDeclarationName {
			if !novaGroundingAdded {
				specs = append(specs, &types.ToolMemberSystemTool{
					Value: types.SystemTool{Name: aws.String(novagrounding.SystemToolName)},
				})
				novaGroundingAdded = true
			}
			continue
		}
		inputSchema, err := functionParametersToToolInputSchema(fd)
		if err != nil {
			return specs, novaGroundingAdded, fmt.Errorf("tool %q: %w", fd.Name, err)
		}
		specs = append(specs, &types.ToolMemberToolSpec{
			Value: types.ToolSpecification{
				Name:        aws.String(fd.Name),
				Description: aws.String(fd.Description),
				InputSchema: inputSchema,
			},
		})
	}
	return specs, novaGroundingAdded, nil
}

func unsupportedToolVariantsFromGenai(t *genai.Tool) []string {
	if t == nil {
		return nil
	}
	unsupported := make([]string, 0, unsupportedToolVariantCount)
	appendUnsupported := func(enabled bool, name string) {
		if !enabled || slices.Contains(unsupported, name) {
			return
		}
		unsupported = append(unsupported, name)
	}

	appendUnsupported(t.Retrieval != nil, "Retrieval")
	appendUnsupported(t.ComputerUse != nil, "ComputerUse")
	appendUnsupported(t.FileSearch != nil, "FileSearch")
	appendUnsupported(t.GoogleSearch != nil, "GoogleSearch")
	appendUnsupported(t.GoogleMaps != nil, "GoogleMaps")
	appendUnsupported(t.CodeExecution != nil, "CodeExecution")
	appendUnsupported(t.EnterpriseWebSearch != nil, "EnterpriseWebSearch")
	appendUnsupported(t.GoogleSearchRetrieval != nil, "GoogleSearchRetrieval")
	appendUnsupported(t.ParallelAISearch != nil, "ParallelAISearch")
	appendUnsupported(t.URLContext != nil, "URLContext")
	appendUnsupported(len(t.MCPServers) > 0, "MCPServers")

	return unsupported
}

func functionParametersToToolInputSchema(fd *genai.FunctionDeclaration) (types.ToolInputSchema, error) {
	if fd.ParametersJsonSchema != nil {
		return &types.ToolInputSchemaMemberJson{Value: brdoc.NewLazyDocument(fd.ParametersJsonSchema)}, nil
	}
	if fd.Parameters == nil {
		return &types.ToolInputSchemaMemberJson{
			Value: brdoc.NewLazyDocument(map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}),
		}, nil
	}
	b, err := json.Marshal(fd.Parameters)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	normalizeSchemaTypes(m)
	return &types.ToolInputSchemaMemberJson{Value: brdoc.NewLazyDocument(m)}, nil
}

// normalizeSchemaTypes recursively lowercases every "type" field value in a JSON
// Schema map. genai.Schema marshals Gemini-style uppercase type names (e.g.
// "STRING", "OBJECT", "ARRAY") but Bedrock Converse requires lowercase JSON
// Schema type names (e.g. "string", "object", "array").
func normalizeSchemaTypes(v any) {
	switch m := v.(type) {
	case map[string]any:
		for k, val := range m {
			if k == "type" {
				if s, ok := val.(string); ok {
					m[k] = strings.ToLower(s)
				}
			} else {
				normalizeSchemaTypes(val)
			}
		}
	case []any:
		for _, item := range m {
			normalizeSchemaTypes(item)
		}
	}
}
