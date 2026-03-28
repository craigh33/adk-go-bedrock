package mappers

import (
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"
)

func toolConfigurationFromGenai(cfg *genai.GenerateContentConfig) (*types.ToolConfiguration, error) {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil, nil //nolint:nilnil // optional ToolConfiguration: nil means no tools
	}
	var specs []types.Tool
	for _, t := range cfg.Tools {
		if t == nil {
			continue
		}
		for _, fd := range t.FunctionDeclarations {
			if fd == nil || fd.Name == "" {
				continue
			}
			inputSchema, err := functionParametersToToolInputSchema(fd)
			if err != nil {
				return nil, fmt.Errorf("tool %q: %w", fd.Name, err)
			}
			specs = append(specs, &types.ToolMemberToolSpec{
				Value: types.ToolSpecification{
					Name:        aws.String(fd.Name),
					Description: aws.String(fd.Description),
					InputSchema: inputSchema,
				},
			})
		}
		// Map supported non-function tool variants to system tools.
		toolVariantSpecs := toolVariantsToSystemTools(t)
		specs = append(specs, toolVariantSpecs...)
	}
	if len(specs) == 0 {
		return nil, nil //nolint:nilnil // optional ToolConfiguration: nil means no tools
	}
	return &types.ToolConfiguration{Tools: specs}, nil
}

func toolVariantsToSystemTools(t *genai.Tool) []types.Tool {
	if t == nil {
		return nil
	}

	// Define tool variants and their mapped system tool names.
	variants := []struct {
		field any
		name  string
	}{
		{t.Retrieval, "retrieval"},
		{t.ComputerUse, "computer_use"},
		{t.FileSearch, "file_search"},
		{t.GoogleSearch, "google_search"},
		{t.GoogleMaps, "google_maps"},
		{t.CodeExecution, "code_execution"},
		{t.EnterpriseWebSearch, "enterprise_web_search"},
		{t.GoogleSearchRetrieval, "google_search_retrieval"},
		{t.ParallelAISearch, "parallel_ai_search"},
		{t.URLContext, "url_context"},
	}

	var specs []types.Tool

	// Map each variant to a system tool.
	for _, v := range variants {
		if v.field != nil {
			specs = append(specs, &types.ToolMemberSystemTool{Value: types.SystemTool{Name: aws.String(v.name)}})
		}
	}

	// Handle MCP servers: each server becomes a system tool with its name.
	if len(t.MCPServers) > 0 {
		for _, mcp := range t.MCPServers {
			if mcp != nil && mcp.Name != "" {
				specs = append(specs, &types.ToolMemberSystemTool{Value: types.SystemTool{Name: aws.String(mcp.Name)}})
			}
		}
	}

	return specs
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
	return &types.ToolInputSchemaMemberJson{Value: brdoc.NewLazyDocument(m)}, nil
}
