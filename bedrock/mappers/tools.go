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
		if len(t.FunctionDeclarations) == 0 {
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
	}
	if len(specs) == 0 {
		return nil, nil //nolint:nilnil // optional ToolConfiguration: nil means no tools
	}
	return &types.ToolConfiguration{Tools: specs}, nil
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
