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
		if unsupported := UnsupportedToolVariantsFromGenai(t); len(unsupported) > 0 {
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

// UnsupportedToolVariantsFromGenai returns the names of [genai.Tool] variants
// set on t that cannot be mapped to a Bedrock backend (Converse or Live). Used
// to fail fast with a clear error rather than silently dropping tools.
func UnsupportedToolVariantsFromGenai(t *genai.Tool) []string {
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
	schema, err := FunctionDeclarationSchema(fd)
	if err != nil {
		return nil, err
	}
	return &types.ToolInputSchemaMemberJson{Value: brdoc.NewLazyDocument(schema)}, nil
}

// FunctionDeclarationSchema extracts a canonical JSON-Schema map from a
// [genai.FunctionDeclaration], honoring the ParametersJsonSchema → Parameters
// → empty-object precedence and lowercasing Gemini-style uppercase type names.
//
// Exported for reuse by the Bedrock Live (Nova Sonic) backend, which serializes
// the same map as a stringified JSON inside its toolSpec.inputSchema.json field.
func FunctionDeclarationSchema(fd *genai.FunctionDeclaration) (map[string]any, error) {
	if fd == nil {
		return emptyObjectSchema(), nil
	}
	if fd.ParametersJsonSchema != nil {
		schema, err := NormalizeSchema(fd.ParametersJsonSchema)
		if err != nil {
			return nil, err
		}
		return schema, nil
	}
	if fd.Parameters == nil {
		return emptyObjectSchema(), nil
	}
	b, err := json.Marshal(fd.Parameters)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	NormalizeSchemaTypes(m)
	return m, nil
}

// jsonSchemaObjectType is the canonical JSON Schema "object" type literal.
// Repeated often enough across tools mapping + tests to warrant a constant.
const jsonSchemaObjectType = "object"

func emptyObjectSchema() map[string]any {
	return map[string]any{
		"type":       jsonSchemaObjectType,
		"properties": map[string]any{},
	}
}

// NormalizeSchemaTypes recursively lowercases every "type" field value in a JSON
// Schema map. genai.Schema marshals Gemini-style uppercase type names (e.g.
// "STRING", "OBJECT", "ARRAY") but JSON Schema (and Bedrock Converse / Nova
// Sonic) require lowercase type names (e.g. "string", "object", "array").
func NormalizeSchemaTypes(v any) {
	switch m := v.(type) {
	case map[string]any:
		for k, val := range m {
			if k == "type" {
				if s, ok := val.(string); ok {
					m[k] = strings.ToLower(s)
				}
			} else {
				NormalizeSchemaTypes(val)
			}
		}
	case []any:
		for _, item := range m {
			NormalizeSchemaTypes(item)
		}
	}
}

// NormalizeSchema turns an arbitrary JSON value into a JSON object map. ADK's
// FunctionTool supplies ParametersJsonSchema as arbitrary JSON (often a struct
// or other non-map value after decoding); consumers that need a map[string]any
// (Bedrock's document layer, or a stringified JSON for Nova Sonic) round-trip
// it through JSON when the value is not already map[string]any.
func NormalizeSchema(schema any) (map[string]any, error) {
	switch s := schema.(type) {
	case map[string]any:
		return s, nil
	default:
		bytes, err := json.Marshal(s)
		if err != nil {
			return nil, fmt.Errorf("bedrock: failed to marshal schema: %w", err)
		}
		var m map[string]any
		if err := json.Unmarshal(bytes, &m); err != nil {
			return nil, fmt.Errorf("bedrock: failed to unmarshal schema: %w", err)
		}
		return m, nil
	}
}

// RawFunctionArgsJSONKey is the map key used by [FunctionArgsFromRawJSON] when
// the raw blob is not parseable as a JSON object. The original bytes are
// preserved under this key so callers can recover them.
const RawFunctionArgsJSONKey = "rawArgsJson"

// FunctionArgsFromRawJSON parses a stringified-JSON tool-arguments blob into a
// map[string]any. Empty input returns an empty map. Unparseable input is
// returned as a single-entry map keyed by [RawFunctionArgsJSONKey].
//
// Both Bedrock Converse (ConverseStream tool input deltas) and Nova Sonic
// (toolUse.content) deliver tool arguments as stringified JSON, so both
// backends share this parser.
func FunctionArgsFromRawJSON(raw string) map[string]any {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]any{}
	}
	var parsed map[string]any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
		return parsed
	}
	return map[string]any{RawFunctionArgsJSONKey: raw}
}
