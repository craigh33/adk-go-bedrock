package novagrounding_test

import (
	"testing"

	"github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

func TestTool_SentinelName(t *testing.T) {
	t.Parallel()
	tool := novagrounding.Tool()
	if len(tool.FunctionDeclarations) != 1 {
		t.Fatalf("expected one declaration, got %d", len(tool.FunctionDeclarations))
	}
	if tool.FunctionDeclarations[0].Name != novagrounding.SentinelFunctionDeclarationName {
		t.Fatalf("unexpected declaration name %q", tool.FunctionDeclarations[0].Name)
	}
}
