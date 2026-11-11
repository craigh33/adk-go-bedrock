package agentcorecodeinterpreter

import (
	"strings"
	"testing"

	"github.com/craigh33/adk-go-bedrock/tools/agentcore/codeinterpreter"
)

func TestCompatibility(t *testing.T) {
	var cfg codeinterpreter.Config = Config{}
	_, err := New(cfg)
	if err == nil || !strings.Contains(err.Error(), "API is required") {
		t.Fatalf("New() error = %v, want missing API", err)
	}
}
