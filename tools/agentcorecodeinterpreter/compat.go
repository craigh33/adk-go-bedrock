// Package agentcorecodeinterpreter preserves the original Code Interpreter import path.
// Deprecated: use github.com/craigh33/adk-go-bedrock/tools/agentcore/codeinterpreter.
package agentcorecodeinterpreter

import (
	"google.golang.org/adk/v2/tool"

	"github.com/craigh33/adk-go-bedrock/tools/agentcore/codeinterpreter"
)

// AgentCoreAPI is the AgentCore API used by the Code Interpreter tool.
//
// Deprecated: use codeinterpreter.AgentCoreAPI.
type AgentCoreAPI = codeinterpreter.AgentCoreAPI

// Config configures the Code Interpreter tool.
//
// Deprecated: use codeinterpreter.Config.
type Config = codeinterpreter.Config

// New creates an ADK-compatible Code Interpreter tool.
//
// Deprecated: use codeinterpreter.New.
func New(cfg Config) (tool.Tool, error) {
	return codeinterpreter.New(cfg)
}
