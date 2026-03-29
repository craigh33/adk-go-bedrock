# bedrock-tool-variants example

This example documents the current Bedrock provider behavior for ADK tool variants. Bedrock tool calling works with function declarations, while non-function ADK tool variants (for example Google Search, Code Execution, Retrieval, URL Context, and MCP servers) are currently rejected early with a clear provider error instead of being sent as invalid Bedrock requests.

## Features

- **Function Declarations** - Supported Bedrock custom function tools
- **Unsupported Variant Detection** - Early, clear errors for non-function ADK tool variants
- **Mixed Tool Validation** - Demonstrates that unsupported variants are detected even when mixed with valid function declarations
- **System Instructions** - Shows that system instructions still work with supported function tools

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID that supports function calling
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)
- Bedrock model should support the specific tool variants you want to use

## Run

```bash
make -C examples/bedrock-tool-variants run
```

## Unsupported Non-Function Tool Variants

The following ADK tool variants are currently **not** mapped to Bedrock Converse by this provider and are rejected locally with a clear error:

```go
// Google Search
Tools: []*genai.Tool{{
    GoogleSearch: &genai.GoogleSearch{},
}},

// Code Execution
Tools: []*genai.Tool{{
    CodeExecution: &genai.ToolCodeExecution{},
}},

// Retrieval
Tools: []*genai.Tool{{
    Retrieval: &genai.Retrieval{},
}},

// URL Context
Tools: []*genai.Tool{{
    URLContext: &genai.URLContext{},
}},

// MCP Servers
Tools: []*genai.Tool{{
    MCPServers: []*genai.MCPServer{{Name: "docs"}},
}},
```

### Combining with Function Declarations

If you mix unsupported variants with function declarations, the provider returns an explicit error instead of sending a partially-invalid request to Bedrock:

```go
Tools: []*genai.Tool{{
    FunctionDeclarations: []*genai.FunctionDeclaration{{
        Name: "save_results",
        // ... parameters ...
    }},
    GoogleSearch:  &genai.GoogleSearch{},
    CodeExecution: &genai.ToolCodeExecution{},
}},
```

### Multiple Variants

Multiple unsupported variants are reported together in one error:

```go
Tools: []*genai.Tool{{
    GoogleSearch:  &genai.GoogleSearch{},
    CodeExecution: &genai.ToolCodeExecution{},
    Retrieval:     &genai.Retrieval{},
}},
```

## Tool Mapping

The Bedrock provider currently supports:

| ADK Tool Type | Bedrock Mapping | Status |
|---|---|---|
| `FunctionDeclarations` | `ToolSpecification` | ✅ Supported |
| Non-function variants (`GoogleSearch`, `CodeExecution`, `Retrieval`, `URLContext`, `MCPServers`, etc.) | none | ❌ Rejected early with a clear error |

## Example Patterns

### Pattern 1: Search and Analyze

```go
GoogleSearch: &genai.GoogleSearch{},
CodeExecution: &genai.ToolCodeExecution{},
```

Request the model to search the web and analyze results with code.

### Pattern 2: Multi-Source Research

```go
GoogleSearch: &genai.GoogleSearch{},
Retrieval: &genai.Retrieval{},
URLContext: &genai.URLContext{},
```

Combine multiple research sources for comprehensive analysis.

### Pattern 3: Function + System Tools

```go
FunctionDeclarations: []*genai.FunctionDeclaration{{
    Name: "log_analysis",
    // ...
}},
CodeExecution: &genai.ToolCodeExecution{},
GoogleSearch: &genai.GoogleSearch{},
```

Use custom functions alongside system tools for complete workflows.

### Pattern 4: Guided Analysis

Combine tools with system instructions:

```go
Config: &genai.GenerateContentConfig{
    SystemInstruction: &genai.Content{
        Parts: []*genai.Part{{
            Text: `You are a research analyst. When using tools:
1. Search for information
2. Analyze with code
3. Cross-reference with knowledge base
4. Provide structured summary`,
        }},
    },
    Tools: []*genai.Tool{{
        GoogleSearch:  &genai.GoogleSearch{},
        CodeExecution: &genai.ToolCodeExecution{},
        Retrieval:     &genai.Retrieval{},
    }},
}
```

## Model Capabilities

For this provider implementation:

- **Supported**: Function declarations
- **Not currently supported**: Non-function ADK tool variants

## Error Handling

If you use an unsupported non-function ADK tool variant:

1. The provider returns an error before sending the request to Bedrock
2. The error message lists the unsupported variant names
3. Replace the variant with `FunctionDeclarations` where possible

## Limitations

- Non-function ADK tool variants are not currently mapped to Bedrock Converse
- Mixed tools are only supported when all entries are function declarations

## Advanced Usage

### Streaming with Tools

```go
for resp, err := range llm.GenerateContent(ctx, req, true) { // true = streaming
    if err != nil {
        // Handle stream error (e.g., log, break, etc.)
        break
    }
    if resp == nil {
        continue
    }
    // Handle streamed tool responses
}
```

### System Instructions with Tools

Guide the model's tool selection and usage with detailed system instructions.

### Multi-Turn Conversations

Tools persist across conversation turns—use them for:

- Progressive research (search → analyze → refine)
- Knowledge gathering (retrieve → analyze → report)
- Complex workflows (function → search → code → save)

## More Resources

- [Bedrock Tool Use Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html)
- [ADK Tool Documentation](https://pkg.go.dev/google.golang.org/genai#Tool)
- [Tool Variant Reference](https://pkg.go.dev/google.golang.org/genai)
