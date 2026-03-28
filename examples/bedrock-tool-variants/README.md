# bedrock-tool-variants example

This example demonstrates how to use non-function tool variants with the Bedrock Converse provider, including Google Search, Code Execution, Retrieval, MCP Servers, and combinations with traditional function declarations.

## Features

- **Google Search** - Enable web search as a tool
- **Code Execution** - Allow the model to write and execute code
- **Retrieval** - Access knowledge bases and documentation
- **URL Context** - Retrieve and analyze content from URLs
- **MCP Servers** - Connect to Model Context Protocol servers
- **Function Declarations** - Traditional custom function tools
- **Mixed Tools** - Combine function declarations with tool variants
- **Multiple Variants** - Use multiple tool types in a single request
- **System Instructions** - Guide tool usage with detailed instructions

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID that supports these tool variants
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)
- Bedrock model should support the specific tool variants you want to use

## Run

```bash
make -C examples/bedrock-tool-variants run
```

## Supported Tool Variants

### Single Tool Variants

Each of these can be used as a standalone tool:

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

Mix tool variants with custom function declarations:

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

Combine multiple tool variants in a single request:

```go
Tools: []*genai.Tool{{
    GoogleSearch:  &genai.GoogleSearch{},
    CodeExecution: &genai.ToolCodeExecution{},
    Retrieval:     &genai.Retrieval{},
}},
```

## Tool Mapping

The provider maps genai tool variants to Bedrock system tools:

| genai Variant | Bedrock System Tool | Purpose |
|---|---|---|
| `GoogleSearch` | `google_search` | Web search |
| `CodeExecution` | `code_execution` | Code running |
| `Retrieval` | `retrieval` | Knowledge base access |
| `URLContext` | `url_context` | URL content analysis |
| `ComputerUse` | `computer_use` | Computer interaction |
| `FileSearch` | `file_search` | File searching |
| `GoogleMaps` | `google_maps` | Geospatial queries |
| `EnterpriseWebSearch` | `enterprise_web_search` | Enterprise search |
| `GoogleSearchRetrieval` | `google_search_retrieval` | Enhanced retrieval |
| `ParallelAISearch` | `parallel_ai_search` | Parallel searching |
| `MCPServers` | `system_tool(name)` | Protocol servers |

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

Not all Bedrock models support all tool variants. Check:

- Model documentation for supported tools
- Bedrock console for availability in your region
- Error messages for specific unsupported variants

Common support levels:

- **Always**: Function declarations
- **Usually**: Google Search, Code Execution
- **Model-dependent**: Retrieval, URL Context, MCP
- **Region-dependent**: Enterprise search features

## Error Handling

If you use an unsupported tool variant:

1. Bedrock will return an error at request time
2. The error message will indicate which variant is unsupported
3. Try a different model or tool combination
4. Check the Bedrock documentation for your model

## Limitations

- Not all models support all tool variants
- Some tools may have regional availability restrictions
- Tool availability may differ across Bedrock regions
- Mixed tools (functions + variants) depend on model capability

## Advanced Usage

### Streaming with Tools

```go
for resp := range llm.GenerateContent(ctx, req, true) { // true = streaming
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
