# bedrock-mcp example

This example is based on the ADK MCP example and shows how to use ADK's `mcptoolset` with the Bedrock Converse provider.

To keep the example simple, it starts an **in-memory MCP server** inside the process and exposes a single `get_weather` MCP tool. ADK discovers that MCP tool, exposes it to the model as a callable tool, executes it when requested, and the Bedrock-backed agent returns the final answer.

## Prerequisites

- Optional: `BEDROCK_MODEL_ID` to override the default Bedrock model ID or inference profile ARN (must support tool use). If unset, this example uses the default configured in `main.go`.
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-mcp run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-mcp run PROMPT='What is the weather in London? Use the MCP tool.'
```

## How It Works

1. Start an in-memory MCP server using `github.com/modelcontextprotocol/go-sdk/mcp`
2. Register a `get_weather` MCP tool on that server
3. Create an ADK MCP toolset with `mcptoolset.New(...)`
4. Attach the MCP toolset to an `llmagent`
5. Run the agent through the Bedrock provider using the standard ADK runner

Because ADK handles MCP tool discovery and tool execution, the Bedrock provider only needs to support the standard ADK function-calling path.
