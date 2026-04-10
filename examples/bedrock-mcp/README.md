# bedrock-mcp example

This example is based on the ADK MCP example and shows how to use ADK's `mcptoolset` with the Bedrock Converse provider.

By default it starts an **in-memory MCP server** inside the process and exposes a single `get_weather` MCP tool. You can also switch to a **remote GitHub MCP endpoint** by setting `AGENT_MODE=github`.

## Prerequisites

- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)
- Optional: `BEDROCK_MODEL_ID` to override the default Bedrock model ID or inference profile ARN (must support tool use). If unset, this example defaults to `eu.amazon.nova-2-lite-v1:0`.

If you want to use the GitHub MCP endpoint mode:

- `AGENT_MODE=github`
- `GITHUB_PAT=<your token>`

## Run

```bash
make -C examples/bedrock-mcp run
```

The Makefile forwards `PROMPT` directly as CLI args to the ADK launcher (`go run . $(PROMPT)`).
For example:

```bash
make -C examples/bedrock-mcp run PROMPT='--help'
```

Run against GitHub MCP transport:

```bash
AGENT_MODE=github GITHUB_PAT=your_token make -C examples/bedrock-mcp run
```

## How It Works

1. Load AWS config from the default chain and resolve a Bedrock model ID
2. Choose MCP transport:
   - Local mode (default): in-memory MCP server with `get_weather`
   - GitHub mode (`AGENT_MODE=github`): streamable HTTP MCP transport
3. Create an ADK MCP toolset with `mcptoolset.New(...)`
4. Attach that toolset to an `llmagent`
5. Run via the ADK full launcher, forwarding CLI args from `os.Args[1:]`

Because ADK handles MCP tool discovery and tool execution, the Bedrock provider only needs to support the standard ADK function-calling path.
