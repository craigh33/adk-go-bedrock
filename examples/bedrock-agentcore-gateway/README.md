# bedrock-agentcore-gateway example

This example runs a Bedrock-backed ADK agent with tools discovered from an existing Amazon Bedrock AgentCore Gateway MCP endpoint.

AgentCore Gateway exposes MCP-compatible tools, so no dedicated library wrapper is needed: use ADK's `mcptoolset` with the MCP SDK streamable HTTP transport.

## Prerequisites

- `AGENTCORE_GATEWAY_ENDPOINT`: AgentCore Gateway MCP endpoint URL
- `AGENTCORE_GATEWAY_ACCESS_TOKEN`: bearer token for the Gateway endpoint
- AWS credentials configured via the default chain
- AWS region configured, for example `AWS_REGION=us-east-1`
- Optional: `BEDROCK_MODEL_ID` to override the default Bedrock model ID or inference profile ARN. If unset, this example defaults to `eu.amazon.nova-2-lite-v1:0`.

## Run

```bash
AGENTCORE_GATEWAY_ENDPOINT='https://example.gateway/mcp' \
AGENTCORE_GATEWAY_ACCESS_TOKEN='token' \
make -C examples/bedrock-agentcore-gateway run
```

The Makefile forwards `PROMPT` directly as CLI args to the ADK launcher:

```bash
make -C examples/bedrock-agentcore-gateway run PROMPT='--help'
```

## How It Works

1. Load AWS config from the default chain and resolve a Bedrock model ID
2. Create an ADK MCP toolset from `AGENTCORE_GATEWAY_ENDPOINT`
3. Attach the toolset to an ADK `llmagent`
4. Run via the ADK full launcher, forwarding CLI args from `os.Args[1:]`

The Gateway must already exist. This example does not create gateways, targets, credential providers, policies, or access tokens.
