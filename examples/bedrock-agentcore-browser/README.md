# Bedrock AgentCore Browser example

This example runs an ADK agent with the `agentcore_browser` tool for Amazon Bedrock AgentCore Browser.

## Setup

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0

# Optional:
export AGENTCORE_BROWSER_ID=aws.browser.v1
export AGENTCORE_BROWSER_ALLOWED_HOSTS=example.com
export AGENTCORE_BROWSER_DENIED_HOSTS=
export AGENTCORE_BROWSER_SESSION_TIMEOUT_SECONDS=900
export AGENTCORE_BROWSER_VIEWPORT_WIDTH=1280
export AGENTCORE_BROWSER_VIEWPORT_HEIGHT=720
export AGENTCORE_BROWSER_NAVIGATION_TIMEOUT=30s
export AGENTCORE_BROWSER_CLEANUP_TIMEOUT=10s
export AGENTCORE_BROWSER_MAX_TEXT_BYTES=65536
export AGENTCORE_BROWSER_MAX_SCREENSHOT_BYTES=16777216
export AGENTCORE_BROWSER_WAIT_UNTIL=load

go run ./examples/bedrock-agentcore-browser
```

Pass a prompt as CLI arguments to browse a different site.

```bash
go run ./examples/bedrock-agentcore-browser "Open https://example.com and save a screenshot as page.png"
```

## IAM

The Bedrock model client needs normal Bedrock inference permissions. The AgentCore client needs:

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:GetBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

Browser sessions have runtime cost and remain active until stopped or timed out. The example asks the agent to stop each session when finished; keep the host allowlist set for constrained or private-network workflows. Empty optional environment variables select the library defaults documented in [`../../tools/agentcorebrowser`](../../tools/agentcorebrowser).
