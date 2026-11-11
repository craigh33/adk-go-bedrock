# Bedrock AgentCore Browser example

This example runs an ADK agent with the `browse_page` tool for Amazon Bedrock AgentCore Browser. Each call takes a `url` and returns visible page text. Use the optional `selector` to extract text from a CSS match and `screenshot` to specify a screenshot artifact filename.

## Setup

Use a non-root AWS identity with the permissions below. If you use a named CLI profile, set `AWS_PROFILE` to that profile. Check the active identity before running the example.

```bash
aws sts get-caller-identity --query Arn --output text
```

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

For London, set `AWS_REGION=eu-west-2` and `BEDROCK_MODEL_ID=amazon.nova-lite-v1:0`. [Nova Lite regional availability](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html)

The example allows `example.com` when `AGENTCORE_BROWSER_ALLOWED_HOSTS` is unset or empty. Pass a prompt as CLI arguments to customize the request. To browse another site, set `AGENTCORE_BROWSER_ALLOWED_HOSTS` to a comma-separated list of hosts the request needs. This list replaces the example default.

```bash
go run ./examples/bedrock-agentcore-browser "Open https://example.com and save a screenshot as page.png"
```

## IAM

The Bedrock model client needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` for the configured model or inference profile. The AgentCore client needs the following permissions.

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

Scope the session lifecycle actions to the configured Browser ARN. For example, the system Browser in London is `arn:aws:bedrock-agentcore:eu-west-2:aws:browser/aws.browser.v1`. Grant `ConnectBrowserAutomationStream` separately with `Resource` set to `*` and restrict it to the Browser region with `aws:RequestedRegion`. [AgentCore service authorization reference](https://docs.aws.amazon.com/service-authorization/latest/reference/list_bedrock-agentcore.html)

The tool opens and stops the Browser session for each call. Browser sessions have runtime cost until stopped or timed out. Other empty optional environment variables select the library defaults documented in [`../../tools/agentcore/browser`](../../tools/agentcore/browser).
