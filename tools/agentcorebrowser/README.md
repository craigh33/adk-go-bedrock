# agentcorebrowser

`agentcorebrowser` provides an ADK tool for constrained Amazon Bedrock AgentCore Browser sessions.

## Usage

```go
awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}

browserTool, err := agentcorebrowser.New(agentcorebrowser.Config{
    API:               bedrockagentcore.NewFromConfig(awsCfg),
    Region:            awsCfg.Region,
    Credentials:       awsCfg.Credentials,
    BrowserIdentifier: "aws.browser.v1",
    AllowedHosts:      []string{"example.com"},
})
if err != nil {
    log.Fatal(err)
}
```

The tool is named `agentcore_browser`. Calls use an `action` value:

- `start`: create a browser session.
- `navigate`: navigate to a URL. If `session_id` is omitted, the tool starts a session.
- `extract_text`: return visible text for the page or an optional CSS `selector`.
- `screenshot`: save a browser screenshot as an ADK artifact.
- `status`: return AgentCore Browser session status.
- `stop`: stop a browser session.

## Behavior

The tool uses AgentCore session APIs plus the automation stream. It only exposes the actions above; it does not expose raw CDP, mouse, keyboard, or arbitrary browser protocol access.

By default sessions use `aws.browser.v1` and a 900 second timeout. `AllowedHosts` and `DeniedHosts` apply to requested and final URLs during `navigate`, and to the current URL during `extract_text` and `screenshot`; deny rules win. An empty `AllowedHosts` list permits public hosts except entries matched by `DeniedHosts`, while localhost and non-public IP literals require an explicit allowlist entry.

## Required IAM Actions

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:GetBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

See [`../../examples/bedrock-agentcore-browser`](../../examples/bedrock-agentcore-browser) for a runnable setup.
