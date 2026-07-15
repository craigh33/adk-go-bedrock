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
- `screenshot`: save a browser screenshot as an ADK artifact and return its URL/title metadata. If `format` is omitted, `.jpg`, `.jpeg`, or `.png` in `file_name` selects the format.
- `status`: return AgentCore Browser session status.
- `stop`: stop a browser session.

## Behavior

The tool uses AgentCore session APIs plus the automation stream. It only exposes the actions above; it does not expose raw CDP, mouse, keyboard, or arbitrary browser protocol access.

By default sessions use `aws.browser.v1` and a 900 second timeout. `NavigationTimeout` defaults to 30 seconds and bounds the AgentCore lookup, WebSocket/CDP work, and artifact save performed by `navigate`, `extract_text`, and `screenshot`. Failed auto-started navigations get a separate best-effort 10 second cleanup window. `MaxTextBytes` defaults to 64 KiB and caps text inside the browser before it crosses the automation stream.

`AllowedHosts` and `DeniedHosts` apply to requested and redirected document URLs while `navigate`, `extract_text`, and `screenshot` run, as well as the current URL before text is returned or an artifact is saved; deny rules win. An empty `AllowedHosts` list permits syntactically public hostnames and public IP literals except entries matched by `DeniedHosts`, while localhost, non-public IP literals, and legacy IPv4 spellings require an explicit allowlist entry. DNS names are matched lexically rather than resolved locally because AgentCore, including VPC-connected browsers, may resolve them differently. Use an explicit `AllowedHosts` list when the browser can reach private networks or the workflow needs a strong domain boundary.

Browser sessions are billable until explicitly stopped or their session timeout expires. Keep `SessionTimeoutSeconds` no longer than the workflow needs (AgentCore allows up to 28,800 seconds) and call `stop` when work is complete.

## Required IAM Actions

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:GetBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

See [`../../examples/bedrock-agentcore-browser`](../../examples/bedrock-agentcore-browser) for a runnable setup.
