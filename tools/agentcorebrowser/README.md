# agentcorebrowser

`agentcorebrowser` provides an ADK tool for constrained Amazon Bedrock AgentCore Browser sessions.

## Usage

```go
awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}
token := os.Getenv("EXAMPLE_TOKEN")

browserTool, err := agentcorebrowser.New(agentcorebrowser.Config{
    API:               bedrockagentcore.NewFromConfig(awsCfg),
    Region:            awsCfg.Region,
    Credentials:       awsCfg.Credentials,
    BrowserIdentifier: "aws.browser.v1",
    AllowedHosts:      []string{"example.com"},
    RequestMiddlewares: []agentcorebrowser.RequestMiddleware{
        func(next agentcorebrowser.RequestHandler) agentcorebrowser.RequestHandler {
            return func(ctx context.Context, req *agentcorebrowser.BrowserRequest) (*agentcorebrowser.BrowserResponse, error) {
                if req.URL == "https://example.com/api" {
                    req.Headers.Set("Authorization", "Bearer "+token)
                }
                return next(ctx, req)
            }
        },
    },
})
if err != nil {
    log.Fatal(err)
}
```

The tool is named `agentcore_browser`. Calls use an `action` value:

- `start`: create a browser session.
- `navigate`: navigate to a URL. If `session_id` is omitted, the tool starts a session.
- `extract_text`: return visible text for the page or an optional CSS `selector`.
- `screenshot`: save a browser screenshot as an ADK artifact and return its URL/title metadata. If `format` is omitted, `.jpg`, `.jpeg`, or `.png` in `file_name` selects the format. Extensionless names are allowed; other extensions are rejected.
- `status`: return AgentCore Browser session status.
- `stop`: stop a browser session.

## Behavior

The tool uses AgentCore session APIs plus the automation stream. It only exposes the actions above; it does not expose raw CDP, mouse, keyboard, or arbitrary browser protocol access.

### Request Middleware

`RequestMiddlewares` wraps request-stage browser interception. Middleware receives URL, method, headers, body, resource type, frame, network, and redirect metadata. It may mutate URL, method, headers, or body before calling `next`, return a `BrowserResponse` to fulfill the request without network access, or return an error to block the request. Middleware is applied in list order, with the first entry outermost.

Calling `next` applies the remaining middleware and the built-in host policy. Omitting `next` replaces request-stage host handling for that request, which supports custom routing, mocking, and policy implementations. The explicit `navigate` input and current/final page URLs are still checked against `AllowedHosts` and `DeniedHosts`. Middleware may run concurrently for separate tool calls and must be concurrency-safe. Its context carries the configured action deadline.

By default sessions use `aws.browser.v1` and a 900 second timeout. `NavigationTimeout` defaults to 30 seconds and bounds the AgentCore lookup, WebSocket/CDP work, and artifact save performed by `navigate`, `extract_text`, and `screenshot`. Failed auto-started navigations get a separate best-effort 10 second cleanup window. `MaxTextBytes` defaults to 64 KiB and caps text inside the browser before it crosses the automation stream.

`AllowedHosts` and `DeniedHosts` apply to all HTTP(S) page and subresource requests while `navigate`, `extract_text`, and `screenshot` run, as well as the current URL before text is returned or an artifact is saved; deny rules win. Browser-local schemes such as `data:` and `blob:` are allowed. An empty `AllowedHosts` list permits syntactically public hostnames and public IP literals except entries matched by `DeniedHosts`, while localhost, non-public IP literals, and legacy IPv4 spellings require an explicit allowlist entry. DNS names are matched lexically rather than resolved locally because AgentCore, including VPC-connected browsers, may resolve them differently. Use an explicit `AllowedHosts` list when the browser can reach private networks or the workflow needs a strong domain boundary, and include any HTTP(S) hosts needed for scripts, styles, images, or API calls.

Browser sessions are billable until explicitly stopped or their session timeout expires. Keep `SessionTimeoutSeconds` no longer than the workflow needs (AgentCore allows up to 28,800 seconds) and call `stop` when work is complete.

## Required IAM Actions

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:GetBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

See [`../../examples/bedrock-agentcore-browser`](../../examples/bedrock-agentcore-browser) for a runnable setup.
