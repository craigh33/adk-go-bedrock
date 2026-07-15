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
    WaitUntil:         agentcorebrowser.WaitUntilDOMContentLoaded,
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
- `navigate`: navigate to a URL. If `session_id` is omitted, the tool starts a session. `wait_until` overrides the configured lifecycle wait and `wait_for_selector` waits for a CSS selector before returning.
- `extract_text`: return visible text for the page or an optional CSS `selector`. `wait_for_selector` waits before extraction and becomes the extraction selector when `selector` is omitted.
- `screenshot`: save a browser screenshot as an ADK artifact and return its URL/title metadata. `full_page` defaults to `true`; JPEG `quality` accepts integral values from 0 through 100. If `format` is omitted, `.jpg`, `.jpeg`, or `.png` in `file_name` selects the format. Extensionless names are allowed; other extensions are rejected.
- `status`: return AgentCore Browser session status.
- `stop`: stop a browser session.

## Behavior

The tool uses AgentCore session APIs plus the automation stream. It only exposes the actions above; it does not expose raw CDP, mouse, keyboard, or arbitrary browser protocol access.

## Configuration

Zero values select the defaults below unless noted otherwise. Negative timeouts and limits are rejected.

| Field | Default | Purpose |
| --- | --- | --- |
| `BrowserIdentifier` | `aws.browser.v1` | AgentCore browser resource identifier. |
| `SessionTimeoutSeconds` | `900` | AgentCore session lifetime, up to 28,800 seconds. |
| `ViewportWidth`, `ViewportHeight` | AgentCore default | Optional viewport dimensions; set both together. |
| `AllowedHosts`, `DeniedHosts` | public hosts allowed | Lexical HTTP(S) host policy; deny rules win. |
| `RequestMiddlewares` | none | Mutate, fulfill, or block intercepted browser requests. |
| `URLMiddlewares` | none | Compose or replace URL host policy for navigate, request, current, and final stages. |
| `Dialer` | `websocket.DefaultDialer` | Open the signed automation WebSocket through a custom proxy or transport. |
| `AuthHandler` | disabled | Answer HTTP authentication challenges with default handling, cancellation, or credentials. |
| `NavigationTimeout` | `30s` | Overall deadline for browser actions, selector waits, and artifact saving. |
| `CleanupTimeout` | `10s` | Separate best-effort stop deadline after a failed auto-started navigation. |
| `MaxTextBytes` | `64 KiB` | Maximum extracted visible text size. |
| `MaxScreenshotBytes` | `16 MiB` | Maximum decoded screenshot artifact size. |
| `WaitUntil` | `load` | Navigation wait mode: `load`, `dom_content_loaded`, or `none`. |

### URL Middleware

`URLMiddlewares` receives a parsed `URLCheck` with a `URLStage` of `navigate`, `request`, `current`, or `final`. Middleware is applied in list order with the first entry outermost. Calling `next` applies the remaining middleware and built-in host policy; omitting `next` replaces host policy. Parsing, user-info rejection, and HTTP(S) requirements for explicit/current/final page URLs remain fixed structural validation.

```go
auditPolicy := func(next agentcorebrowser.URLHandler) agentcorebrowser.URLHandler {
    return func(ctx context.Context, check agentcorebrowser.URLCheck) error {
        log.Printf("browser URL stage=%s url=%s", check.Stage, check.URL.String())
        return next(ctx, check)
    }
}

browserTool, err := agentcorebrowser.New(agentcorebrowser.Config{
    API:            bedrockagentcore.NewFromConfig(awsCfg),
    Region:         awsCfg.Region,
    Credentials:    awsCfg.Credentials,
    AllowedHosts:   []string{"example.com"},
    URLMiddlewares: []agentcorebrowser.URLMiddleware{auditPolicy},
})
```

When URL middleware rewrites a `URLCheck` and calls `next`, the rewritten URL is structurally validated and passed through host policy. Request middleware that deliberately omits its own `next` may bypass request-stage URL policy for custom routing or synthetic responses; explicit navigation and observed current/final page URLs are still checked independently.

### Request Middleware

`RequestMiddlewares` wraps request-stage browser interception. Middleware receives URL, method, headers, body, resource type, frame, network, and redirect metadata. It may mutate URL, method, headers, or body before calling `next`, return a `BrowserResponse` to fulfill the request without network access, or return an error to block the request. Middleware is applied in list order, with the first entry outermost.

Calling `next` applies the remaining middleware and the built-in host policy. Omitting `next` replaces request-stage host handling for that request, which supports custom routing, mocking, and policy implementations. The explicit `navigate` input and current/final page URLs are still checked against `AllowedHosts` and `DeniedHosts`. Middleware may run concurrently for separate tool calls and must be concurrency-safe. Its context carries the configured action deadline.

### Authentication And Transport

Set `AuthHandler` to enable Fetch authentication events. An empty action selects default browser handling; use `AuthActionCancel` or `AuthActionProvideCredentials` for explicit responses.

```go
AuthHandler: func(ctx context.Context, challenge agentcorebrowser.AuthChallenge) (agentcorebrowser.AuthResponse, error) {
    if challenge.Origin != "https://example.com" {
        return agentcorebrowser.AuthResponse{Action: agentcorebrowser.AuthActionCancel}, nil
    }
    return agentcorebrowser.AuthResponse{
        Action:   agentcorebrowser.AuthActionProvideCredentials,
        Username: os.Getenv("BROWSER_USERNAME"),
        Password: os.Getenv("BROWSER_PASSWORD"),
    }, nil
},
```

`Dialer` accepts any `WebSocketDialer` with Gorilla's `DialContext` signature. It receives the AgentCore endpoint and SigV4-signed headers, making proxying, observability, and custom TLS behavior injectable without exposing CDP through the ADK tool.

By default sessions use `aws.browser.v1` and a 900 second timeout. `NavigationTimeout` bounds the AgentCore lookup, WebSocket/CDP work, lifecycle and selector waits, and artifact save performed by `navigate`, `extract_text`, and `screenshot`. Screenshot payloads are bounded before and after base64 decoding and immediately before artifact storage. The automation WebSocket also has an internal read limit derived from the configured text and screenshot bounds.

`AllowedHosts` and `DeniedHosts` apply to all HTTP(S) page and subresource requests while `navigate`, `extract_text`, and `screenshot` run, as well as the current URL before text is returned or an artifact is saved; deny rules win. Browser-local schemes such as `data:` and `blob:` are allowed. An empty `AllowedHosts` list permits syntactically public hostnames and public IP literals except entries matched by `DeniedHosts`, while localhost, non-public IP literals, and legacy IPv4 spellings require an explicit allowlist entry. DNS names are matched lexically rather than resolved locally because AgentCore, including VPC-connected browsers, may resolve them differently. Use an explicit `AllowedHosts` list when the browser can reach private networks or the workflow needs a strong domain boundary, and include any HTTP(S) hosts needed for scripts, styles, images, or API calls.

Browser sessions are billable until explicitly stopped or their session timeout expires. Keep `SessionTimeoutSeconds` no longer than the workflow needs (AgentCore allows up to 28,800 seconds) and call `stop` when work is complete.

## Required IAM Actions

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:GetBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

See [`../../examples/bedrock-agentcore-browser`](../../examples/bedrock-agentcore-browser) for a runnable setup.
