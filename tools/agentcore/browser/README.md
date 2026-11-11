# browser

`browser` provides ADK tools for constrained Amazon Bedrock AgentCore Browser sessions.

## Single page visits

Use `browser.NewPage(cfg)` to register `browse_page`. Pass a `url` and optionally a CSS `selector` to wait for and read, or a `screenshot` artifact filename to capture the full page. The tool returns the final URL, title, visible text, truncation flag, and any screenshot artifact metadata. Each call opens a fresh session, performs the work, and attempts to stop the session with a separate cleanup deadline.

## Interactive browsing

```go
awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}
token := os.Getenv("EXAMPLE_TOKEN")

browserTool, err := browser.New(browser.Config{
    API:               bedrockagentcore.NewFromConfig(awsCfg),
    Region:            awsCfg.Region,
    Credentials:       awsCfg.Credentials,
    BrowserIdentifier: "aws.browser.v1",
    AllowedHosts:      []string{"example.com"},
    WaitUntil:         browser.WaitUntilDOMContentLoaded,
    RequestMiddlewares: []browser.RequestMiddleware{
        func(next browser.RequestHandler) browser.RequestHandler {
            return func(ctx context.Context, req *browser.Request) (*browser.Response, error) {
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

The tool is named `agentcore_browser`. Each call selects an `action`.

- `start` optionally opens a browser session ahead of navigation and reuses the current session if one exists.
- `navigate` opens a URL and starts a session when needed. `wait_until` overrides the configured lifecycle wait and `wait_for_selector` waits for a CSS selector before returning.
- `extract_text` returns visible text from the current page or an optional CSS `selector`. `wait_for_selector` waits before extraction and becomes the extraction selector when `selector` is omitted.
- `screenshot` saves a screenshot of the current page as an ADK artifact and returns its URL and title. `full_page` defaults to `true`. JPEG `quality` accepts integral values from 0 through 100. Use an extensionless `file_name` or one ending in `.jpg`, `.jpeg`, or `.png`. The extension selects the format when `format` is omitted.
- `status` reports the current session's status or that the conversation has no active session.
- `stop` ends the current session if one exists and succeeds when the conversation has no active session.

## Behavior

The tool manages one Browser session per ADK conversation and keeps the AgentCore session identifier internally. This state lives in memory for the lifetime of the tool instance. Run dependent actions in sequence. Wait for `navigate` to finish before extracting text or taking a screenshot, and call `stop` after the requested work is complete.

The tool uses AgentCore session APIs plus the automation stream. It only exposes the actions above; it does not expose raw CDP, mouse, keyboard, or arbitrary browser protocol access.

## Configuration

Zero values select the defaults below unless noted otherwise. Negative timeouts and limits are rejected.

| Field | Default | Purpose |
| --- | --- | --- |
| `BrowserIdentifier` | `aws.browser.v1` | AgentCore browser resource identifier. |
| `SessionTimeoutSeconds` | `900` | AgentCore session lifetime, from 60 through 28,800 seconds. |
| `ViewportWidth`, `ViewportHeight` | AgentCore default | Optional viewport dimensions; set both together. |
| `AllowedHosts`, `DeniedHosts` | empty | Lexical HTTP(S) host policy; deny rules win. |
| `AllowUnrestrictedDNS` | `false` | Allow DNS hostnames when `AllowedHosts` is empty. |
| `RequestMiddlewares` | none | Mutate, fulfill, or block intercepted browser requests. |
| `URLMiddlewares` | none | Compose or replace URL host policy for navigate, request, current, and final stages. |
| `Dialer` | `websocket.Dial` | Open the signed automation WebSocket through a custom proxy or transport. |
| `AuthHandler` | disabled | Answer HTTP authentication challenges with default handling, cancellation, or credentials. |
| `NavigationTimeout` | `30s` | Overall deadline for a page visit or interactive browser action, including selector waits and artifact saving. |
| `CleanupTimeout` | `10s` | Separate stop deadline after a page visit or a failed auto-started interactive navigation. |
| `MaxTextBytes` | `64 KiB` | Maximum extracted visible text size. |
| `MaxScreenshotBytes` | `16 MiB` | Maximum decoded screenshot artifact size. |
| `WaitUntil` | `load` | Navigation wait mode: `load`, `dom_content_loaded`, or `none`. |

### URL Middleware

`URLMiddlewares` receives a parsed `URLCheck` with a `URLStage` of `navigate`, `request`, `current`, or `final`. Middleware is applied in list order with the first entry outermost. URL middleware is validation-only: rewriting a URL and calling `next` is rejected because the checked URL must match the one used by the browser. Use request middleware to rewrite browser requests. Calling `next` applies the remaining URL middleware and built-in host policy; omitting `next` replaces host policy. Parsing, user-info rejection, and HTTP(S) requirements for explicit/current/final page URLs remain fixed structural validation.

```go
auditPolicy := func(next browser.URLHandler) browser.URLHandler {
    return func(ctx context.Context, check browser.URLCheck) error {
        log.Printf("browser URL stage=%s url=%s", check.Stage, check.URL.String())
        return next(ctx, check)
    }
}

browserTool, err := browser.New(browser.Config{
    API:            bedrockagentcore.NewFromConfig(awsCfg),
    Region:         awsCfg.Region,
    Credentials:    awsCfg.Credentials,
    AllowedHosts:   []string{"example.com"},
    URLMiddlewares: []browser.URLMiddleware{auditPolicy},
})
```

URL middleware may inspect and reject URLs, but does not change the browser destination. Request middleware that deliberately omits its own `next` may bypass request-stage URL policy for custom routing or synthetic responses; explicit navigation and observed current/final page URLs are still checked independently.

### Request Middleware

`RequestMiddlewares` wraps request-stage browser interception. Middleware receives URL, method, headers, body, resource type, frame, network, and redirect metadata. It may mutate URL, method, headers, or body before calling `next`, return a `Response` to fulfill the request without network access, or return an error to block the request. Middleware is applied in list order, with the first entry outermost.

Calling `next` applies the remaining middleware and the built-in host policy. Omitting `next` replaces request-stage host handling for that request, which supports custom routing, mocking, and policy implementations. The explicit `navigate` input and current/final page URLs are still checked against `AllowedHosts` and `DeniedHosts`. Middleware may run concurrently for separate tool calls and must be concurrency-safe. Its context carries the configured action deadline.

### Authentication And Transport

Set `AuthHandler` to enable Fetch authentication events. The tool checks the challenge's request URL with the request-stage URL policy before calling the handler and cancels the challenge if the policy rejects it. An empty action selects default browser handling. Use `AuthActionCancel` or `AuthActionProvideCredentials` for explicit responses.

```go
AuthHandler: func(ctx context.Context, challenge browser.AuthChallenge) (browser.AuthResponse, error) {
    if challenge.Origin != "https://example.com" {
        return browser.AuthResponse{Action: browser.AuthActionCancel}, nil
    }
    return browser.AuthResponse{
        Action:   browser.AuthActionProvideCredentials,
        Username: os.Getenv("BROWSER_USERNAME"),
        Password: os.Getenv("BROWSER_PASSWORD"),
    }, nil
},
```

`Dialer` accepts any `WebSocketDialer` with Coder WebSocket's `Dial` signature. Its `DialOptions.HTTPHeader` contains the SigV4-signed headers, making proxying, observability, and custom TLS behavior injectable without exposing CDP through the ADK tool.

By default sessions use `aws.browser.v1` and a 900 second timeout. `wait_until: "none"` waits only for the main frame's navigation commit, not a document lifecycle event, before returning page metadata. `NavigationTimeout` bounds the AgentCore lookup, WebSocket/CDP work, lifecycle and selector waits, and artifact save performed by `navigate`, `extract_text`, and `screenshot`. Screenshot payloads are bounded before and after base64 decoding and immediately before artifact storage. The automation WebSocket also has an internal read limit derived from the configured text and screenshot bounds.

`AllowedHosts` and `DeniedHosts` apply to all HTTP(S) page and subresource requests while `navigate`, `extract_text`, and `screenshot` run, as well as the current URL before text is returned or an artifact is saved; deny rules win. Browser-local schemes such as `data:` and `blob:` are allowed. With an empty `AllowedHosts` list, DNS hostnames are rejected unless `AllowUnrestrictedDNS` is explicitly enabled; public IP literals remain allowed, while localhost, non-public IP literals, and legacy IPv4 spellings require an explicit allowlist entry. DNS names are matched lexically rather than resolved locally because AgentCore, including VPC-connected browsers, may resolve them differently. Enabling unrestricted DNS may allow access to private resources reachable by the browser. Prefer an explicit `AllowedHosts` list and include any HTTP(S) hosts needed for scripts, styles, images, or API calls.

Browser sessions are billable until explicitly stopped or their session timeout expires. Keep `SessionTimeoutSeconds` no longer than the workflow needs (AgentCore allows up to 28,800 seconds) and call `stop` when work is complete.

## Required IAM Actions

- `bedrock-agentcore:StartBrowserSession`
- `bedrock-agentcore:GetBrowserSession`
- `bedrock-agentcore:StopBrowserSession`
- `bedrock-agentcore:ConnectBrowserAutomationStream`

See [`../../../examples/bedrock-agentcore-browser`](../../../examples/bedrock-agentcore-browser) for a runnable setup.
