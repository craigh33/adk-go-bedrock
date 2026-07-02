# bedrock-mantle-chat example

This example runs a simple ADK runner through the **Amazon Bedrock Mantle**
endpoint — an Anthropic-compatible Messages API — instead of the native Bedrock
Converse API. It reuses the same `bedrock.Model` by wrapping a Mantle
`RuntimeAPI` implementation with `bedrock.NewWithAPI`.

## Prerequisites

- `AWS_REGION` set (used for the Mantle base URL and SigV4 signing)
- Authentication, via one of:
  - an API key: `AWS_BEARER_TOKEN_BEDROCK` (or `ANTHROPIC_API_KEY`), or
  - the default AWS credential chain for SigV4 (env vars, shared config,
    SSO / `AWS_PROFILE`, instance role, etc.)
- Optionally `BEDROCK_MANTLE_MODEL_ID` (an Anthropic model ID such as
  `anthropic.claude-sonnet-4-5-20250929-v1:0`; a Converse-style region prefix
  like `us.anthropic.…` is accepted and normalized). Defaults to
  `anthropic.claude-sonnet-4-5-20250929-v1:0`.

## Run

```bash
make -C examples/bedrock-mantle-chat run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-mantle-chat run PROMPT='Summarize static typing in one sentence.'
```

> This example is a self-contained leaf module. It is built and run with
> `GOWORK=off` (the `Makefile` sets this) so it resolves the in-repo core and
> Mantle modules through its `replace` directives rather than the repository
> `go.work` workspace. Running `go run .` directly from the repo root without
> `GOWORK=off` will fail because the module is intentionally not part of the
> workspace.
