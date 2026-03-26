# adk-go-bedrock

[Amazon Bedrock](https://aws.amazon.com/bedrock/) **Converse** / **ConverseStream** implementation of the [`model.LLM`](https://pkg.go.dev/google.golang.org/adk/model#LLM) interface for [adk-go](https://github.com/google/adk-go), so you can run agents on Claude, Nova, and other Bedrock chat models with the same ADK APIs you use for Gemini.

## Requirements

- **Go** 1.25+ (aligned with `google.golang.org/adk`)
- **AWS account** with Bedrock model access in your chosen region
- **Credentials** via the default AWS chain ([environment variables](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html), shared config, IAM role, etc.)
- A **region** where Bedrock is available: `AWS_REGION`, or `region` in `~/.aws/config` for your profile
- IAM permission to call inference, for example:
  - `bedrock:InvokeModel` for `Converse`
  - `bedrock:InvokeModelWithResponseStream` for `ConverseStream` (when ADK uses SSE streaming)
- **[golangci-lint](https://golangci-lint.run/welcome/install/)** if you run `make lint` (uses [.golangci.yaml](.golangci.yaml))

## Install

```bash
go get github.com/craigh33/adk-go-bedrock
```

Replace the module path with your fork or published path if you rename the module in `go.mod`.

## Makefile

| Target | Description |
|--------|-------------|
| `make dev` | Run the example (`go run ./examples/bedrock-chat`) |
| `make test` | Run unit tests |
| `make build` | Compile all packages |
| `make lint` | Run `golangci-lint run ./...` |
| `make pre-commit-install` | Install pre-commit hooks |

## Contributing / Development

### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com) to enforce code quality and commit hygiene. The following tools must be available on your `PATH` before installing the hooks:

| Tool | Purpose | Install |
|------|---------|---------|
| [pre-commit](https://pre-commit.com) | Hook framework | `brew install pre-commit` |
| [golangci-lint](https://golangci-lint.run/welcome/install/) | Go linter (runs `make lint`) | `brew install golangci-lint` |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secret / credential scanner | `brew install gitleaks` |

Once the tools are installed, wire the hooks into your local clone:

```bash
make pre-commit-install
```

This installs hooks for both the `pre-commit` stage and the `commit-msg` stage.

#### What the hooks do

| Hook | Stage | Description |
|------|-------|-------------|
| `trailing-whitespace` | pre-commit | Strips trailing whitespace |
| `end-of-file-fixer` | pre-commit | Ensures files end with a newline |
| `check-yaml` | pre-commit | Validates YAML syntax |
| `no-commit-to-branch` | pre-commit | Prevents direct commits to `main` |
| `conventional-pre-commit` | commit-msg | Enforces [Conventional Commits](https://www.conventionalcommits.org/) message format (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`) |
| `golangci-lint` | pre-commit | Runs `make lint` against all Go files |
| `gitleaks` | pre-commit | Scans staged diff for secrets/credentials |

## Usage

```go
ctx := context.Background()
llm, err := bedrock.New(ctx, "us.anthropic.claude-3-5-sonnet-20241022-v2:0", &bedrock.Options{
    Region: os.Getenv("AWS_REGION"),
})
if err != nil {
    log.Fatal(err)
}

agent, err := llmagent.New(llmagent.Config{
    Name:  "assistant",
    Model: llm,
    Instruction: "You are helpful.",
})
// Wire agent into runner.New(...) as usual.
```

`bedrock.New` accepts a **model ID** or **inference profile ARN** as documented by AWS. [`LLMRequest.Model`](https://pkg.go.dev/google.golang.org/adk/model#LLMRequest) can override the model ID at runtime (e.g. from ADK callbacks).

The [`bedrock/mappers`](bedrock/mappers/) package holds genai ↔ Bedrock conversions (requests, responses, tools, usage). Import it if you need the same mappings outside the default [`bedrock`](bedrock/) package. [`bedrock/client`](bedrock/client/) defines the Bedrock Runtime API surface ([`client.RuntimeAPI`](bedrock/client/client.go)) used by [`converse.go`](bedrock/converse.go).

## Example

The [example](examples/bedrock-chat/main.go) loads AWS configuration with [`config.LoadDefaultConfig`](https://pkg.go.dev/github.com/aws/aws-sdk-go-v2/config#LoadDefaultConfig) (environment variables, `~/.aws/credentials`, `~/.aws/config`, `AWS_PROFILE` / SSO, instance/task role, etc.), then builds the Bedrock Runtime client. Set **`BEDROCK_MODEL_ID`**. Set **`AWS_REGION`** or configure a `region` on your AWS profile so the resolved region is non-empty.

```bash
export BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0
export AWS_REGION=us-east-1   # optional if your profile already defines region
go run ./examples/bedrock-chat
```

Optional: pass a prompt as arguments:

```bash
go run ./examples/bedrock-chat "Summarize the benefits of static typing in one sentence."
```

## How it maps to Bedrock

- **Messages**: `genai` roles `user` and `model` map to Bedrock `user` and `assistant`. Optional `system` role entries in the conversation are mapped to Bedrock `system` blocks.
- **System instruction**: `GenerateContentConfig.SystemInstruction` is sent as Bedrock system content.
- **Tools**: `GenerateContentConfig.Tools` with `FunctionDeclarations` are converted to Bedrock tool specifications. Other `genai.Tool` variants (Google Search, code execution, etc.) are not supported here.
- **Streaming**: When ADK uses SSE streaming, the provider calls `ConverseStream`, emits partial text responses, then a final response with `TurnComplete` set.

## Limitations

- **Non–function-calling tools**: Only function declarations are mapped; retrieval, Google Search, MCP, and similar tool types are ignored.
- **Multimodal**: Inline images are supported for **user** turns with supported MIME types (`image/jpeg`, `image/png`, `image/gif`, `image/webp`). Other modalities may not round-trip.
- **Streaming tool calls**: Tool input is accumulated as JSON text; streamed tool use is best-effort compared to non-streaming `Converse`.
- **Safety / guardrails**: Genai safety settings are not mapped to Bedrock guardrails (you can extend the request builders if needed).

## License

Apache 2.0 — see [LICENSE](LICENSE).
