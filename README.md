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

The [`bedrock/mappers`](bedrock/mappers/) package holds genai ↔ Bedrock conversions (requests, responses, tools, usage). Import it if you need the same mappings outside the default [`bedrock`](bedrock/) package. The Bedrock Runtime API abstraction used by [`converse.go`](bedrock/converse.go) is exported from [`bedrock`](bedrock/) (`RuntimeAPI`, `StreamReader`, and `NewRuntimeAPI`).

## Examples

Each example has its own `README.md` and `Makefile`:

- [`examples/bedrock-a2a`](examples/bedrock-a2a): A2A remote-agent example backed by Bedrock.
- [`examples/bedrock-chat`](examples/bedrock-chat): runner-based chat example.
- [`examples/bedrock-mcp`](examples/bedrock-mcp): MCP support via ADK's `mcptoolset` with an in-memory MCP server ([MCP support](#mcp-support)).
- [`examples/bedrock-tool-calling`](examples/bedrock-tool-calling): tool-calling agent example with function declarations.
- [`examples/bedrock-stream`](examples/bedrock-stream): direct streaming example using `GenerateContent(..., true)`.
- [`examples/bedrock-tool-variants`](examples/bedrock-tool-variants): function declaration support plus early detection of non-function ADK tool variants that Bedrock does not currently support.
- [`examples/bedrock-multimodal`](examples/bedrock-multimodal): comprehensive image analysis, document processing, tool calling with rich media, and vision-based reasoning.
- [`examples/bedrock-guardrails`](examples/bedrock-guardrails): safety assessments, content filtering, and guardrail metadata handling.
- [`examples/bedrock-system-instruction`](examples/bedrock-system-instruction): system instructions for role definition, output formatting, and behavioral control.
- [`examples/bedrock-web-ui`](examples/bedrock-web-ui): ADK local web UI launcher.

All examples load AWS configuration with [`config.LoadDefaultConfig`](https://pkg.go.dev/github.com/aws/aws-sdk-go-v2/config#LoadDefaultConfig) and require **`BEDROCK_MODEL_ID`** plus region configuration (`AWS_REGION` or profile region).

```bash
export BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20241022-v2:0
export AWS_REGION=us-east-1   # optional if your profile already defines region
make -C examples/bedrock-chat run
```

Run streaming example:

```bash
make -C examples/bedrock-stream run
```

## How it maps to Bedrock

- **Messages**: `genai` roles `user` and `model` map to Bedrock `user` and `assistant`. Optional `system` role entries in the conversation are mapped to Bedrock `system` blocks.
- **System instruction**: `GenerateContentConfig.SystemInstruction` is sent as Bedrock system content.
- **Tools**: the mapper converts `GenerateContentConfig.Tools` entries:
  - `FunctionDeclarations` → Bedrock `ToolSpecification` (custom function tools)
  - Non-function ADK variants (Google Search, Code Execution, Retrieval, MCP Servers, Computer Use, File Search, Google Maps, URL Context, etc.) are rejected early with a clear provider error because they are not currently mapped to Bedrock Converse
  - MCP: use ADK `mcptoolset` so MCP tools become function declarations before they reach this provider ([MCP support](#mcp-support)). Other `genai.Tool` variants (Google Search, code execution, etc.) are not supported here.
- **Multimodal parts**: ADK `Part` text, thoughts/reasoning, inline/file-backed images, audio, video, and documents are mapped on the Bedrock-compatible subset. Rich user media is sent as Bedrock content blocks; assistant reasoning is preserved as Bedrock reasoning content.
- **Function responses**: JSON tool output still maps as before, and image/video/document `FunctionResponse.Parts` are preserved through Bedrock tool-result content blocks.
- **Streaming**: When ADK uses SSE streaming, the provider calls `ConverseStream`, emits partial text responses, and buffers streamed tool calls, reasoning blocks, image blocks, usage, and guardrail metadata into the final `TurnComplete` response.
- **Guardrails / safety results**: Bedrock guardrail stop reasons and trace metadata are mapped back into ADK `FinishReason` and `CustomMetadata`, including synthesized `safety_ratings` derived from Bedrock guardrail assessments when available.

## Limitations

- **Bedrock role restrictions**: Rich media input still follows Bedrock Converse constraints (for example, user turns are the interoperable place for images/documents/audio/video, while model turns are reserved for text/tool use/reasoning).
- **Request-side generic guardrails**: ADK `SafetySettings` / `ModelArmorConfig` are not currently supported for Bedrock Converse. Because they do not contain the Bedrock guardrail identifier + version required by `Converse`, the request builder returns an explicit error instead of silently dropping them, and there is currently no supported way to provide `GuardrailIdentifier` / `GuardrailVersion` through this provider.
- **Provider surface mismatch**: Bedrock-specific features that require pre-provisioned AWS resources or have no generic ADK equivalent are exposed back through ADK-friendly `CustomMetadata`, but cannot always be reconstructed into first-class `genai` request fields.
- **Unsupported tool types**: Tool variants not supported by Bedrock or the target model cause a request-time error with details about which variants are unsupported.


## License

Apache 2.0 — see [LICENSE](LICENSE).
