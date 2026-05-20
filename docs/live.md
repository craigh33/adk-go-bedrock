# Bidirectional Live (Nova 2 Sonic) — Design Notes

This document explains how [`bedrock/live`](../bedrock/live) implements bidirectional ("Live") streaming on Amazon Bedrock and why it lives alongside — rather than inside — adk-go's [`Runner.RunLive`](https://pkg.go.dev/google.golang.org/adk/runner#Runner.RunLive).

The default model is [`amazon.nova-2-sonic-v1:0`](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-getting-started.html). The previous-generation `amazon.nova-sonic-v1:0` is wire-compatible and still works against this package, but reaches end-of-life on **2026-09-14**.

**Regional availability** (as of 2026-05):

| Model | Regions |
|---|---|
| `amazon.nova-2-sonic-v1:0` | `us-east-1`, `us-west-2`, `ap-northeast-1` |
| `amazon.nova-sonic-v1:0` (EOL 2026-09-14) | `us-east-1`, `eu-north-1`, `ap-northeast-1` |

## Why a parallel API

adk-go v1.3 introduced a public bidi surface ([`agent.LiveSession`](https://pkg.go.dev/google.golang.org/adk/agent#LiveSession), [`agent.LiveRequest`](https://pkg.go.dev/google.golang.org/adk/agent#LiveRequest), [`agent.LiveRunConfig`](https://pkg.go.dev/google.golang.org/adk/agent#LiveRunConfig), and new [`model.LLMResponse`](https://pkg.go.dev/google.golang.org/adk/model#LLMResponse) fields including `Interrupted`, `InputTranscription`, `OutputTranscription`, and `SessionResumptionHandle`). These types are model-agnostic.

The runner-level integration is not. `internal/llminternal/base_flow.go` (`Flow.RunLive`) type-asserts the model to `interface{ Client() *genai.Client }` and dials Google's [`genai.Client.Live.Connect()`](https://pkg.go.dev/google.golang.org/genai#Client) — a hardcoded Gemini WebSocket path. There is no extension point that lets a Bedrock backend register itself with `Runner.RunLive`.

We could either fork that internal package (brittle) or land an upstream abstraction (slow). Neither belongs in a dependency bump. So `bedrock/live` instead exposes [`Open`](../bedrock/live/session.go) — a constructor that returns the same `(LiveSession, iter.Seq2[*session.Event, error], error)` triple `Runner.RunLive` does, so consumer code reads almost identically. Once adk-go ships a backend abstraction we can deprecate `Open` in favor of the runner path.

## Wire format

Bedrock's [`InvokeModelWithBidirectionalStream`](https://pkg.go.dev/github.com/aws/aws-sdk-go-v2/service/bedrockruntime#Client.InvokeModelWithBidirectionalStream) exposes opaque `BidirectionalInputPayloadPart` / `BidirectionalOutputPayloadPart` byte chunks. Each chunk is a UTF-8 JSON event envelope of the form

```json
{"event":{"<eventName>":{...payload...}}}
```

The single key under `event` is the discriminator — there is no `type` field. See [Nova Sonic input events](https://docs.aws.amazon.com/nova/latest/userguide/input-events.html) and [output events](https://docs.aws.amazon.com/nova/latest/userguide/output-events.html) for the canonical schemas. Struct definitions live in [`bedrock/live/events.go`](../bedrock/live/events.go).

### Identifier hierarchy

- `promptName` (UUID, client-chosen) — ties a turn together. `bedrock/live` opens one perpetual prompt per session.
- `contentName` (UUID, client-chosen) — marks one content block (audio block, text turn, or tool result). Audio reuses one `contentName` across many chunks; text turns get a fresh `contentName` each call.
- `completionId` / `contentId` (server-assigned) — echoed on output events; we track them in [`readState`](../bedrock/live/mapping.go) to route `audioOutput`/`textOutput`/`toolUse` correctly.

## Event mapping

| Direction | Nova Sonic event | adk-go translation |
|---|---|---|
| → out | `sessionStart` | First event written by `Open`. No fields from `LiveRunConfig` are forwarded yet (Sonic's inference config differs from `GenerateContentConfig`). |
| → out | `promptStart` | Built from `LiveRunConfig.ResponseModalities`, `LiveRunConfig.SpeechConfig.VoiceConfig`, and the tool catalog. |
| → out | `contentStart` + `textInput` + `contentEnd` (role=SYSTEM) | `OpenOptions.SystemInstruction` |
| → out | `contentStart` (role=USER, type=AUDIO) + many `audioInput` + `contentEnd` | `LiveRequest.RealtimeInput = *genai.Blob`. Block opens on first chunk and stays open until an `*genai.ActivityEnd` arrives. |
| → out | `contentStart` (role=USER/ASSISTANT/SYSTEM, type=TEXT) + `textInput` + `contentEnd` | `LiveRequest.Content` with text parts |
| → out | `contentStart` (type=TOOL) + `toolResult` + `contentEnd` | `LiveRequest.Content` with a `FunctionResponse` part — `FunctionResponse.ID` is forwarded as `toolUseId` |
| → out | `promptEnd` + `sessionEnd` | Sent by `Session.Close` |
| ← in | `completionStart` | Suppressed (framing only) |
| ← in | `contentStart` | Tracked in `readState` so subsequent events know the content's role/type |
| ← in | `audioOutput` | `LLMResponse{Content: {Parts: [{InlineData: Blob{MIMEType: "audio/pcm;rate=24000", Data: ...}}]}, Partial: true}`. Base64-decoded on the way through. |
| ← in | `textOutput` (role=USER) | `LLMResponse.InputTranscription` |
| ← in | `textOutput` (role=ASSISTANT) | `LLMResponse.OutputTranscription` |
| ← in | `textOutput` content matches `{ "interrupted" : true }` | `LLMResponse.Interrupted = true` |
| ← in | `toolUse` + `contentEnd(stopReason=TOOL_USE)` | Buffered on `toolUse`, emitted as `LLMResponse{Content: {Parts: [{FunctionCall: ...}]}}` on `contentEnd` |
| ← in | `contentEnd(stopReason=INTERRUPTED)` | `LLMResponse.Interrupted = true` |
| ← in | `completionEnd` | `LLMResponse.TurnComplete = true` with mapped `FinishReason` |
| ← in | `usageEvent` | `LLMResponse.UsageMetadata` |

Each emitted `LLMResponse` is wrapped as a [`session.Event`](https://pkg.go.dev/google.golang.org/adk/session#Event) so the iterator returned by `Open` matches `Runner.RunLive`'s shape verbatim.

## Nova 2 vs Nova 1 Sonic schema notes

Nova 2 Sonic uses the same wire envelope as the original Sonic, with two additive differences this package does not exercise by default:

- **`sessionStart.turnDetectionConfiguration`**: an optional field that lets callers tune `endpointingSensitivity` (LOW / MEDIUM / HIGH). Not currently exposed on `OpenOptions`; Sonic falls back to its default.
- **`contentStart.interactive` for SYSTEM/TEXT**: Nova 2 docs show `interactive: true` for system-prompt content blocks; the original Sonic used `false`. This package still sends `false`. Both versions appear to accept either value in practice; revisit if a Nova 2 schema validation rejects it.

## Tools

Tools are first-class. Pass any number of `*genai.Tool` or `*genai.FunctionDeclaration` values to `OpenOptions.Tools` and the library serializes them into `promptStart.toolConfiguration` per the [Nova 2 Sonic tool spec](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-tool-configuration.html).

**Schema normalization.** `genai.Schema` marshals Gemini-style uppercase type names (`"STRING"`, `"OBJECT"`); JSON Schema (and Sonic) expect lowercase. The shared helper [mappers.FunctionDeclarationSchema](../internal/mappers/tools.go) lowercases types recursively before wire serialization. Both `Converse` and `Live` route through it, so a tool that works on one will work on the other modulo backend feature gaps.

**Auto-execute via `RunAgentLoop`.** Because `bedrock/live` bypasses adk-go's `Runner.RunLive`, the tool round-trip is exposed as a Session method instead:

```go
tools := live.ToolRegistry{
    "get_weather": func(ctx context.Context, args map[string]any) (map[string]any, error) {
        return map[string]any{"temp_f": 72, "condition": "sunny"}, nil
    },
}
err := sess.RunAgentLoop(ctx, events, tools, nil)
```

Why a callback registry rather than `map[string]tool.Tool`? adk-go's `tool.Tool` interface deliberately doesn't expose `Run` — that lives on an unexported `runnableTool` and needs a `tool.Context` we can't synthesize outside the ADK runner. Callbacks let callers wrap any tool (function literal, `tool/functiontool`, `mcptoolset`-produced declarations) without fighting visibility.

**MCP tools.** `mcptoolset` produces `*genai.Tool` values whose schemas live in `ParametersJsonSchema` as a raw JSON object. The mapper's `ParametersJsonSchema` precedence covers this; pass the tool through unchanged.

**Nova Web Grounding (unsupported).** [tools/novagrounding](../tools/novagrounding) maps to a Bedrock Converse `SystemTool` for which Sonic has no equivalent. Passing the grounding sentinel through `OpenOptions.Tools` returns [`ErrUnsupportedTool`](../bedrock/live/mapping.go) from `Open` so failures surface at session start, not mid-conversation.

## Inference parameters

`OpenOptions.GenerateContentConfig` plumbs `Temperature`, `TopP`, and `MaxOutputTokens` into Sonic's `sessionStart.inferenceConfiguration` — the same fields the Converse backend honors. Pass it just like you would `llmagent.Config.GenerateContentConfig`:

```go
temp := float32(0.2)
sess, events, err := live.Open(ctx, api, "", liveCfg, &live.OpenOptions{
    GenerateContentConfig: &genai.GenerateContentConfig{
        Temperature:     &temp,
        MaxOutputTokens: 512,
    },
})
```

`StopSequences` are silently dropped — Sonic's schema doesn't include them. `SystemInstruction` on this config is ignored; use [`OpenOptions.SystemInstruction`](../bedrock/live/session.go) instead so the SYSTEM/TEXT content block framing is correct.

## Converse ↔ Live parity

| Feature | Converse | Live | Notes |
|---|---|---|---|
| Tools (function declarations) | ✅ | ✅ | Shared mapper via [mappers.FunctionDeclarationSchema](../internal/mappers/tools.go) |
| Tool args streaming | n/a (single chunk) | ✅ | Live concatenates across `toolUse` events |
| Tool execution loop | manual / via Runner | [Session.RunAgentLoop](../bedrock/live/agentloop.go) | Live can't go through `Runner.RunLive` (Gemini-locked) |
| Inference params (temperature, topP, maxTokens) | ✅ | ✅ | Pass via `OpenOptions.GenerateContentConfig` |
| StopSequences | ✅ | ❌ — Sonic schema omits | Silently dropped on Live |
| System instruction | `GenerateContentConfig.SystemInstruction` | `OpenOptions.SystemInstruction` | Different surface, same effect |
| Multimodal input (image / video / document) | ✅ | ❌ | Sonic is speech-only |
| Audio input | ❌ | ✅ (16 kHz LPCM) | Live-only |
| Audio output | ❌ | ✅ (24 kHz LPCM) | Live-only |
| Transcripts (input / output) | n/a | ✅ via `LLMResponse.Input/OutputTranscription` | Sonic-only feature |
| Barge-in / interrupt | n/a | ✅ via `LLMResponse.Interrupted` | Sonic-only feature |
| Reasoning blocks ("thoughts") | ✅ | ❌ — Sonic doesn't emit | n/a |
| Citations | ✅ | ❌ — Sonic doesn't emit | n/a |
| Prompt caching (`WithCacheSystemPrompt`) | ✅ | ❌ — no Sonic equivalent | Converse-only |
| Nova Web Grounding | ✅ | ❌ — `ErrUnsupportedTool` | Sonic has no SystemTool |
| MCP tools (via `mcptoolset`) | ✅ | ✅ | Both accept `ParametersJsonSchema` |
| Bedrock Guardrails (preconfigured ID) | ❌ — not exposed | ❌ — not exposed | Cross-cutting gap, separate work |
| Usage metadata (token counts) | ✅ | ✅ | Both populate `UsageMetadata` |
| Unknown server metadata → `CustomMetadata` | `bedrock_*` keys (guardrail trace, service tier, …) | `bedrock_live_unknown_events` passthrough | Sonic doesn't emit guardrail trace today; passthrough surfaces any future event |
| Finish reasons | full Bedrock enum via `mappers.FinishReasonFromStopReasonAndTrace` | Sonic subset via `mappers.FinishReasonFromSonicStop` | Both live in the mappers package |
| OTel tracing | ✅ | ✅ | Both wrap their respective API calls |

## Known gaps vs Gemini Live

- **Session resumption**: Sonic's 8-minute cap surfaces as the underlying HTTP/2 stream closing. `LiveRunConfig.SessionResumption` is accepted but ignored — callers must reconnect themselves and replay state.
- **Affective dialog / proactivity**: no Bedrock equivalent.
- **Audio transcription configs**: `LiveRunConfig.InputAudioTranscription` and `LiveRunConfig.OutputAudioTranscription` are ignored. Sonic always emits transcripts when audio is configured.

## Running the ADK web UI's mic button against Sonic

> **This bridge is a workaround**, not a permanent architecture. adk-go v1.3's `Flow.RunLive` is locked to Gemini's `genai.Client.Live.Connect()`. Until an upstream `LiveBackend` interface lands ([Future work](#future-work)), we override `/api/run_live` with a Bedrock-Sonic-backed WebSocket bridge.

adk-go's launcher (`cmd/launcher/full`) ships an embedded Angular UI plus a `/run_live` WebSocket endpoint. The default handler ([`adkrest.RuntimeAPIController.RunLiveHandler`](https://pkg.go.dev/google.golang.org/adk@v1.3.0/server/adkrest)) calls `Runner.RunLive`, which fails against a Bedrock model. The [`bedrock/live/webbridge`](../bedrock/live/webbridge) package provides a drop-in replacement.

[`examples/bedrock-web-live`](../examples/bedrock-web-live) wires it up:

```go
import (
    "github.com/craigh33/adk-go-bedrock/bedrock/live"
    "github.com/craigh33/adk-go-bedrock/bedrock/live/webbridge"
    // ... adk-go launcher imports ...
)

bridge := webbridge.New(
    live.NewBidiRuntimeAPI(bedrockruntime.NewFromConfig(awsCfg)),
    webbridge.Options{
        SystemInstruction: "You are a friendly voice assistant.",
        Logger:            slog.Default(),
    },
)
bridgeSub := webbridge.NewSublauncher(bridge, webbridge.SublauncherOptions{})

l := universal.NewLauncher(
    console.NewLauncher(),
    web.NewLauncher(
        webui.NewLauncher(),
        bridgeSub,                 // ★ registered BEFORE api so its /api/run_live wins
        a2a.NewLauncher(),
        pubsub.NewLauncher(),
        eventarc.NewLauncher(),
        api.NewLauncher(),         // upstream catchall at root
    ),
)
```

The sublauncher activates on the `live` keyword (`go run . web api webui live`), registers `Path("/api/run_live")`, and bridges between the WebSocket and `live.Session`. Wire shape matches `server/adkrest/internal/models` exactly (binary 16 kHz PCM in, JSON `Event` payloads out) so the embedded UI works unmodified. (`bedrock-web-ui`, which only needs text chat, deliberately stays Converse-only and works in any Bedrock region.)

If you don't use the adk-go launcher, mount `webbridge.New(...)` on any router — it's just an `http.Handler`.

## Future work

1. **Upstream issue / PR** to adk-go proposing a `LiveBackend` interface so `internal/llminternal/base_flow.go` can dispatch to multiple backends. Once that lands we can deprecate `Open` and register the Bedrock model with `Runner.RunLive` directly.
2. **Transparent session resumption** by buffering replayable history and reconnecting on the 8-minute boundary.
3. **Bedrock Guardrails plumbing** on both backends, including the `GuardrailIdentifier` config and the resulting trace data.

## See also

- [`bedrock/live`](../bedrock/live) — implementation
- [`examples/bedrock-live`](../examples/bedrock-live) — runnable example
- [Nova Sonic bidirectional streaming](https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html) — AWS docs
- [adk-go v1.3.0 release notes](https://github.com/google/adk-go/releases/tag/v1.3.0)
