# bedrock-web-live example

ADK local web UI with **bidirectional voice** via Amazon Nova 2 Sonic. Text
chat goes through Bedrock Converse; the mic button goes through
[`bedrock/live.Session`](../../bedrock/live). One process, one launcher, one
embedded UI.

For text-only against any Bedrock chat model in any Bedrock region, use
[`bedrock-web-ui`](../bedrock-web-ui) instead — it doesn't need Sonic and
works in regions Sonic isn't in.

## Prerequisites

- AWS region with Nova 2 Sonic access: `us-east-1`, `us-west-2`, or
  `ap-northeast-1`.
- `AWS_REGION` exported (or set in your profile).
- AWS credentials via the default chain.
- IAM permissions:
  - `bedrock:InvokeModel` (Converse)
  - `bedrock:InvokeModelWithResponseStream` (ConverseStream)
  - `bedrock:InvokeModelWithBidirectionalStream` (Nova Sonic)

The default **text** model is `amazon.nova-2-lite-v1:0` (available in every
Sonic region). Override via `BEDROCK_MODEL_ID` if you have access to a
different model in your Sonic region. The **voice** model is hard-wired to
[`live.DefaultModelID`](../../bedrock/live/session.go) (`amazon.nova-2-sonic-v1:0`).

## Run

```bash
AWS_REGION=us-east-1 make -C examples/bedrock-web-live run
```

Open the printed URL (default `http://localhost:8000`), create a session,
and click the **mic** button. The browser captures audio, downsamples to
16 kHz, and streams to Sonic. Audio output plays through Web Audio API;
transcripts appear in the chat panel.

The Makefile expands to:

```bash
go run . web api webui live
```

The `live` keyword is what activates our Sonic-backed `/run_live` route —
omit it and the mic button falls through to the upstream Gemini-only handler
("model does not support live connection"). The web launcher only calls
`SetupSubrouters` on sublaunchers whose keyword you list on the command line.

## How the voice path works

adk-go's default `/run_live` WebSocket handler hard-codes a Gemini-only call
into `genai.Client.Live.Connect()`. This example uses
[`bedrock/live/webbridge`](../../bedrock/live/webbridge) — a reusable
`http.Handler` that bridges the WebSocket to `bedrock/live.Session` — wrapped
as a sublauncher that mounts at `/api/run_live` **before** the upstream API
sublauncher, so gorilla mux matches it first.

See [docs/live.md](../../docs/live.md#running-the-adk-web-uis-mic-button-against-sonic)
for the full design rationale and the future-work note about replacing the
override once an upstream `LiveBackend` interface lands in adk-go.

Sublauncher order (earlier wins for matching paths):

```
webui        → static UI assets at /
live ★       → /api/run_live (Nova Sonic WebSocket bridge) ← this example
a2a / pubsub / eventarc → unchanged upstream
api          → catchall for everything else (sessions, /api/run, /api/run_sse, artifacts)
```

The bridge's wire shape matches `server/adkrest/internal/models` exactly
(binary 16 kHz mono LE16 PCM in, JSON events out matching `models.Event`), so
the embedded Angular UI works without modification.

## Notes & limitations

- **No tools on the voice path.** This example passes no
  `webbridge.Options.Tools` — the voice agent runs with system instruction
  only. To enable tool calls on the mic path, populate `Options.Tools` in
  the call to `webbridge.New(...)`; see [`examples/bedrock-live-tool`](../bedrock-live-tool)
  for the tool round-trip pattern via `Session.RunAgentLoop`.
- **8-minute Sonic session cap.** When the bidi stream closes, the UI gets
  a WebSocket close; re-open the mic to start a new session.
- **Separate text vs voice models.** Text uses `BEDROCK_MODEL_ID` (default
  Nova 2 Lite); voice always uses Nova 2 Sonic. They're independent.
