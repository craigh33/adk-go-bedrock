# bedrock-live-tool example

End-to-end bidirectional voice + tool example. The model receives 16 kHz LPCM
audio, calls a (fake) `get_weather` tool when the user asks about the weather,
and replies in 24 kHz LPCM audio after consuming the tool result. The library
drives the tool round-trip via [`live.Session.RunAgentLoop`](../../bedrock/live/agentloop.go) —
the same shape `Runner.RunLive` uses for Gemini.

## Prerequisites

- Nova 2 Sonic regional access — currently `us-east-1`, `us-west-2`, or `ap-northeast-1`
- AWS credentials via the default chain
- A 16 kHz mono LE16 LPCM file asking about the weather, e.g.:
  ```bash
  # record a phrase like "What's the weather in Tokyo?" and convert:
  ffmpeg -i weather-ask.wav -ar 16000 -ac 1 -f s16le weather-ask-16k.pcm
  ```

## Run

```bash
AWS_REGION=us-east-1 \
go run ./examples/bedrock-live-tool \
  --audio-in weather-ask-16k.pcm \
  --audio-out /tmp/sonic-tool-out.pcm
```

Expected stdout sequence (interleaved with the audio output file):

```
[user] What's the weather in Tokyo?
[tool-call] get_weather(map[location:Tokyo])
[tool-result] get_weather -> map[condition:sunny location:Tokyo temp_f:72 ...]
[model] In Tokyo it's currently 72 degrees and sunny.
```

Play the response:

```bash
ffplay -f s16le -ar 24000 -ac 1 /tmp/sonic-tool-out.pcm
```

## How it works

1. Define the tool's wire-level signature as a `*genai.FunctionDeclaration` and
   register it under `OpenOptions.Tools`. Sonic sees it in the `promptStart`
   payload.
2. Register the actual Go implementation in a `live.ToolRegistry` map keyed by
   the same tool name.
3. Call `Session.RunAgentLoop(ctx, events, tools, emit)`. The loop reads server
   events, invokes the matching `ToolHandler` whenever a `FunctionCall` arrives,
   `Send`s the `FunctionResponse` back, and returns on `TurnComplete`.

You can pass `nil` for `emit` if you don't need to observe individual events —
the loop still completes the tool round-trip. In this example we use `emit` to
print transcripts and write audio bytes to disk.
