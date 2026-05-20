# bedrock-live example

Bidirectional voice example using Amazon Nova 2 Sonic via the `bedrock/live`
package. Reads a 16 kHz LPCM file, streams it to Sonic in real-time-paced
chunks, prints interleaved input/output transcripts, and writes the model's
24 kHz LPCM reply to disk.

## Prerequisites

- Nova 2 Sonic regional access — currently `us-east-1`, `us-west-2`, or `ap-northeast-1`
- AWS credentials via the default chain
- `AWS_REGION` exported or configured in `~/.aws/config`
- A 16 kHz mono LE16 LPCM audio file. To create one from a WAV:
  ```bash
  ffmpeg -i hello.wav -ar 16000 -ac 1 -f s16le hello-16k.pcm
  ```

## Run

```bash
AWS_REGION=us-east-1 make -C examples/bedrock-live run AUDIO_IN=hello-16k.pcm
```

Or directly:

```bash
AWS_REGION=us-east-1 \
go run ./examples/bedrock-live \
  --audio-in hello-16k.pcm \
  --audio-out /tmp/sonic-out.pcm \
  --system "You are a friendly voice assistant. Reply briefly."
```

Play the response:

```bash
ffplay -f s16le -ar 24000 -ac 1 /tmp/sonic-out.pcm
```

## Notes & limitations

- **Model**: defaults to `amazon.nova-2-sonic-v1:0`. The previous-generation
  `amazon.nova-sonic-v1:0` is wire-compatible and can be selected via
  `BEDROCK_MODEL_ID`, but reaches EOL on 2026-09-14.
- **Audio formats** are raw LPCM, never WAV. Sonic accepts 8/16/24 kHz mono
  LE16 in; the output is 24 kHz mono LE16 by default.
- **8-minute session cap**: enforced server-side. Restart the session for
  longer conversations (transparent resumption is not implemented yet).
- **Mic / speaker** integration is out of scope here — the example is
  file-based so it stays portable and CI-runnable.
- This example does NOT use `adk-go`'s `Runner.RunLive`. See the [Live docs](../../docs/live.md)
  for why and what's different.
