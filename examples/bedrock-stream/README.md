# bedrock-stream example

This example calls `GenerateContent(..., stream=true)` directly and prints partial and final responses.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-stream run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-stream run PROMPT='Write three bullet points about event streams.'
```
