# bedrock-chat example

This example runs a simple ADK runner using the Bedrock Converse provider.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-chat run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-chat run PROMPT='Summarize static typing in one sentence.'
```
