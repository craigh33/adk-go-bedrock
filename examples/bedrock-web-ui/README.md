# bedrock-web-ui example

This example launches the ADK local web UI backed by the Bedrock Converse provider.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-web-ui run
```
