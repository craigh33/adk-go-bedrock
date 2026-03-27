# bedrock-a2a example

This example is based on the ADK A2A example and runs an in-process A2A server backed by Bedrock, then connects to it as a remote agent.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-a2a run
```

The launcher accepts ADK launcher commands/flags. For example:

```bash
make -C examples/bedrock-a2a run ARGS='web api webui'
```
