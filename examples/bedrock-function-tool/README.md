# bedrock-function-tool example

This example runs an ADK `llmagent` with a `functiontool`-defined tool against the Bedrock Converse provider. The tool uses typed arguments and return values (JSON schema is inferred and sent to the model as `parametersJsonSchema`).

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN (must support tool use)
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-function-tool run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-function-tool run PROMPT='What is the weather in Paris?'
```
