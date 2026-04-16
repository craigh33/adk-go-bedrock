# bedrock-prompt-cache example

This example shows how to use the `ModelOptions` and namely prompt caching to increase token efficiency.
For more on Bedrock prompt caching, see the [Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-prompt-cache run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-prompt-cache run
```
