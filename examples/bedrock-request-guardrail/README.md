# bedrock-request-guardrail example

This example attaches a preconfigured Bedrock guardrail to one Converse request with `bedrock.WithGuardrail`.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- `BEDROCK_GUARDRAIL_ID` set to a preconfigured Bedrock guardrail identifier
- `BEDROCK_GUARDRAIL_VERSION` set to that guardrail version, for example `DRAFT`
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-request-guardrail run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-request-guardrail run PROMPT='Explain secrets management in one paragraph.'
```
