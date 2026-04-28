# bedrock-nova-grounding example

This example calls Amazon Bedrock **Converse** with Nova Web Grounding enabled via [`tools/novagrounding`](../../tools/novagrounding). This tool is Bedrock-specific: the provider maps the sentinel tool to Converse `SystemTool` name `nova_grounding` and surfaces citation metadata on `genai.Part.PartMetadata`.

## Prerequisites

- **US Bedrock region** (for example `AWS_REGION=us-east-1`) — Web Grounding for Nova is documented for US regions.
- **`BEDROCK_MODEL_ID`** — an inference profile that supports Web Grounding with your account (default in code: `us.amazon.nova-premier-v1:0`). Confirm current IDs in [Amazon Nova Web Grounding](https://docs.aws.amazon.com/nova/latest/userguide/grounding.html).
- **IAM** — access to Bedrock inference and, if needed, `bedrock:InvokeTool` for resource identifier `amazon.nova_grounding` (Converse request payloads use `nova_grounding`).
- AWS credentials via the default chain.

## Run

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.amazon.nova-premier-v1:0
make -C examples/bedrock-nova-grounding run
```

Custom prompt:

```bash
make -C examples/bedrock-nova-grounding run PROMPT='What changed in Fed rates this week?'
```

## What it prints

- The model’s natural-language answer.
- Any structured citation rows under metadata key `bedrock_citations` (URLs, domains, etc.), pretty-printed as JSON.

Grounding and citations may incur additional Bedrock charges.
