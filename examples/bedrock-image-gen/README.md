# bedrock-image-gen example

This example runs an ADK agent that uses the `imagegenerator` tool to generate images via Amazon Nova Canvas (`amazon.nova-canvas-v1:0` by default). The agent receives a text prompt, invokes Bedrock `InvokeModel` to generate the image, and saves the result as an artifact.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a conversational Bedrock model ID (e.g. `eu.amazon.nova-2-lite-v1:0`)
- Nova Canvas enabled in your Bedrock account for the default image model (`amazon.nova-canvas-v1:0`)
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-image-gen run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-image-gen run PROMPT='Generate an image of a sunset over the ocean'
```
