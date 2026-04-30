# bedrock-video-gen example

This example runs an ADK agent that uses the `videogenerator` tool to generate short videos via Amazon Nova Reel (`amazon.nova-reel-v1:0` by default). Bedrock writes async output to the S3 location you configure; the tool polls until completion, then downloads `output.mp4` into the artifact service.

## Prerequisites

- `NOVA_REEL_OUTPUT_S3_URI` — S3 URI where Bedrock async invoke may write results (for example `s3://my-bucket/nova-reel-output`). Your IAM principal needs permission for Bedrock async invoke and S3 access appropriate for that location.
- `BEDROCK_MODEL_ID` — conversational model for the agent (for example `eu.amazon.nova-2-lite-v1:0`)
- Nova Reel enabled in Amazon Bedrock for your account; use a [supported region](https://docs.aws.amazon.com/nova/latest/userguide/video-generation.html) (for example `AWS_REGION=us-east-1`)
- AWS credentials configured via the default chain

## Run

```bash
export NOVA_REEL_OUTPUT_S3_URI=s3://your-bucket/your-prefix
export AWS_REGION=us-east-1
make -C examples/bedrock-video-gen run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-video-gen run PROMPT='Generate a video of waves at sunset'
```

Video generation is **slow** (often minutes). The tool is marked long-running for the ADK runner.
