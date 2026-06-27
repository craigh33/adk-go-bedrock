# bedrock-data-automation example

This example runs an ADK agent with the `tools/bedrockdataautomation` tool. The tool invokes Amazon Bedrock Data Automation asynchronously and returns the S3 output location.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock chat model ID or inference profile ARN
- `BDA_PROFILE_ARN` set to a Bedrock Data Automation profile ARN
- `BDA_OUTPUT_S3_URI` set to an S3 prefix where BDA may write output
- AWS credentials and `AWS_REGION` configured
- Existing BDA project/blueprint resources if you want custom output

Optional environment variables:

- `BDA_PROJECT_ARN`: default project ARN for the tool
- `BDA_INPUT_S3_URI`: S3 prefix used to stage ADK artifact inputs

## Run

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=eu.amazon.nova-2-lite-v1:0
export BDA_PROFILE_ARN=arn:aws:bedrock:us-east-1:123456789012:data-automation-profile/...
export BDA_OUTPUT_S3_URI=s3://your-bucket/bda-output
make -C examples/bedrock-data-automation run PROMPT='Analyze s3://your-bucket/input/document.pdf and save the JSON as bda-result.json'
```

BDA async output is S3-first. The tool returns the invocation ARN and output S3 URI. If the model sets `result_artifact_name`, the tool attempts to save `<output_s3_uri>/output.json` as an ADK artifact.
