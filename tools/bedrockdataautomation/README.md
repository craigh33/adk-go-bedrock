# bedrockdataautomation

`bedrockdataautomation` provides an ADK tool for Amazon Bedrock Data Automation async analysis.

## Usage

```go
awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}

bdaTool, err := bedrockdataautomation.New(bedrockdataautomation.Config{
    API:                      bedrockdataautomationruntime.NewFromConfig(awsCfg),
    S3:                       s3.NewFromConfig(awsCfg),
    DataAutomationProfileARN: "arn:aws:bedrock:us-east-1:123456789012:data-automation-profile/...",
    DataAutomationProjectARN: "arn:aws:bedrock:us-east-1:123456789012:data-automation-project/...",
    OutputS3URI:              "s3://my-bucket/bda-output",
    InputS3URI:               "s3://my-bucket/bda-input",
})
if err != nil {
    log.Fatal(err)
}
```

The tool is named `analyze_data`. A call must provide exactly one of:

- `s3_uri`: S3 object to analyze
- `artifact_name`: ADK artifact to upload to `InputS3URI` before analysis

Optional call arguments include `project_arn`, `blueprint_arn`, `blueprint_version`, `stage`, `output_s3_uri`, and `result_artifact_name`.

## Behavior

This package uses only the async Bedrock Data Automation runtime. It does not create or manage BDA projects, blueprints, or profiles.

On success the tool returns:

- `status`
- `invocation_arn`
- `input_s3_uri`
- `output_s3_uri`

If `result_artifact_name` is set, the tool downloads `<output_s3_uri>/output.json` and saves it as an ADK artifact. Rich discovery of modality-specific output files is intentionally left out of v1.

## Required IAM Actions

The runtime client needs:

- `bedrock:InvokeDataAutomationAsync`
- `bedrock:GetDataAutomationStatus`

For S3 input staging and result saving, the configured S3 client also needs the relevant `s3:PutObject` and `s3:GetObject` permissions on the input/output buckets. If your BDA project or S3 buckets use KMS keys, include the corresponding KMS permissions.

See [`../../examples/bedrock-data-automation`](../../examples/bedrock-data-automation) for a runnable setup.
