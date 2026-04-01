# bedrock-guardrails example

This example demonstrates how to work with Bedrock guardrails, including safety assessments, content filtering, and guardrail metadata handling in ADK responses.

## Features

- Safety ratings extraction and interpretation
- Guardrail intervention detection
- Custom metadata examination
- Multiple safety category assessment
- Streaming with guardrail metadata
- Mapping Bedrock guardrail data to ADK FinishReason and CustomMetadata

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)
- (Optional) Pre-configured Bedrock guardrail ID and version for custom guardrail testing

## Run

```bash
make -C examples/bedrock-guardrails run
```

## Guardrails Overview

Bedrock guardrails provide content filtering and safety assessments:

### Safety Assessment Categories

The example demonstrates these harm categories:

- **Harassment/Insults** - Offensive language and disrespectful content
- **Hate Speech** - Hateful and discriminatory content
- **Sexually Explicit** - Adult content and sexual material
- **Dangerous Content** - Violence, illegal activities, misconduct
- **Jailbreak Attempts** - Prompt injection and manipulation

### Safety Rating Structure

Each safety rating includes:

- **Category**: The harm category being assessed
- **Blocked**: Whether the content was blocked
- **Probability**: Likelihood level (Low, Medium, High)
- **ProbabilityScore**: Numerical score (0-1)
- **Severity**: Impact level when detected (Low, Medium, High)
- **SeverityScore**: Numerical severity score (0-1)

### Guardrail Metadata Mapping

- **CustomMetadata["safety_ratings"]**: Array of `genai.SafetyRating`
- **CustomMetadata["bedrock_guardrail_trace"]**: Raw guardrail assessment data
- **FinishReason**: Indicates type of guardrail intervention
  - `FinishReasonSafety`: General guardrail block
  - `FinishReasonSPII`: Sensitive personally identifiable information
  - `FinishReasonBlocklist`: Word or phrase filter match
  - `FinishReasonProhibitedContent`: Topic-based prohibition

## Using Custom Guardrails

To enable a pre-configured Bedrock guardrail in your applications:

1. Create a guardrail in AWS Bedrock console
2. Note its Identifier and Version
3. Use the Bedrock Runtime API directly to pass guardrail configuration:
   - The current ADK interface doesn't support request-side guardrail configuration
   - See [Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) for details

## Example: Extracting Safety Ratings

```go
// Generate content
for resp := range llm.GenerateContent(ctx, request, false) {
    if resp.CustomMetadata != nil {
        if ratings, ok := resp.CustomMetadata["safety_ratings"].([]*genai.SafetyRating); ok {
            for _, rating := range ratings {
                fmt.Printf("Category: %s, Blocked: %v, Probability: %s\n",
                    rating.Category, rating.Blocked, rating.Probability)
            }
        }
    }
}
```

## Limitations

- Bedrock guardrails require pre-provisioned AWS resources (guardrail ID and version)
- ADK's generic `SafetySettings` cannot be automatically mapped to Bedrock guardrails
- Guardrail metadata is available on response, but request-side configuration requires Bedrock-native APIs
- Only function-based tools are supported; other tool types are ignored

## More Resources

- [Bedrock Guardrails Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [ADK Safety & Moderation](https://pkg.go.dev/google.golang.org/genai#SafetyRating)
