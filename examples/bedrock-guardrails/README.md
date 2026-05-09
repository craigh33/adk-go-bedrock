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

## Request-side configuration (this module)

Pre-provision a guardrail in AWS, then pass its identifier and version to the `bedrock` model:

- **Model default** — set `bedrock.GuardrailConfig` on [`bedrock.Options`](https://pkg.go.dev/github.com/craigh33/adk-go-bedrock/bedrock#Options) when calling [`bedrock.New`](https://pkg.go.dev/github.com/craigh33/adk-go-bedrock/bedrock#New), or use [`bedrock.WithGuardrail`](https://pkg.go.dev/github.com/craigh33/adk-go-bedrock/bedrock#WithGuardrail) with [`bedrock.NewWithAPI`](https://pkg.go.dev/github.com/craigh33/adk-go-bedrock/bedrock#NewWithAPI).
- **Per-request override** — wrap `context.Context` with [`bedrock.ContextWithGuardrail`](https://pkg.go.dev/github.com/craigh33/adk-go-bedrock/bedrock#ContextWithGuardrail); it takes precedence over the model default.

Optional field `Trace` maps to Bedrock’s guardrail trace behavior (`enabled`, `disabled`, `enabled_full`).

Genai `SafetySettings` / `ModelArmorConfig` are **not** translated into Bedrock policies. When a resolved Bedrock guardrail is active, those genai fields are ignored for this provider. If they are set and no Bedrock guardrail is configured, the mapper still errors (same as before).

See [Bedrock guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) for creating guardrails in AWS.

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

- Bedrock guardrails require pre-provisioned AWS resources (guardrail ID and version).
- Genai `SafetySettings` / `ModelArmorConfig` are not mapped to Bedrock; use `bedrock.GuardrailConfig` instead (see above).
- Only function-based tools are supported in this example’s tooling notes; other tool types may be unsupported by Bedrock or this mapper (see main README).

## More Resources

- [Bedrock Guardrails Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)
- [ADK Safety & Moderation](https://pkg.go.dev/google.golang.org/genai#SafetyRating)
