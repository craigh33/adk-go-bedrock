# bedrock-prompt-cache example

This example shows how to use `ModelOption`, specifically prompt caching, to increase token efficiency.
For more on Bedrock prompt caching, see the [Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html).

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-prompt-cache run
```

## Output

Each answer is followed by a token line:

```
[Q1] prompt=12  candidates=1000  fromCache=0    toCache=2605 total=3617 tokens
[Q2] prompt=13  candidates=943   fromCache=2604 toCache=0    total=3560 tokens
[Q3] prompt=9   candidates=950   fromCache=2605 toCache=0    total=3564 tokens
```

`fromCache` is `UsageMetadata.CachedContentTokenCount` (Bedrock `cacheReadInputTokens`).
`toCache` is the cache write, read from `CustomMetadata["bedrock_cache_write_input_tokens"]`
because `genai` has no field for it. The first request pays to populate the cache;
later requests read the system prompt back from it.

The cache outlives the process. Re-running within the cache TTL shows `toCache=0`
and a non-zero `fromCache` on Q1 as well, because the entry is still warm from the
previous run.
