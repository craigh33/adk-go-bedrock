# bedrock-agentcore-memory example

This example wires the [`memory/agentcore`](../../memory/agentcore) service into an
ADK runner. It writes a prior conversation to an Amazon Bedrock AgentCore Memory
resource with `CreateEvent`, then runs an agent (with the `load_memory` /
`preload_memory` tools) in a fresh session so it can recall that history via
`RetrieveMemoryRecords`.

## Prerequisites

- A **provisioned AgentCore Memory resource** with a memory strategy. The strategy's
  namespace template must match `AGENTCORE_NAMESPACE` below. Provisioning is done out
  of band (AWS console, CLI, or the AgentCore control-plane API) — this example only
  uses the data plane.
- `BEDROCK_MODEL_ID` — a Bedrock model ID or inference profile ARN for the agent's LLM.
- `AGENTCORE_MEMORY_ID` — the Memory resource ID.
- `AGENTCORE_NAMESPACE` — the namespace prefix to search, e.g. `/actors/{actorId}/facts`.
  The placeholders `{actorId}`, `{userId}` and `{appName}` are substituted per request
  (by default `{actorId}` resolves to the ADK user ID).
- `AGENTCORE_STRATEGY_ID` — *(optional)* memory strategy ID to filter retrieval.
- AWS credentials via the default chain, and a region (`AWS_REGION` or your profile).
- IAM permissions: `bedrock-agentcore:CreateEvent` and
  `bedrock-agentcore:RetrieveMemoryRecords` (plus `bedrock:InvokeModel*` for the LLM).

## Run

```bash
make -C examples/bedrock-agentcore-memory run
```

Or pass a custom prompt:

```bash
make -C examples/bedrock-agentcore-memory run PROMPT='Where did I eat ramen?'
```

## Note on asynchronous extraction

Long-term memory extraction in AgentCore is **asynchronous**. Events written with
`AddSessionToMemory` are not immediately returned by `SearchMemory`, so on a fresh run
the agent may have nothing to recall. Re-run the recall query after extraction has had
time to complete.
