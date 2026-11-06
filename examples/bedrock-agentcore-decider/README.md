# AgentCore Decider

Requires an existing IAM-authenticated Runtime serving Strands Decider's `/invocations` contract, standard AWS credentials, and permission `bedrock-agentcore:InvokeAgentRuntime`.

```sh
export AWS_REGION=eu-west-1
export AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:eu-west-1:123456789012:runtime/your-runtime
make run ARGS='"I was charged twice"'
```

The default mode calls Decider directly and prints its typed assessment. For the ADK tool demonstration, also configure an accessible Bedrock model and its inference permissions:

```sh
export BEDROCK_MODEL_ID=eu.amazon.nova-2-lite-v1:0
make run ARGS='-tool "I was charged twice"'
```

Optional: `AGENTCORE_RUNTIME_QUALIFIER` selects an endpoint; `AGENTCORE_RUNTIME_SESSION_ID` reuses a caller-owned session across processes. Otherwise each client generates one session ID and reuses it for its calls. Cold GPU sessions can take minutes; this example has a five-minute deadline. The example provisions no resources.
