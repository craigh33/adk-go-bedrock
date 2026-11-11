# Route support requests with AgentCore Decider

This ADK agent sends support requests to the Decider tool and reports the chosen
department and its confidence. The application defines the routing question and
allowed departments; the agent supplies the request text.

Configure AWS credentials for Bedrock inference and
`bedrock-agentcore:InvokeAgentRuntime`. Use an existing IAM-authenticated
Strands Decider Runtime serving `/invocations`.

From the repository root:

```sh
export AWS_REGION=eu-west-1
export BEDROCK_MODEL_ID=eu.amazon.nova-2-lite-v1:0
export AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:eu-west-1:123456789012:runtime/your-runtime

go run ./examples/bedrock-agentcore-decider console
```

Try `I was charged twice for my last invoice` or
`My integration returns a 500 error`. To use the ADK web interface instead:

```sh
go run ./examples/bedrock-agentcore-decider web api webui
```

Set `AGENTCORE_RUNTIME_QUALIFIER` to select a Runtime endpoint, or
`AGENTCORE_RUNTIME_SESSION_ID` to reuse a caller-owned session.
See the [Decider client README](../../bedrock/agentcore/decider) for direct
SDK evaluation without a Bedrock model.
