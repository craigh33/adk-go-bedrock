# bedrock-agentcore-session example

This example runs an ADK runner with `bedrock/agentcore/session/agentcoresession`, so session state and event history are stored in the AgentCore session layer.

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured, for example `AWS_REGION=us-east-1`
- IAM access for Bedrock model invocation and Bedrock Agent Runtime session APIs

## Run

Create a new managed session:

```bash
make -C examples/bedrock-agentcore-session run
```

The program prints a `Session ID`. Resume it on another run:

```bash
AGENTCORE_SESSION_ID='<printed-session-id>' make -C examples/bedrock-agentcore-session run PROMPT='What did I ask last time?'
```

Optional environment variables:

- `ADK_USER_ID`: ADK session user ID; defaults to `local-user`
- `AGENTCORE_SESSION_KMS_KEY_ARN`: KMS key ARN for AgentCore session encryption

This example creates or verifies the session before `runner.Run`. It deliberately leaves `AutoCreateSession` off because AgentCore assigns session IDs.
