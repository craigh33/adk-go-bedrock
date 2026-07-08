# bedrock-agentcore-code-interpreter

This example runs an ADK agent with the `agentcorecodeinterpreter` tool backed by Amazon
Bedrock AgentCore Code Interpreter. It seeds a small `sales.csv` artifact into an
in-memory artifact service, asks the agent to analyze it with Python, and saves
`summary.txt` back as an ADK artifact.

## Environment

Optional:

- `BEDROCK_MODEL_ID`: Bedrock chat model or inference profile for the ADK agent
  (defaults to `global.amazon.nova-2-lite-v1:0`)
- `AWS_REGION`: AWS region, if your profile does not already set one
- `AGENTCORE_REGION`: AgentCore region, if different from `AWS_REGION`
- `AGENTCORE_CODE_INTERPRETER_ID`: Code Interpreter identifier or ARN
  (defaults to `aws.codeinterpreter.v1`)

AWS credentials are loaded from the default AWS SDK chain.

## Run

```bash
export BEDROCK_MODEL_ID=global.amazon.nova-2-lite-v1:0
export AWS_REGION=eu-west-2
export AGENTCORE_REGION=eu-west-2
export AGENTCORE_CODE_INTERPRETER_ID=aws.codeinterpreter.v1

make -C examples/bedrock-agentcore-code-interpreter run
```

You can pass a custom prompt:

```bash
make -C examples/bedrock-agentcore-code-interpreter run PROMPT="Analyze sales.csv and save summary.txt"
```
