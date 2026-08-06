# agentcorecodeinterpreter

`agentcorecodeinterpreter` provides an ADK tool for Amazon Bedrock AgentCore Code Interpreter.

The tool is named `execute_code`. It starts one AgentCore Code Interpreter session per
tool call, writes requested ADK artifacts into the sandbox, executes code, reads requested
output files, saves generated files back to ADK artifacts, and stops the session.

## Usage

```go
awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}

codeTool, err := agentcorecodeinterpreter.New(agentcorecodeinterpreter.Config{
    API:                       bedrockagentcore.NewFromConfig(awsCfg),
    CodeInterpreterIdentifier: "aws.codeinterpreter.v1",
})
if err != nil {
    log.Fatal(err)
}
```

The tool call must provide:

- `code`: source code to execute

Optional call arguments:

- `language`: `python`, `javascript`, or `typescript`; defaults to `python`
- `runtime`: `python`, `nodejs`, or `deno`
- `input_artifacts`: array of `{artifact_name, path}`; `path` defaults to `artifact_name`
- `output_artifacts`: array of `{path, artifact_name}`; `artifact_name` defaults to the path basename

Artifact `path` values must be relative sandbox paths such as `sales.csv` or
`reports/summary.txt`. Absolute paths like `/tmp/sales.csv` and paths containing
`..` are rejected before the request reaches AgentCore.

Use `aws.codeinterpreter.v1` for the AWS-managed Code Interpreter. Custom Code
Interpreter identifiers and ARNs can also be supplied.

## Behavior

Sandbox execution errors are returned as tool results with `status: "error"`,
`is_error`, `stderr`, `exit_code`, and any available task metadata. Integration failures
such as AWS API errors, artifact load/save errors, invalid arguments, and size-limit
violations are returned as Go errors.

The package does not create or manage AgentCore Code Interpreter resources, VPC settings,
or custom runtime environments.

## Required IAM Actions

The configured client needs:

- `bedrock-agentcore:StartCodeInterpreterSession`
- `bedrock-agentcore:InvokeCodeInterpreter`
- `bedrock-agentcore:StopCodeInterpreterSession`

See [`../../examples/bedrock-agentcore-code-interpreter`](../../examples/bedrock-agentcore-code-interpreter)
for a runnable setup.
