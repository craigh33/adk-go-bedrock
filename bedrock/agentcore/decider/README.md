# AgentCore Decider

Typed evaluation of Choice, Noul, and Score questions through an existing Strands Decider Runtime. Uses AWS SDK authentication and invocation; no TypeSafe dependency.

## Requirements

- An IAM-authenticated AgentCore Runtime serving Strands Decider over HTTP at
  `/invocations`. Supply the runtime ARN and configure the AWS SDK for the
  region where it is deployed.
- AWS credentials with `bedrock-agentcore:InvokeAgentRuntime` permission for
  that runtime.
- The ADK console example also needs a Bedrock Converse model with tool support,
  `bedrock:InvokeModel` permission, and available inference quota. Set
  `BEDROCK_MODEL_ID` to a model or inference profile available in the configured
  AWS region, and `AGENTCORE_RUNTIME_ARN` to the deployed Decider runtime ARN.

The [Strands Decider deployment sample](https://github.com/Vivek0712/strands-decider-agentcore)
shows how to package the model in an ARM64 container for an AgentCore CPU microVM.
Allow time for model initialization on the first invocation of a new session
when setting the request deadline.

## Usage

```go
client, err := decider.New(decider.Config{
    API: bedrockagentcore.NewFromConfig(awsCfg),
    AgentRuntimeARN: runtimeARN,
})
if err != nil {
    return err
}

var question decider.Question
if err := question.FromNoulQuestion(decider.NoulQuestion{
    Type: "noul", Instructions: "Does this request require immediate attention?",
}); err != nil {
    return err
}
response, err := client.Evaluate(ctx, &decider.Request{
    State: "Production is unavailable",
    Questions: map[string]decider.Question{"urgent": question},
})
```

Questions and answers expose generated union helpers such as `FromChoiceQuestion` and `AsChoiceAnswer`. Supply the explicit question `Type` to select the intended union member. State and instructions accept strings, JSON objects, or arrays. Score rubrics contain 2–10 strings. Images are currently rejected.

The Runtime deployment selects the checkpoint; the request's optional `Model` field is passed through, and the response reports the served model. Confidence and probabilities are returned unchanged; routing and fallback belong to the application.

Each client reuses one Runtime session. Set `RuntimeSessionID` for caller-owned reuse and create distinct clients/sessions for required isolation boundaries. Callers configure AWS SDK retries and use context deadlines. Runtime errors preserve their causes; responses are capped at 8 MiB.

The [ADK example](../../../examples/bedrock-agentcore-decider) shows how to use
this client with the [Decider tool](../../../tools/agentcore/decider).
