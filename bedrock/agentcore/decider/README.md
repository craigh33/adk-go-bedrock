# AgentCore Decider

Typed evaluation of Choice, Noul, and Score questions through an existing Strands Decider Runtime. Uses AWS SDK authentication and invocation; no TypeSafe dependency.

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

See the [runnable example](../../../examples/bedrock-agentcore-decider) and [contract generation](../../../api/decider). The [ADK tool](../../../tools/agentcore/decider) accepts this client's `Evaluate` interface and fixes questions at construction.
