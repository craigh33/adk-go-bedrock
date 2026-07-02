# agentcoresession

`agentcoresession` implements `google.golang.org/adk/v2/session.Service` with the Amazon Bedrock AgentCore session APIs exposed by Bedrock Agent Runtime.

## Usage

```go
ctx := context.Background()

awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}

svc, err := agentcoresession.NewWithAPI(
    bedrockagentruntime.NewFromConfig(awsCfg),
    nil,
)
if err != nil {
    log.Fatal(err)
}

created, err := svc.Create(ctx, &session.CreateRequest{
    AppName: "my-app",
    UserID:  "user-123",
})
if err != nil {
    log.Fatal(err)
}

r, err := runner.New(runner.Config{
    AppName:        "my-app",
    Agent:          agent,
    SessionService: svc,
})

sessionID := created.Session.ID()
```

AgentCore assigns session IDs. Do not enable ADK `AutoCreateSession` for this service unless ADK adds support for provider-assigned IDs; create the session first and pass the returned ID to `runner.Run`.

`UserID` is the ADK session user scope, not an AWS or Bedrock identity. It controls ownership checks, `List` filtering, and `user:` state sharing.

## Required IAM Actions

The service needs Bedrock Agent Runtime permissions for:

- `bedrock:CreateSession`
- `bedrock:GetSession`
- `bedrock:UpdateSession`
- `bedrock:ListSessions`
- `bedrock:EndSession`
- `bedrock:DeleteSession`
- `bedrock:CreateInvocation`
- `bedrock:ListInvocations`
- `bedrock:PutInvocationStep`
- `bedrock:ListInvocationSteps`
- `bedrock:GetInvocationStep`

Your agent still needs the usual model invocation permissions, such as `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`.

## Storage Shape

The adapter stores ADK app name, user ID, schema version, and current session state in AgentCore session metadata. It stores each non-partial ADK event as a JSON text payload in an invocation step, preserving the full ADK event.

## Limitations

- AgentCore session APIs assign session IDs; caller-provided ADK session IDs are rejected.
- Session metadata is a string map, so very large ADK state may hit Bedrock metadata limits.
- Listing is client-filtered by adapter metadata because Bedrock `ListSessions` has no app/user filter.
- AgentCore session APIs may have regional availability, retention, and quota limits.

See [`examples/bedrock-agentcore-session`](../../../../examples/bedrock-agentcore-session) for a runnable setup.
