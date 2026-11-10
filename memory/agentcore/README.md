# AgentCore Memory service

`agentcorememory` implements ADK's `memory.Service` with Amazon Bedrock
AgentCore Memory. Provision the Memory resource and at least one memory strategy
before using the service. Pass the resource's full ARN in `Config.MemoryID`;
the strategy's namespace must match `Config.Namespace` or `Config.NamespacePath`.

The service needs AWS credentials and a region for the resource. Use `New` to
load the standard AWS configuration chain, or `NewWithAWSConfig` to reuse an
`aws.Config` already loaded by the application. The configured credentials
require `bedrock-agentcore:CreateEvent` and
`bedrock-agentcore:RetrieveMemoryRecords`.

```go
memSvc, err := agentcorememory.New(ctx, &agentcorememory.Config{
	MemoryID:  os.Getenv("AGENTCORE_MEMORY_ARN"),
	Namespace: "/actors/{actorId}/facts",
})
if err != nil {
	return err
}

r, err := runner.New(runner.Config{
	AppName:        "my-app",
	Agent:          agent,
	SessionService: session.InMemoryService(),
	MemoryService:  memSvc,
})
```

By default, the ADK user ID becomes the AgentCore actor ID. `ActorIDTemplate`
can customize that mapping, and namespace templates can use `{actorId}`,
`{userId}`, and `{appName}`. The resource and namespace are user-scoped
according to the configured identity mapping.

Long-term memory extraction is asynchronous. Events written with
`AddSessionToMemory` may not be immediately available through `SearchMemory`.
See [`../../examples/bedrock-agentcore-memory`](../../examples/bedrock-agentcore-memory)
for a runnable example.
