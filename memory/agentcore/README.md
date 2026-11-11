# AgentCore Memory service

`agentcorememory` implements ADK's `memory.Service` with Amazon Bedrock
AgentCore Memory. Provision the Memory resource and at least one memory strategy
before using the service. Pass the resource's full ARN to the constructor.
Configure search with `WithNamespace` or `WithNamespacePath` to match the strategy.

The service needs AWS credentials and a region for the resource. Use `New` to
load the standard AWS configuration chain, `NewWithAWSConfig` to reuse an
`aws.Config` already loaded by the application, or `NewWithAPI` to pass an
existing AgentCore client. The configured credentials require
`bedrock-agentcore:CreateEvent` and `bedrock-agentcore:RetrieveMemoryRecords`.

```go
memSvc, err := agentcorememory.New(ctx, os.Getenv("AGENTCORE_MEMORY_ARN"),
	agentcorememory.WithNamespace("/actors/{actorId}/facts"),
)
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

By default, the service derives the AgentCore actor ID from the ADK user ID.
`WithActorIDTemplate` can combine `{userId}` and `{appName}` before this mapping.
The service preserves valid AgentCore actor and session IDs except for the
reserved `adk-id-` prefix. IDs that use that prefix or exceed AgentCore's
character or length limits become `adk-id-` followed by a SHA-256 hex digest.

Use `{actorId}` in the search namespace to match the actor ID used when storing
events. The `{userId}` and `{appName}` namespace placeholders insert the original
ADK values, which must already fit the namespace's character and length limits.

Event metadata keeps the original ADK values when they fit AgentCore's metadata
constraints. Otherwise, the service stores them in Base64 when they fit in 256
characters after encoding, or stores a SHA-256 hex digest. The corresponding
`<key>_encoding` metadata field identifies the fallback format.

Long-term memory extraction is asynchronous. Events written with
`AddSessionToMemory` may not be immediately available through `SearchMemory`.
See [`../../examples/bedrock-agentcore-memory`](../../examples/bedrock-agentcore-memory)
for a runnable example.
