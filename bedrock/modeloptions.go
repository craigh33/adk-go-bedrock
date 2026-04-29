package bedrock

// ModelOption configures a [Model].
type ModelOption func(*Model)

// WithCacheSystemPrompt appends a Bedrock CachePoint block after the system
// prompt on every request, making the system prompt eligible for prompt
// caching. Has no effect when the request carries no system prompt.
// For more information on Bedrock Prompt Caching see https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html.
func WithCacheSystemPrompt() ModelOption {
	return func(m *Model) { m.cacheSystemPrompt = true }
}

// WithGuardrail sets the default Bedrock guardrail for this model. Per-request
// values override via [ContextWithGuardrail]. When a guardrail is in effect,
// genai safety settings and model armor on the request are ignored (they are
// not mapped to Bedrock).
func WithGuardrail(cfg GuardrailConfig) ModelOption {
	return func(m *Model) {
		c := cfg
		m.guardrail = &c
	}
}
