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
