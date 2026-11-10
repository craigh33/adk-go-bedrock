package converse

import (
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers"
)

// ModelOption configures a [Model].
type ModelOption func(*Model)

// WithCacheSystemPrompt appends a Bedrock CachePoint block after the system
// prompt on every request, making the system prompt eligible for prompt
// caching. Has no effect when the request carries no system prompt.
// Optionally pass one TTL, such as [types.CacheTTLOneHour], to set how long the
// cache lasts. Not all models support every TTL. Passing more than one is an error.
// Parts made with [DynamicSystemPart] are sent after the cache point.
// For more information on Bedrock Prompt Caching see https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html.
func WithCacheSystemPrompt(ttl ...types.CacheTTL) ModelOption {
	return func(m *Model) {
		m.cachePoint = &types.CachePointBlock{Type: types.CachePointTypeDefault}
		m.cacheTTLCount = len(ttl)
		if len(ttl) > 0 {
			m.cachePoint.Ttl = ttl[0]
		}
	}
}

// DynamicSystemPart returns a system prompt part for text that changes between
// requests, such as the current time. With caching on it is moved to the end of
// the system prompt, after the cache point, so it does not break the cache.
// Add it in a before-model callback, so it comes after ADK-GO's own instructions.
func DynamicSystemPart(text string) *genai.Part {
	return &genai.Part{Text: text, PartMetadata: map[string]any{mappers.PartMetadataKeyAfterCachePoint: true}}
}

// WithGuardrail attaches a preconfigured Bedrock guardrail to every request.
func WithGuardrail(identifier, version string, trace types.GuardrailTrace) ModelOption {
	return func(m *Model) {
		m.guardrailConfigured = true
		m.guardrailIdentifier = strings.TrimSpace(identifier)
		m.guardrailVersion = strings.TrimSpace(version)
		m.guardrailTrace = trace
	}
}
