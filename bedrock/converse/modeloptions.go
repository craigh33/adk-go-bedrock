package converse

import (
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// ModelOption configures a [Model].
type ModelOption func(*Model)

// WithCacheSystemPrompt appends a Bedrock CachePoint block after the system
// prompt on every request, making the system prompt eligible for prompt
// caching. Has no effect when the request carries no system prompt.
// For more information on Bedrock Prompt Caching see https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html.
func WithCacheSystemPrompt() ModelOption {
	return func(m *Model) { m.cacheSystemPrompt = true }
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
