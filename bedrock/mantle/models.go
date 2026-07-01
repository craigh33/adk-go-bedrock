package mantle

import (
	"errors"
	"fmt"
	"strings"
)

// anthropicProviderPrefix is the provider segment every Bedrock Mantle model ID
// carries. Mantle serves only Anthropic models, so this is a stable structural
// invariant rather than a snapshot of a mutable catalog.
const anthropicProviderPrefix = "anthropic."

// regionInferenceProfilePrefixes are the cross-region inference-profile prefixes
// used by Bedrock Converse model IDs (for example "us.anthropic.claude-...").
// Bedrock Mantle expects the plain, region-less form, so these are stripped by
// [NormalizeModelID].
func regionInferenceProfilePrefixes() []string {
	return []string{"us-gov.", "us.", "eu.", "apac.", "global."}
}

// NormalizeModelID converts a Converse-style model ID to the plain form Bedrock
// Mantle expects, stripping any leading cross-region inference-profile prefix
// (e.g. "us.anthropic.claude-sonnet-4-5-..." becomes
// "anthropic.claude-sonnet-4-5-..."). IDs without such a prefix are returned
// trimmed but otherwise unchanged.
func NormalizeModelID(id string) string {
	trimmed := strings.TrimSpace(id)
	for _, prefix := range regionInferenceProfilePrefixes() {
		if rest, ok := strings.CutPrefix(trimmed, prefix); ok {
			return rest
		}
	}
	return trimmed
}

// ValidateModelID reports whether a model ID can be served by Bedrock Mantle.
// It rejects empty IDs and any provider other than Anthropic, since Mantle does
// not front the non-Anthropic models available through Bedrock Converse. The ID
// is normalized before the check, so region-prefixed inference-profile IDs are
// accepted.
func ValidateModelID(id string) error {
	normalized := NormalizeModelID(id)
	if normalized == "" {
		return errors.New("model ID is required")
	}
	if !strings.HasPrefix(normalized, anthropicProviderPrefix) {
		return fmt.Errorf(
			"bedrock Mantle serves only Anthropic models; model ID %q is not supported "+
				"(expected an %q identifier, optionally with a region inference-profile prefix)",
			id, anthropicProviderPrefix,
		)
	}
	return nil
}
