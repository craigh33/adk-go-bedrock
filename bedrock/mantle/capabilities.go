package mantle

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Human-readable names for Converse capabilities that have no Bedrock Mantle
// (Anthropic Messages API) equivalent. Following the Converse path's practice of
// rejecting unsupported inputs early, these are surfaced as an explicit error
// rather than silently dropped.
const (
	capGuardrails       = "GuardrailConfiguration (Bedrock Guardrails)"
	capStructuredOutput = "OutputConfig (structured outputs)"
	capAdditionalFields = "AdditionalModelRequestFields (model-specific fields such as Nova Web Grounding)"
)

func checkUnsupportedCapabilities(in *bedrockruntime.ConverseInput) error {
	return capabilityError(collectUnsupported(
		in.GuardrailConfig != nil,
		in.OutputConfig != nil,
		in.AdditionalModelRequestFields != nil,
	))
}

func checkUnsupportedStreamCapabilities(in *bedrockruntime.ConverseStreamInput) error {
	return capabilityError(collectUnsupported(
		in.GuardrailConfig != nil,
		in.OutputConfig != nil,
		in.AdditionalModelRequestFields != nil,
	))
}

func collectUnsupported(hasGuardrails, hasOutputConfig, hasAdditionalFields bool) []string {
	var unsupported []string
	if hasGuardrails {
		unsupported = append(unsupported, capGuardrails)
	}
	if hasOutputConfig {
		unsupported = append(unsupported, capStructuredOutput)
	}
	if hasAdditionalFields {
		unsupported = append(unsupported, capAdditionalFields)
	}
	return unsupported
}

func capabilityError(unsupported []string) error {
	if len(unsupported) == 0 {
		return nil
	}
	return fmt.Errorf(
		"bedrock Mantle (Anthropic Messages API) does not support these Converse capabilities: %s",
		strings.Join(unsupported, ", "),
	)
}
