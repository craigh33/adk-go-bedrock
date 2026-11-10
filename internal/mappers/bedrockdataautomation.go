package mappers

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	bdaruntime "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime"
	bdatypes "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime/types"
	"github.com/google/uuid"
)

const bedrockDataAutomationMaxClientTokenLength = 256

// BedrockDataAutomationInvokeParams is the tool-neutral input for BDA async invoke mapping.
type BedrockDataAutomationInvokeParams struct {
	DataAutomationProfileARN string
	DataAutomationProjectARN string
	InputS3URI               string
	OutputS3URI              string
	ClientToken              string
	BlueprintARN             string
	BlueprintVersion         string
	Stage                    string
}

// BedrockDataAutomationInvokeInput maps tool args/config into the BDA async runtime input shape.
func BedrockDataAutomationInvokeInput(p BedrockDataAutomationInvokeParams) *bdaruntime.InvokeDataAutomationAsyncInput {
	stage := strings.ToUpper(strings.TrimSpace(p.Stage))
	input := &bdaruntime.InvokeDataAutomationAsyncInput{
		DataAutomationProfileArn: aws.String(strings.TrimSpace(p.DataAutomationProfileARN)),
		InputConfiguration:       &bdatypes.InputConfiguration{S3Uri: aws.String(strings.TrimSpace(p.InputS3URI))},
		OutputConfiguration:      &bdatypes.OutputConfiguration{S3Uri: aws.String(strings.TrimSpace(p.OutputS3URI))},
		ClientToken:              aws.String(strings.TrimSpace(p.ClientToken)),
	}

	projectARN := strings.TrimSpace(p.DataAutomationProjectARN)
	if projectARN != "" {
		input.DataAutomationConfiguration = &bdatypes.DataAutomationConfiguration{
			DataAutomationProjectArn: aws.String(projectARN),
		}
		if stage != "" {
			input.DataAutomationConfiguration.Stage = bdatypes.DataAutomationStage(stage)
		}
	}

	blueprintARN := strings.TrimSpace(p.BlueprintARN)
	if blueprintARN != "" {
		bp := bdatypes.Blueprint{BlueprintArn: aws.String(blueprintARN)}
		if blueprintVersion := strings.TrimSpace(p.BlueprintVersion); blueprintVersion != "" {
			bp.Version = aws.String(blueprintVersion)
		}
		if stage != "" {
			bp.Stage = bdatypes.BlueprintStage(stage)
		}
		input.Blueprints = []bdatypes.Blueprint{bp}
	}
	return input
}

// BedrockDataAutomationOutputS3URI maps a BDA output configuration to its S3 URI.
func BedrockDataAutomationOutputS3URI(cfg *bdatypes.OutputConfiguration) string {
	if cfg == nil {
		return ""
	}
	return aws.ToString(cfg.S3Uri)
}

// BedrockDataAutomationStatusIsTerminal reports whether polling should stop for status.
func BedrockDataAutomationStatusIsTerminal(status bdatypes.AutomationJobStatus) bool {
	return status == bdatypes.AutomationJobStatusSuccess ||
		BedrockDataAutomationStatusIsFailure(status)
}

// BedrockDataAutomationStatusIsFailure reports whether status is a failed terminal BDA status.
func BedrockDataAutomationStatusIsFailure(status bdatypes.AutomationJobStatus) bool {
	return status == bdatypes.AutomationJobStatusClientError ||
		status == bdatypes.AutomationJobStatusServiceError
}

// BedrockDataAutomationStatusIsPending reports whether polling should continue for status.
func BedrockDataAutomationStatusIsPending(status bdatypes.AutomationJobStatus) bool {
	return status == bdatypes.AutomationJobStatusCreated ||
		status == bdatypes.AutomationJobStatusInProgress
}

// BedrockDataAutomationFailureError formats a BDA failure response.
func BedrockDataAutomationFailureError(invocationARN string, errorType, errorMessage *string) error {
	msg := strings.TrimSpace(aws.ToString(errorMessage))
	kind := strings.TrimSpace(aws.ToString(errorType))
	if kind == "" {
		kind = "error"
	}
	if msg == "" {
		return fmt.Errorf("data automation failed with %s (invocation %s)", kind, invocationARN)
	}
	return fmt.Errorf("data automation failed with %s: %s (invocation %s)", kind, msg, invocationARN)
}

// BedrockDataAutomationClientToken maps an ADK function-call ID to BDA's clientToken pattern.
func BedrockDataAutomationClientToken(functionCallID string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range strings.TrimSpace(functionCallID) {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if b.Len() > 0 && !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	s := strings.Trim(b.String(), "-")
	if s == "" {
		return uuid.NewString()
	}
	if len(s) > bedrockDataAutomationMaxClientTokenLength {
		return uuid.NewSHA1(uuid.NameSpaceURL, []byte(s)).String()
	}
	return s
}
