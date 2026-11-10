package mappers

import (
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	bdatypes "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime/types"
)

func TestBedrockDataAutomationInvokeInput(t *testing.T) {
	t.Parallel()
	in := BedrockDataAutomationInvokeInput(BedrockDataAutomationInvokeParams{
		DataAutomationProfileARN: " profile ",
		DataAutomationProjectARN: " project ",
		InputS3URI:               " s3://input/doc.pdf ",
		OutputS3URI:              " s3://output ",
		ClientToken:              " token ",
		BlueprintARN:             " blueprint ",
		BlueprintVersion:         " 3 ",
		Stage:                    " live ",
	})

	if aws.ToString(in.DataAutomationProfileArn) != "profile" {
		t.Fatalf("profile ARN = %q", aws.ToString(in.DataAutomationProfileArn))
	}
	if aws.ToString(in.InputConfiguration.S3Uri) != "s3://input/doc.pdf" {
		t.Fatalf("input S3 URI = %q", aws.ToString(in.InputConfiguration.S3Uri))
	}
	if aws.ToString(in.OutputConfiguration.S3Uri) != "s3://output" {
		t.Fatalf("output S3 URI = %q", aws.ToString(in.OutputConfiguration.S3Uri))
	}
	if aws.ToString(in.ClientToken) != "token" {
		t.Fatalf("client token = %q", aws.ToString(in.ClientToken))
	}
	if aws.ToString(in.DataAutomationConfiguration.DataAutomationProjectArn) != "project" {
		t.Fatalf("project ARN = %q", aws.ToString(in.DataAutomationConfiguration.DataAutomationProjectArn))
	}
	if in.DataAutomationConfiguration.Stage != bdatypes.DataAutomationStageLive {
		t.Fatalf("project stage = %q", in.DataAutomationConfiguration.Stage)
	}
	if len(in.Blueprints) != 1 ||
		aws.ToString(in.Blueprints[0].BlueprintArn) != "blueprint" ||
		aws.ToString(in.Blueprints[0].Version) != "3" ||
		in.Blueprints[0].Stage != bdatypes.BlueprintStageLive {
		t.Fatalf("blueprints = %+v", in.Blueprints)
	}
}

func TestBedrockDataAutomationInvokeInputOmitsOptionalBlocks(t *testing.T) {
	t.Parallel()
	in := BedrockDataAutomationInvokeInput(BedrockDataAutomationInvokeParams{
		DataAutomationProfileARN: "profile",
		InputS3URI:               "s3://input/doc.pdf",
		OutputS3URI:              "s3://output",
		ClientToken:              "token",
	})
	if in.DataAutomationConfiguration != nil {
		t.Fatalf("project config = %+v, want nil", in.DataAutomationConfiguration)
	}
	if len(in.Blueprints) != 0 {
		t.Fatalf("blueprints = %+v, want none", in.Blueprints)
	}
}

func TestBedrockDataAutomationStatusMapping(t *testing.T) {
	t.Parallel()
	if !BedrockDataAutomationStatusIsPending(bdatypes.AutomationJobStatusCreated) {
		t.Fatal("Created should be pending")
	}
	if !BedrockDataAutomationStatusIsPending(bdatypes.AutomationJobStatusInProgress) {
		t.Fatal("InProgress should be pending")
	}
	if !BedrockDataAutomationStatusIsTerminal(bdatypes.AutomationJobStatusSuccess) {
		t.Fatal("Success should be terminal")
	}
	if !BedrockDataAutomationStatusIsFailure(bdatypes.AutomationJobStatusClientError) {
		t.Fatal("ClientError should be failure")
	}
	if BedrockDataAutomationStatusIsTerminal(bdatypes.AutomationJobStatus("FutureStatus")) {
		t.Fatal("unknown status should not be terminal")
	}
}

func TestBedrockDataAutomationOutputS3URI(t *testing.T) {
	t.Parallel()
	if got := BedrockDataAutomationOutputS3URI(nil); got != "" {
		t.Fatalf("nil output S3 URI = %q", got)
	}
	got := BedrockDataAutomationOutputS3URI(&bdatypes.OutputConfiguration{S3Uri: aws.String("s3://out")})
	if got != "s3://out" {
		t.Fatalf("output S3 URI = %q", got)
	}
}

func TestBedrockDataAutomationFailureError(t *testing.T) {
	t.Parallel()
	err := BedrockDataAutomationFailureError("arn", aws.String("ValidationException"), aws.String("bad input"))
	if err == nil ||
		!strings.Contains(err.Error(), "ValidationException") ||
		!strings.Contains(err.Error(), "bad input") {
		t.Fatalf("err = %v", err)
	}
	err = BedrockDataAutomationFailureError("arn", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "arn") {
		t.Fatalf("err = %v", err)
	}
}

func TestBedrockDataAutomationClientToken(t *testing.T) {
	t.Parallel()
	got := BedrockDataAutomationClientToken("__tooluse_abc 123!!")
	if got != "tooluse-abc-123" {
		t.Fatalf("got %q", got)
	}
	long := strings.Repeat("a", 300)
	if got := BedrockDataAutomationClientToken(long); len(got) > bedrockDataAutomationMaxClientTokenLength {
		t.Fatalf("long token length = %d", len(got))
	}
	if got := BedrockDataAutomationClientToken(""); got == "" {
		t.Fatal("empty token fallback should not be empty")
	}
}
