package mappers

import (
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	agentcoretypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
)

func TestAgentCoreCodeInterpreterStartInput(t *testing.T) {
	t.Parallel()
	in := AgentCoreCodeInterpreterStartInput(AgentCoreCodeInterpreterStartParams{
		CodeInterpreterIdentifier: " interp ",
		ClientToken:               "__tooluse_abc 123!!",
		MaxExecutionTime:          90 * time.Second,
	})
	if aws.ToString(in.CodeInterpreterIdentifier) != "interp" {
		t.Fatalf("identifier = %q", aws.ToString(in.CodeInterpreterIdentifier))
	}
	if aws.ToString(in.Name) != AgentCoreCodeInterpreterDefaultSessionName {
		t.Fatalf("name = %q", aws.ToString(in.Name))
	}
	if token := aws.ToString(in.ClientToken); !strings.HasPrefix(token, "tooluse-abc-123-") ||
		len(token) < agentCoreCodeInterpreterMinClientTokenLength ||
		len(token) > agentCoreCodeInterpreterMaxClientTokenLength {
		t.Fatalf("client token = %q", token)
	}
	if in.SessionTimeoutSeconds == nil || *in.SessionTimeoutSeconds != 90 {
		t.Fatalf("session timeout = %v", in.SessionTimeoutSeconds)
	}
}

func TestAgentCoreCodeInterpreterSessionTimeoutSecondsClamps(t *testing.T) {
	t.Parallel()
	if got := AgentCoreCodeInterpreterSessionTimeoutSeconds(time.Second); got == nil || *got != 60 {
		t.Fatalf("short timeout = %v, want 60", got)
	}
	if got := AgentCoreCodeInterpreterSessionTimeoutSeconds(9 * time.Hour); got == nil || *got != 28800 {
		t.Fatalf("long timeout = %v, want 28800", got)
	}
	if got := AgentCoreCodeInterpreterSessionTimeoutSeconds(0); got != nil {
		t.Fatalf("zero timeout = %v, want nil", got)
	}
}

func TestAgentCoreCodeInterpreterInvokeInputs(t *testing.T) {
	t.Parallel()
	exec := AgentCoreCodeInterpreterExecuteInput(AgentCoreCodeInterpreterInvokeParams{
		CodeInterpreterIdentifier: "interp",
		SessionID:                 "session",
		Code:                      "print('hi')",
		Language:                  "python",
		Runtime:                   "python",
	})
	if exec.Name != agentcoretypes.ToolNameExecuteCode ||
		aws.ToString(exec.CodeInterpreterIdentifier) != "interp" ||
		aws.ToString(exec.SessionId) != "session" ||
		aws.ToString(exec.Arguments.Code) != "print('hi')" ||
		exec.Arguments.Language != agentcoretypes.ProgrammingLanguagePython ||
		exec.Arguments.Runtime != agentcoretypes.LanguageRuntimePython {
		t.Fatalf("execute input = %+v", exec)
	}

	write := AgentCoreCodeInterpreterWriteFilesInput("interp", "session", []AgentCoreCodeInterpreterInputFile{{
		Path:   " data.csv ",
		Text:   "a,b\n1,2\n",
		IsText: true,
	}})
	if write.Name != agentcoretypes.ToolNameWriteFiles ||
		len(write.Arguments.Content) != 1 ||
		aws.ToString(write.Arguments.Content[0].Path) != "data.csv" ||
		aws.ToString(write.Arguments.Content[0].Text) != "a,b\n1,2\n" {
		t.Fatalf("write input = %+v", write)
	}

	read := AgentCoreCodeInterpreterReadFilesInput("interp", "session", []string{" out.txt "})
	if read.Name != agentcoretypes.ToolNameReadFiles ||
		len(read.Arguments.Paths) != 1 ||
		read.Arguments.Paths[0] != "out.txt" {
		t.Fatalf("read input = %+v", read)
	}
}

func TestAgentCoreCodeInterpreterIdentifierNormalizesAWSOwnedARN(t *testing.T) {
	t.Parallel()
	const arn = "arn:aws:bedrock-agentcore:eu-west-2:aws:code-interpreter/aws.codeinterpreter.v1"
	if got := AgentCoreCodeInterpreterIdentifier(arn); got != "aws.codeinterpreter.v1" {
		t.Fatalf("identifier = %q", got)
	}

	custom := "arn:aws:bedrock-agentcore:eu-west-2:123456789012:code-interpreter/custom-a1b2c3d4e5"
	if got := AgentCoreCodeInterpreterIdentifier(custom); got != custom {
		t.Fatalf("custom identifier = %q", got)
	}
}

func TestAgentCoreCodeInterpreterNormalizeLanguageAndRuntime(t *testing.T) {
	t.Parallel()
	got, err := AgentCoreCodeInterpreterNormalizeLanguage("", " Python ", []string{"python"})
	if err != nil {
		t.Fatal(err)
	}
	if got != "python" {
		t.Fatalf("language = %q", got)
	}
	if _, err := AgentCoreCodeInterpreterNormalizeLanguage("typescript", "python", []string{"python"}); err == nil {
		t.Fatal("expected disallowed language error")
	}
	if _, err := AgentCoreCodeInterpreterNormalizeLanguage("ruby", "python", nil); err == nil {
		t.Fatal("expected unsupported language error")
	}
	if got, err := AgentCoreCodeInterpreterNormalizeRuntime(" NodeJS "); err != nil || got != "nodejs" {
		t.Fatalf("runtime = %q, err = %v", got, err)
	}
	if _, err := AgentCoreCodeInterpreterNormalizeRuntime("ruby"); err == nil {
		t.Fatal("expected unsupported runtime error")
	}
}

func TestAgentCoreCodeInterpreterResultMapsStructuredContent(t *testing.T) {
	t.Parallel()
	exitCode := int32(2)
	executionTime := 12.5
	result := agentcoretypes.CodeInterpreterResult{
		StructuredContent: &agentcoretypes.ToolResultStructuredContent{
			Stdout:        aws.String("out"),
			Stderr:        aws.String("err"),
			ExitCode:      aws.Int32(exitCode),
			ExecutionTime: aws.Float64(executionTime),
			TaskId:        aws.String("task-1"),
			TaskStatus:    agentcoretypes.TaskStatusFailed,
		},
	}
	got, artifacts, err := AgentCoreCodeInterpreterResult(result, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if got["status"] != "error" ||
		got["is_error"] != true ||
		got["stdout"] != "out" ||
		got["stderr"] != "err" ||
		got["exit_code"] != exitCode ||
		got["execution_time_ms"] != executionTime ||
		got["task_id"] != "task-1" ||
		got["task_status"] != "failed" {
		t.Fatalf("result map = %+v", got)
	}
	if len(artifacts) != 0 {
		t.Fatalf("artifacts = %+v, want none", artifacts)
	}
}

func TestAgentCoreCodeInterpreterResultMapsContentArtifacts(t *testing.T) {
	t.Parallel()
	result := agentcoretypes.CodeInterpreterResult{
		Content: []agentcoretypes.ContentBlock{
			{
				Type:     agentcoretypes.ContentBlockTypeText,
				Text:     aws.String("hello world"),
				Name:     aws.String("note.txt"),
				MimeType: aws.String("text/plain"),
			},
			{
				Type:     agentcoretypes.ContentBlockTypeEmbeddedResource,
				Name:     aws.String("/tmp/plot.png"),
				MimeType: aws.String("image/png"),
				Resource: &agentcoretypes.ResourceContent{
					Type:     agentcoretypes.ResourceContentTypeBlob,
					Blob:     []byte("png"),
					MimeType: aws.String("image/png"),
				},
			},
		},
	}
	got, artifacts, err := AgentCoreCodeInterpreterResult(result, 1024)
	if err != nil {
		t.Fatal(err)
	}
	content := got["content"].([]map[string]any)
	if len(content) != 2 || content[0]["text"] != "hello world" {
		t.Fatalf("content = %+v", content)
	}
	if len(artifacts) != 1 ||
		artifacts[0].ArtifactName != "plot.png" ||
		artifacts[0].MIMEType != "image/png" ||
		string(artifacts[0].Data) != "png" {
		t.Fatalf("artifacts = %+v", artifacts)
	}
}

func TestAgentCoreCodeInterpreterResultTruncatesText(t *testing.T) {
	t.Parallel()
	got, _, err := AgentCoreCodeInterpreterResult(agentcoretypes.CodeInterpreterResult{
		StructuredContent: &agentcoretypes.ToolResultStructuredContent{Stdout: aws.String("abcdef")},
	}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if got["truncated"] != true || !strings.Contains(got["stdout"].(string), "[truncated]") {
		t.Fatalf("result = %+v", got)
	}
}

func TestAgentCoreCodeInterpreterResultRejectsLargeArtifact(t *testing.T) {
	t.Parallel()
	_, _, err := AgentCoreCodeInterpreterResult(agentcoretypes.CodeInterpreterResult{
		Content: []agentcoretypes.ContentBlock{{
			Type: agentcoretypes.ContentBlockTypeEmbeddedResource,
			Name: aws.String("large.bin"),
			Data: []byte("abcd"),
		}},
	}, 3)
	if err == nil || !strings.Contains(err.Error(), "exceeds maximum output size") {
		t.Fatalf("err = %v", err)
	}
}

func TestAgentCoreCodeInterpreterArtifactName(t *testing.T) {
	t.Parallel()
	if got := AgentCoreCodeInterpreterArtifactName("/tmp/out.txt", 0); got != "out.txt" {
		t.Fatalf("artifact name = %q", got)
	}
	if got := AgentCoreCodeInterpreterArtifactName("", 2); got != "code_interpreter_output_3" {
		t.Fatalf("fallback artifact name = %q", got)
	}
}

func TestAgentCoreCodeInterpreterClientTokenMeetsAgentCoreLength(t *testing.T) {
	t.Parallel()
	token := AgentCoreCodeInterpreterClientToken("tooluse_abc")
	if !strings.HasPrefix(token, "tooluse-abc-") ||
		len(token) < agentCoreCodeInterpreterMinClientTokenLength ||
		len(token) > agentCoreCodeInterpreterMaxClientTokenLength {
		t.Fatalf("token = %q", token)
	}

	long := strings.Repeat("a", 300)
	if got := AgentCoreCodeInterpreterClientToken(long); len(got) < agentCoreCodeInterpreterMinClientTokenLength ||
		len(got) > agentCoreCodeInterpreterMaxClientTokenLength {
		t.Fatalf("long token length = %d", len(got))
	}
}
