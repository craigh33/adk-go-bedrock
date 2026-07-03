package agentcorecodeinterpreter

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	agentcoretypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

type fakeAgentCoreAPI struct {
	startOut *bedrockagentcore.StartCodeInterpreterSessionOutput
	startErr error

	invokeOuts []invokeOut
	invokeErr  error

	stopErr error

	calls        []string
	startInput   *bedrockagentcore.StartCodeInterpreterSessionInput
	invokeInputs []*bedrockagentcore.InvokeCodeInterpreterInput
	stopInput    *bedrockagentcore.StopCodeInterpreterSessionInput
}

type invokeOut struct {
	results []agentcoretypes.CodeInterpreterResult
	err     error
}

func (f *fakeAgentCoreAPI) StartCodeInterpreterSession(
	_ context.Context,
	in *bedrockagentcore.StartCodeInterpreterSessionInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.StartCodeInterpreterSessionOutput, error) {
	f.calls = append(f.calls, "StartCodeInterpreterSession")
	f.startInput = in
	if f.startErr != nil {
		return nil, f.startErr
	}
	if f.startOut != nil {
		return f.startOut, nil
	}
	return &bedrockagentcore.StartCodeInterpreterSessionOutput{SessionId: aws.String("session-1")}, nil
}

func (f *fakeAgentCoreAPI) InvokeCodeInterpreter(
	_ context.Context,
	in *bedrockagentcore.InvokeCodeInterpreterInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.InvokeCodeInterpreterOutput, error) {
	f.calls = append(f.calls, "InvokeCodeInterpreter:"+string(in.Name))
	f.invokeInputs = append(f.invokeInputs, in)
	if f.invokeErr != nil {
		return nil, f.invokeErr
	}
	if len(f.invokeOuts) == 0 {
		return nil, errors.New("unexpected InvokeCodeInterpreter call")
	}
	next := f.invokeOuts[0]
	f.invokeOuts = f.invokeOuts[1:]
	if next.err != nil {
		return nil, next.err
	}
	return invokeOutput(next.results), nil
}

func (f *fakeAgentCoreAPI) StopCodeInterpreterSession(
	_ context.Context,
	in *bedrockagentcore.StopCodeInterpreterSessionInput,
	_ ...func(*bedrockagentcore.Options),
) (*bedrockagentcore.StopCodeInterpreterSessionOutput, error) {
	f.calls = append(f.calls, "StopCodeInterpreterSession")
	f.stopInput = in
	if f.stopErr != nil {
		return nil, f.stopErr
	}
	return &bedrockagentcore.StopCodeInterpreterSessionOutput{}, nil
}

type fakeStreamReader struct {
	events    chan agentcoretypes.CodeInterpreterStreamOutput
	err       error
	closeOnce sync.Once
}

func newFakeStreamReader(results []agentcoretypes.CodeInterpreterResult) *fakeStreamReader {
	ch := make(chan agentcoretypes.CodeInterpreterStreamOutput, len(results))
	for _, result := range results {
		ch <- &agentcoretypes.CodeInterpreterStreamOutputMemberResult{Value: result}
	}
	close(ch)
	return &fakeStreamReader{events: ch}
}

func (f *fakeStreamReader) Events() <-chan agentcoretypes.CodeInterpreterStreamOutput {
	return f.events
}

func (f *fakeStreamReader) Close() error {
	f.closeOnce.Do(func() {})
	return f.err
}

func (f *fakeStreamReader) Err() error {
	return f.err
}

func invokeOutput(results []agentcoretypes.CodeInterpreterResult) *bedrockagentcore.InvokeCodeInterpreterOutput {
	out := &bedrockagentcore.InvokeCodeInterpreterOutput{}
	stream := bedrockagentcore.NewInvokeCodeInterpreterEventStream(
		func(es *bedrockagentcore.InvokeCodeInterpreterEventStream) {
			es.Reader = newFakeStreamReader(results)
		},
	)
	field := reflect.ValueOf(out).Elem().FieldByName("eventStream")
	reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Set(reflect.ValueOf(stream))
	return out
}

type fakeArtifacts struct {
	loadPart *genai.Part
	loadErr  error

	savedNames []string
	savedParts []*genai.Part
	saveErr    error
	version    int64
}

func (f *fakeArtifacts) Save(_ context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	f.savedNames = append(f.savedNames, name)
	f.savedParts = append(f.savedParts, data)
	if f.saveErr != nil {
		return nil, f.saveErr
	}
	if f.version == 0 {
		f.version = 1
	}
	return &artifact.SaveResponse{Version: f.version}, nil
}

func (f *fakeArtifacts) List(context.Context) (*artifact.ListResponse, error) {
	return &artifact.ListResponse{}, nil
}

func (f *fakeArtifacts) Load(context.Context, string) (*artifact.LoadResponse, error) {
	if f.loadErr != nil {
		return nil, f.loadErr
	}
	return &artifact.LoadResponse{Part: f.loadPart}, nil
}

func (f *fakeArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, errors.New("not implemented")
}

type fakeToolContext struct {
	agent.StrictContextMock

	artifacts agent.Artifacts
}

func (f *fakeToolContext) FunctionCallID() string     { return "tooluse_abc" }
func (f *fakeToolContext) Artifacts() agent.Artifacts { return f.artifacts }

var _ agent.Context = (*fakeToolContext)(nil)

func newFakeToolCtx(arts agent.Artifacts) *fakeToolContext {
	return &fakeToolContext{
		StrictContextMock: agent.StrictContextMock{Ctx: context.Background()},
		artifacts:         arts,
	}
}

func successResult(stdout string) agentcoretypes.CodeInterpreterResult {
	return agentcoretypes.CodeInterpreterResult{
		StructuredContent: &agentcoretypes.ToolResultStructuredContent{
			Stdout:     aws.String(stdout),
			ExitCode:   aws.Int32(0),
			TaskStatus: agentcoretypes.TaskStatusCompleted,
		},
	}
}

func TestNewValidationAndDefaults(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{CodeInterpreterIdentifier: "id"}); err == nil {
		t.Fatal("expected nil API error")
	}
	if _, err := New(Config{API: &fakeAgentCoreAPI{}}); err == nil {
		t.Fatal("expected missing identifier error")
	}
	if _, err := New(
		Config{API: &fakeAgentCoreAPI{}, CodeInterpreterIdentifier: "id", MaxOutputBytes: -1},
	); err == nil {
		t.Fatal("expected negative MaxOutputBytes error")
	}

	tl, err := New(Config{
		API:                       &fakeAgentCoreAPI{},
		CodeInterpreterIdentifier: " id ",
		AllowedLanguages:          []string{"javascript"},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := tl.(*codeInterpreterTool)
	if got.codeInterpreterIdentifier != "id" ||
		got.defaultLanguage != "javascript" ||
		got.maxExecutionTime != defaultMaxExecutionTime ||
		got.maxInputArtifactBytes != defaultMaxArtifactBytes ||
		got.maxOutputBytes != defaultMaxArtifactBytes {
		t.Fatalf("defaults = %+v", got)
	}
}

func TestDeclarationAndProcessRequest(t *testing.T) {
	t.Parallel()
	tl, err := New(Config{API: &fakeAgentCoreAPI{}, CodeInterpreterIdentifier: "id"})
	if err != nil {
		t.Fatal(err)
	}
	ct := tl.(*codeInterpreterTool)
	decl := ct.Declaration()
	if decl.Name != codeInterpreterToolName ||
		decl.Parameters.Properties[paramCode] == nil ||
		decl.Parameters.Properties[paramInputArtifacts].Items.Properties[paramArtifactName] == nil ||
		decl.Parameters.Properties[paramOutputArtifacts].Items.Properties[paramPath] == nil ||
		len(decl.Parameters.Required) != 1 ||
		decl.Parameters.Required[0] != paramCode {
		t.Fatalf("declaration = %+v", decl)
	}
	if ct.IsLongRunning() {
		t.Fatal("Code Interpreter tool should run synchronously")
	}

	req := &model.LLMRequest{}
	if err := ct.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err != nil {
		t.Fatal(err)
	}
	if len(req.Config.Tools) != 1 || len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools = %+v", req.Config.Tools)
	}
	if err := ct.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err == nil {
		t.Fatal("expected duplicate tool error")
	}
}

func TestRunSuccessWithArtifacts(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{
		invokeOuts: []invokeOut{
			{results: []agentcoretypes.CodeInterpreterResult{successResult("wrote input")}},
			{results: []agentcoretypes.CodeInterpreterResult{successResult("ran code")}},
			{results: []agentcoretypes.CodeInterpreterResult{{
				Content: []agentcoretypes.ContentBlock{{
					Type: agentcoretypes.ContentBlockTypeEmbeddedResource,
					Name: aws.String("summary.txt"),
					Resource: &agentcoretypes.ResourceContent{
						Type: agentcoretypes.ResourceContentTypeText,
						Text: aws.String("total,42\n"),
					},
				}},
			}}},
		},
	}
	arts := &fakeArtifacts{loadPart: genai.NewPartFromText("region,total\nEMEA,42\n")}
	tl, err := New(Config{API: api, CodeInterpreterIdentifier: "interp"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(arts), map[string]any{
		paramCode:     "print('hello')",
		paramLanguage: "python",
		paramInputArtifacts: []any{map[string]any{
			paramArtifactName: "sales.csv",
			paramPath:         "sales.csv",
		}},
		paramOutputArtifacts: []any{map[string]any{
			paramPath:         "summary.txt",
			paramArtifactName: "summary.txt",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCalls := strings.Join([]string{
		"StartCodeInterpreterSession",
		"InvokeCodeInterpreter:writeFiles",
		"InvokeCodeInterpreter:executeCode",
		"InvokeCodeInterpreter:readFiles",
		"StopCodeInterpreterSession",
	}, ",")
	if got := strings.Join(api.calls, ","); got != wantCalls {
		t.Fatalf("calls = %s, want %s", got, wantCalls)
	}
	if aws.ToString(api.startInput.CodeInterpreterIdentifier) != "interp" ||
		!strings.HasPrefix(aws.ToString(api.startInput.ClientToken), "tooluse-abc-") ||
		aws.ToString(api.stopInput.SessionId) != "session-1" {
		t.Fatalf("start/stop = %+v %+v", api.startInput, api.stopInput)
	}
	write := api.invokeInputs[0]
	if write.Arguments.Content[0].Text == nil ||
		aws.ToString(write.Arguments.Content[0].Path) != "sales.csv" {
		t.Fatalf("write input = %+v", write.Arguments.Content[0])
	}
	exec := api.invokeInputs[1]
	if exec.Arguments.Language != agentcoretypes.ProgrammingLanguagePython ||
		aws.ToString(exec.Arguments.Code) != "print('hello')" {
		t.Fatalf("execute input = %+v", exec.Arguments)
	}
	read := api.invokeInputs[2]
	if len(read.Arguments.Paths) != 1 || read.Arguments.Paths[0] != "summary.txt" {
		t.Fatalf("read input = %+v", read.Arguments.Paths)
	}
	if result["status"] != "success" ||
		result["stdout"] != "ran code" ||
		result["session_id"] != "session-1" {
		t.Fatalf("result = %+v", result)
	}
	savedResult, ok := result["artifacts"].([]map[string]any)
	if !ok || len(savedResult) != 1 || savedResult[0]["path"] != "summary.txt" {
		t.Fatalf("artifact result = %+v", result["artifacts"])
	}
	if len(arts.savedNames) != 1 ||
		arts.savedNames[0] != "summary.txt" ||
		arts.savedParts[0].Text != "total,42\n" {
		t.Fatalf("saved artifacts = %+v %+v", arts.savedNames, arts.savedParts)
	}
}

func TestRunExecutionErrorReturnsToolResult(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{invokeOuts: []invokeOut{{results: []agentcoretypes.CodeInterpreterResult{{
		IsError: aws.Bool(true),
		StructuredContent: &agentcoretypes.ToolResultStructuredContent{
			Stderr:   aws.String("boom"),
			ExitCode: aws.Int32(1),
		},
	}}}}}
	tl, err := New(Config{API: api, CodeInterpreterIdentifier: "interp"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), map[string]any{paramCode: "raise Exception()"})
	if err != nil {
		t.Fatal(err)
	}
	if result["status"] != "error" ||
		result["is_error"] != true ||
		result["stderr"] != "boom" ||
		result["exit_code"] != int32(1) {
		t.Fatalf("result = %+v", result)
	}
	if api.calls[len(api.calls)-1] != "StopCodeInterpreterSession" {
		t.Fatalf("calls = %+v", api.calls)
	}
}

func TestRunInputArtifactRequiresArtifactServiceAndStopsSession(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	tl, err := New(Config{API: api, CodeInterpreterIdentifier: "interp"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), map[string]any{
		paramCode: "print('x')",
		paramInputArtifacts: []any{map[string]any{
			paramArtifactName: "sales.csv",
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "artifact service is unavailable") {
		t.Fatalf("err = %v", err)
	}
	if got := strings.Join(api.calls, ","); got != "StartCodeInterpreterSession,StopCodeInterpreterSession" {
		t.Fatalf("calls = %s", got)
	}
}

func TestRunRejectsOversizedInputArtifact(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{}
	arts := &fakeArtifacts{loadPart: genai.NewPartFromBytes([]byte("abcd"), "text/plain")}
	tl, err := New(Config{
		API:                       api,
		CodeInterpreterIdentifier: "interp",
		MaxInputArtifactBytes:     3,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*codeInterpreterTool).Run(newFakeToolCtx(arts), map[string]any{
		paramCode: "print('x')",
		paramInputArtifacts: []any{map[string]any{
			paramArtifactName: "large.txt",
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds maximum input artifact size") {
		t.Fatalf("err = %v", err)
	}
}

func TestRunArtifactSaveErrorReturnsPartialResult(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{invokeOuts: []invokeOut{{results: []agentcoretypes.CodeInterpreterResult{{
		Content: []agentcoretypes.ContentBlock{{
			Type: agentcoretypes.ContentBlockTypeEmbeddedResource,
			Name: aws.String("plot.png"),
			Data: []byte("png"),
		}},
	}}}}}
	arts := &fakeArtifacts{saveErr: errors.New("disk full")}
	tl, err := New(Config{API: api, CodeInterpreterIdentifier: "interp"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(arts), map[string]any{paramCode: "plot()"})
	if err == nil || !strings.Contains(err.Error(), "save artifact") {
		t.Fatalf("err = %v", err)
	}
	if result == nil || result["status"] != "success" {
		t.Fatalf("partial result = %+v", result)
	}
}

func TestParseArtifactArgs(t *testing.T) {
	t.Parallel()
	inputs, err := parseInputArtifactArgs([]any{map[string]any{paramArtifactName: "data.csv"}})
	if err != nil {
		t.Fatal(err)
	}
	if inputs[0].Path != "data.csv" {
		t.Fatalf("input path = %q", inputs[0].Path)
	}
	outputs, err := parseOutputArtifactArgs([]any{map[string]any{paramPath: "reports/out.txt"}})
	if err != nil {
		t.Fatal(err)
	}
	if outputs[0].ArtifactName != "out.txt" {
		t.Fatalf("output artifact name = %q", outputs[0].ArtifactName)
	}
	if _, err := parseOutputArtifactArgs("bad"); err == nil {
		t.Fatal("expected bad output args error")
	}
}

func TestParseArtifactArgsRejectUnsafeSandboxPaths(t *testing.T) {
	t.Parallel()
	if _, err := parseInputArtifactArgs([]any{map[string]any{
		paramArtifactName: "data.csv",
		paramPath:         "/tmp/data.csv",
	}}); err == nil || !strings.Contains(err.Error(), "relative sandbox path") {
		t.Fatalf("absolute input path err = %v", err)
	}
	if _, err := parseOutputArtifactArgs([]any{map[string]any{
		paramPath: "../summary.txt",
	}}); err == nil || !strings.Contains(err.Error(), "must not contain '..'") {
		t.Fatalf("traversal output path err = %v", err)
	}
}

func TestRunRejectsBadArgs(t *testing.T) {
	t.Parallel()
	tl, err := New(Config{API: &fakeAgentCoreAPI{}, CodeInterpreterIdentifier: "interp"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), "bad"); err == nil {
		t.Fatal("expected bad args type error")
	}
	if _, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), map[string]any{paramCode: "   "}); err == nil {
		t.Fatal("expected missing code error")
	}
	if _, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), map[string]any{
		paramCode:     "print(1)",
		paramLanguage: "ruby",
	}); err == nil {
		t.Fatal("expected language error")
	}
}

func TestRunAddsCleanupErrorToSuccessfulResult(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{
		invokeOuts: []invokeOut{{results: []agentcoretypes.CodeInterpreterResult{successResult("ok")}}},
		stopErr:    errors.New("stop failed"),
	}
	tl, err := New(Config{API: api, CodeInterpreterIdentifier: "interp"})
	if err != nil {
		t.Fatal(err)
	}
	result, err := tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), map[string]any{paramCode: "print(1)"})
	if err != nil {
		t.Fatal(err)
	}
	if result["cleanup_error"] == nil {
		t.Fatalf("result = %+v", result)
	}
}

func TestRunUsesConfiguredTimeout(t *testing.T) {
	t.Parallel()
	api := &fakeAgentCoreAPI{
		invokeOuts: []invokeOut{{results: []agentcoretypes.CodeInterpreterResult{successResult("ok")}}},
	}
	tl, err := New(Config{
		API:                       api,
		CodeInterpreterIdentifier: "interp",
		MaxExecutionTime:          2 * time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*codeInterpreterTool).Run(newFakeToolCtx(nil), map[string]any{paramCode: "print(1)"})
	if err != nil {
		t.Fatal(err)
	}
	if api.startInput.SessionTimeoutSeconds == nil || *api.startInput.SessionTimeoutSeconds != 120 {
		t.Fatalf("timeout = %v", api.startInput.SessionTimeoutSeconds)
	}
}
