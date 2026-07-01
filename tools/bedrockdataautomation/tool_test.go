package bedrockdataautomation

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	bdaruntime "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime"
	bdatypes "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

type fakeRuntimeAPI struct {
	invokeOut *bdaruntime.InvokeDataAutomationAsyncOutput
	invokeErr error

	lastInvoke *bdaruntime.InvokeDataAutomationAsyncInput
	lastGet    *bdaruntime.GetDataAutomationStatusInput
	getCalls   int
	getOut     []*bdaruntime.GetDataAutomationStatusOutput
	getErr     []error
}

func (f *fakeRuntimeAPI) InvokeDataAutomationAsync(
	_ context.Context,
	in *bdaruntime.InvokeDataAutomationAsyncInput,
	_ ...func(*bdaruntime.Options),
) (*bdaruntime.InvokeDataAutomationAsyncOutput, error) {
	f.lastInvoke = in
	if f.invokeErr != nil {
		return nil, f.invokeErr
	}
	return f.invokeOut, nil
}

func (f *fakeRuntimeAPI) GetDataAutomationStatus(
	_ context.Context,
	in *bdaruntime.GetDataAutomationStatusInput,
	_ ...func(*bdaruntime.Options),
) (*bdaruntime.GetDataAutomationStatusOutput, error) {
	f.lastGet = in
	idx := f.getCalls
	f.getCalls++
	if idx < len(f.getErr) && f.getErr[idx] != nil {
		return nil, f.getErr[idx]
	}
	if idx < len(f.getOut) {
		return f.getOut[idx], nil
	}
	return nil, errors.New("unexpected GetDataAutomationStatus call")
}

type fakeS3 struct {
	objects map[string][]byte

	lastPutBucket      string
	lastPutKey         string
	lastPutBody        []byte
	lastPutContentType *string
	putErr             error

	lastGetBucket string
	lastGetKey    string
	getErr        error
	contentLength *int64
}

func (f *fakeS3) PutObject(
	_ context.Context,
	in *s3.PutObjectInput,
	_ ...func(*s3.Options),
) (*s3.PutObjectOutput, error) {
	f.lastPutBucket = aws.ToString(in.Bucket)
	f.lastPutKey = aws.ToString(in.Key)
	f.lastPutContentType = in.ContentType
	if in.Body != nil {
		f.lastPutBody, _ = io.ReadAll(in.Body)
	}
	if f.putErr != nil {
		return nil, f.putErr
	}
	if f.objects == nil {
		f.objects = map[string][]byte{}
	}
	f.objects[f.lastPutBucket+"/"+f.lastPutKey] = append([]byte(nil), f.lastPutBody...)
	return &s3.PutObjectOutput{}, nil
}

func (f *fakeS3) GetObject(
	_ context.Context,
	in *s3.GetObjectInput,
	_ ...func(*s3.Options),
) (*s3.GetObjectOutput, error) {
	f.lastGetBucket = aws.ToString(in.Bucket)
	f.lastGetKey = aws.ToString(in.Key)
	if f.getErr != nil {
		return nil, f.getErr
	}
	data, ok := f.objects[f.lastGetBucket+"/"+f.lastGetKey]
	if !ok {
		return nil, errors.New("missing fake s3 object")
	}
	out := &s3.GetObjectOutput{Body: io.NopCloser(bytes.NewReader(data))}
	if f.contentLength != nil {
		out.ContentLength = f.contentLength
	}
	return out, nil
}

type fakeArtifacts struct {
	loadPart *genai.Part
	loadErr  error

	savedName string
	savedPart *genai.Part
	saveErr   error
	version   int64
}

func (f *fakeArtifacts) Save(_ context.Context, name string, data *genai.Part) (*artifact.SaveResponse, error) {
	f.savedName = name
	f.savedPart = data
	if f.saveErr != nil {
		return nil, f.saveErr
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

	artifacts      *fakeArtifacts
	functionCallID string
}

func (f *fakeToolContext) FunctionCallID() string {
	if f.functionCallID == "" {
		return "tooluse_test"
	}
	return f.functionCallID
}
func (f *fakeToolContext) Artifacts() agent.Artifacts { return f.artifacts }

var _ agent.Context = (*fakeToolContext)(nil)

func newFakeToolCtx(arts *fakeArtifacts) *fakeToolContext {
	return &fakeToolContext{
		StrictContextMock: agent.StrictContextMock{Ctx: context.Background()},
		artifacts:         arts,
	}
}

func successStatus(outputS3URI string) *bdaruntime.GetDataAutomationStatusOutput {
	return &bdaruntime.GetDataAutomationStatusOutput{
		Status: bdatypes.AutomationJobStatusSuccess,
		OutputConfiguration: &bdatypes.OutputConfiguration{
			S3Uri: aws.String(outputS3URI),
		},
	}
}

func newTestTool(t *testing.T, api *fakeRuntimeAPI, s3api S3API) *dataAutomationTool {
	t.Helper()
	tl, err := New(Config{
		API:                      api,
		S3:                       s3api,
		DataAutomationProfileARN: "arn:aws:bedrock:us-east-1:123456789012:data-automation-profile/profile",
		DataAutomationProjectARN: "arn:aws:bedrock:us-east-1:123456789012:data-automation-project/default",
		OutputS3URI:              "s3://out/default",
		InputS3URI:               "s3://input/staging",
		PollInterval:             time.Millisecond,
		MaxWait:                  30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	return tl.(*dataAutomationTool)
}

func TestNewValidation(t *testing.T) {
	t.Parallel()
	_, err := New(Config{DataAutomationProfileARN: "profile", OutputS3URI: "s3://out"})
	if err == nil {
		t.Fatal("expected nil API error")
	}
	_, err = New(Config{API: &fakeRuntimeAPI{}, OutputS3URI: "s3://out"})
	if err == nil {
		t.Fatal("expected missing profile error")
	}
	_, err = New(Config{API: &fakeRuntimeAPI{}, DataAutomationProfileARN: "profile", OutputS3URI: "out"})
	if err == nil {
		t.Fatal("expected invalid output URI error")
	}
	_, err = New(Config{
		API:                      &fakeRuntimeAPI{},
		DataAutomationProfileARN: "profile",
		OutputS3URI:              "s3://out",
		InputS3URI:               "input",
	})
	if err == nil {
		t.Fatal("expected invalid input URI error")
	}
	_, err = New(Config{
		API:                      &fakeRuntimeAPI{},
		DataAutomationProfileARN: "profile",
		OutputS3URI:              "s3://out",
		MaxInputArtifactBytes:    -1,
	})
	if err == nil {
		t.Fatal("expected negative MaxInputArtifactBytes error")
	}
}

func TestDeclarationAndProcessRequest(t *testing.T) {
	t.Parallel()
	gt := newTestTool(t, &fakeRuntimeAPI{}, nil)
	if gt.Name() != dataAutomationToolName {
		t.Fatalf("Name() = %q", gt.Name())
	}
	if !gt.IsLongRunning() {
		t.Fatal("IsLongRunning() = false, want true")
	}
	decl := gt.Declaration()
	if decl == nil ||
		decl.Parameters.Properties[paramS3URI] == nil ||
		decl.Parameters.Properties[paramArtifactName] == nil {
		t.Fatalf("declaration missing expected params: %+v", decl)
	}

	req := &model.LLMRequest{}
	if err := gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err != nil {
		t.Fatal(err)
	}
	if len(req.Config.Tools) != 1 || len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("tools = %+v", req.Config.Tools)
	}
	if err := gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err == nil {
		t.Fatal("expected duplicate tool error")
	}
}

func TestRunS3InputBuildsRequestAndPolls(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{
			InvocationArn: aws.String("arn:aws:bedrock:us-east-1:123456789012:data-automation-invocation/test"),
		},
		getOut: []*bdaruntime.GetDataAutomationStatusOutput{
			{Status: bdatypes.AutomationJobStatusInProgress},
			successStatus("s3://out/custom"),
		},
	}
	gt := newTestTool(t, api, nil)
	ctx := newFakeToolCtx(&fakeArtifacts{})
	ctx.functionCallID = "tooluse_abc"
	wantProjectARN := "arn:aws:bedrock:us-east-1:123456789012:data-automation-project/override"
	wantBlueprintARN := "arn:aws:bedrock:us-east-1:123456789012:blueprint/custom"

	out, err := gt.Run(ctx, map[string]any{
		paramS3URI:            "s3://source-bucket/path/doc.pdf",
		paramProjectARN:       wantProjectARN,
		paramBlueprintARN:     wantBlueprintARN,
		paramBlueprintVersion: "3",
		paramStage:            "live",
		paramOutputS3URI:      "s3://out/custom",
	})
	if err != nil {
		t.Fatal(err)
	}
	if out["status"] != "success" || out[paramOutputS3URI] != "s3://out/custom" {
		t.Fatalf("result = %+v", out)
	}
	if aws.ToString(api.lastInvoke.ClientToken) != "tooluse-abc" {
		t.Fatalf("ClientToken = %q", aws.ToString(api.lastInvoke.ClientToken))
	}
	if aws.ToString(api.lastInvoke.InputConfiguration.S3Uri) != "s3://source-bucket/path/doc.pdf" {
		t.Fatalf("input S3 URI = %q", aws.ToString(api.lastInvoke.InputConfiguration.S3Uri))
	}
	if aws.ToString(api.lastInvoke.OutputConfiguration.S3Uri) != "s3://out/custom" {
		t.Fatalf("output S3 URI = %q", aws.ToString(api.lastInvoke.OutputConfiguration.S3Uri))
	}
	if aws.ToString(api.lastInvoke.DataAutomationConfiguration.DataAutomationProjectArn) != wantProjectARN {
		t.Fatalf("project ARN = %q", aws.ToString(api.lastInvoke.DataAutomationConfiguration.DataAutomationProjectArn))
	}
	if api.lastInvoke.DataAutomationConfiguration.Stage != bdatypes.DataAutomationStageLive {
		t.Fatalf("project stage = %q", api.lastInvoke.DataAutomationConfiguration.Stage)
	}
	if len(api.lastInvoke.Blueprints) != 1 ||
		aws.ToString(api.lastInvoke.Blueprints[0].BlueprintArn) != wantBlueprintARN ||
		aws.ToString(api.lastInvoke.Blueprints[0].Version) != "3" ||
		api.lastInvoke.Blueprints[0].Stage != bdatypes.BlueprintStageLive {
		t.Fatalf("blueprints = %+v", api.lastInvoke.Blueprints)
	}
	if api.getCalls != 2 {
		t.Fatalf("getCalls = %d, want 2", api.getCalls)
	}
}

func TestRunArtifactInputStagesToS3(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{
			InvocationArn: aws.String("arn:aws:bedrock:us-east-1:123456789012:data-automation-invocation/artifact"),
		},
		getOut: []*bdaruntime.GetDataAutomationStatusOutput{successStatus("s3://out/default")},
	}
	s3api := &fakeS3{}
	gt := newTestTool(t, api, s3api)
	ctx := newFakeToolCtx(&fakeArtifacts{
		loadPart: genai.NewPartFromBytes([]byte("pdf bytes"), "application/pdf"),
	})
	ctx.functionCallID = "call_1"

	_, err := gt.Run(ctx, map[string]any{paramArtifactName: "doc.pdf"})
	if err != nil {
		t.Fatal(err)
	}
	if s3api.lastPutBucket != "input" || s3api.lastPutKey != "staging/call-1/doc.pdf" {
		t.Fatalf("put bucket/key = %s/%s", s3api.lastPutBucket, s3api.lastPutKey)
	}
	if string(s3api.lastPutBody) != "pdf bytes" {
		t.Fatalf("put body = %q", string(s3api.lastPutBody))
	}
	if s3api.lastPutContentType == nil || *s3api.lastPutContentType != "application/pdf" {
		t.Fatalf("content type = %v", s3api.lastPutContentType)
	}
	if aws.ToString(api.lastInvoke.InputConfiguration.S3Uri) != "s3://input/staging/call-1/doc.pdf" {
		t.Fatalf("input S3 URI = %q", aws.ToString(api.lastInvoke.InputConfiguration.S3Uri))
	}
}

func TestRunRequiresExactlyOneInput(t *testing.T) {
	t.Parallel()
	gt := newTestTool(t, &fakeRuntimeAPI{}, nil)
	if _, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{}); err == nil {
		t.Fatal("expected error for no input")
	}
	if _, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramS3URI:        "s3://b/k",
		paramArtifactName: "doc.pdf",
	}); err == nil {
		t.Fatal("expected error for both inputs")
	}
}

func TestRunArtifactInputRequiresS3AndInputPrefix(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{}
	gt := newTestTool(t, api, nil)
	if _, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramArtifactName: "doc.pdf"}); err == nil {
		t.Fatal("expected S3 required error")
	}

	tl, err := New(Config{
		API:                      api,
		S3:                       &fakeS3{},
		DataAutomationProfileARN: "profile",
		OutputS3URI:              "s3://out",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tl.(*dataAutomationTool).Run(
		newFakeToolCtx(&fakeArtifacts{}),
		map[string]any{paramArtifactName: "doc.pdf"},
	); err == nil {
		t.Fatal("expected InputS3URI required error")
	}
}

func TestRunFailureStatus(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{InvocationArn: aws.String("arn")},
		getOut: []*bdaruntime.GetDataAutomationStatusOutput{{
			Status:       bdatypes.AutomationJobStatusClientError,
			ErrorType:    aws.String("ValidationException"),
			ErrorMessage: aws.String("bad input"),
		}},
	}
	gt := newTestTool(t, api, nil)
	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramS3URI: "s3://b/k"})
	if err == nil ||
		!strings.Contains(err.Error(), "bad input") ||
		!strings.Contains(err.Error(), "ValidationException") {
		t.Fatalf("err = %v", err)
	}
}

func TestRunUnknownStatusFailsFast(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{InvocationArn: aws.String("arn")},
		getOut: []*bdaruntime.GetDataAutomationStatusOutput{{
			Status: bdatypes.AutomationJobStatus("FutureStatus"),
		}},
	}
	gt := newTestTool(t, api, nil)
	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramS3URI: "s3://b/k"})
	if err == nil || !strings.Contains(err.Error(), "FutureStatus") {
		t.Fatalf("err = %v", err)
	}
	if api.getCalls != 1 {
		t.Fatalf("getCalls = %d, want 1", api.getCalls)
	}
}

func TestRunTimeout(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{InvocationArn: aws.String("arn")},
		getOut: []*bdaruntime.GetDataAutomationStatusOutput{
			{Status: bdatypes.AutomationJobStatusInProgress},
			{Status: bdatypes.AutomationJobStatusInProgress},
		},
	}
	tl, err := New(Config{
		API:                      api,
		DataAutomationProfileARN: "profile",
		OutputS3URI:              "s3://out",
		PollInterval:             time.Millisecond,
		MaxWait:                  time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*dataAutomationTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{paramS3URI: "s3://b/k"})
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("err = %v", err)
	}
}

func TestRunContextCancellation(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{InvocationArn: aws.String("arn")},
		getOut: []*bdaruntime.GetDataAutomationStatusOutput{
			{Status: bdatypes.AutomationJobStatusInProgress},
		},
	}
	gt := newTestTool(t, api, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := gt.Run(
		&fakeToolContext{
			StrictContextMock: agent.StrictContextMock{Ctx: ctx},
			artifacts:         &fakeArtifacts{},
		},
		map[string]any{paramS3URI: "s3://b/k"},
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestRunSavesResultArtifact(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{InvocationArn: aws.String("arn")},
		getOut:    []*bdaruntime.GetDataAutomationStatusOutput{successStatus("s3://out/job")},
	}
	s3api := &fakeS3{objects: map[string][]byte{"out/job/output.json": []byte(`{"ok":true}`)}}
	arts := &fakeArtifacts{version: 7}
	gt := newTestTool(t, api, s3api)

	out, err := gt.Run(newFakeToolCtx(arts), map[string]any{
		paramS3URI:              "s3://b/k",
		paramResultArtifactName: "bda-result.json",
	})
	if err != nil {
		t.Fatal(err)
	}
	if s3api.lastGetBucket != "out" || s3api.lastGetKey != "job/output.json" {
		t.Fatalf("get bucket/key = %s/%s", s3api.lastGetBucket, s3api.lastGetKey)
	}
	if arts.savedName != "bda-result.json" ||
		arts.savedPart == nil ||
		string(arts.savedPart.InlineData.Data) != `{"ok":true}` {
		t.Fatalf("saved artifact = %q %+v", arts.savedName, arts.savedPart)
	}
	if out["result_s3_uri"] != "s3://out/job/output.json" || out["result_version"] != int64(7) {
		t.Fatalf("out = %+v", out)
	}
}

func TestRunResultArtifactRequiresS3(t *testing.T) {
	t.Parallel()
	gt := newTestTool(t, &fakeRuntimeAPI{}, nil)
	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramS3URI:              "s3://b/k",
		paramResultArtifactName: "result.json",
	})
	if err == nil {
		t.Fatal("expected S3 required error")
	}
}

func TestRunResultTooLarge(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{
		invokeOut: &bdaruntime.InvokeDataAutomationAsyncOutput{InvocationArn: aws.String("arn")},
		getOut:    []*bdaruntime.GetDataAutomationStatusOutput{successStatus("s3://out/job")},
	}
	huge := int64(11)
	s3api := &fakeS3{
		objects:       map[string][]byte{"out/job/output.json": []byte("x")},
		contentLength: aws.Int64(huge),
	}
	tl, err := New(Config{
		API:                      api,
		S3:                       s3api,
		DataAutomationProfileARN: "profile",
		OutputS3URI:              "s3://out",
		MaxResultBytes:           10,
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := tl.(*dataAutomationTool).Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		paramS3URI:              "s3://b/k",
		paramResultArtifactName: "result.json",
	})
	if err == nil || !strings.Contains(err.Error(), "exceeds maximum result size") {
		t.Fatalf("err = %v", err)
	}
	if out == nil || out[paramOutputS3URI] != "s3://out/job" {
		t.Fatalf("partial out = %+v", out)
	}
}

func TestRunArtifactTooLarge(t *testing.T) {
	t.Parallel()
	api := &fakeRuntimeAPI{}
	tl, err := New(Config{
		API:                      api,
		S3:                       &fakeS3{},
		DataAutomationProfileARN: "profile",
		OutputS3URI:              "s3://out",
		InputS3URI:               "s3://input",
		MaxInputArtifactBytes:    3,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = tl.(*dataAutomationTool).Run(newFakeToolCtx(&fakeArtifacts{
		loadPart: genai.NewPartFromBytes([]byte("four"), "text/plain"),
	}), map[string]any{paramArtifactName: "too-big.txt"})
	if err == nil || !strings.Contains(err.Error(), "exceeds maximum artifact size") {
		t.Fatalf("err = %v", err)
	}
}

func TestParseS3URIAndPollBackoff(t *testing.T) {
	t.Parallel()
	if _, _, err := parseS3URI("s3://bucket/key"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := parseS3URI("s3://bucket"); err == nil {
		t.Fatal("expected missing key error")
	}
	if nextPollBackoff(math.MaxInt64, math.MaxInt64) != time.Duration(math.MaxInt64) {
		t.Fatal("max backoff should stay capped")
	}
}
