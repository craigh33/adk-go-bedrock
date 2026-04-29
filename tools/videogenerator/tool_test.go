package videogenerator

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/memory"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/toolconfirmation"
	"google.golang.org/genai"
)

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

type fakeAsyncAPI struct {
	startOut *bedrockruntime.StartAsyncInvokeOutput
	startErr error

	lastClientRequestToken *string

	getCalls int
	getOut   []*bedrockruntime.GetAsyncInvokeOutput
	getErr   []error
}

func (f *fakeAsyncAPI) StartAsyncInvoke(
	_ context.Context,
	in *bedrockruntime.StartAsyncInvokeInput,
	_ ...func(*bedrockruntime.Options),
) (*bedrockruntime.StartAsyncInvokeOutput, error) {
	f.lastClientRequestToken = in.ClientRequestToken
	if f.startErr != nil {
		return nil, f.startErr
	}
	return f.startOut, nil
}

func (f *fakeAsyncAPI) GetAsyncInvoke(
	_ context.Context,
	_ *bedrockruntime.GetAsyncInvokeInput,
	_ ...func(*bedrockruntime.Options),
) (*bedrockruntime.GetAsyncInvokeOutput, error) {
	idx := f.getCalls
	f.getCalls++
	if idx < len(f.getErr) && f.getErr[idx] != nil {
		return nil, f.getErr[idx]
	}
	if idx < len(f.getOut) {
		return f.getOut[idx], nil
	}
	return nil, errors.New("unexpected GetAsyncInvoke call")
}

type fakeS3 struct {
	body          []byte
	contentLength *int64
	err           error
}

func (f *fakeS3) GetObject(
	_ context.Context,
	_ *s3.GetObjectInput,
	_ ...func(*s3.Options),
) (*s3.GetObjectOutput, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := &s3.GetObjectOutput{
		Body: io.NopCloser(bytes.NewReader(f.body)),
	}
	if f.contentLength != nil {
		out.ContentLength = f.contentLength
	}
	return out, nil
}

type fakeArtifacts struct {
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
	return nil, errors.New("not implemented")
}

func (f *fakeArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, errors.New("not implemented")
}

type fakeToolContext struct {
	context.Context

	artifacts *fakeArtifacts
}

func (f *fakeToolContext) FunctionCallID() string         { return "test-call-id" }
func (f *fakeToolContext) Actions() *session.EventActions { return &session.EventActions{} }
func (f *fakeToolContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, errors.New("not implemented")
}
func (f *fakeToolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }
func (f *fakeToolContext) RequestConfirmation(string, any) error                { return nil }
func (f *fakeToolContext) Artifacts() agent.Artifacts                           { return f.artifacts }
func (f *fakeToolContext) State() session.State                                 { return nil }
func (f *fakeToolContext) UserContent() *genai.Content                          { return nil }
func (f *fakeToolContext) InvocationID() string                                 { return "inv-1" }
func (f *fakeToolContext) AgentName() string                                    { return "test-agent" }
func (f *fakeToolContext) ReadonlyState() session.ReadonlyState                 { return nil }
func (f *fakeToolContext) UserID() string                                       { return "user-1" }
func (f *fakeToolContext) AppName() string                                      { return "test-app" }
func (f *fakeToolContext) SessionID() string                                    { return "session-1" }
func (f *fakeToolContext) Branch() string                                       { return "" }

var _ tool.Context = (*fakeToolContext)(nil)

func newFakeToolCtx(arts *fakeArtifacts) *fakeToolContext {
	return &fakeToolContext{Context: context.Background(), artifacts: arts}
}

func completedInvokeOutput(s3URI string) *bedrockruntime.GetAsyncInvokeOutput {
	return &bedrockruntime.GetAsyncInvokeOutput{
		InvocationArn: aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/test"),
		Status:        types.AsyncInvokeStatusCompleted,
		OutputDataConfig: &types.AsyncInvokeOutputDataConfigMemberS3OutputDataConfig{
			Value: types.AsyncInvokeS3OutputDataConfig{
				S3Uri: aws.String(s3URI),
			},
		},
	}
}

// ---------------------------------------------------------------------------
// New
// ---------------------------------------------------------------------------

func TestNew_NilAPI(t *testing.T) {
	t.Parallel()
	_, err := New(Config{S3OutputURI: "s3://b"})
	if err == nil {
		t.Fatal("expected error for nil API")
	}
}

func TestNew_EmptyS3OutputURI(t *testing.T) {
	t.Parallel()
	_, err := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "  "})
	if err == nil {
		t.Fatal("expected error for empty S3OutputURI")
	}
}

func TestNew_InvalidS3OutputURI_Scheme(t *testing.T) {
	t.Parallel()
	_, err := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "my-bucket/prefix"})
	if err == nil {
		t.Fatal("expected error for non-s3 scheme")
	}
}

func TestNew_InvalidS3OutputURI_NoBucket(t *testing.T) {
	t.Parallel()
	for _, raw := range []string{"s3://", "s3:///"} {
		t.Run(raw, func(t *testing.T) {
			t.Parallel()
			_, err := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: raw})
			if err == nil {
				t.Fatal("expected error when bucket is missing")
			}
		})
	}
}

func TestNew_MaxArtifactBytesNegative(t *testing.T) {
	t.Parallel()
	_, err := New(Config{
		API:              &fakeAsyncAPI{},
		S3OutputURI:      "s3://b/p",
		MaxArtifactBytes: -1,
	})
	if err == nil {
		t.Fatal("expected error for negative MaxArtifactBytes")
	}
}

func TestNew_InvalidS3OutputURI_LeadingSlashAfterScheme(t *testing.T) {
	t.Parallel()
	_, err := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3:///my-bucket/out"})
	if err == nil {
		t.Fatal("expected error for s3:///… (slash immediately after scheme)")
	}
}

func TestNew_NormalizesProviderNonPositiveSeed(t *testing.T) {
	t.Parallel()
	prov := &ReelProvider{modelID: DefaultReelModelID, Seed: 0}
	tl, err := New(Config{
		API:         &fakeAsyncAPI{},
		S3OutputURI: "s3://bucket",
		Provider:    prov,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	if gt.provider.Seed <= 0 || gt.provider.Seed > maxNovaReelSeed {
		t.Fatalf("expected randomized positive seed, got %d", gt.provider.Seed)
	}

	neg := &ReelProvider{modelID: DefaultReelModelID, Seed: -3}
	tl2, err := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3://bucket", Provider: neg})
	if err != nil {
		t.Fatal(err)
	}
	gt2 := tl2.(*videoGenTool)
	if gt2.provider.Seed <= 0 || gt2.provider.Seed > maxNovaReelSeed {
		t.Fatalf("expected randomized positive seed, got %d", gt2.provider.Seed)
	}
}

func TestNew_OK(t *testing.T) {
	t.Parallel()
	tl, err := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3://bucket"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tl.Name() != "generate_video" {
		t.Errorf("Name() = %q", tl.Name())
	}
	if !tl.IsLongRunning() {
		t.Error("IsLongRunning should be true")
	}
}

func TestNewReelProvider_NegativeSeed_IsRandomInRange(t *testing.T) {
	t.Parallel()
	p := NewReelProvider("", -1)
	if p.Seed == 0 {
		t.Fatal("negative seed should not produce deterministic 0")
	}
	if p.Seed < 1 || p.Seed > maxNovaReelSeed {
		t.Fatalf("Seed = %d, want [1,%d]", p.Seed, maxNovaReelSeed)
	}
}

func TestProcessRequest_PacksDeclaration(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3://b"})
	gt := tl.(*videoGenTool)
	req := &model.LLMRequest{}
	if err := gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req); err != nil {
		t.Fatalf("ProcessRequest: %v", err)
	}
	if len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatal("expected one function declaration")
	}
	if req.Config.Tools[0].FunctionDeclarations[0].Name != "generate_video" {
		t.Error("wrong declaration name")
	}
}

func TestProcessRequest_DuplicateError(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3://b"})
	gt := tl.(*videoGenTool)
	req := &model.LLMRequest{}
	_ = gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req)
	err := gt.ProcessRequest(newFakeToolCtx(&fakeArtifacts{}), req)
	if err == nil {
		t.Fatal("expected duplicate tool error")
	}
}

func TestProviderForArgs_NonIntegerFloat(t *testing.T) {
	t.Parallel()
	base := NewReelProvider("", 99)
	_, err := providerForArgs(base, map[string]any{"seed": 1.2})
	if err == nil {
		t.Fatal("expected error for non-integer float seed")
	}
}

func TestProviderForArgs_IntegerFloat(t *testing.T) {
	t.Parallel()
	base := NewReelProvider("", 99)
	p, err := providerForArgs(base, map[string]any{"seed": float64(1.0)})
	if err != nil {
		t.Fatal(err)
	}
	if p.Seed != 1 {
		t.Fatalf("Seed = %d, want 1", p.Seed)
	}
}

func TestProviderForArgs_NegativeSeed_IsRandomInRange(t *testing.T) {
	t.Parallel()
	base := NewReelProvider("", 99)
	p, err := providerForArgs(base, map[string]any{"seed": -1})
	if err != nil {
		t.Fatal(err)
	}
	if p.Seed == 0 {
		t.Fatal("negative seed should not produce deterministic 0")
	}
	if p.Seed < 1 || p.Seed > maxNovaReelSeed {
		t.Fatalf("Seed = %d, want [1,%d]", p.Seed, maxNovaReelSeed)
	}
}

func TestProviderForArgs_JSONNumber(t *testing.T) {
	t.Parallel()
	base := NewReelProvider("", 99)
	p, err := providerForArgs(base, map[string]any{"seed": json.Number("42")})
	if err != nil {
		t.Fatal(err)
	}
	if p.Seed != 42 {
		t.Fatalf("Seed = %d, want 42", p.Seed)
	}
}

func TestProviderForArgs_FloatMagnitudeTooLarge(t *testing.T) {
	t.Parallel()
	base := NewReelProvider("", 99)
	_, err := providerForArgs(base, map[string]any{"seed": 1e20})
	if err == nil {
		t.Fatal("expected error for float seed magnitude past safe integer range")
	}
}

func TestProviderForArgs_FloatNonFinite(t *testing.T) {
	t.Parallel()
	base := NewReelProvider("", 99)
	for _, tc := range []struct {
		name string
		v    float64
	}{
		{"nan", math.NaN()},
		{"pos_inf", math.Inf(1)},
		{"neg_inf", math.Inf(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := providerForArgs(base, map[string]any{"seed": tc.v})
			if err == nil {
				t.Fatal("expected error for non-finite float seed")
			}
		})
	}
}

func TestRun_ArtifactTooLarge_ContentLength(t *testing.T) {
	t.Parallel()
	arn := aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/job-big")
	huge := defaultMaxArtifactBytes + int64(1)
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{InvocationArn: arn},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			completedInvokeOutput("s3://out-bucket/prefix"),
		},
	}
	s3api := &fakeS3{
		body:          []byte("x"),
		contentLength: aws.Int64(huge),
	}
	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://staging/prefix",
		S3:           s3api,
		PollInterval: time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	_, err = gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": "short"})
	if err == nil {
		t.Fatal("expected error for oversized ContentLength")
	}
	if !strings.Contains(err.Error(), "exceeds maximum artifact size") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRun_ArtifactTooLarge_Stream(t *testing.T) {
	t.Parallel()
	arn := aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/job-stream")
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{InvocationArn: arn},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			completedInvokeOutput("s3://out-bucket/prefix"),
		},
	}
	const maxSmall int64 = 10
	longBody := bytes.Repeat([]byte("x"), int(maxSmall+50))
	s3api := &fakeS3{body: longBody}
	tl, err := New(Config{
		API:              api,
		S3OutputURI:      "s3://staging/prefix",
		S3:               s3api,
		PollInterval:     time.Millisecond,
		MaxWait:          30 * time.Second,
		MaxArtifactBytes: maxSmall,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	_, err = gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": "short"})
	if err == nil {
		t.Fatal("expected error when stream exceeds limit")
	}
	if !strings.Contains(err.Error(), "exceeds maximum artifact size") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRun_ArtifactSaveFails_ReturnsPartialResult(t *testing.T) {
	t.Parallel()
	arn := aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/job-save-fail")
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{InvocationArn: arn},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			completedInvokeOutput("s3://out-bucket/jobs/prefix"),
		},
	}
	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://staging/prefix",
		S3:           &fakeS3{body: []byte("x")},
		PollInterval: time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	wantURI := "s3://out-bucket/jobs/prefix/output.mp4"
	saveErr := errors.New("artifact store unavailable")
	result, err := gt.Run(newFakeToolCtx(&fakeArtifacts{saveErr: saveErr}), map[string]any{
		"prompt": "waves",
	})
	if err == nil || !errors.Is(err, saveErr) {
		t.Fatalf("Run err = %v, want %v", err, saveErr)
	}
	if result == nil {
		t.Fatal("expected partial result map")
	}
	if result["video_s3_uri"] != wantURI {
		t.Errorf("video_s3_uri = %v, want %q", result["video_s3_uri"], wantURI)
	}
	if result["status"] != "success" {
		t.Errorf("status = %v", result["status"])
	}
}

func TestRun_StartAsyncInvoke_ClientRequestTokenMatchesFunctionCallID(t *testing.T) {
	t.Parallel()
	arn := aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/token-test")
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{InvocationArn: arn},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			completedInvokeOutput("s3://out/prefix"),
		},
	}
	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://staging/prefix",
		PollInterval: time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	_, err = gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": "x"})
	if err != nil {
		t.Fatal(err)
	}
	if api.lastClientRequestToken == nil || *api.lastClientRequestToken != "test-call-id" {
		t.Fatalf("ClientRequestToken = %v, want test-call-id", api.lastClientRequestToken)
	}
}

func TestRun_Success_WithS3Download(t *testing.T) {
	t.Parallel()
	arn := aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/job1")
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{InvocationArn: arn},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			{Status: types.AsyncInvokeStatusInProgress},
			completedInvokeOutput("s3://out-bucket/jobs/prefix"),
		},
	}
	arts := &fakeArtifacts{version: 3}
	videoBytes := []byte("fake-mp4-data")
	s3api := &fakeS3{body: videoBytes}

	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://staging-bucket/prefix",
		S3:           s3api,
		PollInterval: 1 * time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)

	result, err := gt.Run(newFakeToolCtx(arts), map[string]any{
		"prompt":    "waves on a beach",
		"file_name": "wave.mp4",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result["status"] != "success" {
		t.Errorf("status = %v", result["status"])
	}
	if result["file_name"] != "wave.mp4" {
		t.Errorf("file_name = %v", result["file_name"])
	}
	if arts.savedName != "wave.mp4" {
		t.Errorf("artifact %q", arts.savedName)
	}
	if arts.savedPart == nil || arts.savedPart.InlineData == nil {
		t.Fatal("missing inline data")
	}
	if string(arts.savedPart.InlineData.Data) != string(videoBytes) {
		t.Error("artifact bytes mismatch")
	}
	if arts.savedPart.InlineData.MIMEType != "video/mp4" {
		t.Errorf("MIME %q", arts.savedPart.InlineData.MIMEType)
	}
}

func TestRun_Success_NoS3Download(t *testing.T) {
	t.Parallel()
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{
			InvocationArn: aws.String("arn:aws:bedrock:us-east-1:123:async-invoke/job2"),
		},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			completedInvokeOutput("s3://out-bucket"),
		},
	}

	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://staging",
		PollInterval: time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)

	result, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{
		"prompt": "sunset",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result["video_s3_uri"] != "s3://out-bucket/output.mp4" {
		t.Errorf("video_s3_uri = %v", result["video_s3_uri"])
	}
	if _, ok := result["file_name"]; ok {
		t.Error("did not expect file_name without S3 download")
	}
	if _, ok := result["artifact"]; ok {
		t.Error("did not expect artifact key without S3 client")
	}
}

func TestRun_Success_UnknownStatusThenCompleted(t *testing.T) {
	t.Parallel()
	arn := "arn:aws:bedrock:us-east-1:123:async-invoke/unknown-then-ok"
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{
			InvocationArn: aws.String(arn),
		},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			{
				InvocationArn: aws.String(arn),
				Status:        types.AsyncInvokeStatus("FutureOrUnknownStatus"),
			},
			{
				InvocationArn: aws.String(arn),
				Status:        types.AsyncInvokeStatusCompleted,
				OutputDataConfig: &types.AsyncInvokeOutputDataConfigMemberS3OutputDataConfig{
					Value: types.AsyncInvokeS3OutputDataConfig{
						S3Uri: aws.String("s3://out-bucket/prefix"),
					},
				},
			},
		},
	}
	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://staging",
		S3:           &fakeS3{body: []byte("x")},
		PollInterval: time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	result, err := gt.Run(newFakeToolCtx(&fakeArtifacts{version: 1}), map[string]any{
		"prompt": "test",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if result["video_s3_uri"] != "s3://out-bucket/prefix/output.mp4" {
		t.Errorf("video_s3_uri = %v", result["video_s3_uri"])
	}
}

func TestRun_EmptyPrompt(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3://b"})
	gt := tl.(*videoGenTool)

	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": ""})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestRun_Failed_EmptyFailureMessage(t *testing.T) {
	t.Parallel()
	arn := "arn:aws:bedrock:us-east-1:123:async-invoke/failed-job"
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{
			InvocationArn: aws.String(arn),
		},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			{
				InvocationArn:  aws.String(arn),
				Status:         types.AsyncInvokeStatusFailed,
				FailureMessage: nil,
			},
		},
	}
	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://b",
		PollInterval: time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	_, err = gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": "a prompt"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), arn) {
		t.Errorf("error should mention invocation ARN: %v", err)
	}
	if strings.TrimSpace(strings.TrimPrefix(err.Error(), "video generation failed")) == "" ||
		strings.HasSuffix(err.Error(), ": ") {
		t.Errorf("unexpected empty or trailing-colon message: %q", err.Error())
	}
}

func TestRun_Failed_WithFailureMessage(t *testing.T) {
	t.Parallel()
	arn := "arn:aws:bedrock:us-east-1:123:async-invoke/failed-with-msg"
	msg := "quota exceeded"
	api := &fakeAsyncAPI{
		startOut: &bedrockruntime.StartAsyncInvokeOutput{
			InvocationArn: aws.String(arn),
		},
		getOut: []*bedrockruntime.GetAsyncInvokeOutput{
			{
				InvocationArn:  aws.String(arn),
				Status:         types.AsyncInvokeStatusFailed,
				FailureMessage: aws.String(msg),
			},
		},
	}
	tl, err := New(Config{
		API:          api,
		S3OutputURI:  "s3://b",
		PollInterval: time.Millisecond,
		MaxWait:      30 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	gt := tl.(*videoGenTool)
	_, err = gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": "x"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), arn) {
		t.Errorf("error should mention invocation ARN: %v", err)
	}
	if !strings.Contains(err.Error(), msg) {
		t.Errorf("error should mention Bedrock message: %v", err)
	}
}

func TestParseS3URI(t *testing.T) {
	t.Parallel()
	bucket, key, err := parseS3URI("s3://my-bucket/folder/sub/output.mp4")
	if err != nil {
		t.Fatal(err)
	}
	if bucket != "my-bucket" || key != "folder/sub/output.mp4" {
		t.Fatalf("got bucket=%q key=%q", bucket, key)
	}

	_, _, err = parseS3URI("s3://only-bucket")
	if err == nil {
		t.Fatal("expected error for uri without key")
	}

	_, _, err = parseS3URI("https://my-bucket.s3.amazonaws.com/key")
	if err == nil {
		t.Fatal("expected error for non-s3 scheme")
	}

	_, _, err = parseS3URI("s3:///my-bucket/key")
	if err == nil {
		t.Fatal("expected error for s3:///… form")
	}
}

func TestClampNovaReelSeed_PositiveNeverZero(t *testing.T) {
	t.Parallel()
	cases := []int64{
		maxNovaReelSeed + 1,
		maxNovaReelSeed * 2,
		math.MaxInt64,
	}
	for _, s := range cases {
		t.Run(strconv.FormatInt(s, 10), func(t *testing.T) {
			t.Parallel()
			got := clampNovaReelSeed(s)
			if got == 0 {
				t.Fatalf("clampNovaReelSeed(%d) = 0", s)
			}
			if got < 1 || got > maxNovaReelSeed {
				t.Fatalf("clampNovaReelSeed(%d) = %d, want [1,%d]", s, got, maxNovaReelSeed)
			}
		})
	}
}

func TestJoinS3Key(t *testing.T) {
	t.Parallel()
	got := joinS3Key("s3://bucket/prefix/", "output.mp4")
	want := "s3://bucket/prefix/output.mp4"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	if joinS3Key("s3://bucket", "output.mp4") != "s3://bucket/output.mp4" {
		t.Fatal("join without prefix")
	}
}

func TestRun_PromptTooLong(t *testing.T) {
	t.Parallel()
	tl, _ := New(Config{API: &fakeAsyncAPI{}, S3OutputURI: "s3://b"})
	gt := tl.(*videoGenTool)

	prompt := strings.Repeat("あ", maxNovaReelPromptRunes+1)
	if utf8.RuneCountInString(prompt) <= maxNovaReelPromptRunes {
		t.Fatal("test setup: prompt should exceed rune limit")
	}

	_, err := gt.Run(newFakeToolCtx(&fakeArtifacts{}), map[string]any{"prompt": prompt})
	if err == nil {
		t.Fatal("expected error for long prompt")
	}
}
