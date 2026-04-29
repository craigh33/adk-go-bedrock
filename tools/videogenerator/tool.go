package videogenerator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/google/uuid"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// DefaultReelModelID is the Bedrock model for [ReelProvider] when modelID is empty.
// See https://docs.aws.amazon.com/nova/latest/userguide/video-generation.html
const DefaultReelModelID = "amazon.nova-reel-v1:0"

const (
	videoGenToolName        = "generate_video"
	videoGenToolDescription = `Generates a short video from a text prompt using Amazon Nova Reel (Bedrock async invoke), stores output in your configured S3 location, and saves the MP4 as an artifact when S3 download is configured.

NOTE: This is a long-running operation (often minutes). Do not call this tool again for the same request until it returns.`

	defaultDurationSeconds  = 6
	defaultFPS              = 24
	defaultDimension        = "1280x720"
	novaReelOutputKey       = "output.mp4"
	maxNovaReelPromptRunes  = 512
	defaultArtifactFileName = "generated_video.mp4"

	maxNovaReelSeed int64 = 2147483646

	// defaultMaxArtifactBytes caps S3 download size when persisting MP4 artifacts (RAM bounded).
	defaultMaxArtifactBytes int64 = 512 << 20 // 512 MiB

	// defaultMaxPollInterval caps exponential backoff between GetAsyncInvoke polls.
	defaultMaxPollInterval = 30 * time.Second

	// jitterPercentRange is the argument to [rand.IntN]; added to jitterPercentFloor yields up to 120 (%).
	jitterPercentRange   = 41
	jitterPercentFloor   = 80
	jitterPercentDivisor = 100

	// maxExactIntegerFloat is the largest integer magnitude where every integer
	// in [-N, N] is exactly representable as float64 (IEEE 754 doubles).
	maxExactIntegerFloat int64 = 9007199254740991 // 2^53 - 1
)

// validateS3OutputURI ensures cfg uses an S3 URI suitable for Bedrock async output (s3://bucket[/prefix]).
func validateS3OutputURI(uri string) error {
	s := strings.TrimSpace(uri)
	if s == "" {
		return errors.New("videogenerator: S3OutputURI is required")
	}
	const prefix = "s3://"
	if !strings.HasPrefix(s, prefix) {
		return fmt.Errorf(
			"videogenerator: S3OutputURI must use scheme %q (e.g. s3://my-bucket/prefix), got %q",
			prefix,
			uri,
		)
	}
	rest := strings.TrimPrefix(s, prefix)
	if strings.HasPrefix(rest, "/") {
		return fmt.Errorf(
			"videogenerator: S3OutputURI must be s3://bucket or s3://bucket/prefix (no '/' immediately after %q), got %q",
			prefix,
			uri,
		)
	}
	if rest == "" {
		return errors.New(
			"videogenerator: S3OutputURI must include a bucket name after s3:// (e.g. s3://my-bucket/prefix)",
		)
	}
	bucket, _, _ := strings.Cut(rest, "/")
	if strings.TrimSpace(bucket) == "" {
		return errors.New(
			"videogenerator: S3OutputURI must include a bucket name after s3:// (e.g. s3://my-bucket/prefix)",
		)
	}
	return nil
}

// AsyncInvokeAPI is the Bedrock Runtime subset needed for Nova Reel (async-only).
type AsyncInvokeAPI interface {
	StartAsyncInvoke(
		ctx context.Context,
		params *bedrockruntime.StartAsyncInvokeInput,
		optFns ...func(*bedrockruntime.Options),
	) (*bedrockruntime.StartAsyncInvokeOutput, error)
	GetAsyncInvoke(
		ctx context.Context,
		params *bedrockruntime.GetAsyncInvokeInput,
		optFns ...func(*bedrockruntime.Options),
	) (*bedrockruntime.GetAsyncInvokeOutput, error)
}

// S3GetObjectAPI downloads completed video objects from S3.
type S3GetObjectAPI interface {
	GetObject(
		ctx context.Context,
		params *s3.GetObjectInput,
		optFns ...func(*s3.Options),
	) (*s3.GetObjectOutput, error)
}

// ReelProvider builds Amazon Nova Reel TEXT_VIDEO payloads for StartAsyncInvoke.
// See https://docs.aws.amazon.com/nova/latest/userguide/video-gen-code-examples.html
type ReelProvider struct {
	modelID string
	Seed    int64
}

// NewReelProvider returns a single-shot text-to-video provider.
// Pass an empty modelID to use [DefaultReelModelID].
// Seed <= 0 (including zero and negatives) selects a random in-range seed; positive seeds are clamped to Nova’s allowed range.
func NewReelProvider(modelID string, seed int64) *ReelProvider {
	if modelID == "" {
		modelID = DefaultReelModelID
	}
	if seed <= 0 {
		seed = randomNovaReelSeed()
	} else {
		seed = clampNovaReelSeed(seed)
	}
	return &ReelProvider{modelID: modelID, Seed: seed}
}

// clampNovaReelSeed maps seeds into Nova’s usable range. Values <= 0 match [NewReelProvider]
// (random in-range seed). Values above max are reduced; positive values never become 0.
func clampNovaReelSeed(s int64) int64 {
	if s <= 0 {
		return randomNovaReelSeed()
	}
	if s <= maxNovaReelSeed {
		return s
	}
	return 1 + (s % maxNovaReelSeed)
}

// randomNovaReelSeed returns a pseudo-random seed in [1, maxNovaReelSeed].
// math/rand/v2 is auto-seeded per process, avoiding collisions when called
// in quick succession (which a [time.Now]-derived seed would suffer).
//
//nolint:gosec // G404: non-cryptographic seed for Bedrock Nova Reel.
func randomNovaReelSeed() int64 {
	return 1 + rand.Int64N(maxNovaReelSeed)
}

func (p *ReelProvider) ModelID() string { return p.modelID }

func (p *ReelProvider) modelInput(prompt string) any {
	return map[string]any{
		"taskType": "TEXT_VIDEO",
		"textToVideoParams": map[string]any{
			"text": prompt,
		},
		"videoGenerationConfig": map[string]any{
			"durationSeconds": defaultDurationSeconds,
			"fps":             defaultFPS,
			"dimension":       defaultDimension,
			"seed":            clampNovaReelSeed(p.Seed),
		},
	}
}

// Config configures the video generator tool.
type Config struct {
	API AsyncInvokeAPI

	// S3OutputURI is where Bedrock writes async output (e.g. s3://my-bucket or s3://my-bucket/prefix).
	// Required for StartAsyncInvoke.
	S3OutputURI string

	// Provider builds model input; if nil, a default [NewReelProvider]("", 0) is used.
	// If non-nil with Seed <= 0, [New] replaces it with [NewReelProvider](ModelID(), 0) so the payload gets a random in-range seed.
	Provider *ReelProvider

	// S3 downloads output.mp4 after the job completes. If nil, Run returns video_s3_uri only (no artifact).
	S3 S3GetObjectAPI

	// PollInterval is the initial delay before the first sleep while status is InProgress; backoff doubles until MaxPollInterval. Zero defaults to 5s.
	PollInterval time.Duration
	// MaxPollInterval is the maximum delay between GetAsyncInvoke polls under exponential backoff. Zero defaults to defaultMaxPollInterval.
	MaxPollInterval time.Duration
	// MaxWait is the maximum time to wait for job completion. Zero defaults to 30m.
	MaxWait time.Duration

	// MaxArtifactBytes is the maximum S3 object size to read into memory when saving an MP4 artifact.
	// Zero defaults to defaultMaxArtifactBytes. Must not be negative.
	MaxArtifactBytes int64
}

type videoGenTool struct {
	api              AsyncInvokeAPI
	s3               S3GetObjectAPI
	s3OutputURI      string
	provider         *ReelProvider
	pollInterval     time.Duration
	maxPollInterval  time.Duration
	maxWait          time.Duration
	maxArtifactBytes int64
	decl             *genai.FunctionDeclaration
}

// New creates an ADK tool that generates video via Nova Reel async invoke.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("videogenerator: API is required")
	}
	if err := validateS3OutputURI(cfg.S3OutputURI); err != nil {
		return nil, err
	}
	prov := cfg.Provider
	if prov == nil {
		prov = NewReelProvider("", 0)
	} else if prov.Seed <= 0 {
		prov = NewReelProvider(prov.ModelID(), 0)
	}
	poll := cfg.PollInterval
	if poll <= 0 {
		poll = 5 * time.Second
	}
	if cfg.MaxPollInterval < 0 {
		return nil, errors.New("videogenerator: MaxPollInterval cannot be negative")
	}
	maxPoll := cfg.MaxPollInterval
	if maxPoll == 0 {
		maxPoll = defaultMaxPollInterval
	}
	if poll > maxPoll {
		poll = maxPoll
	}
	maxWait := cfg.MaxWait
	if maxWait <= 0 {
		maxWait = 30 * time.Minute
	}
	if cfg.MaxArtifactBytes < 0 {
		return nil, errors.New("videogenerator: MaxArtifactBytes cannot be negative")
	}
	maxArt := cfg.MaxArtifactBytes
	if maxArt == 0 {
		maxArt = defaultMaxArtifactBytes
	}
	return &videoGenTool{
		api:              cfg.API,
		s3:               cfg.S3,
		s3OutputURI:      strings.TrimSpace(cfg.S3OutputURI),
		provider:         prov,
		pollInterval:     poll,
		maxPollInterval:  maxPoll,
		maxWait:          maxWait,
		maxArtifactBytes: maxArt,
		decl:             newFunctionDeclaration(),
	}, nil
}

func (t *videoGenTool) Name() string {
	return videoGenToolName
}

func (t *videoGenTool) Description() string {
	return videoGenToolDescription
}

// newFunctionDeclaration builds the tool schema once; [videoGenTool.decl] reuses this pointer.
func newFunctionDeclaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        videoGenToolName,
		Description: videoGenToolDescription,
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"prompt": {
					Type: "STRING",
					Description: fmt.Sprintf(
						"Text prompt describing the video (English, max %d characters for single-shot)",
						maxNovaReelPromptRunes,
					),
				},
				"file_name": {
					Type: "STRING",
					Description: fmt.Sprintf(
						"Artifact filename for the downloaded MP4 (e.g. 'clip.mp4'). Defaults to %q.",
						defaultArtifactFileName,
					),
				},
				"seed": {
					Type:        "INTEGER",
					Description: "Optional positive integer seed for reproducibility; omit, zero, or non-positive values select a random seed.",
				},
			},
			Required: []string{"prompt"},
		},
	}
}

func (t *videoGenTool) IsLongRunning() bool {
	return true
}

func (t *videoGenTool) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

func (t *videoGenTool) ProcessRequest(_ tool.Context, req *model.LLMRequest) error {
	if req.Tools == nil {
		req.Tools = make(map[string]any)
	}
	name := t.Name()
	if _, ok := req.Tools[name]; ok {
		return fmt.Errorf("duplicate tool: %q", name)
	}
	req.Tools[name] = t

	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	decl := t.Declaration()
	if decl == nil {
		return nil
	}

	var funcTool *genai.Tool
	for _, gt := range req.Config.Tools {
		if gt != nil && gt.FunctionDeclarations != nil {
			funcTool = gt
			break
		}
	}
	if funcTool == nil {
		req.Config.Tools = append(req.Config.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{decl},
		})
	} else {
		funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, decl)
	}
	return nil
}

func providerForArgs(base *ReelProvider, m map[string]any) (*ReelProvider, error) {
	prov := base
	if seedRaw, ok := m["seed"]; ok && seedRaw != nil {
		switch v := seedRaw.(type) {
		case float64:
			if math.Trunc(v) != v {
				return nil, fmt.Errorf("seed: must be an integer, got %v", seedRaw)
			}
			n, err := floatSeedToInt64(v)
			if err != nil {
				return nil, err
			}
			prov = NewReelProvider(base.ModelID(), n)
		case json.Number:
			n, err := v.Int64()
			if err != nil {
				return nil, fmt.Errorf("seed: %w", err)
			}
			prov = NewReelProvider(base.ModelID(), n)
		case int64:
			prov = NewReelProvider(base.ModelID(), v)
		case int:
			prov = NewReelProvider(base.ModelID(), int64(v))
		default:
			return nil, fmt.Errorf("seed: unsupported type %T", seedRaw)
		}
	}
	return prov, nil
}

func floatSeedToInt64(v float64) (int64, error) {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, errors.New("seed: must be a finite number")
	}
	if math.Abs(v) > float64(maxExactIntegerFloat) {
		return 0, fmt.Errorf(
			"seed: magnitude too large for safe conversion from float (max abs is %d)",
			maxExactIntegerFloat,
		)
	}
	return int64(v), nil
}

// readArtifactBytes reads rc fully subject to maxBytes, using ContentLength for an early size check when present.
func readArtifactBytes(rc io.ReadCloser, contentLength *int64, maxBytes int64) ([]byte, error) {
	if rc == nil {
		return nil, errors.New("videogenerator: S3 GetObject returned nil body")
	}
	defer rc.Close()
	if maxBytes == math.MaxInt64 {
		return nil, errors.New(
			"videogenerator: maximum artifact size overflows safe single read limit; use a slightly smaller MaxArtifactBytes",
		)
	}
	if contentLength != nil && *contentLength >= 0 && *contentLength > maxBytes {
		return nil, fmt.Errorf(
			"video object size (%d bytes) exceeds maximum artifact size (%d bytes); "+
				"raise videogenerator.Config.MaxArtifactBytes if appropriate, or verify the S3 object is the expected MP4",
			*contentLength,
			maxBytes,
		)
	}
	limited := io.LimitReader(rc, maxBytes+1) // +1 so exactly maxBytes bytes still succeeds; overflow ruled out above
	data, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf("read video body: %w", err)
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf(
			"video download exceeds maximum artifact size (%d bytes); "+
				"object may be larger than allowed or Content-Length was unavailable",
			maxBytes,
		)
	}
	return data, nil
}

func (t *videoGenTool) appendArtifactFromS3(
	ctx tool.Context,
	fileName, videoS3URI string,
	out map[string]any,
) error {
	if t.s3 == nil {
		return nil
	}

	bucket, key, err := parseS3URI(videoS3URI)
	if err != nil {
		return err
	}
	gobj, err := t.s3.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("s3 get object s3://%s/%s: %w", bucket, key, err)
	}
	videoBytes, err := readArtifactBytes(gobj.Body, gobj.ContentLength, t.maxArtifactBytes)
	if err != nil {
		return err
	}

	part := &genai.Part{
		InlineData: &genai.Blob{
			Data:     videoBytes,
			MIMEType: "video/mp4",
		},
	}
	saveResp, err := ctx.Artifacts().Save(ctx, fileName, part)
	if err != nil {
		return fmt.Errorf("save artifact %q: %w", fileName, err)
	}
	out["file_name"] = fileName
	out["version"] = saveResp.Version
	return nil
}

func (t *videoGenTool) Run(ctx tool.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}

	prompt, _ := m["prompt"].(string)
	if prompt == "" {
		return nil, errors.New("prompt is required")
	}
	if utf8.RuneCountInString(prompt) > maxNovaReelPromptRunes {
		return nil, fmt.Errorf("prompt exceeds %d characters", maxNovaReelPromptRunes)
	}

	fileName, _ := m["file_name"].(string)
	if fileName == "" {
		fileName = defaultArtifactFileName
	}

	prov, err := providerForArgs(t.provider, m)
	if err != nil {
		return nil, err
	}

	clientToken := strings.TrimSpace(ctx.FunctionCallID())
	if clientToken == "" {
		clientToken = uuid.NewString()
	}
	invocationArn, err := t.startAsyncJob(ctx, prov, prompt, clientToken)
	if err != nil {
		return nil, err
	}

	final, err := t.pollUntilTerminal(ctx, invocationArn)
	if err != nil {
		return nil, err
	}
	if final.Status == types.AsyncInvokeStatusFailed {
		return nil, failureError(invocationArn, final.FailureMessage)
	}

	s3Base := extractOutputS3URI(final.OutputDataConfig)
	if s3Base == "" {
		return nil, errors.New("completed job missing S3 output location")
	}
	videoS3URI := joinS3Key(s3Base, novaReelOutputKey)

	out := map[string]any{
		"status":       "success",
		"invocation":   invocationArn,
		"video_s3_uri": videoS3URI,
	}
	if err := t.appendArtifactFromS3(ctx, fileName, videoS3URI, out); err != nil {
		return out, err
	}
	return out, nil
}

// startAsyncJob calls Bedrock StartAsyncInvoke and returns the invocation ARN.
// clientRequestToken should be stable per ADK function call (e.g. ctx.FunctionCallID()) so retries dedupe.
func (t *videoGenTool) startAsyncJob(
	ctx tool.Context,
	prov *ReelProvider,
	prompt string,
	clientRequestToken string,
) (string, error) {
	startOut, err := t.api.StartAsyncInvoke(ctx, &bedrockruntime.StartAsyncInvokeInput{
		ModelId:    aws.String(prov.ModelID()),
		ModelInput: document.NewLazyDocument(prov.modelInput(prompt)),
		OutputDataConfig: &types.AsyncInvokeOutputDataConfigMemberS3OutputDataConfig{
			Value: types.AsyncInvokeS3OutputDataConfig{
				S3Uri: aws.String(t.s3OutputURI),
			},
		},
		ClientRequestToken: aws.String(clientRequestToken),
	})
	if err != nil {
		return "", fmt.Errorf("start async invoke %s: %w", prov.ModelID(), err)
	}
	arn := aws.ToString(startOut.InvocationArn)
	if arn == "" {
		return "", errors.New("start async invoke: empty invocation ARN")
	}
	return arn, nil
}

// failureError formats a Bedrock-failure error including the invocation ARN; messages with only whitespace are treated as empty.
func failureError(invocationArn string, raw *string) error {
	msg := ""
	if raw != nil {
		msg = strings.TrimSpace(*raw)
	}
	if msg == "" {
		return fmt.Errorf(
			"video generation failed (no failure message from Bedrock; invocation %s)",
			invocationArn,
		)
	}
	return fmt.Errorf("video generation failed: %s (invocation %s)", msg, invocationArn)
}

func errVideoGenPollTimeout(maxWait time.Duration) error {
	return fmt.Errorf("video generation timed out after %v", maxWait)
}

// nextPollBackoff returns min(prev*2, maxPoll), handling overflow and cap.
func nextPollBackoff(prev, maxPoll time.Duration) time.Duration {
	if prev >= maxPoll {
		return maxPoll
	}
	doubled := prev * 2
	if doubled < prev || doubled > maxPoll {
		return maxPoll
	}
	return doubled
}

// jitterPollDelay returns d scaled to [80%,120%] to reduce aligned retries.
func jitterPollDelay(d time.Duration) time.Duration {
	if d <= 0 {
		return d
	}
	//nolint:gosec // G404: non-cryptographic jitter for poll spacing only.
	p := jitterPercentFloor + rand.IntN(jitterPercentRange)
	return time.Duration(int64(d) * int64(p) / jitterPercentDivisor)
}

func (t *videoGenTool) pollUntilTerminal(
	ctx context.Context,
	invocationArn string,
) (*bedrockruntime.GetAsyncInvokeOutput, error) {
	deadline := time.Now().Add(t.maxWait)
	sleepDur := t.pollInterval
	for {
		if time.Now().After(deadline) {
			return nil, errVideoGenPollTimeout(t.maxWait)
		}
		out, err := t.api.GetAsyncInvoke(ctx, &bedrockruntime.GetAsyncInvokeInput{
			InvocationArn: aws.String(invocationArn),
		})
		if err != nil {
			return nil, fmt.Errorf("get async invoke (poll): %w", err)
		}
		switch out.Status {
		case types.AsyncInvokeStatusCompleted, types.AsyncInvokeStatusFailed:
			return out, nil
		case types.AsyncInvokeStatusInProgress:
			if err := t.sleepPollBackoff(ctx, deadline, sleepDur); err != nil {
				return nil, err
			}
			sleepDur = nextPollBackoff(sleepDur, t.maxPollInterval)
		default:
			return nil, fmt.Errorf(
				"unexpected async invoke status %q (invocation %s)",
				out.Status,
				invocationArn,
			)
		}
	}
}

// sleepPollBackoff sleeps up to jittered sleepDur, capped by remaining time until deadline.
// Returns ctx.Err() if the context is cancelled, or errVideoGenPollTimeout if the deadline has passed.
func (t *videoGenTool) sleepPollBackoff(ctx context.Context, deadline time.Time, sleepDur time.Duration) error {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return errVideoGenPollTimeout(t.maxWait)
	}
	wait := min(jitterPollDelay(sleepDur), remaining)
	if wait <= 0 {
		return errVideoGenPollTimeout(t.maxWait)
	}
	timer := time.NewTimer(wait)
	select {
	case <-ctx.Done():
		if !timer.Stop() {
			<-timer.C
		}
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func extractOutputS3URI(cfg types.AsyncInvokeOutputDataConfig) string {
	v, ok := cfg.(*types.AsyncInvokeOutputDataConfigMemberS3OutputDataConfig)
	if !ok || v == nil {
		return ""
	}
	return aws.ToString(v.Value.S3Uri)
}

func joinS3Key(s3URI, filename string) string {
	s := strings.TrimRight(strings.TrimSpace(s3URI), "/")
	return s + "/" + filename
}

func parseS3URI(uri string) (string, string, error) {
	s := strings.TrimSpace(uri)
	const prefix = "s3://"
	if !strings.HasPrefix(s, prefix) {
		return "", "", fmt.Errorf("expected s3:// URI for object download, got %q", uri)
	}
	rest := strings.TrimPrefix(s, prefix)
	if strings.HasPrefix(rest, "/") {
		return "", "", fmt.Errorf("invalid s3 URI %q (no '/' immediately after s3://)", uri)
	}
	if rest == "" {
		return "", "", errors.New("empty s3 uri")
	}
	before, after, ok := strings.Cut(rest, "/")
	if !ok {
		return "", "", errors.New("s3 uri missing object key")
	}
	bucket := before
	key := strings.TrimPrefix(after, "/")
	if bucket == "" || key == "" {
		return "", "", fmt.Errorf("invalid s3 uri %q", uri)
	}
	return bucket, key, nil
}
