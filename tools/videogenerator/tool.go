package videogenerator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

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
const DefaultReelModelID = "amazon.nova-reel-v1:1"

const (
	defaultDurationSeconds = 6
	defaultFPS             = 24
	defaultDimension       = "1280x720"
	novaReelOutputKey      = "output.mp4"
	maxNovaReelPromptRunes = 512

	maxNovaReelSeed int64 = 2147483646

	// maxExactIntegerFloat is the largest integer magnitude where every integer
	// in [-N, N] is exactly representable as float64 (IEEE 754 doubles).
	maxExactIntegerFloat = 9007199254740991 // 2^53 - 1
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

func clampNovaReelSeed(s int64) int64 {
	if s < 0 {
		return 0
	}
	if s <= maxNovaReelSeed {
		return s
	}
	return s % (maxNovaReelSeed + 1)
}

func randomNovaReelSeed() int64 {
	n := time.Now().UnixNano()
	if n < 0 {
		n = -n
	}
	return 1 + (n % maxNovaReelSeed)
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

	// PollInterval is the delay between GetAsyncInvoke polls. Zero defaults to 5s.
	PollInterval time.Duration
	// MaxWait is the maximum time to wait for job completion. Zero defaults to 30m.
	MaxWait time.Duration
}

type videoGenTool struct {
	api          AsyncInvokeAPI
	s3           S3GetObjectAPI
	s3OutputURI  string
	provider     *ReelProvider
	pollInterval time.Duration
	maxWait      time.Duration
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
	maxWait := cfg.MaxWait
	if maxWait <= 0 {
		maxWait = 30 * time.Minute
	}
	return &videoGenTool{
		api:          cfg.API,
		s3:           cfg.S3,
		s3OutputURI:  strings.TrimSpace(cfg.S3OutputURI),
		provider:     prov,
		pollInterval: poll,
		maxWait:      maxWait,
	}, nil
}

func (t *videoGenTool) Name() string {
	return "generate_video"
}

func (t *videoGenTool) Description() string {
	return `Generates a short video from a text prompt using Amazon Nova Reel (Bedrock async invoke), stores output in your configured S3 location, and saves the MP4 as an artifact when S3 download is configured.

NOTE: This is a long-running operation (often minutes). Do not call this tool again for the same request until it returns.`
}

func (t *videoGenTool) IsLongRunning() bool {
	return true
}

func (t *videoGenTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        t.Name(),
		Description: t.Description(),
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"prompt": {
					Type:        "STRING",
					Description: "Text prompt describing the video (English, max 512 characters for single-shot)",
				},
				"file_name": {
					Type:        "STRING",
					Description: "Artifact filename for the downloaded MP4 (e.g. 'clip.mp4'). Defaults to 'generated_video.mp4'.",
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
	if math.Abs(v) > maxExactIntegerFloat {
		return 0, fmt.Errorf(
			"seed: magnitude too large for safe conversion from float (max abs is %d)",
			maxExactIntegerFloat,
		)
	}
	return int64(v), nil
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
	defer gobj.Body.Close()
	videoBytes, err := io.ReadAll(gobj.Body)
	if err != nil {
		return fmt.Errorf("read video body: %w", err)
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
	if len([]rune(prompt)) > maxNovaReelPromptRunes {
		return nil, fmt.Errorf("prompt exceeds %d characters", maxNovaReelPromptRunes)
	}

	fileName, _ := m["file_name"].(string)
	if fileName == "" {
		fileName = "generated_video.mp4"
	}

	prov, err := providerForArgs(t.provider, m)
	if err != nil {
		return nil, err
	}

	startOut, err := t.api.StartAsyncInvoke(ctx, &bedrockruntime.StartAsyncInvokeInput{
		ModelId:    aws.String(prov.ModelID()),
		ModelInput: document.NewLazyDocument(prov.modelInput(prompt)),
		OutputDataConfig: &types.AsyncInvokeOutputDataConfigMemberS3OutputDataConfig{
			Value: types.AsyncInvokeS3OutputDataConfig{
				S3Uri: aws.String(t.s3OutputURI),
			},
		},
		ClientRequestToken: aws.String(uuid.NewString()),
	})
	if err != nil {
		return nil, fmt.Errorf("start async invoke %s: %w", prov.ModelID(), err)
	}
	if startOut.InvocationArn == nil || *startOut.InvocationArn == "" {
		return nil, errors.New("start async invoke: empty invocation ARN")
	}
	invocationArn := aws.ToString(startOut.InvocationArn)

	final, err := t.pollUntilTerminal(ctx, invocationArn)
	if err != nil {
		return nil, err
	}
	if final.Status == types.AsyncInvokeStatusFailed {
		msg := ""
		if final.FailureMessage != nil {
			msg = *final.FailureMessage
		}
		msg = strings.TrimSpace(msg)
		if msg == "" {
			return nil, fmt.Errorf(
				"video generation failed (no failure message from Bedrock; invocation %s)",
				invocationArn,
			)
		}
		return nil, fmt.Errorf(
			"video generation failed: %s (invocation %s)",
			msg,
			invocationArn,
		)
	}

	s3Base, ok := extractOutputS3URI(final.OutputDataConfig)
	if !ok || s3Base == "" {
		return nil, errors.New("completed job missing S3 output location")
	}
	videoS3URI := joinS3Key(s3Base, novaReelOutputKey)

	out := map[string]any{
		"status":       "success",
		"invocation":   invocationArn,
		"video_s3_uri": videoS3URI,
	}
	if err := t.appendArtifactFromS3(ctx, fileName, videoS3URI, out); err != nil {
		return nil, err
	}
	return out, nil
}

func errVideoGenPollTimeout(maxWait time.Duration) error {
	return fmt.Errorf("video generation timed out after %v", maxWait)
}

func (t *videoGenTool) pollUntilTerminal(
	ctx context.Context,
	invocationArn string,
) (*bedrockruntime.GetAsyncInvokeOutput, error) {
	ticker := time.NewTicker(t.pollInterval)
	defer ticker.Stop()
	deadline := time.Now().Add(t.maxWait)
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
		if out.Status == types.AsyncInvokeStatusCompleted ||
			out.Status == types.AsyncInvokeStatusFailed {
			return out, nil
		}
		// IN_PROGRESS, unknown future statuses, or empty string: fixed-interval wait then poll again.
		if time.Now().After(deadline) {
			return nil, errVideoGenPollTimeout(t.maxWait)
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

func extractOutputS3URI(cfg types.AsyncInvokeOutputDataConfig) (string, bool) {
	switch v := cfg.(type) {
	case *types.AsyncInvokeOutputDataConfigMemberS3OutputDataConfig:
		if v == nil {
			return "", false
		}
		return aws.ToString(v.Value.S3Uri), true
	default:
		return "", false
	}
}

func joinS3Key(s3URI, filename string) string {
	s := strings.TrimRight(strings.TrimSpace(s3URI), "/")
	return s + "/" + filename
}

func parseS3URI(uri string) (string, string, error) {
	s := strings.TrimSpace(uri)
	s = strings.TrimPrefix(s, "s3://")
	if s == "" {
		return "", "", errors.New("empty s3 uri")
	}
	before, after, ok := strings.Cut(s, "/")
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
