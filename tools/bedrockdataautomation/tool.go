package bedrockdataautomation

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	bdaruntime "github.com/aws/aws-sdk-go-v2/service/bedrockdataautomationruntime"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	bedrockmappers "github.com/craigh33/adk-go-bedrock/internal/mappers"
)

const (
	dataAutomationToolName = "analyze_data"
	resultObjectName       = "output.json"

	paramS3URI              = "s3_uri"
	paramArtifactName       = "artifact_name"
	paramProjectARN         = "project_arn"
	paramBlueprintARN       = "blueprint_arn"
	paramBlueprintVersion   = "blueprint_version"
	paramStage              = "stage"
	paramOutputS3URI        = "output_s3_uri"
	paramResultArtifactName = "result_artifact_name"

	genaiSchemaString = "STRING"

	defaultPollInterval    = 5 * time.Second
	defaultMaxPollInterval = 30 * time.Second
	defaultMaxWait         = 30 * time.Minute

	defaultMaxInputArtifactBytes int64 = 512 << 20 // 512 MiB
	defaultMaxResultBytes        int64 = 16 << 20  // 16 MiB
)

const dataAutomationToolDescription = `Analyzes an existing S3 object or ADK artifact with Amazon Bedrock Data Automation and returns the async output S3 location.

NOTE: This is a long-running operation. Provide exactly one of s3_uri or artifact_name.`

// RuntimeAPI is the Bedrock Data Automation Runtime subset used by this tool.
type RuntimeAPI interface {
	InvokeDataAutomationAsync(
		ctx context.Context,
		params *bdaruntime.InvokeDataAutomationAsyncInput,
		optFns ...func(*bdaruntime.Options),
	) (*bdaruntime.InvokeDataAutomationAsyncOutput, error)
	GetDataAutomationStatus(
		ctx context.Context,
		params *bdaruntime.GetDataAutomationStatusInput,
		optFns ...func(*bdaruntime.Options),
	) (*bdaruntime.GetDataAutomationStatusOutput, error)
}

// S3API stages artifact inputs and downloads optional JSON output artifacts.
type S3API interface {
	PutObject(
		ctx context.Context,
		params *s3.PutObjectInput,
		optFns ...func(*s3.Options),
	) (*s3.PutObjectOutput, error)
	GetObject(
		ctx context.Context,
		params *s3.GetObjectInput,
		optFns ...func(*s3.Options),
	) (*s3.GetObjectOutput, error)
}

// Config configures the Bedrock Data Automation tool.
type Config struct {
	API RuntimeAPI
	S3  S3API

	// DataAutomationProfileARN is the BDA profile ARN required by InvokeDataAutomationAsync.
	DataAutomationProfileARN string
	// DataAutomationProjectARN is used when a tool call does not provide project_arn.
	DataAutomationProjectARN string
	// OutputS3URI is where BDA writes async output.
	OutputS3URI string
	// InputS3URI is where artifact inputs are staged before invoking BDA.
	InputS3URI string

	PollInterval    time.Duration
	MaxPollInterval time.Duration
	MaxWait         time.Duration

	MaxInputArtifactBytes int64
	MaxResultBytes        int64
}

type dataAutomationTool struct {
	api RuntimeAPI
	s3  S3API

	dataAutomationProfileARN string
	dataAutomationProjectARN string
	outputS3URI              string
	inputS3URI               string

	pollInterval    time.Duration
	maxPollInterval time.Duration
	maxWait         time.Duration

	maxInputArtifactBytes int64
	maxResultBytes        int64

	decl *genai.FunctionDeclaration
}

// New creates an ADK tool that analyzes media/documents through Bedrock Data Automation async runtime APIs.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("bedrockdataautomation: API is required")
	}
	if strings.TrimSpace(cfg.DataAutomationProfileARN) == "" {
		return nil, errors.New("bedrockdataautomation: DataAutomationProfileARN is required")
	}
	if err := validateS3PrefixURI("OutputS3URI", cfg.OutputS3URI); err != nil {
		return nil, err
	}
	if cfg.InputS3URI != "" {
		if err := validateS3PrefixURI("InputS3URI", cfg.InputS3URI); err != nil {
			return nil, err
		}
	}
	if cfg.MaxPollInterval < 0 {
		return nil, errors.New("bedrockdataautomation: MaxPollInterval cannot be negative")
	}
	if cfg.MaxInputArtifactBytes < 0 {
		return nil, errors.New("bedrockdataautomation: MaxInputArtifactBytes cannot be negative")
	}
	if cfg.MaxResultBytes < 0 {
		return nil, errors.New("bedrockdataautomation: MaxResultBytes cannot be negative")
	}

	poll := cfg.PollInterval
	if poll <= 0 {
		poll = defaultPollInterval
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
		maxWait = defaultMaxWait
	}
	maxInput := cfg.MaxInputArtifactBytes
	if maxInput == 0 {
		maxInput = defaultMaxInputArtifactBytes
	}
	maxResult := cfg.MaxResultBytes
	if maxResult == 0 {
		maxResult = defaultMaxResultBytes
	}

	return &dataAutomationTool{
		api:                      cfg.API,
		s3:                       cfg.S3,
		dataAutomationProfileARN: strings.TrimSpace(cfg.DataAutomationProfileARN),
		dataAutomationProjectARN: strings.TrimSpace(cfg.DataAutomationProjectARN),
		outputS3URI:              strings.TrimSpace(cfg.OutputS3URI),
		inputS3URI:               strings.TrimSpace(cfg.InputS3URI),
		pollInterval:             poll,
		maxPollInterval:          maxPoll,
		maxWait:                  maxWait,
		maxInputArtifactBytes:    maxInput,
		maxResultBytes:           maxResult,
		decl:                     newFunctionDeclaration(),
	}, nil
}

func (t *dataAutomationTool) Name() string {
	return dataAutomationToolName
}

func (t *dataAutomationTool) Description() string {
	return dataAutomationToolDescription
}

func (t *dataAutomationTool) IsLongRunning() bool {
	return true
}

func (t *dataAutomationTool) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

func newFunctionDeclaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        dataAutomationToolName,
		Description: dataAutomationToolDescription,
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				paramS3URI: {
					Type:        genaiSchemaString,
					Description: "S3 URI of the document, image, audio, or video to analyze. Provide exactly one of s3_uri or artifact_name.",
				},
				paramArtifactName: {
					Type:        genaiSchemaString,
					Description: "ADK artifact name to analyze. Provide exactly one of artifact_name or s3_uri.",
				},
				paramProjectARN: {
					Type:        genaiSchemaString,
					Description: "Optional Bedrock Data Automation project ARN. Defaults to the configured project ARN.",
				},
				paramBlueprintARN: {
					Type:        genaiSchemaString,
					Description: "Optional blueprint ARN for custom output.",
				},
				paramBlueprintVersion: {
					Type:        genaiSchemaString,
					Description: "Optional blueprint version.",
				},
				paramStage: {
					Type:        genaiSchemaString,
					Description: "Optional BDA project/blueprint stage, typically LIVE or DEVELOPMENT.",
				},
				paramOutputS3URI: {
					Type:        genaiSchemaString,
					Description: "Optional S3 output prefix for this invocation. Defaults to the configured output prefix.",
				},
				paramResultArtifactName: {
					Type: genaiSchemaString,
					Description: fmt.Sprintf(
						"Optional artifact name for saving %s from the BDA output prefix.",
						resultObjectName,
					),
				},
			},
		},
	}
}

func (t *dataAutomationTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
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

func (t *dataAutomationTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}

	s3URI, _ := m[paramS3URI].(string)
	artifactName, _ := m[paramArtifactName].(string)
	s3URI = strings.TrimSpace(s3URI)
	artifactName = strings.TrimSpace(artifactName)
	if (s3URI == "") == (artifactName == "") {
		return nil, errors.New("provide exactly one of s3_uri or artifact_name")
	}

	outputS3URI, _ := m[paramOutputS3URI].(string)
	outputS3URI = strings.TrimSpace(outputS3URI)
	if outputS3URI == "" {
		outputS3URI = t.outputS3URI
	}
	if err := validateS3PrefixURI(paramOutputS3URI, outputS3URI); err != nil {
		return nil, err
	}

	resultArtifactName, _ := m[paramResultArtifactName].(string)
	resultArtifactName = strings.TrimSpace(resultArtifactName)
	if resultArtifactName != "" && t.s3 == nil {
		return nil, errors.New("bedrockdataautomation: S3 client is required when result_artifact_name is set")
	}

	clientToken := bedrockmappers.BedrockDataAutomationClientToken(ctx.FunctionCallID())
	if artifactName != "" {
		var err error
		s3URI, err = t.stageArtifact(ctx, artifactName, clientToken)
		if err != nil {
			return nil, err
		}
	} else if _, _, err := parseS3URI(s3URI); err != nil {
		return nil, err
	}

	invocationARN, err := t.invoke(ctx, m, s3URI, outputS3URI, clientToken)
	if err != nil {
		return nil, err
	}
	final, err := t.pollUntilTerminal(ctx, invocationARN)
	if err != nil {
		return nil, err
	}
	if bedrockmappers.BedrockDataAutomationStatusIsFailure(final.Status) {
		return nil, bedrockmappers.BedrockDataAutomationFailureError(
			invocationARN,
			final.ErrorType,
			final.ErrorMessage,
		)
	}

	finalOutputS3URI := bedrockmappers.BedrockDataAutomationOutputS3URI(final.OutputConfiguration)
	if finalOutputS3URI == "" {
		return nil, errors.New("completed data automation job missing S3 output location")
	}

	out := map[string]any{
		"status":         "success",
		"invocation_arn": invocationARN,
		"input_s3_uri":   s3URI,
		paramOutputS3URI: finalOutputS3URI,
	}
	if resultArtifactName != "" {
		if err := t.appendResultArtifact(ctx, resultArtifactName, finalOutputS3URI, out); err != nil {
			return out, err
		}
	}
	return out, nil
}

func (t *dataAutomationTool) stageArtifact(ctx agent.Context, artifactName, clientToken string) (string, error) {
	if t.s3 == nil {
		return "", errors.New("bedrockdataautomation: S3 client is required for artifact input")
	}
	if t.inputS3URI == "" {
		return "", errors.New("bedrockdataautomation: InputS3URI is required for artifact input")
	}
	artifacts := ctx.Artifacts()
	if artifacts == nil {
		return "", errors.New("bedrockdataautomation: artifact service is unavailable")
	}
	resp, err := artifacts.Load(ctx, artifactName)
	if err != nil {
		return "", fmt.Errorf("load artifact %q: %w", artifactName, err)
	}
	data, contentType, err := artifactBytes(resp.Part, t.maxInputArtifactBytes)
	if err != nil {
		return "", fmt.Errorf("artifact %q: %w", artifactName, err)
	}
	stagedURI := joinS3Key(joinS3Key(t.inputS3URI, clientToken), artifactName)
	bucket, key, err := parseS3URI(stagedURI)
	if err != nil {
		return "", err
	}
	input := &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(data),
	}
	if contentType != "" {
		input.ContentType = aws.String(contentType)
	}
	if _, err := t.s3.PutObject(ctx, input); err != nil {
		return "", fmt.Errorf("s3 put object s3://%s/%s: %w", bucket, key, err)
	}
	return stagedURI, nil
}

func artifactBytes(part *genai.Part, maxBytes int64) ([]byte, string, error) {
	if part == nil {
		return nil, "", errors.New("loaded artifact has nil part")
	}
	if part.InlineData != nil {
		if int64(len(part.InlineData.Data)) > maxBytes {
			return nil, "", fmt.Errorf(
				"size (%d bytes) exceeds maximum artifact size (%d bytes)",
				len(part.InlineData.Data),
				maxBytes,
			)
		}
		return part.InlineData.Data, part.InlineData.MIMEType, nil
	}
	if part.Text != "" {
		data := []byte(part.Text)
		if int64(len(data)) > maxBytes {
			return nil, "", fmt.Errorf(
				"size (%d bytes) exceeds maximum artifact size (%d bytes)",
				len(data),
				maxBytes,
			)
		}
		return data, "text/plain; charset=utf-8", nil
	}
	return nil, "", errors.New("loaded artifact must contain InlineData or Text")
}

func (t *dataAutomationTool) invoke(
	ctx context.Context,
	args map[string]any,
	inputS3URI string,
	outputS3URI string,
	clientToken string,
) (string, error) {
	projectARN, _ := args[paramProjectARN].(string)
	projectARN = strings.TrimSpace(projectARN)
	if projectARN == "" {
		projectARN = t.dataAutomationProjectARN
	}
	blueprintARN, _ := args[paramBlueprintARN].(string)
	blueprintARN = strings.TrimSpace(blueprintARN)
	blueprintVersion, _ := args[paramBlueprintVersion].(string)
	blueprintVersion = strings.TrimSpace(blueprintVersion)
	stage, _ := args[paramStage].(string)
	input := bedrockmappers.BedrockDataAutomationInvokeInput(bedrockmappers.BedrockDataAutomationInvokeParams{
		DataAutomationProfileARN: t.dataAutomationProfileARN,
		DataAutomationProjectARN: projectARN,
		InputS3URI:               inputS3URI,
		OutputS3URI:              outputS3URI,
		ClientToken:              clientToken,
		BlueprintARN:             blueprintARN,
		BlueprintVersion:         blueprintVersion,
		Stage:                    stage,
	})

	out, err := t.api.InvokeDataAutomationAsync(ctx, input)
	if err != nil {
		return "", fmt.Errorf("invoke data automation async: %w", err)
	}
	arn := aws.ToString(out.InvocationArn)
	if arn == "" {
		return "", errors.New("invoke data automation async: empty invocation ARN")
	}
	return arn, nil
}

func (t *dataAutomationTool) pollUntilTerminal(
	ctx context.Context,
	invocationARN string,
) (*bdaruntime.GetDataAutomationStatusOutput, error) {
	deadline := time.Now().Add(t.maxWait)
	sleepDur := t.pollInterval
	for {
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("data automation timed out after %v", t.maxWait)
		}
		out, err := t.api.GetDataAutomationStatus(ctx, &bdaruntime.GetDataAutomationStatusInput{
			InvocationArn: aws.String(invocationARN),
		})
		if err != nil {
			return nil, fmt.Errorf("get data automation status: %w", err)
		}
		switch {
		case bedrockmappers.BedrockDataAutomationStatusIsTerminal(out.Status):
			return out, nil
		case bedrockmappers.BedrockDataAutomationStatusIsPending(out.Status):
			if err := t.sleepPollBackoff(ctx, deadline, sleepDur); err != nil {
				return nil, err
			}
			sleepDur = nextPollBackoff(sleepDur, t.maxPollInterval)
		default:
			return nil, fmt.Errorf("unexpected data automation status %q (invocation %s)", out.Status, invocationARN)
		}
	}
}

func (t *dataAutomationTool) sleepPollBackoff(ctx context.Context, deadline time.Time, sleepDur time.Duration) error {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return fmt.Errorf("data automation timed out after %v", t.maxWait)
	}
	wait := min(sleepDur, remaining)
	if wait <= 0 {
		return fmt.Errorf("data automation timed out after %v", t.maxWait)
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

func (t *dataAutomationTool) appendResultArtifact(
	ctx agent.Context,
	fileName, outputS3URI string,
	out map[string]any,
) error {
	resultS3URI := strings.TrimSpace(outputS3URI)
	if !strings.HasSuffix(resultS3URI, ".json") {
		resultS3URI = joinS3Key(resultS3URI, resultObjectName)
	}
	bucket, key, err := parseS3URI(resultS3URI)
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
	data, err := readObjectBytes(gobj.Body, gobj.ContentLength, t.maxResultBytes)
	if err != nil {
		return err
	}
	artifacts := ctx.Artifacts()
	if artifacts == nil {
		return errors.New("bedrockdataautomation: artifact service is unavailable")
	}
	saveResp, err := artifacts.Save(ctx, fileName, genai.NewPartFromBytes(data, "application/json"))
	if err != nil {
		return fmt.Errorf("save artifact %q: %w", fileName, err)
	}
	out["result_s3_uri"] = resultS3URI
	out["result_file_name"] = fileName
	out["result_version"] = saveResp.Version
	return nil
}

func readObjectBytes(rc io.ReadCloser, contentLength *int64, maxBytes int64) ([]byte, error) {
	if rc == nil {
		return nil, errors.New("bedrockdataautomation: S3 GetObject returned nil body")
	}
	defer rc.Close()
	if maxBytes == math.MaxInt64 {
		return nil, errors.New("bedrockdataautomation: maximum result size overflows safe single read limit")
	}
	if contentLength != nil && *contentLength >= 0 && *contentLength > maxBytes {
		return nil, fmt.Errorf(
			"result object size (%d bytes) exceeds maximum result size (%d bytes)",
			*contentLength,
			maxBytes,
		)
	}
	data, err := io.ReadAll(io.LimitReader(rc, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read result body: %w", err)
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("result download exceeds maximum result size (%d bytes)", maxBytes)
	}
	return data, nil
}

func validateS3PrefixURI(field, uri string) error {
	s := strings.TrimSpace(uri)
	if s == "" {
		return fmt.Errorf("bedrockdataautomation: %s is required", field)
	}
	const prefix = "s3://"
	if !strings.HasPrefix(s, prefix) {
		return fmt.Errorf("bedrockdataautomation: %s must use scheme %q, got %q", field, prefix, uri)
	}
	rest := strings.TrimPrefix(s, prefix)
	if strings.HasPrefix(rest, "/") {
		return fmt.Errorf("bedrockdataautomation: %s must be s3://bucket or s3://bucket/prefix, got %q", field, uri)
	}
	bucket, _, _ := strings.Cut(rest, "/")
	if strings.TrimSpace(bucket) == "" {
		return fmt.Errorf("bedrockdataautomation: %s must include a bucket name", field)
	}
	return nil
}

func parseS3URI(uri string) (string, string, error) {
	s := strings.TrimSpace(uri)
	const prefix = "s3://"
	if !strings.HasPrefix(s, prefix) {
		return "", "", fmt.Errorf("expected s3:// URI, got %q", uri)
	}
	rest := strings.TrimPrefix(s, prefix)
	if strings.HasPrefix(rest, "/") {
		return "", "", fmt.Errorf("invalid s3 URI %q (no '/' immediately after s3://)", uri)
	}
	if rest == "" {
		return "", "", errors.New("empty s3 URI")
	}
	before, after, ok := strings.Cut(rest, "/")
	if !ok {
		return "", "", errors.New("s3 URI missing object key")
	}
	bucket := before
	key := strings.TrimPrefix(after, "/")
	if bucket == "" || key == "" {
		return "", "", fmt.Errorf("invalid s3 URI %q", uri)
	}
	return bucket, key, nil
}

func joinS3Key(s3URI, name string) string {
	return strings.TrimRight(strings.TrimSpace(s3URI), "/") + "/" + strings.TrimLeft(name, "/")
}
