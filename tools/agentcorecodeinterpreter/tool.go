package agentcorecodeinterpreter

import (
	"context"
	"errors"
	"fmt"
	"maps"
	pathpkg "path"
	"slices"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	agentcoretypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/genai"

	bedrockmappers "github.com/craigh33/adk-go-bedrock/internal/mappers"
)

const (
	codeInterpreterToolName = "execute_code"

	paramCode            = "code"
	paramLanguage        = "language"
	paramRuntime         = "runtime"
	paramInputArtifacts  = "input_artifacts"
	paramOutputArtifacts = "output_artifacts"
	paramArtifactName    = "artifact_name"
	paramPath            = "path"

	schemaTypeObject = "OBJECT"
	schemaTypeString = "STRING"
	schemaTypeArray  = "ARRAY"

	resultStatusSuccess = "success"

	defaultMaxExecutionTime = 5 * time.Minute
	defaultMaxArtifactBytes = 16 << 20 // 16 MiB
	stopTimeout             = 10 * time.Second
)

const codeInterpreterToolDescription = `Executes code in Amazon Bedrock AgentCore Code Interpreter, optionally loading ADK artifacts as input files and saving requested output files back as artifacts.`

// AgentCoreAPI is the subset of Bedrock AgentCore used by this tool.
type AgentCoreAPI interface {
	StartCodeInterpreterSession(
		context.Context,
		*bedrockagentcore.StartCodeInterpreterSessionInput,
		...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.StartCodeInterpreterSessionOutput, error)
	InvokeCodeInterpreter(
		context.Context,
		*bedrockagentcore.InvokeCodeInterpreterInput,
		...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.InvokeCodeInterpreterOutput, error)
	StopCodeInterpreterSession(
		context.Context,
		*bedrockagentcore.StopCodeInterpreterSessionInput,
		...func(*bedrockagentcore.Options),
	) (*bedrockagentcore.StopCodeInterpreterSessionOutput, error)
}

// Config configures the Code Interpreter tool.
type Config struct {
	API                       AgentCoreAPI
	CodeInterpreterIdentifier string
	SessionName               string
	DefaultLanguage           string
	AllowedLanguages          []string
	MaxExecutionTime          time.Duration
	MaxInputArtifactBytes     int64
	MaxOutputBytes            int64
}

type codeInterpreterTool struct {
	api                       AgentCoreAPI
	codeInterpreterIdentifier string
	sessionName               string
	defaultLanguage           string
	allowedLanguages          []string
	maxExecutionTime          time.Duration
	maxInputArtifactBytes     int64
	maxOutputBytes            int64
	decl                      *genai.FunctionDeclaration
}

type inputArtifactArg struct {
	ArtifactName string
	Path         string
}

type outputArtifactArg struct {
	Path         string
	ArtifactName string
}

// New creates an ADK-compatible Code Interpreter tool.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("agentcorecodeinterpreter: API is required")
	}
	codeInterpreterIdentifier := strings.TrimSpace(cfg.CodeInterpreterIdentifier)
	if codeInterpreterIdentifier == "" {
		return nil, errors.New("agentcorecodeinterpreter: CodeInterpreterIdentifier is required")
	}
	if cfg.MaxExecutionTime < 0 {
		return nil, errors.New("agentcorecodeinterpreter: MaxExecutionTime cannot be negative")
	}
	if cfg.MaxInputArtifactBytes < 0 {
		return nil, errors.New("agentcorecodeinterpreter: MaxInputArtifactBytes cannot be negative")
	}
	if cfg.MaxOutputBytes < 0 {
		return nil, errors.New("agentcorecodeinterpreter: MaxOutputBytes cannot be negative")
	}

	allowed, err := bedrockmappers.AgentCoreCodeInterpreterNormalizeAllowedLanguages(cfg.AllowedLanguages)
	if err != nil {
		return nil, err
	}
	defaultLanguageRaw := cfg.DefaultLanguage
	if strings.TrimSpace(defaultLanguageRaw) == "" && !containsString(allowed, "python") {
		defaultLanguageRaw = allowed[0]
	}
	defaultLanguage, err := bedrockmappers.AgentCoreCodeInterpreterNormalizeLanguage("", defaultLanguageRaw, allowed)
	if err != nil {
		return nil, err
	}
	maxExecutionTime := cfg.MaxExecutionTime
	if maxExecutionTime == 0 {
		maxExecutionTime = defaultMaxExecutionTime
	}
	maxInputArtifactBytes := cfg.MaxInputArtifactBytes
	if maxInputArtifactBytes == 0 {
		maxInputArtifactBytes = defaultMaxArtifactBytes
	}
	maxOutputBytes := cfg.MaxOutputBytes
	if maxOutputBytes == 0 {
		maxOutputBytes = defaultMaxArtifactBytes
	}
	sessionName := strings.TrimSpace(cfg.SessionName)
	if sessionName == "" {
		sessionName = bedrockmappers.AgentCoreCodeInterpreterDefaultSessionName
	}

	t := &codeInterpreterTool{
		api:                       cfg.API,
		codeInterpreterIdentifier: codeInterpreterIdentifier,
		sessionName:               sessionName,
		defaultLanguage:           defaultLanguage,
		allowedLanguages:          allowed,
		maxExecutionTime:          maxExecutionTime,
		maxInputArtifactBytes:     maxInputArtifactBytes,
		maxOutputBytes:            maxOutputBytes,
	}
	t.decl = t.newFunctionDeclaration()
	return t, nil
}

func (t *codeInterpreterTool) Name() string {
	return codeInterpreterToolName
}

func (t *codeInterpreterTool) Description() string {
	return codeInterpreterToolDescription
}

func (t *codeInterpreterTool) IsLongRunning() bool {
	return false
}

func (t *codeInterpreterTool) Declaration() *genai.FunctionDeclaration {
	return t.decl
}

func (t *codeInterpreterTool) newFunctionDeclaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{
		Name:        codeInterpreterToolName,
		Description: codeInterpreterToolDescription,
		Parameters: &genai.Schema{
			Type: schemaTypeObject,
			Properties: map[string]*genai.Schema{
				paramCode: {
					Type:        schemaTypeString,
					Description: "Source code to execute in the sandbox.",
				},
				paramLanguage: {
					Type:        schemaTypeString,
					Format:      "enum",
					Enum:        append([]string(nil), t.allowedLanguages...),
					Description: fmt.Sprintf("Programming language for the code. Defaults to %q.", t.defaultLanguage),
				},
				paramRuntime: {
					Type:        schemaTypeString,
					Format:      "enum",
					Enum:        []string{"python", "nodejs", "deno"},
					Description: "Optional runtime override. JavaScript and TypeScript default to deno when omitted.",
				},
				paramInputArtifacts: {
					Type:        schemaTypeArray,
					Description: "Optional ADK artifacts to write into the sandbox before execution.",
					Items: &genai.Schema{
						Type: schemaTypeObject,
						Properties: map[string]*genai.Schema{
							paramArtifactName: {
								Type:        schemaTypeString,
								Description: "ADK artifact filename to load.",
							},
							paramPath: {
								Type:        schemaTypeString,
								Description: "Relative sandbox path to write. Defaults to artifact_name. Absolute paths and '..' are not allowed.",
							},
						},
						Required: []string{paramArtifactName},
					},
				},
				paramOutputArtifacts: {
					Type:        schemaTypeArray,
					Description: "Optional sandbox files to read and save back as ADK artifacts after execution.",
					Items: &genai.Schema{
						Type: schemaTypeObject,
						Properties: map[string]*genai.Schema{
							paramPath: {
								Type:        schemaTypeString,
								Description: "Relative sandbox path to read. Absolute paths and '..' are not allowed.",
							},
							paramArtifactName: {
								Type:        schemaTypeString,
								Description: "ADK artifact filename to save. Defaults to the path basename.",
							},
						},
						Required: []string{paramPath},
					},
				},
			},
			Required: []string{paramCode},
		},
	}
}

func (t *codeInterpreterTool) ProcessRequest(_ agent.Context, req *model.LLMRequest) error {
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

//nolint:gocognit,funlen,nonamedreturns // Named returns let deferred cleanup update result or err.
func (t *codeInterpreterTool) Run(ctx agent.Context, args any) (result map[string]any, err error) {
	m, ok := args.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected args type: %T", args)
	}

	code, _ := m[paramCode].(string)
	if strings.TrimSpace(code) == "" {
		return nil, errors.New("code is required")
	}
	languageRaw, _ := m[paramLanguage].(string)
	language, err := bedrockmappers.AgentCoreCodeInterpreterNormalizeLanguage(
		languageRaw,
		t.defaultLanguage,
		t.allowedLanguages,
	)
	if err != nil {
		return nil, err
	}
	runtimeRaw, _ := m[paramRuntime].(string)
	runtime, err := bedrockmappers.AgentCoreCodeInterpreterNormalizeRuntime(runtimeRaw)
	if err != nil {
		return nil, err
	}
	inputArtifacts, err := parseInputArtifactArgs(m[paramInputArtifacts])
	if err != nil {
		return nil, err
	}
	outputArtifacts, err := parseOutputArtifactArgs(m[paramOutputArtifacts])
	if err != nil {
		return nil, err
	}

	runCtx, cancel := context.WithTimeout(ctx, t.maxExecutionTime)
	defer cancel()

	clientToken := bedrockmappers.AgentCoreCodeInterpreterClientToken(ctx.FunctionCallID())
	startOut, err := t.api.StartCodeInterpreterSession(runCtx, bedrockmappers.AgentCoreCodeInterpreterStartInput(
		bedrockmappers.AgentCoreCodeInterpreterStartParams{
			CodeInterpreterIdentifier: t.codeInterpreterIdentifier,
			SessionName:               t.sessionName,
			ClientToken:               clientToken,
			MaxExecutionTime:          t.maxExecutionTime,
		},
	))
	if err != nil {
		return nil, fmt.Errorf("start code interpreter session: %w", err)
	}
	sessionID := aws.ToString(startOut.SessionId)
	if sessionID == "" {
		return nil, errors.New("start code interpreter session: empty session ID")
	}
	defer func() {
		stopErr := t.stopSession(sessionID, clientToken)
		if stopErr == nil {
			return
		}
		if result != nil {
			result["cleanup_error"] = stopErr.Error()
			return
		}
		if err == nil {
			err = stopErr
		}
	}()

	if len(inputArtifacts) > 0 {
		files, err := t.loadInputArtifacts(runCtx, ctx, inputArtifacts)
		if err != nil {
			return nil, err
		}
		results, err := t.invoke(runCtx, bedrockmappers.AgentCoreCodeInterpreterWriteFilesInput(
			t.codeInterpreterIdentifier,
			sessionID,
			files,
		))
		if err != nil {
			return nil, err
		}
		if err := t.resultFailure("writeFiles", results); err != nil {
			return nil, err
		}
	}

	execResults, err := t.invoke(runCtx, bedrockmappers.AgentCoreCodeInterpreterExecuteInput(
		bedrockmappers.AgentCoreCodeInterpreterInvokeParams{
			CodeInterpreterIdentifier: t.codeInterpreterIdentifier,
			SessionID:                 sessionID,
			Code:                      code,
			Language:                  language,
			Runtime:                   runtime,
		},
	))
	if err != nil {
		return nil, err
	}
	if len(execResults) == 0 {
		return nil, errors.New("executeCode returned no result")
	}

	result, generatedArtifacts, err := t.mapResults(execResults)
	if err != nil {
		return nil, err
	}
	if len(outputArtifacts) > 0 {
		readArtifacts, err := t.readOutputArtifacts(runCtx, sessionID, outputArtifacts)
		if err != nil {
			return nil, err
		}
		generatedArtifacts = append(generatedArtifacts, readArtifacts...)
	}
	if len(generatedArtifacts) > 0 {
		saved, err := t.saveArtifacts(runCtx, ctx, generatedArtifacts)
		if err != nil {
			return result, err
		}
		result["artifacts"] = saved
	}
	result["session_id"] = sessionID
	return result, nil
}

func (t *codeInterpreterTool) stopSession(sessionID, clientToken string) error {
	ctx, cancel := context.WithTimeout(context.Background(), stopTimeout)
	defer cancel()
	_, err := t.api.StopCodeInterpreterSession(ctx, bedrockmappers.AgentCoreCodeInterpreterStopInput(
		t.codeInterpreterIdentifier,
		sessionID,
		clientToken+"-stop",
	))
	if err != nil {
		return fmt.Errorf("stop code interpreter session %q: %w", sessionID, err)
	}
	return nil
}

func (t *codeInterpreterTool) invoke(
	ctx context.Context,
	input *bedrockagentcore.InvokeCodeInterpreterInput,
) ([]agentcoretypes.CodeInterpreterResult, error) {
	out, err := t.api.InvokeCodeInterpreter(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("invoke code interpreter %s: %w", input.Name, err)
	}
	if out == nil || out.GetStream() == nil {
		return nil, fmt.Errorf("invoke code interpreter %s: missing event stream", input.Name)
	}
	stream := out.GetStream()
	streamClosed := false
	defer func() {
		if !streamClosed {
			_ = stream.Close()
		}
	}()
	var results []agentcoretypes.CodeInterpreterResult
	for ev := range stream.Events() {
		switch v := ev.(type) {
		case *agentcoretypes.CodeInterpreterStreamOutputMemberResult:
			results = append(results, v.Value)
		default:
			return nil, fmt.Errorf("invoke code interpreter %s: unexpected stream event %T", input.Name, ev)
		}
	}
	streamClosed = true
	if err := stream.Close(); err != nil {
		return nil, fmt.Errorf("close code interpreter stream %s: %w", input.Name, err)
	}
	return results, nil
}

func (t *codeInterpreterTool) resultFailure(op string, results []agentcoretypes.CodeInterpreterResult) error {
	for _, result := range results {
		m, _, err := bedrockmappers.AgentCoreCodeInterpreterResult(result, t.maxOutputBytes)
		if err != nil {
			return err
		}
		if isError, _ := m["is_error"].(bool); isError {
			return fmt.Errorf("agentcorecodeinterpreter: %s failed: %v", op, m)
		}
	}
	return nil
}

func (t *codeInterpreterTool) mapResults(
	results []agentcoretypes.CodeInterpreterResult,
) (map[string]any, []bedrockmappers.AgentCoreCodeInterpreterOutputArtifact, error) {
	merged := map[string]any{"status": resultStatusSuccess, "is_error": false}
	var artifacts []bedrockmappers.AgentCoreCodeInterpreterOutputArtifact
	var content []map[string]any
	for _, result := range results {
		m, arts, err := bedrockmappers.AgentCoreCodeInterpreterResult(result, t.maxOutputBytes)
		if err != nil {
			return nil, nil, err
		}
		if c, ok := m["content"].([]map[string]any); ok {
			content = append(content, c...)
			delete(m, "content")
		}
		maps.Copy(merged, m)
		artifacts = append(artifacts, arts...)
	}
	if len(content) > 0 {
		merged["content"] = content
	}
	return merged, artifacts, nil
}

func (t *codeInterpreterTool) loadInputArtifacts(
	ctx context.Context,
	toolCtx agent.Context,
	args []inputArtifactArg,
) ([]bedrockmappers.AgentCoreCodeInterpreterInputFile, error) {
	artifacts := toolCtx.Artifacts()
	if artifacts == nil {
		return nil, errors.New("agentcorecodeinterpreter: artifact service is unavailable")
	}
	files := make([]bedrockmappers.AgentCoreCodeInterpreterInputFile, 0, len(args))
	for _, arg := range args {
		resp, err := artifacts.Load(ctx, arg.ArtifactName)
		if err != nil {
			return nil, fmt.Errorf("load artifact %q: %w", arg.ArtifactName, err)
		}
		file, err := inputFileFromPart(arg.Path, resp.Part, t.maxInputArtifactBytes)
		if err != nil {
			return nil, fmt.Errorf("artifact %q: %w", arg.ArtifactName, err)
		}
		files = append(files, file)
	}
	return files, nil
}

func inputFileFromPart(
	path string,
	part *genai.Part,
	maxBytes int64,
) (bedrockmappers.AgentCoreCodeInterpreterInputFile, error) {
	if part == nil {
		return bedrockmappers.AgentCoreCodeInterpreterInputFile{}, errors.New("loaded artifact has nil part")
	}
	if part.InlineData != nil {
		if int64(len(part.InlineData.Data)) > maxBytes {
			return bedrockmappers.AgentCoreCodeInterpreterInputFile{}, fmt.Errorf(
				"size (%d bytes) exceeds maximum input artifact size (%d bytes)",
				len(part.InlineData.Data),
				maxBytes,
			)
		}
		return bedrockmappers.AgentCoreCodeInterpreterInputFile{Path: path, Blob: part.InlineData.Data}, nil
	}
	if part.Text != "" {
		if int64(len(part.Text)) > maxBytes {
			return bedrockmappers.AgentCoreCodeInterpreterInputFile{}, fmt.Errorf(
				"size (%d bytes) exceeds maximum input artifact size (%d bytes)",
				len(part.Text),
				maxBytes,
			)
		}
		return bedrockmappers.AgentCoreCodeInterpreterInputFile{Path: path, Text: part.Text, IsText: true}, nil
	}
	return bedrockmappers.AgentCoreCodeInterpreterInputFile{}, errors.New(
		"loaded artifact must contain InlineData or Text",
	)
}

func (t *codeInterpreterTool) readOutputArtifacts(
	ctx context.Context,
	sessionID string,
	args []outputArtifactArg,
) ([]bedrockmappers.AgentCoreCodeInterpreterOutputArtifact, error) {
	var artifacts []bedrockmappers.AgentCoreCodeInterpreterOutputArtifact
	for _, arg := range args {
		results, err := t.invoke(ctx, bedrockmappers.AgentCoreCodeInterpreterReadFilesInput(
			t.codeInterpreterIdentifier,
			sessionID,
			[]string{arg.Path},
		))
		if err != nil {
			return nil, err
		}
		if err := t.resultFailure("readFiles", results); err != nil {
			return nil, err
		}
		_, arts, err := t.mapResults(results)
		if err != nil {
			return nil, err
		}
		if len(arts) > 0 && arg.ArtifactName != "" {
			arts[0].ArtifactName = arg.ArtifactName
			arts[0].Path = arg.Path
		}
		artifacts = append(artifacts, arts...)
	}
	return artifacts, nil
}

func (t *codeInterpreterTool) saveArtifacts(
	ctx context.Context,
	toolCtx agent.Context,
	artifacts []bedrockmappers.AgentCoreCodeInterpreterOutputArtifact,
) ([]map[string]any, error) {
	service := toolCtx.Artifacts()
	if service == nil {
		return nil, errors.New("agentcorecodeinterpreter: artifact service is unavailable")
	}
	saved := make([]map[string]any, 0, len(artifacts))
	for _, artifact := range artifacts {
		part := genai.NewPartFromBytes(artifact.Data, artifact.MIMEType)
		if artifact.IsText {
			part = genai.NewPartFromText(artifact.Text)
		}
		resp, err := service.Save(ctx, artifact.ArtifactName, part)
		if err != nil {
			return nil, fmt.Errorf("save artifact %q: %w", artifact.ArtifactName, err)
		}
		saved = append(saved, map[string]any{
			"path":      artifact.Path,
			"file_name": artifact.ArtifactName,
			"mime_type": artifact.MIMEType,
			"version":   resp.Version,
		})
	}
	return saved, nil
}

func parseInputArtifactArgs(raw any) ([]inputArtifactArg, error) {
	items, err := objectSlice(raw, paramInputArtifacts)
	if err != nil {
		return nil, err
	}
	out := make([]inputArtifactArg, 0, len(items))
	for _, item := range items {
		artifactName, _ := item[paramArtifactName].(string)
		artifactName = strings.TrimSpace(artifactName)
		if artifactName == "" {
			return nil, fmt.Errorf("%s.%s is required", paramInputArtifacts, paramArtifactName)
		}
		path, _ := item[paramPath].(string)
		path = strings.TrimSpace(path)
		if path == "" {
			path = artifactName
		}
		path, err = sandboxArtifactPath(path, fmt.Sprintf("%s.%s", paramInputArtifacts, paramPath))
		if err != nil {
			return nil, err
		}
		out = append(out, inputArtifactArg{ArtifactName: artifactName, Path: path})
	}
	return out, nil
}

func parseOutputArtifactArgs(raw any) ([]outputArtifactArg, error) {
	items, err := objectSlice(raw, paramOutputArtifacts)
	if err != nil {
		return nil, err
	}
	out := make([]outputArtifactArg, 0, len(items))
	for i, item := range items {
		path, _ := item[paramPath].(string)
		path = strings.TrimSpace(path)
		if path == "" {
			return nil, fmt.Errorf("%s.%s is required", paramOutputArtifacts, paramPath)
		}
		path, err = sandboxArtifactPath(path, fmt.Sprintf("%s.%s", paramOutputArtifacts, paramPath))
		if err != nil {
			return nil, err
		}
		artifactName, _ := item[paramArtifactName].(string)
		artifactName = strings.TrimSpace(artifactName)
		if artifactName == "" {
			artifactName = pathpkg.Base(strings.ReplaceAll(path, "\\", "/"))
		}
		if artifactName == "." || artifactName == "/" || artifactName == "" {
			artifactName = bedrockmappers.AgentCoreCodeInterpreterArtifactName(path, i)
		}
		out = append(out, outputArtifactArg{Path: path, ArtifactName: artifactName})
	}
	return out, nil
}

func sandboxArtifactPath(raw, field string) (string, error) {
	s := strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	if s == "" {
		return "", fmt.Errorf("%s is required", field)
	}
	if strings.HasPrefix(s, "/") {
		return "", fmt.Errorf("%s must be a relative sandbox path, got %q", field, raw)
	}
	if slices.Contains(strings.Split(s, "/"), "..") {
		return "", fmt.Errorf("%s must not contain '..', got %q", field, raw)
	}
	clean := pathpkg.Clean(s)
	if clean == "." || clean == "" {
		return "", fmt.Errorf("%s is required", field)
	}
	return clean, nil
}

func objectSlice(raw any, name string) ([]map[string]any, error) {
	if raw == nil {
		return nil, nil
	}
	switch v := raw.(type) {
	case []map[string]any:
		return v, nil
	case []any:
		out := make([]map[string]any, 0, len(v))
		for i, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("%s[%d]: expected object, got %T", name, i, item)
			}
			out = append(out, m)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%s: expected array, got %T", name, raw)
	}
}

func containsString(values []string, want string) bool {
	return slices.Contains(values, want)
}
