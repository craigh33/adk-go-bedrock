package codeinterpreter

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	agentcoretypes "github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"github.com/google/uuid"

	bedrockmappers "github.com/craigh33/adk-go-bedrock/internal/mappers"
	codeinterpretermodels "github.com/craigh33/adk-go-bedrock/internal/models/agentcore/codeinterpreter"
)

const (
	DefaultSessionName = "adk-go-bedrock-agentcore-code-interpreter"

	agentCoreCodeInterpreterDefaultIdentifier    = "aws.codeinterpreter.v1"
	agentCoreCodeInterpreterMaxClientTokenLength = 256
	agentCoreCodeInterpreterMinClientTokenLength = 33
	agentCoreCodeInterpreterMinSessionSeconds    = 60
	agentCoreCodeInterpreterMaxSessionSeconds    = 28800

	agentCoreCodeInterpreterLanguagePython     = string(agentcoretypes.ProgrammingLanguagePython)
	agentCoreCodeInterpreterLanguageJavascript = string(agentcoretypes.ProgrammingLanguageJavascript)
	agentCoreCodeInterpreterLanguageTypescript = string(agentcoretypes.ProgrammingLanguageTypescript)

	agentCoreCodeInterpreterRuntimePython = string(agentcoretypes.LanguageRuntimePython)
	agentCoreCodeInterpreterRuntimeNodeJS = string(agentcoretypes.LanguageRuntimeNodejs)
	agentCoreCodeInterpreterRuntimeDeno   = string(agentcoretypes.LanguageRuntimeDeno)

	agentCoreCodeInterpreterStatusError   = "error"
	agentCoreCodeInterpreterStatusSuccess = "success"
)

// StartInput maps config into StartCodeInterpreterSession input.
func StartInput(
	p codeinterpretermodels.StartSessionParams,
) *bedrockagentcore.StartCodeInterpreterSessionInput {
	name := strings.TrimSpace(p.SessionName)
	if name == "" {
		name = DefaultSessionName
	}
	return &bedrockagentcore.StartCodeInterpreterSessionInput{
		CodeInterpreterIdentifier: aws.String(Identifier(p.CodeInterpreterIdentifier)),
		ClientToken:               aws.String(ClientToken(p.ClientToken)),
		Name:                      aws.String(name),
		SessionTimeoutSeconds:     SessionTimeoutSeconds(p.MaxExecutionTime),
	}
}

// SessionTimeoutSeconds maps a duration to AgentCore's session TTL range.
func SessionTimeoutSeconds(d time.Duration) *int32 {
	if d <= 0 {
		return nil
	}
	minDuration := time.Duration(agentCoreCodeInterpreterMinSessionSeconds) * time.Second
	if d <= minDuration {
		return aws.Int32(agentCoreCodeInterpreterMinSessionSeconds)
	}
	maxDuration := time.Duration(agentCoreCodeInterpreterMaxSessionSeconds) * time.Second
	if d >= maxDuration {
		return aws.Int32(agentCoreCodeInterpreterMaxSessionSeconds)
	}
	seconds := int32((d + time.Second - 1) / time.Second)
	return aws.Int32(seconds)
}

// ExecuteInput maps tool args into executeCode input.
func ExecuteInput(
	p codeinterpretermodels.ExecuteParams,
) *bedrockagentcore.InvokeCodeInterpreterInput {
	args := &agentcoretypes.ToolArguments{
		Code:     aws.String(p.Code),
		Language: agentcoretypes.ProgrammingLanguage(p.Language),
	}
	if runtime := strings.TrimSpace(p.Runtime); runtime != "" {
		args.Runtime = agentcoretypes.LanguageRuntime(runtime)
	}
	return &bedrockagentcore.InvokeCodeInterpreterInput{
		CodeInterpreterIdentifier: aws.String(Identifier(p.CodeInterpreterIdentifier)),
		SessionId:                 aws.String(strings.TrimSpace(p.SessionID)),
		Name:                      agentcoretypes.ToolNameExecuteCode,
		Arguments:                 args,
	}
}

// WriteFilesInput maps artifact bytes/text into writeFiles input.
func WriteFilesInput(
	codeInterpreterIdentifier, sessionID string,
	files []codeinterpretermodels.InputFile,
) *bedrockagentcore.InvokeCodeInterpreterInput {
	content := make([]agentcoretypes.InputContentBlock, 0, len(files))
	for _, f := range files {
		block := agentcoretypes.InputContentBlock{Path: aws.String(strings.TrimSpace(f.Path))}
		if f.IsText {
			block.Text = aws.String(f.Text)
		} else {
			block.Blob = f.Blob
		}
		content = append(content, block)
	}
	return &bedrockagentcore.InvokeCodeInterpreterInput{
		CodeInterpreterIdentifier: aws.String(Identifier(codeInterpreterIdentifier)),
		SessionId:                 aws.String(strings.TrimSpace(sessionID)),
		Name:                      agentcoretypes.ToolNameWriteFiles,
		Arguments:                 &agentcoretypes.ToolArguments{Content: content},
	}
}

// ReadFilesInput maps output paths into readFiles input.
func ReadFilesInput(
	codeInterpreterIdentifier, sessionID string,
	paths []string,
) *bedrockagentcore.InvokeCodeInterpreterInput {
	clean := make([]string, 0, len(paths))
	for _, path := range paths {
		clean = append(clean, strings.TrimSpace(path))
	}
	return &bedrockagentcore.InvokeCodeInterpreterInput{
		CodeInterpreterIdentifier: aws.String(Identifier(codeInterpreterIdentifier)),
		SessionId:                 aws.String(strings.TrimSpace(sessionID)),
		Name:                      agentcoretypes.ToolNameReadFiles,
		Arguments:                 &agentcoretypes.ToolArguments{Paths: clean},
	}
}

// StopInput maps session identity into StopCodeInterpreterSession input.
func StopInput(
	codeInterpreterIdentifier, sessionID, clientToken string,
) *bedrockagentcore.StopCodeInterpreterSessionInput {
	return &bedrockagentcore.StopCodeInterpreterSessionInput{
		CodeInterpreterIdentifier: aws.String(Identifier(codeInterpreterIdentifier)),
		SessionId:                 aws.String(strings.TrimSpace(sessionID)),
		ClientToken:               aws.String(ClientToken(clientToken)),
	}
}

// Identifier maps the AWS-owned Code Interpreter ARN to the data-plane ID.
func Identifier(identifier string) string {
	id := strings.TrimSpace(identifier)
	if strings.Contains(id, ":aws:code-interpreter/") &&
		strings.HasSuffix(id, "/"+agentCoreCodeInterpreterDefaultIdentifier) {
		return agentCoreCodeInterpreterDefaultIdentifier
	}
	return id
}

// NormalizeLanguage validates a requested language against supported and allowed values.
func NormalizeLanguage(language, defaultLanguage string, allowed []string) (string, error) {
	normalizedAllowed, err := NormalizeAllowedLanguages(allowed)
	if err != nil {
		return "", err
	}
	lang := strings.ToLower(strings.TrimSpace(language))
	if lang == "" {
		lang = strings.ToLower(strings.TrimSpace(defaultLanguage))
	}
	if lang == "" {
		lang = string(agentcoretypes.ProgrammingLanguagePython)
	}
	if !agentCoreCodeInterpreterLanguageSupported(lang) {
		return "", fmt.Errorf("agentcorecodeinterpreter: unsupported language %q", language)
	}
	if !slices.Contains(normalizedAllowed, lang) {
		return "", fmt.Errorf("agentcorecodeinterpreter: language %q is not allowed", lang)
	}
	return lang, nil
}

// NormalizeAllowedLanguages validates an allowlist.
func NormalizeAllowedLanguages(allowed []string) ([]string, error) {
	if len(allowed) == 0 {
		return agentCoreCodeInterpreterLanguages(), nil
	}
	out := make([]string, 0, len(allowed))
	for _, raw := range allowed {
		lang := strings.ToLower(strings.TrimSpace(raw))
		if lang == "" {
			continue
		}
		if !agentCoreCodeInterpreterLanguageSupported(lang) {
			return nil, fmt.Errorf("agentcorecodeinterpreter: unsupported allowed language %q", raw)
		}
		if !slices.Contains(out, lang) {
			out = append(out, lang)
		}
	}
	if len(out) == 0 {
		return nil, errors.New("agentcorecodeinterpreter: AllowedLanguages cannot contain only empty values")
	}
	return out, nil
}

// NormalizeRuntime validates an optional runtime.
func NormalizeRuntime(runtime string) (string, error) {
	rt := strings.ToLower(strings.TrimSpace(runtime))
	if rt == "" {
		return "", nil
	}
	if !agentCoreCodeInterpreterRuntimeSupported(rt) {
		return "", fmt.Errorf("agentcorecodeinterpreter: unsupported runtime %q", runtime)
	}
	return rt, nil
}

// Result maps one Code Interpreter result into ADK-friendly fields.
//
//nolint:gocognit,nestif // AgentCore result mapping has several optional structured fields and content blocks.
func Result(
	result agentcoretypes.CodeInterpreterResult,
	maxBytes int64,
) (map[string]any, []codeinterpretermodels.OutputArtifact, error) {
	isError := aws.ToBool(result.IsError)
	out := map[string]any{"is_error": isError}
	if isError {
		out["status"] = agentCoreCodeInterpreterStatusError
	} else {
		out["status"] = agentCoreCodeInterpreterStatusSuccess
	}

	truncated := false
	if sc := result.StructuredContent; sc != nil {
		if sc.Stdout != nil {
			out["stdout"], truncated = limitOutputString(aws.ToString(sc.Stdout), maxBytes, truncated)
		}
		if sc.Stderr != nil {
			out["stderr"], truncated = limitOutputString(aws.ToString(sc.Stderr), maxBytes, truncated)
		}
		if sc.ExitCode != nil {
			out["exit_code"] = *sc.ExitCode
			if *sc.ExitCode != 0 {
				out["status"] = agentCoreCodeInterpreterStatusError
				out["is_error"] = true
			}
		}
		if sc.ExecutionTime != nil {
			out["execution_time_ms"] = *sc.ExecutionTime
		}
		if sc.TaskId != nil {
			out["task_id"] = aws.ToString(sc.TaskId)
		}
		if sc.TaskStatus != "" {
			out["task_status"] = string(sc.TaskStatus)
			if sc.TaskStatus == agentcoretypes.TaskStatusFailed ||
				sc.TaskStatus == agentcoretypes.TaskStatusCanceled {
				out["status"] = agentCoreCodeInterpreterStatusError
				out["is_error"] = true
			}
		}
	}

	content := make([]map[string]any, 0, len(result.Content))
	artifacts := make([]codeinterpretermodels.OutputArtifact, 0)
	for i, block := range result.Content {
		item, artifact, blockTruncated, err := contentBlockToMapAndArtifact(block, i, maxBytes)
		if err != nil {
			return nil, nil, err
		}
		truncated = truncated || blockTruncated
		if len(item) > 0 {
			content = append(content, item)
		}
		if artifact.ArtifactName != "" {
			artifacts = append(artifacts, artifact)
		}
	}
	if len(content) > 0 {
		out["content"] = content
	}
	if truncated {
		out["truncated"] = true
	}
	return out, artifacts, nil
}

// ClientToken maps an ADK function-call ID to an AgentCore client token.
func ClientToken(functionCallID string) string {
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
	if len(s) > agentCoreCodeInterpreterMaxClientTokenLength {
		return uuid.NewSHA1(uuid.NameSpaceURL, []byte(s)).String()
	}
	if len(s) < agentCoreCodeInterpreterMinClientTokenLength {
		return s + "-" + uuid.NewSHA1(uuid.NameSpaceURL, []byte(s)).String()
	}
	return s
}

func contentBlockToMapAndArtifact(
	block agentcoretypes.ContentBlock,
	index int,
	maxBytes int64,
) (map[string]any, codeinterpretermodels.OutputArtifact, bool, error) {
	item := map[string]any{"type": string(block.Type)}
	if block.Name != nil {
		item["name"] = aws.ToString(block.Name)
	}
	if block.MimeType != nil {
		item["mime_type"] = aws.ToString(block.MimeType)
	}
	if block.Uri != nil {
		item["uri"] = aws.ToString(block.Uri)
	}
	if block.Size != nil {
		item["size"] = *block.Size
	}

	truncated := false
	if block.Text != nil {
		item["text"], truncated = limitOutputString(aws.ToString(block.Text), maxBytes, truncated)
	}
	if len(block.Data) > 0 {
		artifact, err := outputArtifact(
			aws.ToString(block.Name),
			aws.ToString(block.MimeType),
			aws.ToString(block.Uri),
			block.Data,
			"",
			false,
			index,
			maxBytes,
		)
		if err != nil {
			return nil, codeinterpretermodels.OutputArtifact{}, false, err
		}
		return item, artifact, truncated, nil
	}
	if block.Resource != nil {
		artifact, err := resourceArtifact(
			block.Resource,
			aws.ToString(block.Name),
			aws.ToString(block.Uri),
			index,
			maxBytes,
		)
		if err != nil {
			return nil, codeinterpretermodels.OutputArtifact{}, false, err
		}
		return item, artifact, truncated, nil
	}
	return item, codeinterpretermodels.OutputArtifact{}, truncated, nil
}

func resourceArtifact(
	resource *agentcoretypes.ResourceContent,
	name, uri string,
	index int,
	maxBytes int64,
) (codeinterpretermodels.OutputArtifact, error) {
	if resource == nil {
		return codeinterpretermodels.OutputArtifact{}, nil
	}
	path := firstNonEmpty(name, uri)
	if len(resource.Blob) > 0 {
		return outputArtifact(path, aws.ToString(resource.MimeType), uri, resource.Blob, "", false, index, maxBytes)
	}
	if resource.Text != nil {
		return outputArtifact(
			path,
			aws.ToString(resource.MimeType),
			uri,
			nil,
			aws.ToString(resource.Text),
			true,
			index,
			maxBytes,
		)
	}
	return codeinterpretermodels.OutputArtifact{}, nil
}

func outputArtifact(
	path, mimeType, uri string,
	data []byte,
	text string,
	isText bool,
	index int,
	maxBytes int64,
) (codeinterpretermodels.OutputArtifact, error) {
	size := int64(len(data))
	if isText {
		size = int64(len(text))
	}
	if maxBytes > 0 && size > maxBytes {
		return codeinterpretermodels.OutputArtifact{}, fmt.Errorf(
			"agentcorecodeinterpreter: output artifact size (%d bytes) exceeds maximum output size (%d bytes)",
			size,
			maxBytes,
		)
	}
	if mimeType == "" && path != "" {
		mimeType = bedrockmappers.MIMETypeFromExtension(path)
	}
	if mimeType == "" && isText {
		mimeType = "text/plain; charset=utf-8"
	}
	if mimeType == "" {
		mimeType = bedrockmappers.MIMETypeFromExtension(path)
	}
	if path == "" {
		path = fmt.Sprintf("code_interpreter_output_%d", index+1)
	}
	return codeinterpretermodels.OutputArtifact{
		Path:         firstNonEmpty(path, uri),
		ArtifactName: ArtifactName(path, index),
		MIMEType:     mimeType,
		Data:         data,
		Text:         text,
		IsText:       isText,
	}, nil
}

// ArtifactName returns a valid ADK artifact filename.
func ArtifactName(path string, index int) string {
	name := strings.TrimSpace(strings.ReplaceAll(path, "\\", "/"))
	if name != "" {
		name = filepath.Base(name)
	}
	if name == "" || name == "." || name == "/" {
		return fmt.Sprintf("code_interpreter_output_%d", index+1)
	}
	return name
}

func limitOutputString(s string, maxBytes int64, alreadyTruncated bool) (string, bool) {
	if maxBytes <= 0 || int64(len(s)) <= maxBytes {
		return s, alreadyTruncated
	}
	limit := min(int(maxBytes), len(s))
	for limit > 0 && !utf8.ValidString(s[:limit]) {
		limit--
	}
	return s[:limit] + "\n[truncated]", true
}

func agentCoreCodeInterpreterLanguages() []string {
	return []string{
		agentCoreCodeInterpreterLanguagePython,
		agentCoreCodeInterpreterLanguageJavascript,
		agentCoreCodeInterpreterLanguageTypescript,
	}
}

func agentCoreCodeInterpreterLanguageSupported(language string) bool {
	switch language {
	case agentCoreCodeInterpreterLanguagePython,
		agentCoreCodeInterpreterLanguageJavascript,
		agentCoreCodeInterpreterLanguageTypescript:
		return true
	default:
		return false
	}
}

func agentCoreCodeInterpreterRuntimeSupported(runtime string) bool {
	switch runtime {
	case agentCoreCodeInterpreterRuntimePython,
		agentCoreCodeInterpreterRuntimeNodeJS,
		agentCoreCodeInterpreterRuntimeDeno:
		return true
	default:
		return false
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
