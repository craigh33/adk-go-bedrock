package codeinterpreter

import "time"

// StartSessionParams is the tool-neutral input for starting a Code Interpreter session.
type StartSessionParams struct {
	CodeInterpreterIdentifier string
	SessionName               string
	ClientToken               string
	MaxExecutionTime          time.Duration
}

// ExecuteParams is the tool-neutral input for executing code.
type ExecuteParams struct {
	CodeInterpreterIdentifier string
	SessionID                 string
	Code                      string
	Language                  string
	Runtime                   string
}

// InputFile is one writeFiles input.
type InputFile struct {
	Path   string
	Text   string
	Blob   []byte
	IsText bool
}

// OutputArtifact is file-like content returned by Code Interpreter.
type OutputArtifact struct {
	Path         string
	ArtifactName string
	MIMEType     string
	Data         []byte
	Text         string
	IsText       bool
}
