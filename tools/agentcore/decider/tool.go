// Package decider exposes a configured decision evaluator as an ADK function tool.
package decider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	deciderclient "github.com/craigh33/adk-go-bedrock/bedrock/agentcore/decider"
	mappers "github.com/craigh33/adk-go-bedrock/internal/mappers/agentcore/decider"
)

// Evaluator is the decision operation consumed by the ADK tool.
type Evaluator interface {
	Evaluate(context.Context, *deciderclient.Request) (*deciderclient.Response, error)
}

// Config fixes the questions used for every tool invocation.
type Config struct {
	API         Evaluator
	Name        string
	Description string
	Questions   map[string]deciderclient.Question
}

// Input is the text supplied by the calling agent.
type Input struct {
	State string `json:"state" jsonschema:"Text to evaluate against the configured questions"`
}

// New creates an ADK tool with immutable application-defined questions.
func New(cfg Config) (tool.Tool, error) {
	if cfg.API == nil {
		return nil, errors.New("decider tool: API is required")
	}
	// Copy through the generated contract so later map edits cannot change the tool's questions.
	body, err := json.Marshal(cfg.Questions)
	if err != nil {
		return nil, fmt.Errorf("decider tool: encode questions: %w", err)
	}
	var questions map[string]deciderclient.Question
	if err = json.Unmarshal(body, &questions); err != nil {
		return nil, fmt.Errorf("decider tool: decode questions: %w", err)
	}
	if err = mappers.ValidateRequest(&deciderclient.Request{State: "", Questions: questions}); err != nil {
		return nil, fmt.Errorf("decider tool: invalid questions: %w", err)
	}
	name := strings.TrimSpace(cfg.Name)
	if name == "" {
		name = "agentcore_decider"
	}
	description := cfg.Description
	if description == "" {
		description = "Evaluate text against predefined decision questions and return choices, scores, and probabilities."
	}
	return functiontool.New(functiontool.Config{Name: name, Description: description},
		func(ctx agent.Context, input Input) (map[string]any, error) {
			var callQuestions map[string]deciderclient.Question
			if decodeErr := json.Unmarshal(body, &callQuestions); decodeErr != nil {
				return nil, fmt.Errorf("decider tool: copy questions: %w", decodeErr)
			}
			response, evaluateErr := cfg.API.Evaluate(ctx, &deciderclient.Request{
				State: input.State, Questions: callQuestions,
			})
			if evaluateErr != nil {
				return nil, fmt.Errorf("decider tool: evaluate: %w", evaluateErr)
			}
			if response == nil {
				return nil, errors.New("decider tool: evaluator returned no response")
			}
			result, encodeErr := json.Marshal(response)
			if encodeErr != nil {
				return nil, fmt.Errorf("decider tool: encode result: %w", encodeErr)
			}
			var record map[string]any
			if decodeErr := json.Unmarshal(result, &record); decodeErr != nil {
				return nil, fmt.Errorf("decider tool: decode result: %w", decodeErr)
			}
			return record, nil
		})
}
