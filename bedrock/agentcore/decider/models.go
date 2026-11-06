// Package decider evaluates typed questions using Strands Decider hosted in AgentCore Runtime.
package decider

import models "github.com/craigh33/adk-go-bedrock/internal/models/agentcore/decider"

type (
	// Request is the generated invocation payload.
	Request = models.SystemOneRequest

	// Response is the generated evaluation result.
	Response = models.SystemOneResponse

	// SystemOneRequest retains the upstream schema name.
	SystemOneRequest = models.SystemOneRequest

	// SystemOneResponse retains the upstream schema name.
	SystemOneResponse = models.SystemOneResponse

	// Question is the generated question union.
	Question = models.SystemOneRequest_Questions_AdditionalProperties

	// Answer is the generated answer union.
	Answer = models.SystemOneResponse_Answers_AdditionalProperties

	// ChoiceQuestion selects among named options.
	ChoiceQuestion = models.ChoiceQuestion

	// NoulQuestion asks a yes/no question.
	NoulQuestion = models.NoulQuestion

	// ScoreQuestion evaluates an ordered rubric.
	ScoreQuestion = models.ScoreQuestion

	// ChoiceAnswer contains a selection and its distribution.
	ChoiceAnswer = models.ChoiceAnswer

	// NoulAnswer contains the probability of true.
	NoulAnswer = models.NoulAnswer

	// ScoreAnswer contains a score, distribution, and rubric.
	ScoreAnswer = models.ScoreAnswer

	// Usage contains the upstream token counts.
	Usage = models.Usage
)
