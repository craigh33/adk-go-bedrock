// Package decider maps generated Decider payloads to the AgentCore invocation boundary.
package decider

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"unicode"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/smithy-go"

	models "github.com/craigh33/adk-go-bedrock/internal/models/agentcore/decider"
)

const (
	choiceType       = "choice"
	noulType         = "noul"
	scoreType        = "score"
	maxChoices       = 255
	minLevels        = 2
	maxLevels        = 10
	nullValue        = "null"
	probabilitiesKey = "probabilities"
)

// InvokeInput maps the payload and Runtime identity into the AWS SDK envelope.
func InvokeInput(req *models.SystemOneRequest, runtimeARN, qualifier, sessionID string) (
	*bedrockagentcore.InvokeAgentRuntimeInput, error,
) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encode payload: %w", err)
	}
	input := &bedrockagentcore.InvokeAgentRuntimeInput{
		AgentRuntimeArn: aws.String(runtimeARN), RuntimeSessionId: aws.String(sessionID),
		ContentType: aws.String("application/json"), Accept: aws.String("application/json"), Payload: body,
	}
	if qualifier != "" {
		input.Qualifier = aws.String(qualifier)
	}
	return input, nil
}

// ValidateRequest checks content shapes, upstream Python validators, and text-only capability.
func ValidateRequest(req *models.SystemOneRequest) error {
	if req == nil {
		return errors.New("request is required")
	}
	if req.Images != nil && len(*req.Images) != 0 {
		return errors.New("images are not supported by this adapter")
	}
	if len(req.Questions) == 0 {
		return errors.New("questions must not be empty")
	}
	if err := validateContent(req.State); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	for name, question := range req.Questions {
		if err := validateQuestion(question); err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
	}
	return nil
}

func validateQuestion(question models.SystemOneRequest_Questions_AdditionalProperties) error {
	body, err := question.MarshalJSON()
	if err != nil {
		return fmt.Errorf("encode question: %w", err)
	}
	fields, err := requiredFields(body, "instructions")
	if err != nil {
		return err
	}
	if err = validateContent(fields["instructions"]); err != nil {
		return fmt.Errorf("instructions: %w", err)
	}
	kind, err := questionKind(fields)
	if err != nil {
		return err
	}
	switch kind {
	case choiceType, noulType:
		var criteria map[string]any
		if raw := fields["criteria"]; len(raw) != 0 {
			if err = json.Unmarshal(raw, &criteria); err != nil {
				return fmt.Errorf("decode criteria: %w", err)
			}
		}
		if kind == choiceType {
			return validateChoice(criteria)
		}
		return validateNoul(criteria)
	case scoreType:
		return validateScore(fields["criteria"])
	default:
		return fmt.Errorf("unsupported type %q", kind)
	}
}

func validateChoice(criteria map[string]any) error {
	if len(criteria) < minLevels || len(criteria) > maxChoices {
		return errors.New("choice requires 2 to 255 options")
	}
	for _, description := range criteria {
		if description == nil {
			continue
		}
		if err := validateContent(description); err != nil {
			return fmt.Errorf("choice description: %w", err)
		}
	}
	return nil
}

func validateNoul(criteria map[string]any) error {
	for key, description := range criteria {
		if key != "true" && key != "false" {
			return errors.New("noul criteria must use true or false")
		}
		if description == nil {
			continue
		}
		if err := validateContent(description); err != nil {
			return fmt.Errorf("noul description: %w", err)
		}
	}
	return nil
}

func validateScore(body []byte) error {
	var levels []*string
	if err := json.Unmarshal(body, &levels); err != nil {
		return fmt.Errorf("decode score levels: %w", err)
	}
	if len(levels) < minLevels || len(levels) > maxLevels {
		return errors.New("score requires 2 to 10 string levels")
	}
	for _, level := range levels {
		if level == nil {
			return errors.New("score levels must be strings")
		}
	}
	return nil
}

func validateContent(value any) error {
	body, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode content: %w", err)
	}
	if len(body) == 0 || (body[0] != '"' && body[0] != '{' && body[0] != '[') {
		return errors.New("expected a string, object, or array")
	}
	return nil
}

func questionKind(fields map[string]json.RawMessage) (string, error) {
	if discriminator, ok := fields["type"]; ok {
		var kind string
		if err := json.Unmarshal(discriminator, &kind); err != nil {
			return "", fmt.Errorf("decode question type: %w", err)
		}
		return kind, nil
	}
	criteria := fields["criteria"]
	if len(criteria) != 0 && criteria[0] == '[' {
		return scoreType, nil
	}
	var options map[string]json.RawMessage
	if len(criteria) != 0 {
		if err := json.Unmarshal(criteria, &options); err != nil {
			return "", fmt.Errorf("decode criteria: %w", err)
		}
	}
	for key := range options {
		if key != "true" && key != "false" {
			return choiceType, nil
		}
	}
	return noulType, nil
}

// DecodeResponse decodes generated unions and checks answers against the request.
func DecodeResponse(body []byte, req *models.SystemOneRequest) (*models.SystemOneResponse, error) {
	fields, err := requiredFields(body, "model", "answers", "usage")
	if err != nil {
		var envelope struct {
			Error struct {
				Code    string  `json:"code"`
				Message *string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &envelope) == nil && envelope.Error.Code != "" && envelope.Error.Message != nil {
			return nil, &smithy.GenericAPIError{Code: envelope.Error.Code, Message: *envelope.Error.Message}
		}
		return nil, err
	}
	var response models.SystemOneResponse
	if err = json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if err = validateUsage(fields["usage"]); err != nil {
		return nil, err
	}
	if len(response.Answers) != len(req.Questions) {
		return nil, errors.New("answers do not match the requested questions")
	}
	for name, question := range req.Questions {
		answer, ok := response.Answers[name]
		if !ok {
			return nil, fmt.Errorf("answer %q is missing", name)
		}
		questionBody, encodeErr := question.MarshalJSON()
		if encodeErr != nil {
			return nil, fmt.Errorf("encode question %q: %w", name, encodeErr)
		}
		questionFields, decodeErr := requiredFields(questionBody, "instructions")
		if decodeErr != nil {
			return nil, fmt.Errorf("question %q: %w", name, decodeErr)
		}
		kind, kindErr := questionKind(questionFields)
		if kindErr != nil {
			return nil, fmt.Errorf("question %q: %w", name, kindErr)
		}
		if err = validateAnswer(kind, questionFields["criteria"], answer); err != nil {
			return nil, fmt.Errorf("answer %q: %w", name, err)
		}
	}
	return &response, nil
}

func validateUsage(body []byte) error {
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(body, &usage); err != nil {
		return fmt.Errorf("decode usage: %w", err)
	}
	for _, name := range []string{"input_tokens", "output_tokens"} {
		if raw, ok := usage[name]; ok {
			var value *int
			if err := json.Unmarshal(raw, &value); err != nil || value == nil || *value < 0 {
				return fmt.Errorf("usage %s must be a nonnegative integer", name)
			}
		}
	}
	return nil
}

func requiredFields(body []byte, names ...string) (map[string]json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, fmt.Errorf("decode payload: %w", err)
	}
	for _, name := range names {
		if value := fields[name]; len(value) == 0 || string(value) == nullValue {
			return nil, fmt.Errorf("missing %s", name)
		}
	}
	return fields, nil
}

func validateAnswer(kind string, criteria json.RawMessage,
	answer models.SystemOneResponse_Answers_AdditionalProperties) error {
	body, err := json.Marshal(answer)
	if err != nil {
		return fmt.Errorf("encode answer: %w", err)
	}
	required := []string{"noul"}
	switch kind {
	case choiceType:
		required = []string{"choice", "confidence", probabilitiesKey}
	case scoreType:
		required = []string{"score", "confidence", probabilitiesKey, "legend"}
	}
	fields, err := requiredFields(body, required...)
	if err != nil {
		return err
	}
	if value, ok := fields["type"]; ok {
		var actual string
		if err = json.Unmarshal(value, &actual); err != nil || actual != kind {
			return errors.New("answer type does not match the question")
		}
	}
	switch kind {
	case noulType:
		value, decodeErr := answer.AsNoulAnswer()
		if decodeErr != nil {
			return fmt.Errorf("decode noul: %w", decodeErr)
		}
		if value.Noul < 0 || value.Noul > 1 {
			return errors.New("noul must be between zero and one")
		}
		return nil
	case choiceType:
		var options map[string]any
		if err = json.Unmarshal(criteria, &options); err != nil {
			return fmt.Errorf("decode choice criteria: %w", err)
		}
		value, decodeErr := answer.AsChoiceAnswer()
		if decodeErr != nil {
			return fmt.Errorf("decode choice: %w", decodeErr)
		}
		if _, ok := options[value.Choice]; !ok {
			return errors.New("choice is not a configured option")
		}
		return validateProbabilities(fields[probabilitiesKey], value.Confidence, options)
	case scoreType:
		return validateScoreAnswer(answer, fields, criteria)
	default:
		return fmt.Errorf("unsupported type %q", kind)
	}
}

func validateScoreAnswer(answer models.SystemOneResponse_Answers_AdditionalProperties,
	fields map[string]json.RawMessage, criteria json.RawMessage) error {
	var levels []string
	if err := json.Unmarshal(criteria, &levels); err != nil {
		return fmt.Errorf("decode score criteria: %w", err)
	}
	value, err := answer.AsScoreAnswer()
	if err != nil {
		return fmt.Errorf("decode score: %w", err)
	}
	var legend map[string]*string
	if err := json.Unmarshal(fields["legend"], &legend); err != nil {
		return fmt.Errorf("decode legend: %w", err)
	}
	if value.Score < 0 || value.Score > float64(len(levels)-1) ||
		len(value.Legend) != len(levels) {
		return errors.New("score or legend does not match the rubric")
	}
	options := make(map[string]any, len(levels))
	for index, level := range levels {
		key := strconv.Itoa(index)
		// The upstream renderer uses Python str.strip(), including U+001C through U+001F.
		label := strings.TrimFunc(level, func(r rune) bool {
			return unicode.IsSpace(r) || (r >= '\u001c' && r <= '\u001f')
		})
		if description := legend[key]; description == nil || *description != label {
			return errors.New("score legend does not match the rubric")
		}
		options[key] = level
	}
	return validateProbabilities(fields[probabilitiesKey], value.Confidence, options)
}

func validateProbabilities(body []byte, confidence float64, options map[string]any) error {
	var probabilities map[string]*float64
	if err := json.Unmarshal(body, &probabilities); err != nil {
		return fmt.Errorf("decode probabilities: %w", err)
	}
	if confidence < 0 || confidence > 1 || len(probabilities) != len(options) {
		return errors.New("confidence or probabilities are invalid")
	}
	for key := range options {
		value, ok := probabilities[key]
		if !ok || value == nil || *value < 0 || *value > 1 {
			return errors.New("probabilities do not match the configured options")
		}
	}
	return nil
}
