package decider

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/google/uuid"

	mappers "github.com/craigh33/adk-go-bedrock/internal/mappers/agentcore/decider"
)

// Config identifies an existing IAM-authenticated Decider Runtime.
// RuntimeSessionID binds the client to a caller-owned session when provided.
type Config struct {
	API              AgentCoreAPI
	AgentRuntimeARN  string
	Qualifier        string
	RuntimeSessionID string
}

// New creates a client, generating one reusable session ID when none is supplied.
// Create a separate client/session for each required isolation boundary.
func New(cfg Config) (*Client, error) {
	if cfg.API == nil {
		return nil, errors.New("decider: API is required")
	}
	runtimeARN := strings.TrimSpace(cfg.AgentRuntimeARN)
	if runtimeARN == "" {
		return nil, errors.New("decider: AgentRuntimeARN is required")
	}
	sessionID := strings.TrimSpace(cfg.RuntimeSessionID)
	if sessionID == "" {
		sessionID = uuid.NewString()
	}
	const minSessionIDLength, maxSessionIDLength = 33, 256
	if len(sessionID) < minSessionIDLength || len(sessionID) > maxSessionIDLength {
		return nil, errors.New("decider: RuntimeSessionID must contain 33 to 256 characters")
	}
	return &Client{api: cfg.API, runtimeARN: runtimeARN,
		qualifier: strings.TrimSpace(cfg.Qualifier), sessionID: sessionID}, nil
}

// Evaluate answers text or structured-state questions using the deployed checkpoint.
// Images are not supported. Model is passed through; the deployment selects the checkpoint.
// Routing, confidence thresholds, and fallback remain application decisions.
func (c *Client) Evaluate(ctx context.Context, req *Request) (*Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("decider: evaluate: %w", err)
	}
	if err := mappers.ValidateRequest(req); err != nil {
		return nil, fmt.Errorf("decider: invalid request: %w", err)
	}
	input, err := mappers.InvokeInput(req, c.runtimeARN, c.qualifier, c.sessionID)
	if err != nil {
		return nil, fmt.Errorf("decider: map request: %w", err)
	}
	body, err := c.invoke(ctx, input)
	if err != nil {
		return nil, err
	}
	response, err := mappers.DecodeResponse(body, req)
	if err != nil {
		return nil, fmt.Errorf("decider: invalid response: %w", err)
	}
	return response, nil
}
