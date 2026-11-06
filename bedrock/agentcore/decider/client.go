package decider

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"

	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
)

const maxResponseBytes = 8 << 20

// AgentCoreAPI is the AWS operation consumed by the client.
type AgentCoreAPI interface {
	InvokeAgentRuntime(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput,
		...func(*bedrockagentcore.Options)) (*bedrockagentcore.InvokeAgentRuntimeOutput, error)
}

// Client evaluates questions against one deployed Runtime and session.
// It is safe for concurrent use; callers must not mutate requests during evaluation.
type Client struct {
	api        AgentCoreAPI
	runtimeARN string
	qualifier  string
	sessionID  string
}

func (c *Client) invoke(ctx context.Context, input *bedrockagentcore.InvokeAgentRuntimeInput) ([]byte, error) {
	out, err := c.api.InvokeAgentRuntime(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("decider: invoke runtime: %w", err)
	}
	if out == nil || out.Response == nil {
		return nil, errors.New("decider: runtime returned no response body")
	}
	switch {
	case ctx.Err() != nil:
		err = fmt.Errorf("decider: read response: %w", ctx.Err())
	case out.StatusCode == nil:
		err = errors.New("decider: runtime status is missing")
	case *out.StatusCode != http.StatusOK:
		err = fmt.Errorf("decider: unexpected runtime status: %d", *out.StatusCode)
	case out.ContentType == nil:
		err = errors.New("decider: response content type is missing")
	default:
		mediaType, _, parseErr := mime.ParseMediaType(*out.ContentType)
		if parseErr != nil || mediaType != "application/json" {
			err = errors.New("decider: expected an application/json response")
		}
	}
	if err != nil {
		_ = out.Response.Close()
		return nil, err
	}
	body, readErr := io.ReadAll(io.LimitReader(out.Response, maxResponseBytes+1))
	closeErr := out.Response.Close()
	if readErr != nil {
		return nil, fmt.Errorf("decider: read response: %w", readErr)
	}
	if closeErr != nil {
		return nil, fmt.Errorf("decider: close response: %w", closeErr)
	}
	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("decider: read response: %w", err)
	}
	if len(body) > maxResponseBytes {
		return nil, errors.New("decider: response exceeds 8 MiB")
	}
	return body, nil
}
