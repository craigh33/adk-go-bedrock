package bedrock

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

type guardrailContextKey struct{}

// GuardrailConfig selects a Bedrock guardrail for Converse / ConverseStream.
// Identifier and Version must both be non-empty (after trimming); otherwise the
// config is treated as unset. Trace is optional; when empty it is omitted from
// the AWS request.
type GuardrailConfig struct {
	Identifier string
	Version    string
	Trace      types.GuardrailTrace
}

// ContextWithGuardrail attaches a per-request guardrail that overrides the model
// default from [WithGuardrail] or [Options.Guardrail]. Pass nil to clear (not typical).
func ContextWithGuardrail(ctx context.Context, cfg *GuardrailConfig) context.Context {
	return context.WithValue(ctx, guardrailContextKey{}, cfg)
}

func guardrailFromContext(ctx context.Context) *GuardrailConfig {
	v, _ := ctx.Value(guardrailContextKey{}).(*GuardrailConfig)
	return v
}

func (c *GuardrailConfig) toAWS() (*types.GuardrailConfiguration, error) {
	if c == nil {
		return nil, nil //nolint:nilnil // Explicit unset.
	}
	id := strings.TrimSpace(c.Identifier)
	ver := strings.TrimSpace(c.Version)
	if id == "" && ver == "" {
		return nil, nil //nolint:nilnil // No guardrail.
	}
	if id == "" || ver == "" {
		return nil, errors.New("bedrock guardrail: identifier and version must both be set")
	}
	out := &types.GuardrailConfiguration{
		GuardrailIdentifier: aws.String(id),
		GuardrailVersion:    aws.String(ver),
	}
	if c.Trace != "" {
		out.Trace = c.Trace
	}
	return out, nil
}

func resolveGuardrailConfig(ctx context.Context, defaultCfg *GuardrailConfig) (*types.GuardrailConfiguration, error) {
	if g := guardrailFromContext(ctx); g != nil {
		conv, err := g.toAWS()
		if err != nil {
			return nil, fmt.Errorf("request guardrail: %w", err)
		}
		if conv != nil {
			return conv, nil
		}
	}
	if defaultCfg != nil {
		conv, err := defaultCfg.toAWS()
		if err != nil {
			return nil, fmt.Errorf("model guardrail: %w", err)
		}
		return conv, nil
	}
	return nil, nil //nolint:nilnil // No guardrail.
}
