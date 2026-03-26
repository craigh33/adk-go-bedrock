package bedrock

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craig-hutcheon/adk-go-bedrock/bedrock/client"
	"github.com/craig-hutcheon/adk-go-bedrock/bedrock/mappers"
)

var _ model.LLM = (*Model)(nil)

// Options configures [New] and [NewWithAPI].
type Options struct {
	// Region overrides AWS region (otherwise [config.LoadDefaultConfig] resolution is used).
	Region string
}

// Model implements [model.LLM] using Amazon Bedrock Runtime Converse / ConverseStream.
type Model struct {
	modelID string
	api     client.RuntimeAPI
}

// New creates a [Model] using the default AWS configuration chain and a new
// [bedrockruntime.Client]. ModelID is the Bedrock model ID or inference profile ARN.
func New(ctx context.Context, modelID string, opts *Options) (*Model, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelID is required")
	}
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("load AWS config: %w", err)
	}
	if opts != nil && opts.Region != "" {
		cfg.Region = opts.Region
	}
	cli := bedrockruntime.NewFromConfig(cfg)
	return NewWithAPI(modelID, client.NewFromClient(cli))
}

// NewWithAPI wires a Bedrock runtime implementation (typically [bedrockruntime.Client] via [client.NewFromClient]).
func NewWithAPI(modelID string, api client.RuntimeAPI) (*Model, error) {
	if strings.TrimSpace(modelID) == "" {
		return nil, errors.New("modelID is required")
	}
	if api == nil {
		return nil, errors.New("nil RuntimeAPI")
	}
	return &Model{modelID: modelID, api: api}, nil
}

// Name returns the configured model identifier (see [New]).
func (m *Model) Name() string {
	if m == nil {
		return ""
	}
	return m.modelID
}

// GenerateContent calls Bedrock Converse or ConverseStream.
func (m *Model) GenerateContent(
	ctx context.Context,
	req *model.LLMRequest,
	stream bool,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m == nil || m.api == nil {
			yield(nil, errors.New("nil bedrock Model"))
			return
		}
		modelID := m.modelID
		if req != nil && req.Model != "" {
			modelID = req.Model
		}
		if stream {
			m.generateStream(ctx, modelID, req)(yield)
			return
		}
		m.generateUnary(ctx, modelID, req)(yield)
	}
}

func (m *Model) generateUnary(
	ctx context.Context,
	modelID string,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		in, err := mappers.ConverseInputFromLLMRequest(modelID, req)
		if err != nil {
			yield(nil, err)
			return
		}
		out, err := m.api.Converse(ctx, in)
		if err != nil {
			yield(nil, err)
			return
		}
		resp, err := mappers.LLMResponseFromConverseOutput(out)
		if !yield(resp, err) {
			return
		}
	}
}

//nolint:gocognit,funlen // Single stream handler: text deltas, tool deltas, and metadata into ADK responses.
func (m *Model) generateStream(
	ctx context.Context,
	modelID string,
	req *model.LLMRequest,
) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		in, err := mappers.ConverseStreamInputFromLLMRequest(modelID, req)
		if err != nil {
			yield(nil, err)
			return
		}
		stream, err := m.api.ConverseStream(ctx, in)
		if err != nil {
			yield(nil, err)
			return
		}
		defer stream.Close()

		var textBuf strings.Builder
		toolInputBuf := make(map[string]*strings.Builder)
		toolName := make(map[string]string)
		var lastUsage *genai.GenerateContentResponseUsageMetadata
		var stopReason types.StopReason

		for ev := range stream.Events() {
			switch v := ev.(type) {
			case *types.ConverseStreamOutputMemberContentBlockDelta:
				switch d := v.Value.Delta.(type) {
				case *types.ContentBlockDeltaMemberText:
					if d.Value != "" {
						textBuf.WriteString(d.Value)
						if !yield(&model.LLMResponse{
							Content: &genai.Content{
								Role: "model",
								Parts: []*genai.Part{{
									Text: textBuf.String(),
								}},
							},
							Partial: true,
						}, nil) {
							return
						}
					}
				case *types.ContentBlockDeltaMemberToolUse:
					frag := ""
					if d.Value.Input != nil {
						frag = *d.Value.Input
					}
					if frag != "" {
						var key string
						for k := range toolInputBuf {
							key = k
							break
						}
						if key == "" {
							key = "default"
							toolInputBuf[key] = &strings.Builder{}
						}
						if _, werr := toolInputBuf[key].WriteString(frag); werr != nil {
							yield(nil, werr)
							return
						}
					}
				}
			case *types.ConverseStreamOutputMemberContentBlockStart:
				if st, ok := v.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
					if st.Value.ToolUseId != nil {
						id := *st.Value.ToolUseId
						toolInputBuf[id] = &strings.Builder{}
						if st.Value.Name != nil {
							toolName[id] = *st.Value.Name
						}
					}
				}
			case *types.ConverseStreamOutputMemberMessageStop:
				stopReason = v.Value.StopReason
				_ = stopReason
			case *types.ConverseStreamOutputMemberMetadata:
				lastUsage = mappers.StreamMetadataToUsage(&v.Value)
			default:
				continue
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, err)
			return
		}

		var parts []*genai.Part
		if textBuf.Len() > 0 {
			parts = append(parts, &genai.Part{Text: textBuf.String()})
		}
		for id, b := range toolInputBuf {
			if b == nil {
				continue
			}
			name := toolName[id]
			if name == "" {
				name = id
			}
			args := map[string]any{}
			raw := b.String()
			if raw != "" {
				args["_toolInputJSON"] = raw
			}
			parts = append(parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					ID:   id,
					Name: name,
					Args: args,
				},
			})
		}
		if len(parts) == 0 {
			parts = []*genai.Part{{Text: ""}}
		}
		if !yield(&model.LLMResponse{
			Content: &genai.Content{
				Role:  "model",
				Parts: parts,
			},
			FinishReason:  mappers.StopReasonToFinishReason(stopReason),
			UsageMetadata: lastUsage,
			Partial:       false,
			TurnComplete:  true,
		}, nil) {
			return
		}
	}
}
