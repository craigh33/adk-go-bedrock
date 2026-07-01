package mantle

import (
	"errors"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/craigh33/adk-go-bedrock/bedrock"
)

// Anthropic Messages streaming event type discriminators.
const (
	eventMessageStart      = "message_start"
	eventContentBlockStart = "content_block_start"
	eventContentBlockDelta = "content_block_delta"
	eventMessageDelta      = "message_delta"
)

// Anthropic content-block-delta type discriminators.
const (
	deltaText      = "text_delta"
	deltaThinking  = "thinking_delta"
	deltaSignature = "signature_delta"
	deltaInputJSON = "input_json_delta"
)

const contentBlockTypeToolUse = "tool_use"

// MessageParamsFromConverseStreamInput maps a [bedrockruntime.ConverseStreamInput]
// to Anthropic [anthropic.MessageNewParams]. It mirrors the Converse-side
// ConverseStreamInputFromLLMRequest by reusing the unary mapping over the shared
// request fields.
func MessageParamsFromConverseStreamInput(
	in *bedrockruntime.ConverseStreamInput,
) (anthropic.MessageNewParams, error) {
	if in == nil {
		return anthropic.MessageNewParams{}, errors.New("nil ConverseStreamInput")
	}
	return MessageParamsFromConverseInput(&bedrockruntime.ConverseInput{
		ModelId:         in.ModelId,
		Messages:        in.Messages,
		System:          in.System,
		InferenceConfig: in.InferenceConfig,
		ToolConfig:      in.ToolConfig,
	})
}

// converseStream adapts an Anthropic Messages SSE stream to the Bedrock
// [bedrock.StreamReader] contract, translating each Anthropic event into the
// [types.ConverseStreamOutput] variants that streamState.consumeEvent expects.
type converseStream struct {
	src       *ssestream.Stream[anthropic.MessageStreamEventUnion]
	events    chan types.ConverseStreamOutput
	done      chan struct{}
	closeOnce sync.Once
	err       error
}

var _ bedrock.StreamReader = (*converseStream)(nil)

func newConverseStream(src *ssestream.Stream[anthropic.MessageStreamEventUnion]) *converseStream {
	cs := &converseStream{
		src:    src,
		events: make(chan types.ConverseStreamOutput),
		done:   make(chan struct{}),
	}
	go cs.pump()
	return cs
}

// pump drains the Anthropic stream and forwards translated events. It exits when
// the source is exhausted or the consumer calls Close (via the done channel),
// so an early break in the caller's range loop cannot leak this goroutine.
func (c *converseStream) pump() {
	defer close(c.events)
	var t eventTranslator
	for c.src.Next() {
		for _, out := range t.translate(c.src.Current()) {
			select {
			case c.events <- out:
			case <-c.done:
				return
			}
		}
	}
	// Written before the deferred close(events); the channel close establishes
	// the happens-before edge for a subsequent Err read by the consumer.
	c.err = c.src.Err()
}

func (c *converseStream) Events() <-chan types.ConverseStreamOutput {
	return c.events
}

func (c *converseStream) Err() error {
	return c.err
}

func (c *converseStream) Close() error {
	c.closeOnce.Do(func() { close(c.done) })
	return c.src.Close()
}

// eventTranslator carries the small amount of cross-event state needed to
// reconstruct Converse-shaped usage: Anthropic reports input tokens on
// message_start and output tokens on message_delta.
type eventTranslator struct {
	inputTokens int64
}

func (t *eventTranslator) translate(ev anthropic.MessageStreamEventUnion) []types.ConverseStreamOutput {
	switch ev.Type {
	case eventMessageStart:
		t.inputTokens = ev.Message.Usage.InputTokens
		return nil
	case eventContentBlockStart:
		return contentBlockStartOutputs(ev)
	case eventContentBlockDelta:
		return contentBlockDeltaOutputs(ev)
	case eventMessageDelta:
		return t.messageDeltaOutputs(ev)
	default:
		// content_block_stop, message_stop, and unknown events need no mapping;
		// streamState assembles the final response after the stream closes.
		return nil
	}
}

func contentBlockStartOutputs(ev anthropic.MessageStreamEventUnion) []types.ConverseStreamOutput {
	// Only tool_use blocks carry a start payload (the id and name) that
	// streamState needs; text and thinking blocks are assembled from deltas.
	if ev.ContentBlock.Type != contentBlockTypeToolUse {
		return nil
	}
	return []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockStart{
			Value: types.ContentBlockStartEvent{
				ContentBlockIndex: aws.Int32(int32FromInt64(ev.Index)),
				Start: &types.ContentBlockStartMemberToolUse{
					Value: types.ToolUseBlockStart{
						ToolUseId: aws.String(ev.ContentBlock.ID),
						Name:      aws.String(ev.ContentBlock.Name),
					},
				},
			},
		},
	}
}

func contentBlockDeltaOutputs(ev anthropic.MessageStreamEventUnion) []types.ConverseStreamOutput {
	delta := contentBlockDelta(ev.Delta)
	if delta == nil {
		return nil
	}
	return []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberContentBlockDelta{
			Value: types.ContentBlockDeltaEvent{
				ContentBlockIndex: aws.Int32(int32FromInt64(ev.Index)),
				Delta:             delta,
			},
		},
	}
}

func contentBlockDelta(delta anthropic.MessageStreamEventUnionDelta) types.ContentBlockDelta {
	switch delta.Type {
	case deltaText:
		if delta.Text == "" {
			return nil
		}
		return &types.ContentBlockDeltaMemberText{Value: delta.Text}
	case deltaThinking:
		return &types.ContentBlockDeltaMemberReasoningContent{
			Value: &types.ReasoningContentBlockDeltaMemberText{Value: delta.Thinking},
		}
	case deltaSignature:
		return &types.ContentBlockDeltaMemberReasoningContent{
			Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: delta.Signature},
		}
	case deltaInputJSON:
		return &types.ContentBlockDeltaMemberToolUse{
			Value: types.ToolUseBlockDelta{Input: aws.String(delta.PartialJSON)},
		}
	default:
		return nil
	}
}

func (t *eventTranslator) messageDeltaOutputs(ev anthropic.MessageStreamEventUnion) []types.ConverseStreamOutput {
	out := make([]types.ConverseStreamOutput, 0, 2)
	out = append(out, &types.ConverseStreamOutputMemberMetadata{
		Value: types.ConverseStreamMetadataEvent{
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(int32FromInt64(t.inputTokens)),
				OutputTokens: aws.Int32(int32FromInt64(ev.Usage.OutputTokens)),
				TotalTokens:  aws.Int32(int32FromInt64(t.inputTokens + ev.Usage.OutputTokens)),
			},
		},
	})
	if ev.Delta.StopReason != "" {
		out = append(out, &types.ConverseStreamOutputMemberMessageStop{
			Value: types.MessageStopEvent{StopReason: stopReasonFromAnthropic(ev.Delta.StopReason)},
		})
	}
	return out
}
