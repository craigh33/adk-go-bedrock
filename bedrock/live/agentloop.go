package live

import (
	"context"
	"errors"
	"fmt"
	"iter"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers/sonic"
)

// ToolHandler executes one tool call invoked by the model. It receives the
// arguments map from the model's FunctionCall and returns the JSON-encodable
// result map that will be sent back as a FunctionResponse.
//
// Returning an error sends a FunctionResponse with an `{"error": "..."}` map
// and lets the session continue; the model can then react to the failure.
type ToolHandler func(ctx context.Context, args map[string]any) (map[string]any, error)

// ToolRegistry maps tool name → handler. Use it with [Session.RunAgentLoop].
//
// Callers building on top of adk-go's typed tools (`tool/functiontool`,
// `mcptoolset`) wrap each tool's invocation in a `ToolHandler` closure. We
// can't accept `map[string]tool.Tool` directly because ADK's `Run` method is
// only exposed on an unexported `runnableTool` interface and requires a
// `tool.Context` we can't synthesize outside the ADK runner.
type ToolRegistry map[string]ToolHandler

// ErrToolNotRegistered is the error a [ToolHandler] would have returned for an
// unregistered tool — surfaced via the FunctionResponse `error` field so the
// model sees a clear failure message instead of silent confusion.
var ErrToolNotRegistered = errors.New("bedrock/live: tool not registered")

// RunAgentLoop drains the session's iterator, auto-invoking tools in the
// registry when a FunctionCall arrives. It returns when the iterator emits a
// TurnComplete event, an error, or ctx is cancelled.
//
// This is the Live equivalent of adk-go's [Runner.RunLive] tool-execution
// behavior: the model calls a tool, we call the matching handler, and the
// result is shipped back as a FunctionResponse so the model can continue
// speaking. Manual control via [Session.Send] and the iterator remains
// available — don't combine them.
//
// emit is called for every received event so the caller can render audio,
// transcripts, interrupts, and (if they want) observe tool calls and results.
// Passing nil suppresses callbacks.
func (s *Session) RunAgentLoop(
	ctx context.Context,
	events iter.Seq2[*session.Event, error],
	tools ToolRegistry,
	emit func(*session.Event),
) error {
	ch := drainEvents(ctx, events)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case it, ok := <-ch:
			if !ok {
				return nil
			}
			done, err := s.handleAgentEvent(ctx, it, tools, emit)
			if err != nil {
				return err
			}
			if done {
				return nil
			}
		}
	}
}

type eventItem struct {
	ev  *session.Event
	err error
}

// drainEvents starts a goroutine that forwards iterator events onto a channel,
// closing the channel when the iterator exhausts or ctx is cancelled. Lets
// RunAgentLoop select on cancellation since [iter.Seq2] has no cancel hook.
func drainEvents(ctx context.Context, events iter.Seq2[*session.Event, error]) <-chan eventItem {
	ch := make(chan eventItem)
	go func() {
		defer close(ch)
		for ev, err := range events {
			select {
			case ch <- eventItem{ev, err}:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}

// handleAgentEvent processes one drained event. The first return is true when
// the turn has completed so the outer loop can exit.
func (s *Session) handleAgentEvent(
	ctx context.Context,
	it eventItem,
	tools ToolRegistry,
	emit func(*session.Event),
) (bool, error) {
	if it.err != nil {
		return false, it.err
	}
	if emit != nil && it.ev != nil {
		emit(it.ev)
	}
	if it.ev == nil {
		return false, nil
	}
	if err := s.dispatchToolCalls(ctx, it.ev, tools); err != nil {
		return false, err
	}
	return it.ev.TurnComplete, nil
}

// dispatchToolCalls invokes the matching ToolHandler for every FunctionCall
// part in ev and sends each result back as a FunctionResponse via the
// session's existing Send path.
func (s *Session) dispatchToolCalls(
	ctx context.Context,
	ev *session.Event,
	tools ToolRegistry,
) error {
	if ev.Content == nil {
		return nil
	}
	for _, part := range ev.Content.Parts {
		if part == nil || part.FunctionCall == nil {
			continue
		}
		fc := part.FunctionCall
		result, runErr := runToolHandler(ctx, tools, fc.Name, fc.Args)
		response := buildFunctionResponse(fc, result, runErr)
		if err := s.Send(agent.LiveRequest{Content: response}); err != nil {
			return fmt.Errorf("send tool result for %s: %w", fc.Name, err)
		}
	}
	return nil
}

func runToolHandler(
	ctx context.Context,
	tools ToolRegistry,
	name string,
	args map[string]any,
) (map[string]any, error) {
	handler, ok := tools[name]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrToolNotRegistered, name)
	}
	if args == nil {
		args = map[string]any{}
	}
	return handler(ctx, args)
}

// buildFunctionResponse wraps a tool result (or error) into the
// genai.Content shape Nova Sonic expects on the input side. On error, the
// response carries an {"error": "..."} map so the model can see the failure.
func buildFunctionResponse(fc *genai.FunctionCall, result map[string]any, runErr error) *genai.Content {
	if runErr != nil {
		result = map[string]any{"error": runErr.Error()}
	}
	if result == nil {
		result = map[string]any{}
	}
	return &genai.Content{
		Role: sonic.RoleTool,
		Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       fc.ID,
				Name:     fc.Name,
				Response: result,
			},
		}},
	}
}
