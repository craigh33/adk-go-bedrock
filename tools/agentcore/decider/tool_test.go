package decider

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolconfirmation"

	client "github.com/craigh33/adk-go-bedrock/bedrock/agentcore/decider"
)

type evaluatorFunc func(context.Context, *client.Request) (*client.Response, error)

func (f evaluatorFunc) Evaluate(ctx context.Context, req *client.Request) (*client.Response, error) {
	return f(ctx, req)
}

type toolContext struct{ agent.StrictContextMock }

func (*toolContext) ToolConfirmation() *toolconfirmation.ToolConfirmation { return nil }

type runnableTool interface {
	Run(agent.Context, any) (map[string]any, error)
	ProcessRequest(agent.Context, *model.LLMRequest) error
}

func TestToolUsesFixedQuestionsAndPreservesResult(t *testing.T) {
	var question client.Question
	if err := question.FromNoulQuestion(client.NoulQuestion{Type: "noul", Instructions: "Urgent?"}); err != nil {
		t.Fatal(err)
	}
	questions := map[string]client.Question{"urgent": question}
	var response client.Response
	json.Unmarshal([]byte(`{"model":"decider","answers":{"urgent":{"type":"noul","noul":0}},"usage":{}}`), &response)
	calls := 0
	api := evaluatorFunc(func(ctx context.Context, req *client.Request) (*client.Response, error) {
		calls++
		if ctx.Err() != nil || req.State != "No rush" || len(req.Questions) != 1 {
			t.Fatalf("unexpected request: %+v", req)
		}
		got, err := req.Questions["urgent"].AsNoulQuestion()
		if err != nil || got.Instructions != "Urgent?" {
			t.Fatalf("fixed question changed: %+v, %v", got, err)
		}
		delete(req.Questions, "urgent")
		return &response, nil
	})
	value, err := New(Config{API: api, Questions: questions})
	if err != nil {
		t.Fatal(err)
	}
	delete(questions, "urgent")
	tool := value.(runnableTool)
	ctx := &toolContext{StrictContextMock: agent.StrictContextMock{Ctx: t.Context()}}
	llmRequest := &model.LLMRequest{}
	if err = tool.ProcessRequest(ctx, llmRequest); err != nil {
		t.Fatal(err)
	}
	if llmRequest.Tools[value.Name()] != value {
		t.Fatal("tool was not registered")
	}
	for range 2 {
		result, runErr := tool.Run(ctx, map[string]any{"state": "No rush"})
		if runErr != nil {
			t.Fatal(runErr)
		}
		answers := result["answers"].(map[string]any)
		if result["model"] != "decider" || answers["urgent"].(map[string]any)["noul"] != float64(0) {
			t.Fatalf("result lost fields or zero probability: %+v", result)
		}
	}
	if calls != 2 {
		t.Fatalf("expected two evaluations, got %d", calls)
	}
}

func TestToolErrors(t *testing.T) {
	want := errors.New("evaluation failed")
	var question client.Question
	question.FromNoulQuestion(client.NoulQuestion{Instructions: "Urgent?"})
	api := evaluatorFunc(func(context.Context, *client.Request) (*client.Response, error) { return nil, want })
	for _, cfg := range []Config{{}, {API: api}, {API: api, Questions: map[string]client.Question{"bad": {}}}} {
		if _, err := New(cfg); err == nil {
			t.Fatal("invalid configuration accepted")
		}
	}
	value, err := New(Config{API: api, Name: "urgency", Description: "Assess urgency",
		Questions: map[string]client.Question{"urgent": question}})
	if err != nil {
		t.Fatal(err)
	}
	ctx := &toolContext{StrictContextMock: agent.StrictContextMock{Ctx: t.Context()}}
	tool := value.(runnableTool)
	if _, err = tool.Run(ctx, map[string]any{"state": "hello"}); !errors.Is(err, want) {
		t.Fatalf("evaluation error lost: %v", err)
	}
	if _, err = tool.Run(ctx, map[string]any{}); err == nil {
		t.Fatal("missing state accepted")
	}
}
