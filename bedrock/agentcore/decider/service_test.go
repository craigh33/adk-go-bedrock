package decider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
)

const requestFixture = `{"state":{"text":"Charged twice"},"questions":{
"route":{"type":"choice","instructions":"Route it","criteria":{"billing":"Money","technical":null}},
"urgent":{"type":"noul","instructions":"Urgent?","criteria":{"true":"Immediate","false":null}},
"severity":{"type":"score","instructions":"Severity?","criteria":["Low","High"]}}}`

const responseFixture = `{"model":"decider-test","answers":{
"route":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.9,"technical":0.1}},
"urgent":{"type":"noul","noul":0},
"severity":{"type":"score","score":0.5,"confidence":0.4,"legend":{"0":"Low","1":"High"},"probabilities":{"0":0.5,"1":0.5}}},
"usage":{"input_tokens":10,"output_tokens":3},"latency_ms":123}`

type invokeFunc func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
	*bedrockagentcore.InvokeAgentRuntimeOutput, error,
)

func (f invokeFunc) InvokeAgentRuntime(ctx context.Context, input *bedrockagentcore.InvokeAgentRuntimeInput,
	_ ...func(*bedrockagentcore.Options)) (*bedrockagentcore.InvokeAgentRuntimeOutput, error) {
	return f(ctx, input)
}

type trackedBody struct {
	io.Reader

	closed   bool
	closeErr error
}

func (b *trackedBody) Close() error {
	b.closed = true
	return b.closeErr
}

//nolint:gocognit // Assert all three union members and session identity in the same round trip.
func TestEvaluateRoundTripAndSessionReuse(t *testing.T) {
	var req Request
	requestJSON := strings.Replace(requestFixture, `["Low","High"]`, `["\u001c Low ","High\n"]`, 1)
	if err := json.Unmarshal([]byte(requestJSON), &req); err != nil {
		t.Fatal(err)
	}
	var inputs []*bedrockagentcore.InvokeAgentRuntimeInput
	var bodies []*trackedBody
	api := invokeFunc(func(ctx context.Context, input *bedrockagentcore.InvokeAgentRuntimeInput) (
		*bedrockagentcore.InvokeAgentRuntimeOutput, error,
	) {
		if ctx != t.Context() {
			t.Fatal("evaluation context was replaced")
		}
		inputs = append(inputs, input)
		body := &trackedBody{Reader: strings.NewReader(responseFixture)}
		bodies = append(bodies, body)
		return &bedrockagentcore.InvokeAgentRuntimeOutput{Response: body,
			StatusCode: aws.Int32(http.StatusOK), ContentType: aws.String("application/json; charset=utf-8")}, nil
	})
	client, err := New(Config{API: api, AgentRuntimeARN: "runtime-arn", Qualifier: "production"})
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		response, evaluateErr := client.Evaluate(t.Context(), &req)
		if evaluateErr != nil {
			t.Fatal(evaluateErr)
		}
		route, decodeErr := response.Answers["route"].AsChoiceAnswer()
		if decodeErr != nil || route.Choice != "billing" || route.Confidence != 0.8 {
			t.Fatalf("unexpected choice: %+v, %v", route, decodeErr)
		}
		urgent, decodeErr := response.Answers["urgent"].AsNoulAnswer()
		if decodeErr != nil || urgent.Noul != 0 {
			t.Fatalf("zero probability lost: %+v, %v", urgent, decodeErr)
		}
		severity, decodeErr := response.Answers["severity"].AsScoreAnswer()
		if decodeErr != nil || severity.Score != 0.5 || severity.Legend["1"] != "High" {
			t.Fatalf("unexpected score: %+v, %v", severity, decodeErr)
		}
		if response.Model != "decider-test" || *response.Usage.InputTokens != 10 {
			t.Fatalf("metadata lost: %+v", response)
		}
	}
	if len(inputs) != 2 || *inputs[0].RuntimeSessionId != *inputs[1].RuntimeSessionId ||
		len(*inputs[0].RuntimeSessionId) != 36 {
		t.Fatal("client did not reuse its generated session")
	}
	for index, input := range inputs {
		if !bodies[index].closed || *input.AgentRuntimeArn != "runtime-arn" ||
			*input.Qualifier != "production" || *input.ContentType != "application/json" ||
			*input.Accept != "application/json" {
			t.Fatalf("incorrect invocation or unclosed body: %+v", input)
		}
		var payload Request
		if err = json.Unmarshal(input.Payload, &payload); err != nil {
			t.Fatal(err)
		}
		if payload.Model != nil || len(payload.Questions) != 3 || req.Model != nil {
			t.Fatal("payload changed or Jev defaults leaked into invocation")
		}
	}
}

func TestEvaluateRejectsInvalidRequestsBeforeInvoking(t *testing.T) {
	tests := map[string]string{
		"empty questions":        `{"state":"","questions":{}}`,
		"null question":          `{"state":"","questions":{"x":null}}`,
		"numeric state":          strings.Replace(requestFixture, `{"text":"Charged twice"}`, "42", 1),
		"null instructions":      strings.Replace(requestFixture, `"Urgent?"`, "null", 1),
		"mis-cased instructions": `{"state":"","questions":{"x":{"type":"noul","Instructions":"Urgent?"}}}`,
		"mis-cased criteria":     `{"state":"","questions":{"x":{"type":"choice","instructions":"Route it","Criteria":{"a":null,"b":null}}}}`,
		"single choice":          `{"state":"","questions":{"x":{"type":"choice","instructions":"","criteria":{"a":null}}}}`,
		"invalid noul key":       `{"state":"","questions":{"x":{"type":"noul","instructions":"","criteria":{"other":null}}}}`,
		"single score":           `{"state":"","questions":{"x":{"type":"score","instructions":"","criteria":["Low"]}}}`,
		"null score level":       strings.Replace(requestFixture, `["Low","High"]`, `[null,"High"]`, 1),
		"unknown type":           strings.Replace(requestFixture, `"type":"noul"`, `"type":"box"`, 1),
		"images": strings.Replace(
			requestFixture,
			`"questions":`,
			`"images":["base64"],"questions":`,
			1,
		),
	}
	unexpectedInvocation := errors.New("invalid request reached AWS")
	api := invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
		*bedrockagentcore.InvokeAgentRuntimeOutput, error,
	) {
		return nil, unexpectedInvocation
	})
	client, err := New(Config{API: api, AgentRuntimeARN: "runtime"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = client.Evaluate(t.Context(), nil); err == nil || errors.Is(err, unexpectedInvocation) {
		t.Fatalf("expected preflight error for nil request, got %v", err)
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			var req Request
			if decodeErr := json.Unmarshal([]byte(body), &req); decodeErr != nil {
				t.Fatal(decodeErr)
			}
			if _, evaluateErr := client.Evaluate(t.Context(), &req); evaluateErr == nil ||
				errors.Is(evaluateErr, unexpectedInvocation) {
				t.Fatalf("expected preflight error, got %v", evaluateErr)
			}
		})
	}
}

func TestEvaluateRejectsMalformedResponsesAndClosesBodies(t *testing.T) {
	tests := map[string]string{
		"invalid JSON":       "{",
		"missing model":      strings.Replace(responseFixture, `"model":"decider-test",`, "", 1),
		"null usage":         strings.Replace(responseFixture, `{"input_tokens":10,"output_tokens":3}`, "null", 1),
		"missing answer":     strings.Replace(responseFixture, `"urgent":`, `"other":`, 1),
		"wrong type":         strings.Replace(responseFixture, `"type":"noul"`, `"type":"choice"`, 1),
		"missing noul":       strings.Replace(responseFixture, `,"noul":0`, "", 1),
		"missing confidence": strings.Replace(responseFixture, `"confidence":0.8,`, "", 1),
		"null probability":   strings.Replace(responseFixture, `"billing":0.9`, `"billing":null`, 1),
		"invalid confidence": strings.Replace(responseFixture, `"confidence":0.8`, `"confidence":1.1`, 1),
		"unknown choice":     strings.Replace(responseFixture, `"choice":"billing"`, `"choice":"other"`, 1),
		"negative tokens":    strings.Replace(responseFixture, `"input_tokens":10`, `"input_tokens":-1`, 1),
		"wrong legend":       strings.Replace(responseFixture, `"1":"High"`, `"1":"Different"`, 1),
		"null legend":        strings.Replace(responseFixture, `"1":"High"`, `"1":null`, 1),
		"null tokens":        strings.Replace(responseFixture, `"input_tokens":10`, `"input_tokens":null`, 1),
		"oversized":          strings.Repeat(" ", maxResponseBytes+1),
	}
	for name, payload := range tests {
		t.Run(name, func(t *testing.T) {
			body := &trackedBody{Reader: strings.NewReader(payload)}
			client, err := New(Config{AgentRuntimeARN: "runtime",
				API: invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
					*bedrockagentcore.InvokeAgentRuntimeOutput, error,
				) {
					return &bedrockagentcore.InvokeAgentRuntimeOutput{Response: body,
						ContentType: aws.String("application/json"), StatusCode: aws.Int32(http.StatusOK)}, nil
				})})
			if err != nil {
				t.Fatal(err)
			}
			var req Request
			json.Unmarshal([]byte(requestFixture), &req)
			if _, err = client.Evaluate(t.Context(), &req); err == nil {
				t.Fatal("malformed response accepted")
			}
			if !body.closed {
				t.Fatal("body not closed")
			}
		})
	}
}

func TestRuntimeFailures(t *testing.T) {
	want := errors.New("AWS unavailable")
	var req Request
	json.Unmarshal([]byte(requestFixture), &req)
	api := invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
		*bedrockagentcore.InvokeAgentRuntimeOutput, error,
	) {
		return nil, want
	})
	client, err := New(Config{API: api, AgentRuntimeARN: "runtime"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = client.Evaluate(t.Context(), &req); !errors.Is(err, want) {
		t.Fatalf("AWS error lost: %v", err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err = client.Evaluate(ctx, &req); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
	for _, test := range []struct {
		name, contentType string
		status            int32
		closeErr          error
	}{
		{name: "content type", contentType: "text/event-stream", status: http.StatusOK},
		{name: "status", contentType: "application/json", status: http.StatusBadRequest},
		{name: "close", contentType: "application/json", status: http.StatusOK, closeErr: want},
	} {
		t.Run(test.name, func(t *testing.T) {
			unexpectedRead := errors.New("read body despite invalid response metadata")
			var reader io.Reader = brokenReader{cause: unexpectedRead}
			if test.closeErr != nil {
				reader = strings.NewReader(responseFixture)
			}
			body := &trackedBody{Reader: reader, closeErr: test.closeErr}
			client.api = invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
				*bedrockagentcore.InvokeAgentRuntimeOutput, error,
			) {
				return &bedrockagentcore.InvokeAgentRuntimeOutput{Response: body,
					ContentType: &test.contentType, StatusCode: &test.status}, nil
			})
			if _, err = client.Evaluate(t.Context(), &req); err == nil || !body.closed {
				t.Fatalf("expected failure and closed body: %v", err)
			}
			if errors.Is(err, unexpectedRead) {
				t.Fatal(err)
			}
		})
	}
}

func TestConfigAndExplicitSession(t *testing.T) {
	api := invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
		*bedrockagentcore.InvokeAgentRuntimeOutput, error,
	) {
		return nil, errors.New("unexpected invocation")
	})
	for _, cfg := range []Config{
		{}, {API: api}, {API: api, AgentRuntimeARN: "runtime", RuntimeSessionID: "short"},
		{API: api, AgentRuntimeARN: "runtime", RuntimeSessionID: strings.Repeat("x", 257)},
	} {
		if _, err := New(cfg); err == nil {
			t.Fatalf("invalid configuration accepted: %+v", cfg)
		}
	}
	id := "12345678-1234-1234-1234-123456789012"
	client, err := New(Config{API: api, AgentRuntimeARN: " runtime ", RuntimeSessionID: id})
	if err != nil || client.sessionID != id || client.runtimeARN != "runtime" {
		t.Fatalf("explicit identity changed: %+v, %v", client, err)
	}
}
