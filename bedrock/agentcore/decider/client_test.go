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
	"github.com/aws/smithy-go"
)

type brokenReader struct{ cause error }

func (r brokenReader) Read([]byte) (int, error) { return 0, r.cause }

func TestReadFailuresAndCancellationCloseBody(t *testing.T) {
	want := errors.New("response read failed")
	var req Request
	json.Unmarshal([]byte(requestFixture), &req)
	for _, cancelDuringCall := range []bool{false, true} {
		t.Run(map[bool]string{false: "read failure", true: "cancellation"}[cancelDuringCall], func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			var reader io.Reader = brokenReader{cause: want}
			if cancelDuringCall {
				reader = strings.NewReader(responseFixture)
			}
			body := &trackedBody{Reader: reader}
			api := invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
				*bedrockagentcore.InvokeAgentRuntimeOutput, error,
			) {
				if cancelDuringCall {
					cancel()
				}
				return &bedrockagentcore.InvokeAgentRuntimeOutput{Response: body,
					StatusCode: aws.Int32(http.StatusOK), ContentType: aws.String("application/json")}, nil
			})
			client, err := New(Config{API: api, AgentRuntimeARN: "runtime"})
			if err != nil {
				t.Fatal(err)
			}
			_, err = client.Evaluate(ctx, &req)
			cause := want
			if cancelDuringCall {
				cause = context.Canceled
			}
			if !errors.Is(err, cause) || !body.closed {
				t.Fatalf("error cause or cleanup lost: %v, closed=%v", err, body.closed)
			}
		})
	}
}

func TestIncompleteSDKResponses(t *testing.T) {
	var req Request
	json.Unmarshal([]byte(requestFixture), &req)
	for _, name := range []string{"nil response", "nil body", "no status", "no content type"} {
		t.Run(name, func(t *testing.T) {
			body := &trackedBody{Reader: strings.NewReader(responseFixture)}
			out := &bedrockagentcore.InvokeAgentRuntimeOutput{Response: body,
				StatusCode: aws.Int32(http.StatusOK), ContentType: aws.String("application/json")}
			switch name {
			case "nil response":
				out = nil
			case "nil body":
				out.Response = nil
			case "no status":
				out.StatusCode = nil
			case "no content type":
				out.ContentType = nil
			}
			api := invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
				*bedrockagentcore.InvokeAgentRuntimeOutput, error,
			) {
				return out, nil
			})
			client, err := New(Config{API: api, AgentRuntimeARN: "runtime"})
			if err != nil {
				t.Fatal(err)
			}
			if _, err = client.Evaluate(t.Context(), &req); err == nil {
				t.Fatal("incomplete SDK response accepted")
			}
			if out != nil && out.Response != nil && !body.closed {
				t.Fatal("response body leaked")
			}
		})
	}
}

func TestEvaluatePreservesRuntimeErrorEnvelope(t *testing.T) {
	var req Request
	if err := json.Unmarshal([]byte(requestFixture), &req); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		code, message, body string
	}{
		{"loading", "the model is still loading", `{"error":{"code":"loading","message":"the model is still loading","state":"loading"}}`},
		{"invalid_request", "score requires at least two levels", `{"error":{"code":"invalid_request","message":"score requires at least two levels"}}`},
		{"inference_error", "model failed", `{"error":{"code":"inference_error","message":"model failed"}}`},
	} {
		t.Run(test.code, func(t *testing.T) {
			body := &trackedBody{Reader: strings.NewReader(test.body)}
			api := invokeFunc(func(context.Context, *bedrockagentcore.InvokeAgentRuntimeInput) (
				*bedrockagentcore.InvokeAgentRuntimeOutput, error,
			) {
				return &bedrockagentcore.InvokeAgentRuntimeOutput{Response: body,
					StatusCode: aws.Int32(http.StatusOK), ContentType: aws.String("application/json")}, nil
			})
			client, err := New(Config{API: api, AgentRuntimeARN: "runtime"})
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Evaluate(t.Context(), &req)
			var runtimeErr smithy.APIError
			if response != nil || !errors.As(err, &runtimeErr) || runtimeErr.ErrorCode() != test.code ||
				runtimeErr.ErrorMessage() != test.message {
				t.Fatalf("expected runtime error %q with message %q, got %v", test.code, test.message, err)
			}
			if !body.closed {
				t.Fatal("response body leaked")
			}
		})
	}
}

func TestUpstreamDefaultsAndModelPassThrough(t *testing.T) {
	requestJSON := requestFixture
	responseJSON := responseFixture
	for _, kind := range []string{"choice", "noul", "score"} {
		requestJSON = strings.ReplaceAll(requestJSON, `"type":"`+kind+`",`, "")
		responseJSON = strings.ReplaceAll(responseJSON, `"type":"`+kind+`",`, "")
	}
	responseJSON = strings.Replace(responseJSON, `"usage":{`, `"usage":{"future_metadata":"ignored",`, 1)
	var req Request
	if err := json.Unmarshal([]byte(requestJSON), &req); err != nil {
		t.Fatal(err)
	}
	req.Model = aws.String("caller-alias")
	sessionID := "12345678-1234-1234-1234-123456789012"
	api := invokeFunc(func(_ context.Context, input *bedrockagentcore.InvokeAgentRuntimeInput) (
		*bedrockagentcore.InvokeAgentRuntimeOutput, error,
	) {
		var sent Request
		if err := json.Unmarshal(input.Payload, &sent); err != nil {
			t.Fatal(err)
		}
		if sent.Model == nil || *sent.Model != "caller-alias" || *input.RuntimeSessionId != sessionID ||
			input.Qualifier != nil {
			t.Fatalf("caller configuration changed: %+v", input)
		}
		return &bedrockagentcore.InvokeAgentRuntimeOutput{Response: io.NopCloser(strings.NewReader(responseJSON)),
			StatusCode: aws.Int32(http.StatusOK), ContentType: aws.String("application/json")}, nil
	})
	client, err := New(Config{API: api, AgentRuntimeARN: "runtime", RuntimeSessionID: sessionID})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = client.Evaluate(t.Context(), &req); err != nil {
		t.Fatalf("valid upstream defaults rejected: %v", err)
	}
}
