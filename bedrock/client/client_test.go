package client_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/craigh33/adk-go-bedrock/bedrock/client"
)

func TestNewFromClient_Converse(t *testing.T) {
	t.Parallel()

	const modelID = "test.model-id"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Helper()
		if r.Method != http.MethodPost {
			t.Errorf("got method %q, want POST", r.Method)
		}
		wantPath := "/model/" + modelID + "/converse"
		if r.URL.Path != wantPath {
			t.Errorf("got path %q, want %q", r.URL.Path, wantPath)
		}
		w.Header().Set("Content-Type", "application/json")
		body := map[string]any{
			"output": map[string]any{
				"message": map[string]any{
					"role": "assistant",
					"content": []any{
						map[string]any{"text": "hello"},
					},
				},
			},
			"stopReason": "end_turn",
			"usage": map[string]any{
				"inputTokens":  1,
				"outputTokens": 2,
				"totalTokens":  3,
			},
		}
		if err := json.NewEncoder(w).Encode(body); err != nil {
			t.Fatalf("encode: %v", err)
		}
	}))
	t.Cleanup(srv.Close)

	ctx := t.Context()
	cfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion("us-east-1"),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("AKID", "SECRET", "")),
		config.WithRetryer(func() aws.Retryer {
			return aws.NopRetryer{}
		}),
	)
	if err != nil {
		t.Fatalf("LoadDefaultConfig: %v", err)
	}
	br := bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.BaseEndpoint = aws.String(srv.URL)
	})
	api := client.NewFromClient(br)

	out, err := api.Converse(ctx, &bedrockruntime.ConverseInput{
		ModelId: aws.String(modelID),
		Messages: []types.Message{{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "hi"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("Converse: %v", err)
	}
	msg, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		t.Fatalf("output type %T", out.Output)
	}
	if msg.Value.Role != types.ConversationRoleAssistant {
		t.Fatalf("role: got %v", msg.Value.Role)
	}
	if len(msg.Value.Content) != 1 {
		t.Fatalf("content blocks: %d", len(msg.Value.Content))
	}
	txt, ok := msg.Value.Content[0].(*types.ContentBlockMemberText)
	if !ok || txt.Value != "hello" {
		t.Fatalf("text block: %#v", msg.Value.Content[0])
	}
	if out.StopReason != types.StopReasonEndTurn {
		t.Fatalf("stopReason: %v", out.StopReason)
	}
	if out.Usage == nil || out.Usage.InputTokens == nil || *out.Usage.InputTokens != 1 {
		t.Fatalf("usage: %#v", out.Usage)
	}
}

func TestNewFromClient_ConverseStream_Error(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "server error", http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	ctx := t.Context()
	cfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion("us-east-1"),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("AKID", "SECRET", "")),
		config.WithRetryer(func() aws.Retryer {
			return aws.NopRetryer{}
		}),
	)
	if err != nil {
		t.Fatalf("LoadDefaultConfig: %v", err)
	}
	br := bedrockruntime.NewFromConfig(cfg, func(o *bedrockruntime.Options) {
		o.BaseEndpoint = aws.String(srv.URL)
	})
	api := client.NewFromClient(br)

	_, err = api.ConverseStream(ctx, &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String("any"),
		Messages: []types.Message{{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "hi"},
			},
		}},
	})
	if err == nil {
		t.Fatal("ConverseStream: expected error from HTTP 500")
	}
}
