package bedrock

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/adk/model"
	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/internal/mappers"
)

type fakeAPI struct {
	converseOut *bedrockruntime.ConverseOutput
	converseErr error

	stream    StreamReader
	streamErr error
}

func (f *fakeAPI) Converse(
	ctx context.Context,
	params *bedrockruntime.ConverseInput,
	optFns ...func(*bedrockruntime.Options),
) (*bedrockruntime.ConverseOutput, error) {
	_ = ctx
	_ = params
	_ = optFns
	if f.converseErr != nil {
		return nil, f.converseErr
	}
	return f.converseOut, nil
}

func (f *fakeAPI) ConverseStream(
	ctx context.Context,
	params *bedrockruntime.ConverseStreamInput,
	optFns ...func(*bedrockruntime.Options),
) (StreamReader, error) {
	_ = ctx
	_ = params
	_ = optFns
	if f.streamErr != nil {
		return nil, f.streamErr
	}
	return f.stream, nil
}

type fakeStream struct {
	ch  chan types.ConverseStreamOutput
	err error
}

func (f *fakeStream) Events() <-chan types.ConverseStreamOutput { return f.ch }
func (f *fakeStream) Close() error                              { return nil }
func (f *fakeStream) Err() error                                { return f.err }

func TestConverse_GenerateContent_unary(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{
		converseOut: &bedrockruntime.ConverseOutput{
			Output: &types.ConverseOutputMemberMessage{
				Value: types.Message{
					Role: types.ConversationRoleAssistant,
					Content: []types.ContentBlock{
						&types.ContentBlockMemberText{Value: "ok"},
					},
				},
			},
			StopReason: types.StopReasonEndTurn,
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(1),
				OutputTokens: aws.Int32(1),
				TotalTokens:  aws.Int32(2),
			},
		},
	}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var got int
	for r, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatal(err)
		}
		got++
		if r.Content.Parts[0].Text != "ok" {
			t.Fatalf("text %q", r.Content.Parts[0].Text)
		}
	}
	if got != 1 {
		t.Fatalf("responses: %d", got)
	}
}

func TestConverse_GenerateContent_stream(t *testing.T) {
	t.Parallel()
	idx := int32(0)
	ch := make(chan types.ConverseStreamOutput, 4)
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta:             &types.ContentBlockDeltaMemberText{Value: "hel"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta:             &types.ContentBlockDeltaMemberText{Value: "lo"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberMetadata{
		Value: types.ConverseStreamMetadataEvent{
			Usage: &types.TokenUsage{
				InputTokens:  aws.Int32(2),
				OutputTokens: aws.Int32(2),
				TotalTokens:  aws.Int32(4),
			},
		},
	}
	close(ch)

	api := &fakeAPI{stream: &fakeStream{ch: ch}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}
	var partial, final int
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if r.Partial {
			partial++
		} else {
			final++
		}
	}
	if partial < 1 || final != 1 {
		t.Fatalf("partial=%d final=%d", partial, final)
	}
}

func TestConverse_GenerateContent_streamCitationDelta(t *testing.T) {
	t.Parallel()
	idx := int32(0)
	ch := make(chan types.ConverseStreamOutput, 8)
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta: &types.ContentBlockDeltaMemberCitation{
				Value: types.CitationsDelta{
					Title: aws.String("Story"),
					Location: &types.CitationLocationMemberWeb{
						Value: types.WebLocation{
							Url:    aws.String("https://news.example/item"),
							Domain: aws.String("news.example"),
						},
					},
				},
			},
		},
	}
	ch <- &types.ConverseStreamOutputMemberMessageStop{
		Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn},
	}
	close(ch)

	api := &fakeAPI{stream: &fakeStream{ch: ch}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var final *model.LLMResponse
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if !r.Partial {
			final = r
		}
	}
	if final == nil || final.Content == nil || len(final.Content.Parts) != 1 {
		t.Fatalf("final: %+v", final)
	}
	meta := final.Content.Parts[0].PartMetadata
	raw, ok := meta[mappers.PartMetadataKeyBedrockCitations].([]any)
	if !ok || len(raw) != 1 {
		t.Fatalf("citations metadata: %+v", meta)
	}
	cm, ok := raw[0].(map[string]any)
	if !ok || cm["title"] != "Story" {
		t.Fatalf("citation row: %+v", raw[0])
	}
}

func TestConverse_GenerateContent_streamTextAndCitationSameSlot(t *testing.T) {
	t.Parallel()
	idx := int32(0)
	ch := make(chan types.ConverseStreamOutput, 8)
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta:             &types.ContentBlockDeltaMemberText{Value: "Hello"},
		},
	}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{
		Value: types.ContentBlockDeltaEvent{
			ContentBlockIndex: &idx,
			Delta: &types.ContentBlockDeltaMemberCitation{
				Value: types.CitationsDelta{
					Title: aws.String("Story"),
					Location: &types.CitationLocationMemberWeb{
						Value: types.WebLocation{
							Url:    aws.String("https://news.example/item"),
							Domain: aws.String("news.example"),
						},
					},
				},
			},
		},
	}
	ch <- &types.ConverseStreamOutputMemberMessageStop{
		Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn},
	}
	close(ch)

	api := &fakeAPI{stream: &fakeStream{ch: ch}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var final *model.LLMResponse
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if !r.Partial {
			final = r
		}
	}
	if final == nil || final.Content == nil || len(final.Content.Parts) != 1 {
		t.Fatalf("final: %+v", final)
	}
	p := final.Content.Parts[0]
	if p.Text != "Hello" {
		t.Fatalf("text: got %q", p.Text)
	}
	meta := p.PartMetadata
	raw, ok := meta[mappers.PartMetadataKeyBedrockCitations].([]any)
	if !ok || len(raw) != 1 {
		t.Fatalf("citations metadata: %+v", meta)
	}
	cm, ok := raw[0].(map[string]any)
	if !ok || cm["title"] != "Story" {
		t.Fatalf("citation row: %+v", raw[0])
	}
}

func TestConverse_GenerateContent_streamToolCalls_parseAndRawArgs(t *testing.T) {
	t.Parallel()

	idx0 := int32(0)
	idx1 := int32(1)
	toolAID := "tool-a"
	toolAName := "get_weather"
	toolBID := "tool-b"
	toolBName := "get_time"

	ch := make(chan types.ConverseStreamOutput, 8)
	ch <- &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: &idx0,
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
			ToolUseId: &toolAID,
			Name:      &toolAName,
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &idx0,
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
			Input: aws.String("{\"city\":\"Dub"),
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: &idx1,
		Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{
			ToolUseId: &toolBID,
			Name:      &toolBName,
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &idx0,
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
			Input: aws.String("lin\"}"),
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &idx1,
		Delta: &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{
			Input: aws.String("{not-json}"),
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}}
	close(ch)

	api := &fakeAPI{stream: &fakeStream{ch: ch}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var final *model.LLMResponse
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if !r.Partial {
			final = r
		}
	}
	if final == nil {
		t.Fatal("missing final response")
	}
	if final.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish reason: got %v", final.FinishReason)
	}
	if final.Content == nil || len(final.Content.Parts) != 2 {
		t.Fatalf("parts: %+v", final.Content)
	}
	callA := final.Content.Parts[0].FunctionCall
	callB := final.Content.Parts[1].FunctionCall
	if callA == nil || callB == nil {
		t.Fatalf("expected function calls, got %+v", final.Content.Parts)
	}
	if callA.Name != toolAName || callA.ID != toolAID {
		t.Fatalf("callA: %+v", callA)
	}
	if _, ok := callA.Args[rawFunctionArgsJSONKey]; ok {
		t.Fatalf("callA args should not include rawArgsJson when JSON parses: %+v", callA.Args)
	}
	if callA.Args["city"] != "Dublin" {
		t.Fatalf("callA args: %+v", callA.Args)
	}
	if callB.Name != toolBName || callB.ID != toolBID {
		t.Fatalf("callB: %+v", callB)
	}
	if callB.Args[rawFunctionArgsJSONKey] != "{not-json}" {
		t.Fatalf("callB args: %+v", callB.Args)
	}
}

func TestConverse_GenerateContent_streamError(t *testing.T) {
	t.Parallel()

	ch := make(chan types.ConverseStreamOutput)
	close(ch)
	api := &fakeAPI{stream: &fakeStream{ch: ch, err: errors.New("stream failed")}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var gotErr error
	for _, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			gotErr = err
		}
	}
	if gotErr == nil || gotErr.Error() != "stream failed" {
		t.Fatalf("error: %v", gotErr)
	}
}

func TestConverse_GenerateContent_streamImageReasoningAndGuardrailMetadata(t *testing.T) {
	t.Parallel()

	imgIdx := int32(0)
	reasonIdx := int32(1)
	ch := make(chan types.ConverseStreamOutput, 8)
	ch <- &types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{
		ContentBlockIndex: &imgIdx,
		Start:             &types.ContentBlockStartMemberImage{Value: types.ImageBlockStart{Format: types.ImageFormatPng}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &imgIdx,
		Delta:             &types.ContentBlockDeltaMemberImage{Value: types.ImageBlockDelta{Source: &types.ImageSourceMemberBytes{Value: []byte{0x01, 0x02}}}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &reasonIdx,
		Delta:             &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"}},
	}}
	ch <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		ContentBlockIndex: &reasonIdx,
		Delta:             &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig-1"}},
	}}
	ch <- &types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
		Usage: &types.TokenUsage{InputTokens: aws.Int32(1), OutputTokens: aws.Int32(1), TotalTokens: aws.Int32(2)},
		Trace: &types.ConverseStreamTrace{Guardrail: &types.GuardrailTraceAssessment{
			OutputAssessments: map[string][]types.GuardrailAssessment{
				"0": {{ContentPolicy: &types.GuardrailContentPolicyAssessment{Filters: []types.GuardrailContentFilter{{
					Type:           types.GuardrailContentFilterTypeHate,
					Confidence:     types.GuardrailContentFilterConfidenceHigh,
					FilterStrength: types.GuardrailContentFilterStrengthHigh,
					Action:         types.GuardrailContentPolicyActionBlocked,
				}}}}},
			},
		}},
	}}
	ch <- &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonGuardrailIntervened}}
	close(ch)

	api := &fakeAPI{stream: &fakeStream{ch: ch}}
	m, err := NewWithAPI("mid", api)
	if err != nil {
		t.Fatal(err)
	}
	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", "user")},
		Config:   &genai.GenerateContentConfig{},
	}

	var final *model.LLMResponse
	for r, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatal(err)
		}
		if !r.Partial {
			final = r
		}
	}
	if final == nil {
		t.Fatal("missing final response")
	}
	if final.FinishReason != genai.FinishReasonSafety {
		t.Fatalf("finish reason: %v", final.FinishReason)
	}
	if final.Content == nil || len(final.Content.Parts) != 2 {
		t.Fatalf("parts: %+v", final.Content)
	}
	if final.Content.Parts[0].InlineData == nil || final.Content.Parts[0].InlineData.MIMEType != "image/png" {
		t.Fatalf("image part: %+v", final.Content.Parts[0])
	}
	if !final.Content.Parts[1].Thought || final.Content.Parts[1].Text != "thinking" ||
		string(final.Content.Parts[1].ThoughtSignature) != "sig-1" {
		t.Fatalf("reasoning part: %+v", final.Content.Parts[1])
	}
	ratings, ok := final.CustomMetadata["safety_ratings"].([]*genai.SafetyRating)
	if !ok || len(ratings) != 1 || ratings[0].Category != genai.HarmCategoryHateSpeech {
		t.Fatalf("custom metadata: %+v", final.CustomMetadata)
	}
}

// Minimal successful Converse REST JSON (restjson) shape for a mocked Bedrock endpoint.
const testConverseOKJSON = `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`

func newTestBedrockClient(t *testing.T, srv *httptest.Server) *bedrockruntime.Client {
	t.Helper()
	return bedrockruntime.New(bedrockruntime.Options{
		Region: "us-east-1",
		// Static credentials satisfy the SDK auth middleware against a local httptest server.
		Credentials:  credentials.NewStaticCredentialsProvider("test-akid", "test-secret", ""),
		BaseEndpoint: aws.String(srv.URL),
		HTTPClient:   srv.Client(),
	})
}

func findEndedSpan(recorder *tracetest.SpanRecorder, name string) sdktrace.ReadOnlySpan {
	for _, s := range recorder.Ended() {
		if s.Name() == name {
			return s
		}
	}
	return nil
}

func TestNewRuntimeAPI_WithTracerProvider_Converse_recordsClientSpan(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(testConverseOKJSON))
	}))
	defer srv.Close()

	cli := newTestBedrockClient(t, srv)
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	api := NewRuntimeAPI(cli, WithTracerProvider(tp))

	_, err := api.Converse(context.Background(), &bedrockruntime.ConverseInput{
		ModelId: aws.String("eu.amazon.nova-2-lite-v1:0"),
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

	span := findEndedSpan(sr, "bedrockruntime.Converse")
	if span == nil {
		t.Fatalf("no bedrockruntime.Converse span; ended=%d", len(sr.Ended()))
	}
	if span.InstrumentationScope().Name != otelTracerName {
		t.Fatalf("scope name: got %q want %q", span.InstrumentationScope().Name, otelTracerName)
	}
	if span.SpanKind() != trace.SpanKindClient {
		t.Fatalf("span kind: got %v want client", span.SpanKind())
	}
	if span.Status().Code != codes.Ok {
		t.Fatalf("status: got %v (%q) want Ok", span.Status().Code, span.Status().Description)
	}
}

func TestNewRuntimeAPI_usesGlobalTracerProviderWhenOptionOmitted(t *testing.T) {
	// Not parallel: mutates the process-global OTel tracer provider.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(testConverseOKJSON))
	}))
	defer srv.Close()

	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })

	sr := tracetest.NewSpanRecorder()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr)))

	cli := newTestBedrockClient(t, srv)
	api := NewRuntimeAPI(cli)

	_, err := api.Converse(context.Background(), &bedrockruntime.ConverseInput{
		ModelId: aws.String("eu.amazon.nova-2-lite-v1:0"),
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

	if findEndedSpan(sr, "bedrockruntime.Converse") == nil {
		t.Fatalf("expected global tracer provider to be used; ended=%d", len(sr.Ended()))
	}
}

func TestNewRuntimeAPI_WithTracerProvider_Converse_recordsErrorSpan(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"__type":"ValidationException","message":"invalid"}`))
	}))
	defer srv.Close()

	cli := newTestBedrockClient(t, srv)
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	api := NewRuntimeAPI(cli, WithTracerProvider(tp))

	_, err := api.Converse(context.Background(), &bedrockruntime.ConverseInput{
		ModelId: aws.String("eu.amazon.nova-2-lite-v1:0"),
		Messages: []types.Message{{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "hi"},
			},
		}},
	})
	if err == nil {
		t.Fatal("expected error from Converse")
	}

	span := findEndedSpan(sr, "bedrockruntime.Converse")
	if span == nil {
		t.Fatalf("no bedrockruntime.Converse span; ended=%d", len(sr.Ended()))
	}
	if span.Status().Code != codes.Error {
		t.Fatalf("status: got %v want Error", span.Status().Code)
	}
}

func TestNewRuntimeAPI_WithTracerProvider_ConverseStream_recordsErrorSpan(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"__type":"ValidationException","message":"invalid stream"}`))
	}))
	defer srv.Close()

	cli := newTestBedrockClient(t, srv)
	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	api := NewRuntimeAPI(cli, WithTracerProvider(tp))

	_, err := api.ConverseStream(context.Background(), &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String("eu.amazon.nova-2-lite-v1:0"),
		Messages: []types.Message{{
			Role: types.ConversationRoleUser,
			Content: []types.ContentBlock{
				&types.ContentBlockMemberText{Value: "hi"},
			},
		}},
	})
	if err == nil {
		t.Fatal("expected error from ConverseStream")
	}

	span := findEndedSpan(sr, "bedrockruntime.ConverseStream")
	if span == nil {
		t.Fatalf("no bedrockruntime.ConverseStream span; ended=%d", len(sr.Ended()))
	}
	if span.Status().Code != codes.Error {
		t.Fatalf("status: got %v want Error", span.Status().Code)
	}
}

// TestTracedStreamReader_Close_recordsStreamErr verifies that when Events()
// drains and Err() returns a non-nil error, Close() ends the span with
// status=Error and records the error.
func TestTracedStreamReader_Close_recordsStreamErr(t *testing.T) {
	t.Parallel()

	sr := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(sr))
	_, span := tp.Tracer(otelTracerName).Start(context.Background(),
		"bedrockruntime.ConverseStream", trace.WithSpanKind(trace.SpanKindClient))

	ch := make(chan types.ConverseStreamOutput)
	close(ch)
	streamErr := errors.New("mid-stream failure")
	inner := &fakeStream{ch: ch, err: streamErr}

	reader := &tracedStreamReader{inner: inner, span: span}

	// Drain the (already-closed) events channel.
	for range reader.Events() {
	}

	// Close records stream.Err() and ends the span.
	if err := reader.Close(); err != nil {
		t.Fatalf("Close returned unexpected error: %v", err)
	}

	s := findEndedSpan(sr, "bedrockruntime.ConverseStream")
	if s == nil {
		t.Fatalf("span not ended after Close(); ended=%d", len(sr.Ended()))
	}
	if s.Status().Code != codes.Error {
		t.Fatalf("status: got %v want Error", s.Status().Code)
	}
	if len(s.Events()) == 0 {
		t.Fatal("expected recorded error event on span")
	}
}
