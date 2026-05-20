package sonic

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

func TestSonicInferenceConfigFromGenai(t *testing.T) {
	t.Parallel()
	if got := sonicInferenceConfigFromGenai(nil); got != nil {
		t.Fatalf("nil config should yield nil, got %+v", got)
	}
	temp := float32(0.3)
	topP := float32(0.9)
	cfg := &genai.GenerateContentConfig{
		Temperature:     &temp,
		TopP:            &topP,
		MaxOutputTokens: 256,
		StopSequences:   []string{"###"}, // intentionally dropped
	}
	got := sonicInferenceConfigFromGenai(cfg)
	if got == nil {
		t.Fatal("expected non-nil config")
	}
	if got.Temperature == nil || *got.Temperature != 0.3 {
		t.Fatalf("temperature = %v", got.Temperature)
	}
	if got.TopP == nil || *got.TopP != 0.9 {
		t.Fatalf("topP = %v", got.TopP)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 256 {
		t.Fatalf("maxTokens = %v", got.MaxTokens)
	}
}

func TestBuildSessionStartSerializesInferenceConfig(t *testing.T) {
	t.Parallel()
	temp := float32(0.0)
	raw, err := BuildSessionStart(&genai.GenerateContentConfig{Temperature: &temp})
	if err != nil {
		t.Fatalf("BuildSessionStart: %v", err)
	}
	if !strings.Contains(string(raw), `"temperature":0`) {
		t.Fatalf("temperature missing from sessionStart: %s", raw)
	}
}

func TestBuildSessionStartOmitsInferenceWhenUnset(t *testing.T) {
	t.Parallel()
	raw, err := BuildSessionStart(nil)
	if err != nil {
		t.Fatalf("BuildSessionStart: %v", err)
	}
	if strings.Contains(string(raw), "inferenceConfiguration") {
		t.Fatalf("expected no inferenceConfiguration field: %s", raw)
	}
}

func TestToolEntriesFromADKRejectsUnsupportedVariant(t *testing.T) {
	t.Parallel()
	tools := map[string]any{
		"search": &genai.Tool{GoogleSearch: &genai.GoogleSearch{}},
	}
	_, err := toolEntriesFromADK(tools)
	if err == nil {
		t.Fatal("expected error for GoogleSearch variant")
	}
	if !errors.Is(err, ErrUnsupportedTool) {
		t.Fatalf("err should wrap ErrUnsupportedTool: %v", err)
	}
	if !strings.Contains(err.Error(), "GoogleSearch") {
		t.Fatalf("error should mention variant name: %v", err)
	}
}

func TestToolEntriesFromADKRejectsUnknownValueType(t *testing.T) {
	t.Parallel()
	tools := map[string]any{"x": 42}
	_, err := toolEntriesFromADK(tools)
	if err == nil {
		t.Fatal("expected error for non-tool value")
	}
	if !errors.Is(err, ErrUnsupportedTool) {
		t.Fatalf("err should wrap ErrUnsupportedTool: %v", err)
	}
	if !strings.Contains(err.Error(), "int") {
		t.Fatalf("error should mention type: %v", err)
	}
}

func TestReadStateBuffersUnknownEvent(t *testing.T) {
	t.Parallel()
	rs := NewReadState()
	// Unknown event yields no response but is buffered.
	unknown := []byte(`{"event":{"futureEvent":{"foo":1}}}`)
	resp, err := rs.Consume(unknown)
	if err != nil {
		t.Fatalf("Consume unknown: %v", err)
	}
	if resp != nil {
		t.Fatalf("unknown event shouldn't emit a response: %+v", resp)
	}
	// completionEnd flushes the buffered unknown events into CustomMetadata.
	end := mustWrap(t, "completionEnd", CompletionEndOutput{StopReason: StopReasonEndTurn})
	resp, err = rs.Consume(end)
	if err != nil {
		t.Fatalf("Consume completionEnd: %v", err)
	}
	if resp.CustomMetadata == nil {
		t.Fatal("expected CustomMetadata populated")
	}
	events, ok := resp.CustomMetadata[CustomMetadataKeyUnknownEvents].([]map[string]any)
	if !ok {
		t.Fatalf("CustomMetadata[unknown] = %T", resp.CustomMetadata[CustomMetadataKeyUnknownEvents])
	}
	if len(events) != 1 {
		t.Fatalf("expected 1 unknown event, got %d", len(events))
	}
	if events[0]["event"] != "futureEvent" {
		t.Fatalf("event name lost: %+v", events[0])
	}
}

func TestReadStateUnknownEventsCappedAndCleared(t *testing.T) {
	t.Parallel()
	rs := NewReadState()
	// Feed more than the cap to confirm the slice is bounded.
	overflow := unknownEventsCap + 5
	for i := range overflow {
		_, err := rs.Consume([]byte(`{"event":{"futureEvent":{}}}`))
		if err != nil {
			t.Fatalf("Consume %d: %v", i, err)
		}
	}
	if len(rs.unknownEvents) != unknownEventsCap {
		t.Fatalf("unknownEvents should be capped at %d, got %d", unknownEventsCap, len(rs.unknownEvents))
	}
	// completionEnd flushes and clears the buffer.
	end := mustWrap(t, "completionEnd", CompletionEndOutput{StopReason: StopReasonEndTurn})
	if _, err := rs.Consume(end); err != nil {
		t.Fatalf("completionEnd: %v", err)
	}
	if len(rs.unknownEvents) != 0 {
		t.Fatalf("unknownEvents should be cleared after flush, got %d", len(rs.unknownEvents))
	}
}

func TestEventNameDiscriminator(t *testing.T) {
	raw := json.RawMessage(`{"completionStart":{"sessionId":"s","promptName":"p","completionId":"c"}}`)
	name, payload, err := EventName(raw)
	if err != nil {
		t.Fatalf("EventName: %v", err)
	}
	if name != "completionStart" {
		t.Fatalf("name = %q, want completionStart", name)
	}
	var got CompletionStartOutput
	if err := json.Unmarshal(payload, &got); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	if got.SessionID != "s" || got.PromptName != "p" || got.CompletionID != "c" {
		t.Fatalf("got %+v", got)
	}
}

func TestEventNameRejectsMultipleKeys(t *testing.T) {
	raw := json.RawMessage(`{"a":{},"b":{}}`)
	if _, _, err := EventName(raw); err == nil {
		t.Fatal("expected error on multi-key envelope")
	}
}

func TestWrapShape(t *testing.T) {
	out, err := Wrap("sessionStart", sessionStartInput{})
	if err != nil {
		t.Fatalf("Wrap: %v", err)
	}
	if !strings.HasPrefix(string(out), `{"event":{"sessionStart":`) {
		t.Fatalf("envelope shape unexpected: %s", out)
	}
	// Round-trip parse to ensure valid JSON.
	var env Envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatalf("round-trip unmarshal: %v", err)
	}
	name, _, err := EventName(env.Event)
	if err != nil {
		t.Fatalf("EventName: %v", err)
	}
	if name != "sessionStart" {
		t.Fatalf("name = %q", name)
	}
}

func TestBuildAudioInputBase64(t *testing.T) {
	raw := []byte{0x01, 0x02, 0x03, 0x04}
	out, err := BuildAudioInput("p1", "c1", raw)
	if err != nil {
		t.Fatalf("BuildAudioInput: %v", err)
	}
	// Decode and inspect the inner audioInput.content.
	var env Envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, payload, err := EventName(env.Event)
	if err != nil {
		t.Fatalf("EventName: %v", err)
	}
	var ev audioInputEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		t.Fatalf("decode audioInput: %v", err)
	}
	if ev.PromptName != "p1" || ev.ContentName != "c1" {
		t.Fatalf("ids = %q/%q", ev.PromptName, ev.ContentName)
	}
	decoded, err := base64.StdEncoding.DecodeString(ev.Content)
	if err != nil {
		t.Fatalf("base64 decode: %v", err)
	}
	if string(decoded) != string(raw) {
		t.Fatalf("decoded = %v want %v", decoded, raw)
	}
}

func TestBuildContentStartAudioConfig(t *testing.T) {
	out, err := BuildContentStartAudio("p1", "c1", 8000)
	if err != nil {
		t.Fatalf("BuildContentStartAudio: %v", err)
	}
	var env Envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, payload, _ := EventName(env.Event)
	var ev contentStartInput
	if err := json.Unmarshal(payload, &ev); err != nil {
		t.Fatalf("decode contentStart: %v", err)
	}
	if ev.Type != "AUDIO" || ev.Role != "USER" || !ev.Interactive {
		t.Fatalf("audio framing wrong: %+v", ev)
	}
	if ev.AudioInputConfiguration == nil || ev.AudioInputConfiguration.SampleRateHertz != 8000 {
		t.Fatalf("sample rate = %+v", ev.AudioInputConfiguration)
	}
}

func TestBuildPromptStartDefaultsToTextAndAudio(t *testing.T) {
	out, err := BuildPromptStart("p1", PromptStartOptions{})
	if err != nil {
		t.Fatalf("BuildPromptStart: %v", err)
	}
	var env Envelope
	if err := json.Unmarshal(out, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	_, payload, _ := EventName(env.Event)
	var ps promptStartInput
	if err := json.Unmarshal(payload, &ps); err != nil {
		t.Fatalf("decode promptStart: %v", err)
	}
	if ps.TextOutputConfiguration == nil {
		t.Fatal("expected default text output config")
	}
	if ps.AudioOutputConfiguration == nil {
		t.Fatal("expected default audio output config")
	}
	if ps.AudioOutputConfiguration.VoiceID != defaultVoiceID {
		t.Fatalf("default voice = %q", ps.AudioOutputConfiguration.VoiceID)
	}
}

func TestReadStateAudioOutputEmitsBlob(t *testing.T) {
	rs := NewReadState()
	pcm := []byte{0xAA, 0xBB, 0xCC}
	envBytes := mustWrap(t, "audioOutput", AudioOutputEvent{
		ContentID: "c1",
		Content:   base64.StdEncoding.EncodeToString(pcm),
	})
	// Pre-register the content as ASSISTANT/AUDIO via a synthetic contentStart.
	rs.contentType["c1"] = "AUDIO"
	rs.contentRole["c1"] = "ASSISTANT"

	resp, err := rs.Consume(envBytes)
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) != 1 {
		t.Fatalf("missing content parts: %+v", resp)
	}
	part := resp.Content.Parts[0]
	if part.InlineData == nil {
		t.Fatalf("missing inline data part")
	}
	if !strings.HasPrefix(part.InlineData.MIMEType, "audio/pcm;rate=") {
		t.Fatalf("mime = %q", part.InlineData.MIMEType)
	}
	if string(part.InlineData.Data) != string(pcm) {
		t.Fatalf("audio bytes round-trip mismatch")
	}
	if !resp.Partial {
		t.Fatal("audio chunks should be Partial=true")
	}
}

func TestReadStateTextOutputRoutesByRole(t *testing.T) {
	cases := []struct {
		role         string
		wantInput    bool
		wantOutput   bool
		wantTextPart bool
	}{
		{role: "USER", wantInput: true},
		{role: "ASSISTANT", wantOutput: true},
		{role: "UNKNOWN", wantTextPart: true},
	}
	for _, tc := range cases {
		t.Run(tc.role, func(t *testing.T) {
			rs := NewReadState()
			rs.contentType["c1"] = "TEXT"
			rs.contentRole["c1"] = tc.role
			env := mustWrap(t, "textOutput", TextOutputEvent{ContentID: "c1", Content: "hi"})
			resp, err := rs.Consume(env)
			if err != nil {
				t.Fatalf("Consume: %v", err)
			}
			if (resp.InputTranscription != nil) != tc.wantInput {
				t.Fatalf("InputTranscription = %v", resp.InputTranscription)
			}
			if (resp.OutputTranscription != nil) != tc.wantOutput {
				t.Fatalf("OutputTranscription = %v", resp.OutputTranscription)
			}
			if tc.wantTextPart {
				if resp.Content == nil || len(resp.Content.Parts) == 0 || resp.Content.Parts[0].Text != "hi" {
					t.Fatalf("expected text part, got %+v", resp.Content)
				}
			}
		})
	}
}

func TestReadStateInterruptedMarker(t *testing.T) {
	rs := NewReadState()
	rs.contentType["c1"] = "TEXT"
	rs.contentRole["c1"] = "ASSISTANT"
	env := mustWrap(t, "textOutput", TextOutputEvent{
		ContentID: "c1",
		Content:   interruptedMarker,
	})
	resp, err := rs.Consume(env)
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if !resp.Interrupted {
		t.Fatal("expected Interrupted=true")
	}
}

func TestReadStateToolUseEmitsOnContentEnd(t *testing.T) {
	rs := NewReadState()
	rs.contentType["c1"] = "TOOL"
	rs.contentRole["c1"] = "TOOL"

	// toolUse alone should buffer and not yield.
	env := mustWrap(t, "toolUse", ToolUseOutputEvent{
		ContentID: "c1",
		ToolName:  "weather",
		ToolUseID: "tu-1",
		Content:   `{"location":"Boston"}`,
	})
	if resp, err := rs.Consume(env); err != nil || resp != nil {
		t.Fatalf("toolUse should buffer: resp=%+v err=%v", resp, err)
	}

	// contentEnd(TOOL_USE) flushes the buffered call.
	end := mustWrap(t, "contentEnd", ContentEndOutput{
		ContentID:  "c1",
		Type:       "TOOL",
		StopReason: StopReasonToolUse,
	})
	resp, err := rs.Consume(end)
	if err != nil {
		t.Fatalf("Consume contentEnd: %v", err)
	}
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) != 1 {
		t.Fatalf("expected one part, got %+v", resp)
	}
	fc := resp.Content.Parts[0].FunctionCall
	if fc == nil || fc.Name != "weather" || fc.ID != "tu-1" {
		t.Fatalf("function call wrong: %+v", fc)
	}
	if fc.Args["location"] != "Boston" {
		t.Fatalf("args = %v", fc.Args)
	}
}

func TestReadStateCompletionEndTurnComplete(t *testing.T) {
	rs := NewReadState()
	env := mustWrap(t, "completionEnd", CompletionEndOutput{StopReason: StopReasonEndTurn})
	resp, err := rs.Consume(env)
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if !resp.TurnComplete {
		t.Fatal("expected TurnComplete=true")
	}
	if resp.FinishReason != genai.FinishReasonStop {
		t.Fatalf("finish reason = %v", resp.FinishReason)
	}
}

func TestReadStateUnknownEventDropped(t *testing.T) {
	rs := NewReadState()
	env := []byte(`{"event":{"futureEvent":{"foo":1}}}`)
	resp, err := rs.Consume(env)
	if err != nil {
		t.Fatalf("Consume: %v", err)
	}
	if resp != nil {
		t.Fatalf("expected nil response, got %+v", resp)
	}
}

func TestToolEntryFromFunctionDeclarationLowercasesSchemaTypes(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name: "search",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"q": {Type: genai.TypeString},
			},
		},
	}
	entry, err := toolEntryFromFunctionDeclaration(fd)
	if err != nil {
		t.Fatalf("toolEntryFromFunctionDeclaration: %v", err)
	}
	// genai.TypeString marshals to "STRING"; we must lowercase before sending
	// because Sonic / JSON Schema canonical uses lowercase type names.
	if strings.Contains(entry.ToolSpec.InputSchema.JSON, `"STRING"`) {
		t.Fatalf("schema kept uppercase types: %s", entry.ToolSpec.InputSchema.JSON)
	}
	if !strings.Contains(entry.ToolSpec.InputSchema.JSON, `"type":"string"`) {
		t.Fatalf("schema missing lowercased type: %s", entry.ToolSpec.InputSchema.JSON)
	}
}

func TestToolEntryFromFunctionDeclarationParametersJsonSchemaPrecedence(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name:                 "search",
		ParametersJsonSchema: map[string]any{"type": "object", "marker": "from-json-schema"},
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"unused": {Type: genai.TypeString},
			},
		},
	}
	entry, err := toolEntryFromFunctionDeclaration(fd)
	if err != nil {
		t.Fatalf("toolEntryFromFunctionDeclaration: %v", err)
	}
	if !strings.Contains(entry.ToolSpec.InputSchema.JSON, "from-json-schema") {
		t.Fatalf("ParametersJsonSchema should win: %s", entry.ToolSpec.InputSchema.JSON)
	}
	if strings.Contains(entry.ToolSpec.InputSchema.JSON, "unused") {
		t.Fatalf("Parameters should be ignored when ParametersJsonSchema is set: %s", entry.ToolSpec.InputSchema.JSON)
	}
}

func TestToolEntryFromFunctionDeclarationEmptyParametersFallback(t *testing.T) {
	fd := &genai.FunctionDeclaration{Name: "no_args"}
	entry, err := toolEntryFromFunctionDeclaration(fd)
	if err != nil {
		t.Fatalf("toolEntryFromFunctionDeclaration: %v", err)
	}
	if !strings.Contains(entry.ToolSpec.InputSchema.JSON, `"type":"object"`) {
		t.Fatalf("empty schema should fall back to object: %s", entry.ToolSpec.InputSchema.JSON)
	}
	if !strings.Contains(entry.ToolSpec.InputSchema.JSON, `"properties":{}`) {
		t.Fatalf("empty schema should have empty properties: %s", entry.ToolSpec.InputSchema.JSON)
	}
}

func TestToolEntriesFromADKRejectsNovaGrounding(t *testing.T) {
	tools := map[string]any{
		"grounded": novagrounding.Tool(),
	}
	_, err := toolEntriesFromADK(tools)
	if err == nil {
		t.Fatal("expected error rejecting nova_grounding sentinel")
	}
	if !errors.Is(err, ErrUnsupportedTool) {
		t.Fatalf("error should wrap ErrUnsupportedTool: %v", err)
	}
}

func TestReadStateToolUseConcatenatesChunks(t *testing.T) {
	rs := NewReadState()
	rs.contentType["c1"] = "TOOL"
	rs.contentRole["c1"] = "TOOL"

	// Two toolUse events for the same contentId — args arrive chunked.
	// Identity is set on the first event only; subsequent events carry empty
	// name/id, matching how a streamed JSON body would arrive.
	chunk1 := mustWrap(t, "toolUse", ToolUseOutputEvent{
		ContentID: "c1",
		ToolName:  "weather",
		ToolUseID: "tu-1",
		Content:   `{"loca`,
	})
	if _, err := rs.Consume(chunk1); err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	chunk2 := mustWrap(t, "toolUse", ToolUseOutputEvent{
		ContentID: "c1",
		Content:   `tion":"Boston"}`,
	})
	if _, err := rs.Consume(chunk2); err != nil {
		t.Fatalf("chunk2: %v", err)
	}
	end := mustWrap(t, "contentEnd", ContentEndOutput{
		ContentID:  "c1",
		Type:       "TOOL",
		StopReason: StopReasonToolUse,
	})
	resp, err := rs.Consume(end)
	if err != nil {
		t.Fatalf("contentEnd: %v", err)
	}
	if resp == nil || len(resp.Content.Parts) != 1 {
		t.Fatalf("expected one part, got %+v", resp)
	}
	fc := resp.Content.Parts[0].FunctionCall
	if fc == nil || fc.Name != "weather" || fc.ID != "tu-1" {
		t.Fatalf("function call: %+v", fc)
	}
	if fc.Args["location"] != "Boston" {
		t.Fatalf("concatenated args lost: %v", fc.Args)
	}
}

func TestToolEntriesFromADK(t *testing.T) {
	fd := &genai.FunctionDeclaration{
		Name:        "search",
		Description: "search the web",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"q": {Type: genai.TypeString},
			},
		},
	}
	tools := map[string]any{
		"search":    fd,
		"toolGroup": &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{fd}},
	}
	entries, err := toolEntriesFromADK(tools)
	if err != nil {
		t.Fatalf("toolEntriesFromADK: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	for _, e := range entries {
		if e.ToolSpec.Name != "search" {
			t.Errorf("name = %q", e.ToolSpec.Name)
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(e.ToolSpec.InputSchema.JSON), &parsed); err != nil {
			t.Errorf("inputSchema.json must be valid JSON: %v", err)
		}
	}
}

func mustWrap(t *testing.T, name string, payload any) []byte {
	t.Helper()
	b, err := Wrap(name, payload)
	if err != nil {
		t.Fatalf("Wrap %q: %v", name, err)
	}
	return b
}
