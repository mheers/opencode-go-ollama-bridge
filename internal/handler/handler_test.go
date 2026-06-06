package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mheers/opencode-go-ollama-bridge/internal/adapter"
	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
	"github.com/mheers/opencode-go-ollama-bridge/internal/redact"
)

func newTestHandler() *Handler {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "models") {
			json.NewEncoder(w).Encode(client.ModelsResponse{
				Data: []client.Model{{ID: "glm-5.1"}, {ID: "kimi-k2.6"}},
			})
			return
		}
		if strings.Contains(r.URL.Path, "messages") {
			json.NewEncoder(w).Encode(map[string]interface{}{
				"type":    "message",
				"role":    "assistant",
				"content": []map[string]interface{}{{"type": "text", "text": "hello from anthropic"}},
			})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "hello from openai",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]interface{}{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	c := client.New(srv.URL, "test-key")
	return New(c, "0.24.0", false, redact.NewNoop())
}

func TestVersionEndpoint(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/version", nil)
	w := httptest.NewRecorder()

	h.Version()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp["version"] != "0.24.0" {
		t.Errorf("version = %q", resp["version"])
	}
}

func TestHealthEndpoint(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()

	h.Health()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "Ollama is running") {
		t.Errorf("unexpected body: %s", w.Body.String())
	}
}

func TestTagsEndpoint(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/tags", nil)
	w := httptest.NewRecorder()

	h.Tags()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	models, ok := resp["models"].([]interface{})
	if !ok {
		t.Fatalf("expected models array, got %T", resp["models"])
	}
	if len(models) != 2 {
		t.Errorf("expected 2 models, got %d", len(models))
	}
}

func TestPSEndpoint(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/ps", nil)
	w := httptest.NewRecorder()

	h.PS()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
}

func TestChatEndpoint_OpenAI(t *testing.T) {
	h := newTestHandler()
	body := `{"model":"glm-5.1","messages":[{"role":"user","content":"hello"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Chat()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	if resp["model"] != "glm-5.1" {
		t.Errorf("model = %q", resp["model"])
	}
	done, _ := resp["done"].(bool)
	if !done {
		t.Error("should be done")
	}
}

func TestChatEndpoint_Anthropic(t *testing.T) {
	h := newTestHandler()
	body := `{"model":"minimax-m2.5","messages":[{"role":"user","content":"hello"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Chat()(w, req)

	if w.Code != http.StatusNotImplemented {
		t.Errorf("expected 501 for non-streaming anthropic, got %d: %s", w.Code, w.Body.String())
	}
}

func TestChatEndpoint_InvalidBody(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{invalid`))
	w := httptest.NewRecorder()

	h.Chat()(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestGenerateEndpoint(t *testing.T) {
	h := newTestHandler()
	body := `{"model":"glm-5.1","prompt":"Why is the sky blue?","stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Generate()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}
	done, _ := resp["done"].(bool)
	if !done {
		t.Error("should be done")
	}
}

func TestShowEndpoint(t *testing.T) {
	h := newTestHandler()
	body := `{"model":"glm-5.1:latest"}`
	req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.Show()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	details, ok := resp["details"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected details object, got %T", resp["details"])
	}
	if details["family"] != "glm" {
		t.Errorf("family = %q", details["family"])
	}

	caps, ok := resp["capabilities"].([]interface{})
	if !ok {
		t.Fatalf("expected capabilities array, got %T", resp["capabilities"])
	}
	foundTools := false
	foundCompletion := false
	for _, c := range caps {
		if s, ok := c.(string); ok {
			if s == "tools" {
				foundTools = true
			}
			if s == "completion" {
				foundCompletion = true
			}
		}
	}
	if !foundTools {
		t.Error("capabilities should include 'tools'")
	}
	if !foundCompletion {
		t.Error("capabilities should include 'completion'")
	}

	modelfile, _ := resp["modelfile"].(string)
	if modelfile == "" {
		t.Error("modelfile should not be empty")
	}
}

func TestShowEndpoint_GetNotAllowed(t *testing.T) {
	h := newTestHandler()
	req := httptest.NewRequest(http.MethodGet, "/api/show", nil)
	w := httptest.NewRecorder()

	h.Show()(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestSanitizeTaggedText(t *testing.T) {
	in := `<think>internal notes</think>
Optimized tool selection

]<]minimax[>[<tool_call> - read_file </tool_call>

Visible answer`

	out := sanitizeTaggedText(in)

	if strings.Contains(out, "<think>") {
		t.Fatalf("think block should be removed: %q", out)
	}
	if strings.Contains(out, "<tool_call>") {
		t.Fatalf("tool_call block should be removed: %q", out)
	}
	if strings.Contains(out, "]<]minimax[>[") {
		t.Fatalf("minimax wrapper should be removed: %q", out)
	}
	if !strings.Contains(out, "Visible answer") {
		t.Fatalf("expected remaining visible answer, got: %q", out)
	}
}

func TestParseTaggedAssistantContent_ExtractsToolCalls(t *testing.T) {
	in := `<think>analysis</think>I'll explore now.
<tool_call>{"name":"read_file","parameters":{"filePath":"/tmp/demo/README.md"}} {"name":"read_file","parameters":{"filePath":"/tmp/demo/go.mod"}}</tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)

	if strings.Contains(clean, "<think>") || strings.Contains(clean, "<tool_call>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if !strings.Contains(clean, "I'll explore now.") {
		t.Fatalf("expected visible text to remain, got: %q", clean)
	}
	if len(toolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("unexpected tool name: %+v", fn)
	}
}

func TestParseTaggedAssistantContent_ExtractsMalformedToolCalls(t *testing.T) {
	in := `<think>thinking</think>I'll explore now.
<tool_call> read_file read_file [read_file {"filePath":"/tmp/demo/go.mod"}] [read_file {"filePath":"/tmp/demo/main.go"}] </tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)

	if strings.Contains(clean, "<think>") || strings.Contains(clean, "<tool_call>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if len(toolCalls) < 2 {
		t.Fatalf("expected at least 2 tool calls from malformed block, got %d", len(toolCalls))
	}
}

func TestParseTaggedAssistantContent_FunctionTagStyle(t *testing.T) {
	in := `<tool_call>
<function run_in_terminal>
<parameter=command>
cd /tmp/demo && git log --oneline
</parameter>
</function>
</tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["command"] != "cd /tmp/demo && git log --oneline" {
		t.Fatalf("unexpected command argument: %q", parsed["command"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeStyle(t *testing.T) {
	in := `]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="run_in_terminal">]<]minimax[>[<command>cd /tmp/demo && git log --oneline</command>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["command"] != "cd /tmp/demo && git log --oneline" {
		t.Fatalf("unexpected command argument: %q", parsed["command"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeWithValueTags_ReadFile(t *testing.T) {
	in := `]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="read_file">]<]minimax[>[<filePath>/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt]<]minimax[>[</filePath>]<]minimax[>[<startLine>0]<]minimax[>[</startLine>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("expected read_file, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt" {
		t.Fatalf("unexpected filePath argument: %q", parsed["filePath"])
	}
	if parsed["startLine"] != "0" {
		t.Fatalf("unexpected startLine argument: %q", parsed["startLine"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeWithValueTags_InsertEdit(t *testing.T) {
	in := `]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="insert_edit_into_file">]<]minimax[>[<filePath>/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt]<]minimax[>[</filePath>]<]minimax[>[<newString>test
]<]minimax[>[</newString>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "insert_edit_into_file" {
		t.Fatalf("expected insert_edit_into_file, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt" {
		t.Fatalf("unexpected filePath argument: %q", parsed["filePath"])
	}
	if strings.TrimSpace(parsed["newString"]) != "test" {
		t.Fatalf("unexpected newString argument: %q", parsed["newString"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxFencedToolArgsObject(t *testing.T) {
	in := "I'll read the file first.]<]minimax[>[<tool_call>\n````json\n{ tool: \"read_file\", args: { filePath: \"/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt\" } }\n````"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("expected read_file, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt" {
		t.Fatalf("unexpected filePath argument: %q", parsed["filePath"])
	}
	if strings.Contains(clean, "<tool_call>") {
		t.Fatalf("tool_call block should be removed from clean content: %q", clean)
	}
}

func TestParseTaggedAssistantContent_FilepathSnippetFallbackToCreateFile(t *testing.T) {
	in := "I'll write \"test\" into the file.````\n// filepath: /home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt\ntest\n````"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "create_file" {
		t.Fatalf("expected create_file, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt" {
		t.Fatalf("unexpected filePath argument: %q", parsed["filePath"])
	}
	if parsed["content"] != "test" {
		t.Fatalf("unexpected content argument: %q", parsed["content"])
	}
	if strings.Contains(clean, "// filepath:") {
		t.Fatalf("filepath snippet should be removed from clean content: %q", clean)
	}
}

func TestParseTaggedAssistantContent_TaggedToolEnvelopeParameters(t *testing.T) {
	in := "I'll write \"test\" into the file.]<]minimax[>[<tool_call>\n````\n<tool_name>insert_edit_into_file</tool_name>\n<parameters>\n<relativeWorkspacePath>test.txt</relativeWorkspacePath>\n<file_text>test\n</file_text>\n</parameters>\n````"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "insert_edit_into_file" {
		t.Fatalf("expected insert_edit_into_file, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["relativeWorkspacePath"] != "test.txt" {
		t.Fatalf("unexpected relativeWorkspacePath argument: %q", parsed["relativeWorkspacePath"])
	}
	if strings.TrimSpace(parsed["file_text"]) != "test" {
		t.Fatalf("unexpected file_text argument: %q", parsed["file_text"])
	}
	if strings.Contains(clean, "<tool_name>") {
		t.Fatalf("tagged tool envelope should be removed from clean content: %q", clean)
	}
}

func TestParseTaggedAssistantContent_DeepSeekDSMLStyle(t *testing.T) {
	in := "<｜｜DSML｜｜tool_calls>\n" +
		"<｜｜DSML｜｜invoke name=\"run_in_terminal\">\n" +
		"<｜｜DSML｜｜parameter name=\"command\" string=\"true\">cd /tmp/demo && wc -l * 2>/dev/null</｜｜DSML｜｜parameter>\n" +
		"<｜｜DSML｜｜parameter name=\"explanation\" string=\"true\">Re-check line counts for all files</｜｜DSML｜｜parameter>\n" +
		"<｜｜DSML｜｜parameter name=\"goal\" string=\"true\">Verify line counts</｜｜DSML｜｜parameter>\n" +
		"<｜｜DSML｜｜parameter name=\"mode\" string=\"true\">sync</｜｜DSML｜｜parameter>\n" +
		"</｜｜DSML｜｜invoke>\n" +
		"</｜｜DSML｜｜tool_calls>"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["command"] != "cd /tmp/demo && wc -l * 2>/dev/null" {
		t.Fatalf("unexpected command argument: %q", parsed["command"])
	}
	if parsed["mode"] != "sync" {
		t.Fatalf("unexpected mode argument: %q", parsed["mode"])
	}
}

func TestParseTaggedAssistantContent_PluralToolCallsEmpty(t *testing.T) {
	// hy3-preview emits an empty <tool_calls> block when it attempts a call.
	in := "Now, let me lint the Go files:<tool_calls>\n\n</tool_calls>"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_calls>") {
		t.Fatalf("clean content still has <tool_calls> markup: %q", clean)
	}
	if !strings.Contains(clean, "Now, let me lint the Go files") {
		t.Fatalf("visible text was eaten: %q", clean)
	}
	if len(toolCalls) != 0 {
		t.Fatalf("expected 0 tool calls from empty block, got %d", len(toolCalls))
	}
}

func TestParseTaggedAssistantContent_PluralToolCallsJSON(t *testing.T) {
	in := `Let me list the files.<tool_calls>[{"name":"list_files","arguments":{"path":"/tmp"}}]</tool_calls>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_calls>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "list_files" {
		t.Fatalf("unexpected tool name: %+v", fn)
	}
}

func TestParseTaggedAssistantContent_PluralToolCallsInvokeStyle(t *testing.T) {
	in := `<tool_calls><invoke name="run_in_terminal"><command>go vet ./...</command></invoke></tool_calls>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["command"] != "go vet ./..." {
		t.Fatalf("unexpected command: %q", parsed["command"])
	}
}

func TestParseTaggedAssistantContent_PluralToolCallsSingleObject(t *testing.T) {
	// hy3-preview may emit a single JSON object (not an array) inside <tool_calls>.
	in := `Let me read the file.<tool_calls>{"name":"read_file","arguments":{"path":"main.go"}}</tool_calls>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_calls>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if !strings.Contains(clean, "Let me read the file") {
		t.Fatalf("visible text was eaten: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("unexpected tool name: %+v", fn)
	}
}

func TestParseTaggedAssistantContent_PluralToolCallsOpenAINested(t *testing.T) {
	// OpenAI-nested format inside <tool_calls>: {"type":"function","function":{...}}
	in := `<tool_calls>{"type":"function","function":{"name":"run_in_terminal","arguments":"{\"command\":\"go vet ./...\"}"}}</tool_calls>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %+v", fn)
	}
}

func TestParseTaggedAssistantContent_PluralNestedToolCallsNameOnly(t *testing.T) {
	// hy3-preview exact format from real log:
	// <tool_calls>\n<tool_call>read_file\n<tool_call>run_in_terminal\n</tool_call>\n</tool_calls>
	in := "I'll read the Go file and then run the linter." +
		"<tool_calls>\n<tool_call>read_file\n<tool_call>run_in_terminal\n</tool_call>\n</tool_calls>"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_calls>") || strings.Contains(clean, "<tool_call>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if !strings.Contains(clean, "I'll read the Go file") {
		t.Fatalf("visible text was eaten: %q", clean)
	}
	if len(toolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d: %+v", len(toolCalls), toolCalls)
	}
	name0, _ := toolCalls[0]["function"].(map[string]interface{})
	name1, _ := toolCalls[1]["function"].(map[string]interface{})
	if name0["name"] != "read_file" {
		t.Fatalf("expected read_file, got %q", name0["name"])
	}
	if name1["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %q", name1["name"])
	}
}

func TestParseTaggedAssistantContent_PluralNestedToolCallsWithArgs(t *testing.T) {
	// <tool_call>name\n{json_args}</tool_call> format inside <tool_calls>.
	in := "<tool_calls>\n<tool_call>read_file\n{\"filePath\":\"main.go\",\"startLine\":1,\"endLine\":50}</tool_call>\n</tool_calls>"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if clean != "" {
		t.Fatalf("expected empty visible content, got: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("unexpected tool name: %+v", fn)
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "main.go" {
		t.Fatalf("unexpected filePath: %v", parsed["filePath"])
	}
}

func TestParseTaggedAssistantContent_PluralNestedToolCallJunkTags(t *testing.T) {
	// Real observed format: model appends </arg_value> on the same line as the name.
	// <tool_calls>\n<tool_call>read_file</arg_value>\n</tool_call>\n</tool_calls>
	in := "I'll read the Go file and then run the linter." +
		"<tool_calls>\n<tool_call>read_file</arg_value>\n</tool_call>\n</tool_calls>"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_calls>") || strings.Contains(clean, "<tool_call>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("expected read_file (without junk tags), got %q", fn["name"])
	}
}

func TestParseTaggedAssistantContent_PluralNestedArgKeyValueFormat(t *testing.T) {
	// hy3-preview exact format: name and args on same line using <arg_key>/<arg_value> pairs.
	in := "I'll count the lines using the terminal." +
		"<tool_calls>\n<tool_call>run_in_terminal " +
		"<arg_key>command</arg_key> <arg_value>wc -l /tmp/demo/* 2>/dev/null</arg_value> " +
		"<arg_key>explanation</arg_key> <arg_value>Count lines in all files</arg_value>" +
		"</tool_call>\n</tool_calls>"

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_calls>") || strings.Contains(clean, "<arg_key>") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if !strings.Contains(clean, "I'll count the lines") {
		t.Fatalf("visible text was eaten: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "run_in_terminal" {
		t.Fatalf("expected run_in_terminal, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["command"] != "wc -l /tmp/demo/* 2>/dev/null" {
		t.Fatalf("unexpected command: %q", parsed["command"])
	}
	if parsed["explanation"] != "Count lines in all files" {
		t.Fatalf("unexpected explanation: %q", parsed["explanation"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxM3JSObjectFormat(t *testing.T) {
	// MiniMax-M3 format: ]<]minimax[>[<tool_call> with JS-style unquoted keys.
	// { tool: "name", args: { ... } }
	in := `I'll explore the codebase.]<]minimax[>[<tool_call>
{ tool: "manage_todo_list",args: { "todoList": [{"id":1,"title":"Explore","status":"in-progress"}] } }
`
	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_call>") || strings.Contains(clean, "minimax") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if !strings.Contains(clean, "I'll explore") {
		t.Fatalf("visible text was eaten: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "manage_todo_list" {
		t.Fatalf("expected manage_todo_list, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	todos, _ := parsed["todoList"].([]interface{})
	if len(todos) != 1 {
		t.Fatalf("expected 1 todo, got %d", len(todos))
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeWithParameters(t *testing.T) {
	// MiniMax-M3 format: invoke with <parameter name="key"> attributes.
	// Each line is prefixed with the minimax wrapper.
	in := `]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="runSubagent">]<]minimax[>[<parameter name="agentName">Explore</parameter>
<parameter name="prompt">Explore the repo thoroughly.</parameter>
</invoke>
]<]minimax[>[</tool_call>`

	clean, toolCalls := parseTaggedAssistantContent(in)
	if strings.Contains(clean, "<tool_call>") || strings.Contains(clean, "minimax") {
		t.Fatalf("clean content still has markup: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "runSubagent" {
		t.Fatalf("expected runSubagent, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["agentName"] != "Explore" {
		t.Fatalf("unexpected agentName: %q", parsed["agentName"])
	}
	if parsed["prompt"] != "Explore the repo thoroughly." {
		t.Fatalf("unexpected prompt: %q", parsed["prompt"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeReadFile(t *testing.T) {
	// read_file via MiniMax invoke format — the exact failure reported as
	// "must have required property 'filePath'".
	in := `]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="read_file">]<]minimax[>[<parameter name="filePath">/home/foo/handler.go</parameter>
<parameter name="startLine">1</parameter>
<parameter name="endLine">50</parameter>
]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>`

	_, toolCalls := parseTaggedAssistantContent(in)
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("expected read_file, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "/home/foo/handler.go" {
		t.Fatalf("filePath wrong: %q", parsed["filePath"])
	}
	if parsed["startLine"] != "1" {
		t.Fatalf("startLine wrong: %q", parsed["startLine"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeMalformedAttr(t *testing.T) {
	// MiniMax emits <invoke name>funcname"> (drops the = and opening "),
	// which closes the tag at 'name>' making the function name appear as text.
	// Log excerpt: ]<]minimax[>[<invoke name>read_file">]<]minimax[>[</invoke>
	in := `]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name>read_file">]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>`

	_, toolCalls := parseTaggedAssistantContent(in)
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call (name only, empty args), got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("expected read_file, got %q", fn["name"])
	}
}

func TestParseTaggedAssistantContent_MiniMaxInvokeCollapsedIntoNameField(t *testing.T) {
	// MiniMax-M3 occasionally collapses a single-argument tool call into a
	// malformed invoke tag where the argument key/value is emitted in the name
	// position and the body only contains a stray closing tag.
	in := `]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name "filePath": "/home/marcel/workspace/heers/opencode-go-ollama-bridge/internal/redact/gitleaks.go"]<]minimax[>[</command>]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>`

	_, toolCalls := parseTaggedAssistantContent(in)
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Fatalf("expected read_file, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/internal/redact/gitleaks.go" {
		t.Fatalf("filePath wrong: %q", parsed["filePath"])
	}
}

func TestParseTaggedAssistantContent_LooseToolTextKeepsLeadingContent(t *testing.T) {
	hints := map[string]string{"semantic_search": "query"}
	in := "Let me find all the `io.ReadAll(r.Body)` sites:\n\nsemantic_search\">\nSearch for all \"io.ReadAll(r.Body)\" invocations in handler.go\n\n</tool_call>"

	clean, toolCalls := parseTaggedAssistantContentWithHints(in, hints)
	if clean != "Let me find all the `io.ReadAll(r.Body)` sites:" {
		t.Fatalf("clean content wrong: %q", clean)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(toolCalls), toolCalls)
	}
	fn, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn["name"] != "semantic_search" {
		t.Fatalf("expected semantic_search, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments json: %v (%s)", err, args)
	}
	if parsed["query"] != "Search for all \"io.ReadAll(r.Body)\" invocations in handler.go" {
		t.Fatalf("query wrong: %q", parsed["query"])
	}
}

func TestParseTaggedAssistantContent_TranscriptToolBlocks(t *testing.T) {
	in := "The AGENTS.md file was not written. Let me create it now.\n\n```text\n[create_file] creating /home/marcel/workspace/heers/opencode-go-ollama-bridge/AGENTS.md\n\n- one\n- two\n```\n\nLet me verify it now exists.\n\n```text\n[read_file] reading /home/marcel/workspace/heers/opencode-go-ollama-bridge/AGENTS.md\n```\n\nConfirmed."

	clean, toolCalls := parseTaggedAssistantContentWithHints(in, nil)
	if len(toolCalls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d: %+v", len(toolCalls), toolCalls)
	}

	fn0, _ := toolCalls[0]["function"].(map[string]interface{})
	if fn0["name"] != "create_file" {
		t.Fatalf("expected first tool create_file, got %q", fn0["name"])
	}
	args0, _ := fn0["arguments"].(string)
	var parsed0 map[string]interface{}
	if err := json.Unmarshal([]byte(args0), &parsed0); err != nil {
		t.Fatalf("invalid create_file args json: %v (%s)", err, args0)
	}
	if parsed0["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/AGENTS.md" {
		t.Fatalf("create_file filePath wrong: %v", parsed0["filePath"])
	}
	if content, _ := parsed0["content"].(string); !strings.Contains(content, "- one") {
		t.Fatalf("create_file content wrong: %q", content)
	}

	fn1, _ := toolCalls[1]["function"].(map[string]interface{})
	if fn1["name"] != "read_file" {
		t.Fatalf("expected second tool read_file, got %q", fn1["name"])
	}
	args1, _ := fn1["arguments"].(string)
	var parsed1 map[string]interface{}
	if err := json.Unmarshal([]byte(args1), &parsed1); err != nil {
		t.Fatalf("invalid read_file args json: %v (%s)", err, args1)
	}
	if parsed1["filePath"] != "/home/marcel/workspace/heers/opencode-go-ollama-bridge/AGENTS.md" {
		t.Fatalf("read_file filePath wrong: %v", parsed1["filePath"])
	}
	if start, ok := parsed1["startLine"].(float64); !ok || int(start) != 1 {
		t.Fatalf("read_file startLine wrong: %v", parsed1["startLine"])
	}
	if end, ok := parsed1["endLine"].(float64); !ok || int(end) != 200 {
		t.Fatalf("read_file endLine wrong: %v", parsed1["endLine"])
	}

	if strings.Contains(clean, "[create_file]") || strings.Contains(clean, "[read_file]") {
		t.Fatalf("clean content should not retain transcript tool blocks: %q", clean)
	}
	if !strings.Contains(clean, "Confirmed.") {
		t.Fatalf("clean content missing surrounding assistant text: %q", clean)
	}
}

func TestShouldRetryMiniMaxForToolCall_IntentNoToolCalls(t *testing.T) {
	body := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"Let me actually invoke the tool now: I will use read_file."},"finish_reason":"stop"}]}`)
	proxyReq := map[string]interface{}{"tool_choice": "required"}
	if !shouldRetryMiniMaxForToolCall(proxyReq, 1, body) {
		t.Fatal("expected retry when response promises tool invocation without structured tool_calls")
	}
}

func TestShouldRetryMiniMaxForToolCall_HasToolCalls(t *testing.T) {
	body := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"ok","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`)
	proxyReq := map[string]interface{}{"tool_choice": "required"}
	if shouldRetryMiniMaxForToolCall(proxyReq, 1, body) {
		t.Fatal("did not expect retry when structured tool_calls are already present")
	}
}

func TestShouldRetryMiniMaxForToolCall_DirectFileSnippetNoToolCalls(t *testing.T) {
	body := []byte("{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"````markdown\\n// filepath: test.txt\\ntest\\n````\"},\"finish_reason\":\"stop\"}]}")
	proxyReq := map[string]interface{}{"tool_choice": "required"}
	if !shouldRetryMiniMaxForToolCall(proxyReq, 1, body) {
		t.Fatal("expected retry when response returns direct file snippet while tool call is required")
	}
}

func TestShouldRetryMiniMaxForToolCall_ExplicitAutoDoesNotRetry(t *testing.T) {
	body := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"I will use read_file now"},"finish_reason":"stop"}]}`)
	proxyReq := map[string]interface{}{"tool_choice": "auto"}
	if shouldRetryMiniMaxForToolCall(proxyReq, 1, body) {
		t.Fatal("did not expect retry when tool_choice is explicitly auto")
	}
}

func TestShouldRetryMiniMaxForToolCall_PlainFencedTextNoToolCalls(t *testing.T) {
	body := []byte("{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"````text\\ntest\\n````\"},\"finish_reason\":\"stop\"}]}")
	proxyReq := map[string]interface{}{"tool_choice": "required"}
	if !shouldRetryMiniMaxForToolCall(proxyReq, 1, body) {
		t.Fatal("expected retry when required tool call is missing and assistant returns plain fenced text")
	}
}

func TestShouldRetryMiniMaxForToolCall_PlainProseNoToolCalls(t *testing.T) {
	body := []byte(`{"choices":[{"index":0,"message":{"role":"assistant","content":"test"},"finish_reason":"stop"}]}`)
	proxyReq := map[string]interface{}{"tool_choice": "required"}
	if !shouldRetryMiniMaxForToolCall(proxyReq, 1, body) {
		t.Fatal("expected retry when required tool call is missing and assistant returns plain prose")
	}
}

func newOpenAIV1Handler(t *testing.T, response string, stream bool) *Handler {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if ct := r.Header.Get("Authorization"); ct != "Bearer test-key" {
			t.Fatalf("unexpected auth header: %s", ct)
		}

		body, _ := io.ReadAll(r.Body)
		if stream && !strings.Contains(string(body), `"stream":true`) {
			t.Fatalf("expected stream=true to be forwarded for non-minimax, body=%s", string(body))
		}

		if stream {
			w.Header().Set("Content-Type", "text/event-stream")
		} else {
			w.Header().Set("Content-Type", "application/json")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	return New(c, "0.24.0", false, redact.NewNoop())
}

func newMiniMaxV1Handler(t *testing.T, response string) *Handler {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if ct := r.Header.Get("Authorization"); ct != "Bearer test-key" {
			t.Fatalf("unexpected auth header: %s", ct)
		}

		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"stream":false`) {
			t.Fatalf("expected bridge to force stream=false for minimax repair path, body=%s", string(body))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	return New(c, "0.24.0", false, redact.NewNoop())
}

func newMiniMaxV1HandlerWithAssert(t *testing.T, response string, assertBody func(string)) *Handler {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if ct := r.Header.Get("Authorization"); ct != "Bearer test-key" {
			t.Fatalf("unexpected auth header: %s", ct)
		}

		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"stream":false`) {
			t.Fatalf("expected bridge to force stream=false for minimax repair path, body=%s", string(body))
		}
		if assertBody != nil {
			assertBody(string(body))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	return New(c, "0.24.0", false, redact.NewNoop())
}

func TestV1ChatCompletions_MiniMax_ForceToolChoiceRequiredWhenToolsPresent(t *testing.T) {
	upstream := `{"id":"mmx-tc-required","object":"chat.completion","created":1,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`

	h := newMiniMaxV1HandlerWithAssert(t, upstream, func(body string) {
		if !strings.Contains(body, `"tools":[`) {
			t.Fatalf("test setup expected tools in request body: %s", body)
		}
		if !strings.Contains(body, `"tool_choice":"required"`) {
			t.Fatalf("expected tool_choice=required when tools present, body=%s", body)
		}
	})

	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"write file"}],"tools":[{"type":"function","function":{"name":"create_file","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"content":{"type":"string"}},"required":["filePath","content"]}}}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestV1ChatCompletions_MiniMax_KeepExplicitToolChoice(t *testing.T) {
	upstream := `{"id":"mmx-tc-explicit","object":"chat.completion","created":1,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`

	h := newMiniMaxV1HandlerWithAssert(t, upstream, func(body string) {
		var payload map[string]interface{}
		if err := json.Unmarshal([]byte(body), &payload); err != nil {
			t.Fatalf("failed to decode proxied body: %v (%s)", err, body)
		}
		tc, ok := payload["tool_choice"].(map[string]interface{})
		if !ok {
			t.Fatalf("expected explicit tool_choice object to be preserved, body=%s", body)
		}
		fn, ok := tc["function"].(map[string]interface{})
		if !ok || fn["name"] != "create_file" {
			t.Fatalf("expected explicit tool_choice.function.name=create_file, body=%s", body)
		}
	})

	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"write file"}],"tools":[{"type":"function","function":{"name":"create_file","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"content":{"type":"string"}},"required":["filePath","content"]}}}],"tool_choice":{"type":"function","function":{"name":"create_file"}},"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestV1ChatCompletions_MiniMax_RetryIntentOnlyResponseOnce(t *testing.T) {
	var callCount int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		callCount++

		if callCount == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"id":"mmx-intent-1","object":"chat.completion","created":1,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"I will now use read_file. Let me actually invoke the tool now:"},"finish_reason":"stop"}]}`))
			return
		}

		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "Tool execution is required") {
			t.Fatalf("expected retry nudge to be injected into second request, body=%s", string(body))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"id":"mmx-intent-2","object":"chat.completion","created":2,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"<tool_call>{\"name\":\"read_file\",\"parameters\":{\"filePath\":\"/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt\",\"startLine\":1,\"endLine\":50}}</tool_call>"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	h := New(c, "0.24.0", false, redact.NewNoop())

	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"write test into test.txt"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"startLine":{"type":"integer"},"endLine":{"type":"integer"}},"required":["filePath","startLine","endLine"]}}}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if callCount != 2 {
		t.Fatalf("expected exactly one retry call (2 total upstream calls), got %d", callCount)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid response JSON: %v (%s)", err, w.Body.String())
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice, got %d: %s", len(choices), w.Body.String())
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})
	toolCalls, _ := msg["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 recovered tool_call after retry, got %d: %s", len(toolCalls), w.Body.String())
	}
}

func TestV1ChatCompletions_MiniMax_RecoversDirectFileSnippetWithoutRetry(t *testing.T) {
	var callCount int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		callCount++

		if callCount == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("{\"id\":\"mmx-file-1\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"MiniMax-M3\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"<think>plan</think>````markdown\\n// filepath: test.txt\\ntest\\n````\"},\"finish_reason\":\"stop\"}]}"))
			return
		}

		t.Fatalf("expected no retry; received unexpected second upstream call")
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	h := New(c, "0.24.0", false, redact.NewNoop())

	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"write test into test.txt"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"startLine":{"type":"integer"},"endLine":{"type":"integer"}},"required":["filePath","startLine","endLine"]}}}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if callCount != 1 {
		t.Fatalf("expected no retry for directly recoverable filepath snippet (1 upstream call), got %d", callCount)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid response JSON: %v (%s)", err, w.Body.String())
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice, got %d: %s", len(choices), w.Body.String())
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})
	toolCalls, _ := msg["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 recovered tool_call from filepath snippet, got %d: %s", len(toolCalls), w.Body.String())
	}
	fn, _ := toolCalls[0].(map[string]interface{})
	function, _ := fn["function"].(map[string]interface{})
	if function["name"] != "create_file" {
		t.Fatalf("expected create_file from filepath snippet recovery, got %v", function["name"])
	}
}

func TestV1ChatCompletions_MiniMax_RetryPlainFencedTextResponseOnce(t *testing.T) {
	var callCount int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		callCount++

		if callCount == 1 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("{\"id\":\"mmx-plain-1\",\"object\":\"chat.completion\",\"created\":1,\"model\":\"MiniMax-M3\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"<think>plan</think>````text\\ntest\\n````\"},\"finish_reason\":\"stop\"}]}") )
			return
		}

		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "Tool execution is required") {
			t.Fatalf("expected retry nudge to be injected into second request, body=%s", string(body))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"id":"mmx-plain-2","object":"chat.completion","created":2,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"<tool_call>{\"name\":\"read_file\",\"parameters\":{\"filePath\":\"/home/marcel/workspace/heers/opencode-go-ollama-bridge/test.txt\",\"startLine\":1,\"endLine\":50}}</tool_call>"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	h := New(c, "0.24.0", false, redact.NewNoop())

	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"write test into test.txt"}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"filePath":{"type":"string"},"startLine":{"type":"integer"},"endLine":{"type":"integer"}},"required":["filePath","startLine","endLine"]}}}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if callCount != 2 {
		t.Fatalf("expected exactly one retry call (2 total upstream calls), got %d", callCount)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid response JSON: %v (%s)", err, w.Body.String())
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice, got %d: %s", len(choices), w.Body.String())
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})
	toolCalls, _ := msg["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 recovered tool_call after retry, got %d: %s", len(toolCalls), w.Body.String())
	}
}

func TestV1ChatCompletions_MiniMax_SynthesizeWriteToolCallFromProseOnly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"id":"mmx-synth-1","object":"chat.completion","created":1,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"I'll write \"test\" directly into the file using the file editing tool."},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	c := client.New(srv.URL, "test-key")
	h := New(c, "0.24.0", false, redact.NewNoop())

	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"write \"test\" into test.txt without reading first"}],"tools":[{"type":"function","function":{"name":"insert_edit_into_file","parameters":{"type":"object","properties":{"relativeWorkspacePath":{"type":"string"},"file_text":{"type":"string"}},"required":["relativeWorkspacePath","file_text"]}}}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid response JSON: %v (%s)", err, w.Body.String())
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice, got %d: %s", len(choices), w.Body.String())
	}
	choice, _ := choices[0].(map[string]interface{})
	if fr, _ := choice["finish_reason"].(string); fr != "tool_calls" {
		t.Fatalf("expected finish_reason=tool_calls, got %q", fr)
	}
	msg, _ := choice["message"].(map[string]interface{})
	toolCalls, _ := msg["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 synthesized tool_call, got %d: %s", len(toolCalls), w.Body.String())
	}
	tc, _ := toolCalls[0].(map[string]interface{})
	fn, _ := tc["function"].(map[string]interface{})
	if fn["name"] != "insert_edit_into_file" {
		t.Fatalf("expected insert_edit_into_file, got %v", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid args JSON: %v (%s)", err, args)
	}
	if parsed["relativeWorkspacePath"] != "test.txt" {
		t.Fatalf("expected relativeWorkspacePath=test.txt, got %q", parsed["relativeWorkspacePath"])
	}
	if parsed["file_text"] != "test" {
		t.Fatalf("expected file_text=test, got %q", parsed["file_text"])
	}
}

func TestV1ChatCompletions_MiniMax_RepairsNonStreamingContent(t *testing.T) {
	upstream := `{
		"id":"chatcmpl-1",
		"object":"chat.completion",
		"created":1,
		"model":"minimax-m2.5",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"<think>plan</think>Final answer ]<]minimax[>[<tool_call>{\"name\":\"read_file\",\"parameters\":{\"filePath\":\"/tmp/demo/README.md\"}}</tool_call>"
			},
			"finish_reason":"stop"
		}]
	}`

	h := newMiniMaxV1Handler(t, upstream)
	body := `{"model":"minimax-m2.5","messages":[{"role":"user","content":"validate repo"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	choices, _ := resp["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(choices))
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})
	content, _ := msg["content"].(string)
	toolCalls, _ := msg["tool_calls"].([]interface{})

	if strings.Contains(content, "<think>") || strings.Contains(content, "<tool_call>") || strings.Contains(content, "]<]minimax[>[") {
		t.Fatalf("content still contains unsupported tags: %q", content)
	}
	if content != "Final answer" {
		t.Fatalf("unexpected repaired content: %q", content)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 structured tool_call, got %d", len(toolCalls))
	}
	if fr, _ := choice["finish_reason"].(string); fr != "tool_calls" {
		t.Fatalf("expected finish_reason=tool_calls, got %q", fr)
	}
}

func TestV1ChatCompletions_MiniMax_RepairsStreamingContent(t *testing.T) {
	upstream := `{
		"id":"chatcmpl-2",
		"object":"chat.completion",
		"created":2,
		"model":"minimax-m2.5",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"<think>plan</think>Answer text ]<]minimax[>[<tool_call>{\"name\":\"run_in_terminal\",\"parameters\":{\"command\":\"go test ./...\"}}</tool_call>"
			},
			"finish_reason":"stop"
		}]
	}`

	h := newMiniMaxV1Handler(t, upstream)
	body := `{"model":"minimax-m2.5","messages":[{"role":"user","content":"validate repo"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if !strings.Contains(bodyOut, "data: ") || !strings.Contains(bodyOut, "data: [DONE]") {
		t.Fatalf("expected SSE output with DONE marker, got: %s", bodyOut)
	}
	if strings.Contains(bodyOut, "<think>") || strings.Contains(bodyOut, "<tool_call>") || strings.Contains(bodyOut, "]<]minimax[>[") {
		t.Fatalf("stream output still contains unsupported tags: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, "Answer text") {
		t.Fatalf("expected repaired answer in stream output: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"tool_calls"`) {
		t.Fatalf("expected structured tool_calls in SSE output: %s", bodyOut)
	}
}

func TestV1ChatCompletions_OpenAI_RepairsNonStreamingContent(t *testing.T) {
	upstream := `{
		"id":"chatcmpl-3",
		"object":"chat.completion",
		"created":3,
		"model":"glm-5.1",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"<think>hidden</think>Clean output ]<]minimax[>[<tool_call>{\"name\":\"read_file\",\"parameters\":{\"filePath\":\"/tmp/demo/go.mod\"}}</tool_call>"
			},
			"finish_reason":"stop"
		}]
	}`

	h := newOpenAIV1Handler(t, upstream, false)
	body := `{"model":"glm-5.1","messages":[{"role":"user","content":"hello"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "<think>") || strings.Contains(w.Body.String(), "<tool_call>") {
		t.Fatalf("non-stream response still contains unsupported tags: %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "Clean output") {
		t.Fatalf("expected sanitized response content: %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"tool_calls"`) {
		t.Fatalf("expected structured tool_calls in response: %s", w.Body.String())
	}
}

func TestV1ChatCompletions_OpenAI_RepairsStreamingContent(t *testing.T) {
	upstream := "data: {\"id\":\"chatcmpl-4\",\"object\":\"chat.completion.chunk\",\"created\":4,\"model\":\"glm-5.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"<think>plan</think>Hello ]<]minimax[>[<tool_call>{\\\"name\\\":\\\"run\\\",\\\"parameters\\\":{\\\"x\\\":1}}</tool_call>\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n"

	h := newOpenAIV1Handler(t, upstream, true)
	body := `{"model":"glm-5.1","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if strings.Contains(bodyOut, "<think>") || strings.Contains(bodyOut, "<tool_call>") || strings.Contains(bodyOut, "]<]minimax[>[") {
		t.Fatalf("stream response still contains unsupported tags: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, "Hello") || !strings.Contains(bodyOut, "data: [DONE]") {
		t.Fatalf("expected sanitized SSE output with DONE marker: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"tool_calls"`) {
		t.Fatalf("expected structured tool_calls in stream output: %s", bodyOut)
	}
}

func TestV1ChatCompletions_OpenAI_StreamWithSplitTagsStillSanitizes(t *testing.T) {
	upstream := "data: {\"id\":\"chatcmpl-5\",\"object\":\"chat.completion.chunk\",\"created\":5,\"model\":\"glm-5.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"<think>hidden\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"chatcmpl-5\",\"object\":\"chat.completion.chunk\",\"created\":5,\"model\":\"glm-5.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" text</think>Answer <tool_call> [read_file {\\\"filePath\\\":\\\"/tmp/demo/go.mod\\\"}] </tool_call>\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"

	h := newOpenAIV1Handler(t, upstream, true)
	body := `{"model":"glm-5.1","messages":[{"role":"user","content":"hello"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if strings.Contains(bodyOut, "<think>") || strings.Contains(bodyOut, "<tool_call>") {
		t.Fatalf("split-tag stream still contains markup: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, "Answer") || !strings.Contains(bodyOut, `"tool_calls"`) {
		t.Fatalf("expected sanitized answer and tool_calls in rewritten stream: %s", bodyOut)
	}
}

func TestV1ChatCompletions_OpenAI_StreamPreservesNativeToolCalls(t *testing.T) {
	upstream := "data: {\"id\":\"chatcmpl-native\",\"object\":\"chat.completion.chunk\",\"created\":10,\"model\":\"qwen3.7-plus\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"run_in_terminal\",\"arguments\":\"\"}}],\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"chatcmpl-native\",\"object\":\"chat.completion.chunk\",\"created\":10,\"model\":\"qwen3.7-plus\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"\",\"type\":\"function\",\"function\":{\"arguments\":\"{\\\"command\\\":\\\"git log --oneline\\\"}\"}}],\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"chatcmpl-native\",\"object\":\"chat.completion.chunk\",\"created\":10,\"model\":\"qwen3.7-plus\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	h := newOpenAIV1Handler(t, upstream, true)
	body := `{"model":"qwen3.7-plus","messages":[{"role":"user","content":"check latest diff"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if !strings.Contains(bodyOut, `"name":"run_in_terminal"`) {
		t.Fatalf("expected native run_in_terminal tool call to be preserved: %s", bodyOut)
	}
	if strings.Contains(bodyOut, `"name":"cd"`) || strings.Contains(bodyOut, `"name":"git"`) {
		t.Fatalf("unexpected fake tool names should not be emitted: %s", bodyOut)
	}
}

func TestV1ChatCompletions_OpenAI_StreamPreservesSparseNativeToolCallIndex(t *testing.T) {
	upstream := "data: {\"id\":\"chatcmpl-sparse\",\"object\":\"chat.completion.chunk\",\"created\":11,\"model\":\"qwen3.7-plus\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":3,\"id\":\"call_sparse\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}],\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
		"data: {\"id\":\"chatcmpl-sparse\",\"object\":\"chat.completion.chunk\",\"created\":11,\"model\":\"qwen3.7-plus\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":3,\"function\":{\"arguments\":\"{\\\"filePath\\\":\\\"/tmp/demo/go.mod\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	h := newOpenAIV1Handler(t, upstream, true)
	body := `{"model":"qwen3.7-plus","messages":[{"role":"user","content":"read go.mod"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if !strings.Contains(bodyOut, `"name":"read_file"`) {
		t.Fatalf("expected sparse-index tool call name to be preserved: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `/tmp/demo/go.mod`) {
		t.Fatalf("expected sparse-index tool call arguments to be preserved: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"finish_reason":"tool_calls"`) {
		t.Fatalf("expected finish_reason tool_calls in rewritten stream: %s", bodyOut)
	}
	if strings.Contains(bodyOut, `"arguments":"{}"`) {
		t.Fatalf("sparse-index tool call should not be dropped to empty args: %s", bodyOut)
	}
}

func TestWriteOpenAIStreamFromResponse_EmitsPerChoiceFinishChunks(t *testing.T) {
	frStop := "stop"
	frTools := "tool_calls"
	repaired := adapter.OpenAIResponse{
		ID:      "chatcmpl-multi",
		Object:  "chat.completion",
		Created: 12,
		Model:   "qwen3.7-plus",
		Choices: []adapter.OpenAIChoice{
			{
				Index: 0,
				Message: &adapter.OpenAIMessageResp{
					Role:    "assistant",
					Content: "first",
				},
				FinishReason: &frStop,
			},
			{
				Index: 1,
				Message: &adapter.OpenAIMessageResp{
					Role:    "assistant",
					Content: "second",
					ToolCalls: []adapter.OpenAIToolCall{{
						ID:   "call_1",
						Type: "function",
						Function: adapter.OpenAIToolFunction{
							Name:      "read_file",
							Arguments: `{"filePath":"main.go"}`,
						},
					}},
				},
				FinishReason: &frTools,
			},
		},
	}

	body, err := json.Marshal(repaired)
	if err != nil {
		t.Fatalf("marshal repaired response: %v", err)
	}

	var out bytes.Buffer
	if err := writeOpenAIStreamFromResponse(&out, body, "qwen3.7-plus"); err != nil {
		t.Fatalf("writeOpenAIStreamFromResponse: %v", err)
	}

	bodyOut := out.String()
	if !strings.Contains(bodyOut, `"index":0`) || !strings.Contains(bodyOut, `"index":1`) {
		t.Fatalf("expected SSE output for both choice indices: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"finish_reason":"stop","index":0`) {
		t.Fatalf("expected final stop chunk for choice index 0: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"finish_reason":"tool_calls","index":1`) {
		t.Fatalf("expected final tool_calls chunk for choice index 1: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, "data: [DONE]") {
		t.Fatalf("expected DONE marker in SSE output: %s", bodyOut)
	}
}

func TestV1ChatCompletions_MiniMaxM3_StreamInvokeSetsToolCallsFinish(t *testing.T) {
	upstream := `{
		"id":"mmx-1",
		"object":"chat.completion",
		"created":1780734753,
		"model":"MiniMax-M3",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"run_in_terminal\">]<]minimax[>[<command>cd /tmp/demo && git log --oneline</command>]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>"
			},
			"finish_reason":"stop"
		}]
	}`

	h := newMiniMaxV1Handler(t, upstream)
	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"check latest diff"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if !strings.Contains(bodyOut, `"name":"run_in_terminal"`) {
		t.Fatalf("expected run_in_terminal tool call in rewritten stream: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"finish_reason":"tool_calls"`) {
		t.Fatalf("expected finish_reason tool_calls in rewritten stream: %s", bodyOut)
	}
}

func TestV1ChatCompletions_MiniMaxM3_StreamBareObjectToolCalls(t *testing.T) {
	upstream := `{
		"id":"mmx-obj-1",
		"object":"chat.completion",
		"created":1780743307,
		"model":"MiniMax-M3",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"<think>reasoning</think>I will inspect files.]<]minimax[>[<tool_call>\n{file_search:{root:\"/tmp/repo\",pattern:\"**/*.go\"}}\n{read_file:{filepath:\"/tmp/repo/README.md\"}}\n"
			},
			"finish_reason":"stop"
		}]
	}`

	h := newMiniMaxV1Handler(t, upstream)
	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"inspect repo"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	if strings.Contains(bodyOut, "<think>") || strings.Contains(bodyOut, "<tool_call>") || strings.Contains(bodyOut, "]<]minimax[>") {
		t.Fatalf("stream response still contains unsupported tags: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"name":"file_search"`) {
		t.Fatalf("expected file_search tool call in rewritten stream: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"name":"read_file"`) {
		t.Fatalf("expected read_file tool call in rewritten stream: %s", bodyOut)
	}
	if !strings.Contains(bodyOut, `"finish_reason":"tool_calls"`) {
		t.Fatalf("expected finish_reason tool_calls in rewritten stream: %s", bodyOut)
	}
}

func TestV1ChatCompletions_MiniMax_LooseToolTextUsesRequestHints(t *testing.T) {
	upstream := `{
		"id":"mmx-loose-1",
		"object":"chat.completion",
		"created":1780750522,
		"model":"MiniMax-M3",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":"Let me find all the ` + "`io.ReadAll(r.Body)`" + ` sites:\n\nsemantic_search\">\nSearch for all \"io.ReadAll(r.Body)\" invocations in handler.go\n\n</tool_call>"
			},
			"finish_reason":"stop"
		}]
	}`

	h := newMiniMaxV1Handler(t, upstream)
	body := `{"model":"minimax-m3","messages":[{"role":"user","content":"find the body reads"}],"tools":[{"type":"function","function":{"name":"semantic_search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.V1ChatCompletions()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	bodyOut := w.Body.String()
	var payload map[string]interface{}
	if err := json.Unmarshal([]byte(bodyOut), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v (%s)", err, bodyOut)
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) != 1 {
		t.Fatalf("expected 1 choice, got %d: %s", len(choices), bodyOut)
	}
	choice, _ := choices[0].(map[string]interface{})
	if fr, _ := choice["finish_reason"].(string); fr != "tool_calls" {
		t.Fatalf("expected finish_reason tool_calls, got %q: %s", fr, bodyOut)
	}
	msg, _ := choice["message"].(map[string]interface{})
	if content, _ := msg["content"].(string); content != "Let me find all the `io.ReadAll(r.Body)` sites:" {
		t.Fatalf("expected leading assistant content to be preserved, got %q", content)
	}
	toolCalls, _ := msg["tool_calls"].([]interface{})
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %s", len(toolCalls), bodyOut)
	}
	tc, _ := toolCalls[0].(map[string]interface{})
	fn, _ := tc["function"].(map[string]interface{})
	if fn["name"] != "semantic_search" {
		t.Fatalf("expected semantic_search tool call, got %q", fn["name"])
	}
	args, _ := fn["arguments"].(string)
	var parsed map[string]string
	if err := json.Unmarshal([]byte(args), &parsed); err != nil {
		t.Fatalf("invalid arguments JSON: %v (%s)", err, args)
	}
	if parsed["query"] != "Search for all \"io.ReadAll(r.Body)\" invocations in handler.go" {
		t.Fatalf("expected query argument, got %q", parsed["query"])
	}
}

// newTestHandlerWithRedaction builds a handler whose upstream records the
// raw request body it receives. The handler itself is configured with a
// real gitleaks redactor in hide mode so we can assert that secrets
// embedded in the client request never reach the upstream.
func newTestHandlerWithRedaction(t *testing.T) (*Handler, *[]byte) {
	t.Helper()
	var captured []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			b, _ := io.ReadAll(r.Body)
			r.Body.Close()
			captured = b
		}
		// Reply with a minimal valid OpenAI-shape chat completion.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
			},
			"usage": map[string]interface{}{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	}))
	t.Cleanup(upstream.Close)

	redactor, err := redact.New(true, redact.ModeHide)
	if err != nil {
		t.Fatalf("redact.New: %v", err)
	}
	c := client.New(upstream.URL, "test-key")
	return New(c, "0.24.0", false, redactor), &captured
}

func TestChatEndpoint_RedactsSecretBeforeUpstream(t *testing.T) {
	h, captured := newTestHandlerWithRedaction(t)

	// A clearly high-entropy OpenAI-style key that gitleaks flags by default.
	const secret = "ghp_1234567890abcdefghijABCDEFGHIJKLMNopqrstuvwxyz"
	body := `{"model":"glm-5.1","messages":[{"role":"user","content":"here is a key: ` + secret + `"}],"stream":false}`

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Chat()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if len(*captured) == 0 {
		t.Fatal("upstream did not receive a body")
	}
	if strings.Contains(string(*captured), secret) {
		t.Fatalf("secret leaked to upstream; body=%s", string(*captured))
	}
}

func TestGenerateEndpoint_RedactsSecretBeforeUpstream(t *testing.T) {
	h, captured := newTestHandlerWithRedaction(t)

	const secret = "ghp_1234567890abcdefghijABCDEFGHIJKLMNopqrstuvwxyz"
	body := `{"model":"glm-5.1","prompt":"token: ` + secret + `","stream":false}`

	req := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Generate()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if strings.Contains(string(*captured), secret) {
		t.Fatalf("secret leaked to upstream; body=%s", string(*captured))
	}
}

func TestChatEndpoint_NoopRedactorPassesBodyThrough(t *testing.T) {
	h := newTestHandler() // uses redact.NewNoop()
	const secret = "sk-IMO8bkaQGwuUzfnMTLZzSV0FCToTrfG6h3qgTYZ6wm9HRZ6ImkZbED6NKSembk49"
	body := `{"model":"glm-5.1","messages":[{"role":"user","content":"key: ` + secret + `"}],"stream":false}`

	req := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.Chat()(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	// Noop should not alter the body. We can't inspect what hit the upstream
	// in this fixture, but the request must at least still succeed and the
	// handler must not error out trying to parse the redacted body.
}
