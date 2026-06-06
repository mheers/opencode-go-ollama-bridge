package handler

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
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
	return New(c, "0.24.0", false)
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
	return New(c, "0.24.0", false)
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
	return New(c, "0.24.0", false)
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
