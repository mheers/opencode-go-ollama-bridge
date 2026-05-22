package adapter

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
)

func TestChatRequestToOpenAI_Basic(t *testing.T) {
	ollamaReq := &OllamaChatRequest{
		Model: "glm-5.1",
		Messages: []OllamaMessage{
			{Role: "user", Content: "hello"},
		},
	}

	req := ChatRequestToOpenAI(ollamaReq)

	if req.Model != "glm-5.1" {
		t.Errorf("model = %q", req.Model)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != "user" {
		t.Errorf("role = %q", req.Messages[0].Role)
	}
	if req.Messages[0].Content != "hello" {
		t.Errorf("content = %q", req.Messages[0].Content)
	}
	if req.Stream != false {
		t.Errorf("stream should be false by default")
	}
}

func TestChatRequestToOpenAI_Streaming(t *testing.T) {
	stream := true
	ollamaReq := &OllamaChatRequest{
		Model:    "kimi-k2.6",
		Messages: []OllamaMessage{{Role: "user", Content: "test"}},
		Stream:   &stream,
	}

	req := ChatRequestToOpenAI(ollamaReq)
	if !req.Stream {
		t.Error("stream should be true")
	}
}

func TestChatRequestToOpenAI_SystemMessage(t *testing.T) {
	ollamaReq := &OllamaChatRequest{
		Model: "deepseek-v4-pro",
		Messages: []OllamaMessage{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "test"},
		},
	}

	req := ChatRequestToOpenAI(ollamaReq)
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != "system" {
		t.Errorf("first message role = %q", req.Messages[0].Role)
	}
}

func TestChatRequestToOpenAI_WithTools(t *testing.T) {
	ollamaReq := &OllamaChatRequest{
		Model: "glm-5.1",
		Messages: []OllamaMessage{
			{Role: "user", Content: "weather?"},
		},
		Tools: []OllamaTool{
			{
				Type: "function",
				Function: OllamaToolFunctionDef{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
			},
		},
	}

	req := ChatRequestToOpenAI(ollamaReq)
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "get_weather" {
		t.Errorf("tool name = %q", req.Tools[0].Function.Name)
	}
}

func TestChatRequestToOpenAI_WithOptions(t *testing.T) {
	temp := 0.7
	ollamaReq := &OllamaChatRequest{
		Model:    "qwen3.6-plus",
		Messages: []OllamaMessage{{Role: "user", Content: "test"}},
		Options: &OllamaOptions{
			Temperature: &temp,
		},
	}

	req := ChatRequestToOpenAI(ollamaReq)
	if req.Temperature == nil || *req.Temperature != 0.7 {
		t.Errorf("temperature = %v", req.Temperature)
	}
}

func TestChatRequestToOpenAI_WithImages(t *testing.T) {
	base64img := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
	ollamaReq := &OllamaChatRequest{
		Model: "glm-5.1",
		Messages: []OllamaMessage{
			{Role: "user", Content: "what is this?", Images: []string{base64img}},
		},
	}

	req := ChatRequestToOpenAI(ollamaReq)
	content, ok := req.Messages[0].Content.([]map[string]interface{})
	if !ok {
		t.Fatalf("content should be array, got %T", req.Messages[0].Content)
	}
	if len(content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(content))
	}
}

func TestChatRequestToOpenAI_ToolMessage(t *testing.T) {
	ollamaReq := &OllamaChatRequest{
		Model: "glm-5.1",
		Messages: []OllamaMessage{
			{Role: "user", Content: "weather?"},
			{Role: "assistant", Content: "", ToolCalls: []OllamaToolCall{
				{Function: OllamaToolFunction{Name: "get_weather", Arguments: json.RawMessage(`{"city":"Tokyo"}`)}},
			}},
			{Role: "tool", Content: "25°C", ToolName: "get_weather"},
		},
	}

	req := ChatRequestToOpenAI(ollamaReq)
	if len(req.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(req.Messages))
	}
	toolMsg := req.Messages[2]
	if toolMsg.Role != "tool" {
		t.Errorf("role = %q", toolMsg.Role)
	}
	if toolMsg.ToolCallID != "get_weather" {
		t.Errorf("tool_call_id = %q", toolMsg.ToolCallID)
	}
}

func TestOpenAIResponseToOllamaChat(t *testing.T) {
	openaiResp := &OpenAIResponse{
		ID:      "chatcmpl-123",
		Model:   "glm-5.1",
		Created: 1234567890,
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: &OpenAIMessageResp{
					Role:    "assistant",
					Content: "Hello!",
				},
				FinishReason: strPtr("stop"),
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}

	resp := OpenAIResponseToOllamaChat(openaiResp, "glm-5.1")
	if resp.Model != "glm-5.1" {
		t.Errorf("model = %q", resp.Model)
	}
	if !resp.Done {
		t.Error("done should be true")
	}
	if resp.Message.Content != "Hello!" {
		t.Errorf("content = %q", resp.Message.Content)
	}
	if resp.DoneReason != "stop" {
		t.Errorf("done_reason = %q", resp.DoneReason)
	}
	if resp.PromptEvalCount != 10 {
		t.Errorf("prompt_eval_count = %d", resp.PromptEvalCount)
	}
	if resp.EvalCount != 5 {
		t.Errorf("eval_count = %d", resp.EvalCount)
	}
}

func TestOpenAIResponseToOllamaChat_WithToolCalls(t *testing.T) {
	openaiResp := &OpenAIResponse{
		ID:      "chatcmpl-456",
		Model:   "glm-5.1",
		Created: 1234567890,
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: &OpenAIMessageResp{
					Role:    "assistant",
					Content: "",
					ToolCalls: []OpenAIToolCall{
						{
							Type: "function",
							Function: OpenAIToolFunction{
								Name:      "get_weather",
								Arguments: `{"city":"Tokyo"}`,
							},
						},
					},
				},
			},
		},
	}

	resp := OpenAIResponseToOllamaChat(openaiResp, "glm-5.1")
	if resp.Message.ToolCalls == nil {
		t.Fatal("expected tool_calls")
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(resp.Message.ToolCalls))
	}
	if resp.Message.ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("tool name = %q", resp.Message.ToolCalls[0].Function.Name)
	}
}

func TestGenerateRequestToOpenAI(t *testing.T) {
	genReq := &OllamaGenerateRequest{
		Model:  "glm-5.1",
		Prompt: "Why is the sky blue?",
		System: "You are helpful.",
	}

	req := GenerateRequestToOpenAI(genReq)

	if req.Model != "glm-5.1" {
		t.Errorf("model = %q", req.Model)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(req.Messages))
	}
	if req.Messages[0].Role != "system" {
		t.Errorf("first message role = %q", req.Messages[0].Role)
	}
	if req.Messages[1].Content != "Why is the sky blue?" {
		t.Errorf("prompt = %q", req.Messages[1].Content)
	}
}

func TestIsMiniMaxModel(t *testing.T) {
	if !IsMiniMaxModel("minimax-m2.5") {
		t.Error("minimax-m2.5 should be detected")
	}
	if !IsMiniMaxModel("minimax-m2.7") {
		t.Error("minimax-m2.7 should be detected")
	}
	if IsMiniMaxModel("glm-5.1") {
		t.Error("glm-5.1 should not be detected as minimax")
	}
}

func TestMapModelsToOllama(t *testing.T) {
	models := []client.Model{
		{ID: "glm-5.1"},
		{ID: "kimi-k2.6"},
	}

	result := MapModelsToOllama(models)
	if len(result) != 2 {
		t.Fatalf("expected 2 models, got %d", len(result))
	}
	if result[0]["model"] != "glm-5.1:latest" {
		t.Errorf("first model = %q", result[0]["model"])
	}
}

func TestStreamOpenAI_TextContent(t *testing.T) {
	chunk := &OpenAIResponse{
		Choices: []OpenAIChoice{
			{
				Delta: &OpenAIDelta{
					Content: "Hello",
				},
				FinishReason: nil,
			},
		},
	}

	resp := StreamOpenAI(chunk, "glm-5.1")
	if resp.Done {
		t.Error("should not be done")
	}
	if resp.Message.Content != "Hello" {
		t.Errorf("content = %q", resp.Message.Content)
	}
}

func TestStreamOpenAI_Finish(t *testing.T) {
	chunk := &OpenAIResponse{
		Choices: []OpenAIChoice{
			{
				Delta:        &OpenAIDelta{},
				FinishReason: strPtr("stop"),
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}

	resp := StreamOpenAI(chunk, "glm-5.1")
	if !resp.Done {
		t.Error("should be done")
	}
	if resp.DoneReason != "stop" {
		t.Errorf("done_reason = %q", resp.DoneReason)
	}
	if resp.EvalCount != 5 {
		t.Errorf("eval_count = %d", resp.EvalCount)
	}
}

func TestOpenAIStreamToOllamaNDJSON(t *testing.T) {
	input := `data: {"id":"1","choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"2","choices":[{"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"3","choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}

data: [DONE]
`

	var output strings.Builder
	err := OpenAIStreamToOllamaNDJSON(strings.NewReader(input), "glm-5.1", &output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) < 3 {
		t.Fatalf("expected at least 3 lines, got %d: %s", len(lines), output.String())
	}

	var firstResp OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[0]), &firstResp); err != nil {
		t.Fatalf("failed to parse first chunk: %v", err)
	}
	if firstResp.Done {
		t.Error("first chunk should not be done")
	}
	if firstResp.Message.Content != "Hello" {
		t.Errorf("first content = %q", firstResp.Message.Content)
	}

	var lastResp OllamaChatResponse
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &lastResp); err != nil {
		t.Fatalf("failed to parse last chunk: %v", err)
	}
	if !lastResp.Done {
		t.Error("last chunk should be done")
	}
}

func TestChatRequestToAnthropic_Basic(t *testing.T) {
	ollamaReq := &OllamaChatRequest{
		Model: "minimax-m2.5",
		Messages: []OllamaMessage{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "hello"},
		},
	}

	req := ChatRequestToAnthropic(ollamaReq)
	if req.Model != "minimax-m2.5" {
		t.Errorf("model = %q", req.Model)
	}
	if req.System != "You are helpful." {
		t.Errorf("system = %q", req.System)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message (system extracted), got %d", len(req.Messages))
	}
	if req.MaxTokens == 0 {
		t.Error("max_tokens should be set")
	}
}

func TestChatRequestToAnthropic_WithTools(t *testing.T) {
	ollamaReq := &OllamaChatRequest{
		Model: "minimax-m2.7",
		Messages: []OllamaMessage{
			{Role: "user", Content: "weather?"},
		},
		Tools: []OllamaTool{
			{
				Type: "function",
				Function: OllamaToolFunctionDef{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
			},
		},
	}

	req := ChatRequestToAnthropic(ollamaReq)
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Name != "get_weather" {
		t.Errorf("tool name = %q", req.Tools[0].Name)
	}
}

func TestAnthropicStreamToOllamaNDJSON(t *testing.T) {
	input := `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
`

	var output strings.Builder
	err := AnthropicStreamToOllamaNDJSON(strings.NewReader(input), "minimax-m2.5", &output)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) == 0 {
		t.Fatal("expected output lines")
	}
}

func strPtr(s string) *string {
	return &s
}
