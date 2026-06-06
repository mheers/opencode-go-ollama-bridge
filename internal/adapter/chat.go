package adapter

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
)

type OllamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []OllamaMessage `json:"messages"`
	Tools     []OllamaTool    `json:"tools,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Options   *OllamaOptions  `json:"options,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	KeepAlive *string         `json:"keep_alive,omitempty"`
}

type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
}

type OllamaToolCall struct {
	Function OllamaToolFunction `json:"function"`
}

type OllamaToolFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type OllamaTool struct {
	Type     string                `json:"type"`
	Function OllamaToolFunctionDef `json:"function"`
}

type OllamaToolFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type OllamaOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
	Seed        *int     `json:"seed,omitempty"`
	NumPredict  *int     `json:"num_predict,omitempty"`
}

type OpenAIRequest struct {
	Model       string          `json:"model"`
	Messages    []OpenAIMessage `json:"messages"`
	Stream      bool            `json:"stream,omitempty"`
	Tools       []OpenAITool    `json:"tools,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Seed        *int            `json:"seed,omitempty"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
}

type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    interface{}      `json:"content"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type OpenAIToolCall struct {
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

type OpenAIToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type OpenAITool struct {
	Type     string                `json:"type"`
	Function OpenAIToolFunctionDef `json:"function"`
}

type OpenAIToolFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
	Choices []OpenAIChoice `json:"choices"`
}

type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type OpenAIChoice struct {
	Index        int                `json:"index"`
	Message      *OpenAIMessageResp `json:"message,omitempty"`
	Delta        *OpenAIDelta       `json:"delta,omitempty"`
	FinishReason *string            `json:"finish_reason,omitempty"`
}

type OpenAIMessageResp struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []OpenAIToolCall `json:"tool_calls,omitempty"`
}

type OpenAIDelta struct {
	Role      string                `json:"role,omitempty"`
	Content   string                `json:"content,omitempty"`
	ToolCalls []OpenAIToolCallDelta `json:"tool_calls,omitempty"`
}

type OpenAIToolCallDelta struct {
	Index    int                `json:"index"`
	ID       string             `json:"id,omitempty"`
	Type     string             `json:"type,omitempty"`
	Function OpenAIToolFunction `json:"function,omitempty"`
}

func ChatRequestToOpenAI(ollamaReq *OllamaChatRequest) *OpenAIRequest {
	stream := false
	if ollamaReq.Stream != nil {
		stream = *ollamaReq.Stream
	}

	req := &OpenAIRequest{
		Model:    ollamaReq.Model,
		Messages: make([]OpenAIMessage, 0, len(ollamaReq.Messages)),
		Stream:   stream,
	}

	for _, msg := range ollamaReq.Messages {
		openaiMsg := OpenAIMessage{
			Role: msg.Role,
		}

		if len(msg.Images) > 0 {
			contentParts := make([]map[string]interface{}, 0, 1+len(msg.Images))
			if msg.Content != "" {
				contentParts = append(contentParts, map[string]interface{}{
					"type": "text",
					"text": msg.Content,
				})
			}
			for _, img := range msg.Images {
				contentParts = append(contentParts, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]string{
						"url": "data:image/jpeg;base64," + img,
					},
				})
			}
			openaiMsg.Content = contentParts
		} else {
			openaiMsg.Content = msg.Content
		}

		if msg.Role == "tool" {
			openaiMsg.Role = "tool"
			openaiMsg.ToolCallID = msg.ToolName
		}

		if len(msg.ToolCalls) > 0 {
			openaiMsg.ToolCalls = make([]OpenAIToolCall, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				openaiMsg.ToolCalls[i] = OpenAIToolCall{
					Type: "function",
					Function: OpenAIToolFunction{
						Name:      tc.Function.Name,
						Arguments: string(tc.Function.Arguments),
					},
				}
			}
		}

		req.Messages = append(req.Messages, openaiMsg)
	}

	if len(ollamaReq.Tools) > 0 {
		req.Tools = make([]OpenAITool, len(ollamaReq.Tools))
		for i, t := range ollamaReq.Tools {
			req.Tools[i] = OpenAITool{
				Type: t.Type,
				Function: OpenAIToolFunctionDef{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			}
		}
	}

	if ollamaReq.Options != nil {
		req.Temperature = ollamaReq.Options.Temperature
		req.TopP = ollamaReq.Options.TopP
		req.Seed = ollamaReq.Options.Seed
		req.MaxTokens = ollamaReq.Options.NumPredict
	}

	return req
}

type OllamaChatResponse struct {
	Model              string         `json:"model"`
	CreatedAt          string         `json:"created_at"`
	Message            *OllamaRespMsg `json:"message"`
	Done               bool           `json:"done"`
	DoneReason         string         `json:"done_reason,omitempty"`
	TotalDuration      int64          `json:"total_duration,omitempty"`
	LoadDuration       int64          `json:"load_duration,omitempty"`
	PromptEvalCount    int            `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64          `json:"prompt_eval_duration,omitempty"`
	EvalCount          int            `json:"eval_count,omitempty"`
	EvalDuration       int64          `json:"eval_duration,omitempty"`
}

type OllamaRespMsg struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    interface{}      `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}

func OpenAIResponseToOllamaChat(openaiResp *OpenAIResponse, model string) *OllamaChatResponse {
	resp := &OllamaChatResponse{
		Model:     model,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Done:      true,
	}

	if len(openaiResp.Choices) > 0 {
		choice := openaiResp.Choices[0]
		msg := choice.Message
		if msg == nil {
			resp.Message = &OllamaRespMsg{Role: "assistant", Content: ""}
		} else {
			ollamaMsg := &OllamaRespMsg{
				Role:    msg.Role,
				Content: msg.Content,
			}
			if len(msg.ToolCalls) > 0 {
				ollamaMsg.ToolCalls = make([]OllamaToolCall, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					var args json.RawMessage
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
						args = json.RawMessage(tc.Function.Arguments)
					}
					ollamaMsg.ToolCalls[i] = OllamaToolCall{
						Function: OllamaToolFunction{
							Name:      tc.Function.Name,
							Arguments: args,
						},
					}
				}
			}
			resp.Message = ollamaMsg
		}
		if choice.FinishReason != nil {
			resp.DoneReason = *choice.FinishReason
		}
	} else {
		resp.Message = &OllamaRespMsg{Role: "assistant", Content: ""}
	}

	if openaiResp.Usage != nil {
		resp.PromptEvalCount = openaiResp.Usage.PromptTokens
		resp.EvalCount = openaiResp.Usage.CompletionTokens
	}

	return resp
}

type OllamaGenerateRequest struct {
	Model   string          `json:"model"`
	Prompt  string          `json:"prompt"`
	System  string          `json:"system,omitempty"`
	Stream  *bool           `json:"stream,omitempty"`
	Options *OllamaOptions  `json:"options,omitempty"`
	Format  json.RawMessage `json:"format,omitempty"`
}

func GenerateRequestToOpenAI(generateReq *OllamaGenerateRequest) *OpenAIRequest {
	stream := false
	if generateReq.Stream != nil {
		stream = *generateReq.Stream
	}

	messages := []OpenAIMessage{
		{Role: "user", Content: generateReq.Prompt},
	}
	if generateReq.System != "" {
		messages = append([]OpenAIMessage{{Role: "system", Content: generateReq.System}}, messages...)
	}

	req := &OpenAIRequest{
		Model:    generateReq.Model,
		Messages: messages,
		Stream:   stream,
	}

	if generateReq.Options != nil {
		req.Temperature = generateReq.Options.Temperature
		req.TopP = generateReq.Options.TopP
		req.Seed = generateReq.Options.Seed
		req.MaxTokens = generateReq.Options.NumPredict
	}

	return req
}

// OpenAIRequestToOllama converts an OpenAI /v1/chat/completions request to the
// internal OllamaChatRequest representation so it can be forwarded to the
// Anthropic messages endpoint via ChatRequestToAnthropic.
func OpenAIRequestToOllama(req *OpenAIRequest) *OllamaChatRequest {
	stream := req.Stream
	ollamaReq := &OllamaChatRequest{
		Model:    req.Model,
		Stream:   &stream,
		Messages: make([]OllamaMessage, 0, len(req.Messages)),
	}

	if req.Temperature != nil || req.TopP != nil || req.MaxTokens != nil {
		ollamaReq.Options = &OllamaOptions{
			Temperature: req.Temperature,
			TopP:        req.TopP,
		}
		if req.MaxTokens != nil {
			ollamaReq.Options.NumPredict = req.MaxTokens
		}
	}

	for _, msg := range req.Messages {
		ollamaMsg := OllamaMessage{Role: msg.Role}

		switch v := msg.Content.(type) {
		case string:
			ollamaMsg.Content = v
		case []interface{}:
			// Multi-part content: collect text parts only.
			for _, part := range v {
				if pm, ok := part.(map[string]interface{}); ok {
					if pm["type"] == "text" {
						if text, ok := pm["text"].(string); ok {
							ollamaMsg.Content += text
						}
					}
				}
			}
		}

		if msg.Role == "tool" {
			ollamaMsg.ToolName = msg.ToolCallID
		}

		if len(msg.ToolCalls) > 0 {
			ollamaMsg.ToolCalls = make([]OllamaToolCall, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				var args json.RawMessage
				if tc.Function.Arguments != "" {
					args = json.RawMessage(tc.Function.Arguments)
				}
				ollamaMsg.ToolCalls[i] = OllamaToolCall{
					Function: OllamaToolFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				}
			}
		}

		ollamaReq.Messages = append(ollamaReq.Messages, ollamaMsg)
	}

	if len(req.Tools) > 0 {
		ollamaReq.Tools = make([]OllamaTool, len(req.Tools))
		for i, t := range req.Tools {
			ollamaReq.Tools[i] = OllamaTool{
				Type: t.Type,
				Function: OllamaToolFunctionDef{
					Name:        t.Function.Name,
					Description: t.Function.Description,
					Parameters:  t.Function.Parameters,
				},
			}
		}
	}

	return ollamaReq
}

func IsMiniMaxModel(modelID string) bool {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	return strings.HasPrefix(modelID, "minimax-")
}

// IsAnthropicOnlyModel returns true for models that only support the Anthropic
// messages API and return an error on the OpenAI-compatible /chat/completions
// endpoint ("not supported for format oa-compat").
func IsAnthropicOnlyModel(modelID string) bool {
	return modelID == "qwen3.7-max"
}

func MapModelsToOllama(models []client.Model) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(models))
	now := time.Now().UTC().Format(time.RFC3339)
	for _, m := range models {
		result = append(result, map[string]interface{}{
			"name":        m.ID + ":latest",
			"model":       m.ID + ":latest",
			"modified_at": now,
			"size":        0,
			"digest":      sha256Short(m.ID),
			"details": map[string]interface{}{
				"family":         m.ID,
				"parameter_size": "unknown",
				"format":         "gguf",
			},
		})
	}
	return result
}

func sha256Short(s string) string {
	h := 0
	for _, c := range []byte(s) {
		h = h*31 + int(c)
	}
	result := fmt.Sprintf("%08x", uint32(h))
	for len(result) < 64 {
		result += "0"
	}
	return result[:64]
}
