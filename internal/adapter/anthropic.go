package adapter

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

type AnthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Messages    []AnthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	Tools       []AnthropicTool    `json:"tools,omitempty"`
	Temperature *float64           `json:"temperature,omitempty"`
	TopP        *float64           `json:"top_p,omitempty"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type AnthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type AnthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

func ChatRequestToAnthropic(ollamaReq *OllamaChatRequest) *AnthropicRequest {
	stream := false
	if ollamaReq.Stream != nil {
		stream = *ollamaReq.Stream
	}

	const maxTokens = 65536

	req := &AnthropicRequest{
		Model:     ollamaReq.Model,
		MaxTokens: maxTokens,
		Messages:  make([]AnthropicMessage, 0, len(ollamaReq.Messages)),
		Stream:    stream,
	}

	for _, msg := range ollamaReq.Messages {
		if msg.Role == "system" {
			req.System = msg.Content
			continue
		}

		anthMsg := AnthropicMessage{
			Role: msg.Role,
		}

		if msg.Role == "tool" {
			contentBlocks := []AnthropicContentBlock{
				{
					Type:      "tool_result",
					ToolUseID: msg.ToolName,
					Content:   msg.Content,
				},
			}
			anthMsg.Role = "user"
			anthMsg.Content = contentBlocks
		} else if len(msg.Images) > 0 {
			contentBlocks := make([]AnthropicContentBlock, 0, 1+len(msg.Images))
			if msg.Content != "" {
				contentBlocks = append(contentBlocks, AnthropicContentBlock{
					Type: "text",
					Text: msg.Content,
				})
			}
			for _, img := range msg.Images {
				contentBlocks = append(contentBlocks, AnthropicContentBlock{
					Type: "image",
					ID:   img,
				})
			}
			anthMsg.Content = contentBlocks
		} else if len(msg.ToolCalls) > 0 {
			contentBlocks := make([]AnthropicContentBlock, 0)
			for _, tc := range msg.ToolCalls {
				var input json.RawMessage
				if len(tc.Function.Arguments) > 0 {
					input = tc.Function.Arguments
				}
				contentBlocks = append(contentBlocks, AnthropicContentBlock{
					Type:  "tool_use",
					ID:    fmt.Sprintf("toolu_%s", tc.Function.Name),
					Name:  tc.Function.Name,
					Input: input,
				})
			}
			anthMsg.Content = contentBlocks
		} else {
			anthMsg.Content = msg.Content
		}

		req.Messages = append(req.Messages, anthMsg)
	}

	if len(ollamaReq.Tools) > 0 {
		req.Tools = make([]AnthropicTool, len(ollamaReq.Tools))
		for i, t := range ollamaReq.Tools {
			req.Tools[i] = AnthropicTool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: t.Function.Parameters,
			}
		}
	}

	if ollamaReq.Options != nil {
		req.Temperature = ollamaReq.Options.Temperature
		req.TopP = ollamaReq.Options.TopP
	}

	return req
}

type AnthropicSSEEvent struct {
	Type         string                 `json:"type"`
	Index        *int                   `json:"index,omitempty"`
	Delta        *AnthropicDelta        `json:"delta,omitempty"`
	ContentBlock *AnthropicContentBlock `json:"content_block,omitempty"`
	Message      *AnthropicMsgMeta      `json:"message,omitempty"`
	Usage        *AnthropicUsage        `json:"usage,omitempty"`
}

type AnthropicDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}

type AnthropicMsgMeta struct {
	ID    string `json:"id"`
	Type  string `json:"type"`
	Role  string `json:"role"`
	Model string `json:"model"`
}

type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

func AnthropicStreamToOllamaNDJSON(reader io.Reader, model string, writer io.Writer) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var (
		toolCalls       []OllamaToolCall
		textContent     strings.Builder
		currentTool     int = -1
		toolAccumulator strings.Builder
		sawMessageStop  bool
	)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			writeEvent := true
			var resp *OllamaChatResponse

			if sawMessageStop {
				resp = &OllamaChatResponse{
					Model:      model,
					CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
					Message:    &OllamaRespMsg{Role: "assistant", Content: ""},
					Done:       true,
					DoneReason: "stop",
				}
				sawMessageStop = false
			} else if textContent.Len() > 0 || toolAccumulator.Len() > 0 {
				msg := &OllamaRespMsg{Role: "assistant"}
				if textContent.Len() > 0 {
					msg.Content = textContent.String()
					textContent.Reset()
				}
				if toolAccumulator.Len() > 0 {
					var args json.RawMessage
					s := toolAccumulator.String()
					if json.Valid([]byte(s)) {
						args = json.RawMessage(s)
					} else {
						args = json.RawMessage(s)
					}
					if len(toolCalls) == 0 {
						toolCalls = append(toolCalls, OllamaToolCall{
							Function: OllamaToolFunction{Arguments: args},
						})
					}
					toolAccumulator.Reset()
				}
				if len(toolCalls) > 0 {
					msg.ToolCalls = toolCalls
				}
				resp = &OllamaChatResponse{
					Model:     model,
					CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
					Message:   msg,
					Done:      false,
				}
			} else {
				writeEvent = false
			}

			if writeEvent {
				out, _ := json.Marshal(resp)
				if _, err := writer.Write(append(out, '\n')); err != nil {
					return err
				}
			}
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var event AnthropicSSEEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "content_block_start":
			if event.ContentBlock != nil {
				switch event.ContentBlock.Type {
				case "text":
					textContent.Reset()
				case "tool_use":
					currentTool = *event.Index
					toolCalls = append(toolCalls, OllamaToolCall{
						Function: OllamaToolFunction{
							Name: event.ContentBlock.Name,
						},
					})
				}
			}
		case "content_block_delta":
			if event.Delta != nil {
				switch event.Delta.Type {
				case "text_delta":
					textContent.WriteString(event.Delta.Text)
				case "input_json_delta":
					toolAccumulator.WriteString(event.Delta.PartialJSON)
				}
			}
		case "content_block_stop":
			if toolAccumulator.Len() > 0 && currentTool >= 0 && currentTool < len(toolCalls) {
				toolCalls[currentTool].Function.Arguments = json.RawMessage(toolAccumulator.String())
			}
			toolAccumulator.Reset()
			currentTool = -1
		case "message_stop":
			sawMessageStop = true
		}
	}

	if scanner.Err() != nil {
		return scanner.Err()
	}

	final := &OllamaChatResponse{
		Model:      model,
		CreatedAt:  time.Now().UTC().Format(time.RFC3339Nano),
		Message:    &OllamaRespMsg{Role: "assistant", Content: ""},
		Done:       true,
		DoneReason: "stop",
	}
	out, _ := json.Marshal(final)
	writer.Write(append(out, '\n'))

	return nil
}
