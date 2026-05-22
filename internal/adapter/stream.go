package adapter

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
	"time"
)

func StreamOpenAI(chunk *OpenAIResponse, model string) *OllamaChatResponse {
	resp := &OllamaChatResponse{
		Model:     model,
		CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Done:      false,
	}

	if len(chunk.Choices) == 0 {
		resp.Message = &OllamaRespMsg{Role: "assistant", Content: ""}
		return resp
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	msg := &OllamaRespMsg{Role: "assistant"}

	if delta != nil {
		msg.Content = delta.Content

		if len(delta.ToolCalls) > 0 {
			msg.ToolCalls = make([]OllamaToolCall, 0, len(delta.ToolCalls))
			for _, dtc := range delta.ToolCalls {
				var args json.RawMessage
				if dtc.Function.Arguments != "" {
					if err := json.Unmarshal([]byte(dtc.Function.Arguments), &args); err != nil {
						args = json.RawMessage(dtc.Function.Arguments)
					}
				}
				msg.ToolCalls = append(msg.ToolCalls, OllamaToolCall{
					Function: OllamaToolFunction{
						Name:      dtc.Function.Name,
						Arguments: args,
					},
				})
			}
		}
	}

	resp.Message = msg

	if choice.FinishReason != nil && *choice.FinishReason != "" {
		resp.Done = true
		resp.DoneReason = *choice.FinishReason
		if chunk.Usage != nil {
			resp.PromptEvalCount = chunk.Usage.PromptTokens
			resp.EvalCount = chunk.Usage.CompletionTokens
		}
	}

	return resp
}

func OpenAIStreamToOllamaNDJSON(reader io.Reader, model string, writer io.Writer) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line == "data: [DONE]" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")

		var chunk OpenAIResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		ollamaResp := StreamOpenAI(&chunk, model)

		out, err := json.Marshal(ollamaResp)
		if err != nil {
			continue
		}
		if _, err := writer.Write(append(out, '\n')); err != nil {
			return err
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
