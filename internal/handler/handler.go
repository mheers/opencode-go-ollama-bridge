package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/mheers/opencode-go-ollama-bridge/internal/adapter"
	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
)

// dsmlSep is the fence used by DeepSeek's DSML tool-call markup:
// <｜｜DSML｜｜tool_calls> / <｜｜DSML｜｜invoke name="…"> / <｜｜DSML｜｜parameter name="…">
const dsmlSep = "｜｜DSML｜｜"

var (
	miniMaxWrapperRE = regexp.MustCompile(`\]<\]minimax\[>\[`)
	thinkBlockRE     = regexp.MustCompile(`(?is)<think>.*?</think>`)
	toolCallInnerRE  = regexp.MustCompile(`(?is)<tool_call\b[^>]*>(.*?)</tool_call>`)
	toolCallBlockRE  = regexp.MustCompile(`(?is)<tool_call\b[^>]*>.*?</tool_call>`)
	functionTagRE    = regexp.MustCompile(`(?is)<function\s+([a-zA-Z_][a-zA-Z0-9_-]*)>(.*?)</function>`)
	parameterTagRE   = regexp.MustCompile(`(?is)<parameter=([a-zA-Z_][a-zA-Z0-9_-]*)>(.*?)</parameter>`)
	invokeTagRE      = regexp.MustCompile(`(?is)<invoke\s+name=\"([a-zA-Z_][a-zA-Z0-9_-]*)\">(.*?)</invoke>`)
	commandTagRE     = regexp.MustCompile(`(?is)<command>(.*?)</command>`)
	thinkTailRE      = regexp.MustCompile(`(?is)<think>.*$`)
	toolCallTailRE   = regexp.MustCompile(`(?is)<tool_call\b[^>]*>.*$`)
	bracketCallRE    = regexp.MustCompile(`\[\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s+(\{[^\]]*\})\s*\]`)
	bareCallNameRE   = regexp.MustCompile(`^[a-z_][a-z0-9_]{1,63}$`)
	multiBlankRE     = regexp.MustCompile(`\n{3,}`)

	// DeepSeek DSML format regexes — built at init time to embed the const.
	dsmlToolCallsBlockRE = regexp.MustCompile(`(?is)<` + dsmlSep + `tool_calls>(.*?)</` + dsmlSep + `tool_calls>`)
	dsmlInvokeRE         = regexp.MustCompile(`(?is)<` + dsmlSep + `invoke\s+name="([^"]+)">(.*?)</` + dsmlSep + `invoke>`)
	dsmlParamRE          = regexp.MustCompile(`(?is)<` + dsmlSep + `parameter\s+name="([^"]+)"[^>]*>(.*?)</` + dsmlSep + `parameter>`)
	dsmlToolCallsTailRE  = regexp.MustCompile(`(?is)<` + dsmlSep + `tool_calls>.*$`)

	// <tool_calls>…</tool_calls> (plural, no word-boundary) — used by hy3-preview
	// and possibly others routing via OpenRouter. May contain JSON, invoke-style
	// sub-tags, or be completely empty.
	toolCallsPluralInnerRE = regexp.MustCompile(`(?is)<tool_calls>(.*?)</tool_calls>`)
	toolCallsPluralBlockRE = regexp.MustCompile(`(?is)<tool_calls>.*?</tool_calls>`)
	toolCallsPluralTailRE  = regexp.MustCompile(`(?is)<tool_calls>.*$`)
)

type Handler struct {
	client        *client.Client
	ollamaVersion string
	debug         bool
}

func New(c *client.Client, ollamaVersion string, debug bool) *Handler {
	return &Handler{client: c, ollamaVersion: ollamaVersion, debug: debug}
}

func (h *Handler) logf(format string, args ...interface{}) {
	if h.debug {
		log.Printf(format, args...)
	}
}

func (h *Handler) logBody(tag string, body []byte) {
	if !h.debug {
		return
	}

	const maxLogBytes = 20000
	if len(body) <= maxLogBytes {
		h.logf("%s %s", tag, string(body))
		return
	}

	h.logf("%s %s ... [truncated %d bytes]", tag, string(body[:maxLogBytes]), len(body)-maxLogBytes)
}

func (h *Handler) Health() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.Write([]byte("Ollama is running"))
	}
}

func (h *Handler) Version() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h.logf("[VERSION] returning version %s", h.ollamaVersion)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"version": h.ollamaVersion,
		})
	}
}

func (h *Handler) Tags() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		models, err := h.client.ListModels()
		if err != nil {
			h.logf("[TAGS] failed to fetch models from OpenCode Go: %v", err)
			http.Error(w, `{"error":"failed to list models"}`, http.StatusInternalServerError)
			return
		}

		h.logf("[TAGS] fetched %d models from upstream", len(models))
		for _, m := range models {
			h.logf("[TAGS]   model: %s", m.ID)
		}

		ollamaModels := adapter.MapModelsToOllama(models)
		for _, m := range ollamaModels {
			name, _ := m["name"].(string)
			name = strings.TrimSuffix(name, ":latest")
			if details, ok := m["details"].(map[string]interface{}); ok {
				details["context_length"] = modelContextLength(name)
				details["parameter_size"] = modelParameterSize(name)
			}
		}

		resp := map[string]interface{}{
			"models": ollamaModels,
		}
		body, _ := json.Marshal(resp)
		h.logf("[TAGS] response: %s", string(body))

		w.Header().Set("Content-Type", "application/json")
		w.Write(body)
	}
}

func (h *Handler) PS() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"models":[]}`))
	}
}

func (h *Handler) Show() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		var req struct {
			Model   string `json:"model"`
			Verbose bool   `json:"verbose"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		modelID := strings.TrimSuffix(req.Model, ":latest")

		h.logf("[SHOW] model=%s verbose=%v", modelID, req.Verbose)

		ctxLen := modelContextLength(modelID)
		paramSize := modelParameterSize(modelID)
		arch := modelArchitecture(modelID)

		resp := map[string]interface{}{
			"modelfile":  "# Generated by opencode-go-ollama-bridge\nFROM " + modelID + "\n",
			"parameters": fmt.Sprintf("num_ctx %d", ctxLen),
			"template":   "{{ .Prompt }}",
			"details": map[string]interface{}{
				"parent_model":       "",
				"format":             "gguf",
				"family":             arch,
				"families":           []string{arch},
				"parameter_size":     paramSize,
				"quantization_level": "F16",
			},
			"model_info": map[string]interface{}{
				"general.architecture":         arch,
				"general.quantization_version": 2,
				"general.file_type":            1,
				"general.context_length":       ctxLen,
				arch + ".context_length":       ctxLen,
				arch + ".block_count":          32,
			},
			"capabilities": []string{"completion", "tools"},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func modelContextLength(modelID string) int {
	switch modelID {
	case "minimax-m2.7", "minimax-m2.5":
		return 1000000
	case "mimo-v2-pro", "mimo-v2-omni", "mimo-v2.5-pro", "mimo-v2.5":
		return 262144
	default:
		return 131072
	}
}

func modelParameterSize(modelID string) string {
	switch modelID {
	case "minimax-m2.7", "minimax-m2.5":
		return "unknown"
	case "deepseek-v4-pro":
		return "671B"
	case "deepseek-v4-flash":
		return "21B"
	case "qwen3.5-plus", "qwen3.6-plus":
		return "72B"
	case "glm-5", "glm-5.1":
		return "130B"
	case "kimi-k2.5", "kimi-k2.6":
		return "1T"
	case "mimo-v2.5", "mimo-v2.5-pro", "mimo-v2-pro", "mimo-v2-omni":
		return "unknown"
	default:
		return "unknown"
	}
}

func modelArchitecture(modelID string) string {
	switch {
	case strings.HasPrefix(modelID, "deepseek"):
		return "deepseek2"
	case strings.HasPrefix(modelID, "qwen"):
		return "qwen2"
	case strings.HasPrefix(modelID, "glm"):
		return "glm"
	case strings.HasPrefix(modelID, "kimi"):
		return "kimi"
	case strings.HasPrefix(modelID, "minimax"):
		return "minimax"
	case strings.HasPrefix(modelID, "mimo"):
		return "mimo"
	default:
		return modelID
	}
}

func (h *Handler) V1ChatCompletions() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
			return
		}

		var req struct {
			Model       string          `json:"model"`
			Messages    json.RawMessage `json:"messages"`
			Stream      bool            `json:"stream"`
			Tools       json.RawMessage `json:"tools"`
			Temperature *float64        `json:"temperature"`
			MaxTokens   *int            `json:"max_tokens"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			h.logf("[V1/CHAT] parse error: %v", err)
			http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
			return
		}

		modelID := strings.TrimSuffix(req.Model, ":latest")

		h.logf("[V1/CHAT] model=%s stream=%v body_has_reasoning=%v", modelID, req.Stream, strings.Contains(string(body), "reasoning_content"))

		// Anthropic-only models: translate OpenAI request → Anthropic messages API → OpenAI response.
		if adapter.IsAnthropicOnlyModel(modelID) {
			h.logf("[V1/CHAT] routing to anthropic backend (anthropic-only model) for %s", modelID)
			h.handleV1Anthropic(w, body, modelID, req.Stream)
			return
		}

		var upstreamResp *http.Response
		var upstreamErr error

		if adapter.IsMiniMaxModel(modelID) {
			h.logf("[V1/CHAT] routing to openai backend (repaired minimax path) for %s", modelID)
			var proxyReq map[string]interface{}
			if err := json.Unmarshal(body, &proxyReq); err != nil {
				http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
				return
			}
			proxyReq["model"] = modelID
			proxyReq["stream"] = false
			upstreamResp, upstreamErr = h.client.ChatCompletions(proxyReq)
		} else {
			h.logf("[V1/CHAT] routing to openai backend for %s", modelID)
			var proxyReq map[string]interface{}
			if err := json.Unmarshal(body, &proxyReq); err != nil {
				http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
				return
			}
			proxyReq["model"] = modelID
			h.fixDeepSeekMessages(proxyReq)
			upstreamResp, upstreamErr = h.client.ChatCompletions(proxyReq)
		}

		if upstreamErr != nil {
			h.logf("[V1/CHAT] upstream error: %v", upstreamErr)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": map[string]interface{}{
					"message": fmt.Sprintf("upstream request failed: %v", upstreamErr),
					"type":    "upstream_error",
				},
			})
			return
		}
		defer upstreamResp.Body.Close()

		if upstreamResp.StatusCode >= 400 {
			body, _ := io.ReadAll(io.LimitReader(upstreamResp.Body, 16384))
			h.logf("[V1/CHAT] upstream error status %d: %s", upstreamResp.StatusCode, string(body))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(upstreamResp.StatusCode)
			w.Write(body)
			return
		}

		if adapter.IsMiniMaxModel(modelID) {
			rawBody, repairedBody, err := repairTaggedV1Body(upstreamResp.Body)
			if err != nil {
				h.logf("[V1/CHAT] minimax repair decode error: %v", err)
				http.Error(w, `{"error":"failed to decode upstream response"}`, http.StatusBadGateway)
				return
			}

			h.logBody("[V1/CHAT] upstream raw body:", rawBody)
			h.logBody("[V1/CHAT] fixed body:", repairedBody)

			if req.Stream {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.Header().Set("Connection", "keep-alive")
				w.WriteHeader(http.StatusOK)
				if err := writeOpenAIStreamFromResponse(w, repairedBody, modelID); err != nil {
					h.logf("[V1/CHAT] minimax stream write error: %v", err)
				}
				return
			}

			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(repairedBody)
			return
		}

		copyHeader := upstreamResp.Header.Get("Content-Type")
		if copyHeader != "" {
			w.Header().Set("Content-Type", copyHeader)
		} else {
			w.Header().Set("Content-Type", "application/json")
		}

		if req.Stream {
			flusher, ok := w.(http.Flusher)
			if !ok {
				h.logf("[V1/CHAT] streaming not supported by http.ResponseWriter")
				http.Error(w, `{"error":"streaming not supported"}`, http.StatusInternalServerError)
				return
			}
			// Buffer + rewrite stream to make tag/tool-call normalization robust across split chunks.
			rawSSE, err := io.ReadAll(upstreamResp.Body)
			if err != nil {
				h.logf("[V1/CHAT] stream read error: %v", err)
				http.Error(w, `{"error":"failed to read upstream stream"}`, http.StatusBadGateway)
				return
			}

			h.logBody("[V1/CHAT] upstream raw stream body:", rawSSE)
			w.WriteHeader(http.StatusOK)

			repairedBody, err := rewriteBufferedOpenAISSE(bytes.NewReader(rawSSE), w, flusher, modelID)
			if err != nil {
				h.logf("[V1/CHAT] stream sanitize error: %v", err)
				return
			}

			h.logBody("[V1/CHAT] fixed stream body:", repairedBody)
			return
		}

		rawBody, repairedBody, err := repairTaggedV1Body(upstreamResp.Body)
		if err != nil {
			h.logf("[V1/CHAT] repair decode error: %v", err)
			http.Error(w, `{"error":"failed to decode upstream response"}`, http.StatusBadGateway)
			return
		}

		h.logBody("[V1/CHAT] upstream raw body:", rawBody)
		h.logBody("[V1/CHAT] fixed body:", repairedBody)

		w.WriteHeader(upstreamResp.StatusCode)
		w.Write(repairedBody)
	}
}

// handleV1Anthropic routes a /v1/chat/completions request through the Anthropic
// messages API and converts the response back to OpenAI format.
func (h *Handler) handleV1Anthropic(w http.ResponseWriter, rawBody []byte, modelID string, stream bool) {
	// Parse the incoming OpenAI-format request.
	var openAIReq adapter.OpenAIRequest
	if err := json.Unmarshal(rawBody, &openAIReq); err != nil {
		h.logf("[V1/ANTHROPIC] parse error: %v", err)
		http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
		return
	}
	openAIReq.Model = modelID

	// Convert OpenAI messages → OllamaChatRequest → AnthropicRequest.
	ollamaReq := adapter.OpenAIRequestToOllama(&openAIReq)
	anthReq := adapter.ChatRequestToAnthropic(ollamaReq)
	anthReq.Stream = stream

	resp, err := h.client.Messages(anthReq)
	if err != nil {
		h.logf("[V1/ANTHROPIC] upstream error: %v", err)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{
				"message": fmt.Sprintf("upstream request failed: %v", err),
				"type":    "upstream_error",
			},
		})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 16384))
		h.logf("[V1/ANTHROPIC] upstream error status %d: %s", resp.StatusCode, string(body))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(body)
		return
	}

	rawResp, err := io.ReadAll(resp.Body)
	if err != nil {
		h.logf("[V1/ANTHROPIC] read error: %v", err)
		http.Error(w, `{"error":"failed to read upstream response"}`, http.StatusBadGateway)
		return
	}

	h.logBody("[V1/ANTHROPIC] upstream raw body:", rawResp)

	if stream {
		// Anthropic SSE → OpenAI SSE
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		if err := rewriteAnthropicSSEToOpenAI(bytes.NewReader(rawResp), w, flusher, modelID); err != nil {
			h.logf("[V1/ANTHROPIC] stream rewrite error: %v", err)
		}
		return
	}

	// Non-stream: Anthropic JSON → OpenAI JSON
	openAIBody, err := anthropicBodyToOpenAIJSON(rawResp, modelID)
	if err != nil {
		h.logf("[V1/ANTHROPIC] convert error: %v", err)
		http.Error(w, `{"error":"failed to convert upstream response"}`, http.StatusBadGateway)
		return
	}

	h.logBody("[V1/ANTHROPIC] converted body:", openAIBody)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(openAIBody)
}

// anthropicBodyToOpenAIJSON converts a non-stream Anthropic /messages response
// to an OpenAI /chat/completions response.
func anthropicBodyToOpenAIJSON(raw []byte, model string) ([]byte, error) {
	var anthResp struct {
		ID         string `json:"id"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type     string          `json:"type"`
			Text     string          `json:"text,omitempty"`
			ID       string          `json:"id,omitempty"`
			Name     string          `json:"name,omitempty"`
			Input    json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &anthResp); err != nil {
		return nil, fmt.Errorf("unmarshal anthropic response: %w", err)
	}

	var textParts []string
	toolCalls := make([]adapter.OpenAIToolCall, 0)
	for _, block := range anthResp.Content {
		switch block.Type {
		case "text":
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
		case "tool_use":
			argsStr := "{}"
			if len(block.Input) > 0 {
				argsStr = string(block.Input)
			}
			toolCalls = append(toolCalls, adapter.OpenAIToolCall{
				ID:   block.ID,
				Type: "function",
				Function: adapter.OpenAIToolFunction{
					Name:      block.Name,
					Arguments: argsStr,
				},
			})
		// "thinking" and "thinking" blocks are intentionally skipped
		}
	}

	content := strings.Join(textParts, "")
	finishReason := "stop"
	if len(toolCalls) > 0 || anthResp.StopReason == "tool_use" {
		finishReason = "tool_calls"
	}

	resp := adapter.OpenAIResponse{
		ID:      anthResp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []adapter.OpenAIChoice{
			{
				Index: 0,
				Message: &adapter.OpenAIMessageResp{
					Role:      "assistant",
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: &finishReason,
			},
		},
		Usage: &adapter.OpenAIUsage{
			PromptTokens:     anthResp.Usage.InputTokens,
			CompletionTokens: anthResp.Usage.OutputTokens,
			TotalTokens:      anthResp.Usage.InputTokens + anthResp.Usage.OutputTokens,
		},
	}
	return json.Marshal(resp)
}

// rewriteAnthropicSSEToOpenAI reads a buffered Anthropic SSE body and emits
// OpenAI-format SSE chunks.
func rewriteAnthropicSSEToOpenAI(reader io.Reader, writer io.Writer, flusher http.Flusher, model string) error {
	type toolState struct {
		id   string
		name string
		args strings.Builder
	}

	msgID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	tools := map[int]*toolState{}
	order := []int{}
	finishReason := "stop"
	headerSent := false

	flush := func(data map[string]interface{}) error {
		b, err := json.Marshal(data)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(writer, "data: %s\n\n", b); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
		return nil
	}

	mkChunk := func(delta map[string]interface{}, fr interface{}) map[string]interface{} {
		return map[string]interface{}{
			"id":      msgID,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []interface{}{map[string]interface{}{
				"index":         0,
				"delta":         delta,
				"finish_reason": fr,
			}},
		}
	}

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		var event struct {
			Type  string `json:"type"`
			Index *int   `json:"index"`
			Delta *struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
			ContentBlock *struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
		}
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "message_start":
			if !headerSent {
				if err := flush(mkChunk(map[string]interface{}{"role": "assistant", "content": ""}, nil)); err != nil {
					return err
				}
				headerSent = true
			}

		case "content_block_start":
			if event.ContentBlock == nil || event.Index == nil {
				continue
			}
			idx := *event.Index
			if event.ContentBlock.Type == "tool_use" {
				tools[idx] = &toolState{id: event.ContentBlock.ID, name: event.ContentBlock.Name}
				order = append(order, idx)
				finishReason = "tool_calls"
				tcIdx := len(order) - 1
				tc := map[string]interface{}{
					"index": tcIdx,
					"id":    event.ContentBlock.ID,
					"type":  "function",
					"function": map[string]interface{}{
						"name":      event.ContentBlock.Name,
						"arguments": "",
					},
				}
				if err := flush(mkChunk(map[string]interface{}{"tool_calls": []interface{}{tc}}, nil)); err != nil {
					return err
				}
			}
			// "thinking" content blocks are silently skipped

		case "content_block_delta":
			if event.Delta == nil || event.Index == nil {
				continue
			}
			idx := *event.Index
			switch event.Delta.Type {
			case "text_delta":
				if err := flush(mkChunk(map[string]interface{}{"content": event.Delta.Text}, nil)); err != nil {
					return err
				}
			case "input_json_delta":
				if tc, ok := tools[idx]; ok {
					tc.args.WriteString(event.Delta.PartialJSON)
					tcIdx := 0
					for i, o := range order {
						if o == idx {
							tcIdx = i
							break
						}
					}
					if err := flush(mkChunk(map[string]interface{}{
						"tool_calls": []interface{}{map[string]interface{}{
							"index":    tcIdx,
							"function": map[string]interface{}{"arguments": event.Delta.PartialJSON},
						}},
					}, nil)); err != nil {
						return err
					}
				}
			}

		case "message_delta":
			if event.Delta != nil && event.Delta.StopReason != "" {
				if event.Delta.StopReason == "tool_use" {
					finishReason = "tool_calls"
				} else {
					finishReason = event.Delta.StopReason
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// Final chunk with finish_reason.
	if err := flush(mkChunk(map[string]interface{}{}, finishReason)); err != nil {
		return err
	}
	_, err := fmt.Fprintf(writer, "data: [DONE]\n\n")
	if flusher != nil {
		flusher.Flush()
	}
	return err
}

func repairTaggedV1Body(r io.Reader) ([]byte, []byte, error) {
	body, err := io.ReadAll(r)
	if err != nil {
		return nil, nil, err
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return body, body, nil
	}

	choices, _ := payload["choices"].([]interface{})
	for _, ch := range choices {
		choice, ok := ch.(map[string]interface{})
		if !ok {
			continue
		}
		if msg, ok := choice["message"].(map[string]interface{}); ok {
			if content, ok := msg["content"].(string); ok {
				clean, toolCalls := parseTaggedAssistantContent(content)
				msg["content"] = clean
				if len(toolCalls) > 0 {
					msg["tool_calls"] = toolCalls
					if _, hasFinishReason := choice["finish_reason"]; !hasFinishReason {
						choice["finish_reason"] = "tool_calls"
					} else if fr, ok := choice["finish_reason"].(string); ok && (fr == "" || fr == "stop") {
						choice["finish_reason"] = "tool_calls"
					}
				}
			}
		}
		if delta, ok := choice["delta"].(map[string]interface{}); ok {
			if content, ok := delta["content"].(string); ok {
				clean, toolCalls := parseTaggedAssistantContent(content)
				delta["content"] = clean
				if len(toolCalls) > 0 {
					delta["tool_calls"] = toolCallsToDelta(toolCalls)
					if _, hasFinishReason := choice["finish_reason"]; !hasFinishReason {
						choice["finish_reason"] = "tool_calls"
					} else if fr, ok := choice["finish_reason"].(string); ok && (fr == "" || fr == "stop") {
						choice["finish_reason"] = "tool_calls"
					}
				}
			}
		}
	}

	repaired, err := json.Marshal(payload)
	if err != nil {
		return nil, nil, err
	}

	return body, repaired, nil
}

func sanitizeTaggedText(s string) string {
	s = miniMaxWrapperRE.ReplaceAllString(s, "")
	s = thinkBlockRE.ReplaceAllString(s, "")
	s = toolCallBlockRE.ReplaceAllString(s, "")
	s = toolCallsPluralBlockRE.ReplaceAllString(s, "")
	s = dsmlToolCallsBlockRE.ReplaceAllString(s, "")
	s = thinkTailRE.ReplaceAllString(s, "")
	s = toolCallTailRE.ReplaceAllString(s, "")
	s = toolCallsPluralTailRE.ReplaceAllString(s, "")
	s = dsmlToolCallsTailRE.ReplaceAllString(s, "")
	s = strings.TrimSpace(s)
	s = multiBlankRE.ReplaceAllString(s, "\n\n")
	return s
}

func parseTaggedAssistantContent(content string) (string, []map[string]interface{}) {
	clean := sanitizeTaggedText(content)
	toolCalls := extractToolCalls(content)
	return clean, toolCalls
}

// extractPluralToolCalls handles the <tool_calls>…</tool_calls> (plural) format
// used by hy3-preview and other models routed via OpenRouter.
//
// The inner content can be:
//   - A JSON array: [{"name":"fn","arguments":{…}}] or the full OpenAI shape
//   - Invoke-style sub-tags: <invoke name="fn">…</invoke>
//   - Empty (model attempted a call but emitted nothing) → returns nil
func extractPluralToolCalls(raw string) []map[string]interface{} {
	out := make([]map[string]interface{}, 0)
	callIdx := 0

	for _, blockMatch := range toolCallsPluralInnerRE.FindAllStringSubmatch(raw, -1) {
		if len(blockMatch) < 2 {
			continue
		}
		inner := strings.TrimSpace(blockMatch[1])
		if inner == "" {
			continue
		}

		// Try JSON array first.
		dec := json.NewDecoder(strings.NewReader(inner))
		var arr []interface{}
		if err := dec.Decode(&arr); err == nil {
			for _, item := range arr {
				obj, ok := item.(map[string]interface{})
				if !ok {
					continue
				}
				name := ""
				if n, ok := obj["name"].(string); ok {
					name = n
				} else if fn, ok := obj["function"].(map[string]interface{}); ok {
					name, _ = fn["name"].(string)
				}
				if name == "" {
					continue
				}
				argsJSON := "{}"
				if a, ok := obj["arguments"]; ok {
					switch v := a.(type) {
					case string:
						if json.Valid([]byte(v)) {
							argsJSON = v
						} else if b, err := json.Marshal(v); err == nil {
							argsJSON = string(b)
						}
					default:
						if b, err := json.Marshal(v); err == nil {
							argsJSON = string(b)
						}
					}
				} else if p, ok := obj["parameters"]; ok {
					if b, err := json.Marshal(p); err == nil {
						argsJSON = string(b)
					}
				}
				id := fmt.Sprintf("call_%d", callIdx)
				if rawID, ok := obj["id"].(string); ok && rawID != "" {
					id = rawID
				}
				out = append(out, map[string]interface{}{
					"id":   id,
					"type": "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": argsJSON,
					},
				})
				callIdx++
			}
			continue
		}

		// Fall back to invoke-style sub-tags (same as minimax handler).
		for _, im := range invokeTagRE.FindAllStringSubmatch(inner, -1) {
			if len(im) < 3 {
				continue
			}
			name := strings.TrimSpace(im[1])
			if !bareCallNameRE.MatchString(strings.ToLower(name)) {
				continue
			}
			params := map[string]string{}
			if cm := commandTagRE.FindStringSubmatch(im[2]); len(cm) >= 2 {
				params["command"] = strings.TrimSpace(cm[1])
			}
			for _, pm := range parameterTagRE.FindAllStringSubmatch(im[2], -1) {
				if len(pm) < 3 {
					continue
				}
				if k := strings.TrimSpace(pm[1]); k != "" {
					params[k] = strings.TrimSpace(pm[2])
				}
			}
			argsJSON := "{}"
			if b, err := json.Marshal(params); err == nil {
				argsJSON = string(b)
			}
			out = append(out, map[string]interface{}{
				"id":   fmt.Sprintf("call_%d", callIdx),
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": argsJSON,
				},
			})
			callIdx++
		}
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

// extractDSMLToolCalls handles DeepSeek's DSML markup format:
//
//	<｜｜DSML｜｜tool_calls>
//	<｜｜DSML｜｜invoke name="run_in_terminal">
//	<｜｜DSML｜｜parameter name="command" string="true">cd /tmp && ls</｜｜DSML｜｜parameter>
//	</｜｜DSML｜｜invoke>
//	</｜｜DSML｜｜tool_calls>
func extractDSMLToolCalls(raw string) []map[string]interface{} {
	out := make([]map[string]interface{}, 0)
	callIdx := 0

	for _, blockMatch := range dsmlToolCallsBlockRE.FindAllStringSubmatch(raw, -1) {
		if len(blockMatch) < 2 {
			continue
		}
		blockInner := blockMatch[1]

		for _, im := range dsmlInvokeRE.FindAllStringSubmatch(blockInner, -1) {
			if len(im) < 3 {
				continue
			}
			name := strings.TrimSpace(im[1])
			if name == "" {
				continue
			}

			params := map[string]string{}
			for _, pm := range dsmlParamRE.FindAllStringSubmatch(im[2], -1) {
				if len(pm) < 3 {
					continue
				}
				k := strings.TrimSpace(pm[1])
				v := strings.TrimSpace(pm[2])
				if k == "" {
					continue
				}
				params[k] = v
			}

			argsJSON := "{}"
			if b, err := json.Marshal(params); err == nil {
				argsJSON = string(b)
			}

			out = append(out, map[string]interface{}{
				"id":   fmt.Sprintf("call_%d", callIdx),
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": argsJSON,
				},
			})
			callIdx++
		}
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func extractToolCalls(raw string) []map[string]interface{} {
	// DeepSeek DSML format: <｜｜DSML｜｜tool_calls>...<｜｜DSML｜｜invoke name="fn">...<｜｜DSML｜｜invoke>...
	if dsmlMatches := dsmlToolCallsBlockRE.FindAllStringSubmatch(raw, -1); len(dsmlMatches) > 0 {
		return extractDSMLToolCalls(raw)
	}

	// <tool_calls>…</tool_calls> (plural) — hy3-preview / OpenRouter models.
	if toolCallsPluralInnerRE.MatchString(raw) {
		if tcs := extractPluralToolCalls(raw); len(tcs) > 0 {
			return tcs
		}
		// Block was present but empty — return nil so caller strips it cleanly.
		return nil
	}

	matches := toolCallInnerRE.FindAllStringSubmatch(raw, -1)
	if len(matches) == 0 {
		return nil
	}

	out := make([]map[string]interface{}, 0)
	callIdx := 0

	for _, m := range matches {
		if len(m) < 2 {
			continue
		}
		inner := strings.TrimSpace(miniMaxWrapperRE.ReplaceAllString(m[1], ""))
		if inner == "" {
			continue
		}

		dec := json.NewDecoder(strings.NewReader(inner))
		parsedAny := false
		for {
			var obj map[string]interface{}
			if err := dec.Decode(&obj); err != nil {
				break
			}
			parsedAny = true

			name := ""
			if n, ok := obj["name"].(string); ok {
				name = n
			} else if fn, ok := obj["function"].(map[string]interface{}); ok {
				name, _ = fn["name"].(string)
			}
			if name == "" {
				continue
			}

			argsJSON := "{}"
			if p, ok := obj["parameters"]; ok {
				if b, err := json.Marshal(p); err == nil {
					argsJSON = string(b)
				}
			} else if a, ok := obj["arguments"]; ok {
				switch v := a.(type) {
				case string:
					if json.Valid([]byte(v)) {
						argsJSON = v
					} else {
						if b, err := json.Marshal(v); err == nil {
							argsJSON = string(b)
						}
					}
				default:
					if b, err := json.Marshal(v); err == nil {
						argsJSON = string(b)
					}
				}
			}

			id := ""
			if rawID, ok := obj["id"].(string); ok && rawID != "" {
				id = rawID
			} else {
				id = fmt.Sprintf("call_%d", callIdx)
			}

			out = append(out, map[string]interface{}{
				"id":   id,
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": argsJSON,
				},
			})
			callIdx++
		}

		if parsedAny {
			continue
		}

		for _, im := range invokeTagRE.FindAllStringSubmatch(inner, -1) {
			if len(im) < 3 {
				continue
			}
			name := strings.TrimSpace(im[1])
			if !bareCallNameRE.MatchString(strings.ToLower(name)) {
				continue
			}

			params := map[string]string{}
			if cm := commandTagRE.FindStringSubmatch(im[2]); len(cm) >= 2 {
				params["command"] = strings.TrimSpace(cm[1])
			}
			for _, pm := range parameterTagRE.FindAllStringSubmatch(im[2], -1) {
				if len(pm) < 3 {
					continue
				}
				k := strings.TrimSpace(pm[1])
				v := strings.TrimSpace(pm[2])
				if k == "" {
					continue
				}
				params[k] = v
			}

			argsJSON := "{}"
			if b, err := json.Marshal(params); err == nil {
				argsJSON = string(b)
			}

			out = append(out, map[string]interface{}{
				"id":   fmt.Sprintf("call_%d", callIdx),
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": argsJSON,
				},
			})
			callIdx++
		}

		for _, b := range bracketCallRE.FindAllStringSubmatch(inner, -1) {
			if len(b) < 3 {
				continue
			}
			name := strings.TrimSpace(b[1])
			argsJSON := strings.TrimSpace(b[2])
			if !bareCallNameRE.MatchString(strings.ToLower(name)) {
				continue
			}
			if !json.Valid([]byte(argsJSON)) {
				continue
			}
			out = append(out, map[string]interface{}{
				"id":   fmt.Sprintf("call_%d", callIdx),
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": argsJSON,
				},
			})
			callIdx++
		}

		for _, fm := range functionTagRE.FindAllStringSubmatch(inner, -1) {
			if len(fm) < 3 {
				continue
			}
			name := strings.TrimSpace(fm[1])
			if !bareCallNameRE.MatchString(strings.ToLower(name)) {
				continue
			}
			params := map[string]string{}
			for _, pm := range parameterTagRE.FindAllStringSubmatch(fm[2], -1) {
				if len(pm) < 3 {
					continue
				}
				k := strings.TrimSpace(pm[1])
				v := strings.TrimSpace(pm[2])
				if k == "" {
					continue
				}
				params[k] = v
			}
			argsJSON := "{}"
			if b, err := json.Marshal(params); err == nil {
				argsJSON = string(b)
			}
			out = append(out, map[string]interface{}{
				"id":   fmt.Sprintf("call_%d", callIdx),
				"type": "function",
				"function": map[string]interface{}{
					"name":      name,
					"arguments": argsJSON,
				},
			})
			callIdx++
		}

		// Last fallback: plain tool names inside the block (no args available).
		// Restrict to function-like names (must contain underscore) to avoid false calls like cd/git/log.
		for _, tok := range strings.Fields(inner) {
			t := strings.TrimSpace(strings.Trim(tok, "[](),"))
			if !bareCallNameRE.MatchString(strings.ToLower(t)) {
				continue
			}
			if !strings.Contains(t, "_") {
				continue
			}
			if t == "name" || t == "parameters" || t == "arguments" || t == "https" || t == "http" {
				continue
			}
			out = append(out, map[string]interface{}{
				"id":   fmt.Sprintf("call_%d", callIdx),
				"type": "function",
				"function": map[string]interface{}{
					"name":      t,
					"arguments": "{}",
				},
			})
			callIdx++
		}
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func toolCallsToDelta(toolCalls []map[string]interface{}) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(toolCalls))
	for i, tc := range toolCalls {
		entry := map[string]interface{}{
			"index": i,
		}
		if id, ok := tc["id"].(string); ok {
			entry["id"] = id
		}
		if typ, ok := tc["type"].(string); ok {
			entry["type"] = typ
		}
		if fn, ok := tc["function"].(map[string]interface{}); ok {
			entry["function"] = fn
		}
		out = append(out, entry)
	}
	return out
}

func proxySanitizedOpenAISSE(reader io.Reader, writer io.Writer, flusher http.Flusher) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data != "[DONE]" {
				var payload map[string]interface{}
				if err := json.Unmarshal([]byte(data), &payload); err == nil {
					choices, _ := payload["choices"].([]interface{})
					for _, ch := range choices {
						choice, ok := ch.(map[string]interface{})
						if !ok {
							continue
						}
						if msg, ok := choice["message"].(map[string]interface{}); ok {
							if content, ok := msg["content"].(string); ok {
								clean, toolCalls := parseTaggedAssistantContent(content)
								msg["content"] = clean
								if len(toolCalls) > 0 {
									msg["tool_calls"] = toolCalls
									if fr, ok := choice["finish_reason"].(string); !ok || fr == "" || fr == "stop" {
										choice["finish_reason"] = "tool_calls"
									}
								}
							}
						}
						if delta, ok := choice["delta"].(map[string]interface{}); ok {
							if content, ok := delta["content"].(string); ok {
								clean, toolCalls := parseTaggedAssistantContent(content)
								delta["content"] = clean
								if len(toolCalls) > 0 {
									delta["tool_calls"] = toolCallsToDelta(toolCalls)
									if fr, ok := choice["finish_reason"].(string); !ok || fr == "" || fr == "stop" {
										choice["finish_reason"] = "tool_calls"
									}
								}
							}
						}
					}

					if normalized, err := json.Marshal(payload); err == nil {
						line = "data: " + string(normalized)
					}
				}
			}
		}

		if _, err := io.WriteString(writer, line+"\n"); err != nil {
			return err
		}
		flusher.Flush()
	}

	return scanner.Err()
}

func rewriteBufferedOpenAISSE(reader io.Reader, writer io.Writer, flusher http.Flusher, fallbackModel string) ([]byte, error) {
	type toolCallState struct {
		id      string
		typ     string
		name    string
		argsBuf strings.Builder
	}

	type choiceState struct {
		index        int
		content      strings.Builder
		toolCalls    map[int]*toolCallState
		finishReason string
	}

	states := map[int]*choiceState{}
	order := make([]int, 0)

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	respID := ""
	model := fallbackModel
	var created int64

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var payload map[string]interface{}
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			continue
		}

		if respID == "" {
			if id, ok := payload["id"].(string); ok {
				respID = id
			}
		}
		if model == "" {
			if m, ok := payload["model"].(string); ok {
				model = m
			}
		}
		if created == 0 {
			switch v := payload["created"].(type) {
			case float64:
				created = int64(v)
			case int64:
				created = v
			}
		}

		choices, _ := payload["choices"].([]interface{})
		for _, c := range choices {
			choice, ok := c.(map[string]interface{})
			if !ok {
				continue
			}

			idx := 0
			switch v := choice["index"].(type) {
			case float64:
				idx = int(v)
			case int:
				idx = v
			}

			st, ok := states[idx]
			if !ok {
				st = &choiceState{index: idx, toolCalls: make(map[int]*toolCallState)}
				states[idx] = st
				order = append(order, idx)
			}

			if fr, ok := choice["finish_reason"].(string); ok && fr != "" {
				st.finishReason = fr
			}

			if delta, ok := choice["delta"].(map[string]interface{}); ok {
				if content, ok := delta["content"].(string); ok {
					st.content.WriteString(content)
				}
				if dToolCalls, ok := delta["tool_calls"].([]interface{}); ok {
					for _, dtc := range dToolCalls {
						tcm, ok := dtc.(map[string]interface{})
						if !ok {
							continue
						}

						tcIndex := 0
						switch v := tcm["index"].(type) {
						case float64:
							tcIndex = int(v)
						case int:
							tcIndex = v
						}

						tcs, ok := st.toolCalls[tcIndex]
						if !ok {
							tcs = &toolCallState{}
							st.toolCalls[tcIndex] = tcs
						}

						if id, ok := tcm["id"].(string); ok && id != "" {
							tcs.id = id
						}
						if typ, ok := tcm["type"].(string); ok && typ != "" {
							tcs.typ = typ
						}
						if fn, ok := tcm["function"].(map[string]interface{}); ok {
							if name, ok := fn["name"].(string); ok && name != "" {
								tcs.name = name
							}
							if args, ok := fn["arguments"].(string); ok {
								tcs.argsBuf.WriteString(args)
							}
						}
					}
				}
			}
			if msg, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := msg["content"].(string); ok {
					st.content.WriteString(content)
				}
				if mToolCalls, ok := msg["tool_calls"].([]interface{}); ok {
					for tcIdx, mtc := range mToolCalls {
						tcm, ok := mtc.(map[string]interface{})
						if !ok {
							continue
						}
						tcs, ok := st.toolCalls[tcIdx]
						if !ok {
							tcs = &toolCallState{}
							st.toolCalls[tcIdx] = tcs
						}
						if id, ok := tcm["id"].(string); ok && id != "" {
							tcs.id = id
						}
						if typ, ok := tcm["type"].(string); ok && typ != "" {
							tcs.typ = typ
						}
						if fn, ok := tcm["function"].(map[string]interface{}); ok {
							if name, ok := fn["name"].(string); ok && name != "" {
								tcs.name = name
							}
							if args, ok := fn["arguments"].(string); ok {
								tcs.argsBuf.WriteString(args)
							}
						}
					}
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if respID == "" {
		respID = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	if created == 0 {
		created = time.Now().Unix()
	}

	resp := adapter.OpenAIResponse{
		ID:      respID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: make([]adapter.OpenAIChoice, 0, len(order)),
	}

	for _, idx := range order {
		st := states[idx]
		clean, tcs := parseTaggedAssistantContent(st.content.String())

		if len(st.toolCalls) > 0 {
			fromUpstream := make([]map[string]interface{}, 0, len(st.toolCalls))
			for i := 0; i < len(st.toolCalls); i++ {
				tc, ok := st.toolCalls[i]
				if !ok {
					continue
				}
				if tc.name == "" {
					continue
				}
				typ := tc.typ
				if typ == "" {
					typ = "function"
				}
				id := tc.id
				if id == "" {
					id = fmt.Sprintf("call_%d", i)
				}
				args := tc.argsBuf.String()
				if args == "" {
					args = "{}"
				}
				if !json.Valid([]byte(args)) {
					args = "{}"
				}
				fromUpstream = append(fromUpstream, map[string]interface{}{
					"id":   id,
					"type": typ,
					"function": map[string]interface{}{
						"name":      tc.name,
						"arguments": args,
					},
				})
			}
			if len(fromUpstream) > 0 {
				tcs = fromUpstream
			}
		}

		choice := adapter.OpenAIChoice{Index: idx}
		choice.Message = &adapter.OpenAIMessageResp{Role: "assistant", Content: clean}
		if len(tcs) > 0 {
			converted := make([]adapter.OpenAIToolCall, 0, len(tcs))
			for _, tc := range tcs {
				fn, _ := tc["function"].(map[string]interface{})
				name, _ := fn["name"].(string)
				args, _ := fn["arguments"].(string)
				id, _ := tc["id"].(string)
				typ, _ := tc["type"].(string)
				if typ == "" {
					typ = "function"
				}
				converted = append(converted, adapter.OpenAIToolCall{
					ID:   id,
					Type: typ,
					Function: adapter.OpenAIToolFunction{
						Name:      name,
						Arguments: args,
					},
				})
			}
			choice.Message.ToolCalls = converted
		}

		fr := st.finishReason
		if len(choice.Message.ToolCalls) > 0 {
			if fr == "" || fr == "stop" {
				fr = "tool_calls"
			}
		} else if fr == "" {
			fr = "stop"
		}
		choice.FinishReason = &fr
		resp.Choices = append(resp.Choices, choice)
	}

	body, err := json.Marshal(resp)
	if err != nil {
		return nil, err
	}
	if err := writeOpenAIStreamFromResponse(writer, body, fallbackModel); err != nil {
		return nil, err
	}
	flusher.Flush()
	return body, nil
}

func writeOpenAIStreamFromResponse(w io.Writer, repairedBody []byte, fallbackModel string) error {
	var resp adapter.OpenAIResponse
	if err := json.Unmarshal(repairedBody, &resp); err != nil {
		return err
	}

	id := resp.ID
	if id == "" {
		id = fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	}
	created := resp.Created
	if created == 0 {
		created = time.Now().Unix()
	}
	model := resp.Model
	if model == "" {
		model = fallbackModel
	}

	for i, choice := range resp.Choices {
		content := ""
		var toolCalls []adapter.OpenAIToolCall
		if choice.Message != nil {
			content = choice.Message.Content
			toolCalls = choice.Message.ToolCalls
		}
		if content == "" && len(toolCalls) == 0 {
			continue
		}

		if content != "" {
			chunk := map[string]interface{}{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]interface{}{
					{
						"index": i,
						"delta": map[string]interface{}{
							"role":    "assistant",
							"content": content,
						},
						"finish_reason": nil,
					},
				},
			}

			data, err := json.Marshal(chunk)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
				return err
			}
		}

		if len(toolCalls) > 0 {
			deltaToolCalls := make([]map[string]interface{}, 0, len(toolCalls))
			for idx, tc := range toolCalls {
				deltaToolCalls = append(deltaToolCalls, map[string]interface{}{
					"index": idx,
					"id":    tc.ID,
					"type":  tc.Type,
					"function": map[string]interface{}{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments,
					},
				})
			}

			chunk := map[string]interface{}{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]interface{}{
					{
						"index": i,
						"delta": map[string]interface{}{
							"role":       "assistant",
							"tool_calls": deltaToolCalls,
						},
						"finish_reason": nil,
					},
				},
			}

			data, err := json.Marshal(chunk)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
				return err
			}
		}
	}

	finishReason := "stop"
	if len(resp.Choices) > 0 {
		if resp.Choices[0].FinishReason != nil && *resp.Choices[0].FinishReason != "" {
			finishReason = *resp.Choices[0].FinishReason
		} else if resp.Choices[0].Message != nil && len(resp.Choices[0].Message.ToolCalls) > 0 {
			finishReason = "tool_calls"
		}
	}

	finalChunk := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"delta":         map[string]interface{}{},
				"finish_reason": finishReason,
			},
		},
	}
	data, err := json.Marshal(finalChunk)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
		return err
	}

	_, err = io.WriteString(w, "data: [DONE]\n\n")
	return err
}

func (h *Handler) convertOAToolsToAnthropic(toolsRaw json.RawMessage, out *[]adapter.AnthropicTool) {
	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal(toolsRaw, &tools); err != nil {
		return
	}
	for _, t := range tools {
		*out = append(*out, adapter.AnthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}
}

func (h *Handler) V1Models() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		models, err := h.client.ListModels()
		if err != nil {
			h.logf("[V1/MODELS] failed: %v", err)
			http.Error(w, `{"error":"failed to list models"}`, http.StatusInternalServerError)
			return
		}

		data := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			data = append(data, map[string]interface{}{
				"id":             m.ID,
				"object":         "model",
				"created":        0,
				"owned_by":       "opencode-go",
				"context_length": modelContextLength(m.ID),
				"max_tokens":     modelContextLength(m.ID),
			})
		}

		h.logf("[V1/MODELS] returning %d models", len(data))

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data":   data,
		})
	}
}

func injectEmptyReasoningContent(req map[string]interface{}) {
	msgs, ok := req["messages"].([]interface{})
	if !ok {
		return
	}
	for _, m := range msgs {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); role == "assistant" {
			if _, exists := msg["reasoning_content"]; !exists {
				msg["reasoning_content"] = ""
			}
		}
	}
}

func (h *Handler) fixDeepSeekMessages(req map[string]interface{}) {
	msgs, ok := req["messages"].([]interface{})
	if !ok {
		return
	}

	h.logf("[FIX] scanning %d messages", len(msgs))
	for i, m := range msgs {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		_, hasTC := msg["tool_calls"]
		h.logf("[FIX]   [%d] role=%s has_tool_calls=%v", i, role, hasTC)
	}

	fixed := make([]interface{}, 0, len(msgs))

	for i, m := range msgs {
		msg, ok := m.(map[string]interface{})
		if !ok {
			fixed = append(fixed, m)
			continue
		}

		role, _ := msg["role"].(string)

		if role == "assistant" {
			if _, exists := msg["reasoning_content"]; !exists {
				msg["reasoning_content"] = ""
			}
		}

		if role == "assistant" && msg["tool_calls"] != nil {
			hasToolResponses := false
			for j := i + 1; j < len(msgs); j++ {
				next, ok := msgs[j].(map[string]interface{})
				if !ok {
					continue
				}
				nextRole, _ := next["role"].(string)
				if nextRole == "tool" {
					hasToolResponses = true
				}
				if nextRole != "tool" {
					break
				}
			}
			if !hasToolResponses {
				h.logf("[FIX] removing orphaned tool_calls from assistant[%d]", i)
				delete(msg, "tool_calls")
			} else {
				h.logf("[FIX] keeping tool_calls on assistant[%d] (followed by tool messages)", i)
			}
		}

		if role == "tool" {
			toolCallID, _ := msg["tool_call_id"].(string)
			hasPreceding := false
			for k := i - 1; k >= 0; k-- {
				prev, ok := msgs[k].(map[string]interface{})
				if !ok {
					continue
				}
				prevRole, _ := prev["role"].(string)
				if prevRole == "tool" {
					continue
				}
				if prevRole == "assistant" {
					toolCalls, _ := prev["tool_calls"].([]interface{})
					for _, tc := range toolCalls {
						tcm, _ := tc.(map[string]interface{})
						if tid, _ := tcm["id"].(string); tid == toolCallID {
							hasPreceding = true
							break
						}
					}
				}
				break
			}
			if !hasPreceding {
				h.logf("[FIX] dropping orphaned tool message[%d]", i)
				continue
			}
		}

		fixed = append(fixed, m)
	}

	req["messages"] = fixed

	h.logf("[FIX] result: %d messages", len(fixed))
	for i, m := range fixed {
		if msg, ok := m.(map[string]interface{}); ok {
			role, _ := msg["role"].(string)
			_, hasTC := msg["tool_calls"]
			h.logf("[FIX]   [%d] role=%s has_tool_calls=%v", i, role, hasTC)
		}
	}
}

func (h *Handler) Chat() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		var ollamaReq adapter.OllamaChatRequest
		if err := json.Unmarshal(body, &ollamaReq); err != nil {
			h.logf("[CHAT] parse error: %v | body: %s", err, string(body))
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		h.logf("[CHAT] model=%s stream=%v messages=%d tools=%d",
			ollamaReq.Model, boolPtr(ollamaReq.Stream), len(ollamaReq.Messages), len(ollamaReq.Tools))

		if adapter.IsMiniMaxModel(ollamaReq.Model) {
			h.logf("[CHAT] routing to anthropic backend for %s", ollamaReq.Model)
			h.handleChatAnthropic(w, &ollamaReq)
			return
		}

		if adapter.IsAnthropicOnlyModel(ollamaReq.Model) {
			h.logf("[CHAT] routing to anthropic backend (anthropic-only model) for %s", ollamaReq.Model)
			h.handleChatAnthropic(w, &ollamaReq)
			return
		}

		h.logf("[CHAT] routing to openai backend for %s", ollamaReq.Model)
		h.handleChatOpenAI(w, &ollamaReq)
	}
}

func (h *Handler) handleChatOpenAI(w http.ResponseWriter, ollamaReq *adapter.OllamaChatRequest) {
	openaiReq := adapter.ChatRequestToOpenAI(ollamaReq)

	resp, err := h.client.ChatCompletions(openaiReq)
	if err != nil {
		h.logf("[CHAT] upstream error (openai): %v", err)
		http.Error(w, `{"error":"upstream request failed"}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		h.logf("[CHAT] upstream error status %d: %s", resp.StatusCode, string(body))
		http.Error(w, string(body), resp.StatusCode)
		return
	}

	if openaiReq.Stream {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		if err := adapter.OpenAIStreamToOllamaNDJSON(resp.Body, ollamaReq.Model, w); err != nil {
			h.logf("[CHAT] stream error: %v", err)
			return
		}
		flusher.Flush()
		return
	}

	var openaiResp adapter.OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		h.logf("[CHAT] decode error: %v", err)
		http.Error(w, `{"error":"failed to decode response"}`, http.StatusInternalServerError)
		return
	}

	ollamaResp := adapter.OpenAIResponseToOllamaChat(&openaiResp, ollamaReq.Model)
	h.logf("[CHAT] response: done=%v content_len=%d", ollamaResp.Done, len(ollamaResp.Message.Content))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaResp)
}

func (h *Handler) handleChatAnthropic(w http.ResponseWriter, ollamaReq *adapter.OllamaChatRequest) {
	anthReq := adapter.ChatRequestToAnthropic(ollamaReq)

	var resp *http.Response
	var err error
	resp, err = h.client.Messages(anthReq)
	if err != nil {
		h.logf("[CHAT] upstream error (anthropic): %v", err)
		http.Error(w, `{"error":"upstream request failed"}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		h.logf("[CHAT] upstream error status %d: %s", resp.StatusCode, string(body))
		http.Error(w, string(body), resp.StatusCode)
		return
	}

	if anthReq.Stream {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		if err := adapter.AnthropicStreamToOllamaNDJSON(resp.Body, ollamaReq.Model, w); err != nil {
			h.logf("[CHAT] stream error: %v", err)
			return
		}
		flusher.Flush()
		return
	}

	http.Error(w, `{"error":"non-streaming anthropic not yet supported"}`, http.StatusNotImplemented)
}

func (h *Handler) Generate() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		var genReq adapter.OllamaGenerateRequest
		if err := json.Unmarshal(body, &genReq); err != nil {
			h.logf("[GENERATE] parse error: %v | body: %s", err, string(body))
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		h.logf("[GENERATE] model=%s stream=%v prompt_len=%d",
			genReq.Model, boolPtr(genReq.Stream), len(genReq.Prompt))

		openaiReq := adapter.GenerateRequestToOpenAI(&genReq)

		resp, err := h.client.ChatCompletions(openaiReq)
		if err != nil {
			h.logf("[GENERATE] upstream error: %v", err)
			http.Error(w, `{"error":"upstream request failed"}`, http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			h.logf("[GENERATE] upstream error status %d: %s", resp.StatusCode, string(body))
			http.Error(w, string(body), resp.StatusCode)
			return
		}

		if openaiReq.Stream {
			w.Header().Set("Content-Type", "application/x-ndjson")
			w.WriteHeader(http.StatusOK)
			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "streaming not supported", http.StatusInternalServerError)
				return
			}

			if err := adapter.OpenAIStreamToOllamaNDJSON(resp.Body, genReq.Model, w); err != nil {
				h.logf("[GENERATE] stream error: %v", err)
				return
			}
			flusher.Flush()
			return
		}

		var openaiResp adapter.OpenAIResponse
		if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
			h.logf("[GENERATE] decode error: %v", err)
			http.Error(w, `{"error":"failed to decode response"}`, http.StatusInternalServerError)
			return
		}

		ollamaResp := adapter.OpenAIResponseToOllamaChat(&openaiResp, genReq.Model)

		if len(ollamaResp.DoneReason) == 0 && ollamaResp.Done {
			ollamaResp.DoneReason = "stop"
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ollamaResp)
	}
}

func boolPtr(b *bool) bool {
	if b == nil {
		return false
	}
	return *b
}
