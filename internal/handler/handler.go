package handler

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/mheers/opencode-go-ollama-bridge/internal/adapter"
	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
	"github.com/mheers/opencode-go-ollama-bridge/internal/redact"
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
	// <parameter=name>value</parameter> — function-tag style
	parameterTagRE = regexp.MustCompile(`(?is)<parameter=([a-zA-Z_][a-zA-Z0-9_-]*)>(.*?)</parameter>`)

	// <parameter name="name">value</parameter> — invoke-tag style (MiniMax, Claude-style)
	// Also handles spaces around = and single-quoted attributes.
	namedParamTagRE = regexp.MustCompile(`(?is)<parameter\s+name\s*=\s*["']([a-zA-Z_][a-zA-Z0-9_-]*)["']>(.*?)</parameter>`)
	// <filePath>...</filePath>, <newString>...</newString>, etc. inside <invoke ...>...</invoke>
	invokeValueTagRE = regexp.MustCompile(`(?is)<([a-zA-Z_][a-zA-Z0-9_-]{1,63})>(.*?)</([a-zA-Z_][a-zA-Z0-9_-]{1,63})>`)
	// <invoke name="funcname"> — standard form
	// <invoke name>funcname"> — MiniMax malformed form (= and opening " dropped,
	//   > closes the tag mid-attribute, funcname" becomes "text" before second >)
	// Both are handled by treating [=>] as the separator between "name" and the value.
	invokeTagRE = regexp.MustCompile(`(?is)<invoke\s+name\s*[=>]\s*"?([a-zA-Z_][a-zA-Z0-9_-]*)"?>(.*?)</invoke>`)
	// <invoke name "filePath": "/tmp/file.go"]> — MiniMax malformed form where
	// the function name is lost and a single argument key/value is emitted inside
	// the opening tag itself. The trailing ] before > is present in observed logs.
	malformedInvokeParamRE = regexp.MustCompile(`(?is)<invoke\s+name\s+"?([a-zA-Z_][a-zA-Z0-9_-]*)"?\s*:\s*"([^"]+)"(?:\]?>)?(.*?)</invoke>`)
	commandTagRE           = regexp.MustCompile(`(?is)<command>(.*?)</command>`)
	thinkTailRE            = regexp.MustCompile(`(?is)<think>.*$`)
	toolCallTailRE         = regexp.MustCompile(`(?is)<tool_call\b[^>]*>.*$`)
	bracketCallRE          = regexp.MustCompile(`\[\s*([a-zA-Z_][a-zA-Z0-9_-]*)\s+(\{[^\]]*\})\s*\]`)
	bareCallNameRE         = regexp.MustCompile(`^[a-z_][a-z0-9_]{1,63}$`)
	looseToolTextRE        = regexp.MustCompile(`(?is)(^|\n\s*\n)\s*([a-zA-Z_][a-zA-Z0-9_-]{1,63})"\s*>\s*(.*?)(?:</tool_call>|$)`)
	transcriptFenceRE      = regexp.MustCompile("(?is)```+[^\\n]*\\n(.*?)```+")
	transcriptHeaderRE     = regexp.MustCompile(`^\s*\[([a-zA-Z_][a-zA-Z0-9_-]{1,63})\]\s*(.*?)\s*$`)
	filepathHeaderRE       = regexp.MustCompile(`(?is)^\s*(?://\s*)?filepath\s*:\s*(.+?)\s*$`)
	toolNameTagRE          = regexp.MustCompile(`(?is)<tool_name>\s*([a-zA-Z_][a-zA-Z0-9_-]{1,63})\s*</tool_name>`)
	toolParametersTagRE    = regexp.MustCompile(`(?is)<parameters>(.*?)</parameters>`)
	toolIntentPromiseRE    = regexp.MustCompile(`(?is)\b(let me|i will|i'll)\b.*\b(tool|invoke|call)\b`)
	toolActionNameRE       = regexp.MustCompile(`(?is)\b(read_file|create_file|replace_string_in_file|insert_edit_into_file|run_in_terminal|semantic_search|grep_search|file_search)\b`)
	directFileSnippetRE    = regexp.MustCompile("(?is)`{3,}[a-zA-Z0-9_-]*\\n\\s*(//\\s*)?filepath\\s*:\\s*[^\\n]+\\n")
	quotedWriteValueRE     = regexp.MustCompile(`(?is)\b(?:write|paste|insert|put)\b[^"'\n]{0,80}["']([^"'\n]{1,400})["']`)
	filePathHintRE         = regexp.MustCompile("(?i)(/[\\w./-]+|[\\w.-]+\\.[a-z0-9]{1,8})")
	multiBlankRE           = regexp.MustCompile(`\n{3,}`)

	// DeepSeek DSML format regexes — built at init time to embed the const.
	dsmlToolCallsBlockRE = regexp.MustCompile(`(?is)<` + dsmlSep + `tool_calls>(.*?)</` + dsmlSep + `tool_calls>`)
	dsmlInvokeRE         = regexp.MustCompile(`(?is)<` + dsmlSep + `invoke\s+name="([^"]+)">(.*?)</` + dsmlSep + `invoke>`)
	dsmlParamRE          = regexp.MustCompile(`(?is)<` + dsmlSep + `parameter\s+name="([^"]+)"[^>]*>(.*?)</` + dsmlSep + `parameter>`)
	dsmlToolCallsTailRE  = regexp.MustCompile(`(?is)<` + dsmlSep + `tool_calls>.*$`)

	// <tool_calls>…</tool_calls> (plural, no word-boundary) — used by hy3-preview
	// and possibly others routing via OpenRouter. May contain JSON, invoke-style
	// sub-tags, <tool_call>name style lines, or be completely empty.
	toolCallsPluralInnerRE = regexp.MustCompile(`(?is)<tool_calls>(.*?)</tool_calls>`)
	toolCallsPluralBlockRE = regexp.MustCompile(`(?is)<tool_calls>.*?</tool_calls>`)
	toolCallsPluralTailRE  = regexp.MustCompile(`(?is)<tool_calls>.*$`)

	// Matches a single <tool_call>…</tool_call> segment. The body may be empty
	// (just a name) or contain optional JSON arguments.
	// Used for the hy3-preview nested format inside <tool_calls>.
	toolCallNameSegmentRE = regexp.MustCompile(`(?is)<tool_call>(.*?)(?:</tool_call>|$)`)

	// Matches any XML/HTML tag, used to strip stray tags (e.g. </arg_value>)
	// from extracted tool names.
	xmlTagRE = regexp.MustCompile(`<[^>]+>`)

	// <arg_key>name</arg_key> <arg_value>value</arg_value> — hy3-preview argument format.
	argPairRE = regexp.MustCompile(`(?is)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>`)

	// Detects unquoted JavaScript object keys (e.g. { tool: "name", args: {...} })
	// Only matches an identifier key that is NOT preceded by a quote character.
	unquotedKeyRE = regexp.MustCompile(`([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)`)
)

type Handler struct {
	client        *client.Client
	ollamaVersion string
	debug         bool
	redactor      redact.Redactor
}

func New(c *client.Client, ollamaVersion string, debug bool, redactor redact.Redactor) *Handler {
	if redactor == nil {
		redactor = redact.NewNoop()
	}
	return &Handler{client: c, ollamaVersion: ollamaVersion, debug: debug, redactor: redactor}
}

func (h *Handler) logf(format string, args ...interface{}) {
	if h.debug {
		log.Printf(format, args...)
	}
}

// redactBody runs the configured redactor over body. If redaction is
// disabled (no-op) it returns the input untouched. Any findings are
// surfaced in the debug log. The redactor itself handles all error and
// context-cancellation cases; we only forward an error if the call
// actually returned one.
func (h *Handler) redactBody(ctx context.Context, tag string, body []byte) []byte {
	out, rep, err := h.redactor.Redact(ctx, body)
	if err != nil {
		// Redaction is best-effort: log and fall back to the original
		// body so a misconfigured detector never breaks the proxy.
		h.logf("[%s] redactor error: %v (passing body through)", tag, err)
		return body
	}
	if n := len(rep.Findings); n > 0 {
		h.logf("[%s] redacted %d secret(s): %s", tag, n, summariseFindings(rep.Findings))
	}
	return out
}

func summariseFindings(findings []redact.Finding) string {
	if len(findings) == 0 {
		return ""
	}
	var b strings.Builder
	for i, f := range findings {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(f.Rule)
	}
	return b.String()
}

func (h *Handler) logBody(tag string, body []byte) {
	if !h.debug {
		return
	}

	const maxLogBytes = 60000
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
		body = h.redactBody(r.Context(), "SHOW", body)

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
		body = h.redactBody(r.Context(), "V1/CHAT", body)

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
		var miniMaxProxyReq map[string]interface{}
		var miniMaxToolCount int

		if adapter.IsMiniMaxModel(modelID) {
			h.logf("[V1/CHAT] routing to openai backend (repaired minimax path) for %s", modelID)
			if err := json.Unmarshal(body, &miniMaxProxyReq); err != nil {
				http.Error(w, `{"error":"bad request"}`, http.StatusBadRequest)
				return
			}
			miniMaxProxyReq["model"] = modelID
			miniMaxProxyReq["stream"] = false
			// Log tool count so we can tell if tools are reaching the model.
			if tools, ok := miniMaxProxyReq["tools"].([]interface{}); ok {
				miniMaxToolCount = len(tools)
				h.logf("[V1/CHAT] minimax upstream request: %d tools", len(tools))
				if len(tools) > 0 {
					if _, hasToolChoice := miniMaxProxyReq["tool_choice"]; !hasToolChoice {
						miniMaxProxyReq["tool_choice"] = "required"
						h.logf("[V1/CHAT] minimax upstream request: forcing tool_choice=required")
					}
				}
			}
			upstreamResp, upstreamErr = h.client.ChatCompletions(miniMaxProxyReq)
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
			toolHints := buildToolArgHints(req.Tools)
			rawBody, repairedBody, err := repairTaggedV1BodyWithHints(upstreamResp.Body, toolHints)
			if err != nil {
				h.logf("[V1/CHAT] minimax repair decode error: %v", err)
				http.Error(w, `{"error":"failed to decode upstream response"}`, http.StatusBadGateway)
				return
			}

			if shouldRetryMiniMaxForToolCall(miniMaxProxyReq, miniMaxToolCount, repairedBody) {
				h.logf("[V1/CHAT] minimax response promised tool use without tool_calls; retrying once with strict tool-call nudge")
				retryReq, ok := buildMiniMaxToolRetryRequest(miniMaxProxyReq)
				if ok {
					retryResp, retryErr := h.client.ChatCompletions(retryReq)
					if retryErr != nil {
						h.logf("[V1/CHAT] minimax retry request failed: %v", retryErr)
					} else {
						defer retryResp.Body.Close()
						if retryResp.StatusCode < 400 {
							rawBody2, repairedBody2, err2 := repairTaggedV1BodyWithHints(retryResp.Body, toolHints)
							if err2 != nil {
								h.logf("[V1/CHAT] minimax retry repair decode error: %v", err2)
							} else {
								rawBody = rawBody2
								repairedBody = repairedBody2
								h.logf("[V1/CHAT] minimax retry succeeded")
							}
						} else {
							respBody, _ := io.ReadAll(io.LimitReader(retryResp.Body, 16384))
							h.logf("[V1/CHAT] minimax retry status %d: %s", retryResp.StatusCode, string(respBody))
						}
					}
				}
			}

			if miniMaxRequestRequiresToolCall(miniMaxProxyReq, miniMaxToolCount) {
				if synthesized, ok := synthesizeMiniMaxWriteToolCallFromContext(req.Messages, req.Tools, repairedBody); ok {
					repairedBody = synthesized
					h.logf("[V1/CHAT] minimax synthesized write tool call from context after prose-only response")
				}
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

			repairedBody, err := rewriteBufferedOpenAISSE(bytes.NewReader(rawSSE), w, flusher, modelID, h.logf)
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
			Type  string          `json:"type"`
			Text  string          `json:"text,omitempty"`
			ID    string          `json:"id,omitempty"`
			Name  string          `json:"name,omitempty"`
			Input json.RawMessage `json:"input,omitempty"`
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
	return repairTaggedV1BodyWithHints(r, nil)
}

func repairTaggedV1BodyWithHints(r io.Reader, toolArgHints map[string]string) ([]byte, []byte, error) {
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
				clean, toolCalls := parseTaggedAssistantContentWithHints(content, toolArgHints)
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
				clean, toolCalls := parseTaggedAssistantContentWithHints(content, toolArgHints)
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

func shouldRetryMiniMaxForToolCall(proxyReq map[string]interface{}, toolCount int, repairedBody []byte) bool {
	if !miniMaxRequestRequiresToolCall(proxyReq, toolCount) {
		return false
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(repairedBody, &payload); err != nil {
		return false
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) == 0 {
		return false
	}
	sawNonEmptyAssistantContent := false

	for _, c := range choices {
		choice, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		msg, ok := choice["message"].(map[string]interface{})
		if !ok {
			continue
		}
		if tcs, ok := msg["tool_calls"].([]interface{}); ok && len(tcs) > 0 {
			return false
		}
		content, _ := msg["content"].(string)
		content = strings.TrimSpace(content)
		if content == "" {
			continue
		}
		sawNonEmptyAssistantContent = true
		if toolIntentPromiseRE.MatchString(content) && toolActionNameRE.MatchString(content) {
			return true
		}
		if directFileSnippetRE.MatchString(content) {
			return true
		}
	}

	// If tool use is required and the assistant still returned plain content
	// without structured tool_calls, do a one-shot retry with stricter nudge.
	return sawNonEmptyAssistantContent
}

func miniMaxRequestRequiresToolCall(proxyReq map[string]interface{}, toolCount int) bool {
	if toolCount == 0 || proxyReq == nil {
		return false
	}

	tc, ok := proxyReq["tool_choice"]
	if !ok {
		return true
	}

	switch v := tc.(type) {
	case string:
		s := strings.ToLower(strings.TrimSpace(v))
		if s == "none" || s == "auto" {
			return false
		}
		return true
	case map[string]interface{}:
		return true
	default:
		return true
	}
}

func buildMiniMaxToolRetryRequest(orig map[string]interface{}) (map[string]interface{}, bool) {
	if orig == nil {
		return nil, false
	}

	b, err := json.Marshal(orig)
	if err != nil {
		return nil, false
	}
	var retry map[string]interface{}
	if err := json.Unmarshal(b, &retry); err != nil {
		return nil, false
	}

	tools, ok := retry["tools"].([]interface{})
	if !ok || len(tools) == 0 {
		return nil, false
	}

	if _, hasToolChoice := retry["tool_choice"]; !hasToolChoice {
		retry["tool_choice"] = "required"
	}

	msgs, ok := retry["messages"].([]interface{})
	if !ok {
		return nil, false
	}
	msgs = append(msgs, map[string]interface{}{
		"role":    "system",
		"content": "Tool execution is required for this turn. Respond with tool_calls only and no explanatory text.",
	})
	retry["messages"] = msgs
	retry["stream"] = false

	return retry, true
}

func synthesizeMiniMaxWriteToolCallFromContext(messagesRaw, toolsRaw, repairedBody []byte) ([]byte, bool) {
	var payload map[string]interface{}
	if err := json.Unmarshal(repairedBody, &payload); err != nil {
		return nil, false
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) == 0 {
		return nil, false
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil, false
	}
	msg, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil, false
	}
	if tcs, ok := msg["tool_calls"].([]interface{}); ok && len(tcs) > 0 {
		return nil, false
	}

	userText := extractLatestUserMessageText(messagesRaw)
	if userText == "" {
		return nil, false
	}
	filePath := extractFilePathHint(userText)
	text := extractQuotedWriteValue(userText)
	if filePath == "" || text == "" {
		return nil, false
	}

	toolName, pathKey, contentKey := selectWriteToolAndKeys(toolsRaw)
	if toolName == "" || pathKey == "" || contentKey == "" {
		return nil, false
	}

	if pathKey == "relativeWorkspacePath" {
		filePath = filepath.Base(filePath)
	}

	args := map[string]string{
		pathKey:    filePath,
		contentKey: text,
	}
	argsJSON := "{}"
	if b, err := json.Marshal(args); err == nil {
		argsJSON = string(b)
	}

	msg["tool_calls"] = []interface{}{map[string]interface{}{
		"id":   "call_0",
		"type": "function",
		"function": map[string]interface{}{
			"name":      toolName,
			"arguments": argsJSON,
		},
	}}
	if fr, ok := choice["finish_reason"].(string); !ok || fr == "" || fr == "stop" {
		choice["finish_reason"] = "tool_calls"
	}

	b, err := json.Marshal(payload)
	if err != nil {
		return nil, false
	}
	return b, true
}

func extractLatestUserMessageText(messagesRaw []byte) string {
	if len(messagesRaw) == 0 {
		return ""
	}
	var msgs []map[string]interface{}
	if err := json.Unmarshal(messagesRaw, &msgs); err != nil {
		return ""
	}
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if role, _ := m["role"].(string); role != "user" {
			continue
		}
		switch c := m["content"].(type) {
		case string:
			return strings.TrimSpace(c)
		case []interface{}:
			parts := make([]string, 0)
			for _, p := range c {
				obj, ok := p.(map[string]interface{})
				if !ok {
					continue
				}
				if t, _ := obj["type"].(string); t != "text" {
					continue
				}
				if txt, _ := obj["text"].(string); strings.TrimSpace(txt) != "" {
					parts = append(parts, strings.TrimSpace(txt))
				}
			}
			return strings.TrimSpace(strings.Join(parts, "\n"))
		}
	}
	return ""
}

func extractQuotedWriteValue(s string) string {
	m := quotedWriteValueRE.FindStringSubmatch(s)
	if len(m) < 2 {
		return ""
	}
	return strings.TrimSpace(m[1])
}

func extractFilePathHint(s string) string {
	all := filePathHintRE.FindAllString(s, -1)
	if len(all) == 0 {
		return ""
	}
	v := strings.TrimSpace(all[len(all)-1])
	v = strings.Trim(v, "`\"'.,:;)")
	return v
}

func selectWriteToolAndKeys(toolsRaw []byte) (toolName, pathKey, contentKey string) {
	if len(toolsRaw) == 0 {
		return "", "", ""
	}
	var tools []adapter.OpenAITool
	if err := json.Unmarshal(toolsRaw, &tools); err != nil {
		return "", "", ""
	}

	for _, preferred := range []string{"create_file", "insert_edit_into_file"} {
		for _, t := range tools {
			if strings.TrimSpace(t.Function.Name) != preferred {
				continue
			}
			pk, ck := inferWriteParamKeys(t.Function.Parameters)
			if pk != "" && ck != "" {
				return preferred, pk, ck
			}
		}
	}
	return "", "", ""
}

func inferWriteParamKeys(parameters json.RawMessage) (pathKey, contentKey string) {
	if len(parameters) == 0 {
		return "", ""
	}
	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
	}
	if err := json.Unmarshal(parameters, &schema); err != nil {
		return "", ""
	}
	for _, k := range []string{"filePath", "relativeWorkspacePath", "path", "filepath"} {
		if _, ok := schema.Properties[k]; ok {
			pathKey = k
			break
		}
	}
	for _, k := range []string{"content", "file_text", "newString", "newText", "text"} {
		if _, ok := schema.Properties[k]; ok {
			contentKey = k
			break
		}
	}
	return pathKey, contentKey
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
	return parseTaggedAssistantContentWithHints(content, nil)
}

func parseTaggedAssistantContentWithHints(content string, toolArgHints map[string]string) (string, []map[string]interface{}) {
	toolCalls := extractToolCallsWithHints(content, toolArgHints)
	cleanSource := content
	if len(toolCalls) > 0 {
		cleanSource = stripRecoveredLooseToolText(cleanSource)
		cleanSource = stripRecoveredTranscriptToolText(cleanSource)
	}
	clean := sanitizeTaggedText(cleanSource)
	return clean, toolCalls
}

func buildToolArgHints(toolsRaw json.RawMessage) map[string]string {
	if len(toolsRaw) == 0 {
		return nil
	}

	var tools []adapter.OpenAITool
	if err := json.Unmarshal(toolsRaw, &tools); err != nil {
		return nil
	}

	hints := make(map[string]string)
	for _, tool := range tools {
		name := strings.TrimSpace(tool.Function.Name)
		if name == "" {
			continue
		}
		if key := inferPrimaryToolArgKey(tool.Function.Parameters); key != "" {
			hints[name] = key
		}
	}
	if len(hints) == 0 {
		return nil
	}
	return hints
}

func inferPrimaryToolArgKey(parameters json.RawMessage) string {
	if len(parameters) == 0 {
		return ""
	}

	var schema struct {
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required"`
	}
	if err := json.Unmarshal(parameters, &schema); err != nil {
		return ""
	}
	if len(schema.Properties) == 1 {
		for key := range schema.Properties {
			return key
		}
	}
	if len(schema.Required) == 1 {
		if _, ok := schema.Properties[schema.Required[0]]; ok {
			return schema.Required[0]
		}
	}
	return ""
}

// pluralSingleObjectToTC converts a single JSON object (from inside a
// <tool_calls> block) to an OpenAI-shaped tool-call map.  Returns nil if the
// object cannot be identified as a valid tool call.
func pluralSingleObjectToTC(obj map[string]interface{}, idx int) map[string]interface{} {
	name := ""
	argsRaw := interface{}(nil)

	// OpenAI-nested: {"type":"function","function":{"name":"fn","arguments":"..."}}
	if fn, ok := obj["function"].(map[string]interface{}); ok {
		name, _ = fn["name"].(string)
		argsRaw = fn["arguments"]
	}

	// Simple: {"name":"fn","arguments":{...}} or {"name":"fn","parameters":{...}}
	if name == "" {
		name, _ = obj["name"].(string)
		if argsRaw == nil {
			if a, ok := obj["arguments"]; ok {
				argsRaw = a
			} else if p, ok := obj["parameters"]; ok {
				argsRaw = p
			}
		}
	}

	// tool_name / tool_input style (Anthropic-like)
	if name == "" {
		if tn, ok := obj["tool_name"].(string); ok {
			name = tn
			if ti, ok := obj["tool_input"]; ok {
				argsRaw = ti
			}
		}
	}

	if name == "" {
		return nil
	}

	argsJSON := "{}"
	if argsRaw != nil {
		switch v := argsRaw.(type) {
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
	}

	id := fmt.Sprintf("call_%d", idx)
	if rawID, ok := obj["id"].(string); ok && rawID != "" {
		id = rawID
	}
	return map[string]interface{}{
		"id":   id,
		"type": "function",
		"function": map[string]interface{}{
			"name":      name,
			"arguments": argsJSON,
		},
	}
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

		// Try a single JSON object (no array wrapper).
		// Models sometimes emit <tool_calls>{"name":"fn","arguments":{...}}</tool_calls>
		// or the OpenAI-nested form {"type":"function","function":{"name":"fn",...}}.
		{
			var obj map[string]interface{}
			if err := json.NewDecoder(strings.NewReader(inner)).Decode(&obj); err == nil {
				if tc := pluralSingleObjectToTC(obj, callIdx); tc != nil {
					out = append(out, tc)
					callIdx++
					continue
				}
			}
		}

		// Fall back to invoke-style sub-tags (same as minimax handler).
		invokeMatched := false
		for _, im := range invokeTagRE.FindAllStringSubmatch(inner, -1) {
			if len(im) < 3 {
				continue
			}
			name := strings.TrimSpace(im[1])
			if !bareCallNameRE.MatchString(strings.ToLower(name)) {
				continue
			}
			invokeMatched = true
			params := collectInvokeParams(im[2])
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
		if invokeMatched {
			continue
		}

		// Handle hy3-preview format: <tool_call>name\n{optional_json}</tool_call>
		// Multiple calls may be present with each name on the line after <tool_call>.
		// The closing </tool_call> may be absent for all but the last.
		// Strategy: split inner content on <tool_call> to get one segment per call.
		if strings.Contains(inner, "<tool_call>") {
			// Split on <tool_call> (case-insensitive equivalent via strings.Split on lowercased positions).
			// We rebuild by splitting the original inner on the literal "<tool_call>".
			parts := splitOnToolCallTag(inner)
			for _, seg := range parts {
				// Each segment is the content after one <tool_call> opener.
				// Strip any trailing </tool_call> or </tool_calls>.
				seg = strings.TrimSpace(seg)
				seg = stripTrailingCloseTags(seg)
				seg = strings.TrimSpace(seg)
				if seg == "" {
					continue
				}
				// Extract tool name: first word before any whitespace or XML tag opener.
				// This handles both:
				//   <tool_call>read_file\n{json}  (name on own line, json follows)
				//   <tool_call>run_in_terminal <arg_key>command</arg_key> <arg_value>...</arg_value>
				nameEnd := strings.IndexAny(seg, " \t\n<")
				var name, rest string
				if nameEnd < 0 {
					name = strings.TrimSpace(seg)
					rest = ""
				} else {
					name = strings.TrimSpace(seg[:nameEnd])
					rest = strings.TrimSpace(seg[nameEnd:])
				}
				// Strip any remaining XML tags from name for safety.
				name = strings.TrimSpace(xmlTagRE.ReplaceAllString(name, ""))
				// Must be a valid tool name; reject garbage.
				if name == "" || !bareCallNameRE.MatchString(strings.ToLower(name)) {
					continue
				}
				argsJSON := "{}"
				// Try <arg_key>k</arg_key> <arg_value>v</arg_value> pair format.
				if argPairRE.MatchString(rest) {
					params := map[string]string{}
					for _, pm := range argPairRE.FindAllStringSubmatch(rest, -1) {
						if len(pm) >= 3 {
							k := strings.TrimSpace(pm[1])
							v := strings.TrimSpace(pm[2])
							if k != "" {
								params[k] = v
							}
						}
					}
					if len(params) > 0 {
						if b, err := json.Marshal(params); err == nil {
							argsJSON = string(b)
						}
					}
				} else if rest != "" {
					// Fall back: the rest (after the name line) may be JSON.
					argContent := rest
					// If name was on its own line, rest is everything after the first \n.
					if lines := strings.SplitN(seg, "\n", 2); len(lines) > 1 {
						argContent = strings.TrimSpace(lines[1])
					}
					if argContent != "" && json.Valid([]byte(argContent)) {
						argsJSON = argContent
					}
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
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

// normalizeJSObject quotes unquoted JS object keys so the result can be
// parsed by encoding/json.  It handles the MiniMax-M3 format:
//
//	{ tool: "name", args: { "key": "value" } }
//
// Only top-level-like unquoted keys (after { or ,) are affected; keys that
// are already preceded by a quote character are left alone.
func normalizeJSObject(s string) string {
	// Replace  {key:  and  ,key:  where key is a bare identifier not preceded
	// by a quote.  The replacement adds quotes around the key.
	result := unquotedKeyRE.ReplaceAllStringFunc(s, func(match string) string {
		// match is like ",  key  :" – capture groups are groups 1,2,3.
		m := unquotedKeyRE.FindStringSubmatch(match)
		if len(m) < 4 {
			return match
		}
		// m[1]=prefix ({,), m[2]=key, m[3]=: suffix
		return m[1] + `"` + m[2] + `"` + m[3]
	})
	return result
}

func inferMalformedInvokeToolName(params map[string]string) string {
	if _, ok := params["command"]; ok {
		return "run_in_terminal"
	}
	if _, ok := params["filePath"]; ok {
		return "read_file"
	}
	return ""
}

func collectInvokeParams(body string) map[string]string {
	params := map[string]string{}
	if cm := commandTagRE.FindStringSubmatch(body); len(cm) >= 2 {
		params["command"] = strings.TrimSpace(cm[1])
	}
	for _, pm := range namedParamTagRE.FindAllStringSubmatch(body, -1) {
		if len(pm) < 3 {
			continue
		}
		k := strings.TrimSpace(pm[1])
		v := strings.TrimSpace(pm[2])
		if k != "" {
			params[k] = v
		}
	}
	for _, pm := range parameterTagRE.FindAllStringSubmatch(body, -1) {
		if len(pm) < 3 {
			continue
		}
		k := strings.TrimSpace(pm[1])
		v := strings.TrimSpace(pm[2])
		if k != "" {
			params[k] = v
		}
	}
	for _, pm := range invokeValueTagRE.FindAllStringSubmatch(body, -1) {
		if len(pm) < 4 {
			continue
		}
		k := strings.TrimSpace(pm[1])
		end := strings.TrimSpace(pm[3])
		if k == "" {
			continue
		}
		if !strings.EqualFold(k, end) {
			continue
		}
		switch strings.ToLower(k) {
		case "invoke", "function", "parameter", "tool_call", "tool_calls", "think", "command":
			continue
		}
		v := strings.TrimSpace(pm[2])
		if v == "" {
			continue
		}
		if _, exists := params[k]; !exists {
			params[k] = v
		}
	}
	return params
}

func extractTaggedToolEnvelopeCall(body string, callIdx int) map[string]interface{} {
	nameMatch := toolNameTagRE.FindStringSubmatch(body)
	if len(nameMatch) < 2 {
		return nil
	}
	name := strings.TrimSpace(nameMatch[1])
	if !bareCallNameRE.MatchString(strings.ToLower(name)) {
		return nil
	}

	params := map[string]string{}
	if pm := toolParametersTagRE.FindStringSubmatch(body); len(pm) >= 2 {
		params = collectInvokeParams(pm[1])
	}

	argsJSON := "{}"
	if b, err := json.Marshal(params); err == nil {
		argsJSON = string(b)
	}

	return map[string]interface{}{
		"id":   fmt.Sprintf("call_%d", callIdx),
		"type": "function",
		"function": map[string]interface{}{
			"name":      name,
			"arguments": argsJSON,
		},
	}
}

func extractLooseToolText(raw string, toolArgHints map[string]string) []map[string]interface{} {
	if len(toolArgHints) == 0 {
		return nil
	}

	normalized := miniMaxWrapperRE.ReplaceAllString(raw, "")
	normalized = thinkBlockRE.ReplaceAllString(normalized, "")
	match := looseToolTextRE.FindStringSubmatch(normalized)
	if len(match) < 4 {
		return nil
	}

	name := strings.TrimSpace(match[2])
	argKey := toolArgHints[name]
	if argKey == "" {
		return nil
	}
	body := strings.TrimSpace(match[3])
	body = strings.TrimSuffix(body, "</tool_call>")
	body = strings.TrimSpace(body)
	if body == "" {
		return nil
	}

	argsJSON := "{}"
	if b, err := json.Marshal(map[string]string{argKey: body}); err == nil {
		argsJSON = string(b)
	}

	return []map[string]interface{}{{
		"id":   "call_0",
		"type": "function",
		"function": map[string]interface{}{
			"name":      name,
			"arguments": argsJSON,
		},
	}}
}

func stripRecoveredLooseToolText(raw string) string {
	normalized := miniMaxWrapperRE.ReplaceAllString(raw, "")
	match := looseToolTextRE.FindStringSubmatchIndex(normalized)
	if len(match) < 6 {
		return raw
	}
	nameStart := match[4]
	if nameStart < 0 || nameStart > len(normalized) {
		return raw
	}
	return normalized[:nameStart]
}

func stripRecoveredTranscriptToolText(raw string) string {
	normalized := miniMaxWrapperRE.ReplaceAllString(raw, "")
	cleaned := transcriptFenceRE.ReplaceAllString(normalized, "")
	cleaned = strings.TrimSpace(cleaned)
	cleaned = multiBlankRE.ReplaceAllString(cleaned, "\n\n")
	return cleaned
}

// extractFilepathSnippetToolCall recovers a pseudo tool call from fenced text like:
// ```
// // filepath: /tmp/demo/test.txt
// hello
// ```
// This pattern appears when a model intends an edit but emits markdown instead
// of a structured tool call. We map it to create_file(filePath, content).
func extractFilepathSnippetToolCall(raw string) []map[string]interface{} {
	normalized := miniMaxWrapperRE.ReplaceAllString(raw, "")
	normalized = thinkBlockRE.ReplaceAllString(normalized, "")

	for _, fm := range transcriptFenceRE.FindAllStringSubmatch(normalized, -1) {
		if len(fm) < 2 {
			continue
		}
		fenced := strings.TrimSpace(fm[1])
		if fenced == "" {
			continue
		}

		lines := strings.Split(fenced, "\n")
		if len(lines) < 2 {
			continue
		}
		header := strings.TrimSpace(lines[0])
		hm := filepathHeaderRE.FindStringSubmatch(header)
		if len(hm) < 2 {
			continue
		}
		filePath := strings.TrimSpace(hm[1])
		filePath = strings.Trim(filePath, "`\"'")
		if filePath == "" {
			continue
		}

		content := strings.Join(lines[1:], "\n")
		content = strings.TrimSpace(content)

		args := map[string]string{
			"filePath": filePath,
			"content":  content,
		}
		argsJSON := "{}"
		if b, err := json.Marshal(args); err == nil {
			argsJSON = string(b)
		}

		return []map[string]interface{}{{
			"id":   "call_0",
			"type": "function",
			"function": map[string]interface{}{
				"name":      "create_file",
				"arguments": argsJSON,
			},
		}}
	}

	return nil
}

// splitOnToolCallTag splits s on every occurrence of "<tool_call>" (case-insensitive)
// and returns the segments that follow each opener (the first empty segment before
// the first opener is dropped).
func splitOnToolCallTag(s string) []string {
	const open = "<tool_call>"
	lower := strings.ToLower(s)
	var parts []string
	pos := 0
	for {
		idx := strings.Index(lower[pos:], open)
		if idx < 0 {
			break
		}
		abs := pos + idx + len(open)
		pos = abs
		// Collect until the next <tool_call> or end of string.
		next := strings.Index(lower[abs:], open)
		var seg string
		if next < 0 {
			seg = s[abs:]
		} else {
			seg = s[abs : abs+next]
		}
		parts = append(parts, seg)
	}
	return parts
}

// stripTrailingCloseTags removes </tool_call> and </tool_calls> suffixes from s.
func stripTrailingCloseTags(s string) string {
	for {
		lower := strings.ToLower(s)
		if strings.HasSuffix(lower, "</tool_call>") {
			s = s[:len(s)-len("</tool_call>")]
			s = strings.TrimRight(s, " \t\r\n")
			continue
		}
		if strings.HasSuffix(lower, "</tool_calls>") {
			s = s[:len(s)-len("</tool_calls>")]
			s = strings.TrimRight(s, " \t\r\n")
			continue
		}
		break
	}
	return s
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
	return extractToolCallsWithHints(raw, nil)
}

func extractToolCallsWithHints(raw string, toolArgHints map[string]string) []map[string]interface{} {
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
		// No closed </tool_call> — try tail variant (opener to end of string).
		// MiniMax-M3 doesn't emit </tool_call>.
		if tail := toolCallTailRE.FindString(raw); tail != "" {
			// Strip the opener tag itself to get inner content.
			openerEnd := strings.Index(tail, ">")
			if openerEnd >= 0 {
				inner := strings.TrimSpace(tail[openerEnd+1:])
				if inner != "" {
					matches = [][]string{{"", inner}}
				}
			}
		}
	}
	if len(matches) == 0 {
		if tcs := extractTranscriptToolText(raw, toolArgHints); len(tcs) > 0 {
			return tcs
		}
		if tcs := extractLooseToolText(raw, toolArgHints); len(tcs) > 0 {
			return tcs
		}
		return extractFilepathSnippetToolCall(raw)
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

		jsonInner := inner
		if fence := transcriptFenceRE.FindStringSubmatch(inner); len(fence) >= 2 {
			fenced := strings.TrimSpace(fence[1])
			if fenced != "" {
				jsonInner = fenced
			}
		}

		callsBeforeBlock := len(out)

		dec := json.NewDecoder(strings.NewReader(jsonInner))
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
			} else if t, ok := obj["tool"].(string); ok {
				// MiniMax-M3: { tool: "name", args: {...} }
				name = t
			}
			if name == "" {
				// MiniMax stream style may emit bare object calls:
				// {file_search:{...}} {read_file:{...}}
				if len(obj) == 1 {
					for k, v := range obj {
						candidate := strings.TrimSpace(k)
						if bareCallNameRE.MatchString(strings.ToLower(candidate)) {
							name = candidate
							if b, err := json.Marshal(v); err == nil {
								out = append(out, map[string]interface{}{
									"id":   fmt.Sprintf("call_%d", callIdx),
									"type": "function",
									"function": map[string]interface{}{
										"name":      name,
										"arguments": string(b),
									},
								})
								callIdx++
							}
						}
					}
				}
				if len(out) > callsBeforeBlock {
					continue
				}
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
			} else if a, ok := obj["args"]; ok {
				// MiniMax-M3: args key instead of arguments
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

		if parsedAny && len(out) > callsBeforeBlock {
			continue
		}

		// JSON parse failed — try normalizing JavaScript-style object syntax
		// where keys may be unquoted: { tool: "name", args: {...} }
		// This is the MiniMax-M3 format as of 2026-06.
		if normalized := normalizeJSObject(jsonInner); normalized != jsonInner {
			dec2 := json.NewDecoder(strings.NewReader(normalized))
			for {
				var obj map[string]interface{}
				if err := dec2.Decode(&obj); err != nil {
					break
				}
				parsedAny = true

				name := ""
				if n, ok := obj["name"].(string); ok {
					name = n
				} else if fn, ok := obj["function"].(map[string]interface{}); ok {
					name, _ = fn["name"].(string)
				} else if t, ok := obj["tool"].(string); ok {
					name = t
				}
				if name == "" {
					if len(obj) == 1 {
						for k, v := range obj {
							candidate := strings.TrimSpace(k)
							if bareCallNameRE.MatchString(strings.ToLower(candidate)) {
								name = candidate
								if b, err := json.Marshal(v); err == nil {
									out = append(out, map[string]interface{}{
										"id":   fmt.Sprintf("call_%d", callIdx),
										"type": "function",
										"function": map[string]interface{}{
											"name":      name,
											"arguments": string(b),
										},
									})
									callIdx++
								}
							}
						}
					}
					if len(out) > callsBeforeBlock {
						continue
					}
					continue
				}

				argsJSON := "{}"
				for _, key := range []string{"arguments", "args", "parameters"} {
					if a, ok := obj[key]; ok {
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
						break
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
		}

		if parsedAny && len(out) > callsBeforeBlock {
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

			params := collectInvokeParams(im[2])

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

		for _, im := range malformedInvokeParamRE.FindAllStringSubmatch(inner, -1) {
			if len(im) < 4 {
				continue
			}

			params := map[string]string{}
			key := strings.TrimSpace(im[1])
			value := strings.TrimSpace(im[2])
			if key != "" && value != "" {
				params[key] = value
			}

			body := im[3]
			if cm := commandTagRE.FindStringSubmatch(body); len(cm) >= 2 {
				params["command"] = strings.TrimSpace(cm[1])
			}
			for _, pm := range namedParamTagRE.FindAllStringSubmatch(body, -1) {
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
			for _, pm := range parameterTagRE.FindAllStringSubmatch(body, -1) {
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

			name := inferMalformedInvokeToolName(params)
			if name == "" {
				continue
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

		if tc := extractTaggedToolEnvelopeCall(jsonInner, callIdx); tc != nil {
			out = append(out, tc)
			callIdx++
			continue
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
		if tcs := extractTranscriptToolText(raw, toolArgHints); len(tcs) > 0 {
			return tcs
		}
		return extractLooseToolText(raw, toolArgHints)
	}
	return out
}

func extractTranscriptToolText(raw string, toolArgHints map[string]string) []map[string]interface{} {
	normalized := miniMaxWrapperRE.ReplaceAllString(raw, "")
	normalized = thinkBlockRE.ReplaceAllString(normalized, "")

	blocks := transcriptFenceRE.FindAllStringSubmatch(normalized, -1)
	if len(blocks) == 0 {
		return nil
	}

	out := make([]map[string]interface{}, 0)
	callIdx := 0

	for _, block := range blocks {
		if len(block) < 2 {
			continue
		}
		body := block[1]
		lines := strings.Split(body, "\n")

		headIdx := -1
		toolName := ""
		headTail := ""
		for i, line := range lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}
			if m := transcriptHeaderRE.FindStringSubmatch(trimmed); len(m) >= 3 {
				headIdx = i
				toolName = strings.TrimSpace(m[1])
				headTail = strings.TrimSpace(m[2])
			}
			break
		}
		if headIdx < 0 || toolName == "" {
			continue
		}

		remainder := ""
		if headIdx+1 < len(lines) {
			remainder = strings.TrimSpace(strings.Join(lines[headIdx+1:], "\n"))
		}

		argsMap := map[string]interface{}{}
		switch toolName {
		case "create_file":
			if path := parseTranscriptPath(headTail, "creating ", "create "); path != "" {
				argsMap["filePath"] = path
			}
			if remainder != "" {
				argsMap["content"] = remainder
			}
			if _, ok := argsMap["filePath"]; !ok {
				continue
			}
			if _, ok := argsMap["content"]; !ok {
				continue
			}
		case "read_file":
			path := parseTranscriptPath(headTail, "reading ", "read ")
			if path == "" && remainder != "" {
				path = strings.TrimSpace(strings.Split(remainder, "\n")[0])
			}
			if path == "" {
				continue
			}
			argsMap["filePath"] = path
			argsMap["startLine"] = 1
			argsMap["endLine"] = 200
		default:
			if argKey := toolArgHints[toolName]; argKey != "" {
				value := strings.TrimSpace(remainder)
				if value == "" {
					value = strings.TrimSpace(headTail)
				}
				if value == "" {
					continue
				}
				argsMap[argKey] = value
			} else {
				continue
			}
		}

		argsJSON := "{}"
		if b, err := json.Marshal(argsMap); err == nil {
			argsJSON = string(b)
		}

		out = append(out, map[string]interface{}{
			"id":   fmt.Sprintf("call_%d", callIdx),
			"type": "function",
			"function": map[string]interface{}{
				"name":      toolName,
				"arguments": argsJSON,
			},
		})
		callIdx++
	}

	if len(out) == 0 {
		return nil
	}
	return out
}

func parseTranscriptPath(s string, prefixes ...string) string {
	text := strings.TrimSpace(s)
	lower := strings.ToLower(text)
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			path := strings.TrimSpace(text[len(p):])
			path = strings.Trim(path, "`\"')(")
			return strings.TrimSpace(path)
		}
	}
	return ""
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

func rewriteBufferedOpenAISSE(reader io.Reader, writer io.Writer, flusher http.Flusher, fallbackModel string, logf func(string, ...interface{})) ([]byte, error) {
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

						storedIndex := tcIdx
						switch v := tcm["index"].(type) {
						case float64:
							storedIndex = int(v)
						case int:
							storedIndex = v
						}

						tcs, ok := st.toolCalls[storedIndex]
						if !ok {
							tcs = &toolCallState{}
							st.toolCalls[storedIndex] = tcs
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

		// Debug: log the accumulated content before tag extraction so we can see
		// any un-extracted <tool_calls> markup when things go wrong.
		if logf != nil {
			rawContent := st.content.String()
			snippet := rawContent
			if len(snippet) > 500 {
				snippet = snippet[:500] + "…"
			}
			logf("[SSE/REWRITE] choice[%d] native_tool_calls=%d accumulated_content(%d): %s",
				idx, len(st.toolCalls), len(rawContent), snippet)
		}

		clean, tcs := parseTaggedAssistantContent(st.content.String())

		if len(st.toolCalls) > 0 {
			fromUpstream := make([]map[string]interface{}, 0, len(st.toolCalls))
			tcKeys := make([]int, 0, len(st.toolCalls))
			for k := range st.toolCalls {
				tcKeys = append(tcKeys, k)
			}
			sort.Ints(tcKeys)
			for i, k := range tcKeys {
				tc, ok := st.toolCalls[k]
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

	for _, choice := range resp.Choices {
		choiceIndex := choice.Index
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
						"index": choiceIndex,
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
			for tcIdx, tc := range toolCalls {
				deltaToolCalls = append(deltaToolCalls, map[string]interface{}{
					"index": tcIdx,
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
						"index": choiceIndex,
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

	if len(resp.Choices) == 0 {
		finalChunk := map[string]interface{}{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": "stop",
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
	} else {
		for _, choice := range resp.Choices {
			finishReason := "stop"
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finishReason = *choice.FinishReason
			} else if choice.Message != nil && len(choice.Message.ToolCalls) > 0 {
				finishReason = "tool_calls"
			}

			finalChunk := map[string]interface{}{
				"id":      id,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   model,
				"choices": []map[string]interface{}{
					{
						"index":         choice.Index,
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
		}
	}

	_, err := io.WriteString(w, "data: [DONE]\n\n")
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
		body = h.redactBody(r.Context(), "CHAT", body)

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
		body = h.redactBody(r.Context(), "GENERATE", body)

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
