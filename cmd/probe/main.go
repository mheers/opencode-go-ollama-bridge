// probe is a diagnostic tool that sends standardised tool-calling requests to
// every available model on the upstream OpenCode Go API and prints the raw
// responses.  It runs two rounds per model:
//
//  1. Single-turn: bare tool-call request (tells us the basic format).
//  2. Multi-turn:  injects a fake tool result and asks the model to continue
//     (reveals how models behave in a real agentic conversation where they have
//     already called a tool once).
//
// Results are saved to probe-results/<model>.<turn>.<backend>.json.
//
// Usage:
//
//	OPENCODE_GO_API_KEY=<key> go run ./cmd/probe [flags]
//	OPENCODE_GO_API_KEY=<key> make probe
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var (
	flagBaseURL    string
	flagAPIKey     string
	flagModels     string
	flagStream     bool
	flagTimeout    time.Duration
	flagOutputJSON bool
	flagOutputDir  string
	flagSingleOnly bool
)

func main() {
	root := &cobra.Command{
		Use:   "probe",
		Short: "Probe each model with tool-calling requests (single-turn + multi-turn) and print results",
		RunE:  run,
	}

	root.Flags().StringVar(&flagBaseURL, "base-url", envOrDefault("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1"), "Upstream API base URL")
	root.Flags().StringVar(&flagAPIKey, "api-key", os.Getenv("OPENCODE_GO_API_KEY"), "API key (or set OPENCODE_GO_API_KEY)")
	root.Flags().StringVar(&flagModels, "models", "", "Comma-separated model IDs to probe (default: all from /models)")
	root.Flags().BoolVar(&flagStream, "stream", false, "Use streaming mode (SSE)")
	root.Flags().DurationVar(&flagTimeout, "timeout", 90*time.Second, "Per-model request timeout")
	root.Flags().BoolVar(&flagOutputJSON, "json", false, "Print raw JSON response for each model (default: pretty summary)")
	root.Flags().StringVar(&flagOutputDir, "output-dir", "probe-results", "Directory to save per-model raw JSON results (empty string to disable)")
	root.Flags().BoolVar(&flagSingleOnly, "single-only", false, "Only run the single-turn probe (skip multi-turn)")

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

func run(_ *cobra.Command, _ []string) error {
	if flagAPIKey == "" {
		return fmt.Errorf("API key is required: set OPENCODE_GO_API_KEY or --api-key")
	}

	baseURL := strings.TrimRight(flagBaseURL, "/")

	models, err := resolveModels(baseURL)
	if err != nil {
		return fmt.Errorf("list models: %w", err)
	}

	if flagOutputDir != "" {
		if err := os.MkdirAll(flagOutputDir, 0o755); err != nil {
			return fmt.Errorf("create output dir: %w", err)
		}
	}

	type result struct {
		model       string
		backend     string
		duration    time.Duration
		single      modelSummary
		multiTurn   modelSummary
		singleRaw   []byte
		multiRaw    []byte
		singleErr   error
		multiErr    error
	}
	results := make([]result, 0, len(models))

	sep := strings.Repeat("─", 80)

	for _, model := range models {
		fmt.Printf("\n%s\n▶  Probing model: %s\n%s\n", sep, model, sep)
		r := result{model: model, backend: "openai"}

		// ── Round 1: single-turn ──────────────────────────────────────────────
		fmt.Printf("  [round 1] single-turn tool call…\n")
		start := time.Now()
		r.singleRaw, r.single, r.singleErr = probeModel(baseURL, model)
		r.duration = time.Since(start)

		if r.singleErr != nil && isOACompatError(r.singleErr) {
			fmt.Printf("   OpenAI path failed, retrying via Anthropic messages API…\n")
			start = time.Now()
			r.singleRaw, r.single, r.singleErr = probeModelAnthropic(baseURL, model)
			r.duration = time.Since(start)
			r.backend = "anthropic"
		}

		if r.singleErr != nil {
			fmt.Printf("   ERROR (round 1): %v\n", r.singleErr)
		} else {
			printSummary("single", r.single)
			saveResult(model, "single", r.backend, r.singleRaw)
		}

		// ── Round 2: multi-turn with injected tool result ─────────────────────
		if !flagSingleOnly && r.singleErr == nil && r.single.hasToolCalls {
			fmt.Printf("  [round 2] multi-turn (injecting fake tool result)…\n")
			start = time.Now()
			if r.backend == "anthropic" {
				r.multiRaw, r.multiTurn, r.multiErr = probeMultiTurnAnthropic(baseURL, model, r.single.toolNames)
			} else {
				r.multiRaw, r.multiTurn, r.multiErr = probeMultiTurn(baseURL, model, r.single.toolNames)
			}
			r.duration += time.Since(start)
			if r.multiErr != nil {
				fmt.Printf("   ERROR (round 2): %v\n", r.multiErr)
			} else {
				printSummary("multi ", r.multiTurn)
				saveResult(model, "multi", r.backend, r.multiRaw)
			}
		}

		if flagOutputJSON && len(r.singleRaw) > 0 {
			fmt.Printf("\n--- RAW (round 1) ---\n%s\n", indentJSON(r.singleRaw))
		}
		if flagOutputJSON && len(r.multiRaw) > 0 {
			fmt.Printf("\n--- RAW (round 2) ---\n%s\n", indentJSON(r.multiRaw))
		}

		results = append(results, r)
	}

	// ── Compatibility table ───────────────────────────────────────────────────
	fmt.Printf("\n%s\n  COMPATIBILITY TABLE\n%s\n", sep, sep)
	fmt.Printf("%-28s  %-9s  %-22s  %-22s\n", "MODEL", "BACKEND", "ROUND-1 FORMAT", "ROUND-2 FORMAT")
	fmt.Printf("%-28s  %-9s  %-22s  %-22s\n",
		strings.Repeat("-", 28), strings.Repeat("-", 9), strings.Repeat("-", 22), strings.Repeat("-", 22))
	for _, r := range results {
		r1fmt := "ERROR"
		r2fmt := "(skipped)"
		if r.singleErr == nil {
			r1fmt = r.single.formatTag
		}
		if r.multiErr == nil && len(r.multiRaw) > 0 {
			r2fmt = r.multiTurn.formatTag
		} else if r.multiErr != nil {
			r2fmt = "ERROR"
		}
		fmt.Printf("%-28s  %-9s  %-22s  %-22s\n", r.model, r.backend, truncate(r1fmt, 22), truncate(r2fmt, 22))
	}
	fmt.Println()
	return nil
}

func saveResult(model, turn, backend string, raw []byte) {
	if flagOutputDir == "" || len(raw) == 0 {
		return
	}
	filename := fmt.Sprintf("%s/%s.%s.%s.json",
		flagOutputDir, strings.ReplaceAll(model, "/", "_"), turn, backend)
	if err := os.WriteFile(filename, []byte(indentJSON(raw)), 0o644); err != nil {
		fmt.Printf("   WARN: could not save %s: %v\n", filename, err)
	} else {
		fmt.Printf("  saved      : %s\n", filename)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Model resolution
// ─────────────────────────────────────────────────────────────────────────────

func resolveModels(baseURL string) ([]string, error) {
	if flagModels != "" {
		out := []string{}
		for _, m := range strings.Split(flagModels, ",") {
			if m := strings.TrimSpace(m); m != "" {
				out = append(out, m)
			}
		}
		return out, nil
	}

	req, err := http.NewRequest("GET", baseURL+"/models", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+flagAPIKey)

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var ms struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ms); err != nil {
		return nil, err
	}

	out := make([]string, 0, len(ms.Data))
	for _, m := range ms.Data {
		out = append(out, m.ID)
	}
	return out, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-model probes
// ─────────────────────────────────────────────────────────────────────────────

type modelSummary struct {
	hasToolCalls bool
	hasReasoning bool
	formatTag    string
	contentSnip  string
	toolNames    []string
}

func isOACompatError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "oa-compat") || strings.Contains(s, "not supported for format")
}

func doPost(url string, headers map[string]string, body interface{}) ([]byte, int, error) {
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, 0, err
	}
	httpClient := &http.Client{Timeout: flagTimeout}
	req, err := http.NewRequest("POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	return raw, resp.StatusCode, err
}

// probeModel sends a fresh single-turn tool-calling request via /chat/completions.
func probeModel(baseURL, model string) ([]byte, modelSummary, error) {
	raw, code, err := doPost(baseURL+"/chat/completions",
		map[string]string{"Authorization": "Bearer " + flagAPIKey},
		buildProbeRequest(model, flagStream))
	if err != nil {
		return nil, modelSummary{}, err
	}
	if code >= 400 {
		return raw, modelSummary{}, fmt.Errorf("HTTP %d: %s", code, truncate(string(raw), 200))
	}
	var summary modelSummary
	if flagStream {
		summary = analyseSSE(raw)
	} else {
		summary = analyseJSON(raw)
	}
	return raw, summary, nil
}

// probeMultiTurn continues the conversation after a fake tool result is injected.
// It sends: user → assistant(tool_call) → tool(result) → user(follow-up).
func probeMultiTurn(baseURL, model string, calledTools []string) ([]byte, modelSummary, error) {
	toolName := "list_files"
	if len(calledTools) > 0 {
		toolName = calledTools[0]
	}

	// Build the history: user → assistant with tool call → tool result → follow-up user.
	messages := []map[string]interface{}{
		{"role": "user", "content": "Please list the files in the /tmp directory using the list_files tool."},
		{
			"role": "assistant",
			"content": nil,
			"tool_calls": []map[string]interface{}{
				{
					"id":   "call_probe_1",
					"type": "function",
					"function": map[string]interface{}{
						"name":      toolName,
						"arguments": `{"path":"/tmp"}`,
					},
				},
			},
		},
		{
			"role":         "tool",
			"tool_call_id": "call_probe_1",
			"content":      "file1.txt\nfile2.go\nREADME.md",
		},
		{"role": "user", "content": "Now read the contents of file1.txt using the read_file tool."},
	}

	reqBody := buildProbeRequestWithMessages(model, messages, flagStream)
	raw, code, err := doPost(baseURL+"/chat/completions",
		map[string]string{"Authorization": "Bearer " + flagAPIKey},
		reqBody)
	if err != nil {
		return nil, modelSummary{}, err
	}
	if code >= 400 {
		return raw, modelSummary{}, fmt.Errorf("HTTP %d: %s", code, truncate(string(raw), 200))
	}
	var summary modelSummary
	if flagStream {
		summary = analyseSSE(raw)
	} else {
		summary = analyseJSON(raw)
	}
	return raw, summary, nil
}

// probeModelAnthropic probes a model via the Anthropic /messages endpoint.
func probeModelAnthropic(baseURL, model string) ([]byte, modelSummary, error) {
	type Tool struct {
		Name        string      `json:"name"`
		Description string      `json:"description"`
		InputSchema interface{} `json:"input_schema"`
	}
	reqBody := map[string]interface{}{
		"model":      model,
		"max_tokens": 1024,
		"stream":     false,
		"tools": []Tool{{
			Name:        "list_files",
			Description: "List the files in a given directory path on the local filesystem.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{"type": "string", "description": "Absolute directory path"},
				},
				"required": []string{"path"},
			},
		}},
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Please list the files in the /tmp directory using the list_files tool."},
		},
	}

	raw, code, err := doPost(baseURL+"/messages",
		map[string]string{
			"x-api-key":         flagAPIKey,
			"anthropic-version": "2023-06-01",
		}, reqBody)
	if err != nil {
		return nil, modelSummary{}, err
	}
	if code >= 400 {
		return raw, modelSummary{}, fmt.Errorf("HTTP %d: %s", code, truncate(string(raw), 200))
	}
	return raw, analyseAnthropicJSON(raw), nil
}

// probeMultiTurnAnthropic runs a multi-turn test via the Anthropic messages API.
func probeMultiTurnAnthropic(baseURL, model string, calledTools []string) ([]byte, modelSummary, error) {
	toolName := "list_files"
	if len(calledTools) > 0 {
		toolName = calledTools[0]
	}

	type Tool struct {
		Name        string      `json:"name"`
		Description string      `json:"description"`
		InputSchema interface{} `json:"input_schema"`
	}
	tools := []Tool{
		{
			Name:        "list_files",
			Description: "List the files in a given directory path on the local filesystem.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{"type": "string"},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "read_file",
			Description: "Read the contents of a file at the given path.",
			InputSchema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{"type": "string"},
				},
				"required": []string{"path"},
			},
		},
	}

	messages := []map[string]interface{}{
		{"role": "user", "content": "Please list the files in the /tmp directory using the list_files tool."},
		{
			"role": "assistant",
			"content": []map[string]interface{}{
				{
					"type":  "tool_use",
					"id":    "toolu_probe_1",
					"name":  toolName,
					"input": map[string]string{"path": "/tmp"},
				},
			},
		},
		{
			"role": "user",
			"content": []map[string]interface{}{
				{
					"type":        "tool_result",
					"tool_use_id": "toolu_probe_1",
					"content":     "file1.txt\nfile2.go\nREADME.md",
				},
			},
		},
		{"role": "user", "content": "Now read the contents of file1.txt using the read_file tool."},
	}

	reqBody := map[string]interface{}{
		"model":      model,
		"max_tokens": 1024,
		"stream":     false,
		"tools":      tools,
		"messages":   messages,
	}

	raw, code, err := doPost(baseURL+"/messages",
		map[string]string{
			"x-api-key":         flagAPIKey,
			"anthropic-version": "2023-06-01",
		}, reqBody)
	if err != nil {
		return nil, modelSummary{}, err
	}
	if code >= 400 {
		return raw, modelSummary{}, fmt.Errorf("HTTP %d: %s", code, truncate(string(raw), 200))
	}
	return raw, analyseAnthropicJSON(raw), nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Request builders
// ─────────────────────────────────────────────────────────────────────────────

func buildProbeRequest(model string, stream bool) map[string]interface{} {
	return buildProbeRequestWithMessages(model,
		[]map[string]interface{}{
			{"role": "user", "content": "Please list the files in the /tmp directory using the list_files tool."},
		}, stream)
}

func buildProbeRequestWithMessages(model string, messages []map[string]interface{}, stream bool) map[string]interface{} {
	return map[string]interface{}{
		"model":  model,
		"stream": stream,
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "list_files",
					"description": "List the files in a given directory path on the local filesystem.",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"path": map[string]interface{}{
								"type":        "string",
								"description": "The absolute directory path to list, e.g. /tmp",
							},
						},
						"required": []string{"path"},
					},
				},
			},
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "read_file",
					"description": "Read the contents of a file at the given path.",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"path": map[string]interface{}{
								"type":        "string",
								"description": "The absolute file path to read",
							},
						},
						"required": []string{"path"},
					},
				},
			},
		},
		"messages": messages,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Response analysers
// ─────────────────────────────────────────────────────────────────────────────

func analyseJSON(raw []byte) modelSummary {
	var payload map[string]interface{}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return modelSummary{formatTag: "invalid JSON"}
	}
	choices, _ := payload["choices"].([]interface{})
	if len(choices) == 0 {
		return modelSummary{formatTag: "no choices"}
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})
	if msg == nil {
		msg, _ = choice["delta"].(map[string]interface{})
	}
	return analyseMessage(msg)
}

func analyseAnthropicJSON(raw []byte) modelSummary {
	var payload struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text,omitempty"`
			Thinking string `json:"thinking,omitempty"`
			Name     string `json:"name,omitempty"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return modelSummary{formatTag: "invalid JSON"}
	}

	summary := modelSummary{}
	var textParts []string
	for _, block := range payload.Content {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "thinking":
			if block.Thinking != "" {
				summary.hasReasoning = true
			}
		case "tool_use":
			summary.hasToolCalls = true
			if block.Name != "" {
				summary.toolNames = append(summary.toolNames, block.Name)
			}
		}
	}
	if summary.hasToolCalls {
		summary.formatTag = "anthropic native tool_use"
	} else {
		summary.formatTag = "text only (no tool calls)"
	}
	summary.contentSnip = truncate(strings.Join(textParts, ""), 300)
	return summary
}

func analyseSSE(raw []byte) modelSummary {
	type accTC struct {
		id   string
		typ  string
		name string
		args strings.Builder
	}
	contentBuf := strings.Builder{}
	reasonBuf := strings.Builder{}
	toolCallMap := map[int]*accTC{}

	scanner := bufio.NewScanner(bytes.NewReader(raw))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk map[string]interface{}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		choices, _ := chunk["choices"].([]interface{})
		for _, c := range choices {
			choice, ok := c.(map[string]interface{})
			if !ok {
				continue
			}
			delta, _ := choice["delta"].(map[string]interface{})
			if delta == nil {
				continue
			}
			if v, ok := delta["content"].(string); ok {
				contentBuf.WriteString(v)
			}
			if v, ok := delta["reasoning_content"].(string); ok {
				reasonBuf.WriteString(v)
			}
			if v, ok := delta["reasoning"].(string); ok {
				reasonBuf.WriteString(v)
			}
			if dTCs, ok := delta["tool_calls"].([]interface{}); ok {
				for _, dtc := range dTCs {
					tcm, ok := dtc.(map[string]interface{})
					if !ok {
						continue
					}
					idx := 0
					if v, ok := tcm["index"].(float64); ok {
						idx = int(v)
					}
					acc, exists := toolCallMap[idx]
					if !exists {
						acc = &accTC{}
						toolCallMap[idx] = acc
					}
					if id, ok := tcm["id"].(string); ok && id != "" {
						acc.id = id
					}
					if typ, ok := tcm["type"].(string); ok && typ != "" {
						acc.typ = typ
					}
					if fn, ok := tcm["function"].(map[string]interface{}); ok {
						if name, ok := fn["name"].(string); ok && name != "" {
							acc.name = name
						}
						if args, ok := fn["arguments"].(string); ok {
							acc.args.WriteString(args)
						}
					}
				}
			}
		}
	}

	summary := modelSummary{}
	content := contentBuf.String()
	if reasonBuf.Len() > 0 {
		summary.hasReasoning = true
	}
	if len(toolCallMap) > 0 {
		summary.hasToolCalls = true
		for i := 0; i < len(toolCallMap); i++ {
			if tc, ok := toolCallMap[i]; ok && tc.name != "" {
				summary.toolNames = append(summary.toolNames, tc.name)
			}
		}
		summary.formatTag = "openai native tool_calls (SSE)"
	}
	if tagFormat := detectTagFormat(content); tagFormat != "" {
		if !summary.hasToolCalls {
			summary.hasToolCalls = true
		}
		summary.formatTag = tagFormat
	}
	if summary.formatTag == "" {
		if summary.hasToolCalls {
			summary.formatTag = "openai native tool_calls (SSE)"
		} else {
			summary.formatTag = "text only (no tool calls)"
		}
	}
	summary.contentSnip = truncate(content, 300)
	return summary
}

func analyseMessage(msg map[string]interface{}) modelSummary {
	if msg == nil {
		return modelSummary{formatTag: "no message"}
	}
	summary := modelSummary{}
	if rc, ok := msg["reasoning_content"].(string); ok && rc != "" {
		summary.hasReasoning = true
	}
	if rc, ok := msg["reasoning"].(string); ok && rc != "" {
		summary.hasReasoning = true
	}
	if tcs, ok := msg["tool_calls"].([]interface{}); ok && len(tcs) > 0 {
		summary.hasToolCalls = true
		for _, tc := range tcs {
			tcm, ok := tc.(map[string]interface{})
			if !ok {
				continue
			}
			if fn, ok := tcm["function"].(map[string]interface{}); ok {
				if name, ok := fn["name"].(string); ok {
					summary.toolNames = append(summary.toolNames, name)
				}
			}
		}
		summary.formatTag = "openai native tool_calls"
	}
	content := ""
	switch v := msg["content"].(type) {
	case string:
		content = v
	case []interface{}:
		for _, block := range v {
			if bm, ok := block.(map[string]interface{}); ok {
				if t, _ := bm["type"].(string); t == "text" {
					if text, ok := bm["text"].(string); ok {
						content += text
					}
				}
			}
		}
	}
	if tagFormat := detectTagFormat(content); tagFormat != "" {
		if !summary.hasToolCalls {
			summary.hasToolCalls = true
		}
		summary.formatTag = tagFormat
	}
	if summary.formatTag == "" {
		summary.formatTag = "text only (no tool calls)"
	}
	summary.contentSnip = truncate(content, 300)
	return summary
}

func detectTagFormat(content string) string {
	if content == "" {
		return ""
	}
	if strings.Contains(content, "｜｜DSML｜｜tool_calls") {
		return "deepseek DSML"
	}
	if strings.Contains(content, "]<]minimax[>[") {
		return "minimax wrapped <tool_call>"
	}
	if strings.Contains(content, "<tool_calls>") {
		inner := ""
		if i := strings.Index(content, "<tool_calls>"); i >= 0 {
			end := strings.Index(content[i:], "</tool_calls>")
			if end >= 0 {
				inner = strings.TrimSpace(content[i+len("<tool_calls>") : i+end])
			}
		}
		if inner == "" {
			return "<tool_calls> (plural, empty)"
		}
		if strings.HasPrefix(inner, "[") {
			return "<tool_calls>[json]</tool_calls>"
		}
		if strings.Contains(inner, "<invoke") {
			return "<tool_calls><invoke> style"
		}
		return "<tool_calls> (plural, unknown inner)"
	}
	if strings.Contains(content, "<tool_call>") || strings.Contains(content, "<tool_call ") {
		if strings.Contains(content, "<invoke") {
			return "<tool_call><invoke name=...> style"
		}
		if strings.Contains(content, "<function ") {
			return "<tool_call><function name> style"
		}
		return "<tool_call>{json}</tool_call>"
	}
	if strings.Contains(content, "[") && strings.Contains(content, "{") {
		return "bracket call [name {json}]"
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// Output helpers
// ─────────────────────────────────────────────────────────────────────────────

func printSummary(label string, s modelSummary) {
	fmt.Printf("  [%s] tool_calls=%s  reasoning=%s  format=%s\n",
		label, boolMark(s.hasToolCalls), boolMark(s.hasReasoning), s.formatTag)
	if len(s.toolNames) > 0 {
		fmt.Printf("  [%s] tools used : %s\n", label, strings.Join(s.toolNames, ", "))
	}
	if s.contentSnip != "" {
		fmt.Printf("  [%s] content    : %s\n", label, s.contentSnip)
	}
}

func boolMark(b bool) string {
	if b {
		return "✓"
	}
	return "✗"
}

func fmtDur(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func indentJSON(raw []byte) string {
	var out bytes.Buffer
	if err := json.Indent(&out, raw, "", "  "); err != nil {
		return string(raw)
	}
	return out.String()
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
