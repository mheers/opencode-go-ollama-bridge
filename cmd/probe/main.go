// probe is a diagnostic tool that sends a standardised tool-calling request to
// every available model on the upstream OpenCode Go API and prints the raw
// response.  This helps discover and document non-standard tool-call markup
// formats so that corresponding adapters can be written in the bridge.
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
)

func main() {
	root := &cobra.Command{
		Use:   "probe",
		Short: "Probe each model with a tool-calling request and print the raw upstream response",
		RunE:  run,
	}

	root.Flags().StringVar(&flagBaseURL, "base-url", envOrDefault("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1"), "Upstream API base URL")
	root.Flags().StringVar(&flagAPIKey, "api-key", os.Getenv("OPENCODE_GO_API_KEY"), "API key (or set OPENCODE_GO_API_KEY)")
	root.Flags().StringVar(&flagModels, "models", "", "Comma-separated model IDs to probe (default: all from /models)")
	root.Flags().BoolVar(&flagStream, "stream", false, "Use streaming mode (SSE)")
	root.Flags().DurationVar(&flagTimeout, "timeout", 90*time.Second, "Per-model request timeout")
	root.Flags().BoolVar(&flagOutputJSON, "json", false, "Print raw JSON response for each model (default: pretty summary)")

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Core logic
// ---------------------------------------------------------------------------

func run(_ *cobra.Command, _ []string) error {
	if flagAPIKey == "" {
		return fmt.Errorf("API key is required: set OPENCODE_GO_API_KEY or --api-key")
	}

	baseURL := strings.TrimRight(flagBaseURL, "/")

	models, err := resolveModels(baseURL)
	if err != nil {
		return fmt.Errorf("list models: %w", err)
	}

	type result struct {
		model    string
		duration time.Duration
		summary  modelSummary
		rawBody  []byte
		err      error
	}
	results := make([]result, 0, len(models))

	sep := strings.Repeat("─", 80)

	for _, model := range models {
		fmt.Printf("\n%s\n▶  Probing model: %s\n%s\n", sep, model, sep)

		start := time.Now()
		raw, summary, probeErr := probeModel(baseURL, model)
		dur := time.Since(start)

		if probeErr != nil {
			fmt.Printf("   ERROR: %v\n", probeErr)
		} else {
			printSummary(model, summary)
			if flagOutputJSON {
				fmt.Printf("\n--- RAW RESPONSE ---\n%s\n", indentJSON(raw))
			}
		}

		results = append(results, result{
			model: model, duration: dur, summary: summary, rawBody: raw, err: probeErr,
		})
	}

	// Print compatibility table
	fmt.Printf("\n%s\n  COMPATIBILITY TABLE\n%s\n", sep, sep)
	fmt.Printf("%-30s  %-8s  %-10s  %-10s  %s\n", "MODEL", "LATENCY", "TOOL_CALLS", "REASONING", "FORMAT DETECTED")
	fmt.Printf("%-30s  %-8s  %-10s  %-10s  %s\n",
		strings.Repeat("-", 30), strings.Repeat("-", 8), strings.Repeat("-", 10), strings.Repeat("-", 10), strings.Repeat("-", 24))
	for _, r := range results {
		if r.err != nil {
			fmt.Printf("%-30s  %-8s  %-10s  %-10s  %s\n", r.model, fmtDur(r.duration), "ERROR", "ERROR", r.err.Error())
			continue
		}
		fmt.Printf("%-30s  %-8s  %-10s  %-10s  %s\n",
			r.model, fmtDur(r.duration),
			boolMark(r.summary.hasToolCalls), boolMark(r.summary.hasReasoning),
			r.summary.formatTag)
	}
	fmt.Println()
	return nil
}

// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Per-model probe
// ---------------------------------------------------------------------------

type modelSummary struct {
	hasToolCalls bool
	hasReasoning bool
	formatTag    string   // human-readable format name
	contentSnip  string   // first 200 chars of raw content
	toolNames    []string // tool names found in tool_calls
}

func probeModel(baseURL, model string) ([]byte, modelSummary, error) {
	reqBody := buildProbeRequest(model, flagStream)

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, modelSummary{}, err
	}

	httpClient := &http.Client{Timeout: flagTimeout}
	req, err := http.NewRequest("POST", baseURL+"/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, modelSummary{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+flagAPIKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, modelSummary{}, err
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, modelSummary{}, err
	}

	if resp.StatusCode >= 400 {
		return raw, modelSummary{}, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}

	var summary modelSummary
	if flagStream {
		summary = analyseSSE(raw)
	} else {
		summary = analyseJSON(raw)
	}

	return raw, summary, nil
}

// ---------------------------------------------------------------------------
// Request builder
// ---------------------------------------------------------------------------

func buildProbeRequest(model string, stream bool) map[string]interface{} {
	tools := []map[string]interface{}{
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
	}

	return map[string]interface{}{
		"model":  model,
		"stream": stream,
		"tools":  tools,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": "Please list the files in the /tmp directory using the list_files tool.",
			},
		},
	}
}

// ---------------------------------------------------------------------------
// Response analysers
// ---------------------------------------------------------------------------

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

func analyseSSE(raw []byte) modelSummary {
	// Accumulate all delta content + tool_calls across SSE chunks.
	type accTC struct {
		id   string
		typ  string
		name string
		args strings.Builder
	}
	contentBuf := strings.Builder{}
	var reasonBuf strings.Builder
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
	reasoning := reasonBuf.String()

	if reasoning != "" {
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

	// Check for tag-based formats inside content
	tagFormat := detectTagFormat(content)
	if tagFormat != "" {
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
		// Anthropic-style content blocks
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

	// Check content for non-standard tag formats
	tagFormat := detectTagFormat(content)
	if tagFormat != "" {
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

// detectTagFormat identifies non-standard tool-call markup inside content text.
func detectTagFormat(content string) string {
	if content == "" {
		return ""
	}

	// DeepSeek DSML format
	if strings.Contains(content, "｜｜DSML｜｜tool_calls") {
		return "deepseek DSML (<｜｜DSML｜｜tool_calls>)"
	}

	// MiniMax wrapper + tool_call
	if strings.Contains(content, "]<]minimax[>[") {
		return "minimax wrapped (<tool_call> + ]<]minimax[>[)"
	}

	// Generic tool_call tags
	if strings.Contains(content, "<tool_call>") || strings.Contains(content, "<tool_call ") {
		// Check for invoke sub-tag (MiniMax / Qwen style)
		if strings.Contains(content, "<invoke") {
			return "<tool_call><invoke name=...> style"
		}
		// Check for function sub-tag
		if strings.Contains(content, "<function ") {
			return "<tool_call><function name> style"
		}
		return "<tool_call>{json}</tool_call> style"
	}

	// Bracket call style: [function_name {"arg": "val"}]
	if strings.Contains(content, "[") && strings.Contains(content, "{") {
		return "bracket call [name {json}] style"
	}

	return ""
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

func printSummary(model string, s modelSummary) {
	fmt.Printf("  tool_calls : %s\n", boolMark(s.hasToolCalls))
	fmt.Printf("  reasoning  : %s\n", boolMark(s.hasReasoning))
	fmt.Printf("  format     : %s\n", s.formatTag)
	if len(s.toolNames) > 0 {
		fmt.Printf("  tools used : %s\n", strings.Join(s.toolNames, ", "))
	}
	if s.contentSnip != "" {
		fmt.Printf("  content    : %s\n", s.contentSnip)
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
