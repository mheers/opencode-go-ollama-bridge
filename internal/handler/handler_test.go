package handler

import (
	"encoding/json"
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
