package client

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("expected GET, got %s", r.Method)
		}
		if r.URL.Path != "/models" {
			t.Errorf("expected /models, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer auth, got %s", r.Header.Get("Authorization"))
		}

		json.NewEncoder(w).Encode(ModelsResponse{
			Data: []Model{
				{ID: "glm-5.1"},
				{ID: "kimi-k2.6"},
			},
		})
	}))
	defer srv.Close()

	c := New(srv.URL, "test-key")
	models, err := c.ListModels()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].ID != "glm-5.1" {
		t.Errorf("expected glm-5.1, got %s", models[0].ID)
	}
	if models[1].ID != "kimi-k2.6" {
		t.Errorf("expected kimi-k2.6, got %s", models[1].ID)
	}
}

func TestListModelsErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"unauthorized"}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "bad-key")
	_, err := c.ListModels()
	if err == nil {
		t.Fatal("expected error for 401")
	}
}

func TestChatCompletions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer auth, got %s", r.Header.Get("Authorization"))
		}
		w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hello"}}]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "test-key")
	resp, err := c.ChatCompletions(map[string]string{
		"model": "test",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestMessages(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("expected x-api-key, got %s", r.Header.Get("x-api-key"))
		}
		if r.URL.Path != "/messages" {
			t.Errorf("expected /messages, got %s", r.URL.Path)
		}
		w.Write([]byte(`{"type":"message","role":"assistant","content":[{"type":"text","text":"hello"}]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "test-key")
	resp, err := c.Messages(map[string]string{
		"model": "test",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}
