package config

import (
	"os"
	"testing"
)

func TestLoadMissingAPIKey(t *testing.T) {
	os.Unsetenv("OPENCODE_GO_API_KEY")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() should not error, ApplyOverrides validates API key")
	}
	err = cfg.ApplyOverrides(&Overrides{})
	if err == nil {
		t.Fatal("expected error for missing API key")
	}
}

func TestLoadDefaults(t *testing.T) {
	os.Setenv("OPENCODE_GO_API_KEY", "test-key")
	defer os.Unsetenv("OPENCODE_GO_API_KEY")

	os.Unsetenv("OPENCODE_GO_BASE_URL")
	os.Unsetenv("OLLAMA_BRIDGE_LISTEN")
	os.Unsetenv("OLLAMA_BRIDGE_VERSION")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := cfg.ApplyOverrides(&Overrides{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.APIKey != "test-key" {
		t.Errorf("APIKey = %q, want %q", cfg.APIKey, "test-key")
	}
	if cfg.BaseURL != "https://opencode.ai/zen/go/v1" {
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
	if cfg.ListenAddr != ":11434" {
		t.Errorf("ListenAddr = %q", cfg.ListenAddr)
	}
	if cfg.OllamaVersion != "0.24.0" {
		t.Errorf("OllamaVersion = %q", cfg.OllamaVersion)
	}
}

func TestLoadCustomValues(t *testing.T) {
	os.Setenv("OPENCODE_GO_API_KEY", "sk-abc")
	os.Setenv("OPENCODE_GO_BASE_URL", "https://custom.example.com/v1")
	os.Setenv("OLLAMA_BRIDGE_LISTEN", ":9999")
	os.Setenv("OLLAMA_BRIDGE_VERSION", "0.7.0")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := cfg.ApplyOverrides(&Overrides{}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.APIKey != "sk-abc" {
		t.Errorf("APIKey = %q", cfg.APIKey)
	}
	if cfg.BaseURL != "https://custom.example.com/v1" {
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
	if cfg.ListenAddr != ":9999" {
		t.Errorf("ListenAddr = %q", cfg.ListenAddr)
	}
	if cfg.OllamaVersion != "0.7.0" {
		t.Errorf("OllamaVersion = %q", cfg.OllamaVersion)
	}
}

func TestApplyOverridesFromFlags(t *testing.T) {
	os.Unsetenv("OPENCODE_GO_API_KEY")
	os.Unsetenv("OPENCODE_GO_BASE_URL")
	os.Unsetenv("OLLAMA_BRIDGE_LISTEN")
	os.Unsetenv("OLLAMA_BRIDGE_VERSION")

	cfg, _ := Load()
	err := cfg.ApplyOverrides(&Overrides{
		APIKey:        "sk-override",
		BaseURL:       "https://flag.example.com/v1",
		ListenAddr:    ":8080",
		OllamaVersion: "0.7.5",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.APIKey != "sk-override" {
		t.Errorf("APIKey = %q", cfg.APIKey)
	}
	if cfg.BaseURL != "https://flag.example.com/v1" {
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
	if cfg.ListenAddr != ":8080" {
		t.Errorf("ListenAddr = %q", cfg.ListenAddr)
	}
	if cfg.OllamaVersion != "0.7.5" {
		t.Errorf("OllamaVersion = %q", cfg.OllamaVersion)
	}
}

func TestApplyOverridesPartial(t *testing.T) {
	os.Setenv("OPENCODE_GO_API_KEY", "sk-env")
	os.Setenv("OLLAMA_BRIDGE_LISTEN", ":11434")
	defer func() {
		os.Unsetenv("OPENCODE_GO_API_KEY")
		os.Unsetenv("OLLAMA_BRIDGE_LISTEN")
	}()

	cfg, _ := Load()
	err := cfg.ApplyOverrides(&Overrides{
		ListenAddr: ":9999",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.APIKey != "sk-env" {
		t.Errorf("APIKey = %q", cfg.APIKey)
	}
	if cfg.ListenAddr != ":9999" {
		t.Errorf("ListenAddr = %q, flag should override env", cfg.ListenAddr)
	}
	if cfg.BaseURL != "https://opencode.ai/zen/go/v1" {
		t.Errorf("BaseURL = %q", cfg.BaseURL)
	}
}
