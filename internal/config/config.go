package config

import (
	"fmt"
	"os"
)

type Config struct {
	APIKey        string
	BaseURL       string
	ListenAddr    string
	OllamaVersion string

	// RedactSecrets toggles in-process secret redaction of HTTP request
	// bodies. When false (the default) the redactor is a no-op and there
	// is zero overhead per request.
	RedactSecrets bool

	// RedactMode selects how matches are replaced. "hide" (default)
	// inserts a deterministic placeholder; "drop" removes the entire
	// line that contains the secret.
	RedactMode string
}

type Overrides struct {
	APIKey        string
	BaseURL       string
	ListenAddr    string
	OllamaVersion string
	RedactSecrets *bool
	RedactMode    string
}

func Load() (*Config, error) {
	cfg := &Config{
		APIKey:        os.Getenv("OPENCODE_GO_API_KEY"),
		BaseURL:       getEnv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1"),
		ListenAddr:    getEnv("OLLAMA_BRIDGE_LISTEN", ":11434"),
		OllamaVersion: getEnv("OLLAMA_BRIDGE_VERSION", "0.24.0"),
		RedactSecrets: getEnv("OLLAMA_BRIDGE_REDACT_SECRETS", "") == "1" || getEnv("OLLAMA_BRIDGE_REDACT_SECRETS", "") == "true",
		RedactMode:    getEnv("OLLAMA_BRIDGE_REDACT_MODE", "hide"),
	}
	return cfg, nil
}

func (c *Config) ApplyOverrides(o *Overrides) error {
	if o.APIKey != "" {
		c.APIKey = o.APIKey
	}
	if o.BaseURL != "" {
		c.BaseURL = o.BaseURL
	}
	if o.ListenAddr != "" {
		c.ListenAddr = o.ListenAddr
	}
	if o.OllamaVersion != "" {
		c.OllamaVersion = o.OllamaVersion
	}
	if o.RedactSecrets != nil {
		c.RedactSecrets = *o.RedactSecrets
	}
	if o.RedactMode != "" {
		c.RedactMode = o.RedactMode
	}
	if c.APIKey == "" {
		return fmt.Errorf("OPENCODE_GO_API_KEY environment variable or --api-key flag is required")
	}
	return nil
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
