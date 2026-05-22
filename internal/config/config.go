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
}

type Overrides struct {
	APIKey        string
	BaseURL       string
	ListenAddr    string
	OllamaVersion string
}

func Load() (*Config, error) {
	cfg := &Config{
		APIKey:        os.Getenv("OPENCODE_GO_API_KEY"),
		BaseURL:       getEnv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1"),
		ListenAddr:    getEnv("OLLAMA_BRIDGE_LISTEN", ":11434"),
		OllamaVersion: getEnv("OLLAMA_BRIDGE_VERSION", "0.24.0"),
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
