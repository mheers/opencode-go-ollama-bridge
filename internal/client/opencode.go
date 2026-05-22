package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

func New(baseURL, apiKey string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 10 * time.Minute,
		},
	}
}

type ModelsResponse struct {
	Data []Model `json:"data"`
}

type Model struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name,omitempty"`
	OwnedBy     string `json:"owned_by,omitempty"`
}

func (c *Client) ListModels() ([]Model, error) {
	req, err := http.NewRequest("GET", c.baseURL+"/models", nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("list models failed (status %d): %s", resp.StatusCode, string(body))
	}

	var ms ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&ms); err != nil {
		return nil, fmt.Errorf("decode models response: %w", err)
	}
	return ms.Data, nil
}

func (c *Client) ChatCompletions(reqBody any) (*http.Response, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	c.setAuth(req)

	return c.httpClient.Do(req)
}

func (c *Client) Messages(reqBody any) (*http.Response, error) {
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", c.baseURL+"/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	return c.httpClient.Do(req)
}

func (c *Client) setAuth(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
}
