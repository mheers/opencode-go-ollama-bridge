# opencode-go-ollama-bridge

A lightweight bridge that exposes an **Ollama-compatible HTTP API** and transparently forwards requests to the [OpenCode Go](https://opencode.ai) API. This lets any tool that already speaks the Ollama protocol (Open WebUI, Continue.dev, Cursor, LM Studio, …) work out-of-the-box with OpenCode Go's models.

## How it works

```
Ollama client  ──►  opencode-go-ollama-bridge  ──►  OpenCode Go API
(any tool)           localhost:11434                opencode.ai/zen/go/v1
```

The bridge translates:
- `GET  /api/tags`            → fetches and maps models from `/models`
- `POST /api/chat`            → streams or buffers via `/chat/completions`
- `POST /api/generate`        → same, using a single-turn prompt
- `GET  /api/version`         → returns the configured Ollama version string
- `POST /api/show`            → returns basic model metadata
- `GET  /api/ps`              → returns an empty running-models list
- `GET  /v1/models`           → OpenAI-compatible model list
- `POST /v1/chat/completions` → OpenAI-compatible chat completions

## Requirements

- Go 1.22 or later (for `go install`)
- An [OpenCode Go](https://opencode.ai) account and API key

## Installation

### go install (recommended)

```bash
go install github.com/mheers/opencode-go-ollama-bridge/cmd/bridge@latest
```

The binary will be placed in `$(go env GOPATH)/bin/bridge`. Make sure that directory is on your `$PATH`.

### Build from source

```bash
git clone https://github.com/mheers/opencode-go-ollama-bridge.git
cd opencode-go-ollama-bridge
make build          # produces bin/opencode-go-ollama-bridge
```

### Docker

```bash
docker build -t opencode-go-ollama-bridge .
docker run -e OPENCODE_GO_API_KEY=<your-key> -p 11434:11434 opencode-go-ollama-bridge
```

## Configuration

All settings can be provided via **environment variables** or **CLI flags** (flags take precedence).

| Environment variable      | CLI flag          | Default                             | Description                                  |
|---------------------------|-------------------|-------------------------------------|----------------------------------------------|
| `OPENCODE_GO_API_KEY`     | `--api-key`       | *(required)*                        | Your OpenCode Go API key                     |
| `OPENCODE_GO_BASE_URL`    | `--base-url`      | `https://opencode.ai/zen/go/v1`     | OpenCode Go API base URL                     |
| `OLLAMA_BRIDGE_LISTEN`    | `--listen` / `-l` | `:11434`                            | Listen address, e.g. `0.0.0.0:11434`         |
| `OLLAMA_BRIDGE_VERSION`   | `--version` / `-v`| `0.24.0`                            | Ollama version string reported to clients     |
| *(none)*                  | `--port` / `-p`   | *(overridden by `--listen`)*        | Short-form port only, e.g. `11434`            |
| *(none)*                  | `--debug` / `-d`  | `false`                             | Log all requests and responses to stdout      |
| `OLLAMA_BRIDGE_REDACT_SECRETS` | *(none)*     | `false`           | Run gitleaks over request bodies before forwarding upstream |
| `OLLAMA_BRIDGE_REDACT_MODE`    | *(none)*     | `hide`            | Redaction mode: `hide` replaces with `[REDACTED:<rule>:<hash>]`, `drop` removes the whole line |

## Usage

### Quickstart

```bash
export OPENCODE_GO_API_KEY=your-api-key-here
bridge
# INFO  OpenCode Go Ollama Bridge listening on :11434 …
```

Then point any Ollama-compatible client at `http://localhost:11434`.

### Override the listen address

```bash
bridge --listen 0.0.0.0:8080
# or
bridge --port 8080
```

### Custom base URL

```bash
bridge --base-url https://my-opencode-instance.example.com/v1 --api-key ...
```

### Debug logging

```bash
bridge --debug
```

All incoming HTTP requests (headers + body) and outgoing upstream requests/responses are printed to stdout.

### CLI reference

```
bridge --help

Usage:
  bridge [flags]

Flags:
  -d, --debug             Enable debug logging of all requests and responses
      --api-key string    OpenCode Go API key (also set via OPENCODE_GO_API_KEY env)
      --base-url string   OpenCode Go API base URL (also set via OPENCODE_GO_BASE_URL env)
  -l, --listen string     Listen address e.g. :11434 or 0.0.0.0:11434 (also set via OLLAMA_BRIDGE_LISTEN env)
  -p, --port string       Listen port e.g. 11434 (overridden by --listen)
  -v, --version string    Ollama version to report (also set via OLLAMA_BRIDGE_VERSION env, default 0.6.4)
      --redact-secrets    Run gitleaks over request bodies before forwarding upstream (also set via OLLAMA_BRIDGE_REDACT_SECRETS env)
      --redact-mode string Redaction mode: "hide" (default) replaces secrets with placeholders, "drop" removes the whole line (also set via OLLAMA_BRIDGE_REDACT_MODE env)
  -h, --help              help for bridge
```

### Secret redaction

Pass `--redact-secrets` (or set `OLLAMA_BRIDGE_REDACT_SECRETS=true`) to run [gitleaks](https://github.com/gitleaks/gitleaks) over every request body before it is forwarded to the upstream OpenCode Go API. The detector runs in-process; the bridge never shells out to the gitleaks CLI.

When redaction is enabled, request bodies are scanned for API keys, tokens, private keys, and similar patterns. `--redact-mode` (or `OLLAMA_BRIDGE_REDACT_MODE`) selects how matches are handled:

- `hide` (default) — replaces each detected secret with a deterministic placeholder of the form `[REDACTED:<rule>:<short-hash>]`. The body stays valid JSON and the same secret always produces the same placeholder.
- `drop` — removes the entire line containing the secret. Use this when you want the secret to disappear completely, at the cost of possibly breaking long string fields that span multiple lines.

Redaction is best-effort: if the redactor itself errors, the bridge logs the error and forwards the original body unchanged so a misconfigured detector cannot break the proxy. Debug logging (`--debug`) also applies the redactor to request bodies before printing them.

## Connecting popular clients

### Open WebUI

Set **Ollama Base URL** to `http://localhost:11434` in the admin settings.

### Continue.dev (`~/.continue/config.json`)

```json
{
  "models": [
    {
      "title": "OpenCode Go",
      "provider": "ollama",
      "model": "deepseek-v4-flash",
      "apiBase": "http://localhost:11434"
    }
  ]
}
```

### curl

```bash
# List models
curl http://localhost:11434/api/tags

# Chat (streaming)
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

## Development

```bash
# Run tests
make test

# Run tests with coverage report
make test-cover

# Format & vet
make lint

# Build binary
make build
```

## Model compatibility

The bridge normalises non-standard tool-call markup and routes models to the correct upstream endpoint automatically.
Run `make probe` to refresh results against your API key.

| Model | Backend path | Tool calls | Reasoning | Notes |
|-------|-------------|------------|-----------|-------|
| `minimax-m3` | OpenAI compat | ✓ | ✗ | Native `tool_calls`; `<think>` stripped from content |
| `minimax-m2.7` | OpenAI compat | ✓ | ✗ | Native `tool_calls` |
| `minimax-m2.5` | OpenAI compat (repaired) | ✓ | ✓ | Bridge forces `stream=false`, strips `<tool_call>` markup |
| `deepseek-v4-pro` | OpenAI compat | ✓ | ✓ | Native `tool_calls`; DSML fallback parser handles edge cases |
| `deepseek-v4-flash` | OpenAI compat | ✓ | ✓ | Native `tool_calls`; DSML fallback parser handles edge cases |
| `qwen3.7-max` | **Anthropic messages API** | ✓ | ✓ | Only model requiring `/messages`; bridge translates request+response |
| `qwen3.7-plus` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `qwen3.6-plus` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `qwen3.5-plus` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `glm-5.1` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `glm-5` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `kimi-k2.6` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `kimi-k2.5` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `mimo-v2-pro` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `mimo-v2-omni` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `mimo-v2.5-pro` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `mimo-v2.5` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |
| `hy3-preview` | OpenAI compat | ✓ | ✓ | Native `tool_calls` |

To add support for a new model that uses a non-standard format: add a regex pattern and parser in `internal/handler/handler.go`.
If the model only supports the Anthropic messages API, add it to `IsAnthropicOnlyModel` in `internal/adapter/chat.go`.

## License

[MIT](LICENSE)
