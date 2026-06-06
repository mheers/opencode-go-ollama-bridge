.PHONY: build test lint clean docker-build probe probe-stream probe-json probe-multi probe-single probe-bridge

BINARY_NAME=opencode-go-ollama-bridge
DOCKER_IMAGE=opencode-go-ollama-bridge
GOFILES=$(shell find . -name '*.go' -type f)

build:
	go build -o bin/$(BINARY_NAME) ./cmd/bridge

test:
	go test -v -count=1 -race ./...

test-cover:
	go test -v -count=1 -race -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# probe: query every model with a tool-calling prompt and print a compatibility table.
# Requires OPENCODE_GO_API_KEY to be set.
probe:
	go run ./cmd/probe

# probe-stream: same but uses streaming (SSE) mode.
probe-stream:
	go run ./cmd/probe --stream

# probe-json: same but prints the full raw JSON response per model.
probe-json:
	go run ./cmd/probe --json

# probe-single: single-turn only (no multi-turn follow-up).
probe-single:
	go run ./cmd/probe --single-only

# probe-multi: run both single-turn and multi-turn probes.
probe-multi:
	go run ./cmd/probe

# probe-bridge: full probe including bridge validation round (bridge must be running on :11433).
probe-bridge:
	go run ./cmd/probe --bridge-url http://localhost:11433/v1

lint:
	go vet ./...
	go fmt ./...

clean:
	rm -rf bin/
	rm -f coverage.out coverage.html

docker-build:
	docker build -t $(DOCKER_IMAGE):latest .

fmt:
	go fmt ./...

tidy:
	go mod tidy
