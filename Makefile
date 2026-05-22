.PHONY: build test lint clean docker-build

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
