FROM golang:1.25-alpine AS builder

WORKDIR /app

COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null; true

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /bridge ./cmd/bridge

FROM alpine:3.21

RUN apk --no-cache add ca-certificates tzdata

COPY --from=builder /bridge /usr/local/bin/bridge

ENV OLLAMA_BRIDGE_LISTEN=:11434

EXPOSE 11434

ENTRYPOINT ["bridge"]
