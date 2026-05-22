package server

import (
	"bytes"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/mheers/opencode-go-ollama-bridge/internal/handler"
)

func New(h *handler.Handler, debug bool) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", h.Health())
	mux.HandleFunc("/api/version", h.Version())
	mux.HandleFunc("/api/tags", h.Tags())
	mux.HandleFunc("/api/chat", h.Chat())
	mux.HandleFunc("/api/generate", h.Generate())
	mux.HandleFunc("/api/ps", h.PS())
	mux.HandleFunc("/api/show", h.Show())
	mux.HandleFunc("/v1/chat/completions", h.V1ChatCompletions())
	mux.HandleFunc("/v1/models", h.V1Models())

	if debug {
		return debugMiddleware(mux)
	}
	return mux
}

func debugMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Printf("[REQ] %s %s | Remote: %s | Content-Type: %s",
			r.Method, r.URL.String(), r.RemoteAddr, r.Header.Get("Content-Type"))

		if r.Body != nil && r.Method != http.MethodGet {
			bodyBytes, _ := io.ReadAll(r.Body)
			r.Body.Close()
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))

			bodyStr := string(bodyBytes)
			if len(bodyStr) > 1024 {
				bodyStr = bodyStr[:1024] + "..."
			}
			bodyStr = strings.ReplaceAll(bodyStr, "\n", "\\n")
			log.Printf("[REQ] Body: %s", bodyStr)
		}

		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)

		log.Printf("[RES] %s %s → %d (%d bytes)",
			r.Method, r.URL.String(), rw.status, rw.size)
	})
}

type responseWriter struct {
	http.ResponseWriter
	status int
	size   int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.size += n
	return n, err
}

func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

var _ http.Flusher = (*responseWriter)(nil)
