package main

import (
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/mheers/opencode-go-ollama-bridge/internal/client"
	"github.com/mheers/opencode-go-ollama-bridge/internal/config"
	"github.com/mheers/opencode-go-ollama-bridge/internal/handler"
	"github.com/mheers/opencode-go-ollama-bridge/internal/redact"
	"github.com/mheers/opencode-go-ollama-bridge/internal/server"
)

var (
	flagAPIKey        string
	flagBaseURL       string
	flagListenAddr    string
	flagListenPort    string
	flagOllamaVersion string
	flagDebug         bool
	flagRedactSecrets bool
	flagRedactMode    string
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "opencode-go-ollama-bridge",
		Short: "Ollama-compatible API bridge for OpenCode Go",
		Long:  "A bridge that exposes an Ollama-compatible API endpoint and forwards requests to the OpenCode Go API.",
		RunE:  run,
	}

	rootCmd.Flags().StringVar(&flagAPIKey, "api-key", "", "OpenCode Go API key (also set via OPENCODE_GO_API_KEY env)")
	rootCmd.Flags().StringVar(&flagBaseURL, "base-url", "", "OpenCode Go API base URL (also set via OPENCODE_GO_BASE_URL env)")
	rootCmd.Flags().StringVarP(&flagListenAddr, "listen", "l", "", "Listen address e.g. :11434 or 0.0.0.0:11434 (also set via OLLAMA_BRIDGE_LISTEN env)")
	rootCmd.Flags().StringVarP(&flagListenPort, "port", "p", "", "Listen port e.g. 11434 (overridden by --listen)")
	rootCmd.Flags().StringVarP(&flagOllamaVersion, "version", "v", "", "Ollama version to report (also set via OLLAMA_BRIDGE_VERSION env, default 0.6.4)")
	rootCmd.Flags().BoolVarP(&flagDebug, "debug", "d", false, "Enable debug logging of all requests and responses")
	rootCmd.Flags().BoolVar(&flagRedactSecrets, "redact-secrets", false, "Redact secrets in HTTP request bodies before forwarding upstream (also set via OLLAMA_BRIDGE_REDACT_SECRETS=1)")
	rootCmd.Flags().StringVar(&flagRedactMode, "redact-mode", "hide", "Redaction mode when --redact-secrets is enabled: 'hide' (default) replaces matches with a placeholder, 'drop' removes the entire line containing a match")

	if err := rootCmd.Execute(); err != nil {
		log.Fatalf("error: %v", err)
	}
}

func run(cmd *cobra.Command, args []string) error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	listenAddr := flagListenAddr
	if listenAddr == "" && flagListenPort != "" {
		listenAddr = ":" + flagListenPort
	}

	if err := cfg.ApplyOverrides(&config.Overrides{
		APIKey:        flagAPIKey,
		BaseURL:       flagBaseURL,
		ListenAddr:    listenAddr,
		OllamaVersion: flagOllamaVersion,
		RedactSecrets: &flagRedactSecrets,
		RedactMode:    flagRedactMode,
	}); err != nil {
		return err
	}

	redactor, err := redact.New(cfg.RedactSecrets, redact.Mode(cfg.RedactMode))
	if err != nil {
		return err
	}
	if cfg.RedactSecrets {
		log.Printf("secret redaction enabled (mode=%s)", cfg.RedactMode)
	}

	c := client.New(cfg.BaseURL, cfg.APIKey)
	h := handler.New(c, cfg.OllamaVersion, flagDebug, redactor)

	srv := &http.Server{
		Addr:    cfg.ListenAddr,
		Handler: server.New(h, flagDebug, redactor),
	}

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		srv.Close()
	}()

	log.Printf("OpenCode Go Ollama Bridge listening on %s (version %s) debug=%v", cfg.ListenAddr, cfg.OllamaVersion, flagDebug)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}
