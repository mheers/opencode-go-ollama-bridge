// Package redact scans input byte slices for secrets and either hides or drops
// matched spans, returning redacted bytes and a Report describing what was
// found. The default implementation is backed by the gitleaks v8 detection
// engine used as a Go library (no external binary required).
//
// Typical use is to redact request bodies just before they are forwarded to an
// upstream LLM:
//
//	r, _ := redact.New(true, redact.ModeHide)
//	body, report, _ := r.Redact(ctx, body)
//	if len(report.Findings) > 0 {
//	    log.Printf("redacted %d secrets in %s", len(report.Findings), time.Since(start))
//	}
package redact

import (
	"context"
	"fmt"
)

// Mode controls how matched secrets are handled.
//
//   - ModeHide replaces the matched bytes with a placeholder string.
//   - ModeDrop removes the entire line that contains the match (newline
//     preserved).
type Mode string

const (
	// ModeHide is the default mode. Matched bytes are replaced with a
	// placeholder of the form `[REDACTED:<rule>:<hash>]`.
	ModeHide Mode = "hide"

	// ModeDrop removes the entire line that contains a match. The newline
	// (if any) is preserved, but every other byte of the line is discarded.
	// This is line-oriented; field-aware JSON redaction is intentionally out
	// of scope for v1.
	ModeDrop Mode = "drop"
)

// Valid reports whether m is one of the supported mode values.
func (m Mode) Valid() bool {
	switch m {
	case ModeHide, ModeDrop:
		return true
	default:
		return false
	}
}

// Finding describes a single secret match within the input.
//
// Start and End are byte offsets into the original input slice (End is
// exclusive). Line is the 1-based line number on which the match started.
// Match is the exact substring that triggered the rule.
type Finding struct {
	// Rule is the gitleaks rule ID that matched (e.g. "aws-access-token",
	// "github-pat", "openai-api-key", "private-key", "generic-api-key").
	Rule string

	// Start is the inclusive byte offset of the match in the original input.
	Start int

	// End is the exclusive byte offset of the match in the original input.
	End int

	// Match is the matched substring from the original input.
	Match string

	// Line is the 1-based line number on which the match started.
	Line int
}

// Report aggregates the findings produced by a single Redact call, along
// with the mode that was applied.
type Report struct {
	// Findings is the list of secrets detected in the input, in the order
	// they were encountered.
	Findings []Finding

	// Mode is the redaction mode that was applied to produce Out.
	Mode Mode
}

// Redactor scans byte slices for secrets and returns a redacted copy along
// with a Report describing what was found.
type Redactor interface {
	// Redact scans in for secrets and returns a redacted copy. When no
	// secrets are present, the returned slice may be the same backing array
	// as in.
	//
	// Implementations must not retain references to in after returning.
	Redact(ctx context.Context, in []byte) (out []byte, report Report, err error)
}

// New returns a Redactor configured by the given flag and mode.
//
// When enabled is false, a Noop redactor is returned regardless of mode;
// the redaction overhead in that case is a single byte slice copy.
//
// When enabled is true, a gitleaks-backed redactor is returned. If mode is
// empty, ModeHide is used. If mode is not one of {hide, drop}, an error is
// returned.
func New(enabled bool, mode Mode) (Redactor, error) {
	if !enabled {
		return Noop{}, nil
	}
	if mode == "" {
		mode = ModeHide
	}
	if !mode.Valid() {
		return nil, fmt.Errorf("redact: invalid mode %q (want %q or %q)", mode, ModeHide, ModeDrop)
	}
	return newGitleaks(mode)
}
