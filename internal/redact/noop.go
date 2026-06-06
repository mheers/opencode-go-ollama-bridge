package redact

import "context"

// Noop is a Redactor that returns the input unchanged. It is used when
// secret redaction is disabled via the --redact-secrets flag.
type Noop struct{}

// Redact returns in untouched with an empty Report.
func (Noop) Redact(_ context.Context, in []byte) ([]byte, Report, error) {
	return in, Report{}, nil
}

// Compile-time guarantee that Noop satisfies Redactor.
var _ Redactor = Noop{}

// NewNoop returns a Redactor that does nothing. It is the safe default
// when callers have not configured secret redaction.
func NewNoop() Redactor { return Noop{} }
