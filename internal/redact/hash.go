package redact

import (
	"crypto/sha256"
	"encoding/hex"
)

// shortHash returns a stable, 8-character hex digest derived from rule and
// secret. It is used to make hide-mode placeholders reproducible: the same
// secret matched by the same rule produces the same placeholder text, which
// makes it easy to verify in tests that redaction was applied without
// exposing the underlying secret value in logs or responses.
//
// The salt (rule ID) is mixed in so that two distinct rules accidentally
// matching the same substring still produce distinct placeholders.
func shortHash(rule, secret string) string {
	h := sha256.Sum256([]byte(rule + "\x00" + secret))
	return hex.EncodeToString(h[:])[:8]
}
