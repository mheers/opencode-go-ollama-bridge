package redact

import (
	"context"
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

// Test fixtures: these are *test* secrets, not real credentials, chosen so
// that they actually match gitleaks' default rule set. gitleaks applies
// shannon-entropy filters, so the secret body must be high-entropy; the
// gitleaks testdata provides examples we mimic below.
//
//   - AWS:  AKIA + 16 chars from [A-Z2-7]
//   - GitHub PAT: ghp_ + 36 chars from [0-9a-zA-Z]
//   - OpenAI: sk- + 20 chars + T3BlbkFJ + 20 chars
//   - private-key: PEM block with >= 64 chars of body
//   - generic-api-key: <key/password/...>=<high-entropy>
const (
	openCodeKey   = "sk-IMO8bkaQGwuUzfnMTLZzSV0FCToTrfG6h3qgTYZ6wm9HRZ6ImkZbED6NKSembk49"
	plainEnglish  = "Hello, world! This is just a regular prompt with no secrets."
)

// fixtureRand produces a deterministic random generator so the test
// fixtures are reproducible.
func fixtureRand() *rand.Rand {
	return rand.New(rand.NewSource(0xC0FFEE))
}

func mkSecret(prefix, alphabet string, n int) string {
	r := fixtureRand()
	b := make([]byte, n)
	for i := range b {
		b[i] = alphabet[r.Intn(len(alphabet))]
	}
	return prefix + string(b)
}

func awsKey() string {
	// gitleaks aws-access-token regex: AKIA + 16 chars from [A-Z2-7].
	return mkSecret("AKIA", "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", 16)
}

func githubPAT() string {
	// gitleaks github-pat regex: ghp_ + 36 chars from [0-9a-zA-Z].
	return mkSecret("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", 36)
}

func openAIToken() string {
	// gitleaks openai-api-key regex: sk- + 20 chars + T3BlbkFJ + 20 chars.
	body := "abcdefghijklmnopqrstuvwxyz0123456789"
	r := fixtureRand()
	mk := func(n int) string {
		b := make([]byte, n)
		for i := range b {
			b[i] = body[r.Intn(len(body))]
		}
		return string(b)
	}
	return "sk-" + mk(20) + "T3BlbkFJ" + mk(20)
}

func pemKey() string {
	body := "abcdefghijklmnopqrstuvwxyz0123456789"
	r := fixtureRand()
	mk := func(n int) string {
		b := make([]byte, n)
		for i := range b {
			b[i] = body[r.Intn(len(body))]
		}
		return string(b)
	}
	return "-----BEGIN RSA PRIVATE KEY-----\n" +
		mk(40) + "\n" +
		mk(40) + "\n" +
		"-----END RSA PRIVATE KEY-----"
}

func apiKeyAssignment() string {
	return "api_key=" + mkSecret("", "abcdefghijklmnopqrstuvwxyz0123456789", 32)
}

func TestNewDisabledIsNoop(t *testing.T) {
	r, err := New(false, "drop")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	out, rep, err := r.Redact(context.Background(), []byte(awsKey()))
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if string(out) != awsKey() {
		t.Errorf("noop should not modify input; got %q", out)
	}
	if len(rep.Findings) != 0 {
		t.Errorf("noop should report no findings; got %d", len(rep.Findings))
	}
}

func TestNewInvalidMode(t *testing.T) {
	if _, err := New(true, "scramble"); err == nil {
		t.Fatal("expected error for invalid mode")
	}
}

func TestNewEmptyModeDefaultsToHide(t *testing.T) {
	r, err := New(true, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	gr, ok := r.(*gitleaksRedactor)
	if !ok {
		t.Fatalf("expected *gitleaksRedactor, got %T", r)
	}
	if gr.mode != ModeHide {
		t.Errorf("mode = %q, want %q", gr.mode, ModeHide)
	}
}

func TestRedact_Hide_AWSKey(t *testing.T) {
	r, _ := New(true, ModeHide)
	in := []byte("AWS_ACCESS_KEY_ID=" + awsKey())
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected at least one finding for AWS key")
	}
	// Use the actual matched secret from the report, not the input fixture.
	if strings.Contains(string(out), rep.Findings[0].Match) {
		t.Errorf("redacted output still contains the AWS key: %q", out)
	}
	if !strings.Contains(string(out), "[REDACTED:") {
		t.Errorf("expected placeholder in output, got %q", out)
	}
	found := false
	for _, f := range rep.Findings {
		if strings.Contains(f.Rule, "aws") {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected an aws-* rule in %v", rep.Findings)
	}
}

func TestRedact_Hide_GitHubPAT(t *testing.T) {
	r, _ := New(true, ModeHide)
	in := []byte("token: " + githubPAT())
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected at least one finding for GitHub PAT")
	}
	if strings.Contains(string(out), rep.Findings[0].Match) {
		t.Errorf("redacted output still contains the GitHub PAT: %q", out)
	}
}

func TestRedact_Hide_OpenAIKey(t *testing.T) {
	r, _ := New(true, ModeHide)
	in := []byte(openAIToken())
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected at least one finding for OpenAI key")
	}
	if strings.Contains(string(out), rep.Findings[0].Match) {
		t.Errorf("redacted output still contains the OpenAI key: %q", out)
	}
}

func TestRedact_Hide_PrivateKey(t *testing.T) {
	r, _ := New(true, ModeHide)
	in := []byte(pemKey())
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected at least one finding for PEM private key block")
	}
	if strings.Contains(string(out), "BEGIN RSA PRIVATE KEY") {
		t.Errorf("redacted output still contains PEM header: %q", out)
	}
}

func TestRedact_Hide_OpenCodeEnvVar(t *testing.T) {
	// Demonstrates that the actual `OPENCODE_GO_API_KEY` from the
	// project README is caught when wrapped in a `KEY=` assignment,
	// even though gitleaks has no specific rule for the OpenCode
	// issuer (the generic-api-key rule fires instead).
	r, _ := New(true, ModeHide)
	in := []byte("OPENCODE_GO_API_KEY=" + openCodeKey)
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected generic-api-key to match when OpenCode key appears in KEY= context")
	}
	if strings.Contains(string(out), openCodeKey) {
		t.Errorf("redacted output still contains the OpenCode key: %q", out)
	}
}

func TestRedact_PlainEnglishNoOp(t *testing.T) {
	r, _ := New(true, ModeHide)
	in := []byte(plainEnglish)
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if string(out) != plainEnglish {
		t.Errorf("plain input was modified: got %q", out)
	}
	if len(rep.Findings) != 0 {
		t.Errorf("expected no findings, got %d: %v", len(rep.Findings), rep.Findings)
	}
}

func TestRedact_Hide_PreservesJSONStructure(t *testing.T) {
	r, _ := New(true, ModeHide)
	secret := awsKey()
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type req struct {
		Model    string `json:"model"`
		Messages []msg  `json:"messages"`
	}
	original, _ := json.Marshal(req{
		Model: "test-model",
		Messages: []msg{
			{Role: "user", Content: "here is the key: " + secret},
		},
	})
	out, _, err := r.Redact(context.Background(), original)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	var parsed req
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("redacted output is not valid JSON: %v\n%s", err, out)
	}
	if strings.Contains(parsed.Messages[0].Content, secret) {
		t.Errorf("redacted JSON still contains the AWS key: %q", parsed.Messages[0].Content)
	}
	if parsed.Model != "test-model" {
		t.Errorf("non-secret fields should be preserved; got model=%q", parsed.Model)
	}
}

func TestRedact_Drop_RemovesLineWithSecret(t *testing.T) {
	r, _ := New(true, ModeDrop)
	secret := awsKey()
	in := []byte("safe line 1\nAWS_ACCESS_KEY_ID=" + secret + "\nsafe line 2\n")
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected at least one finding")
	}
	got := string(out)
	if strings.Contains(got, secret) {
		t.Errorf("drop mode left the AWS key in output: %q", got)
	}
	if !strings.Contains(got, "safe line 1") {
		t.Errorf("drop mode removed an unrelated line: %q", got)
	}
	if !strings.Contains(got, "safe line 2") {
		t.Errorf("drop mode removed an unrelated line: %q", got)
	}
}

func TestRedact_Drop_MultipleSecretsOnOneLine(t *testing.T) {
	r, _ := New(true, ModeDrop)
	aws := awsKey()
	gh := githubPAT()
	in := []byte("safe\n" + aws + " and " + gh + "\nsafe\n")
	out, _, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	got := string(out)
	if strings.Contains(got, aws) || strings.Contains(got, gh) {
		t.Errorf("drop mode left secrets in output: %q", got)
	}
	if !strings.Contains(got, "safe") {
		t.Errorf("drop mode removed unrelated content: %q", got)
	}
}

func TestRedact_ContextCancelled(t *testing.T) {
	r, _ := New(true, ModeHide)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := r.Redact(ctx, []byte(awsKey()))
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestRedact_EmptyInput(t *testing.T) {
	r, _ := New(true, ModeHide)
	out, rep, err := r.Redact(context.Background(), nil)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("expected empty output, got %d bytes", len(out))
	}
	if len(rep.Findings) != 0 {
		t.Errorf("expected no findings, got %v", rep.Findings)
	}
}

func TestRedact_AssignmentContextGenericAPIKey(t *testing.T) {
	r, _ := New(true, ModeHide)
	in := []byte("env:\n  " + apiKeyAssignment() + "\n  other: value\n")
	out, rep, err := r.Redact(context.Background(), in)
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if len(rep.Findings) == 0 {
		t.Fatal("expected generic-api-key finding")
	}
	got := string(out)
	if strings.Contains(got, rep.Findings[0].Match) {
		t.Errorf("secret value survived redaction: %q", got)
	}
}

func TestShortHash_Stable(t *testing.T) {
	a := shortHash("aws-access-token", "AKIAIOSFODNN7EXAMPLE")
	b := shortHash("aws-access-token", "AKIAIOSFODNN7EXAMPLE")
	if a != b {
		t.Errorf("shortHash not stable: %q != %q", a, b)
	}
	if len(a) != 8 {
		t.Errorf("shortHash length = %d, want 8", len(a))
	}
	c := shortHash("generic-api-key", "AKIAIOSFODNN7EXAMPLE")
	if a == c {
		t.Errorf("shortHash should differ across rules: %q == %q", a, c)
	}
}
