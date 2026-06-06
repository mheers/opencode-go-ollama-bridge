package redact

import (
	"context"
	"fmt"
	"sort"

	"github.com/zricethezav/gitleaks/v8/detect"
	glreport "github.com/zricethezav/gitleaks/v8/report"
)

// gitleaksRedactor is a Redactor backed by the gitleaks v8 detection engine
// used as an in-process Go library. The default gitleaks rule set is used;
// no external config file or binary is required.
type gitleaksRedactor struct {
	detector *detect.Detector
	mode     Mode
}

// newGitleaks returns a gitleaks-backed redactor configured to apply the
// given mode.
func newGitleaks(mode Mode) (*gitleaksRedactor, error) {
	d, err := detect.NewDetectorDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("redact: init gitleaks: %w", err)
	}
	return &gitleaksRedactor{detector: d, mode: mode}, nil
}

// matchSpan is the internal representation of a single detected secret
// translated into absolute byte offsets into the original input.
type matchSpan struct {
	start int
	end   int
	f     Finding
}

// Redact scans in for secrets and returns a redacted copy. See the Redactor
// interface contract in redact.go.
//
// The input is fed to gitleaks as a single string; gitleaks itself is
// line-oriented internally, but the DetectString API handles the line
// splitting. We translate the per-line StartColumn/EndColumn back into
// absolute byte offsets suitable for splicing the input.
//
// IMPORTANT: gitleaks reports line numbers as 0-based and columns as
// 1-based. The public Finding.Line field is converted to 1-based to match
// what users normally expect from a text editor.
func (r *gitleaksRedactor) Redact(ctx context.Context, in []byte) ([]byte, Report, error) {
	if len(in) == 0 {
		return in, Report{Mode: r.mode}, nil
	}

	// Honour cancellation: gitleaks Detect is synchronous and does not
	// take a context, so the best we can do is bail before the call.
	if err := ctx.Err(); err != nil {
		return nil, Report{Mode: r.mode}, err
	}

	raw := string(in)
	rawFindings := r.detector.DetectString(raw)
	if len(rawFindings) == 0 {
		return in, Report{Mode: r.mode}, nil
	}

	spans := make([]matchSpan, 0, len(rawFindings))
	for _, gf := range rawFindings {
		s, e, ok := absoluteSpan(raw, gf)
		if !ok {
			continue
		}
		spans = append(spans, matchSpan{
			start: s,
			end:   e,
			f: Finding{
				Rule:  gf.RuleID,
				Start: s,
				End:   e,
				Match: gf.Secret,
				Line:  gf.StartLine + 1, // convert 0-based → 1-based
			},
		})
	}
	if len(spans) == 0 {
		return in, Report{Mode: r.mode}, nil
	}

	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })

	report := Report{
		Mode:     r.mode,
		Findings: make([]Finding, len(spans)),
	}
	for i, s := range spans {
		report.Findings[i] = s.f
	}

	var out []byte
	switch r.mode {
	case ModeDrop:
		out = applyDrop(in, spans)
	case ModeHide:
		out = applyHide(in, spans)
	default:
		// Defensive: should be caught by New.
		return in, report, fmt.Errorf("redact: unhandled mode %q", r.mode)
	}
	return out, report, nil
}

// absoluteSpan converts a gitleaks line/column finding into absolute byte
// offsets into raw. Returns false if the offsets fall outside the input.
//
// gitleaks uses 0-based line numbers and 1-based columns. The reported
// EndLine/EndColumn pair is unreliable for multi-line matches: gitleaks'
// location function only updates endLine/endColumn when the match end
// falls inside a tracked newline window, so a multi-line match that ends
// past the final newline (e.g. a PEM block whose last byte is on the
// last line) reports EndLine==StartLine and EndColumn==0 even though
// the match extends further. To stay robust we derive the end offset
// from the trimmed secret length instead. The detector updates
// matchIndex[1] to matchIndex[0]+len(secret) before the location
// calculation, so startByte+len(secret) is the correct exclusive end
// for the redacted splice.
func absoluteSpan(raw string, f glreport.Finding) (start, end int, ok bool) {
	if f.StartLine < 0 {
		return 0, 0, false
	}
	lineStart, _ := lineBounds(raw, f.StartLine)
	if lineStart < 0 {
		return 0, 0, false
	}
	// Columns are 1-based; translate to a 0-based byte offset on the line.
	sc := f.StartColumn - 1
	if sc < 0 {
		sc = 0
	}
	s := lineStart + sc
	if s < 0 || s > len(raw) {
		return 0, 0, false
	}
	// Prefer the secret length when available; fall back to the column
	// pair for unusual cases (e.g. secret with embedded newlines that
	// shifted the reported EndColumn).
	if n := len(f.Secret); n > 0 {
		e := s + n
		if e > len(raw) {
			e = len(raw)
		}
		if e < s {
			return 0, 0, false
		}
		return s, e, true
	}
	// Fallback: column-based end on the same line.
	ec := f.EndColumn
	if ec < sc+1 {
		ec = sc + 1
	}
	e := lineStart + ec
	if f.EndLine != f.StartLine {
		_, lineEnd := lineBounds(raw, f.EndLine)
		if lineEnd > 0 {
			e = lineEnd
		}
	}
	if e > len(raw) {
		e = len(raw)
	}
	if e < s {
		return 0, 0, false
	}
	return s, e, true
}

// lineBounds returns the [start, end) byte offsets of the given 0-based
// line in raw. end is one past the trailing newline (if any). Returns -1
// for an out-of-range line.
func lineBounds(raw string, zeroBasedLine int) (start, end int) {
	if zeroBasedLine < 0 {
		return -1, -1
	}
	cur := 0
	for i := 0; i < len(raw); i++ {
		if cur == zeroBasedLine {
			start = i
			j := i
			for j < len(raw) && raw[j] != '\n' {
				j++
			}
			if j < len(raw) && raw[j] == '\n' {
				return start, j + 1
			}
			return start, j
		}
		if raw[i] == '\n' {
			cur++
		}
	}
	if cur == zeroBasedLine {
		return len(raw), len(raw)
	}
	return -1, -1
}

// applyHide replaces every matched span with a deterministic placeholder.
// Spans are pre-sorted by start offset; overlapping spans are coalesced by
// skipping the second of any pair so we never produce nested edits.
func applyHide(in []byte, spans []matchSpan) []byte {
	out := make([]byte, 0, len(in))
	cursor := 0
	for i, s := range spans {
		if i > 0 && s.start < cursor {
			// Overlap: skip.
			continue
		}
		out = append(out, in[cursor:s.start]...)
		out = append(out, []byte(placeholderFor(s.f))...)
		cursor = s.end
	}
	out = append(out, in[cursor:]...)
	return out
}

// applyDrop removes the entire line on which any match occurred,
// preserving the line's trailing newline. This is the line-oriented
// equivalent of scrubbing the secret and its surrounding context from the
// input.
func applyDrop(in []byte, spans []matchSpan) []byte {
	type lineSpan struct {
		start, end int
	}
	raw := string(in)
	lineSet := make([]lineSpan, 0, len(spans))
	seen := make(map[[2]int]bool)
	for _, s := range spans {
		// spans store 1-based Line in Finding.Line, but the byte-offset
		// helpers expect 0-based, so convert back here.
		ls, le := lineBounds(raw, s.f.Line-1)
		if ls < 0 {
			continue
		}
		key := [2]int{ls, le}
		if seen[key] {
			continue
		}
		seen[key] = true
		lineSet = append(lineSet, lineSpan{ls, le})
	}
	sort.Slice(lineSet, func(i, j int) bool { return lineSet[i].start < lineSet[j].start })

	// Coalesce overlapping/adjacent line spans.
	merged := make([]lineSpan, 0, len(lineSet))
	for _, ls := range lineSet {
		if n := len(merged); n > 0 && ls.start <= merged[n-1].end {
			if ls.end > merged[n-1].end {
				merged[n-1].end = ls.end
			}
		} else {
			merged = append(merged, ls)
		}
	}

	out := make([]byte, 0, len(in))
	cursor := 0
	for _, m := range merged {
		if m.start < cursor {
			continue
		}
		out = append(out, in[cursor:m.start]...)
		cursor = m.end
	}
	out = append(out, in[cursor:]...)
	return out
}

// placeholderFor returns the replacement text used for a single finding in
// hide mode. It is a deterministic function of the rule and the matched
// secret's short hash, so two equal secrets in the same input produce
// identical placeholders.
func placeholderFor(f Finding) string {
	return "[REDACTED:" + f.Rule + ":" + shortHash(f.Rule, f.Match) + "]"
}

// Compile-time guarantee that gitleaksRedactor satisfies Redactor.
var _ Redactor = (*gitleaksRedactor)(nil)
