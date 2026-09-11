package ui

import (
	"strings"
	"testing"
	"time"

	"github.com/andreitelteu/qwen-engine/agent-evals/internal/engine"
)

func TestTailTruncatePreservesTheEndOfAnswerAndThinking(t *testing.T) {
	if got, want := tailTruncate("abcdefghij", 5), "…ghij"; got != want {
		t.Fatalf("tail truncation = %q, want %q", got, want)
	}
	if got, want := truncate("abcdefghij", 5), "abcd…"; got != want {
		t.Fatalf("ordinary truncation = %q, want %q", got, want)
	}
}

func TestHeaderOnlyShowsGenerationRateWhileActiveOrHeld(t *testing.T) {
	m := model{
		width:        160,
		engineOnline: true,
		metrics: engine.Metrics{
			GenerationPerSecond: 42,
			GenerationActive:    true,
		},
	}
	if !strings.Contains(m.header(), "42.0 tok/s") {
		t.Fatalf("active generation rate missing from header: %q", m.header())
	}

	m.metrics.GenerationActive = false
	m.metrics.GenerationHoldUntil = time.Now().Add(time.Second)
	if !strings.Contains(m.header(), "42.0 tok/s") {
		t.Fatalf("held generation rate missing from header: %q", m.header())
	}

	m.metrics.GenerationHoldUntil = time.Now().Add(-time.Second)
	if strings.Contains(m.header(), "tok/s") {
		t.Fatalf("expired generation rate remained in header: %q", m.header())
	}
}
