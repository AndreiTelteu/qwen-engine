package ui

import "testing"

func TestTailTruncatePreservesTheEndOfAnswerAndThinking(t *testing.T) {
	if got, want := tailTruncate("abcdefghij", 5), "…ghij"; got != want {
		t.Fatalf("tail truncation = %q, want %q", got, want)
	}
	if got, want := truncate("abcdefghij", 5), "abcd…"; got != want {
		t.Fatalf("ordinary truncation = %q, want %q", got, want)
	}
}
