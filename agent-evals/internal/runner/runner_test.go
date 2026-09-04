package runner

import "testing"

func TestTailPreviewTextKeepsTheNewestStreamCharacters(t *testing.T) {
	if got, want := tailPreviewText("abcdefghij", 5), "…ghij"; got != want {
		t.Fatalf("tail preview = %q, want %q", got, want)
	}
	if got, want := previewText("abcdefghij", 5), "abcd…"; got != want {
		t.Fatalf("tool preview = %q, want %q", got, want)
	}
}
