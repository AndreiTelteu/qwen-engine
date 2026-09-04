package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAcceptsMinimalHandwrittenEvaluation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "evals.toml")
	content := "[[eval]]\nid = \"react-fix-form\"\ntitle = \"Fix form\"\nsample = \"react-app\"\nagent_prompt = \"Fix it\"\nexpected = \"It is fixed\"\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	suite, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(suite.Eval) != 1 || !suite.Eval[0].IsEnabled() {
		t.Fatalf("unexpected suite: %#v", suite)
	}
}

func TestLoadRejectsDuplicateIDs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "evals.toml")
	content := "[[eval]]\nid=\"duplicate\"\ntitle=\"One\"\nsample=\"one\"\nagent_prompt=\"a\"\nexpected=\"b\"\n[[eval]]\nid=\"duplicate\"\ntitle=\"Two\"\nsample=\"two\"\nagent_prompt=\"a\"\nexpected=\"b\"\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Fatal("expected duplicate ID error")
	}
}
