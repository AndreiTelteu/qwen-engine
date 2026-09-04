package config

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/joho/godotenv"
	"github.com/pelletier/go-toml/v2"
)

type Suite struct {
	Eval []Evaluation `toml:"eval"`
}

type Evaluation struct {
	ID          string   `toml:"id"`
	Title       string   `toml:"title"`
	Sample      string   `toml:"sample"`
	Timeout     string   `toml:"timeout"`
	AgentPrompt string   `toml:"agent_prompt"`
	Expected    string   `toml:"expected"`
	Setup       []string `toml:"setup"`
	Verify      []string `toml:"verify"`
	JudgePrompt string   `toml:"judge_prompt"`
	Enabled     *bool    `toml:"enabled"`
}

func (e Evaluation) IsEnabled() bool { return e.Enabled == nil || *e.Enabled }

var validID = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{2,63}$`)

func Load(path string) (Suite, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Suite{}, fmt.Errorf("read evals TOML: %w", err)
	}
	var suite Suite
	if err := toml.Unmarshal(raw, &suite); err != nil {
		return Suite{}, fmt.Errorf("parse evals TOML: %w", err)
	}
	if len(suite.Eval) == 0 {
		return Suite{}, fmt.Errorf("evals TOML contains no [[eval]] entries")
	}
	seen := map[string]bool{}
	for _, e := range suite.Eval {
		if !validID.MatchString(e.ID) {
			return Suite{}, fmt.Errorf("eval id %q must use lowercase letters, digits, and hyphens", e.ID)
		}
		if seen[e.ID] {
			return Suite{}, fmt.Errorf("duplicate eval id %q", e.ID)
		}
		seen[e.ID] = true
		if strings.TrimSpace(e.Title) == "" || strings.TrimSpace(e.Sample) == "" {
			return Suite{}, fmt.Errorf("eval %q requires title and sample", e.ID)
		}
		if strings.TrimSpace(e.AgentPrompt) == "" || strings.TrimSpace(e.Expected) == "" {
			return Suite{}, fmt.Errorf("eval %q requires agent_prompt and expected", e.ID)
		}
	}
	return suite, nil
}

type Provider struct{ ID, BaseURL, APIKey, Model string }
type Environment struct{ Agent, Judge Provider }

func LoadEnvironment(dir string) (Environment, error) {
	_ = godotenv.Load(filepath.Join(dir, ".env"))
	env := Environment{
		Agent: providerFrom("AGENT", "llama-hip"),
		Judge: providerFrom("JUDGE", "external-judge"),
	}
	if err := validateProvider("AGENT", env.Agent); err != nil {
		return Environment{}, err
	}
	if err := validateProvider("JUDGE", env.Judge); err != nil {
		return Environment{}, err
	}
	return env, nil
}

func providerFrom(prefix, fallbackID string) Provider {
	value := func(name, fallback string) string {
		if v := strings.TrimSpace(os.Getenv(prefix + "_" + name)); v != "" {
			return v
		}
		return fallback
	}
	return Provider{ID: value("PROVIDER_ID", fallbackID), BaseURL: value("BASE_URL", ""), APIKey: value("API_KEY", ""), Model: value("MODEL", "")}
}

func validateProvider(name string, p Provider) error {
	for label, value := range map[string]string{"PROVIDER_ID": p.ID, "BASE_URL": p.BaseURL, "API_KEY": p.APIKey, "MODEL": p.Model} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("%s_%s is required in .env", name, label)
		}
	}
	return nil
}
