package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/andreitelteu/qwen-engine/agent-evals/internal/config"
	"github.com/andreitelteu/qwen-engine/agent-evals/internal/ui"
	tea "github.com/charmbracelet/bubbletea"
)

func main() {
	root, err := filepath.Abs(".")
	if err != nil {
		fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "evals.toml")); err != nil {
		fatal(fmt.Errorf("run this command from agent-evals: %w", err))
	}
	suite, err := config.Load(filepath.Join(root, "evals.toml"))
	if err != nil {
		fatal(err)
	}
	environment, err := config.LoadEnvironment(root)
	if err != nil {
		fatal(err)
	}
	program := tea.NewProgram(ui.New(root, suite, environment), tea.WithAltScreen())
	if _, err := program.Run(); err != nil {
		fatal(err)
	}
}

func fatal(err error) { fmt.Fprintln(os.Stderr, "agent-evals:", err); os.Exit(1) }
