package ui

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/andreitelteu/qwen-engine/agent-evals/internal/config"
	"github.com/andreitelteu/qwen-engine/agent-evals/internal/proxy"
	"github.com/andreitelteu/qwen-engine/agent-evals/internal/runner"
)

var (
	ink         = lipgloss.Color("#E8EDF7")
	muted       = lipgloss.Color("#72829E")
	panel       = lipgloss.Color("#101827")
	line        = lipgloss.Color("#23314A")
	cyan        = lipgloss.Color("#48D6E5")
	lime        = lipgloss.Color("#B7F34B")
	amber       = lipgloss.Color("#FFB454")
	red         = lipgloss.Color("#FF7189")
	titleStyle  = lipgloss.NewStyle().Foreground(ink).Bold(true)
	dimStyle    = lipgloss.NewStyle().Foreground(muted)
	cyanStyle   = lipgloss.NewStyle().Foreground(cyan).Bold(true)
	hotStyle    = lipgloss.NewStyle().Foreground(lime).Bold(true)
	borderStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(line).Padding(0, 1)
)

type model struct {
	root          string
	configPath    string
	suite         config.Suite
	environment   config.Environment
	selected      int
	width, height int
	events        chan runner.Event
	logs          []string
	phase         string
	active        bool
	cancel        context.CancelFunc
	metrics       proxy.Metrics
	engineOnline  bool
	err           string
}

type engineStatus bool
type editorDone struct{ err error }

func New(root string, suite config.Suite, environment config.Environment) tea.Model {
	return model{root: root, configPath: filepath.Join(root, "evals.toml"), suite: suite, environment: environment, events: make(chan runner.Event, 256), logs: []string{"Ready. Engine is manually controlled."}}
}

func (m model) Init() tea.Cmd {
	return tea.Batch(waitEvent(m.events), checkEngine(m.environment.Agent.BaseURL), tick())
}

func (m model) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := message.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case tea.KeyPressMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		case "up", "k":
			if !m.active && m.selected > 0 {
				m.selected--
			}
		case "down", "j":
			if !m.active && m.selected < len(m.suite.Eval)-1 {
				m.selected++
			}
		case "enter", "r":
			if !m.active {
				return m.start([]config.Evaluation{m.suite.Eval[m.selected]})
			}
		case "a":
			if !m.active {
				return m.start(enabled(m.suite.Eval))
			}
		case "esc":
			if m.cancel != nil {
				m.cancel()
				m.addLog("Cancellation requested — waiting for Pi to exit.")
			}
		case "e":
			if !m.active {
				return m, m.editConfig()
			}
		case "n":
			if !m.active {
				return m, m.addTemplateAndEdit()
			}
		}
	case runner.Event:
		if msg.Metrics != nil {
			m.metrics = *msg.Metrics
		}
		switch msg.Kind {
		case "phase":
			m.phase = msg.Detail
			m.addLog(msg.Detail)
		case "log":
			m.addLog(msg.Detail)
		case "failed":
			m.active = false
			m.cancel = nil
			m.err = msg.Detail
			m.phase = "Run failed"
			m.addLog("FAILED · " + msg.Detail)
		case "complete":
			m.active = false
			m.cancel = nil
			m.phase = "Completed"
			m.addLog(msg.Detail)
		}
		return m, waitEvent(m.events)
	case engineStatus:
		m.engineOnline = bool(msg)
	case time.Time:
		return m, tea.Batch(checkEngine(m.environment.Agent.BaseURL), tick())
	case editorDone:
		if msg.err != nil {
			m.err = "editor: " + msg.err.Error()
			return m, nil
		}
		suite, err := config.Load(m.configPath)
		if err != nil {
			m.err = err.Error()
			return m, nil
		}
		m.suite, m.err = suite, ""
		if m.selected >= len(m.suite.Eval) {
			m.selected = len(m.suite.Eval) - 1
		}
		m.addLog("Reloaded evals.toml")
	}
	return m, nil
}

func (m model) start(evaluations []config.Evaluation) (tea.Model, tea.Cmd) {
	if len(evaluations) == 0 {
		m.err = "No enabled evaluations."
		return m, nil
	}
	m.active, m.err, m.phase, m.metrics = true, "", "Queued", proxy.Metrics{}
	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m.addLog(fmt.Sprintf("Queued %d evaluation(s).", len(evaluations)))
	go func() {
		for _, evaluation := range evaluations {
			if ctx.Err() != nil {
				m.events <- runner.Event{Kind: "failed", EvalID: evaluation.ID, Detail: "Run cancelled"}
				return
			}
			r := runner.Runner{Root: m.root, Env: m.environment, Emit: func(event runner.Event) { m.events <- event }}
			runContext, timeoutCancel := withTimeout(ctx, evaluation.Timeout)
			finished := r.Run(runContext, evaluation)
			timeoutCancel()
			if !finished {
				return
			}
			if ctx.Err() != nil {
				return
			}
		}
	}()
	return m, nil
}

func (m model) View() tea.View {
	if m.width == 0 {
		view := tea.NewView("Loading Agent Evals…")
		view.AltScreen = true
		return view
	}
	header := m.header()
	content := lipgloss.JoinHorizontal(lipgloss.Top, m.listPanel(), m.detailPanel())
	logPanel := m.logPanel()
	footer := dimStyle.Render("  ↑/↓ select  •  enter/r run  •  a suite  •  e edit TOML  •  n new template  •  esc cancel  •  q quit")
	view := tea.NewView(lipgloss.JoinVertical(lipgloss.Left, header, "", content, "", logPanel, "", footer))
	view.AltScreen = true
	return view
}

func (m model) header() string {
	status := hotStyle.Render("ENGINE ONLINE")
	if !m.engineOnline {
		status = lipgloss.NewStyle().Foreground(amber).Bold(true).Render("ENGINE OFFLINE")
	}
	pp := dimStyle.Render(fmt.Sprintf("PROMPT  %6.1f tok/s", m.metrics.PromptPerSecond))
	generation := hotStyle.Render(fmt.Sprintf("GENERATION  %6.1f TOK/S", m.metrics.GenerationPerSecond))
	left := titleStyle.Render("AGENT EVALS") + dimStyle.Render("  /  mission control")
	right := status + "   " + pp + "   " + generation
	gap := m.width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 2 {
		gap = 2
	}
	return lipgloss.NewStyle().Foreground(panel).Background(panel).Padding(0, 1).Render(left + strings.Repeat(" ", gap) + right)
}

func (m model) listPanel() string {
	width := max(30, m.width*38/100)
	var rows []string
	for i, e := range m.suite.Eval {
		status := "ready"
		color := muted
		if !e.IsEnabled() {
			status, color = "disabled", amber
		}
		row := fmt.Sprintf("%s\n%s  %s", titleStyle.Render(e.Title), lipgloss.NewStyle().Foreground(color).Render(status), dimStyle.Render(e.ID))
		if i == m.selected {
			row = lipgloss.NewStyle().Foreground(ink).Background(lipgloss.Color("#17253A")).Padding(0, 1).Width(width - 4).Render("› " + row)
		}
		rows = append(rows, row)
	}
	body := strings.Join(rows, "\n\n")
	return borderStyle.Width(width - 2).Render(cyanStyle.Render("EVALUATIONS") + "\n\n" + body)
}

func (m model) detailPanel() string {
	width := max(44, m.width*62/100-2)
	e := m.suite.Eval[m.selected]
	phase := m.phase
	if phase == "" {
		phase = "Standing by"
	}
	state := cyanStyle.Render("RUNNING")
	if !m.active {
		state = dimStyle.Render("READY")
	}
	if m.err != "" {
		state = lipgloss.NewStyle().Foreground(red).Bold(true).Render("ATTENTION")
	}
	lines := []string{
		fmt.Sprintf("%s  %s", titleStyle.Render(e.Title), state),
		dimStyle.Render("sample  /  " + e.Sample + "    timeout  /  " + e.Timeout),
		"",
		cyanStyle.Render("CURRENT PHASE"),
		phase,
		"",
		cyanStyle.Render("AGENT BRIEF"),
		wrap(e.AgentPrompt, width-6),
		"",
		cyanStyle.Render("SUCCESS SIGNAL"),
		wrap(e.Expected, width-6),
	}
	if m.err != "" {
		lines = append(lines, "", lipgloss.NewStyle().Foreground(red).Render("Recovery: "+m.err))
	}
	return borderStyle.Width(width - 2).Render(strings.Join(lines, "\n"))
}

func (m model) logPanel() string {
	width := max(76, m.width-2)
	logs := m.logs
	maxLines := max(4, m.height-25)
	if len(logs) > maxLines {
		logs = logs[len(logs)-maxLines:]
	}
	return borderStyle.Width(width - 2).Render(cyanStyle.Render("LIVE RUN LOG") + "\n" + strings.Join(logs, "\n"))
}

func (m *model) addLog(line string) {
	if strings.TrimSpace(line) == "" {
		return
	}
	m.logs = append(m.logs, dimStyle.Render(time.Now().Format("15:04:05"))+"  "+line)
	if len(m.logs) > 150 {
		m.logs = m.logs[len(m.logs)-150:]
	}
}

func (m model) editConfig() tea.Cmd {
	editor := os.Getenv("EDITOR")
	if editor == "" {
		editor = "nano"
	}
	parts := strings.Fields(editor)
	command := exec.Command(parts[0], append(parts[1:], m.configPath)...)
	return tea.ExecProcess(command, func(err error) tea.Msg { return editorDone{err} })
}

func (m model) addTemplateAndEdit() tea.Cmd {
	id := fmt.Sprintf("new-eval-%d", time.Now().Unix())
	template := fmt.Sprintf("\n[[eval]]\nid = %q\ntitle = %q\nsample = %q\ntimeout = %q\n\nagent_prompt = \"\"\"\nDescribe the coding task here.\n\"\"\"\n\nexpected = \"\"\"\nDescribe the observable successful outcome here.\n\"\"\"\n\nsetup = []\nverify = []\n\njudge_prompt = \"\"\"\nAssess correctness, tests, and scope discipline.\n\"\"\"\n", id, "New evaluation", "your-sample", "15m")
	file, err := os.OpenFile(m.configPath, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return func() tea.Msg { return editorDone{err} }
	}
	_, err = file.WriteString(template)
	_ = file.Close()
	if err != nil {
		return func() tea.Msg { return editorDone{err} }
	}
	return m.editConfig()
}

func enabled(all []config.Evaluation) []config.Evaluation {
	var result []config.Evaluation
	for _, e := range all {
		if e.IsEnabled() {
			result = append(result, e)
		}
	}
	return result
}
func waitEvent(events <-chan runner.Event) tea.Cmd { return func() tea.Msg { return <-events } }
func tick() tea.Cmd                                { return tea.Tick(2*time.Second, func(t time.Time) tea.Msg { return t }) }
func checkEngine(base string) tea.Cmd {
	return func() tea.Msg { return engineStatus(engineHealth(base)) }
}
func engineHealth(base string) bool {
	target, err := url.Parse(base)
	if err != nil {
		return false
	}
	target.Path = "/health"
	target.RawQuery = ""
	response, err := (&http.Client{Timeout: time.Second}).Get(target.String())
	if err != nil {
		return false
	}
	defer response.Body.Close()
	return response.StatusCode >= 200 && response.StatusCode < 300
}
func withTimeout(parent context.Context, raw string) (context.Context, context.CancelFunc) {
	duration, err := time.ParseDuration(raw)
	if err != nil || duration <= 0 {
		return parent, func() {}
	}
	return context.WithTimeout(parent, duration)
}
func wrap(text string, width int) string {
	words := strings.Fields(text)
	var lines []string
	var line string
	for _, word := range words {
		if lipgloss.Width(line)+len(word)+1 > width && line != "" {
			lines = append(lines, line)
			line = word
		} else if line == "" {
			line = word
		} else {
			line += " " + word
		}
	}
	if line != "" {
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
