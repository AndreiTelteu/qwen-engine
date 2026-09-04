package engine

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Metrics struct {
	TaskID              int
	PromptPerSecond     float64
	FreshPromptTokens   int
	CachedPromptTokens  int
	CacheRatio          float64
	GenerationPerSecond float64 // llama.cpp tg_3s: its real rolling three-second value
	DraftAcceptance     float64
	DraftAccepted       int
	DraftGenerated      int
	UpdatedAt           time.Time
}

type Event struct {
	Kind    string
	Detail  string
	Metrics *Metrics
}

type Manager struct {
	workspaceRoot string
	baseURL       string
	emit          func(Event)
	parser        *Parser
	mu            sync.Mutex
	command       *exec.Cmd
}

func New(workspaceRoot, baseURL string, emit func(Event)) *Manager {
	return &Manager{workspaceRoot: workspaceRoot, baseURL: baseURL, emit: emit, parser: NewParser()}
}

func (m *Manager) Start(ctx context.Context) {
	if Healthy(m.baseURL) {
		m.emit(Event{Kind: "external", Detail: "Engine already listens on the configured port; it is external, so its logs cannot be managed here."})
		return
	}
	endpoint, err := url.Parse(m.baseURL)
	if err != nil {
		m.emit(Event{Kind: "failed", Detail: "Invalid AGENT_BASE_URL: " + err.Error()})
		return
	}
	port := endpoint.Port()
	if port == "" {
		port = "8080"
	}
	script := filepath.Join(m.workspaceRoot, "start-llama-hip.sh")
	logDir := filepath.Join(m.workspaceRoot, "agent-evals", "results", "engine")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		m.emit(Event{Kind: "failed", Detail: err.Error()})
		return
	}
	logFile, err := os.Create(filepath.Join(logDir, "llama-"+time.Now().Format("20060102-150405")+".log"))
	if err != nil {
		m.emit(Event{Kind: "failed", Detail: err.Error()})
		return
	}
	defer logFile.Close()

	command := exec.CommandContext(ctx, script)
	command.Dir = m.workspaceRoot
	command.Env = append(os.Environ(), "PORT="+port, "VERBOSITY=4")
	stdout, err := command.StdoutPipe()
	if err != nil {
		m.emit(Event{Kind: "failed", Detail: err.Error()})
		return
	}
	stderr, err := command.StderrPipe()
	if err != nil {
		m.emit(Event{Kind: "failed", Detail: err.Error()})
		return
	}
	m.mu.Lock()
	m.command = command
	m.mu.Unlock()
	m.emit(Event{Kind: "starting", Detail: "Starting managed llama-hip; loading model into ROCm."})
	if err := command.Start(); err != nil {
		m.emit(Event{Kind: "failed", Detail: "Start llama-hip: " + err.Error()})
		return
	}

	var readers sync.WaitGroup
	consume := func(stream io.Reader) {
		defer readers.Done()
		scanner := bufio.NewScanner(stream)
		scanner.Buffer(make([]byte, 16*1024), 2*1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			_, _ = fmt.Fprintln(logFile, line)
			if metrics, update, note := m.parser.Parse(line); update {
				m.emit(Event{Kind: "metrics", Metrics: &metrics})
				if note != "" {
					m.emit(Event{Kind: "engine-log", Detail: note})
				}
			}
		}
	}
	readers.Add(2)
	go consume(stdout)
	go consume(stderr)
	err = command.Wait()
	readers.Wait()
	m.mu.Lock()
	m.command = nil
	m.mu.Unlock()
	if ctx.Err() != nil {
		m.emit(Event{Kind: "stopped", Detail: "Managed llama-hip stopped."})
		return
	}
	if err != nil {
		m.emit(Event{Kind: "failed", Detail: "llama-hip exited: " + err.Error()})
		return
	}
	m.emit(Event{Kind: "stopped", Detail: "Managed llama-hip exited."})
}

func (m *Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.command != nil && m.command.Process != nil {
		_ = m.command.Process.Signal(os.Interrupt)
	}
}

func Healthy(baseURL string) bool {
	target, err := url.Parse(baseURL)
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

type taskState struct{ cached, fresh int }
type Parser struct {
	tasks   map[int]taskState
	metrics Metrics
}

func NewParser() *Parser { return &Parser{tasks: make(map[int]taskState)} }

var (
	taskPattern        = regexp.MustCompile(`task\s+(-?\d+)`)
	cachedPattern      = regexp.MustCompile(`cached n_tokens =\s*(\d+)`)
	promptFinalPattern = regexp.MustCompile(`prompt eval time.*?/\s*(\d+) tokens .*?,\s*([0-9]+(?:\.[0-9]+)?) tokens per second`)
	promptLivePattern  = regexp.MustCompile(`prompt processing, n_tokens =\s*(\d+).*?/\s*([0-9]+(?:\.[0-9]+)?) tokens per second`)
	generationPattern  = regexp.MustCompile(`n_gen =\s*(\d+), tg =\s*([0-9]+(?:\.[0-9]+)?) t/s, tg_3s =\s*([0-9]+(?:\.[0-9]+)?) t/s`)
	acceptancePattern  = regexp.MustCompile(`draft acceptance =\s*([0-9]+(?:\.[0-9]+)?) \(\s*(\d+) accepted /\s*(\d+) generated\)`)
)

func (p *Parser) Parse(line string) (Metrics, bool, string) {
	task, hasTask := parseTask(line)
	now := time.Now()
	if hasTask && strings.Contains(line, "processing task") {
		p.metrics.TaskID = task
		return p.snapshot(now), true, fmt.Sprintf("engine task %d started", task)
	}
	if hasTask && cachedPattern.MatchString(line) {
		cached, _ := parseInt(cachedPattern.FindStringSubmatch(line)[1])
		state := p.tasks[task]
		state.cached = cached
		p.tasks[task] = state
		return p.snapshot(now), false, ""
	}
	if hasTask && (promptFinalPattern.MatchString(line) || promptLivePattern.MatchString(line)) {
		parts := promptFinalPattern.FindStringSubmatch(line)
		if len(parts) == 0 {
			parts = promptLivePattern.FindStringSubmatch(line)
		}
		tokens, _ := parseInt(parts[1])
		perSecond, _ := strconv.ParseFloat(parts[2], 64)
		state := p.tasks[task]
		state.fresh = tokens
		p.tasks[task] = state
		p.metrics.TaskID, p.metrics.FreshPromptTokens, p.metrics.CachedPromptTokens, p.metrics.PromptPerSecond = task, tokens, state.cached, perSecond
		total := tokens + state.cached
		if total > 0 {
			p.metrics.CacheRatio = float64(state.cached) / float64(total)
		}
		return p.snapshot(now), true, fmt.Sprintf("task %d · PP %.1f tok/s · CR %.0f%%", task, perSecond, p.metrics.CacheRatio*100)
	}
	if hasTask && generationPattern.MatchString(line) {
		parts := generationPattern.FindStringSubmatch(line)
		rolling, _ := strconv.ParseFloat(parts[3], 64)
		p.metrics.TaskID, p.metrics.GenerationPerSecond = task, rolling
		return p.snapshot(now), true, ""
	}
	if hasTask && acceptancePattern.MatchString(line) {
		parts := acceptancePattern.FindStringSubmatch(line)
		rate, _ := strconv.ParseFloat(parts[1], 64)
		accepted, _ := parseInt(parts[2])
		generated, _ := parseInt(parts[3])
		p.metrics.TaskID, p.metrics.DraftAcceptance, p.metrics.DraftAccepted, p.metrics.DraftGenerated = task, rate, accepted, generated
		return p.snapshot(now), true, fmt.Sprintf("task %d · MTP %.0f%% (%d/%d)", task, rate*100, accepted, generated)
	}
	if hasTask && strings.Contains(line, "stop processing") {
		p.metrics.TaskID, p.metrics.GenerationPerSecond = task, 0
		return p.snapshot(now), true, fmt.Sprintf("engine task %d completed", task)
	}
	return Metrics{}, false, ""
}

func (p *Parser) snapshot(now time.Time) Metrics {
	result := p.metrics
	result.UpdatedAt = now
	return result
}
func parseTask(line string) (int, bool) {
	parts := taskPattern.FindStringSubmatch(line)
	if len(parts) != 2 {
		return 0, false
	}
	task, err := parseInt(parts[1])
	return task, err == nil
}
func parseInt(value string) (int, error) { return strconv.Atoi(value) }
