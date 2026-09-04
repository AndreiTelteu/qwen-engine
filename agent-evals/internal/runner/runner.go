package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/andreitelteu/qwen-engine/agent-evals/internal/config"
)

type Event struct {
	Kind   string
	EvalID string
	Title  string
	Detail string
}

type Runner struct {
	Root string
	Env  config.Environment
	Emit func(Event)

	previewMu sync.Mutex
	previews  map[string]string
}

func (r *Runner) Run(ctx context.Context, evaluation config.Evaluation) bool {
	r.previewMu.Lock()
	r.previews = make(map[string]string)
	r.previewMu.Unlock()
	emit := func(kind, detail string) {
		r.Emit(Event{Kind: kind, EvalID: evaluation.ID, Title: evaluation.Title, Detail: detail})
	}
	emit("phase", "Checking manually started llama-hip engine")
	if err := health(r.Env.Agent.BaseURL); err != nil {
		emit("failed", err.Error())
		return false
	}

	workspace, err := r.prepare(ctx, evaluation, emit)
	if err != nil {
		emit("failed", err.Error())
		return false
	}
	emit("phase", "Running setup commands")
	if err := r.commands(ctx, workspace, evaluation.Setup, "setup", emit); err != nil {
		emit("failed", err.Error())
		return false
	}

	emit("phase", "Running coding agent; telemetry comes from managed llama-hip logs")
	agentOutput, err := r.runPi(ctx, workspace, evaluation.ID, "agent", r.Env.Agent, evaluation.AgentPrompt, true)
	if err != nil {
		emit("failed", fmt.Sprintf("agent: %v", err))
		return false
	}
	if err := r.writeResult(evaluation.ID, "agent.jsonl", agentOutput); err != nil {
		emit("failed", err.Error())
		return false
	}

	emit("phase", "Collecting diff and running verification")
	diff, err := r.capture(ctx, workspace, "git", "diff", "--binary")
	if err != nil {
		emit("failed", err.Error())
		return false
	}
	_ = r.writeResult(evaluation.ID, "changes.diff", diff)
	verifyOutput, verifyErr := r.commandsOutput(ctx, workspace, evaluation.Verify, "verify", emit)
	_ = r.writeResult(evaluation.ID, "verify.log", verifyOutput)
	if verifyErr != nil {
		emit("log", "Verification failed; judge will receive the failure output.")
	}

	emit("phase", "Running independent judge")
	judgeTask := judgePrompt(evaluation, diff, verifyOutput, verifyErr)
	judgeOutput, err := r.runPi(ctx, workspace, evaluation.ID, "judge", r.Env.Judge, judgeTask, false)
	if err != nil {
		emit("failed", fmt.Sprintf("judge: %v", err))
		return false
	}
	if err := r.writeResult(evaluation.ID, "judge.jsonl", judgeOutput); err != nil {
		emit("failed", err.Error())
		return false
	}
	_ = r.writeResult(evaluation.ID, "summary.md", summary(evaluation, verifyErr, diff, judgeOutput))
	emit("complete", "Run complete — diff, checks, transcript, and judge verdict saved.")
	return true
}

func (r *Runner) prepare(ctx context.Context, evaluation config.Evaluation, emit func(string, string)) (string, error) {
	source := filepath.Join(r.Root, "samples", evaluation.Sample)
	if _, err := os.Stat(filepath.Join(source, ".git")); err != nil {
		return "", fmt.Errorf("sample %q is not a Git checkout at %s", evaluation.Sample, source)
	}
	workspace := filepath.Join(r.Root, "runs", evaluation.ID)
	if _, err := os.Stat(filepath.Join(workspace, ".git")); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(workspace), 0o755); err != nil {
			return "", err
		}
		emit("log", "Creating detached worktree from pristine sample")
		if _, err := r.capture(ctx, source, "git", "worktree", "add", "--detach", "--force", workspace, "HEAD"); err != nil {
			return "", fmt.Errorf("create run worktree: %w", err)
		}
	}
	emit("log", "Resetting tracked files, untracked changes, and Git objects")
	for _, args := range [][]string{{"reset", "--hard"}, {"clean", "-ffd"}, {"gc", "--prune=now"}} {
		if _, err := r.capture(ctx, workspace, "git", args...); err != nil {
			return "", fmt.Errorf("prepare workspace: %w", err)
		}
	}
	return workspace, nil
}

func (r *Runner) commands(ctx context.Context, workspace string, commands []string, phase string, emit func(string, string)) error {
	_, err := r.commandsOutput(ctx, workspace, commands, phase, emit)
	return err
}

func (r *Runner) commandsOutput(ctx context.Context, workspace string, commands []string, phase string, emit func(string, string)) (string, error) {
	var output strings.Builder
	for _, command := range commands {
		emit("log", fmt.Sprintf("%s › %s", phase, command))
		text, err := r.capture(ctx, workspace, "bash", "-lc", command)
		output.WriteString("$ " + command + "\n" + text + "\n")
		if err != nil {
			return output.String(), fmt.Errorf("%s command failed: %w", phase, err)
		}
	}
	return output.String(), nil
}

func (r *Runner) runPi(ctx context.Context, workspace, evalID, stage string, provider config.Provider, prompt string, tools bool) (string, error) {
	home := filepath.Join(r.Root, "results", evalID, "pi-"+stage)
	if err := os.MkdirAll(home, 0o755); err != nil {
		return "", err
	}
	if err := writeModels(home, provider); err != nil {
		return "", err
	}
	args := []string{"--mode", "json", "--print", "--no-session", "--no-extensions", "--no-skills", "--provider", provider.ID, "--model", provider.Model}
	if tools {
		args = append(args, "--tools", "read,write,edit,bash,grep,find,ls")
	} else {
		args = append(args, "--no-tools")
	}
	args = append(args, prompt)
	command := exec.CommandContext(ctx, "pi", args...)
	command.Dir = workspace
	command.Env = append(os.Environ(), "PI_CODING_AGENT_DIR="+home, "PI_OFFLINE=1", "AGENT_EVAL_PROVIDER_API_KEY="+provider.APIKey)
	stdout, err := command.StdoutPipe()
	if err != nil {
		return "", err
	}
	stderr, err := command.StderrPipe()
	if err != nil {
		return "", err
	}
	if err := command.Start(); err != nil {
		return "", fmt.Errorf("start pi: %w", err)
	}
	var output strings.Builder
	var mu sync.Mutex
	consume := func(reader io.Reader, parse bool) {
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 16*1024), 2*1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			mu.Lock()
			output.WriteString(line + "\n")
			mu.Unlock()
			if parse {
				r.emitPiEvent(evalID, line)
			} else {
				r.Emit(Event{Kind: "log", EvalID: evalID, Detail: "pi: " + line})
			}
		}
	}
	var group sync.WaitGroup
	group.Add(2)
	go func() { defer group.Done(); consume(stdout, true) }()
	go func() { defer group.Done(); consume(stderr, false) }()
	err = command.Wait()
	group.Wait()
	if err != nil {
		return output.String(), err
	}
	return output.String(), nil
}

func (r *Runner) emitPiEvent(evalID, line string) {
	var event struct {
		Type                  string          `json:"type"`
		ToolName              string          `json:"toolName"`
		IsError               bool            `json:"isError"`
		Args                  json.RawMessage `json:"args"`
		Result                json.RawMessage `json:"result"`
		AssistantMessageEvent struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
		} `json:"assistantMessageEvent"`
	}
	if json.Unmarshal([]byte(line), &event) != nil {
		return
	}
	switch event.Type {
	case "tool_execution_start":
		detail := "tool › " + event.ToolName
		if args := compactJSON(event.Args, 120); args != "" {
			detail += " · " + args
		}
		r.Emit(Event{Kind: "tool", EvalID: evalID, Detail: detail})
	case "tool_execution_end":
		detail := "tool ✓ " + event.ToolName
		if event.IsError {
			detail = "tool ! " + event.ToolName
		}
		if result := compactJSON(event.Result, 100); result != "" {
			detail += " · " + result
		}
		r.Emit(Event{Kind: "tool", EvalID: evalID, Detail: detail})
	case "turn_start":
		r.resetPreviews(evalID)
		r.Emit(Event{Kind: "turn", EvalID: evalID, Detail: "model turn started"})
	case "turn_end":
		r.Emit(Event{Kind: "log", EvalID: evalID, Detail: "model turn completed"})
	case "message_update":
		switch event.AssistantMessageEvent.Type {
		case "text_delta":
			r.appendPreview(evalID, "answer", event.AssistantMessageEvent.Delta)
		default:
			kind := strings.ToLower(event.AssistantMessageEvent.Type)
			if strings.Contains(kind, "think") || strings.Contains(kind, "reason") {
				r.appendPreview(evalID, "thinking", event.AssistantMessageEvent.Delta)
			}
		}
	}
}

func (r *Runner) resetPreviews(evalID string) {
	r.previewMu.Lock()
	defer r.previewMu.Unlock()
	if r.previews == nil {
		r.previews = make(map[string]string)
	}
	delete(r.previews, evalID+":answer")
	delete(r.previews, evalID+":thinking")
}

func (r *Runner) appendPreview(evalID, kind, delta string) {
	if strings.TrimSpace(delta) == "" {
		return
	}
	key := evalID + ":" + kind
	r.previewMu.Lock()
	if r.previews == nil {
		r.previews = make(map[string]string)
	}
	previous := r.previews[key]
	current := tailPreviewText(previous+delta, 180)
	if current == previous {
		r.previewMu.Unlock()
		return
	}
	r.previews[key] = current
	r.previewMu.Unlock()
	r.Emit(Event{Kind: kind + "_preview", EvalID: evalID, Detail: current})
}

func compactJSON(raw json.RawMessage, limit int) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return previewText(string(raw), limit)
	}
	compact, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return previewText(string(compact), limit)
}

func previewText(text string, limit int) string {
	text = strings.Join(strings.Fields(text), " ")
	characters := []rune(text)
	if len(characters) <= limit {
		return text
	}
	return string(characters[:limit-1]) + "…"
}

func tailPreviewText(text string, limit int) string {
	text = strings.Join(strings.Fields(text), " ")
	characters := []rune(text)
	if len(characters) <= limit {
		return text
	}
	return "…" + string(characters[len(characters)-limit+1:])
}

func (r *Runner) capture(ctx context.Context, directory, binary string, args ...string) (string, error) {
	command := exec.CommandContext(ctx, binary, args...)
	command.Dir = directory
	raw, err := command.CombinedOutput()
	if err != nil {
		return string(raw), fmt.Errorf("%s %s: %w", binary, strings.Join(args, " "), err)
	}
	return string(raw), nil
}

func (r *Runner) writeResult(evalID, name, body string) error {
	dir := filepath.Join(r.Root, "results", evalID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644)
}

func health(baseURL string) error {
	target, err := url.Parse(baseURL)
	if err != nil {
		return err
	}
	target.Path = "/health"
	target.RawQuery = ""
	client := http.Client{Timeout: 3 * time.Second}
	response, err := client.Get(target.String())
	if err != nil {
		return fmt.Errorf("local engine is offline at %s; start ../start-llama-hip.sh first", target.Host)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return fmt.Errorf("local engine health returned %s", response.Status)
	}
	return nil
}

func writeModels(dir string, provider config.Provider) error {
	document := map[string]any{"providers": map[string]any{provider.ID: map[string]any{
		"baseUrl": provider.BaseURL, "api": "openai-completions", "apiKey": "$AGENT_EVAL_PROVIDER_API_KEY",
		"compat": map[string]bool{"supportsDeveloperRole": false, "supportsReasoningEffort": false},
		"models": []map[string]any{{"id": provider.Model, "name": provider.Model, "reasoning": false, "input": []string{"text"}, "contextWindow": 131072, "maxTokens": 32768, "cost": map[string]int{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}},
	}}}
	raw, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "models.json"), raw, 0o600)
}

func judgePrompt(e config.Evaluation, diff, verify string, verifyErr error) string {
	outcome := "passed"
	if verifyErr != nil {
		outcome = "failed: " + verifyErr.Error()
	}
	return fmt.Sprintf("%s\n\nTASK:\n%s\n\nEXPECTED OUTCOME:\n%s\n\nVERIFICATION (%s):\n%s\n\nGIT DIFF:\n%s\n\nReturn a concise verdict. End with one line exactly in this format: SCORE: N/10.", e.JudgePrompt, e.AgentPrompt, e.Expected, outcome, verify, diff)
}

func summary(e config.Evaluation, verifyErr error, diff, judge string) string {
	verification := "passed"
	if verifyErr != nil {
		verification = "failed: " + verifyErr.Error()
	}
	return fmt.Sprintf("# %s\n\n- Evaluation: `%s`\n- Verification: **%s**\n- Diff bytes: %d\n\n## Judge transcript\n\n```jsonl\n%s```\n", e.Title, e.ID, verification, len(diff), judge)
}
