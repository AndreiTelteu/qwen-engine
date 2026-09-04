package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Metrics struct {
	PromptTokens        float64
	PromptPerSecond     float64
	GenerationPerSecond float64
	UpdatedAt           time.Time
}

type Observer func(Metrics)

type Server struct {
	base     *url.URL
	server   *http.Server
	listener net.Listener
	observer Observer
	client   *http.Client
	mu       sync.Mutex
	metrics  Metrics
}

func Start(baseURL string, observer Observer) (*Server, error) {
	base, err := url.Parse(strings.TrimRight(baseURL, "/"))
	if err != nil {
		return nil, fmt.Errorf("parse agent URL: %w", err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("listen for metrics proxy: %w", err)
	}
	s := &Server{base: base, listener: listener, observer: observer, client: &http.Client{Timeout: 0}}
	s.server = &http.Server{Handler: http.HandlerFunc(s.handle), ReadHeaderTimeout: 15 * time.Second}
	go func() { _ = s.server.Serve(listener) }()
	return s, nil
}

func (s *Server) URL() string  { return "http://" + s.listener.Addr().String() + "/v1" }
func (s *Server) Close() error { return s.server.Close() }

func (s *Server) handle(w http.ResponseWriter, r *http.Request) {
	target := *s.base
	suffix := strings.TrimPrefix(r.URL.Path, "/v1")
	target.Path = strings.TrimRight(s.base.Path, "/") + suffix
	target.RawQuery = r.URL.RawQuery

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "cannot read request", http.StatusBadRequest)
		return
	}
	if patched, ok := injectProgress(body); ok {
		body = patched
	}
	request, err := http.NewRequestWithContext(r.Context(), r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		http.Error(w, "cannot create upstream request", http.StatusBadGateway)
		return
	}
	request.Header = r.Header.Clone()
	request.ContentLength = int64(len(body))
	response, err := s.client.Do(request)
	if err != nil {
		http.Error(w, "llama-hip is unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	for k, values := range response.Header {
		for _, v := range values {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(response.StatusCode)
	flush, _ := w.(http.Flusher)
	buffer := make([]byte, 4096)
	pending := ""
	for {
		n, readErr := response.Body.Read(buffer)
		if n > 0 {
			chunk := buffer[:n]
			_, _ = w.Write(chunk)
			if flush != nil {
				flush.Flush()
			}
			pending += string(chunk)
			for {
				at := strings.IndexByte(pending, byte(10))
				if at < 0 {
					break
				}
				s.observeLine(strings.TrimSpace(pending[:at]))
				pending = pending[at+1:]
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return
		}
	}
	if pending != "" {
		s.observeLine(strings.TrimSpace(pending))
	}
}

func injectProgress(body []byte) ([]byte, bool) {
	var request map[string]any
	if json.Unmarshal(body, &request) != nil {
		return body, false
	}
	request["return_progress"] = true
	patched, err := json.Marshal(request)
	return patched, err == nil
}

func (s *Server) observeLine(line string) {
	if !strings.HasPrefix(line, "data:") {
		return
	}
	raw := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if raw == "[DONE]" || raw == "" {
		return
	}
	var data map[string]any
	if json.Unmarshal([]byte(raw), &data) != nil {
		return
	}
	var changed bool
	s.mu.Lock()
	if progress, ok := data["prompt_progress"].(map[string]any); ok {
		processed, okP := number(progress["processed"])
		millis, okT := number(progress["time_ms"])
		if okP {
			s.metrics.PromptTokens = processed
			changed = true
		}
		if okP && okT && millis > 0 {
			s.metrics.PromptPerSecond = processed / (millis / 1000)
			changed = true
		}
	}
	if value, ok := number(data["gen_second"]); ok {
		s.metrics.GenerationPerSecond = value
		changed = true
	}
	if timings, ok := data["timings"].(map[string]any); ok {
		if value, ok := number(timings["prompt_per_second"]); ok {
			s.metrics.PromptPerSecond = value
			changed = true
		}
		if value, ok := number(timings["predicted_per_second"]); ok {
			s.metrics.GenerationPerSecond = value
			changed = true
		}
	}
	if changed {
		s.metrics.UpdatedAt = time.Now()
		snapshot := s.metrics
		s.mu.Unlock()
		if s.observer != nil {
			s.observer(snapshot)
		}
		return
	}
	s.mu.Unlock()
}

func number(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	case string:
		f, err := strconv.ParseFloat(v, 64)
		return f, err == nil
	default:
		return 0, false
	}
}
