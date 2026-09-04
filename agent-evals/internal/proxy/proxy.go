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

const MinFreshPromptTokens = 16

type GenerationSample struct {
	At        time.Time
	PerSecond float64
}

type Metrics struct {
	PromptTokens       float64
	PromptCachedTokens float64
	CacheRatio         float64
	PromptPerSecond    float64
	PromptMeasured     bool
	GenerationSamples  []GenerationSample
	UpdatedAt          time.Time
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
	if patched, isInference := injectProgress(body); isInference {
		body = patched
		s.beginInference()
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
	for key, values := range response.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	flush, _ := w.(http.Flusher)
	buffer := make([]byte, 4096)
	pending := ""
	for {
		count, readErr := response.Body.Read(buffer)
		if count > 0 {
			chunk := buffer[:count]
			_, _ = w.Write(chunk)
			if flush != nil {
				flush.Flush()
			}
			pending += string(chunk)
			for {
				newline := strings.IndexByte(pending, byte(10))
				if newline < 0 {
					break
				}
				s.observeLine(strings.TrimSpace(pending[:newline]))
				pending = pending[newline+1:]
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
	if _, hasMessages := request["messages"]; !hasMessages {
		return body, false
	}
	request["return_progress"] = true
	patched, err := json.Marshal(request)
	return patched, err == nil
}

func (s *Server) beginInference() {
	s.mu.Lock()
	s.metrics.PromptTokens = 0
	s.metrics.PromptCachedTokens = 0
	s.metrics.CacheRatio = 0
	s.metrics.PromptPerSecond = 0
	s.metrics.PromptMeasured = false
	s.metrics.UpdatedAt = time.Now()
	snapshot := s.snapshotLocked()
	s.mu.Unlock()
	s.notify(snapshot)
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
		total, hasTotal := number(progress["total"])
		cached, hasCached := number(progress["cache"])
		processed, hasProcessed := number(progress["processed"])
		millis, hasMillis := number(progress["time_ms"])
		if hasTotal && hasCached && total > 0 {
			s.metrics.PromptCachedTokens = cached
			s.metrics.CacheRatio = min(cached/total, 1)
			changed = true
		}
		if hasProcessed {
			fresh := processed
			if hasCached {
				fresh = max(fresh-cached, 0)
			}
			s.metrics.PromptTokens = fresh
			if hasMillis && millis > 0 && fresh >= MinFreshPromptTokens {
				s.metrics.PromptPerSecond = fresh / (millis / 1000)
				s.metrics.PromptMeasured = true
			}
			changed = true
		}
	}
	if value, ok := number(data["gen_second"]); ok && value >= 0 {
		s.metrics.GenerationSamples = append(s.metrics.GenerationSamples, GenerationSample{At: time.Now(), PerSecond: value})
		s.metrics.GenerationSamples = keepRecent(s.metrics.GenerationSamples, time.Now())
		changed = true
	}
	if timings, ok := data["timings"].(map[string]any); ok {
		// llama.cpp computes this from n_prompt_processed, which excludes cache.
		if value, ok := number(timings["prompt_per_second"]); ok && s.metrics.PromptTokens >= MinFreshPromptTokens {
			s.metrics.PromptPerSecond = value
			s.metrics.PromptMeasured = true
			changed = true
		}
		if value, ok := number(timings["predicted_per_second"]); ok && value >= 0 {
			s.metrics.GenerationSamples = append(s.metrics.GenerationSamples, GenerationSample{At: time.Now(), PerSecond: value})
			s.metrics.GenerationSamples = keepRecent(s.metrics.GenerationSamples, time.Now())
			changed = true
		}
	}
	if changed {
		s.metrics.UpdatedAt = time.Now()
		snapshot := s.snapshotLocked()
		s.mu.Unlock()
		s.notify(snapshot)
		return
	}
	s.mu.Unlock()
}

func (s *Server) snapshotLocked() Metrics {
	snapshot := s.metrics
	snapshot.GenerationSamples = append([]GenerationSample(nil), s.metrics.GenerationSamples...)
	return snapshot
}

func (s *Server) notify(metrics Metrics) {
	if s.observer != nil {
		s.observer(metrics)
	}
}
func keepRecent(samples []GenerationSample, now time.Time) []GenerationSample {
	cutoff := now.Add(-15 * time.Second)
	index := 0
	for index < len(samples) && samples[index].At.Before(cutoff) {
		index++
	}
	return append([]GenerationSample(nil), samples[index:]...)
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
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
