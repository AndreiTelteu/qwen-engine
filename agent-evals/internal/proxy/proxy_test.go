package proxy

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestProxyReportsFreshPromptSpeedAndCacheRatio(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "return_progress") {
			t.Fatal("return_progress was not injected")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"prompt_progress":{"total":200,"cache":80,"processed":120,"time_ms":400}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"gen_second":55.5}`+"\n\n")
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()
	var latest Metrics
	gateway, err := Start(upstream.URL+"/v1", func(m Metrics) { latest = m })
	if err != nil {
		t.Fatal(err)
	}
	defer gateway.Close()
	response, err := http.Post(gateway.URL()+"/chat/completions", "application/json", strings.NewReader(`{"messages":[],"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if _, err := io.ReadAll(response.Body); err != nil {
		t.Fatal(err)
	}
	if latest.PromptTokens != 40 {
		t.Fatalf("fresh prompt tokens = %v", latest.PromptTokens)
	}
	if latest.PromptPerSecond != 100 {
		t.Fatalf("prompt speed = %v", latest.PromptPerSecond)
	}
	if latest.CacheRatio != .4 {
		t.Fatalf("cache ratio = %v", latest.CacheRatio)
	}
	if !latest.PromptMeasured {
		t.Fatal("fresh prompt should be measured")
	}
	if len(latest.GenerationSamples) != 1 || latest.GenerationSamples[0].PerSecond != 55.5 {
		t.Fatalf("generation samples = %#v", latest.GenerationSamples)
	}
}

func TestProxySuppressesPromptSpeedForCacheOnlyRequest(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"prompt_progress":{"total":100,"cache":99,"processed":100,"time_ms":1}}`+"\n\n")
		_, _ = fmt.Fprint(w, `data: {"timings":{"prompt_per_second":4000}}`+"\n\n")
	}))
	defer upstream.Close()
	var latest Metrics
	gateway, err := Start(upstream.URL+"/v1", func(m Metrics) { latest = m })
	if err != nil {
		t.Fatal(err)
	}
	defer gateway.Close()
	response, err := http.Post(gateway.URL()+"/chat/completions", "application/json", strings.NewReader(`{"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	_, _ = io.ReadAll(response.Body)
	if latest.PromptMeasured || latest.PromptPerSecond != 0 {
		t.Fatalf("cache hit polluted prompt speed: %#v", latest)
	}
	if latest.CacheRatio != .99 {
		t.Fatalf("cache ratio = %v", latest.CacheRatio)
	}
}
