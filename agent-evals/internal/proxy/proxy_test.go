package proxy

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestProxyInjectsProgressAndObservesServerSentMetrics(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "return_progress") {
			t.Fatal("return_progress was not injected")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(w, `data: {"prompt_progress":{"processed":120,"time_ms":400}}

`)
		_, _ = fmt.Fprint(w, `data: {"gen_second":55.5}

`)
		_, _ = fmt.Fprint(w, `data: [DONE]

`)
	}))
	defer upstream.Close()
	var latest Metrics
	gateway, err := Start(upstream.URL+"/v1", func(m Metrics) { latest = m })
	if err != nil {
		t.Fatal(err)
	}
	defer gateway.Close()
	response, err := http.Post(gateway.URL()+"/chat/completions", "application/json", strings.NewReader(`{"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if _, err := io.ReadAll(response.Body); err != nil {
		t.Fatal(err)
	}
	if latest.PromptPerSecond != 300 {
		t.Fatalf("prompt speed = %v", latest.PromptPerSecond)
	}
	if latest.GenerationPerSecond != 55.5 {
		t.Fatalf("generation speed = %v", latest.GenerationPerSecond)
	}
}
