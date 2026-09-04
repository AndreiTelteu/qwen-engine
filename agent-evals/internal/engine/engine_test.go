package engine

import "testing"

func TestParserUsesLlamaTimingLinesAsTheOnlyMetricSource(t *testing.T) {
	parser := NewParser()
	lines := []string{
		"26.29.904.554 I slot launch_slot_: id  0 | task 4034 | processing task, is_child = 0",
		"26.29.904.700 D slot      update: id  0 | task 4034 | cached n_tokens = 1090, memory_seq_rm [1090, end)",
		"26.35.103.784 I slot print_timing: id  0 | task 4034 | n_gen =    146, tg =  48.20 t/s, tg_3s =  48.52 t/s",
		"26.39.175.819 I slot print_timing: id  0 | task 4034 | prompt eval time =    2190.39 ms /   910 tokens (    2.41 ms per token,   415.45 tokens per second)",
		"26.39.175.898 I slot print_timing: id  0 | task 4034 | draft acceptance = 0.64379 (  197 accepted /   306 generated), mean len =  2.29",
	}
	var metric Metrics
	for _, line := range lines {
		if update, changed, _ := parser.Parse(line); changed {
			metric = update
		}
	}
	if metric.TaskID != 4034 || metric.GenerationPerSecond != 48.52 {
		t.Fatalf("generation metric: %#v", metric)
	}
	if metric.PromptPerSecond != 415.45 || metric.FreshPromptTokens != 910 {
		t.Fatalf("prompt metric: %#v", metric)
	}
	if metric.CachedPromptTokens != 1090 {
		t.Fatalf("cached tokens: %#v", metric)
	}
	if metric.CacheRatio != 1090.0/2000.0 {
		t.Fatalf("cache ratio: %v", metric.CacheRatio)
	}
	if metric.DraftAcceptance != .64379 || metric.DraftAccepted != 197 || metric.DraftGenerated != 306 {
		t.Fatalf("draft metric: %#v", metric)
	}
}

func TestParserReadsLivePromptRateAndClearsGenerationWhenTaskEnds(t *testing.T) {
	parser := NewParser()
	_, _, _ = parser.Parse("26.56.633.838 I slot launch_slot_: id 0 | task 4355 | processing task, is_child = 0")
	metric, changed, _ := parser.Parse("26.58.775.010 I slot print_timing: id 0 | task 4355 | prompt processing, n_tokens = 2164, progress = 1.00, t = 3.03 s / 713.98 tokens per second")
	if !changed || metric.PromptPerSecond != 713.98 || metric.FreshPromptTokens != 2164 {
		t.Fatalf("live prompt metric: %#v", metric)
	}
	metric, changed, _ = parser.Parse("27.00.395.844 I slot release: id 0 | task 4355 | stop processing: n_tokens = 16138, truncated = 0")
	if !changed || metric.GenerationPerSecond != 0 {
		t.Fatalf("release metric: %#v", metric)
	}
}
