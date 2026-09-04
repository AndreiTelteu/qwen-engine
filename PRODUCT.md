# Product

<!-- impeccable:product-schema 1 -->

## Platform

adaptive

## Stack

Go 1.23, Bubble Tea v2, Lip Gloss, TOML definitions, Git worktrees, and Pi CLI.

## Users

A developer benchmarking a locally served coding model against repeatable Laravel and React repository tasks.

## Product Purpose

Agent Evals runs a coding agent against isolated working samples, verifies the result, and asks an independently configured judge to assess it. Success is a repeatable comparison based on real code changes, test output, and inference throughput.

## Positioning

It treats a local llama.cpp/HIP server as a manually controlled dependency while providing a keyboard-first evaluation cockpit that observes every agent turn and model-stream metric.

## Operating Context

Runs occur in WSL2. Working samples are Git submodules; each evaluation uses a disposable/reusable detached Git worktree. Pi operates non-interactively against a local OpenAI-compatible coding model and an optional external OpenAI-compatible judge.

## Capabilities and Constraints

Definitions live in a manually editable TOML file. The engine is never launched or stopped by Agent Evals. Before every run, the isolated worktree is reset, cleaned of untracked agent files, and garbage-collected. Secrets exist only in an ignored `.env`.

## Evidence on Hand

The local Qwen3.8-27B llama.cpp/HIP server serves at a user-configured OpenAI-compatible URL and exposes SSE prompt-progress and generation timing fields. No samples or external judge credentials have been supplied yet.

## Product Principles

- A benchmark must be reproducible before it is impressive.
- Agent mutations never touch the pristine sample checkout.
- Generation throughput is operationally prominent, without obscuring correctness.
- A model judge supplements automated checks; it never silently replaces them.
- Setup must stay legible enough to author a new evaluation by hand.
