# Design — Agent Evals TUI

## Intent

Midnight Mission Control is a compact operating surface for watching a coding agent alter a real repository. The interface privileges the current inference stream and recovery state over decoration.

## Visual language

- Backgrounds: deep midnight blue surfaces, not terminal-black emptiness.
- Structure: one restrained rounded hairline border per panel; no stacked cards.
- Primary ink: cool near-white. Secondary text is blue-tinted slate rather than neutral gray.
- Cyan identifies navigation, labels, and active operational state.
- Electric lime is reserved for the most recently logged **GENERATION TOK/S**, making the live speed signal unmistakable at a glance.
- Amber describes manual attention or the engine being offline; red names a failed run and its recovery message.
- Measurement values use tabular numeric rhythm where the terminal supports it; prose stays proportional.

## Layout

A full-width status band holds engine state, prompt throughput, and generation throughput. Below it, the evaluation queue stays on the left while the selected task and live phase use the dominant right pane. The lower log is chronological and deliberately quiet, so the highlighted speed metric remains the visual peak.

## Interaction

Keyboard is the primary control plane: `↑/↓` select, `Enter`/`r` execute, `a` run the suite, `Esc` cancels, and `e`/`n` move directly into TOML authoring. The application communicates engine-offline state before a run and never takes control of the engine process.
