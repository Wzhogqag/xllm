# Claude Code project guide

This repository contains the xLLM multi-model development branch and its experiment records.

## Project knowledge

Before non-trivial work, use the relevant repository skill under `.claude/skills/`:

- `xllm-arch`: architecture and code navigation
- `xllm-multimodel`: multi-model lifecycle and replica behavior
- `xllm-memory-mgmt`: XTensor/VMM memory management
- `xllm-scheduler`: scheduling and SLO behavior
- `xllm-conventions`: local coding conventions and known traps
- `xllm-debug`: build, runtime, and debugging workflow
- `xllm-feature-recipes`: implementation recipes

Durable project notes and experiment retrospectives are indexed in
`docs/claude-memory/README.md`.

## Verification rule

The knowledge documents contain point-in-time observations from the
`feat/final_multi_model` development line. Before changing code or relying on a
file path, line number, flag, or runtime assumption, verify it against the
current checkout.

## Local-only state

Do not assume that device IDs, model paths, ports, container isolation, or
process-management commands recorded in the notes apply on another machine.
Inspect the current environment first. Machine-local Claude settings and the
external `$HOME/.claude/projects/.../memory` directory are not repository
state.
