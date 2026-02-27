# Dynamic Context Scaffold (DCS)

YAMS-backed retrieval harness for small LLM agents.

**Thesis**: Scaffold beats scale — small models (1-4B) with intelligent, iterative retrieval
can approach large model performance on knowledge-intensive tasks.

## Architecture

```
task → decompose → multi-hop retrieve (YAMS MCP) → budget-aware assemble → execute (LM Studio) → self-critique → adjust → iterate
```

## Quick Start

```bash
uv sync
dcs run "Explain how YAMS hybrid search works"
dcs eval eval/tasks/coding/
dcs compare --models nemotron-nano,qwen3-4b
# retrieval-only (no generation)
uv run python benchmarks/retrieval_benchmark.py --task-dir eval/tasks --out results/retrieval_baseline.json
```

## Current Defaults

- Executor: `qwen/qwen3-4b-thinking-2507`
- Critic: `mistralai/ministral-3-14b-reasoning`
- No-ground-truth mode: enabled
- DSPy-first faithfulness extractor: enabled (falls back to deterministic parser)

## Context Profiles

- `--context-profile auto` (default): uses large profile when detected model context window is high.
- `--context-profile standard`: keeps baseline budgets.
- `--context-profile large`: forces larger budgets for retrieval/system/output/codemap.

At run start, DCS prints `requested` vs `actual` context window for executor/critic.

## Requirements

- YAMS daemon running (`yams serve` for MCP)
- LM Studio with models loaded (OpenAI-compatible API at `localhost:1234`)
- Python 3.11+

## Git hooks (optional)

This submodule includes repo-local hooks to keep diffs clean:

- `pre-commit`: `ruff` lint + auto-fix + format on staged Python files
- `pre-push`: `pytest` suite

Enable once (run inside `external/agent/`):

```bash
git config core.hooksPath .githooks
```
