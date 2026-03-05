# Dynamic Context Scaffold (DCS)

[![CI](https://github.com/trvon/agent/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/trvon/agent/actions/workflows/ci.yml)

YAMS-backed retrieval harness for small LLM agents.

Thesis: scaffold quality can close part of the gap between smaller models and larger models on
knowledge-intensive tasks.

## Architecture

```
task -> decompose -> retrieve (YAMS MCP) -> assemble -> execute (LM Studio) -> critique -> iterate
```

## Quick Start

```bash
uv sync
research-agent status
research-agent run "Explain how YAMS hybrid search works"

# Eval suite
research-agent eval --task-dir eval/tasks --type coding

# Scaffolded vs vanilla comparison
research-agent compare --task-dir eval/tasks
```

`dcs` remains available as a compatibility alias.

## Benchmarking

Use benchmark commands for reproducible measurement, not ad-hoc runs.

```bash
# Retrieval-only benchmark (deterministic decomposition by default)
research-agent-benchmark-retrieval \
  --task-dir eval/tasks \
  --decompose-mode heuristic \
  --task-seeding=false \
  --out results/retrieval_baseline.json

# Coverage benchmark (full pipeline)
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen3-4b \
  --task-seeding=false \
  --checkpoint results/coverage_checkpoint.json \
  --out results/coverage_results.json
```

Notes:
- Benchmark checkpoint keys now include a config fingerprint to avoid stale-result reuse.
- Set `YAMS_CWD` if you want to override repository scope.
- Use `--task-seeding=false` for fairer generalization measurements.

See `docs/benchmarking.md` for the full protocol.

## Context Profiles

- `--context-profile auto` (default): switches to larger budgets when model context is large.
- `--context-profile standard`: baseline budgets.
- `--context-profile large`: force larger retrieval/system/output/codemap budgets.

At run start, DCS prints requested vs actual context window for executor/critic.

## Requirements

- YAMS daemon available (`yams serve` for MCP)
- LM Studio with models loaded (OpenAI-compatible API at `http://localhost:1234/v1`)
- Python 3.11+

## Repo-Local Hooks (Optional)

- `pre-commit`: `ruff` lint/fix/format on staged Python files
- `pre-push`: `pytest -q`

Enable once (inside `external/agent/`):

```bash
git config core.hooksPath .githooks
```

## CI

GitHub Actions runs lint, format checks, tests with coverage, and package builds on push/PR:

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest -q --cov=dcs.cli --cov=dcs.router --cov=dcs.types --cov=eval.metrics --cov=eval.runner --cov-report=term-missing --cov-report=xml:coverage-core.xml --cov-fail-under=80`
- `uv run pytest -q --cov=dcs --cov=benchmarks --cov=eval --cov-report=term --cov-report=xml:coverage-full.xml` (snapshot, non-gating)
- `uv build`

Coverage XML artifacts are uploaded per Python version.
Current CI coverage floor: 80% for the core control-plane modules.

### Branch Protection (Recommended)

Require these status checks on `main`:

- `lint-test-build (3.11)`
- `lint-test-build (3.12)`
