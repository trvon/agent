# Benchmarking Guide

This project is accuracy-first. Benchmarks should prioritize correctness and reproducibility over
speed.

## Principles

- Keep decomposition deterministic when measuring retrieval behavior.
- Separate policy-tuned experiments from generalization checks.
- Avoid stale checkpoint carryover when run settings change.
- Record enough metadata to replay results.

## Retrieval Benchmark (No Generation)

Use this to evaluate retrieval quality (hit/recall/MRR/noise/graph utility).

```bash
research-agent-benchmark-retrieval \
  --task-dir eval/tasks \
  --decompose-mode heuristic \
  --task-seeding=false \
  --out results/retrieval_baseline.json
```

Recommended settings:

- `--decompose-mode heuristic`: deterministic query planning.
- `--task-seeding=false`: disables hardcoded task-family seeds to reduce benchmark leakage.
- `--tag-match all`: require all tags when filtering.

If you explicitly benchmark model-driven decomposition, set:

```bash
research-agent-benchmark-retrieval --decompose-mode model --decomposer-temperature 0
```

## Coverage Benchmark (Full Pipeline)

Use this for end-to-end answer quality and routing behavior.

```bash
research-agent-benchmark-coverage \
  --task-dir eval/tasks \
  --models qwen3-4b \
  --task-seeding=false \
  --checkpoint results/coverage_checkpoint.json \
  --out results/coverage_results.json
```

Checkpoint behavior:

- Checkpoint keys include a config fingerprint (model + thresholds + filters + policy flags).
- This prevents accidental reuse of old task results under new settings.

## Scope and Paths

- Override retrieval scope with `YAMS_CWD=/path/to/repo` or `--yams-cwd`.
- Default scope resolves dynamically from the current checkout.

## Report Checklist

When publishing benchmark outputs, include:

- command line used
- models (executor/critic/fallback)
- decomposition mode and task-seeding mode
- task filters (`task_type`, `tags`, `tag_match`)
- output JSON path and checkpoint path
