# DCS Agent Notes (external/agent)

This submodule is accuracy-first. Latency is a secondary concern; long-running
benchmarks are expected when measuring retrieval quality.

## Priorities
- Optimize for correctness and coverage, not speed.
- Prefer robust retrieval (path-hinted grep, local file context) over model guesses.
- Use evaluation suites to compare models and settings.
- Treat smoke tests as acceptance gates before commits.

## Runtime Conventions
- Use `uv` for all Python runs.
- Default LM Studio endpoint: `http://localhost:1234/v1`.
- Prefer longer timeouts for benchmark runs.

## Benchmarking
- Use `benchmarks/coverage_benchmark.py` for multi-task evaluation.
- Record source coverage and quality metrics.
- Favor wide task coverage over short, single-task checks.

## Hygiene
- Do not commit unless explicitly asked.
- Keep outputs concise and reproducible.
