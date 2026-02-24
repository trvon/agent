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
```

## Requirements

- YAMS daemon running (`yams serve` for MCP)
- LM Studio with models loaded (OpenAI-compatible API at `localhost:1234`)
- Python 3.11+
