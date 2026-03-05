from __future__ import annotations

import pytest

from dcs.router import RoutingPolicy, TieredRouter
from dcs.types import (
    ContextBlock,
    Critique,
    IterationRecord,
    ModelConfig,
    PipelineConfig,
    PipelineResult,
)


def _mk_cfg(name: str) -> PipelineConfig:
    return PipelineConfig(
        executor_model=ModelConfig(name=name), critic_model=ModelConfig(name=name)
    )


def _mk_result(
    task: str,
    quality: float,
    output: str = "This output is sufficiently detailed and non-error for router acceptance.",
    source: str = "src/a.cpp",
) -> PipelineResult:
    crit = Critique(context_utilization=0.5, quality_score=quality)
    rec = IterationRecord(iteration=1, critique=crit)
    rec.context = ContextBlock(content="", sources=[source])
    res = PipelineResult(task=task, iterations=[rec], final_output=output, converged=True)
    return res


@pytest.mark.asyncio
async def test_router_keeps_base_when_acceptable() -> None:
    cfg = _mk_cfg("base")
    fallback = _mk_cfg("fallback")
    seq = [_mk_result("t", 0.8)]

    class StubPipeline:
        def __init__(self, _cfg):
            self._cfg = _cfg

        async def run(self, _task: str):
            return seq.pop(0)

    router = TieredRouter(
        base_config=cfg,
        fallback_configs=[fallback],
        policy=RoutingPolicy(quality_threshold=0.7),
        pipeline_factory=StubPipeline,  # type: ignore[arg-type]
    )
    out = await router.run("task")
    assert out.selected_tier == 0
    assert not out.escalated


@pytest.mark.asyncio
async def test_router_escalates_to_fallback() -> None:
    cfg = _mk_cfg("base")
    fallback = _mk_cfg("fallback")
    seq = [_mk_result("t", 0.3), _mk_result("t", 0.85)]

    class StubPipeline:
        def __init__(self, _cfg):
            self._cfg = _cfg

        async def run(self, _task: str):
            return seq.pop(0)

    router = TieredRouter(
        base_config=cfg,
        fallback_configs=[fallback],
        policy=RoutingPolicy(quality_threshold=0.7),
        pipeline_factory=StubPipeline,  # type: ignore[arg-type]
    )
    out = await router.run("task")
    assert out.selected_tier == 1
    assert out.escalated
    assert out.selected_result.final_critique is not None
    assert out.selected_result.final_critique.quality_score >= 0.8


@pytest.mark.asyncio
async def test_router_accepts_relaxed_grounded_base() -> None:
    cfg = _mk_cfg("base")
    fallback = _mk_cfg("fallback")
    seq = [
        _mk_result(
            "Describe MCP transport and tool flow",
            0.62,
            output=(
                "YAMS MCP uses stdio NDJSON/JSON-RPC transport and exposes tool handlers "
                "for search, grep, get and graph."
            ),
            source="src/mcp/stdio_transport.cpp",
        )
    ]

    class StubPipeline:
        def __init__(self, _cfg):
            self._cfg = _cfg

        async def run(self, _task: str):
            return seq.pop(0)

    router = TieredRouter(
        base_config=cfg,
        fallback_configs=[fallback],
        policy=RoutingPolicy(
            quality_threshold=0.7,
            relaxed_quality_floor=0.6,
            score_threshold=0.53,
            min_task_term_coverage=0.2,
            min_output_chars=40,
        ),
        pipeline_factory=StubPipeline,  # type: ignore[arg-type]
    )
    out = await router.run("task")
    assert out.selected_tier == 0
    assert not out.escalated


@pytest.mark.asyncio
async def test_router_rejects_low_task_coverage_even_if_quality_ok() -> None:
    cfg = _mk_cfg("base")
    fallback = _mk_cfg("fallback")
    seq = [
        _mk_result("Explain MCP stdio transport", quality=0.65, output="Generic summary text"),
        _mk_result("Explain MCP stdio transport", quality=0.85, output="Grounded MCP stdio answer"),
    ]

    class StubPipeline:
        def __init__(self, _cfg):
            self._cfg = _cfg

        async def run(self, _task: str):
            return seq.pop(0)

    router = TieredRouter(
        base_config=cfg,
        fallback_configs=[fallback],
        policy=RoutingPolicy(
            quality_threshold=0.7,
            relaxed_quality_floor=0.6,
            score_threshold=0.55,
            min_task_term_coverage=0.5,
            min_output_chars=10,
        ),
        pipeline_factory=StubPipeline,  # type: ignore[arg-type]
    )
    out = await router.run("task")
    assert out.selected_tier == 1
    assert out.escalated
