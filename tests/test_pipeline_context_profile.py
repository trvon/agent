from __future__ import annotations

from dcs.pipeline import DCSPipeline
from dcs.types import PipelineConfig


def test_apply_context_profile_large_updates_default_budgets() -> None:
    cfg = PipelineConfig()
    p = DCSPipeline(cfg)
    p._apply_context_profile(16384)

    assert cfg.context_budget == cfg.large_context_budget
    assert cfg.system_prompt_budget == cfg.large_system_prompt_budget
    assert cfg.output_reserve == cfg.large_output_reserve
    assert cfg.codemap_budget == cfg.large_codemap_budget


def test_apply_context_profile_standard_keeps_budgets() -> None:
    cfg = PipelineConfig(context_profile="standard")
    p = DCSPipeline(cfg)
    p._apply_context_profile(16384)

    assert cfg.context_budget == 2048
    assert cfg.system_prompt_budget == 512
    assert cfg.output_reserve == 1024
    assert cfg.codemap_budget == 256
