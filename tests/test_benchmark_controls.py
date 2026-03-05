from __future__ import annotations

from benchmarks.coverage_benchmark import (
    _checkpoint_key,
)
from benchmarks.coverage_benchmark import (
    _filter_by_tags as coverage_filter_by_tags,
)
from benchmarks.retrieval_benchmark import (
    _filter_by_tags as retrieval_filter_by_tags,
)
from benchmarks.retrieval_benchmark import _heuristic_decompose
from dcs.types import EvalTask, TaskType


def _task(task_id: str, tags: list[str]) -> EvalTask:
    return EvalTask(id=task_id, task_type=TaskType.QA, description=f"task-{task_id}", tags=tags)


def test_retrieval_tag_filter_all_requires_subset() -> None:
    tasks = [_task("a", ["qa", "mcp"]), _task("b", ["qa"]), _task("c", ["mcp"])]
    out = retrieval_filter_by_tags(tasks, {"qa", "mcp"}, mode="all")
    assert [t.id for t in out] == ["a"]


def test_retrieval_tag_filter_any_accepts_intersection() -> None:
    tasks = [_task("a", ["qa", "mcp"]), _task("b", ["qa"]), _task("c", ["mcp"])]
    out = retrieval_filter_by_tags(tasks, {"mcp", "vector"}, mode="any")
    assert [t.id for t in out] == ["a", "c"]


def test_coverage_tag_filter_all_and_any_modes() -> None:
    tasks = [_task("a", ["qa", "mcp"]), _task("b", ["qa"]), _task("c", ["mcp"])]

    out_all = coverage_filter_by_tags(tasks, {"qa", "mcp"}, mode="all")
    out_any = coverage_filter_by_tags(tasks, {"qa", "mcp"}, mode="any")

    assert [t.id for t in out_all] == ["a"]
    assert [t.id for t in out_any] == ["a", "b", "c"]


def test_checkpoint_key_is_stable_for_same_payload() -> None:
    payload_a = {"executor": "qwen3-4b", "tags": ["mcp", "qa"], "quality_threshold": 0.7}
    payload_b = {"quality_threshold": 0.7, "tags": ["mcp", "qa"], "executor": "qwen3-4b"}

    key_a = _checkpoint_key("qwen3-4b", payload_a)
    key_b = _checkpoint_key("qwen3-4b", payload_b)
    assert key_a == key_b


def test_checkpoint_key_changes_when_payload_changes() -> None:
    payload_a = {"executor": "qwen3-4b", "quality_threshold": 0.7}
    payload_b = {"executor": "qwen3-4b", "quality_threshold": 0.8}

    key_a = _checkpoint_key("qwen3-4b", payload_a)
    key_b = _checkpoint_key("qwen3-4b", payload_b)
    assert key_a != key_b


def test_heuristic_decompose_is_deterministic_and_bounded() -> None:
    task = "How does ResourceGovernor canLoadModel work in include/yams/daemon/components/ResourceGovernor.h?"
    out1 = _heuristic_decompose(task, max_queries=4)
    out2 = _heuristic_decompose(task, max_queries=4)

    assert out1 == out2
    assert 1 <= len(out1) <= 4
