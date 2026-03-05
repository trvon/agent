from __future__ import annotations

from pathlib import Path

import pytest

from dcs.types import (
    ComparisonResult,
    EvalResult,
    EvalTask,
    PipelineConfig,
    PipelineResult,
    TaskType,
)
from eval import runner as runner_mod
from eval.runner import EvalRunner, _as_task_type, run_comparison_report


def test_as_task_type_parsing() -> None:
    assert _as_task_type(TaskType.QA) == TaskType.QA
    assert _as_task_type(" QA ") == TaskType.QA
    assert _as_task_type("unknown") is None
    assert _as_task_type(123) is None


def test_load_tasks_filters_and_normalizes(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "good.yaml").write_text(
        """
id: qa-1
task_type: qa
description: hello
ground_truth:
  expected: hi
evaluation:
  pass_metric: quality_score
tags: [a, b]
""".strip(),
        encoding="utf-8",
    )
    (task_dir / "bad.yaml").write_text("[]", encoding="utf-8")
    (task_dir / "unknown_type.yaml").write_text("task_type: ???", encoding="utf-8")

    runner = EvalRunner(PipelineConfig(), task_dir=task_dir)
    all_tasks = runner.load_tasks(task_dir)
    qa_tasks = runner.load_tasks(task_dir, task_type=TaskType.QA)
    coding_tasks = runner.load_tasks(task_dir, task_type=TaskType.CODING)

    assert len(all_tasks) == 1
    assert all_tasks[0].id == "qa-1"
    assert all_tasks[0].tags == ["a", "b"]
    assert len(qa_tasks) == 1
    assert coding_tasks == []


def test_decide_pass_priority_paths() -> None:
    runner = EvalRunner(PipelineConfig(), task_dir=Path("."))
    task = EvalTask(id="t", task_type=TaskType.QA, description="x", evaluation={})

    assert runner._decide_pass(task, {"exact_match": 1.0})
    assert runner._decide_pass(task, {"contains_pattern": 1.0})
    assert runner._decide_pass(task, {"quality_score": 0.8})
    assert not runner._decide_pass(task, {"quality_score": 0.2})

    pass_metric_task = EvalTask(
        id="p",
        task_type=TaskType.QA,
        description="x",
        evaluation={"pass_metric": "custom", "pass_threshold": 0.6},
    )
    assert runner._decide_pass(pass_metric_task, {"custom": 0.7})
    assert not runner._decide_pass(pass_metric_task, {"custom": 0.5})


@pytest.mark.asyncio
async def test_run_task_scaffolded_and_vanilla(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"run": 0, "run_vanilla": 0}

    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        async def run(self, task: str) -> PipelineResult:
            calls["run"] += 1
            return PipelineResult(task=task, final_output="scaffolded")

        async def run_vanilla(self, task: str) -> PipelineResult:
            calls["run_vanilla"] += 1
            return PipelineResult(task=task, final_output="vanilla")

    monkeypatch.setattr(runner_mod, "DCSPipeline", FakePipeline)
    monkeypatch.setattr(runner_mod, "evaluate_task", lambda task, pr: {"quality_score": 1.0})

    runner = EvalRunner(PipelineConfig(), task_dir=Path("."))
    task = EvalTask(id="t", task_type=TaskType.QA, description="hello")

    r1 = await runner.run_task(task, scaffolded=True)
    r2 = await runner.run_task(task, scaffolded=False)

    assert r1.passed and r2.passed
    assert calls["run"] == 1
    assert calls["run_vanilla"] == 1


@pytest.mark.asyncio
async def test_run_task_handles_pipeline_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePipeline:
        def __init__(self, cfg):
            self.cfg = cfg

        async def run(self, task: str) -> PipelineResult:
            raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "DCSPipeline", FakePipeline)

    runner = EvalRunner(PipelineConfig(), task_dir=Path("."))
    task = EvalTask(id="t", task_type=TaskType.QA, description="hello")
    result = await runner.run_task(task, scaffolded=True)
    assert not result.passed
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_run_suite_and_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = EvalRunner(PipelineConfig(), task_dir=Path("."))

    async def fake_run_task(task: EvalTask, scaffolded: bool = True) -> EvalResult:
        return EvalResult(task_id=task.id, passed=scaffolded)

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    tasks = [EvalTask(id="a", task_type=TaskType.QA, description="A")]

    suite = await runner.run_suite(tasks, scaffolded=True)
    assert suite[0].passed

    a, b = await runner.run_comparison(tasks)
    assert a.scaffolded is True
    assert b.scaffolded is False
    assert a.model == runner.config.executor_model.name


@pytest.mark.asyncio
async def test_run_comparison_report_prints(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = EvalRunner(PipelineConfig(), task_dir=Path("."))
    printed: list[str] = []

    def fake_print_results(results, title):
        printed.append(title)

    async def fake_run_comparison(tasks):
        return (
            ComparisonResult(
                config_name="default",
                model="m",
                scaffolded=True,
                tasks=[EvalResult(task_id="a", passed=True)],
            ),
            ComparisonResult(
                config_name="default",
                model="m",
                scaffolded=False,
                tasks=[EvalResult(task_id="a", passed=False)],
            ),
        )

    monkeypatch.setattr(runner, "print_results", fake_print_results)
    monkeypatch.setattr(runner, "run_comparison", fake_run_comparison)
    monkeypatch.setattr(runner.console, "print", lambda *args, **kwargs: None)

    a, b = await run_comparison_report(runner, [])
    assert printed == ["Scaffolded", "Vanilla"]
    assert a.success_rate == 1.0
    assert b.success_rate == 0.0
