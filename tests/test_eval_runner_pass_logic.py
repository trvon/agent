from __future__ import annotations

from pathlib import Path

from dcs.types import EvalTask, PipelineConfig, TaskType
from eval.runner import EvalRunner


def _runner() -> EvalRunner:
    return EvalRunner(PipelineConfig(), task_dir=Path("eval/tasks"))


def test_decide_pass_uses_faithfulness_without_patterns() -> None:
    r = _runner()
    t = EvalTask(
        id="x",
        task_type=TaskType.QA,
        description="no-ground-truth task",
        evaluation={"faithfulness_threshold": 0.6},
    )
    assert r._decide_pass(
        t,
        {
            "faithfulness_confidence": 0.72,
            "faithfulness_should_abstain": 0.0,
        },
    )


def test_decide_pass_fails_when_faithfulness_abstains() -> None:
    r = _runner()
    t = EvalTask(
        id="x",
        task_type=TaskType.QA,
        description="no-ground-truth task",
        evaluation={"faithfulness_threshold": 0.6},
    )
    assert not r._decide_pass(
        t,
        {
            "faithfulness_confidence": 0.9,
            "faithfulness_should_abstain": 1.0,
        },
    )
