from __future__ import annotations

from dcs.types import ContextBlock, Critique, EvalTask, IterationRecord, PipelineResult, TaskType
from eval.metrics import evaluate_task


def test_evaluate_task_uses_best_iteration_for_quality_and_faithfulness() -> None:
    t = EvalTask(
        id="t1",
        task_type=TaskType.QA,
        description="task",
        evaluation={"quality_threshold": 0.5},
    )
    it1 = IterationRecord(
        iteration=1,
        context=ContextBlock(content="ctx", token_count=10, budget=100),
        critique=Critique(context_utilization=0.9, quality_score=0.8),
    )
    it2 = IterationRecord(
        iteration=2,
        context=ContextBlock(content="ctx", token_count=10, budget=100),
        critique=Critique(context_utilization=0.4, quality_score=0.1),
    )

    # Inject a lightweight faithfulness-like object for metric extraction.
    class F:
        confidence = 0.77
        supported_ratio = 0.66
        should_abstain = False

    it1.faithfulness = F()

    pr = PipelineResult(task="task", iterations=[it1, it2], best_iteration=1)
    m = evaluate_task(t, pr)
    assert m["quality_score"] == 0.8
    assert m["faithfulness_confidence"] == 0.77
