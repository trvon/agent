from __future__ import annotations

from types import SimpleNamespace

from dcs.types import ContextBlock, Critique, EvalTask, IterationRecord, PipelineResult, TaskType
from eval.metrics import (
    _token_count,
    context_efficiency,
    evaluate_task,
    retrieval_precision,
    score_contains_pattern,
    score_exact_match,
    score_output_length,
)


def test_token_count_and_length_scoring() -> None:
    assert _token_count("  a  b   c ") == 3
    assert score_output_length("a b c", 2, 3) == 1.0
    assert score_output_length("a", 2, 3) == 0.0
    assert score_output_length("a b c d", 1, 3) == 0.0


def test_contains_pattern_supports_literal_regex_and_invalid_regex() -> None:
    out = "ResourceGovernor canLoadModel with json-rpc transport"
    pats = ["resourcegovernor", "re:canLoadModel", "re:[invalid"]
    assert score_contains_pattern(out, pats) == 2 / 3
    assert score_contains_pattern(out, []) == 0.0


def test_exact_match_trims_whitespace() -> None:
    assert score_exact_match("  hello\n", "hello") == 1.0
    assert score_exact_match("hello", "world") == 0.0


def test_context_efficiency_and_retrieval_precision_bounds() -> None:
    ctx = ContextBlock(content="ctx", chunk_ids=["a", "b", "c"], chunks_included=3)
    crt = Critique(context_utilization=0.8, irrelevant_chunks=["b"])
    assert context_efficiency(ctx, crt) == 0.8 * (2 / 3)
    assert retrieval_precision(crt, ctx) == 2 / 3

    zero_ctx = ContextBlock(content="ctx", chunk_ids=[], chunks_included=0)
    assert context_efficiency(zero_ctx, crt) == 0.8
    assert retrieval_precision(crt, zero_ctx) == 0.0

    assert context_efficiency(None, crt) == 0.0
    assert retrieval_precision(crt, None) == 0.0


def test_evaluate_task_populates_all_metric_families() -> None:
    task = EvalTask(
        id="qa-1",
        task_type=TaskType.QA,
        description="demo",
        ground_truth={"expected": "final answer daemon mcp", "patterns": ["daemon", "re:mcp"]},
        evaluation={
            "length": {"min_tokens": 1, "max_tokens": 10},
        },
    )
    faith = SimpleNamespace(confidence=0.9, supported_ratio=0.75, should_abstain=False)
    it = IterationRecord(
        iteration=1,
        context=ContextBlock(content="ctx", chunk_ids=["c1", "c2"], chunks_included=2),
        critique=Critique(context_utilization=0.5, irrelevant_chunks=["c2"], quality_score=0.8),
        faithfulness=faith,
    )
    pr = PipelineResult(
        task="demo",
        iterations=[it],
        final_output="final answer daemon mcp",
        total_latency_ms=42.0,
    )

    m = evaluate_task(task, pr)
    assert m["exact_match"] == 1.0
    assert m["contains_pattern"] == 1.0
    assert m["output_length"] == 1.0
    assert m["quality_score"] == 0.8
    assert m["context_efficiency"] > 0
    assert m["retrieval_precision"] > 0
    assert m["faithfulness_confidence"] == 0.9
    assert m["faithfulness_supported_ratio"] == 0.75
    assert m["faithfulness_should_abstain"] == 0.0
    assert m["iterations"] == 1.0
    assert m["total_latency_ms"] == 42.0


def test_evaluate_task_defaults_when_no_iteration_data() -> None:
    task = EvalTask(id="qa-2", task_type=TaskType.QA, description="demo")
    pr = PipelineResult(task="demo", iterations=[], final_output="x")
    m = evaluate_task(task, pr)

    assert m["context_efficiency"] == 0.0
    assert m["retrieval_precision"] == 0.0
    assert m["faithfulness_confidence"] == 0.0
    assert m["faithfulness_supported_ratio"] == 0.0
    assert m["faithfulness_should_abstain"] == 0.0
