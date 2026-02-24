from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from rich.console import Console
from rich.table import Table

from dcs.assembler import ContextAssembler
from dcs.client import YAMSClient
from dcs.codemap import CodemapBuilder
from dcs.critic import SelfCritic
from dcs.decomposer import TaskDecomposer
from dcs.executor import ModelExecutor
from dcs.optimizer import RetrievalOptimizer
from dcs.planner import QueryPlanner
from dcs.types import (
    ContextBlock,
    Critique,
    ExecutionResult,
    IterationRecord,
    PipelineConfig,
    PipelineResult,
    QuerySpec,
    YAMSQueryResult,
)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _merge_weights(base: dict[str, float], update: dict[str, float]) -> dict[str, float]:
    merged = dict(base or {})
    for k, v in (update or {}).items():
        try:
            merged[str(k)] = float(v)
        except Exception:
            continue
    return merged


class DCSPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.console = Console()

    def _print_iteration_table(self, record: IterationRecord) -> None:
        t = Table(title=f"Iteration {record.iteration}")
        t.add_column("Specs", justify="right")
        t.add_column("Chunks", justify="right")
        t.add_column("Ctx tokens", justify="right")
        t.add_column("Ctx util", justify="right")
        t.add_column("Exec ms", justify="right")
        t.add_column("Quality", justify="right")
        t.add_column("Missing", justify="right")
        t.add_column("Irrelevant", justify="right")

        specs_n = len(record.specs)
        chunks_considered = record.context.chunks_considered if record.context else 0
        chunks_included = record.context.chunks_included if record.context else 0
        chunks_text = f"{chunks_included}/{chunks_considered}"
        ctx_tokens = record.context.token_count if record.context else 0
        ctx_util = record.context.utilization if record.context else 0.0
        exec_ms = record.result.latency_ms if record.result else 0.0
        quality = record.critique.quality_score if record.critique else 0.0
        missing = len(record.critique.missing_info) if record.critique else 0
        irrelevant = len(record.critique.irrelevant_chunks) if record.critique else 0

        t.add_row(
            str(specs_n),
            chunks_text,
            str(ctx_tokens),
            f"{ctx_util:.2f}",
            f"{exec_ms:.0f}",
            f"{quality:.2f}",
            str(missing),
            str(irrelevant),
        )
        self.console.print(t)

    def _converged(self, prev_quality: float | None, critique: Critique) -> bool:
        q = float(critique.quality_score or 0.0)
        if q >= float(self.config.quality_threshold):
            return True
        if prev_quality is None:
            return False
        return abs(q - float(prev_quality)) < float(self.config.convergence_delta)

    def _build_client_kwargs(self, weights: dict[str, float]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        # These are best-effort; client implementation may accept or ignore.
        kwargs["yams_binary"] = self.config.yams_binary
        kwargs["yams_data_dir"] = self.config.yams_data_dir
        kwargs["search_weights"] = weights
        if self.config.yams_cwd:
            kwargs["cwd"] = self.config.yams_cwd
        return kwargs

    def _init_client(self, weights: dict[str, float]) -> YAMSClient:
        """Construct YAMSClient with best-effort config injection.

        The interface spec for YAMSClient in this project doesn't guarantee an __init__
        signature, so we attempt kwargs then fall back to attribute assignment.
        """

        kwargs = self._build_client_kwargs(weights)
        try:
            return YAMSClient(**kwargs)  # type: ignore[arg-type]
        except TypeError:
            c = YAMSClient()  # type: ignore[call-arg]
            for k, v in kwargs.items():
                if hasattr(c, k):
                    try:
                        setattr(c, k, v)
                    except Exception:
                        pass
            return c

    async def run(self, task: str) -> PipelineResult:
        start_total = _now_ms()
        result = PipelineResult(task=task)

        critic_cfg = self.config.critic_model or self.config.executor_model
        base_weights = dict(self.config.search_weights or {})

        self.console.rule("DCS Pipeline")
        self.console.print(f"Task: {task}")
        self.console.print(
            f"Executor: {self.config.executor_model.name} | Critic: {critic_cfg.name} | "
            f"Budget: {self.config.context_budget} | Iterations: {self.config.max_iterations}"
        )

        prev_specs: list[QuerySpec] = []
        prev_critique: Critique | None = None
        prev_quality: float | None = None

        try:
            weights = dict(base_weights)
            async with self._init_client(weights) as client:
                executor = ModelExecutor(self.config.executor_model)
                decomposer = TaskDecomposer(self.config.executor_model)
                planner = QueryPlanner(client)
                assembler = ContextAssembler(budget=self.config.context_budget, model=self.config.executor_model.name)
                critic = SelfCritic(critic_cfg)
                optimizer = RetrievalOptimizer(yams_client=client)

                # Build structural codemap from YAMS knowledge graph (once, before iterations)
                codemap_prefix = ""
                codemap_tokens = 0
                if self.config.codemap_budget > 0:
                    try:
                        t0 = _now_ms()
                        codemap_builder = CodemapBuilder(
                            client,
                            token_budget=self.config.codemap_budget,
                            max_files=30,
                            max_symbols_per_file=15,
                        )
                        codemap_result = await codemap_builder.build(task=task)
                        codemap_prefix = codemap_result.tree_text
                        codemap_tokens = codemap_result.context_block.token_count
                        self.console.print(
                            f"[dim]Codemap: {codemap_result.node_count} nodes, "
                            f"{codemap_tokens} tokens, {codemap_result.latency_ms:.0f}ms[/dim]"
                        )
                    except Exception as e:
                        self.console.print(f"[dim]Codemap: skipped ({e})[/dim]")

                # Adjust assembler budget to reserve room for codemap prefix.
                effective_budget = int(self.config.context_budget)
                if codemap_tokens > 0:
                    effective_budget = max(64, effective_budget - codemap_tokens)
                    assembler = ContextAssembler(budget=effective_budget, model=self.config.executor_model.name)
                    self.console.print(
                        f"[dim]Assembler budget adjusted: {self.config.context_budget} "
                        f"- {codemap_tokens} codemap = {effective_budget}[/dim]"
                    )

                for it in range(1, int(self.config.max_iterations) + 1):
                    iter_start = _now_ms()
                    record = IterationRecord(iteration=it)
                    result.iterations.append(record)

                    stage_table = Table(title=f"Iteration {it} Stages")
                    stage_table.add_column("Stage")
                    stage_table.add_column("ms", justify="right")
                    stage_table.add_column("Details")

                    try:
                        # a) Decompose/refine
                        t0 = _now_ms()
                        if it == 1 or prev_critique is None:
                            specs = await decomposer.decompose(task, max_queries=self.config.max_queries_per_iteration)
                        else:
                            specs = await decomposer.refine(task, prev_critique, prev_specs)
                        specs = list(specs or [])[: int(self.config.max_queries_per_iteration)]
                        record.specs = specs
                        stage_table.add_row(
                            "decompose" if it == 1 else "refine",
                            f"{_now_ms() - t0:.0f}",
                            f"specs={len(specs)}",
                        )

                        # b) Plan/retrieve
                        t0 = _now_ms()
                        query_results: list[YAMSQueryResult] = await planner.execute(specs)

                        # Enforce local caps/filters regardless of planner implementation.
                        for qr in query_results:
                            chunks = list(qr.chunks or [])
                            chunks = [c for c in chunks if float(getattr(c, "score", 0.0) or 0.0) >= float(self.config.min_chunk_score)]
                            chunks.sort(key=lambda c: float(getattr(c, "score", 0.0) or 0.0), reverse=True)
                            qr.chunks = chunks[: int(self.config.max_chunks_per_query)]

                        record.query_results = query_results
                        stage_table.add_row(
                            "retrieve",
                            f"{_now_ms() - t0:.0f}",
                            f"queries={len(query_results)}",
                        )

                        # c) Assemble
                        t0 = _now_ms()
                        context = assembler.assemble(query_results, task=task)

                        # Prepend codemap prefix to assembled context so the
                        # model sees structural codebase awareness before the
                        # task-specific retrieved chunks.
                        if codemap_prefix:
                            combined_content = codemap_prefix.rstrip() + "\n\n" + context.content
                            combined_tokens = context.token_count + codemap_tokens
                            context = ContextBlock(
                                content=combined_content,
                                sources=["yams-knowledge-graph"] + list(context.sources or []),
                                chunk_ids=["codemap"] + list(context.chunk_ids or []),
                                token_count=combined_tokens,
                                budget=self.config.context_budget,  # report against full budget
                                utilization=combined_tokens / max(1, self.config.context_budget),
                                chunks_included=context.chunks_included + 1,
                                chunks_considered=context.chunks_considered,
                            )

                        record.context = context
                        stage_table.add_row(
                            "assemble",
                            f"{_now_ms() - t0:.0f}",
                            f"tokens={context.token_count} util={context.utilization:.2f}",
                        )

                        # d) Execute
                        t0 = _now_ms()
                        exec_result = await executor.execute(task=task, context=context)
                        record.result = exec_result
                        stage_table.add_row(
                            "execute",
                            f"{_now_ms() - t0:.0f}",
                            f"model={exec_result.model}",
                        )

                        # e) Critique
                        t0 = _now_ms()
                        critique = await critic.critique(task=task, context=context, result=exec_result)
                        record.critique = critique
                        stage_table.add_row(
                            "critique",
                            f"{_now_ms() - t0:.0f}",
                            f"quality={critique.quality_score:.2f}",
                        )

                        # f) Optimize
                        t0 = _now_ms()
                        try:
                            optimizer.record_feedback(query_results=query_results, critique=critique)
                            adjusted = optimizer.get_adjusted_weights() or {}
                            weights = _merge_weights(base_weights, adjusted)
                            # Best-effort: update client weights if it exposes an attribute.
                            if hasattr(client, "search_weights"):
                                setattr(client, "search_weights", weights)
                        except Exception as e:
                            stage_table.add_row(
                                "optimize",
                                f"{_now_ms() - t0:.0f}",
                                f"error={e}",
                            )
                        else:
                            stage_table.add_row(
                                "optimize",
                                f"{_now_ms() - t0:.0f}",
                                f"weights={len(weights)}",
                            )

                        record.latency_ms = _now_ms() - iter_start

                        self.console.print(stage_table)
                        self._print_iteration_table(record)

                        prev_specs = specs
                        prev_critique = critique

                        if self._converged(prev_quality, critique):
                            result.converged = True
                            break
                        prev_quality = float(critique.quality_score or 0.0)

                    except Exception as e:
                        record.latency_ms = _now_ms() - iter_start
                        record.result = ExecutionResult(output=f"Error in iteration {it}: {e}")
                        self.console.print(stage_table)
                        self.console.print(f"Iteration {it} failed: {e}")
                        break

        except Exception as e:
            # Failed to even initialize pipeline dependencies.
            err_record = IterationRecord(
                iteration=1,
                specs=[],
                query_results=[],
                context=ContextBlock(content="", token_count=0, budget=self.config.context_budget),
                result=ExecutionResult(output=f"Pipeline initialization error: {e}"),
                critique=None,
                latency_ms=0.0,
            )
            result.iterations.append(err_record)

        # Finalize — pick the best iteration by quality score (not blindly last).
        best_iter: IterationRecord | None = None
        if result.iterations:
            # Prefer iteration with highest critique quality_score.
            scored = [
                it for it in result.iterations
                if it.critique is not None and it.result is not None
            ]
            if scored:
                best_iter = max(scored, key=lambda it: (
                    float((it.critique.quality_score if it.critique else 0.0) or 0.0),
                    int(it.context.token_count if it.context else 0),  # tiebreak: more context = better
                ))
            else:
                # No critiques succeeded — pick iteration with most context tokens.
                with_result = [it for it in result.iterations if it.result is not None]
                if with_result:
                    best_iter = max(
                        with_result,
                        key=lambda it: int(it.context.token_count if it.context else 0),
                    )

        if best_iter is not None and best_iter.result is not None:
            result.final_output = best_iter.result.output
        result.total_latency_ms = _now_ms() - start_total

        final_quality = 0.0
        best_iter_num = 0
        if best_iter is not None:
            best_iter_num = best_iter.iteration
            if best_iter.critique:
                final_quality = float(best_iter.critique.quality_score or 0.0)

        final_tbl = Table(title="Run Summary")
        final_tbl.add_column("Iterations", justify="right")
        final_tbl.add_column("Best iter", justify="right")
        final_tbl.add_column("Converged", justify="center")
        final_tbl.add_column("Total ms", justify="right")
        final_tbl.add_column("Final quality", justify="right")
        final_tbl.add_row(
            str(len(result.iterations)),
            str(best_iter_num),
            "yes" if result.converged else "no",
            f"{result.total_latency_ms:.0f}",
            f"{final_quality:.2f}",
        )
        self.console.print(final_tbl)
        return result

    async def run_vanilla(self, task: str) -> PipelineResult:
        start_total = _now_ms()
        result = PipelineResult(task=task)

        self.console.rule("Vanilla Run")
        self.console.print(f"Task: {task}")

        executor = ModelExecutor(self.config.executor_model)
        iter_start = _now_ms()
        record = IterationRecord(iteration=1)
        result.iterations.append(record)

        try:
            exec_result = await executor.execute(task=task, context=None)
            record.result = exec_result
            record.context = None
            record.critique = None
            record.query_results = []
            record.specs = []
        except Exception as e:
            record.result = ExecutionResult(output=f"Error: vanilla execution failed: {e}")

        record.latency_ms = _now_ms() - iter_start
        result.final_output = record.result.output if record.result else ""
        result.total_latency_ms = _now_ms() - start_total
        result.converged = True
        return result


def pipeline_result_to_dict(res: PipelineResult) -> dict[str, Any]:
    """Best-effort conversion for logging/serialization."""

    try:
        return asdict(res)
    except Exception:
        # Avoid hard failures when some types have non-serializable fields.
        out: dict[str, Any] = {
            "task": res.task,
            "final_output": res.final_output,
            "total_latency_ms": res.total_latency_ms,
            "converged": res.converged,
            "iterations": [],
        }
        for it in res.iterations:
            out["iterations"].append(
                {
                    "iteration": it.iteration,
                    "latency_ms": it.latency_ms,
                    "specs": [asdict(s) for s in it.specs],
                    "query_results": [
                        {
                            "spec": asdict(q.spec),
                            "latency_ms": q.latency_ms,
                            "error": q.error,
                            "chunks": [
                                {
                                    "chunk_id": c.chunk_id,
                                    "score": c.score,
                                    "source": c.source,
                                    "token_count": c.token_count,
                                }
                                for c in (q.chunks or [])
                            ],
                        }
                        for q in (it.query_results or [])
                    ],
                    "context": None
                    if it.context is None
                    else {
                        "token_count": it.context.token_count,
                        "budget": it.context.budget,
                        "utilization": it.context.utilization,
                        "chunks_included": it.context.chunks_included,
                        "chunks_considered": it.context.chunks_considered,
                    },
                    "result": None
                    if it.result is None
                    else {
                        "model": it.result.model,
                        "latency_ms": it.result.latency_ms,
                        "tokens_prompt": it.result.tokens_prompt,
                        "tokens_completion": it.result.tokens_completion,
                    },
                    "critique": None
                    if it.critique is None
                    else {
                        "quality_score": it.critique.quality_score,
                        "context_utilization": it.critique.context_utilization,
                        "missing_info": list(it.critique.missing_info or []),
                        "irrelevant_chunks": list(it.critique.irrelevant_chunks or []),
                        "suggested_queries": list(it.critique.suggested_queries or []),
                    },
                }
            )
        return out
