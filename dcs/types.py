"""Shared types for the Dynamic Context Scaffold pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Query / Decomposition types
# ---------------------------------------------------------------------------


class QueryType(str, Enum):
    """Kind of YAMS query to execute."""

    SEMANTIC = "semantic"
    GREP = "grep"
    GRAPH = "graph"
    GET = "get"
    LIST = "list"


@dataclass
class QuerySpec:
    """A single information need produced by the decomposer."""

    query: str
    query_type: QueryType
    importance: float  # 0.0–1.0
    reason: str = ""


# ---------------------------------------------------------------------------
# YAMS result types
# ---------------------------------------------------------------------------


@dataclass
class YAMSChunk:
    """A single chunk/result returned by YAMS."""

    chunk_id: str  # hash or path
    content: str
    score: float = 0.0
    source: str = ""  # origin path or collection
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0  # filled by assembler


@dataclass
class YAMSQueryResult:
    """Aggregated result of a single YAMS query."""

    spec: QuerySpec
    chunks: list[YAMSChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Context assembly types
# ---------------------------------------------------------------------------


@dataclass
class ContextBlock:
    """Token-bounded assembled context ready for model injection."""

    content: str
    sources: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    token_count: int = 0
    budget: int = 0
    utilization: float = 0.0  # token_count / budget
    chunks_included: int = 0
    chunks_considered: int = 0


# ---------------------------------------------------------------------------
# Execution types
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result from running a model on assembled context."""

    output: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    model: str = ""
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Critique / Feedback types
# ---------------------------------------------------------------------------


@dataclass
class Critique:
    """Self-critique feedback on context quality and output."""

    context_utilization: float  # 0.0–1.0
    missing_info: list[str] = field(default_factory=list)
    irrelevant_chunks: list[str] = field(default_factory=list)  # chunk_ids
    quality_score: float = 0.0  # 0.0–1.0
    suggested_queries: list[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Pipeline types
# ---------------------------------------------------------------------------


class PipelineStage(str, Enum):
    """Stages of the DCS pipeline."""

    DECOMPOSE = "decompose"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    ASSEMBLE = "assemble"
    EXECUTE = "execute"
    CRITIQUE = "critique"
    OPTIMIZE = "optimize"


@dataclass
class IterationRecord:
    """Record of a single pipeline iteration."""

    iteration: int
    specs: list[QuerySpec] = field(default_factory=list)
    query_results: list[YAMSQueryResult] = field(default_factory=list)
    context: ContextBlock | None = None
    result: ExecutionResult | None = None
    critique: Critique | None = None
    latency_ms: float = 0.0


@dataclass
class PipelineResult:
    """Full result of a DCS pipeline run (possibly multi-iteration)."""

    task: str
    iterations: list[IterationRecord] = field(default_factory=list)
    final_output: str = ""
    total_latency_ms: float = 0.0
    converged: bool = False

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def final_critique(self) -> Critique | None:
        if self.iterations:
            return self.iterations[-1].critique
        return None


# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a model backend."""

    name: str  # model identifier (e.g. "qwen/qwen3-4b-thinking-2507")
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"  # LM Studio default
    context_window: int = 4096
    max_output_tokens: int = 1024
    temperature: float = 0.7
    # Optional suffix appended to system prompts. Use "/no_think" for qwen3
    # thinking models to disable chain-of-thought and save output tokens.
    system_suffix: str = ""


@dataclass
class PipelineConfig:
    """Configuration for a DCS pipeline run."""

    # Model settings
    executor_model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="qwen/qwen3-4b-thinking-2507")
    )
    critic_model: ModelConfig | None = None  # defaults to executor_model

    # Context budget
    context_budget: int = 2048  # tokens reserved for retrieved context
    system_prompt_budget: int = 512  # tokens for system prompt
    output_reserve: int = 1024  # tokens reserved for model output
    codemap_budget: int = 256  # tokens reserved for structural codemap prefix

    # Retrieval settings
    max_queries_per_iteration: int = 5
    max_chunks_per_query: int = 10
    min_chunk_score: float = 0.1

    # Iteration settings
    max_iterations: int = 3
    quality_threshold: float = 0.7  # stop if critique quality >= this
    convergence_delta: float = 0.05  # stop if quality improvement < this

    # YAMS settings
    yams_binary: str = "yams"
    yams_data_dir: str | None = None
    yams_cwd: str | None = None  # scope search/grep to this directory tree

    # Search weight overrides (passed to YAMS search)
    search_weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Category of evaluation task."""

    CODING = "coding"
    QA = "qa"
    AGENT = "agent"


class EvalMetric(str, Enum):
    """Evaluation metric names."""

    TASK_SUCCESS = "task_success"
    CONTEXT_EFFICIENCY = "context_efficiency"
    RETRIEVAL_PRECISION = "retrieval_precision"
    ITERATIONS_TO_CONVERGE = "iterations_to_converge"
    TOTAL_LATENCY = "total_latency"
    TOKEN_COST = "token_cost"


@dataclass
class EvalTask:
    """Definition of a single evaluation task."""

    id: str
    task_type: TaskType
    description: str
    ground_truth: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single task."""

    task_id: str
    pipeline_result: PipelineResult | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    error: str | None = None


@dataclass
class ComparisonResult:
    """Comparison of scaffolded vs baseline across a task suite."""

    config_name: str
    model: str
    scaffolded: bool
    tasks: list[EvalResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(1 for t in self.tasks if t.passed) / len(self.tasks)

    @property
    def avg_metric(self) -> dict[str, float]:
        if not self.tasks:
            return {}
        metrics: dict[str, list[float]] = {}
        for t in self.tasks:
            for k, v in t.metrics.items():
                metrics.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in metrics.items()}
