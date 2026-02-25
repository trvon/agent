from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

from dcs.executor import ModelExecutor
from dcs.pipeline import DCSPipeline
from dcs.types import ModelConfig, PipelineConfig, TaskType


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_models(config_dir: Path) -> tuple[ModelConfig, ModelConfig | None]:
    cfg = _read_yaml(config_dir / "models.yaml")
    backends = cfg.get("backends") or {}
    models = cfg.get("models") or {}
    defaults = cfg.get("defaults") or {}

    def build(key: str) -> ModelConfig:
        entry = models.get(key) or {}
        backend_key = entry.get("backend")
        backend = backends.get(backend_key) if isinstance(backends, dict) else None
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"
        if isinstance(backend, dict):
            base_url = str(backend.get("base_url") or base_url)
            api_key = str(backend.get("api_key") or api_key)
        return ModelConfig(
            name=str(entry.get("name") or key),
            base_url=base_url,
            api_key=api_key,
            context_window=int(entry.get("context_window") or 4096),
            max_output_tokens=int(entry.get("max_output_tokens") or 1024),
            temperature=float(entry.get("temperature") or 0.7),
            system_suffix=str(entry.get("system_suffix") or ""),
            request_timeout_s=float(entry.get("request_timeout_s") or 120.0),
            max_retries=int(entry.get("max_retries") or 2),
            retry_backoff_s=float(entry.get("retry_backoff_s") or 2.0),
        )

    executor_key = str(defaults.get("executor") or "")
    critic_key = str(defaults.get("critic") or "")

    executor = (
        build(executor_key) if executor_key else ModelConfig(name="qwen/qwen3-4b-thinking-2507")
    )
    critic = build(critic_key) if critic_key else None
    return executor, critic


def _load_search_weights(config_dir: Path) -> dict[str, float]:
    cfg = _read_yaml(config_dir / "search_weights.yaml")
    search = cfg.get("search") or {}
    fusion = {}
    if isinstance(search, dict):
        fusion = search.get("fusion_weights") or search.get("fusion") or {}
    out: dict[str, float] = {}
    if isinstance(fusion, dict):
        for k, v in fusion.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
    return out


def load_pipeline_config(config_dir: Path) -> PipelineConfig:
    executor, critic = _load_models(config_dir)
    weights = _load_search_weights(config_dir)
    return PipelineConfig(executor_model=executor, critic_model=critic, search_weights=weights)


def _apply_runtime_overrides(args: argparse.Namespace, cfg: PipelineConfig) -> PipelineConfig:
    ngt = getattr(args, "ground_truth_mode", None)
    if ngt is not None:
        cfg.no_ground_truth_mode = bool(ngt)
    dspy_flag = getattr(args, "dspy_faithfulness", None)
    if dspy_flag is not None:
        cfg.use_dspy_faithfulness = bool(dspy_flag)
    profile = getattr(args, "context_profile", None)
    if profile:
        cfg.context_profile = str(profile)
    return cfg


def _init_yams_client(cfg: PipelineConfig):
    from dcs.client import YAMSClient

    try:
        return YAMSClient(yams_binary=cfg.yams_binary, yams_data_dir=cfg.yams_data_dir)  # type: ignore[arg-type]
    except TypeError:
        c = YAMSClient()  # type: ignore[call-arg]
        if hasattr(c, "yams_binary"):
            try:
                c.yams_binary = cfg.yams_binary
            except Exception:
                pass
        if hasattr(c, "yams_data_dir"):
            try:
                c.yams_data_dir = cfg.yams_data_dir
            except Exception:
                pass
        return c


async def _cmd_run(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    pipe = DCSPipeline(cfg)
    res = await pipe.run(args.task)
    console.rule("Output")
    console.print(res.final_output)
    return 0


async def _cmd_eval(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    try:
        from eval.runner import EvalRunner
    except Exception as e:
        console.print(f"Eval runner import failed: {e}")
        return 2

    task_dir = Path(args.task_dir)
    runner = EvalRunner(cfg, task_dir)
    ttype = TaskType(args.type) if args.type else None
    tasks = runner.load_tasks(task_dir, task_type=ttype)
    if not tasks:
        console.print(f"No tasks found in {task_dir}")
        return 1

    results = await runner.run_suite(tasks, scaffolded=True)
    runner.print_results(results, "Evaluation")
    return 0


async def _cmd_compare(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    try:
        from eval.runner import EvalRunner, run_comparison_report
    except Exception as e:
        console.print(f"Eval runner import failed: {e}")
        return 2

    task_dir = Path(args.task_dir)
    runner = EvalRunner(cfg, task_dir)
    tasks = runner.load_tasks(task_dir)
    if not tasks:
        console.print(f"No tasks found in {task_dir}")
        return 1
    await run_comparison_report(runner, tasks)
    return 0


async def _cmd_status(args: argparse.Namespace, cfg: PipelineConfig, console: Console) -> int:
    tbl = Table(title="Status")
    tbl.add_column("Service")
    tbl.add_column("OK")
    tbl.add_column("Details")

    # YAMS
    yams_ok = False
    yams_detail = ""
    try:
        async with _init_yams_client(cfg) as c:
            st = await c.status()
            yams_ok = True
            yams_detail = str(st)
    except Exception as e:
        yams_detail = str(e)
    tbl.add_row("yams", "yes" if yams_ok else "no", yams_detail)

    # LM Studio/OpenAI-compatible
    llm_ok = False
    llm_detail = ""
    try:
        ex = ModelExecutor(cfg.executor_model)
        llm_ok = await ex.health_check()
        llm_detail = f"model={cfg.executor_model.name} base_url={cfg.executor_model.base_url}"
    except Exception as e:
        llm_detail = str(e)
    tbl.add_row("llm", "yes" if llm_ok else "no", llm_detail)

    console.print(tbl)
    return 0 if (yams_ok and llm_ok) else 1


def _build_parser(default_task_dir: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dcs", description="Dynamic Context Scaffold")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="run a single task")
    runp.add_argument("task", type=str, help="task description")
    runp.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    runp.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    runp.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default=None,
        help="Context budget profile",
    )

    evalp = sub.add_parser("eval", help="run evaluation suite")
    evalp.add_argument("--task-dir", type=str, default=str(default_task_dir))
    evalp.add_argument("--type", type=str, choices=[t.value for t in TaskType], default=None)
    evalp.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    evalp.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    evalp.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default=None,
        help="Context budget profile",
    )

    compp = sub.add_parser("compare", help="compare scaffolded vs vanilla")
    compp.add_argument("--task-dir", type=str, default=str(default_task_dir))
    compp.add_argument(
        "--ground-truth-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ground-truth-free faithfulness policy",
    )
    compp.add_argument(
        "--dspy-faithfulness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DSPy-first structured faithfulness extraction",
    )
    compp.add_argument(
        "--context-profile",
        choices=["auto", "standard", "large"],
        default=None,
        help="Context budget profile",
    )

    sub.add_parser("status", help="check YAMS + model connectivity")
    return p


def main() -> None:
    console = Console()
    base_dir = Path(__file__).resolve().parents[1]
    config_dir = base_dir / "configs"
    default_task_dir = base_dir / "eval" / "tasks"

    parser = _build_parser(default_task_dir)
    args = parser.parse_args()

    cfg = _apply_runtime_overrides(args, load_pipeline_config(config_dir))

    async def run_cmd() -> int:
        if args.cmd == "run":
            return await _cmd_run(args, cfg, console)
        if args.cmd == "eval":
            return await _cmd_eval(args, cfg, console)
        if args.cmd == "compare":
            return await _cmd_compare(args, cfg, console)
        if args.cmd == "status":
            return await _cmd_status(args, cfg, console)
        console.print(f"Unknown command: {args.cmd}")
        return 2

    raise SystemExit(asyncio.run(run_cmd()))
