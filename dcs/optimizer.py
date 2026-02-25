"""Feedback store and lightweight retrieval weight optimizer.

This module maintains simple statistics over critique feedback to adjust
retrieval fusion weights over time.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Any

import yaml

from dcs.types import Critique, QueryType, YAMSQueryResult

logger = logging.getLogger(__name__)


DEFAULT_WEIGHTS: dict[str, Any] = {
    "fusion_weights": {"lexical": 0.3, "semantic": 0.4, "graph": 0.2, "metadata": 0.1},
    "query_type_success": {"semantic": [], "grep": [], "graph": [], "get": []},
}


def _clamp01(x: float) -> float:
    try:
        xf = float(x)
    except Exception:
        return 0.0
    if xf < 0.0:
        return 0.0
    if xf > 1.0:
        return 1.0
    return xf


def _normalize_weights(w: dict[str, float], floor: float = 0.05) -> dict[str, float]:
    out = {k: max(float(v), 0.0) for k, v in w.items()}
    if floor > 0:
        for k in out:
            out[k] = max(out[k], floor)
    s = sum(out.values())
    if s <= 0:
        # Shouldn't happen; fall back to even split.
        n = max(1, len(out))
        return {k: 1.0 / n for k in out}
    return {k: v / s for k, v in out.items()}


class RetrievalOptimizer:
    """Track retrieval performance and adjust fusion weights."""

    def __init__(self, yams_client: Any | None = None, config_path: str | None = None):
        self.yams_client = yams_client
        self.config_path = config_path
        self._lock = threading.Lock()
        self.reset()
        if config_path:
            self.load(config_path)

    def reset(self) -> None:
        with self._lock:
            self.weights = {
                "fusion_weights": dict(DEFAULT_WEIGHTS["fusion_weights"]),
                "query_type_success": {k: [] for k in DEFAULT_WEIGHTS["query_type_success"]},
            }

    def record_feedback(self, query_results: list[YAMSQueryResult], critique: Critique) -> None:
        irrelevant = set(critique.irrelevant_chunks or [])
        max_window = 50
        alpha = 0.1

        by_qtype: dict[str, list[float]] = {}
        for qr in query_results:
            qtype = getattr(getattr(qr, "spec", None), "query_type", None)
            qtype_str = ""
            if isinstance(qtype, QueryType):
                qtype_str = qtype.value
            elif isinstance(qtype, str):
                qtype_str = qtype
            qtype_str = qtype_str or ""
            if qtype_str not in ("semantic", "grep", "graph", "get", "list"):
                continue

            score = self._score_query_result(qr, irrelevant)
            by_qtype.setdefault(qtype_str, []).append(score)

        with self._lock:
            qsucc: dict[str, list[float]] = self.weights.get("query_type_success", {})
            for qtype_str, obs in by_qtype.items():
                # Map "list" into "get" bucket (both are lookup-ish).
                bucket = "get" if qtype_str == "list" else qtype_str
                if bucket not in qsucc:
                    continue
                qsucc[bucket].extend(float(x) for x in obs)
                if len(qsucc[bucket]) > max_window:
                    qsucc[bucket] = qsucc[bucket][-max_window:]

            # Update fusion weights via EMA toward per-type success rates.
            scores = self._query_type_scores_locked()
            target = {
                "lexical": scores.get("grep", 0.5),
                "semantic": scores.get("semantic", 0.5),
                "graph": scores.get("graph", 0.5),
                "metadata": scores.get("get", 0.5),
            }
            fw: dict[str, float] = dict(self.weights.get("fusion_weights", {}))
            for k, t in target.items():
                if k not in fw:
                    fw[k] = float(t)
                    continue
                fw[k] = (1.0 - alpha) * float(fw[k]) + alpha * float(t)
            self.weights["fusion_weights"] = _normalize_weights(fw)

        self._store_feedback_in_yams(query_results=query_results, critique=critique)

    def _score_query_result(self, qr: YAMSQueryResult, irrelevant: set[str]) -> float:
        chunks = getattr(qr, "chunks", None) or []
        if not chunks:
            return 0.0

        useful = 0
        total = 0
        enriched = 0
        for ch in chunks:
            cid = getattr(ch, "chunk_id", "") or ""
            total += 1
            if cid and cid in irrelevant:
                continue
            useful += 1
            if getattr(ch, "metadata", None) and ch.metadata.get("enriched"):
                enriched += 1

        base = useful / max(1, total)

        # Penalize path-only grep results (enriched=False)
        if total > 0 and enriched == 0:
            return max(0.05, base * 0.3)

        # Reward enriched content modestly
        if enriched > 0:
            bonus = min(0.2, enriched / max(1, total) * 0.2)
            return _clamp01(base + bonus)

        return _clamp01(base)

    def get_adjusted_weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self.weights.get("fusion_weights", {}))

    def _query_type_scores_locked(self) -> dict[str, float]:
        qsucc: dict[str, list[float]] = self.weights.get("query_type_success", {})
        out: dict[str, float] = {}
        for k in ("semantic", "grep", "graph", "get"):
            vals = [float(v) for v in (qsucc.get(k) or [])]
            if not vals:
                out[k] = 0.5
            else:
                out[k] = _clamp01(sum(vals) / len(vals))
        return out

    def get_query_type_scores(self) -> dict[str, float]:
        with self._lock:
            return self._query_type_scores_locked()

    def suggest_strategy(self, task: str, history: list[Critique]) -> dict:
        scores = self.get_query_type_scores()
        fw = self.get_adjusted_weights()
        last = history[-1] if history else None

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        emphasize = [k for k, _ in ranked[:2]]
        deemphasize = [k for k, _ in ranked[-1:]]

        budget_multiplier = 1.0
        more_iterations = False
        notes: list[str] = []

        if last is not None:
            if (last.missing_info or []) and last.quality_score < 0.7:
                budget_multiplier = 1.25
                more_iterations = True
                notes.append("Missing info detected; increase context budget and iterate.")

            if last.context_utilization < 0.25 and last.quality_score >= 0.6:
                budget_multiplier = 0.9
                notes.append("Low context utilization; trim budget or increase precision.")

            if last.quality_score < 0.4:
                more_iterations = True
                notes.append("Low quality score; additional retrieval iteration likely helpful.")

        # If semantic is underperforming, bias toward grep/graph depending on what's strongest.
        if scores.get("semantic", 0.5) < 0.45:
            if "grep" not in emphasize and scores.get("grep", 0.5) >= scores.get("graph", 0.5):
                emphasize = ["grep", emphasize[0]] if emphasize else ["grep"]
            elif "graph" not in emphasize:
                emphasize = ["graph", emphasize[0]] if emphasize else ["graph"]

        return {
            "task": (task or "").strip(),
            "emphasize_query_types": emphasize,
            "de_emphasize_query_types": deemphasize,
            "context_budget_multiplier": budget_multiplier,
            "more_iterations": bool(more_iterations),
            "fusion_weights": fw,
            "query_type_scores": scores,
            "notes": notes,
        }

    def save(self, path: str | None = None) -> None:
        save_path = path or self.config_path
        if not save_path:
            return

        with self._lock:
            data = {
                "fusion_weights": dict(self.weights.get("fusion_weights", {})),
                "query_type_success": {
                    k: list(v)
                    for k, v in (self.weights.get("query_type_success", {}) or {}).items()
                },
            }

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fd = None
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="search_weights_", suffix=".yaml")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None
                yaml.safe_dump(data, f, sort_keys=True)
            os.replace(tmp_path, save_path)
            tmp_path = None
            logger.debug("Saved retrieval weights to %s", save_path)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def load(self, path: str | None = None) -> None:
        load_path = path or self.config_path
        if not load_path or not os.path.exists(load_path):
            return

        try:
            with open(load_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            logger.warning("Failed to load retrieval weights from %s", load_path, exc_info=True)
            return

        if not isinstance(data, dict):
            return

        fusion = data.get("fusion_weights")
        qsucc = data.get("query_type_success")
        if not isinstance(fusion, dict) or not isinstance(qsucc, dict):
            return

        with self._lock:
            fw: dict[str, float] = dict(self.weights.get("fusion_weights", {}))
            for k in ("lexical", "semantic", "graph", "metadata"):
                if k in fusion:
                    fw[k] = float(fusion[k])
            self.weights["fusion_weights"] = _normalize_weights(fw)

            qt: dict[str, list[float]] = self.weights.get("query_type_success", {})
            for k in ("semantic", "grep", "graph", "get"):
                vals = qsucc.get(k)
                if isinstance(vals, list):
                    qt[k] = [float(v) for v in vals][-50:]
            self.weights["query_type_success"] = qt

        logger.debug("Loaded retrieval weights from %s", load_path)

    def _store_feedback_in_yams(
        self, query_results: list[YAMSQueryResult], critique: Critique
    ) -> None:
        if self.yams_client is None:
            return
        add = getattr(self.yams_client, "add", None)
        if not callable(add):
            return

        # Best-effort, generic payload; avoid depending on a specific YAMSClient signature.
        try:
            query_types: list[str] = []
            for qr in query_results:
                qt = getattr(getattr(qr, "spec", None), "query_type", "")
                if isinstance(qt, QueryType):
                    query_types.append(qt.value)
                else:
                    query_types.append(str(qt or ""))

            payload = {
                "critique": {
                    "context_utilization": float(critique.context_utilization),
                    "quality_score": float(critique.quality_score),
                    "missing_info": list(critique.missing_info or []),
                    "irrelevant_chunks": list(critique.irrelevant_chunks or []),
                    "suggested_queries": list(critique.suggested_queries or []),
                    "reasoning": critique.reasoning,
                },
                "query_types": query_types,
            }
            add(
                content=yaml.safe_dump(payload, sort_keys=True),
                metadata={
                    "owner": "dcs",
                    "source": "feedback",
                    "kind": "retrieval_optimizer",
                },
            )
        except TypeError:
            # Fallback to a simpler signature.
            try:
                add(yaml.safe_dump({"critique": critique.__dict__}, sort_keys=True))
            except Exception:
                logger.debug("YAMS feedback store failed", exc_info=True)
        except Exception:
            logger.debug("YAMS feedback store failed", exc_info=True)
