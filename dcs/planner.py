from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Protocol

from dcs.types import QuerySpec, QueryType, YAMSChunk, YAMSQueryResult


logger = logging.getLogger(__name__)


class YAMSClientLike(Protocol):
    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[YAMSChunk]: ...

    async def grep(self, pattern: str, **kwargs: Any) -> list[YAMSChunk]: ...

    async def graph(self, query: str) -> list[YAMSChunk]: ...

    async def get(self, name_or_hash: str) -> YAMSChunk | None: ...

    async def list_docs(self, **kwargs: Any) -> list[YAMSChunk]: ...

    async def execute_spec(self, spec: QuerySpec) -> YAMSQueryResult: ...


_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:\\)?(?:\.?\.?/)?[\w.\-~/]+(?:/[\w.\-]+)+)"
)
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _spec_key(spec: QuerySpec) -> tuple[str, str]:
    return (spec.query_type.value, spec.query.strip())


def _dedupe_chunks(chunks: list[YAMSChunk]) -> list[YAMSChunk]:
    seen: set[str] = set()
    out: list[YAMSChunk] = []
    for c in chunks:
        cid = (c.chunk_id or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


class QueryPlanner:
    """Maps QuerySpecs to concrete YAMS executions (including expansions)."""

    def __init__(self, yams: YAMSClientLike):
        self._yams = yams

    async def execute(self, specs: list[QuerySpec]) -> list[YAMSQueryResult]:
        """Execute all query specs concurrently; keep input order."""
        specs = self._dedupe_specs(specs)
        if not specs:
            return []

        t0 = time.perf_counter()
        tasks = [self._timed_execute_spec(s) for s in specs]
        results = await asyncio.gather(*tasks)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("executed %d specs in %dms", len(specs), int(dt_ms))
        return self._merge_results_by_spec(results)

    async def execute_multihop(
        self, specs: list[QuerySpec], depth: int = 2
    ) -> list[YAMSQueryResult]:
        """Execute specs, then follow up on high-importance results."""
        depth = max(1, int(depth))
        primary = await self.execute(specs)
        all_results = list(primary)

        # Breadth-first expansion driven by result contents.
        visited_specs = {_spec_key(r.spec) for r in all_results}
        frontier = list(primary)

        for hop in range(1, depth + 1):
            follow_specs: list[QuerySpec] = []
            for res in frontier:
                if res.spec.importance < 0.7:
                    continue
                follow_specs.extend(self._followups_from_result(res, hop=hop))

            follow_specs = [s for s in self._dedupe_specs(follow_specs) if _spec_key(s) not in visited_specs]
            if not follow_specs:
                break

            logger.debug("multihop hop=%d follow_specs=%d", hop, len(follow_specs))
            follow_results = await self.execute(follow_specs)
            for r in follow_results:
                visited_specs.add(_spec_key(r.spec))

            all_results.extend(follow_results)
            frontier = follow_results

        return self._merge_results_by_spec(all_results)

    async def execute_with_expansion(self, specs: list[QuerySpec]) -> list[YAMSQueryResult]:
        """Execute specs, then graph-traverse top chunks to pull related chunks."""
        primary = await self.execute(specs)
        if not primary:
            return []

        graph_specs: list[QuerySpec] = []
        for res in primary:
            # Prefer top scored chunks; fall back to first.
            chunks = sorted(res.chunks or [], key=lambda c: c.score, reverse=True)
            for c in chunks[:2]:
                q = (c.chunk_id or "").strip()
                if not q:
                    continue
                graph_specs.append(
                    QuerySpec(
                        query=q,
                        query_type=QueryType.GRAPH,
                        importance=max(0.4, res.spec.importance - 0.2),
                        reason=f"graph expansion from {res.spec.query_type.value}:{res.spec.query}",
                    )
                )

        expanded = await self.execute(graph_specs)
        return self._merge_results_by_spec(primary + expanded)

    async def _timed_execute_spec(self, spec: QuerySpec) -> YAMSQueryResult:
        t0 = time.perf_counter()
        try:
            res = await self._yams.execute_spec(spec)
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning("execute_spec failed (%s %r): %s", spec.query_type.value, spec.query, e)
            return YAMSQueryResult(spec=spec, chunks=[], latency_ms=dt_ms, error=str(e))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        res.latency_ms = float(res.latency_ms or dt_ms)
        res.chunks = _dedupe_chunks(res.chunks or [])
        return res

    def _dedupe_specs(self, specs: list[QuerySpec]) -> list[QuerySpec]:
        # Keep first occurrence; callers can pre-sort by importance.
        out: list[QuerySpec] = []
        seen: set[tuple[str, str]] = set()
        for s in specs or []:
            if not s or not s.query.strip():
                continue
            key = _spec_key(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(QuerySpec(
                query=s.query.strip(),
                query_type=s.query_type,
                importance=float(max(0.0, min(1.0, s.importance))),
                reason=s.reason or "",
            ))
        return out

    def _merge_results_by_spec(self, results: list[YAMSQueryResult]) -> list[YAMSQueryResult]:
        merged: dict[tuple[str, str], YAMSQueryResult] = {}
        order: list[tuple[str, str]] = []

        for r in results or []:
            key = _spec_key(r.spec)
            if key not in merged:
                merged[key] = YAMSQueryResult(
                    spec=r.spec,
                    chunks=_dedupe_chunks(list(r.chunks or [])),
                    latency_ms=float(r.latency_ms or 0.0),
                    error=r.error,
                )
                order.append(key)
                continue

            cur = merged[key]
            cur.chunks = _dedupe_chunks(cur.chunks + list(r.chunks or []))
            cur.latency_ms = max(float(cur.latency_ms or 0.0), float(r.latency_ms or 0.0))
            if cur.error is None and r.error:
                cur.error = r.error

        return [merged[k] for k in order]

    def _followups_from_result(self, res: YAMSQueryResult, hop: int) -> list[QuerySpec]:
        text = "\n".join(c.content for c in (res.chunks or []) if c.content)
        if not text:
            return []

        follow: list[QuerySpec] = []
        # Files mentioned -> GET
        paths: set[str] = set()
        for m in _PATH_RE.finditer(text):
            p = m.group("path").strip().strip("'\"")
            if "/" not in p:
                continue
            # Avoid obvious URLs.
            if p.startswith("http://") or p.startswith("https://"):
                continue
            paths.add(p)

        for p in list(sorted(paths))[:5]:
            follow.append(
                QuerySpec(
                    query=p,
                    query_type=QueryType.GET,
                    importance=max(0.4, res.spec.importance - 0.1),
                    reason=f"follow-up hop {hop}: file mentioned",
                )
            )

        # Function/class identifiers -> SEMANTIC
        idents: set[str] = set()
        for m in _IDENT_RE.finditer(text):
            ident = m.group(1)
            if len(ident) < 3:
                continue
            if ident in {"self", "None", "True", "False"}:
                continue
            idents.add(ident)

        for ident in list(sorted(idents))[:5]:
            follow.append(
                QuerySpec(
                    query=ident,
                    query_type=QueryType.SEMANTIC,
                    importance=max(0.3, res.spec.importance - 0.25),
                    reason=f"follow-up hop {hop}: symbol mentioned",
                )
            )

        return follow
