from __future__ import annotations

import pytest

from dcs.planner import QueryPlanner
from dcs.types import QuerySpec, QueryType, YAMSChunk, YAMSQueryResult


class _StubYAMS:
    async def search(self, query, limit=10, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def grep(self, pattern, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def graph(self, query):  # pragma: no cover - protocol shim
        return []

    async def get(self, name_or_hash):  # pragma: no cover - protocol shim
        return None

    async def list_docs(self, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def execute_spec(self, spec):  # pragma: no cover - not used in this unit test
        return YAMSQueryResult(spec=spec, chunks=[])


def test_adaptive_followups_add_graph_from_file_chunks() -> None:
    planner = QueryPlanner(_StubYAMS())
    res = YAMSQueryResult(
        spec=QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.9),
        chunks=[
            YAMSChunk(
                chunk_id="c1",
                content="registerTool",
                score=0.9,
                source="/Users/trevon/work/tools/yams/src/mcp/mcp_server.cpp",
                metadata={"enriched": True},
            )
        ],
    )

    follow = planner._adaptive_followups([res])
    graph = [s for s in follow if s.query_type == QueryType.GRAPH]
    assert graph
    assert "mcp_server.cpp" in graph[0].query


class _StageStubYAMS:
    def __init__(self) -> None:
        self.calls: list[tuple[QueryType, str]] = []

    async def search(self, query, limit=10, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def grep(self, pattern, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def graph(self, query):  # pragma: no cover - protocol shim
        return []

    async def get(self, name_or_hash):  # pragma: no cover - protocol shim
        return None

    async def list_docs(self, **kwargs):  # pragma: no cover - protocol shim
        return []

    async def execute_spec(self, spec):
        self.calls.append((spec.query_type, spec.query))
        if spec.query_type == QueryType.SEMANTIC:
            return YAMSQueryResult(
                spec=spec,
                chunks=[
                    YAMSChunk(
                        chunk_id="s1",
                        content="semantic hit",
                        source="/repo/src/mcp/mcp_server.cpp",
                        score=0.95,
                    )
                ],
            )
        if spec.query_type == QueryType.GRAPH:
            return YAMSQueryResult(
                spec=spec,
                chunks=[
                    YAMSChunk(
                        chunk_id="g1",
                        content="[file] /repo/src/mcp/mcp_server.cpp",
                        source="/repo/src/mcp/mcp_server.cpp",
                        score=0.9,
                    )
                ],
            )
        return YAMSQueryResult(spec=spec, chunks=[])


@pytest.mark.asyncio
async def test_execute_runs_search_then_graph_then_grep_get() -> None:
    yams = _StageStubYAMS()
    planner = QueryPlanner(yams)
    specs = [
        QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.8),
        QuerySpec(query="MCP tool registration", query_type=QueryType.SEMANTIC, importance=0.95),
        QuerySpec(query="mcp_server.cpp", query_type=QueryType.GET, importance=0.7),
    ]

    await planner.execute(specs, allow_adaptive=False)

    call_types = [qt for qt, _ in yams.calls]
    first_graph_idx = call_types.index(QueryType.GRAPH)
    first_grep_idx = call_types.index(QueryType.GREP)
    first_get_idx = call_types.index(QueryType.GET)

    assert call_types[0] == QueryType.SEMANTIC
    assert first_graph_idx < first_grep_idx
    assert first_graph_idx < first_get_idx


@pytest.mark.asyncio
async def test_execute_adds_graph_guided_grep_and_get_specs() -> None:
    yams = _StageStubYAMS()
    planner = QueryPlanner(yams)
    specs = [
        QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.8),
        QuerySpec(query="MCP tool registration", query_type=QueryType.SEMANTIC, importance=0.95),
        QuerySpec(query="mcp_server.cpp", query_type=QueryType.GET, importance=0.7),
    ]

    await planner.execute(specs, allow_adaptive=False)

    grep_queries = [q for qt, q in yams.calls if qt == QueryType.GREP]
    get_queries = [q for qt, q in yams.calls if qt == QueryType.GET]

    assert any("path:/repo/src/mcp/mcp_server.cpp" in q for q in grep_queries)
    assert "/repo/src/mcp/mcp_server.cpp" in get_queries
