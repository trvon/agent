from __future__ import annotations

from dcs.pipeline import DCSPipeline
from dcs.types import PipelineConfig, QuerySpec, QueryType, YAMSChunk


def _mk_pipeline() -> DCSPipeline:
    cfg = PipelineConfig()
    cfg.max_chunks_per_query = 10
    cfg.min_chunk_score = 0.1
    return DCSPipeline(cfg)


def test_rerank_prefers_non_test_sources_for_broad_grep() -> None:
    p = _mk_pipeline()
    spec = QuerySpec(query="registerTool", query_type=QueryType.GREP, importance=0.9)
    chunks = [
        YAMSChunk(chunk_id="t1", content="registerTool test", score=0.9, source="tests/a_test.cpp"),
        YAMSChunk(
            chunk_id="p1", content="registerTool impl", score=0.6, source="src/mcp/mcp_server.cpp"
        ),
    ]

    out = p._rerank_and_cap_chunks(task="List MCP tools", spec=spec, chunks=chunks)
    assert out
    assert all("tests/" not in c.source for c in out)


def test_rerank_caps_per_source_for_broad_grep() -> None:
    p = _mk_pipeline()
    spec = QuerySpec(query="search", query_type=QueryType.GREP, importance=0.8)
    chunks = [
        YAMSChunk(
            chunk_id=f"a{i}",
            content="search implementation",
            score=0.7 - (i * 0.01),
            source="src/search/search_engine.cpp",
        )
        for i in range(5)
    ] + [
        YAMSChunk(chunk_id="b1", content="search impl", score=0.69, source="src/search/hybrid.cpp"),
    ]

    out = p._rerank_and_cap_chunks(task="Explain hybrid search", spec=spec, chunks=chunks)
    from_first = [c for c in out if c.source == "src/search/search_engine.cpp"]
    assert len(from_first) <= 2


def test_rerank_caps_get_chunks() -> None:
    p = _mk_pipeline()
    spec = QuerySpec(
        query="src/vector/embedding_service.cpp", query_type=QueryType.GET, importance=1.0
    )
    chunks = [
        YAMSChunk(
            chunk_id=f"c{i}",
            content="EmbeddingService details",
            score=0.8,
            source="src/vector/embedding_service.cpp",
        )
        for i in range(8)
    ]

    out = p._rerank_and_cap_chunks(task="Describe EmbeddingService", spec=spec, chunks=chunks)
    assert len(out) <= 4
