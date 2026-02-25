from __future__ import annotations

from dcs.decomposer import TaskDecomposer
from dcs.types import QuerySpec, QueryType


def _mk_decomposer() -> TaskDecomposer:
    # bypass network client init; tests only use pure helper methods
    return TaskDecomposer.__new__(TaskDecomposer)


def test_apply_type_bias_reweights_importance() -> None:
    d = _mk_decomposer()
    specs = [
        QuerySpec(query="foo", query_type=QueryType.SEMANTIC, importance=0.8),
        QuerySpec(query="bar", query_type=QueryType.GREP, importance=0.5),
    ]
    out = d._apply_type_bias(
        specs,
        {
            QueryType.SEMANTIC.value: 0.5,
            QueryType.GREP.value: 1.5,
        },
    )
    assert out[0].importance == 0.4
    assert out[1].importance == 0.75


def test_ensure_required_types_adds_grep_and_get() -> None:
    d = _mk_decomposer()
    task = "Find tools registered in mcp_server.cpp"
    specs = [QuerySpec(query="architecture", query_type=QueryType.SEMANTIC, importance=0.6)]
    out = d._ensure_required_types(task, specs, {QueryType.GREP, QueryType.GET})

    qtypes = {s.query_type for s in out}
    assert QueryType.GREP in qtypes
    assert QueryType.GET in qtypes


def test_inject_task_specific_specs_for_mcp_protocol() -> None:
    d = _mk_decomposer()
    specs = [QuerySpec(query="mcp architecture", query_type=QueryType.SEMANTIC, importance=0.5)]
    task = "How is MCP started, which transport is used, and how do clients communicate?"

    out = d._inject_task_specific_specs(task, specs)
    queries = {s.query for s in out}

    assert "registerTool" in queries
    assert "ndjson" in queries
    assert "json-rpc" in queries
    assert "stdio_transport" in queries
    assert "serve_command" in queries


def test_inject_task_specific_specs_for_embedding_service() -> None:
    d = _mk_decomposer()
    specs = [
        QuerySpec(query="embedding architecture", query_type=QueryType.SEMANTIC, importance=0.5)
    ]
    task = (
        "How does the YAMS EmbeddingService work? What model does it use by default, "
        "how are embeddings stored, and how does batch processing work?"
    )

    out = d._inject_task_specific_specs(task, specs)
    queries = {s.query for s in out}

    assert "EmbeddingService" in queries
    assert "all-MiniLM-L6-v2" in queries
    assert "generateEmbeddingsInternal" in queries
    assert "src/vector/embedding_service.cpp" in queries
    assert "include/yams/vector/dim_resolver.h" in queries


def test_inject_task_specific_specs_for_knowledge_graph() -> None:
    d = _mk_decomposer()
    specs = [QuerySpec(query="kg architecture", query_type=QueryType.SEMANTIC, importance=0.5)]
    task = "How does the YAMS knowledge graph work and how is it used during search?"

    out = d._inject_task_specific_specs(task, specs)
    queries = {s.query for s in out}

    assert "KnowledgeGraphStore|knowledge graph|KG" in queries
    assert "node|edge|relation path:src/metadata/knowledge_graph_store_sqlite.cpp" in queries
    assert "search|query|travers path:src/metadata/knowledge_graph_store_sqlite.cpp" in queries
    assert "src/metadata/knowledge_graph_store_sqlite.cpp" in queries
    assert "src/metadata/knowledge_graph_store_sqlite.cpp depth:1 limit:40" in queries


def test_inject_task_specific_specs_for_storage_model() -> None:
    d = _mk_decomposer()
    specs = [QuerySpec(query="storage", query_type=QueryType.SEMANTIC, importance=0.4)]
    task = (
        "How does YAMS store documents? Explain the content-addressable storage "
        "model, how documents are indexed, and how metadata is tracked."
    )

    out = d._inject_task_specific_specs(task, specs)
    queries = {s.query for s in out}

    assert "include/yams/api/content_store.h" in queries
    assert "src/api/content_store_impl.cpp" in queries
    assert "src/metadata/metadata_repository.cpp" in queries
    assert "content-addressable|sha256|metadata|index" in queries


def test_inject_task_specific_specs_for_daemon_architecture() -> None:
    d = _mk_decomposer()
    specs = [QuerySpec(query="daemon", query_type=QueryType.SEMANTIC, importance=0.4)]
    task = (
        "Describe the YAMS daemon architecture. What major components does it have, "
        "how do they coordinate, and what role does the event bus play in the system?"
    )

    out = d._inject_task_specific_specs(task, specs)
    queries = {s.query for s in out}

    assert "include/yams/daemon/components/ServiceManager.h" in queries
    assert "include/yams/daemon/components/InternalEventBus.h" in queries
    assert "include/yams/daemon/components/ResourceGovernor.h" in queries
