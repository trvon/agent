from __future__ import annotations

from dcs.executor import ModelExecutor
from dcs.types import ContextBlock, ModelConfig


def _mk_executor() -> ModelExecutor:
    ex = ModelExecutor.__new__(ModelExecutor)
    ex.config = ModelConfig(name="test-model")
    return ex


def test_build_messages_includes_response_contract_with_context() -> None:
    ex = _mk_executor()
    ctx = ContextBlock(content="example context", sources=["src/a.cpp"], chunk_ids=["c1"])
    msgs = ex._build_messages("Describe defaults and batching", ctx, None)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    sys = str(msgs[0]["content"])
    assert "# Response Contract" in sys
    assert "Answer every part of the task directly" in sys


def test_build_messages_contract_includes_task_specific_requirements() -> None:
    ex = _mk_executor()
    ctx = ContextBlock(content="example context", sources=["src/a.cpp"], chunk_ids=["c1"])
    task = "What model is default and how are embeddings stored with batch queue behavior?"
    msgs = ex._build_messages(task, ctx, None)
    sys = str(msgs[0]["content"])
    assert "state the default model name exactly" in sys
    assert "explain where and how data is stored" in sys
    assert "describe batching/queue behavior" in sys


def test_build_messages_contract_includes_mcp_tool_name_requirement() -> None:
    ex = _mk_executor()
    ctx = ContextBlock(
        content="example context", sources=["src/mcp/mcp_server.cpp"], chunk_ids=["c1"]
    )
    task = "What MCP tools are registered? List each tool name and purpose."
    msgs = ex._build_messages(task, ctx, None)
    sys = str(msgs[0]["content"])
    assert "search, grep, add, get, graph" in sys
    assert "literal terms search, grep, store, get, and graph" in sys
