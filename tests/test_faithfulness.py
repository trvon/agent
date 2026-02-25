from __future__ import annotations

import dcs.faithfulness as faith
from dcs.faithfulness import build_abstention_output, build_faithfulness_report
from dcs.types import ContextBlock, ModelConfig


def _mk_context(content: str) -> ContextBlock:
    return ContextBlock(
        content=content,
        sources=["src/mcp/mcp_server.cpp"],
        chunk_ids=["abc123"],
        token_count=120,
        budget=512,
        utilization=0.23,
        chunks_included=1,
        chunks_considered=1,
    )


def test_faithfulness_report_supports_grounded_claims() -> None:
    ctx = _mk_context(
        "## Retrieved Context\n\n"
        "### [1] src/mcp/mcp_server.cpp (relevance: 0.92)\n"
        "```code\n"
        'toolRegistry_->registerTool<MCPSearchRequest, MCPSearchResponse>("search", ...);\n'
        'toolRegistry_->registerTool<MCPGrepRequest, MCPGrepResponse>("grep", ...);\n'
        "```\n"
    )
    out = (
        "The MCP server registers the search and grep tools in mcp_server.cpp "
        "using registerTool for each request/response type."
    )

    rep = build_faithfulness_report(task="List MCP tools", context=ctx, output=out)
    assert rep.claims
    assert rep.evidence
    assert rep.supported_ratio > 0.5
    assert rep.confidence >= 0.6
    assert not rep.should_abstain


def test_faithfulness_report_abstains_on_unsupported_output() -> None:
    ctx = _mk_context(
        "## Retrieved Context\n\n"
        "### [1] src/vector/embedding_service.cpp (relevance: 0.88)\n"
        "```code\n"
        "class EmbeddingService {};\n"
        "```\n"
    )
    out = "The daemon uses a Kafka event stream with S3 snapshots and Terraform modules."

    rep = build_faithfulness_report(task="Describe embedding service", context=ctx, output=out)
    assert rep.claims
    assert rep.supported_ratio == 0.0
    assert rep.should_abstain


def test_build_abstention_output_contains_structured_fields() -> None:
    ctx = _mk_context("### [1] src/a.cpp (relevance: 0.9)\n```code\nfoo\n```\n")
    rep = build_faithfulness_report(task="Task text", context=ctx, output="unrelated sentence")
    txt = build_abstention_output("Task text", rep)
    assert "Faithfulness confidence:" in txt
    assert "Supported claims:" in txt


def test_faithfulness_falls_back_when_dspy_unavailable() -> None:
    ctx = _mk_context("### [1] src/a.cpp (relevance: 0.9)\n```code\nregisterTool\n```\n")
    old = faith.dspy
    faith.dspy = None
    try:
        rep = build_faithfulness_report(
            task="List tools",
            context=ctx,
            output="registerTool is used for MCP tool registration.",
            dspy_model_config=ModelConfig(name="openai/gpt-oss-20b"),
            use_dspy=True,
        )
    finally:
        faith.dspy = old

    assert rep.rationale.startswith("deterministic")
