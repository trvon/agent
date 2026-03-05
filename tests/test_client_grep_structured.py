from __future__ import annotations

from pathlib import Path

from dcs.client import YAMSClient


def test_parse_grep_file_paths_prefers_structured_matches() -> None:
    data = {
        "matches": [
            {"file": "src/mcp/stdio_transport.cpp", "file_matches": 3},
            {"file": "src/mcp/mcp_server.cpp", "file_matches": 12},
            {"file": "src/mcp/mcp_server.cpp", "file_matches": 12},
        ],
        "output": "",
    }

    entries = YAMSClient._parse_grep_file_paths(data)
    assert entries[0] == ("src/mcp/mcp_server.cpp", 12)
    assert entries[1] == ("src/mcp/stdio_transport.cpp", 3)


def test_enrich_grep_results_uses_structured_matches() -> None:
    data = {
        "match_count": 2,
        "matches": [
            {
                "file": "src/mcp/mcp_server.cpp",
                "line_number": 4245,
                "line_text": (
                    "toolRegistry_->registerTool<MCPSearchRequest, MCPSearchResponse>("
                    '"search", ...);'
                ),
                "context_before": ["// Register core tools"],
                "context_after": ["// More registrations"],
                "file_matches": 2,
            },
            {
                "file": "src/mcp/mcp_server.cpp",
                "line_number": 4263,
                "line_text": (
                    "toolRegistry_->registerTool<MCPUpdateMetadataRequest, "
                    'MCPUpdateMetadataResponse>("update_metadata", ...);'
                ),
                "context_before": [],
                "context_after": [],
                "file_matches": 2,
            },
        ],
    }

    chunks = YAMSClient._enrich_grep_results(data, "registerTool")
    assert len(chunks) == 1
    ch = chunks[0]
    assert "mcp_server.cpp" in ch.source
    assert "registerTool" in ch.content
    assert "4245:" in ch.content
    assert ch.metadata.get("structured") is True
    assert ch.metadata.get("enriched") is True


def test_structured_empty_lines_fallback_to_file_context(tmp_path: Path) -> None:
    fp = tmp_path / "sample.cpp"
    fp.write_text(
        "int x = 0;\nvoid registerTool() {}\nvoid other() {}\n",
        encoding="utf-8",
    )

    data = {
        "matches": [
            {
                "file": str(fp),
                "line_number": 0,
                "line_text": "",
                "context_before": [],
                "context_after": [],
                "file_matches": 1,
            }
        ]
    }

    chunks = YAMSClient._enrich_grep_results(data, "registerTool")
    assert len(chunks) == 1
    assert "registerTool" in chunks[0].content
    assert ": void registerTool() {}" in chunks[0].content
