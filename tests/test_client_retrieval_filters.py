from __future__ import annotations

from pathlib import Path

from dcs.client import YAMSClient


def test_split_grep_pattern_supports_include_exclude_tokens() -> None:
    pat, path_hint, includes, excludes = YAMSClient._split_grep_pattern(
        "registerTool path:src/mcp/mcp_server.cpp include:src/mcp/* exclude:tests/*"
    )
    assert pat == "registerTool"
    assert path_hint == "src/mcp/mcp_server.cpp"
    assert includes == ["src/mcp/*"]
    assert excludes == ["tests/*"]


def test_source_matches_filters_honors_hints() -> None:
    src = "/Users/trevon/work/tools/yams/src/mcp/mcp_server.cpp"
    assert YAMSClient._source_matches_filters(
        src,
        cwd="/Users/trevon/work/tools/yams",
        include_hints=["*src/mcp/*"],
        exclude_hints=["*tests/*"],
    )
    assert not YAMSClient._source_matches_filters(
        "/Users/trevon/work/tools/yams/tests/unit/foo_test.cpp",
        cwd="/Users/trevon/work/tools/yams",
        include_hints=["*src/*"],
    )


def test_parse_graph_query_builds_node_key_for_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target = root / "src" / "mcp" / "mcp_server.cpp"
    target.parent.mkdir(parents=True)
    target.write_text("// test\n", encoding="utf-8")

    c = YAMSClient(cwd=str(root))
    args = c._parse_graph_query("src/mcp/mcp_server.cpp depth:2 limit:33")
    assert args["depth"] == 2
    assert args["limit"] == 33
    assert "node_key" in args
    assert str(target.resolve()) in str(args["node_key"])
