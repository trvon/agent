"""Async YAMS MCP client (stdio JSON-RPC over NDJSON).

This client spawns `yams serve` as a subprocess and communicates with it using the
MCP stdio transport: newline-delimited JSON-RPC 2.0 messages.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import logging
import os
import re as _re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import QuerySpec, QueryType, YAMSChunk, YAMSQueryResult

logger = logging.getLogger(__name__)


class YAMSClientError(RuntimeError):
    pass


class YAMSProtocolError(YAMSClientError):
    pass


class YAMSProcessError(YAMSClientError):
    pass


@dataclass
class _Pending:
    future: asyncio.Future[dict[str, Any]]
    method: str


class YAMSClient:
    """YAMS MCP client over stdio.

    Usage:
        async with YAMSClient() as client:
            chunks = await client.search("foo")
    """

    def __init__(
        self,
        *,
        yams_binary: str = "yams",
        data_dir: str | None = None,
        # Back-compat with callers that use PipelineConfig.yams_data_dir naming.
        yams_data_dir: str | None = None,
        # Scope search/grep results to files under this directory tree.
        cwd: str | None = None,
        # Best-effort: callers may provide fusion-style weights. We don't have a
        # first-class MCP field for these, but we can use them for heuristics.
        search_weights: dict[str, float] | None = None,
        request_timeout_s: float = 15.0,
        start_timeout_s: float = 10.0,
        stop_timeout_s: float = 5.0,
        protocol_version: str = "2025-06-18",
        client_name: str = "dcs-yams-client",
        client_version: str = "0.0.1",
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._yams_binary = yams_binary
        self._data_dir = data_dir if data_dir is not None else yams_data_dir
        self._cwd = cwd
        self._request_timeout_s = request_timeout_s
        self._start_timeout_s = start_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._protocol_version = protocol_version
        self._client_name = client_name
        self._client_version = client_version
        self._extra_env = dict(extra_env or {})

        self.search_weights: dict[str, float] = dict(search_weights or {})

        self._proc: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._wait_task: asyncio.Task[None] | None = None

        self._send_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}

        self._initialized = False
        self._stopping = False
        self._tool_names: set[str] | None = None

    @property
    def yams_binary(self) -> str:
        return self._yams_binary

    @yams_binary.setter
    def yams_binary(self, value: str) -> None:
        if self.is_running:
            raise YAMSClientError("Cannot change yams_binary while running")
        self._yams_binary = str(value)

    @property
    def yams_data_dir(self) -> str | None:
        return self._data_dir

    @yams_data_dir.setter
    def yams_data_dir(self, value: str | None) -> None:
        if self.is_running:
            raise YAMSClientError("Cannot change yams_data_dir while running")
        self._data_dir = None if value is None else str(value)

    async def __aenter__(self) -> "YAMSClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        if self._proc is not None:
            return

        env = os.environ.copy()
        # Keep stdout clean (protocol stream). Server logs to stderr.
        env.setdefault("YAMS_MCP_QUIET", "1")
        if self._data_dir is not None:
            env["YAMS_DATA_DIR"] = self._data_dir
        env.update(self._extra_env)

        logger.info("Starting yams MCP server: %s serve", self._yams_binary)
        self._proc = await asyncio.create_subprocess_exec(
            self._yams_binary,
            "serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None

        self._stdout_task = asyncio.create_task(self._read_stdout_loop(), name="yams-mcp-stdout")
        self._stderr_task = asyncio.create_task(self._read_stderr_loop(), name="yams-mcp-stderr")
        self._wait_task = asyncio.create_task(self._wait_loop(), name="yams-mcp-wait")

        try:
            await asyncio.wait_for(self._initialize_handshake(), timeout=self._start_timeout_s)
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return

        logger.info("Stopping yams MCP server")

        self._stopping = True

        # Best-effort graceful shutdown.
        try:
            if proc.returncode is None:
                try:
                    await self._request("shutdown", params={}, timeout_s=1.5)
                except Exception:
                    pass
                try:
                    await self._notify("exit", params={})
                except Exception:
                    pass

                if proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

                try:
                    await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout_s)
                except asyncio.TimeoutError:
                    logger.warning("yams serve did not exit in time; terminating")
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=self._stop_timeout_s)
                    except asyncio.TimeoutError:
                        logger.error("yams serve did not terminate; killing")
                        proc.kill()
                        await proc.wait()
        finally:
            self._proc = None
            self._initialized = False
            self._stopping = False

            for task in (self._stdout_task, self._stderr_task, self._wait_task):
                if task is not None and not task.done():
                    task.cancel()

            self._stdout_task = None
            self._stderr_task = None
            self._wait_task = None

            self._fail_all_pending(YAMSProcessError("YAMS MCP client stopped"))

    async def _initialize_handshake(self) -> None:
        init_params: dict[str, Any] = {
            "protocolVersion": self._protocol_version,
            "capabilities": {},
            "clientInfo": {"name": self._client_name, "version": self._client_version},
        }
        res = await self._request("initialize", params=init_params, timeout_s=self._start_timeout_s)
        logger.debug("MCP initialize result: %s", res)
        await self._notify("notifications/initialized", params={})
        self._initialized = True

        # Best-effort tool cache for feature detection (e.g. composite "query").
        try:
            await self.refresh_tools()
        except Exception:
            self._tool_names = None

    def _require_process(self) -> None:
        if self._proc is None:
            raise YAMSProcessError("YAMS server is not started")
        if self._proc.returncode is not None:
            raise YAMSProcessError(f"YAMS server exited with code {self._proc.returncode}")

    def _require_ready(self) -> None:
        self._require_process()
        if not self._initialized:
            raise YAMSProtocolError("MCP handshake not completed")

    async def _notify(self, method: str, *, params: dict[str, Any] | None = None) -> None:
        self._require_process()
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        await self._send_message(msg)

    async def _request(
        self,
        method: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self._require_process()
        timeout_s = self._request_timeout_s if timeout_s is None else timeout_s

        async with self._send_lock:
            req_id = self._next_id
            self._next_id += 1
            fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
            self._pending[req_id] = _Pending(future=fut, method=method)

            msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}
            await self._send_message(msg)

        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError as e:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"Timed out waiting for JSON-RPC response: {method}") from e

    async def _send_message(self, msg: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise YAMSProcessError("YAMS subprocess stdin is not available")
        if proc.returncode is not None:
            raise YAMSProcessError(f"YAMS subprocess exited with code {proc.returncode}")

        line = json.dumps(msg, ensure_ascii=True, separators=(",", ":")) + "\n"
        logger.debug("MCP send: %s", line.rstrip("\n"))
        proc.stdin.write(line.encode("utf-8"))
        await proc.stdin.drain()

    async def _read_stdout_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        stream = self._proc.stdout
        try:
            while True:
                line = await stream.readline()
                if not line:
                    # During intentional shutdown we expect stdout to close.
                    if self._stopping:
                        return
                    raise EOFError("YAMS MCP stdout closed")
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                logger.debug("MCP recv: %s", raw)
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON from yams stdout: %r", raw)
                    continue

                # Server can respond with batch arrays.
                if isinstance(msg, list):
                    for entry in msg:
                        if isinstance(entry, dict):
                            self._handle_incoming(entry)
                    continue

                if isinstance(msg, dict):
                    self._handle_incoming(msg)
        except asyncio.CancelledError:
            return
        except Exception as e:
            # Avoid noisy errors for graceful stop.
            if self._stopping and isinstance(e, EOFError):
                return
            logger.error("YAMS MCP stdout loop ended: %s", e)
            self._fail_all_pending(YAMSProcessError(str(e)))

    @staticmethod
    def _looks_like_hash(s: str) -> bool:
        s = (s or "").strip()
        if not (8 <= len(s) <= 64):
            return False
        for c in s.lower():
            if c not in "0123456789abcdef":
                return False
        return True

    async def _read_stderr_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None
        stream = self._proc.stderr
        try:
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text:
                    logger.debug("yams serve stderr: %s", text)
        except asyncio.CancelledError:
            return

    async def _wait_loop(self) -> None:
        assert self._proc is not None
        try:
            rc = await self._proc.wait()
            if rc != 0:
                logger.error("yams serve exited with code %s", rc)
            self._fail_all_pending(YAMSProcessError(f"YAMS server exited with code {rc}"))
        except asyncio.CancelledError:
            return

    def _handle_incoming(self, msg: dict[str, Any]) -> None:
        # Notifications
        if "id" not in msg:
            return

        req_id = msg.get("id")
        if not isinstance(req_id, int):
            return

        pending = self._pending.pop(req_id, None)
        if pending is None:
            return

        fut = pending.future
        if fut.done():
            return

        if "error" in msg and msg["error"] is not None:
            err = msg["error"]
            if isinstance(err, dict):
                code = err.get("code")
                message = err.get("message", "JSON-RPC error")
                data = err.get("data")
                fut.set_exception(YAMSProtocolError(f"{pending.method}: {code}: {message} ({data})"))
            else:
                fut.set_exception(YAMSProtocolError(f"{pending.method}: {err}"))
            return

        result = msg.get("result")
        if not isinstance(result, dict):
            fut.set_exception(YAMSProtocolError(f"{pending.method}: invalid result shape"))
            return

        fut.set_result(result)

    def _fail_all_pending(self, exc: Exception) -> None:
        pending = self._pending
        self._pending = {}
        for p in pending.values():
            if not p.future.done():
                p.future.set_exception(exc)

    async def _call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_ready()
        res = await self._request(
            "tools/call",
            params={"name": name, "arguments": arguments or {}},
        )
        # res is a tool-result object, not the structured data.
        return res

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the raw tool list from MCP (best-effort)."""
        self._require_ready()
        res = await self._request("tools/list", params={})
        tools = res.get("tools")
        if isinstance(tools, list):
            return [t for t in tools if isinstance(t, dict)]
        return []

    async def refresh_tools(self) -> set[str]:
        tools = await self.list_tools()
        names: set[str] = set()
        for t in tools:
            n = t.get("name")
            if isinstance(n, str) and n:
                names.add(n)
        self._tool_names = names
        return names

    async def has_tool(self, name: str) -> bool:
        if self._tool_names is None:
            try:
                await self.refresh_tools()
            except Exception:
                return False
        return name in (self._tool_names or set())

    @staticmethod
    def _get_str(mapping: dict[str, Any], key: str) -> str:
        val = mapping.get(key)
        return val if isinstance(val, str) else ""

    @staticmethod
    def _extract_tool_data(tool_result: dict[str, Any]) -> Any:
        if not isinstance(tool_result, dict):
            return tool_result

        sc = tool_result.get("structuredContent")
        if isinstance(sc, dict) and isinstance(sc.get("data"), (dict, list, str, int, float, bool)):
            return sc.get("data")

        content = tool_result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                text = first["text"]
                try:
                    return json.loads(text)
                except Exception:
                    return {"text": text}
        return tool_result

    @staticmethod
    def _chunks_from_search_data(data: Any) -> list[YAMSChunk]:
        if not isinstance(data, dict):
            return []

        if isinstance(data.get("paths"), list):
            out: list[YAMSChunk] = []
            for p in data["paths"]:
                if isinstance(p, str):
                    out.append(YAMSChunk(chunk_id=p, content=p, score=0.0, source=p))
            return out

        results = data.get("results")
        if not isinstance(results, list):
            return []

        out: list[YAMSChunk] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            path = YAMSClient._get_str(r, "path")
            h = YAMSClient._get_str(r, "hash")
            rid = YAMSClient._get_str(r, "id")
            chunk_id: str = h or rid or path
            if not chunk_id:
                chunk_id = json.dumps(r, ensure_ascii=True)
            snippet = YAMSClient._get_str(r, "snippet")
            title = YAMSClient._get_str(r, "title")
            content: str = snippet or title or path or chunk_id
            score = float(r.get("score") or 0.0)
            meta = {k: v for k, v in r.items() if k not in {"snippet", "score"}}
            out.append(YAMSChunk(chunk_id=chunk_id, content=content, score=score, source=path, metadata=meta))
        return out

    @staticmethod
    def _chunks_from_list_data(data: Any) -> list[YAMSChunk]:
        if not isinstance(data, dict):
            return []
        docs = data.get("documents")
        if not isinstance(docs, list):
            return []

        out: list[YAMSChunk] = []
        for d in docs:
            if isinstance(d, str):
                out.append(YAMSChunk(chunk_id=d, content=d, score=0.0, source=d))
                continue
            if not isinstance(d, dict):
                continue
            h = YAMSClient._get_str(d, "hash")
            path = YAMSClient._get_str(d, "path")
            name = YAMSClient._get_str(d, "name")
            chunk_id: str = h or path or name
            if not chunk_id:
                chunk_id = json.dumps(d, ensure_ascii=True)
            content: str = name or path or h or chunk_id
            out.append(YAMSChunk(chunk_id=chunk_id, content=content, score=0.0, source=path, metadata=d))
        return out

    @staticmethod
    def _parse_grep_file_paths(data: Any) -> list[tuple[str, int]]:
        """Extract unique file paths with match counts from YAMS grep output.

        Returns a list of (path, match_count) tuples sorted by match count
        descending so files with more matches (more relevant) come first.

        YAMS grep output format:
            /path/to/file.cpp (15 matches) [cpp]
              0: [R] line content here
        """
        if not isinstance(data, dict):
            return []
        output = data.get("output")
        if not isinstance(output, str) or not output.strip():
            return []

        _HEADER_RE = _re.compile(r'^(/[^\s(]+)\s*\((\d+)\s+match', _re.MULTILINE)

        seen: set[str] = set()
        entries: list[tuple[str, int]] = []
        skip_exts = {".json", ".lock", ".log", ".bin", ".dat"}

        for m in _HEADER_RE.finditer(output):
            path = m.group(1)
            count = int(m.group(2))
            if path in seen:
                continue
            # Skip non-code files
            if Path(path).suffix.lower() in skip_exts:
                continue
            seen.add(path)
            entries.append((path, count))

        # If no header-style matches found, fall back to colon-delimited parsing
        if not entries:
            for ln in output.splitlines():
                ln = ln.strip()
                if not ln or ":" not in ln:
                    continue
                path = ln.split(":", 1)[0]
                if not path or path in seen:
                    continue
                if not (path.startswith("/") or path.startswith(".")):
                    continue
                if Path(path).suffix.lower() in skip_exts:
                    continue
                seen.add(path)
                entries.append((path, 1))

        # Sort by match count descending — files with more matches are more relevant
        entries.sort(key=lambda e: e[1], reverse=True)
        return entries

    @staticmethod
    def _read_file_context(
        filepath: str,
        pattern: str,
        *,
        context_lines: int = 10,
        max_matches: int = 10,
        max_chars: int = 4000,
    ) -> str | None:
        """Read a file from disk and extract context windows around pattern matches.

        Returns a formatted string with match regions, or None if the file
        can't be read or the pattern doesn't match.
        """
        fpath = Path(filepath)
        if not fpath.is_file():
            return None
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except (OSError, PermissionError):
            return None

        lines = text.splitlines()
        if not lines:
            return None

        # Find matching line numbers
        try:
            pat = _re.compile(pattern, _re.IGNORECASE)
        except _re.error:
            # Fall back to literal substring search
            pat = None

        match_line_nums: list[int] = []
        for i, line in enumerate(lines):
            if pat is not None:
                if pat.search(line):
                    match_line_nums.append(i)
            elif pattern.lower() in line.lower():
                match_line_nums.append(i)
            if len(match_line_nums) >= max_matches:
                break

        if not match_line_nums:
            return None

        # Build context windows, merging overlapping regions
        regions: list[tuple[int, int]] = []
        for lno in match_line_nums:
            start = max(0, lno - context_lines)
            end = min(len(lines), lno + context_lines + 1)
            if regions and start <= regions[-1][1]:
                # Merge with previous region
                regions[-1] = (regions[-1][0], end)
            else:
                regions.append((start, end))

        # Format output with line numbers
        parts: list[str] = []
        total_chars = 0
        for start, end in regions:
            region_lines: list[str] = []
            for i in range(start, end):
                numbered = f"{i + 1:>5}: {lines[i]}"
                region_lines.append(numbered)
            region_text = "\n".join(region_lines)
            total_chars += len(region_text)
            if total_chars > max_chars:
                # Truncate this region to fit budget
                remaining = max_chars - (total_chars - len(region_text))
                if remaining > 100:
                    parts.append(region_text[:remaining] + "\n  ... [truncated]")
                break
            parts.append(region_text)

        if not parts:
            return None

        return "\n  ---\n".join(parts)

    @staticmethod
    def _enrich_grep_results(
        data: Any,
        pattern: str,
        *,
        path_hint: str | None = None,
        max_files: int = 12,
        context_lines: int = 10,
        max_chars_per_file: int = 4000,
    ) -> list[YAMSChunk]:
        """Parse grep output, read matched files from disk, return enriched chunks.

        Each chunk contains actual code context around match regions, not just
        file paths.  Falls back to path-only chunks for files that can't be read.
        Files are sorted by match count descending so the most relevant files
        (with the most matches) are processed first.
        """
        file_entries = YAMSClient._parse_grep_file_paths(data)
        if not file_entries:
            return []

        if path_hint:
            hint = path_hint.strip()
            if hint:
                if "/" in hint:
                    filtered = [e for e in file_entries if hint in e[0]]
                else:
                    filtered = [e for e in file_entries if Path(e[0]).name == hint]
                if filtered:
                    file_entries = filtered

        match_count = data.get("match_count", 0) if isinstance(data, dict) else 0
        file_count = data.get("file_count", 0) if isinstance(data, dict) else 0
        pattern_tag = hashlib.md5((pattern or "").encode("utf-8")).hexdigest()[:8]

        chunks: list[YAMSChunk] = []
        pat_lower = (pattern or "").lower()
        tuned_context_lines = context_lines
        tuned_max_chars = max_chars_per_file
        tuned_max_matches = 10
        if "registertool" in pat_lower:
            tuned_context_lines = 1
            tuned_max_chars = 8000
            tuned_max_matches = 24

        for i, (fpath, fmatches) in enumerate(file_entries[:max_files]):
            context = YAMSClient._read_file_context(
                fpath,
                pattern,
                context_lines=tuned_context_lines,
                max_matches=tuned_max_matches,
                max_chars=tuned_max_chars,
            )
            # Score higher for files with more matches
            base_score = min(1.0, 0.3 + (fmatches / max(1, match_count)) * 0.7) if match_count else 1.0
            if context:
                # Rich chunk with actual code context
                header = f"# {fpath} ({fmatches} matches)\n"
                content = header + context
                chunks.append(
                    YAMSChunk(
                        chunk_id=f"grep:{i}:{Path(fpath).name}:{pattern_tag}",
                        content=content,
                        score=base_score,
                        source=fpath,
                        metadata={
                            "match_count": match_count,
                            "file_count": file_count,
                            "file_matches": fmatches,
                            "enriched": True,
                        },
                    )
                )
                logger.debug(
                    "grep enriched %s: %d matches, %d chars of context",
                    fpath,
                    fmatches,
                    len(content),
                )
            else:
                # Fallback: path-only chunk (file not on disk or pattern mismatch)
                chunks.append(
                    YAMSChunk(
                        chunk_id=f"grep:{i}:{Path(fpath).name}:{pattern_tag}",
                        content=f"[file: {fpath}]",
                        score=0.3,  # lower score for path-only
                        source=fpath,
                        metadata={
                            "match_count": match_count,
                            "file_count": file_count,
                            "enriched": False,
                        },
                    )
                )
                logger.debug("grep fallback (path only) %s", fpath)

        return chunks

    @staticmethod
    def _single_chunk_json(kind: str, data: Any) -> list[YAMSChunk]:
        try:
            text = json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True)
        except Exception:
            text = str(data)
        return [YAMSChunk(chunk_id=kind, content=text, score=0.0, source=kind, metadata={})]

    async def search(self, query: str, limit: int = 10, **kwargs: Any) -> list[YAMSChunk]:
        args = {"query": query, "limit": limit}

        # WORKAROUND: hybrid search via MCP returns 0 results (daemon bug).
        # Default to keyword until hybrid is fixed.  Callers can still pass
        # type="hybrid" explicitly to re-test.
        if "type" not in kwargs:
            args["type"] = "keyword"

        # Scope to cwd if configured and not overridden by caller.
        if self._cwd and "cwd" not in kwargs:
            args["cwd"] = self._cwd

        args.update(kwargs)
        tool_result = await self._call_tool("search", args)
        data = self._extract_tool_data(tool_result)
        chunks = self._chunks_from_search_data(data)
        # YAMS keyword search doesn't include FTS5 rank scores in the MCP
        # response, so all chunks arrive with score=0.0.  Assign positional
        # scores (1.0 → 0.1) so the assembler can differentiate them.
        self._backfill_positional_scores(chunks)
        return chunks

    @staticmethod
    def _backfill_positional_scores(chunks: list[YAMSChunk]) -> None:
        """Assign descending positional scores when YAMS returns all-zero scores."""
        if not chunks:
            return
        if any(c.score > 0.0 for c in chunks):
            return  # real scores present; leave as-is
        n = len(chunks)
        for i, c in enumerate(chunks):
            # Linear decay: first result = 1.0, last = max(0.1, 1/n)
            c.score = max(0.1, 1.0 - (i / max(1, n)))

    async def grep(self, pattern: str, **kwargs: Any) -> list[YAMSChunk]:
        pat, path_hint = self._split_grep_pattern(pattern)
        args = {"pattern": pat}
        # Scope to cwd if configured and not overridden by caller.
        if self._cwd and "cwd" not in kwargs:
            args["cwd"] = self._cwd
        args.update(kwargs)
        tool_result = await self._call_tool("grep", args)
        data = self._extract_tool_data(tool_result)
        chunks = self._enrich_grep_results(data, pat, path_hint=path_hint)
        # Assign positional scores so the assembler can rank earlier
        # (more relevant) matches higher.
        self._backfill_positional_scores(chunks)
        return chunks

    @staticmethod
    def _split_grep_pattern(pattern: str) -> tuple[str, str | None]:
        """Split a grep pattern from an optional path hint.

        Accepts formats like:
          "registerTool path:mcp_server.cpp"
          "registerTool path:src/mcp/mcp_server.cpp"
        """
        raw = (pattern or "").strip()
        if not raw:
            return "", None

        m = _re.search(r"\bpath:([^\s]+)", raw)
        if not m:
            return raw, None

        hint = m.group(1).strip().strip("'\"")
        # Remove the path hint token from the pattern
        cleaned = _re.sub(r"\s*\bpath:[^\s]+", "", raw).strip()
        return cleaned or raw, hint

    async def graph(self, query: str) -> list[YAMSChunk]:
        # YAMS graph tool expects a structured request. For DCS we accept a single string and
        # treat it as either a document hash or a name.
        q = (query or "").strip()
        args: dict[str, Any]
        if self._looks_like_hash(q):
            args = {"hash": q}
        else:
            args = {"name": q}
        tool_result = await self._call_tool("graph", args)
        data = self._extract_tool_data(tool_result)
        return self._single_chunk_json("graph", data)

    async def graph_query(self, **kwargs: Any) -> dict[str, Any]:
        """Low-level graph tool call with full parameter control.

        Accepts any parameters the YAMS graph tool supports:
          - list_types: bool (return node type counts)
          - list_type: str (list nodes of a specific type)
          - node_key: str (e.g. "file:/path/to/file.cpp")
          - name: str, hash: str, node_id: int
          - relation: str (filter to specific edge type)
          - depth: int (BFS depth, 1-5)
          - reverse: bool (walk incoming edges)
          - limit: int, offset: int
          - include_properties: bool
          - isolated: bool (find disconnected nodes)
          - action: str ("query" or "ingest")

        Returns the raw structured data dict from the tool response.
        """
        tool_result = await self._call_tool("graph", kwargs)
        data = self._extract_tool_data(tool_result)
        if isinstance(data, dict):
            return data
        return {"raw": data}

    async def get(self, name_or_hash: str) -> YAMSChunk | None:
        q = (name_or_hash or "").strip()
        args: dict[str, Any]
        try_hash_first = self._looks_like_hash(q)

        def hash_args() -> dict[str, Any]:
            return {"hash": q, "include_content": True}

        def name_args() -> dict[str, Any]:
            return {"name": q, "include_content": True}

        try:
            tool_result = await self._call_tool("get", hash_args() if try_hash_first else name_args())
        except YAMSProtocolError as e:
            msg = str(e).lower()
            not_found = "not found" in msg or "missing" in msg
            if not not_found:
                raise
            # Fallback: try the other selector.
            tool_result = await self._call_tool("get", name_args() if try_hash_first else hash_args())

        data = self._extract_tool_data(tool_result)
        if not isinstance(data, dict):
            return None
        content: str = self._get_str(data, "content")
        h = self._get_str(data, "hash")
        path = self._get_str(data, "path")
        name = self._get_str(data, "name")
        chunk_id: str = h or path or name or q
        source: str = path
        return YAMSChunk(chunk_id=chunk_id, content=content, score=0.0, source=source, metadata=data)

    async def list_docs(self, **kwargs: Any) -> list[YAMSChunk]:
        tool_result = await self._call_tool("list", kwargs)
        data = self._extract_tool_data(tool_result)
        return self._chunks_from_list_data(data)

    async def add(
        self,
        content: str,
        name: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        args: dict[str, Any] = {"content": content, "name": name}
        if tags is not None:
            args["tags"] = tags
        if metadata is not None:
            args["metadata"] = metadata

        tool_result = await self._call_tool("add", args)
        data = self._extract_tool_data(tool_result)
        if isinstance(data, dict) and isinstance(data.get("hash"), str) and data["hash"]:
            return data["hash"]
        raise YAMSProtocolError(f"add: unexpected response: {data}")

    async def status(self) -> dict[str, Any]:
        tool_result = await self._call_tool("status", {})
        data = self._extract_tool_data(tool_result)
        if isinstance(data, dict):
            return data
        return {"data": data}

    async def pipeline(self, steps: list[dict[str, Any]]) -> list[Any]:
        """Execute a multi-step read-only pipeline.

        Steps format: [{"op": "search", "params": {...}}, ...]
        Supports "$prev" expansion (server-side when the composite "query" tool exists;
        client-side fallback otherwise).
        """

        # Prefer server-side pipeline if available.
        if await self.has_tool("query"):
            try:
                tool_result = await self._call_tool("query", {"steps": steps})
                data = self._extract_tool_data(tool_result)
                return self._pipeline_results_from_query_data(data)
            except YAMSProtocolError as e:
                msg = str(e)
                if "Unknown tool" not in msg and "Method not found" not in msg and "not found" not in msg:
                    raise
                logger.debug("Falling back to client-side pipeline: %s", e)

        return await self._pipeline_client_side(steps)

    @staticmethod
    def _pipeline_results_from_query_data(data: Any) -> list[Any]:
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            out: list[Any] = []
            for step in data["steps"]:
                if isinstance(step, dict) and "result" in step:
                    out.append(step["result"])
            return out
        return [data]

    async def _pipeline_client_side(self, steps: list[dict[str, Any]]) -> list[Any]:
        prev: Any = {}
        results: list[Any] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict) or "op" not in step:
                raise YAMSProtocolError(f"pipeline: step {i} missing 'op'")
            op = step["op"]
            params = step.get("params", {})
            if not isinstance(params, dict):
                params = {}
            params = self._resolve_prev_refs(params, prev)
            tool_result = await self._call_tool(str(op), params)
            prev = self._extract_tool_data(tool_result)
            results.append(prev)
        return results

    @staticmethod
    def _resolve_prev_refs(params: dict[str, Any], prev: Any) -> dict[str, Any]:
        # Mirror the server-side rules: replace string values starting with "$prev".
        def resolve_one(s: str) -> Any:
            if s == "$prev":
                return prev
            if not s.startswith("$prev"):
                return s
            cur: Any = prev
            path = s[5:]
            while path and cur is not None:
                if path.startswith("."):
                    path = path[1:]
                    end = len(path)
                    for j, ch in enumerate(path):
                        if ch in ".[":
                            end = j
                            break
                    key = path[:end]
                    path = path[end:]
                    if isinstance(cur, dict):
                        cur = cur.get(key)
                    else:
                        return None
                elif path.startswith("["):
                    close = path.find("]")
                    if close == -1:
                        return None
                    idx_str = path[1:close]
                    path = path[close + 1 :]
                    try:
                        idx = int(idx_str)
                    except Exception:
                        return None
                    if isinstance(cur, list) and 0 <= idx < len(cur):
                        cur = cur[idx]
                    else:
                        return None
                else:
                    return None
            return cur

        resolved: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("$prev"):
                resolved[k] = resolve_one(v)
            else:
                resolved[k] = v
        return resolved

    async def execute_spec(self, spec: QuerySpec) -> YAMSQueryResult:
        t0 = time.perf_counter()
        try:
            if spec.query_type == QueryType.SEMANTIC:
                chunks = await self.search(spec.query)
            elif spec.query_type == QueryType.GREP:
                chunks = await self.grep(spec.query)
            elif spec.query_type == QueryType.GRAPH:
                chunks = await self.graph(spec.query)
            elif spec.query_type == QueryType.GET:
                ch = await self.get(spec.query)
                chunks = [ch] if ch is not None else []
            elif spec.query_type == QueryType.LIST:
                chunks = await self.list_docs(pattern=spec.query, limit=50)
            else:
                raise ValueError(f"Unknown QueryType: {spec.query_type}")

            return YAMSQueryResult(
                spec=spec,
                chunks=chunks,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=None,
            )
        except Exception as e:
            return YAMSQueryResult(
                spec=spec,
                chunks=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(e),
            )
