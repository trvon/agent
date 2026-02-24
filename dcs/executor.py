"""Model executor for the Dynamic Context Scaffold (DCS) pipeline.

This module talks to LM Studio (or any OpenAI-compatible API) using the
official OpenAI Python SDK.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, NotFoundError

from .types import ContextBlock, ExecutionResult, ModelConfig

logger = logging.getLogger(__name__)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from thinking-mode model output.

    Models like qwen3 and deepseek-r1 emit reasoning in <think> blocks.
    The actual answer follows after the closing tag. If the model ran out of
    tokens while still thinking (no closing </think>), strip the unclosed
    <think> prefix and return whatever remains. If nothing remains after
    stripping, return the thinking content itself (best-effort).
    """
    if not text or not text.strip():
        return text.strip() if text else ""

    # First: strip closed <think>...</think> blocks
    cleaned = _THINK_RE.sub("", text).strip()
    if cleaned:
        return cleaned

    # If nothing left, check for unclosed <think> (model ran out of tokens mid-think)
    if "<think>" in text:
        # Extract content after the last <think> tag as best-effort output
        idx = text.rfind("<think>")
        inner = text[idx + len("<think>"):].strip()
        if inner:
            return inner

    return text.strip()


def format_context_prompt(context: ContextBlock) -> str:
    """Format a ContextBlock into a clean, parseable context section."""

    header = "# Retrieved Context"

    raw_content = (context.content or "").strip()
    if not raw_content:
        raw_content = "(empty)"

    # If the assembler used an explicit delimiter, keep chunks distinct. Otherwise treat as 1 chunk.
    if "\n\n---\n\n" in raw_content:
        chunk_texts = [c.strip() for c in raw_content.split("\n\n---\n\n") if c.strip()]
    elif "\n---\n" in raw_content:
        chunk_texts = [c.strip() for c in raw_content.split("\n---\n") if c.strip()]
    else:
        chunk_texts = [raw_content]

    sources = list(context.sources or [])
    chunk_ids = list(context.chunk_ids or [])

    def _pick(meta: list[str], idx: int) -> str:
        if not meta:
            return ""
        if len(meta) == 1:
            return meta[0]
        if idx < len(meta):
            return meta[idx]
        return ""

    chunks_out: list[str] = []
    for i, text in enumerate(chunk_texts, start=1):
        src = _pick(sources, i - 1)
        cid = _pick(chunk_ids, i - 1)
        cite_line_parts: list[str] = []
        if src:
            cite_line_parts.append(f"source={src}")
        if cid:
            cite_line_parts.append(f"chunk_id={cid}")
        cite_line = ""
        if cite_line_parts:
            cite_line = "[" + ", ".join(cite_line_parts) + "]"
        chunks_out.append(f"## Chunk {i} {cite_line}\n\n{text}")

    stats = (
        f"- token_count: {context.token_count}\n"
        f"- budget: {context.budget}\n"
        f"- utilization: {context.utilization:.3f}\n"
        f"- chunks_included: {context.chunks_included}\n"
        f"- chunks_considered: {context.chunks_considered}"
    )

    meta_block = "## Context Stats\n" + stats
    return f"{header}\n\n{meta_block}\n\n" + "\n\n".join(chunks_out) + "\n"


class ModelExecutor:
    """Async model executor backed by an OpenAI-compatible API."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=60.0,  # 60s hard timeout per API call
        )

    def _build_messages(
        self,
        task: str,
        context: ContextBlock | None,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        sys_parts: list[str] = []
        if system_prompt:
            sys_parts.append(system_prompt.strip())

        task_text = (task or "").strip()

        # When context is present, keep the full prompt (context + task) in the system message.
        # This makes the layout deterministic for smaller models.
        if context is not None:
            sys_parts.append(format_context_prompt(context).strip())
            sys_parts.append("# Task\n\n" + (task_text or "(empty)"))
            sys_content = "\n\n".join(p for p in sys_parts if p)
            # Append system_suffix (e.g. "/no_think" for qwen3 thinking models)
            suffix = (self.config.system_suffix or "").strip()
            if suffix:
                sys_content = sys_content + "\n" + suffix
            user_content = "Respond to the task above.".strip()
            return [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ]

        # No context: use a normal system+user split.
        if sys_parts:
            sys_content = "\n\n".join(p for p in sys_parts if p)
        else:
            sys_content = (
                "You are a helpful assistant. If you are missing required information, say so."
            )
        # Append system_suffix (e.g. "/no_think" for qwen3 thinking models)
        suffix = (self.config.system_suffix or "").strip()
        if suffix:
            sys_content = sys_content + "\n" + suffix
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": task_text},
        ]

    def _usage_from_response(self, raw: dict[str, Any]) -> tuple[int, int]:
        usage = raw.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return prompt, completion

    def _extract_text_from_response(self, raw: dict[str, Any]) -> str:
        # Chat Completions
        choices = raw.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return _strip_thinking(content)
            # Some servers may respond with list parts; join best-effort.
            if isinstance(content, list):
                parts: list[str] = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        txt = p.get("text")
                        if isinstance(txt, str):
                            parts.append(txt)
                return _strip_thinking("".join(parts))
        return ""

    async def execute(
        self,
        task: str,
        context: ContextBlock | None,
        system_prompt: str | None = None,
    ) -> ExecutionResult:
        messages = self._build_messages(
            task=task,
            context=context,
            system_prompt=system_prompt,
        )
        return await self.execute_raw(messages)

    async def execute_raw(self, messages: list[dict], **kwargs: Any) -> ExecutionResult:
        """Run a raw chat completion request.

        Supports streaming and non-streaming modes.
        Default is non-streaming for simplicity.
        """

        stream = bool(kwargs.pop("stream", False))
        model = str(kwargs.pop("model", self.config.name))
        temperature = float(kwargs.pop("temperature", self.config.temperature))
        max_tokens = int(kwargs.pop("max_tokens", self.config.max_output_tokens))

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        request_kwargs.update(kwargs)

        start = time.perf_counter()
        try:
            if not stream:
                resp = await self.client.chat.completions.create(**request_kwargs)
                raw = resp.model_dump()  # OpenAI pydantic model
                output = self._extract_text_from_response(raw)
                tokens_prompt, tokens_completion = self._usage_from_response(raw)
                latency_ms = (time.perf_counter() - start) * 1000.0
                return ExecutionResult(
                    output=output,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    model=model,
                    latency_ms=latency_ms,
                    raw_response=raw,
                )

            # Streaming: accumulate deltas, but still return a single ExecutionResult.
            output_parts: list[str] = []
            raw_chunks: list[dict[str, Any]] = []

            stream_iter = await self.client.chat.completions.create(**request_kwargs)
            async for event in stream_iter:
                # event is a ChatCompletionChunk
                try:
                    raw_event = event.model_dump()
                except Exception:  # pragma: no cover
                    raw_event = {"_unserializable": True}
                raw_chunks.append(raw_event)

                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    output_parts.append(content)

            latency_ms = (time.perf_counter() - start) * 1000.0
            output = _strip_thinking("".join(output_parts))

            # LM Studio may not provide usage in streaming chunks; leave zeros.
            return ExecutionResult(
                output=output,
                tokens_prompt=0,
                tokens_completion=0,
                model=model,
                latency_ms=latency_ms,
                raw_response={"stream": True, "chunks": raw_chunks},
            )

        except NotFoundError as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error("Model not found: %s", model, exc_info=True)
            return ExecutionResult(
                output=f"Error: model not found: {model}. Details: {e}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "model_not_found", "detail": str(e)},
            )
        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError) as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.error("Model API connection/timeout error", exc_info=True)
            return ExecutionResult(
                output=f"Error: connection/timeout talking to model API: {e}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "connection_timeout", "detail": str(e)},
            )
        except Exception as e:  # defensive: capture server quirks
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.exception("Unexpected model execution error")
            return ExecutionResult(
                output=f"Error: model execution failed: {e}",
                model=model,
                latency_ms=latency_ms,
                raw_response={"error": "unknown", "detail": str(e)},
            )

    async def list_models(self) -> list[str]:
        start = time.perf_counter()
        try:
            resp = await self.client.models.list()
            raw = resp.model_dump()
            data = raw.get("data") or []
            models: list[str] = []
            for item in data:
                mid = (item or {}).get("id")
                if isinstance(mid, str) and mid:
                    models.append(mid)
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.debug("Listed %d models in %.1fms", len(models), latency_ms)
            return models
        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError):
            logger.warning("Model list failed: API unreachable", exc_info=True)
            return []
        except Exception:
            logger.warning("Model list failed", exc_info=True)
            return []

    async def health_check(self) -> bool:
        # Cheapest signal: /models reachable.
        try:
            await self.client.models.list()
            return True
        except (APIConnectionError, APITimeoutError, asyncio.TimeoutError):
            return False
        except Exception:
            return False


__all__ = ["ModelExecutor", "format_context_prompt"]
