"""Token-budget-aware context assembler for Dynamic Context Scaffold (DCS).

The assembler takes raw YAMS retrieval results and produces a compact, ordered
context block suitable for small LLMs with tight context windows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any

try:
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

from .types import ContextBlock, QuerySpec, YAMSChunk, YAMSQueryResult

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


class ContextAssembler:
    """Budget-aware context assembler with novelty and structure heuristics."""

    _encoder_cache: dict[str, Any] = {}

    def __init__(
        self,
        budget: int = 2048,
        model: str = "gpt-4",
        config: dict | None = None,
    ):
        self.budget = int(budget)
        self.model = str(model)

        cfg = dict(config or {})
        self.decay_factor = float(cfg.get("novelty_decay", cfg.get("decay_factor", 0.85)))
        self.structural_bonus_factor = float(cfg.get("structural_bonus", 1.2))
        self.fragment_penalty_factor = float(cfg.get("fragment_penalty", 0.8))
        self.min_threshold = float(cfg.get("min_threshold", 0.05))
        self.max_chunks = int(cfg.get("max_chunks", 64))

        # Internal caches for speed.
        self._token_set_cache: dict[str, set[int]] = {}

        cache_key = self.model or "__default__"
        if cache_key not in self._encoder_cache:
            enc = None
            if tiktoken is not None:
                try:
                    enc = tiktoken.encoding_for_model(self.model)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
            self._encoder_cache[cache_key] = enc

        self._encoder = self._encoder_cache[cache_key]
        if self._encoder is None:
            logger.warning(
                "tiktoken unavailable; using approximate token counting (model=%s)",
                self.model,
            )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens using a cached tiktoken encoder."""

        if not text:
            return 0
        try:
            if self._encoder is None:
                raise RuntimeError("no_encoder")
            return len(self._encoder.encode(text))
        except Exception:
            # Defensive: if tokenization fails, fall back to a rough estimate.
            return max(1, len(text) // 4)

    @staticmethod
    def estimate_budget(
        model_config_context_window: int,
        system_prompt_tokens: int,
        output_reserve: int,
    ) -> int:
        """Estimate available retrieval budget inside a fixed context window."""

        return max(
            0, int(model_config_context_window) - int(system_prompt_tokens) - int(output_reserve)
        )

    def assemble(self, results: list[YAMSQueryResult], task: str = "") -> ContextBlock:
        budget = int(self.budget)
        if budget <= 0:
            return ContextBlock(
                content="",
                budget=budget,
                token_count=0,
                utilization=0.0,
                chunks_included=0,
                chunks_considered=0,
            )

        if not results:
            return ContextBlock(
                content="",
                budget=budget,
                token_count=0,
                utilization=0.0,
                chunks_included=0,
                chunks_considered=0,
            )

        # Flatten + deduplicate by chunk_id.
        candidates_by_id: dict[str, tuple[YAMSChunk, QuerySpec]] = {}
        total_seen = 0
        for r in results:
            spec = getattr(r, "spec", None)
            if spec is None:
                continue
            for ch in getattr(r, "chunks", []) or []:
                total_seen += 1
                cid = (ch.chunk_id or "").strip()
                if not cid:
                    continue
                content = (ch.content or "").strip("\n")
                if not content.strip():
                    continue

                prev = candidates_by_id.get(cid)
                if prev is None:
                    candidates_by_id[cid] = (replace(ch, content=content), spec)
                else:
                    # Keep the chunk/spec pairing with higher combined base score.
                    prev_chunk, prev_spec = prev
                    prev_base = _clamp01(float(prev_chunk.score)) * _clamp01(
                        float(prev_spec.importance)
                    )
                    new_base = _clamp01(float(ch.score)) * _clamp01(float(spec.importance))
                    if new_base > prev_base:
                        candidates_by_id[cid] = (replace(ch, content=content), spec)

        candidates: list[tuple[YAMSChunk, QuerySpec]] = list(candidates_by_id.values())
        if not candidates:
            return ContextBlock(
                content="",
                budget=budget,
                token_count=0,
                utilization=0.0,
                chunks_included=0,
                chunks_considered=total_seen,
            )

        # Cap candidate pool by a cheap pre-score (yams_score * importance).
        candidates.sort(
            key=lambda x: _clamp01(float(x[0].score)) * _clamp01(float(x[1].importance)),
            reverse=True,
        )
        if len(candidates) > self.max_chunks:
            candidates = candidates[: self.max_chunks]

        # Fill token_count per chunk.
        for ch, _ in candidates:
            ch.token_count = self.count_tokens(ch.content)

        selected: list[tuple[YAMSChunk, float]] = []
        remaining: list[tuple[YAMSChunk, QuerySpec]] = candidates[:]
        considered = 0
        stopped_due_to_threshold = False
        skipped_due_to_budget = 0
        truncation_failed = False

        while remaining:
            best_idx = -1
            best_score = -1.0

            # Compute scores relative to current selected set.
            selected_chunks = [c for c, _ in selected]
            for i, (ch, spec) in enumerate(remaining):
                considered += 1
                s = self._score_chunk(ch, spec, selected_chunks)
                if s > best_score:
                    best_score = s
                    best_idx = i

            if best_idx < 0:
                break
            if best_score < self.min_threshold:
                logger.debug(
                    "Stopping assembly: best_score %.4f < min_threshold %.4f",
                    best_score,
                    self.min_threshold,
                )
                stopped_due_to_threshold = True
                break

            ch, spec = remaining.pop(best_idx)

            tentative = selected + [(ch, best_score)]
            formatted = self._format_context(tentative)
            tokens_total = self.count_tokens(formatted)

            if tokens_total <= budget:
                selected = tentative
                continue

            # If nothing selected yet, truncate this single chunk to fit.
            if not selected:
                truncated_chunk = self._truncate_chunk_to_budget(ch, best_score, budget=budget)
                if truncated_chunk is not None:
                    selected = [(truncated_chunk, best_score)]
                else:
                    truncation_failed = True
                break

            # Otherwise, skip this chunk and try the next best.
            logger.debug(
                "Skipping chunk %s (would exceed budget: %d > %d)",
                ch.chunk_id,
                tokens_total,
                budget,
            )
            skipped_due_to_budget += 1

        if not selected:
            if truncation_failed or skipped_due_to_budget:
                note = "(no retrieved context: unable to fit any chunk within budget)"
            elif stopped_due_to_threshold:
                note = "(no retrieved context: all candidate chunks below threshold)"
            else:
                note = "(no retrieved context)"
            return ContextBlock(
                content=note,
                budget=budget,
                token_count=self.count_tokens(note),
                utilization=min(1.0, self.count_tokens(note) / budget) if budget else 0.0,
                chunks_included=0,
                chunks_considered=len(candidates_by_id),
            )

        # Ensure final formatting stays within budget; if not, drop from end.
        while selected:
            content = self._format_context(selected)
            tok = self.count_tokens(content)
            if tok <= budget:
                break
            selected = selected[:-1]

        if not selected:
            note = "(no retrieved context: unable to fit any chunk within budget)"
            return ContextBlock(
                content=note,
                budget=budget,
                token_count=self.count_tokens(note),
                utilization=min(1.0, self.count_tokens(note) / budget) if budget else 0.0,
                chunks_included=0,
                chunks_considered=len(candidates_by_id),
            )

        final_content = self._format_context(selected)
        final_tokens = self.count_tokens(final_content)

        sources: list[str] = []
        seen_sources: set[str] = set()
        chunk_ids: list[str] = []
        seen_ids: set[str] = set()
        for ch, _ in selected:
            if ch.source and ch.source not in seen_sources:
                sources.append(ch.source)
                seen_sources.add(ch.source)
            if ch.chunk_id and ch.chunk_id not in seen_ids:
                chunk_ids.append(ch.chunk_id)
                seen_ids.add(ch.chunk_id)

        return ContextBlock(
            content=final_content,
            sources=sources,
            chunk_ids=chunk_ids,
            token_count=final_tokens,
            budget=budget,
            utilization=(final_tokens / budget) if budget else 0.0,
            chunks_included=len(selected),
            chunks_considered=len(candidates_by_id),
        )

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------

    def _score_chunk(self, chunk: YAMSChunk, spec: QuerySpec, selected: list[YAMSChunk]) -> float:
        yams_score = _clamp01(float(chunk.score))
        importance = _clamp01(float(getattr(spec, "importance", 0.0)))
        novelty = self._compute_novelty(chunk, selected)
        structural = self._compute_structural_bonus(chunk)
        final = yams_score * importance * novelty * structural
        return float(final)

    def _compute_novelty(self, chunk: YAMSChunk, selected: list[YAMSChunk]) -> float:
        if not selected:
            return 1.0
        if self.decay_factor <= 0.0:
            return 1.0

        cand_set = self._token_set(chunk)
        if not cand_set:
            return 1.0

        union_selected: set[int] = set()
        for s in selected:
            union_selected.update(self._token_set(s))

        if not union_selected:
            return 1.0

        overlap = len(cand_set.intersection(union_selected))
        overlap_ratio = overlap / max(1, len(cand_set))

        # novelty = decay_factor ^ overlap_ratio
        novelty = float(self.decay_factor) ** float(overlap_ratio)
        # Guardrails: keep within [0.0, 1.0].
        if novelty < 0.0:
            return 0.0
        if novelty > 1.0:
            return 1.0
        return novelty

    def _compute_structural_bonus(self, chunk: YAMSChunk) -> float:
        text = (chunk.content or "").rstrip()
        if not text:
            return 1.0

        bonus = 1.0
        looks_code = self._looks_like_code(text)

        balanced = self._balanced_delimiters(text)
        ends_clean = self._ends_cleanly(text)
        fragment = self._looks_like_fragment(text)
        complete_para = self._looks_like_complete_paragraph(text)
        complete_code = looks_code and balanced and ends_clean

        if complete_code or complete_para:
            bonus *= self.structural_bonus_factor
        if fragment:
            bonus *= self.fragment_penalty_factor

        # Keep a sane range.
        if not math.isfinite(bonus) or bonus <= 0.0:
            return 1.0
        return float(bonus)

    # ---------------------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------------------

    def _format_context(self, chunks: list[tuple[YAMSChunk, float]]) -> str:
        lines: list[str] = []
        lines.append("## Retrieved Context")
        lines.append("")

        for i, (ch, score) in enumerate(chunks, start=1):
            src = (ch.source or ch.metadata.get("source") or "unknown").strip()
            lines.append(f"### [{i}] {src} (relevance: {score:.2f})")
            lines.append("```code")
            lines.append((ch.content or "").rstrip())
            lines.append("```")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _token_set(self, chunk: YAMSChunk) -> set[int]:
        cid = (chunk.chunk_id or "").strip()
        if not cid:
            # Fallback: cache by content hash for chunks without ids.
            cid = str(hash(chunk.content or ""))

        cached = self._token_set_cache.get(cid)
        if cached is not None:
            return cached

        try:
            if self._encoder is None:
                raise RuntimeError("no_encoder")
            toks = self._encoder.encode(chunk.content or "")
            s = set(int(t) for t in toks)
        except Exception:
            s = set()

        self._token_set_cache[cid] = s
        return s

    def _truncate_chunk_to_budget(
        self, chunk: YAMSChunk, score: float, budget: int
    ) -> YAMSChunk | None:
        marker = "... [truncated]"
        base_empty = replace(chunk, content="")
        base = self._format_context([(base_empty, score)])
        base_tokens = self.count_tokens(base)
        allowed_content_tokens = budget - base_tokens
        if allowed_content_tokens <= 0:
            # Try returning only the marker if even empty block doesn't fit.
            tiny = replace(chunk, content=marker)
            formatted = self._format_context([(tiny, score)])
            if self.count_tokens(formatted) <= budget:
                tiny.token_count = self.count_tokens(tiny.content)
                return tiny
            return None

        truncated = self._truncate_text_to_tokens(
            chunk.content, allowed_content_tokens, marker=marker
        )
        truncated_chunk = replace(chunk, content=truncated)
        truncated_chunk.token_count = self.count_tokens(truncated_chunk.content)

        # Verify fit; if slightly over due to encoding variance/newlines, shrink further.
        for _ in range(6):
            formatted = self._format_context([(truncated_chunk, score)])
            if self.count_tokens(formatted) <= budget:
                return truncated_chunk
            # Reduce by 10% each iteration.
            allowed_content_tokens = int(max(0, allowed_content_tokens * 0.9))
            truncated = self._truncate_text_to_tokens(
                chunk.content, allowed_content_tokens, marker=marker
            )
            truncated_chunk = replace(chunk, content=truncated)
            truncated_chunk.token_count = self.count_tokens(truncated_chunk.content)

        return None

    def _truncate_text_to_tokens(self, text: str, max_tokens: int, marker: str) -> str:
        if max_tokens <= 0:
            return marker
        text = text or ""
        if not text:
            return marker

        try:
            if self._encoder is None:
                raise RuntimeError("no_encoder")
            toks = self._encoder.encode(text)
        except Exception:
            return (text[: max(0, max_tokens * 4)] + marker).strip()

        if len(toks) <= max_tokens:
            return text

        marker_tokens = self.count_tokens(marker)
        allowed = max(0, max_tokens - marker_tokens)
        if allowed <= 0:
            return marker
        truncated = self._encoder.decode(toks[:allowed]).rstrip()  # type: ignore[union-attr]
        return f"{truncated}{marker}"

    def _looks_like_code(self, text: str) -> bool:
        # Cheap heuristics: keywords + braces + indentation + semicolons.
        if "def " in text or "class " in text:
            return True
        if "{" in text and "}" in text:
            return True
        if ";" in text and "\n" in text:
            return True
        if "```" in text:
            return True
        # Indentation-heavy blocks look like code.
        indented = sum(1 for ln in text.splitlines() if ln.startswith(("    ", "\t")))
        return indented >= 2

    def _balanced_delimiters(self, text: str) -> bool:
        # Approximate balance check. Not string-aware, but works well enough for chunks.
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]
        for o, c in pairs:
            if text.count(o) != text.count(c):
                return False
        return True

    def _ends_cleanly(self, text: str) -> bool:
        t = text.rstrip()
        if not t:
            return False
        bad_suffixes = (
            ",",
            "\\",
            "(",
            "[",
            "{",
            "->",
        )
        if t.endswith(bad_suffixes):
            return False
        if t.endswith(":"):
            # Often a suite header in Python.
            return False
        return True

    def _looks_like_fragment(self, text: str) -> bool:
        t = text.rstrip()
        if not t:
            return True
        if t.endswith("..."):
            return True
        if not self._balanced_delimiters(t):
            return True
        if t.endswith((",", "\\", "(", "[", "{", ":")):
            return True
        # Very short snippets are often fragments.
        if self.count_tokens(t) < 20 and "\n" not in t and not t.endswith((".", "!", "?")):
            return True
        return False

    def _looks_like_complete_paragraph(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        # Multi-sentence/paragraph endings.
        if t.endswith((".", "!", "?")):
            return True
        if "\n\n" in t and len(t) > 120:
            return True
        return False
