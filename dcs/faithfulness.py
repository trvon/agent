"""Ground-truth-free structured faithfulness checks.

This module builds deterministic management objects from context + output:
- EvidenceItem[] extracted from assembled context
- ClaimItem[] extracted from model output
- FaithfulnessReport with confidence/abstain signals
"""

from __future__ import annotations

import logging
import re
from typing import Any

from dcs.types import ClaimItem, ContextBlock, EvidenceItem, FaithfulnessReport, ModelConfig

try:  # optional dependency
    import dspy  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "how",
    "does",
    "into",
    "over",
    "when",
    "where",
    "which",
    "used",
    "using",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "can",
    "could",
    "should",
    "would",
    "about",
    "than",
    "then",
    "also",
    "only",
    "some",
    "many",
    "most",
    "more",
    "less",
}


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _tokenize(text: str) -> set[str]:
    toks: set[str] = set()
    for w in re.findall(r"[A-Za-z0-9_\-]+", text or ""):
        t = w.lower()
        if len(t) < 3 or t in _STOPWORDS:
            continue
        toks.add(t)
    return toks


def _extract_evidence(context: ContextBlock) -> list[EvidenceItem]:
    text = (context.content or "").strip()
    if not text:
        return []

    # Parse sections from assembler format:
    # ### [i] <source> (relevance: x)
    lines = text.splitlines()
    evidence: list[EvidenceItem] = []
    cur_source = ""
    cur_snippet: list[str] = []
    idx = 0

    def flush() -> None:
        nonlocal idx, cur_source, cur_snippet
        if not cur_source:
            return
        snippet = "\n".join(cur_snippet).strip()
        if not snippet:
            cur_source = ""
            cur_snippet = []
            return
        idx += 1
        chunk_id = ""
        if idx - 1 < len(context.chunk_ids or []):
            chunk_id = str((context.chunk_ids or [])[idx - 1])
        evidence.append(
            EvidenceItem(
                evidence_id=f"ev-{idx}",
                source=cur_source,
                snippet=snippet[:1000],
                chunk_id=chunk_id,
            )
        )
        cur_source = ""
        cur_snippet = []

    in_code = False
    for ln in lines:
        m = re.match(r"^###\s+\[\d+\]\s+(.+?)\s+\(relevance:\s*[0-9.]+\)\s*$", ln.strip())
        if m:
            flush()
            cur_source = m.group(1).strip()
            continue

        if ln.strip().startswith("```"):
            in_code = not in_code
            continue

        if not cur_source:
            continue
        if in_code:
            cur_snippet.append(ln)

    flush()

    if evidence:
        return evidence

    # Fallback: use whole context as one evidence blob.
    return [
        EvidenceItem(
            evidence_id="ev-1",
            source=(context.sources or ["context"])[0],
            snippet=text[:1000],
            chunk_id=(context.chunk_ids or [""])[0] if context.chunk_ids else "",
        )
    ]


def _extract_claims(output: str) -> list[ClaimItem]:
    out = (output or "").strip()
    if not out:
        return []

    parts = re.split(r"\n+|(?<=[.!?;])\s+", out)
    claims: list[ClaimItem] = []
    seen: set[str] = set()
    idx = 0

    for raw in parts:
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", raw.strip())
        if len(line) < 24:
            continue
        norm = re.sub(r"\s+", " ", line).strip().lower()
        if norm in seen:
            continue
        seen.add(norm)
        idx += 1
        claims.append(ClaimItem(claim_id=f"cl-{idx}", text=line))
        if len(claims) >= 16:
            break

    return claims


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _render_evidence_catalog(evidence: list[EvidenceItem], max_chars: int = 4500) -> str:
    lines: list[str] = []
    for ev in evidence:
        lines.append(f"[{ev.evidence_id}] source={ev.source}")
        lines.append(ev.snippet.strip())
        lines.append("")
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _finalize_report(
    *,
    claims: list[ClaimItem],
    evidence: list[EvidenceItem],
    min_confidence: float,
    max_unsupported_ratio: float,
    min_supported_claims: int,
    model_confidence_hint: float | None = None,
    rationale_prefix: str = "",
) -> FaithfulnessReport:
    unsupported = [c.claim_id for c in claims if not c.supported]
    supported_n = sum(1 for c in claims if c.supported)
    total_claims = len(claims)
    supported_ratio = supported_n / max(1, total_claims)

    used_evidence: set[str] = set()
    for c in claims:
        used_evidence.update(c.evidence_ids)
    evidence_cov = len(used_evidence) / max(1, len(evidence))

    conf = _clamp01((0.75 * supported_ratio) + (0.25 * evidence_cov))
    if model_confidence_hint is not None:
        conf = _clamp01((0.6 * conf) + (0.4 * _clamp01(model_confidence_hint)))

    unsupported_ratio = len(unsupported) / max(1, total_claims)
    should_abstain = (
        total_claims == 0
        or conf < float(min_confidence)
        or unsupported_ratio > float(max_unsupported_ratio)
        or supported_n < int(min_supported_claims)
    )

    rationale = (
        f"claims={total_claims}, supported={supported_n}, "
        f"supported_ratio={supported_ratio:.2f}, evidence_coverage={evidence_cov:.2f}, "
        f"confidence={conf:.2f}, unsupported_ratio={unsupported_ratio:.2f}"
    )
    if rationale_prefix:
        rationale = rationale_prefix.strip() + " | " + rationale
    if should_abstain:
        rationale += " | abstain=true"

    return FaithfulnessReport(
        claims=claims,
        evidence=evidence,
        unsupported_claim_ids=unsupported,
        supported_ratio=supported_ratio,
        evidence_coverage_ratio=evidence_cov,
        confidence=conf,
        should_abstain=should_abstain,
        rationale=rationale,
    )


def _build_report_with_dspy(
    *,
    task: str,
    context: ContextBlock,
    output: str,
    model_config: ModelConfig,
    min_confidence: float,
    max_unsupported_ratio: float,
    min_supported_claims: int,
) -> FaithfulnessReport | None:
    if dspy is None:
        return None

    evidence = _extract_evidence(context)
    if not evidence:
        return None

    evidence_catalog = _render_evidence_catalog(evidence)
    if not evidence_catalog.strip():
        return None

    # Keep DSPy inputs bounded for local models.
    task_text = (task or "")[:1200]
    answer_text = (output or "")[:3000]

    class FaithfulnessSig(dspy.Signature):
        """Ground output claims in supplied evidence IDs only."""

        task: str = dspy.InputField()
        evidence_catalog: str = dspy.InputField(
            desc="Evidence snippets keyed by [ev-N] IDs; cite only these IDs"
        )
        answer: str = dspy.InputField(desc="Candidate answer to validate")
        claims: list[str] = dspy.OutputField(desc="Atomic factual claims from answer")
        claim_to_evidence: dict[str, list[str]] = dspy.OutputField(
            desc="Map each claim text to supporting evidence IDs like ev-1"
        )
        unsupported_claims: list[str] = dspy.OutputField(
            desc="Claims with insufficient support from evidence catalog"
        )
        confidence: float = dspy.OutputField(desc="0..1 grounded confidence")
        rationale: str = dspy.OutputField()

    candidates = [model_config.name]
    if not model_config.name.startswith("openai/"):
        candidates.append(f"openai/{model_config.name}")

    pred = None
    last_err: Exception | None = None
    for model_name in candidates:
        try:
            lm = dspy.LM(
                model_name,
                api_base=model_config.base_url,
                api_key=model_config.api_key,
                temperature=0.1,
                max_tokens=int(min(700, model_config.max_output_tokens)),
                timeout=float(model_config.request_timeout_s),
            )
            dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
            pred = dspy.Predict(FaithfulnessSig)(
                task=task_text,
                evidence_catalog=evidence_catalog,
                answer=answer_text,
            )
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue

    if pred is None:
        if last_err is not None:
            logger.debug("DSPy faithfulness extraction failed: %s", last_err)
        return None

    raw_claims = [str(x).strip() for x in (getattr(pred, "claims", []) or []) if str(x).strip()]
    raw_map = getattr(pred, "claim_to_evidence", {}) or {}
    raw_unsupported = {
        _normalize_text(str(x))
        for x in (getattr(pred, "unsupported_claims", []) or [])
        if str(x).strip()
    }
    rationale = str(getattr(pred, "rationale", "") or "")
    try:
        model_conf = float(getattr(pred, "confidence", 0.0) or 0.0)
    except Exception:
        model_conf = 0.0

    valid_ids = {ev.evidence_id for ev in evidence}
    map_norm: dict[str, list[str]] = {}
    if isinstance(raw_map, dict):
        for k, v in raw_map.items():
            nk = _normalize_text(str(k))
            ids: list[str] = []
            if isinstance(v, list):
                for item in v:
                    sid = str(item).strip()
                    if sid in valid_ids and sid not in ids:
                        ids.append(sid)
            map_norm[nk] = ids

    claims: list[ClaimItem] = []
    for i, ctext in enumerate(raw_claims, start=1):
        n = _normalize_text(ctext)
        ev_ids = list(map_norm.get(n, []))
        supported = bool(ev_ids) and n not in raw_unsupported
        claims.append(
            ClaimItem(
                claim_id=f"cl-{i}",
                text=ctext,
                evidence_ids=ev_ids,
                supported=supported,
                confidence=_clamp01(model_conf if supported else model_conf * 0.5),
            )
        )

    # If DSPy emitted nothing useful, let caller fallback to deterministic path.
    if not claims:
        return None

    return _finalize_report(
        claims=claims,
        evidence=evidence,
        min_confidence=min_confidence,
        max_unsupported_ratio=max_unsupported_ratio,
        min_supported_claims=min_supported_claims,
        model_confidence_hint=model_conf,
        rationale_prefix=("dspy" + (f": {rationale}" if rationale else "")),
    )


def _match_claim_to_evidence(
    claim: ClaimItem,
    evidence: list[EvidenceItem],
    *,
    min_overlap: float,
) -> ClaimItem:
    ctoks = _tokenize(claim.text)
    if not ctoks or not evidence:
        claim.supported = False
        claim.confidence = 0.0
        return claim

    scored: list[tuple[float, str]] = []
    claim_l = claim.text.lower()
    for ev in evidence:
        etoks = _tokenize(ev.snippet)
        if not etoks:
            continue
        inter = len(ctoks.intersection(etoks))
        base = inter / max(1, len(ctoks))
        src_base = ev.source.rsplit("/", 1)[-1].lower()
        if src_base and src_base in claim_l:
            base += 0.12
        scored.append((base, ev.evidence_id))

    if not scored:
        claim.supported = False
        claim.confidence = 0.0
        return claim

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][0]
    ev_ids = [eid for s, eid in scored[:2] if s >= max(0.08, min_overlap * 0.66)]
    claim.evidence_ids = ev_ids
    claim.supported = top >= float(min_overlap)
    claim.confidence = _clamp01(0.2 + (0.7 * top) + (0.1 if ev_ids else 0.0))
    return claim


def build_faithfulness_report(
    *,
    task: str,
    context: ContextBlock,
    output: str,
    min_overlap: float = 0.12,
    min_confidence: float = 0.60,
    max_unsupported_ratio: float = 0.40,
    min_supported_claims: int = 1,
    use_dspy: bool = True,
    dspy_model_config: ModelConfig | None = None,
) -> FaithfulnessReport:
    if use_dspy and dspy_model_config is not None:
        dspy_report = _build_report_with_dspy(
            task=task,
            context=context,
            output=output,
            model_config=dspy_model_config,
            min_confidence=min_confidence,
            max_unsupported_ratio=max_unsupported_ratio,
            min_supported_claims=min_supported_claims,
        )
        if dspy_report is not None:
            return dspy_report

    evidence = _extract_evidence(context)
    claims = _extract_claims(output)

    checked: list[ClaimItem] = []
    for c in claims:
        checked.append(_match_claim_to_evidence(c, evidence, min_overlap=min_overlap))

    return _finalize_report(
        claims=checked,
        evidence=evidence,
        min_confidence=min_confidence,
        max_unsupported_ratio=max_unsupported_ratio,
        min_supported_claims=min_supported_claims,
        rationale_prefix="deterministic",
    )


def build_abstention_output(task: str, report: FaithfulnessReport) -> str:
    lines = [
        "I cannot provide a high-confidence grounded answer from the retrieved context.",
        "",
        f"Task: {task}",
        f"Faithfulness confidence: {report.confidence:.2f}",
        f"Supported claims: {sum(1 for c in report.claims if c.supported)}/{len(report.claims)}",
    ]
    if report.unsupported_claim_ids:
        lines.append("Unsupported claim ids: " + ", ".join(report.unsupported_claim_ids[:8]))
    lines.extend(
        [
            "",
            "Suggested next step: retrieve additional source-grounded evidence and rerun.",
        ]
    )
    return "\n".join(lines).strip()


__all__ = [
    "build_abstention_output",
    "build_faithfulness_report",
]
