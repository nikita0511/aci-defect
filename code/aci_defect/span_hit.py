"""Shared planted-span hit rule for judge-vs-judge comparison on ACI-Defect.

Both the objectified judges and the composite baseline are scored with this
one function so that any recall difference is attributable to the criterion
wording, not to how credit is assigned.

A judge scores a hit on a plant when it emits at least one *flagged* instance
whose quoted note span overlaps the planted span. Overlap is deliberately
generous (containment either direction, or token-F1 >= 0.5) so the baseline is
not penalised for quoting a longer or shorter window than the planter used.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "he", "she", "his", "her",
    "they", "them", "that", "this", "as", "at", "by", "it", "if", "will",
}

OVERLAP_THRESHOLD = 0.5


def norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()


def _content_tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if t not in _STOP and len(t) > 2}


def token_f1(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if not inter:
        return 0.0
    prec, rec = inter / len(tb), inter / len(ta)
    return 2 * prec * rec / (prec + rec)


def spans_overlap(planted: str, quoted: str) -> bool:
    """True when the judge's quoted span refers to the planted span."""
    np_, nq = norm(planted), norm(quoted)
    if not np_ or not nq:
        return False
    if np_ in nq or nq in np_:
        return True
    return token_f1(np_, nq) >= OVERLAP_THRESHOLD


def best_overlap(planted: str, quotes: Iterable[str]) -> float:
    return max((token_f1(planted, q) for q in quotes), default=0.0)


def score_instances(
    planted_span: str,
    instances: Iterable[dict[str, Any]],
    quote_keys: tuple[str, ...],
    flagged_values: tuple[str, ...],
    class_key: str = "classification",
) -> dict[str, Any]:
    """Score one note. Returns hit/flag counts and the localising quote."""
    flagged_quotes: list[str] = []
    all_quotes: list[str] = []
    for inst in instances or []:
        quote = ""
        for k in quote_keys:
            if inst.get(k):
                quote = str(inst[k])
                break
        if not quote:
            continue
        all_quotes.append(quote)
        cls = str(inst.get(class_key) or "").strip().lower()
        if cls in flagged_values:
            flagged_quotes.append(quote)

    hit_quote = next(
        (q for q in flagged_quotes if spans_overlap(planted_span, q)), None
    )
    return {
        "hit": hit_quote is not None,
        "hit_quote": hit_quote,
        "n_flagged": len(flagged_quotes),
        "n_instances": len(all_quotes),
        # Did the judge quote the planted span at all, even if it called it fine?
        "saw_span": any(spans_overlap(planted_span, q) for q in all_quotes),
        "best_overlap": round(best_overlap(planted_span, flagged_quotes), 3),
    }
