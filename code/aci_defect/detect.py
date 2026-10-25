"""Span-localized objectified detectors for ACI-Defect plants.

These encode the *determinate decision* of each target evaluator, not the
production LLM prompt. P8 uses the same brand-drug capitalization rule as
production deterministic style checks. Detection is instance-level: did the
detector flag the planted span (not merely note-level fail)?
"""

from __future__ import annotations

import re
from typing import Any

from .sections import parse_note, split_units

_SYMPTOM_CUES = re.compile(
    r"(?i)\b(reports?|denies|endorses|pain|aching|worse in the evening|symptom)\b"
)
_IMAGING_CUES = re.compile(
    r"(?i)\b(ct |mri |x-ray|radiograph|ultrasound|interpretation|no acute abnormality)\b"
)
_HISTORY_OF_RE = re.compile(r"(?i)\bhistory of\b")
_PLAN_CUES = re.compile(
    r"(?i)\b(we will|i will|i recommend|will increase|will continue|will order|"
    r"will refer|prescribe|mg (?:once |twice )?daily|follow[- ]up)\b"
)
_HPI_CUES = re.compile(
    r"(?i)\b(reports?|denies|endorses|states|started to|over the (?:last |past )?|"
    r"for the past)\b"
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def span_in_section(note: str, span: str, section_key: str) -> bool:
    doc = parse_note(note)
    body = doc.ap_body() if section_key == "ap" else doc.body(section_key)
    return _norm(span) in _norm(body) if span else False


def detect_p1(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_hpi = span_in_section(note, span, "hpi")
    still_in_ap = span_in_section(note, span, "ap")
    flagged = in_hpi and (not still_in_ap) and bool(_PLAN_CUES.search(span) or True)
    return {
        "detected": bool(in_hpi and not still_in_ap),
        "rationale": (
            f"plan-language span in HPI={in_hpi}, remaining in A/P={still_in_ap}"
        ),
        "localized_to_plant": in_hpi,
    }


def detect_p2(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_ap = span_in_section(note, span, "ap")
    still_in_hpi = span_in_section(note, span, "hpi")
    return {
        "detected": bool(in_ap and not still_in_hpi),
        "rationale": f"HPI narrative in A/P={in_ap}, remaining in HPI={still_in_hpi}",
        "localized_to_plant": in_ap,
    }


def detect_p3(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_pe = span_in_section(note, span, "pe")
    subjective = bool(_SYMPTOM_CUES.search(span))
    return {
        "detected": bool(in_pe and subjective),
        "rationale": f"symptom sentence in PE={in_pe}",
        "localized_to_plant": in_pe,
    }


def detect_p4(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_pe = span_in_section(note, span, "pe")
    in_results = span_in_section(note, span, "results")
    return {
        "detected": bool(in_pe and in_results),
        "rationale": f"RESULTS sentence copied into PE={in_pe}, still in RESULTS={in_results}",
        "localized_to_plant": in_pe,
    }


def detect_p5(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_pe = span_in_section(note, span, "pe")
    was_absent = span.lower() not in (orig.get("note") or "").lower()
    t = (orig.get("transcript") or "").lower()
    tokens = [
        w
        for w in re.findall(r"[a-z]{4,}", span.lower())
        if w not in {"with", "that", "this", "from", "have", "been", "were", "they", "them"}
    ]
    support = sum(1 for w in tokens if w in t) / max(1, len(tokens))
    return {
        "detected": bool(in_pe and was_absent),
        "rationale": f"fabricated PE finding in PE={in_pe}, absent from gold={was_absent}, transcript support={support:.2f}",
        "localized_to_plant": in_pe,
        "transcript_support": round(support, 3),
    }


def detect_p6(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    mod = plant.get("planted_span") or ""
    before = plant.get("modifier_count_before")
    after = plant.get("modifier_count_after")
    if before is None:
        before = len(re.findall(r"(?i)" + re.escape(mod), orig.get("note") or "")) if mod else 0
        after = len(re.findall(r"(?i)" + re.escape(mod), note)) if mod else 0
    dropped = (after or 0) == 0 and (before or 0) >= 1
    return {
        "detected": bool(dropped),
        "rationale": f"modifier {mod!r} count {before}→{after}",
        "localized_to_plant": dropped,
    }


def detect_p7(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    label = plant.get("active_problem") or ""
    in_ap = span_in_section(note, span, "ap") or (
        _HISTORY_OF_RE.search(parse_note(note).ap_body() or "") is not None
        and label.lower() in (parse_note(note).ap_body() or "").lower()
    )
    return {
        "detected": bool(in_ap),
        "rationale": f"'history of' + active problem {label!r} in A/P={in_ap}",
        "localized_to_plant": in_ap,
    }


def detect_p8(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    drugs = plant.get("drugs") or []
    hits = []
    for d in drugs:
        if re.search(r"\b" + re.escape(d.lower()) + r"\b", note) and not re.search(
            r"\b" + re.escape(d) + r"\b", note
        ):
            hits.append(d)
    return {
        "detected": bool(hits),
        "rationale": f"uncapitalized brand drugs: {hits}",
        "localized_to_plant": bool(hits),
        "flagged_drugs": hits,
    }


def detect_p9(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    labels = plant.get("labels") or []
    in_ap = span_in_section(note, span, "ap") or (
        " and " in span and span.rstrip(".") in parse_note(note).ap_body()
    )
    bundled = bool(labels) and all(
        lab.lower() in (parse_note(note).ap_body() or "").lower() for lab in labels
    )
    return {
        "detected": bool(in_ap or (bundled and " and " in (parse_note(note).ap_body() or ""))),
        "rationale": f"bundled heading {span!r} in A/P",
        "localized_to_plant": bool(in_ap),
    }


def detect_p10(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    cc = parse_note(note).body("cc")
    words = re.findall(r"[A-Za-z0-9']+", cc or "")
    narrative = bool(re.search(r"(?i)\b(who presents|the patient is)\b", cc or ""))
    fail = len(words) > 12 or narrative
    return {
        "detected": bool(fail),
        "rationale": f"CC words={len(words)} narrative={narrative} (max 12, label-not-sentence)",
        "localized_to_plant": fail,
        "word_count": len(words),
    }


def detect_p11(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_note = _norm(span) in _norm(note) if span else False
    orig_has = _norm(span) in _norm(orig.get("note") or "")
    return {
        "detected": bool(in_note and not orig_has),
        "rationale": f"swapped sex span {span!r} in note={in_note} absent from gold={not orig_has}",
        "localized_to_plant": in_note,
    }


def detect_p12(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    span = plant.get("planted_span") or ""
    in_note = _norm(span) in _norm(note) if span else False
    has_i = bool(re.search(r"\bI\b", span or ""))
    return {
        "detected": bool(in_note and has_i),
        "rationale": f"first-person plant in note={in_note}",
        "localized_to_plant": in_note,
    }


def detect_ctrl(note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    return {
        "detected": False,
        "rationale": "generic control — no objectified evaluator",
        "localized_to_plant": False,
    }


DETECTORS = {
    "P1": detect_p1,
    "P2": detect_p2,
    "P3": detect_p3,
    "P4": detect_p4,
    "P5": detect_p5,
    "P6": detect_p6,
    "P7": detect_p7,
    "P8": detect_p8,
    "P9": detect_p9,
    "P10": detect_p10,
    "P11": detect_p11,
    "P12": detect_p12,
    "CTRL": detect_ctrl,
}


def detect_plant(plant_id: str, note: str, plant: dict[str, Any], orig: dict[str, Any]) -> dict[str, Any]:
    fn = DETECTORS[plant_id]
    return fn(note, plant, orig)
