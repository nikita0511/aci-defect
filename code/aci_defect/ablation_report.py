"""Scoring helpers shared by the composite-vs-objectified ablation.

Kept separate from the runners so that the released archive and the internal
scoring pass apply the same rules to the same judge output.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from aci_defect.span_hit import score_instances

PLANT_FILES = {
    "P1": "P1_ap_into_hpi",
    "P2": "P2_hpi_into_ap",
    "P3": "P3_symptom_into_pe",
    "P4": "P4_imaging_into_pe",
}

PLANT_TITLE = {
    "P1": "A&P sentence into History",
    "P2": "HPI narrative into A&P",
    "P3": "Symptom sentence into PE",
    "P4": "RESULTS sentence into PE",
}

# Buckets are assigned from the composite judge's *own* stated rationale, to
# characterise where an un-scoped criterion sends content. The first three are
# content the objectified rubric explicitly lists as acceptable in History.
RUBRIC_ACCEPTABLE = (
    "ROS / pertinent negatives out of History",
    "Past/family/social history out of History",
    "Chronic-condition narrative out of History",
)


def rows_of(blob: Any) -> list[dict]:
    return blob["results"] if isinstance(blob, dict) else blob


def planted_spans(plants_dir: Path, plant_file: str) -> dict[str, str]:
    blob = json.loads((plants_dir / f"{plant_file}.json").read_text())
    return {i["note_id"]: (i["plant"].get("planted_span") or "") for i in blob["instances"]}


def score_arm(
    spans: dict[str, str],
    rows: list[dict],
    quote_key: str,
    flagged: tuple[str, ...],
) -> dict[str, Any]:
    """Planted-span recall and flag burden for one arm on one plant."""
    n = hits = note_fail = parse_fail = 0
    flag_counts: list[int] = []
    for r in rows:
        nid = r.get("note_id")
        if nid not in spans:
            continue
        n += 1
        er = r.get("eval_result")
        if not er:
            parse_fail += 1
            flag_counts.append(0)
            continue
        if er.get("is_failure"):
            note_fail += 1
        s = score_instances(spans[nid], er.get("instances"), (quote_key,), flagged)
        hits += s["hit"]
        flag_counts.append(s["n_flagged"])
    return {
        "n": n,
        "hits": hits,
        "recall": hits / n if n else None,
        "note_fail": note_fail,
        "note_fail_rate": note_fail / n if n else None,
        "mean_flags": statistics.fmean(flag_counts) if flag_counts else 0.0,
        "parse_fail": parse_fail,
    }


def note_flag_rate(rows: list[dict], flagged_value: str, keep: set[str] | None = None) -> dict[str, Any]:
    """Note-level flag rate and flagged sentences per note, for clean notes."""
    n = fail = 0
    flags: list[int] = []
    for r in rows:
        if keep is not None and r.get("note_id") not in keep:
            continue
        er = r.get("eval_result")
        if not er:
            continue
        n += 1
        fail += bool(er.get("is_failure"))
        flags.append(
            sum(
                1
                for i in (er.get("instances") or [])
                if str(i.get("classification") or "").lower() == flagged_value
            )
        )
    return {
        "n": n,
        "fail": fail,
        "rate": fail / n if n else None,
        "mean_flags": statistics.fmean(flags) if flags else 0.0,
    }


def bucket(sentence: str, rationale: str) -> str:
    t = f"{rationale} {sentence}".lower()
    if re.search(r"review of systems|\bros\b|pertinent negative", t):
        return RUBRIC_ACCEPTABLE[0]
    if re.search(r"past surgical|past medical|family history|social history|dedicated .*history", t):
        return RUBRIC_ACCEPTABLE[1]
    if re.search(r"chronic", t):
        return RUBRIC_ACCEPTABLE[2]
    if re.search(r"assessment|\bplan\b|impression", t):
        return "Assessment/plan placement (in objectified scope)"
    if re.search(r"physical exam|examination|\bexam\b", t):
        return "Exam content placement"
    return "other"


def characterise_gold_flags(rows: list[dict]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    notes: dict[str, set[str]] = {}
    all_acceptable_notes = flagged_notes = 0
    for r in rows:
        er = r.get("eval_result")
        if not er:
            continue
        viols = [
            i
            for i in (er.get("instances") or [])
            if str(i.get("classification") or "").lower() == "violation"
        ]
        if viols:
            flagged_notes += 1
            if all(
                bucket(i["note_sentence"], i["rationale"]) in RUBRIC_ACCEPTABLE for i in viols
            ):
                all_acceptable_notes += 1
        for i in viols:
            b = bucket(i["note_sentence"], i["rationale"])
            counts[b] += 1
            notes.setdefault(b, set()).add(r["note_id"])
    total = sum(counts.values())
    acceptable = sum(n for b, n in counts.items() if b in RUBRIC_ACCEPTABLE)
    return {
        "total_flagged_sentences": total,
        "by_bucket": {
            b: {"n": n, "share": round(n / total, 3), "n_notes": len(notes[b])}
            for b, n in counts.most_common()
        },
        "rubric_acceptable_flags": acceptable,
        "rubric_acceptable_share": round(acceptable / total, 3) if total else None,
        "flagged_notes": flagged_notes,
        "notes_all_flags_rubric_acceptable": all_acceptable_notes,
    }
