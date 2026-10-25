#!/usr/bin/env python3
"""Recompute second-rater agreement from the released label file.

Scores the independent second pass against the first pass on the 10 doubly
labeled ACI-bench test1 notes and checks the result against
results/irr_summary.json. Standard library only.

Usage (from the archive root):
  python code/aci_defect/verify_irr.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ROOT / "data" / "labels" / "second_rater_labels.csv"
DEFAULT_SUMMARY = ROOT / "results" / "irr_summary.json"
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_ITERS = 10000


def cohen_kappa(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    marg_a = Counter(a for a, _ in pairs)
    marg_b = Counter(b for _, b in pairs)
    categories = sorted({c for pair in pairs for c in pair})
    expected = sum((marg_a[c] / n) * (marg_b[c] / n) for c in categories)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return {"n": n, "observed_agreement": observed, "kappa": kappa}


def bootstrap_ci(pairs: list[tuple[str, str]]) -> Optional[list[float]]:
    rng = random.Random(BOOTSTRAP_SEED)
    stats: list[float] = []
    for _ in range(BOOTSTRAP_ITERS):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        k = cohen_kappa(sample)["kappa"]
        if not math.isnan(k):
            stats.append(k)
    stats.sort()
    return [stats[int(0.025 * len(stats))], stats[min(int(0.975 * len(stats)), len(stats) - 1)]]


def score(rows: list[dict[str, str]], dimension: str) -> dict[str, Any]:
    subset = [
        r for r in rows
        if r["dimension"] == dimension and r["evaluation_inclusion"] == "included"
    ]
    pairs = [(r["first_pass_binary"], r["second_pass_binary"]) for r in subset]
    out = cohen_kappa(pairs)
    out["kappa_ci95"] = bootstrap_ci(pairs)
    notes = sorted({r["note_id"] for r in subset})
    out["note_level_agree"] = sum(
        any(r["first_pass_binary"] == "positive" for r in subset if r["note_id"] == nid)
        == any(r["second_pass_binary"] == "positive" for r in subset if r["note_id"] == nid)
        for nid in notes
    )
    out["notes"] = len(notes)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = ap.parse_args()

    with args.labels.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(r["first_pass_rater"] == r["second_pass_rater"] for r in rows):
        raise SystemExit("an item was relabeled by its own first-pass rater")
    reported = json.loads(args.summary.read_text(encoding="utf-8"))["results"]

    status = 0
    for dimension in ("hallucination", "omission"):
        got = score(rows, dimension)
        want = reported[dimension]["included_only"]["binary"]
        match = (
            got["n"] == want["n"]
            and abs(got["kappa"] - want["kappa"]) < 1e-9
            and all(abs(a - b) < 1e-9 for a, b in zip(got["kappa_ci95"], want["kappa_ci95"]))
        )
        status |= not match
        lo, hi = got["kappa_ci95"]
        print(
            f"{dimension:14} n={got['n']} agreement={got['observed_agreement']:.3f} "
            f"kappa={got['kappa']:.2f} [{lo:.2f}, {hi:.2f}] "
            f"note-level {got['note_level_agree']}/{got['notes']} "
            f"{'matches' if match else 'DOES NOT MATCH'} irr_summary.json"
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
