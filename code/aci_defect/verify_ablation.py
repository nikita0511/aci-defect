#!/usr/bin/env python3
"""Recompute the composite arms of the ablation from the released archive.

Reads the raw composite judge output in `results/composite_ablation/`, applies
the shared planted-span hit rule, and checks the result against the shipped
`results/ablation_summary.csv`. The objectified arm is reported from that file
but cannot be recomputed here: its judge prompts are not part of the release.

Usage:
  python -m aci_defect.verify_ablation            # run from the archive root
  python -m aci_defect.verify_ablation --root /path/to/archive
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from aci_defect.ablation_report import (
    PLANT_FILES,
    PLANT_TITLE,
    characterise_gold_flags,
    note_flag_rate,
    planted_spans,
    rows_of,
    score_arm,
)

ARM_FILES = {
    "complaint": "complaint_P1P2P3P4.json",
    "zhou": "zhou_P1P2.json",
}
ZHOU_PLANTS = ("P1", "P2")
TOL = 5e-3


def _load(path: Path) -> list[dict]:
    return rows_of(json.loads(path.read_text()))


def _expected(results_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    with (results_dir / "ablation_summary.csv").open(encoding="utf-8") as fh:
        return {(r["plant_id"], r["arm"]): r for r in csv.DictReader(fh)}


def _check(label: str, got: float | int, want: str, failures: list[str]) -> str:
    if want in ("", None):
        return "n/a"
    ok = abs(float(got) - float(want)) <= TOL
    if not ok:
        failures.append(f"{label}: recomputed {got} but archive says {want}")
    return "ok" if ok else "MISMATCH"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()

    plants_dir = args.root / "data" / "aci_defect_plants"
    results_dir = args.root / "results"
    raw_dir = results_dir / "composite_ablation"
    for path in (plants_dir, raw_dir, results_dir / "ablation_summary.csv"):
        if not path.exists():
            raise SystemExit(f"not found: {path} (pass --root <archive root>)")

    expected = _expected(results_dir)
    arms = {arm: _load(raw_dir / name) for arm, name in ARM_FILES.items()}
    failures: list[str] = []

    print(f"{'plant':6} {'arm':10} {'recomputed':>12} {'recall':>7} {'archive':>8}  check")
    print("-" * 62)
    for pid in PLANT_FILES:
        spans = planted_spans(plants_dir, PLANT_FILES[pid])
        for arm, rows in arms.items():
            if arm == "zhou" and pid not in ZHOU_PLANTS:
                continue
            sel = [r for r in rows if r.get("plant_id") == pid]
            s = score_arm(spans, sel, "note_sentence", ("violation",))
            want = expected.get((pid, arm), {})
            state = _check(
                f"{pid}/{arm} recall", s["recall"], want.get("planted_span_recall", ""), failures
            )
            print(
                f"{pid:6} {arm:10} {s['hits']:>5}/{s['n']:<6} {s['recall']:>7.3f} "
                f"{want.get('planted_span_recall', '—'):>8}  {state}"
            )

    gold_rows = _load(raw_dir / "complaint_GOLD.json")
    g = note_flag_rate(gold_rows, "violation")
    want = expected.get(("GOLD", "complaint"), {})
    state = _check("GOLD/complaint flag rate", g["rate"], want.get("note_flag_rate", ""), failures)
    burden = _check(
        "GOLD/complaint flagged sentences",
        round(g["mean_flags"], 3),
        want.get("mean_flagged_sentences", ""),
        failures,
    )
    print("-" * 62)
    print(
        f"{'GOLD':6} {'complaint':10} {g['fail']:>5}/{g['n']:<6} {g['rate']:>7.3f} "
        f"{want.get('note_flag_rate', '—'):>8}  {state}; burden {burden}"
    )
    print(f"       flagged sentences per clean note: {g['mean_flags']:.2f}")

    ch = characterise_gold_flags(gold_rows)
    print(
        f"\nClean-note flags: {ch['total_flagged_sentences']} sentences over "
        f"{ch['flagged_notes']} notes; {ch['rubric_acceptable_flags']} "
        f"({ch['rubric_acceptable_share']:.0%}) fall in categories the objectified "
        f"rubric rules acceptable in History."
    )
    bucket_csv = results_dir / "ablation_clean_note_flags.csv"
    if bucket_csv.exists():
        with bucket_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                got = ch["by_bucket"].get(row["bucket"], {}).get("n")
                if got is None or int(got) != int(row["flagged_sentences"]):
                    failures.append(
                        f"bucket {row['bucket']}: recomputed {got} but archive says "
                        f"{row['flagged_sentences']}"
                    )
        print(f"Bucket counts checked against {bucket_csv.name}.")

    obj: dict[str, Any] = {
        pid: expected[(pid, "objectified")]
        for pid in PLANT_FILES
        if (pid, "objectified") in expected
    }
    if obj:
        print(
            "\nObjectified arm (from ablation_summary.csv; not recomputable here "
            "because the judge prompts are not released):"
        )
        for pid, row in obj.items():
            print(
                f"  {pid:4} {PLANT_TITLE[pid]:28} "
                f"{row['planted_span_hits']}/{row['n']} "
                f"recall {float(row['planted_span_recall']):.3f}"
            )

    if failures:
        print("\nFAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: composite arms reproduce the released summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
