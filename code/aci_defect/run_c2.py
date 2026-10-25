"""ACI-Defect C2 runner.

Builds one-defect-at-a-time plants on ACI-bench test1, scores each perturbed
note against the unperturbed gold with ROUGE / MedCon-proxy / char-3gram
cosine, and measures span-localized objectified detection plus evaluator
false-positive rate on unperturbed test1.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from aci_defect.detect import detect_plant
from aci_defect.metrics import score_pair
from aci_defect.plants import (
    PLANT_SPECS,
    SkipPlant,
    plant_control_delete_15pct,
)
from aci_defect.sections import parse_note

ACI_DIR = HERE.parent
# Judge runs on unperturbed gold are not part of the release, so the
# gold-comparison block below is skipped.
RESULTS_DIR = HERE.parents[1] / "results" / "gold_judge_runs"
DEFAULT_INPUT = HERE.parents[1] / "data" / "aci_bench_test1_source.json"
DEFAULT_OUT = HERE / "output"


def load_test1(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    encs = data["encounters"] if isinstance(data, dict) else data
    return [e for e in encs if (e.get("meta_data") or {}).get("split") == "test1"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _note_ids_from_eval(blob: Any) -> dict[str, dict[str, Any]]:
    """Index evaluator outputs by note_id. Handles several streamlit dump shapes."""
    rows: list[Any]
    if isinstance(blob, dict):
        for key in ("results", "encounters", "items", "evaluations", "data"):
            if isinstance(blob.get(key), list):
                rows = blob[key]
                break
        else:
            # maybe {note_id: result}
            if blob and all(isinstance(v, dict) for v in blob.values()):
                return {str(k): v for k, v in blob.items()}
            rows = [blob]
    elif isinstance(blob, list):
        rows = blob
    else:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        nid = (
            row.get("note_id")
            or row.get("id")
            or (row.get("encounter") or {}).get("note_id")
            or (row.get("input") or {}).get("note_id")
            or (row.get("meta_data") or {}).get("note_id")
        )
        result = (
            row.get("eval_result")
            or row.get("result")
            or row.get("output")
            or row.get("eval")
            or row
        )
        if nid:
            out[str(nid)] = result if isinstance(result, dict) else {"raw": result}
    return out


def _is_failure(result: dict[str, Any]) -> bool:
    if "is_failure" in result:
        return bool(result["is_failure"])
    nested = result.get("evaluation") or result.get("verdict") or {}
    if isinstance(nested, dict) and "is_failure" in nested:
        return bool(nested["is_failure"])
    return False


def orig_fp_rates(test1_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Note-level fail rate of each production evaluator on unperturbed test1."""
    mapping = {
        "P2": ("HPI→A/P leakage", "hpi_leakage_into_ap", "HPI_in_AP"),
        "P3": ("subjective→PE leakage", "subjective_leakage_pe", "PE Leakage"),
        "P5": ("hallucination", "hallucination", "Hallucination"),
        "P6": ("omission", "omission", "Omission"),
        "P7": ("history-of misuse", "history_of_misuse", "'History of'"),
        "P9": ("diagnosis structure", "diagnosis_structure", "Diagnosis Structure"),
        "P8": ("deterministic style", None, None),
        "P1": ("A&P→HPI leakage", None, None),
        "P4": ("results→PE leakage", None, None),
        "P10": ("CC conciseness", None, None),
    }
    files = list(RESULTS_DIR.glob("streamlit_*.json")) if RESULTS_DIR.exists() else []
    out: dict[str, dict[str, Any]] = {}
    for pid, (label, needle, title_bit) in mapping.items():
        if needle is None:
            out[pid] = {
                "evaluator": label,
                "n_test1": None,
                "n_fail": None,
                "fp_rate": None,
                "source": "not_run_on_unperturbed_gold",
            }
            continue
        match = None
        for f in files:
            name = f.name
            if needle in name or (title_bit and title_bit in name):
                match = f
                break
        if not match:
            out[pid] = {
                "evaluator": label,
                "n_test1": None,
                "n_fail": None,
                "fp_rate": None,
                "source": "file_missing",
            }
            continue
        idx = _note_ids_from_eval(_load_json(match))
        ids = [i for i in test1_ids if i in idx]
        n_fail = sum(1 for i in ids if _is_failure(idx[i]))
        out[pid] = {
            "evaluator": label,
            "n_test1": len(ids),
            "n_fail": n_fail,
            "fp_rate": round(n_fail / len(ids), 4) if ids else None,
            "source": match.name,
        }
    return out


def apply_plant(spec: dict[str, Any], enc: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    if spec["id"] == "CTRL":
        return plant_control_delete_15pct(enc, rng)
    return spec["fn"](enc)


def mean(xs: list[float]) -> float:
    return round(statistics.fmean(xs), 6) if xs else float("nan")


def run(input_path: Path, out_dir: Path, seed: int = 20260821) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plants_dir = out_dir / "plants"
    plants_dir.mkdir(exist_ok=True)

    encs = load_test1(input_path)
    test1_ids = {e["note_id"] for e in encs}
    rng = random.Random(seed)
    fp = orig_fp_rates(test1_ids)

    # Self-similarity ceiling
    self_scores = [score_pair(e["note"], e["note"]) for e in encs]
    ceiling = {k: mean([s[k] for s in self_scores]) for k in self_scores[0]}

    records: list[dict[str, Any]] = []
    skip_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_plant_notes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for spec in PLANT_SPECS:
        for enc in encs:
            try:
                planted = apply_plant(spec, enc, rng)
            except SkipPlant as e:
                skip_counts[spec["id"]][e.reason] += 1
                continue
            metrics = score_pair(planted["note"], enc["note"])
            det = detect_plant(spec["id"], planted["note"], planted, enc)
            rec = {
                "plant_id": spec["id"],
                "plant_name": spec["name"],
                "note_id": enc["note_id"],
                "target_evaluator": spec["target_evaluator"],
                "operation": planted.get("operation"),
                "planted_span": planted.get("planted_span"),
                "detected": det["detected"],
                "detect_rationale": det["rationale"],
                "localized_to_plant": det.get("localized_to_plant"),
                **{f"m_{k}": v for k, v in metrics.items()},
                "rougeL_drop": round(1.0 - metrics["rougeL"], 6),
                "rouge1_drop": round(1.0 - metrics["rouge1"], 6),
                "medcon_drop": round(1.0 - metrics["medcon_proxy"], 6),
            }
            records.append(rec)
            by_plant_notes[spec["id"]].append(
                {
                    "note_id": enc["note_id"],
                    "original_note": enc["note"],
                    "perturbed_note": planted["note"],
                    "transcript": enc.get("transcript"),
                    "case_facts": enc.get("case_facts") or [],
                    "plant": {k: v for k, v in planted.items() if k != "note"},
                    "metrics": metrics,
                    "detection": det,
                    "meta_data": enc.get("meta_data"),
                }
            )

    # Per-plant summaries
    summaries = []
    for spec in PLANT_SPECS:
        pid = spec["id"]
        rows = [r for r in records if r["plant_id"] == pid]
        n_skip = sum(skip_counts[pid].values())
        n_ok = len(rows)
        summaries.append(
            {
                "plant_id": pid,
                "name": spec["name"],
                "title": spec["title"],
                "target_evaluator": spec["target_evaluator"],
                "why_reference_blind": spec["why_reference_blind"],
                "n_applied": n_ok,
                "n_skipped": n_skip,
                "coverage": round(n_ok / max(1, n_ok + n_skip), 4),
                "skip_reasons": dict(skip_counts[pid]),
                "objectified_recall": round(
                    sum(1 for r in rows if r["detected"]) / n_ok, 4
                )
                if n_ok
                else None,
                "mean_rouge1": mean([r["m_rouge1"] for r in rows]),
                "mean_rouge2": mean([r["m_rouge2"] for r in rows]),
                "mean_rougeL": mean([r["m_rougeL"] for r in rows]),
                "mean_rouge1_drop": mean([r["rouge1_drop"] for r in rows]),
                "mean_rougeL_drop": mean([r["rougeL_drop"] for r in rows]),
                "mean_medcon_proxy": mean([r["m_medcon_proxy"] for r in rows]),
                "mean_medcon_drop": mean([r["medcon_drop"] for r in rows]),
                "mean_tfidf_cosine": mean([r["m_tfidf_cosine"] for r in rows]),
                "mean_delta_words": mean([r["m_delta_words"] for r in rows]),
                "orig_fp": fp.get(pid),
            }
        )
        payload = {
            "plant_id": pid,
            "spec": {k: v for k, v in spec.items() if k != "fn"},
            "n": n_ok,
            "instances": by_plant_notes[pid],
        }
        (plants_dir / f"{pid}_{spec['name']}.json").write_text(
            json.dumps(payload, indent=2)
        )

    run_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "n_test1": len(encs),
        "seed": seed,
        "citation": "Yim et al., Scientific Data 2023, doi:10.1038/s41597-023-02487-3",
        "license": "CC BY 4.0 overlay on ACI-bench",
        "self_similarity_ceiling": ceiling,
        "metric_notes": {
            "rouge": "ROUGE-1/2/L F1, locally implemented (whitespace/alnum tokens)",
            "medcon_proxy": (
                "UMLS-free concept F1: gazetteer + dose/lab patterns. Not QuickUMLS. "
                "Use as a blindness diagnostic, not as a published MedCon number until UMLS is wired."
            ),
            "tfidf_cosine": "character 3-gram cosine; BERTScore stand-in when transformers unavailable",
            "detection": (
                "Span-localized objectified rules encoding each evaluator's determinate "
                "decision. These integrity checks are distinct from the evaluated "
                "clinical checks reported in the paper."
            ),
        },
    }

    (out_dir / "aci_defect_records.json").write_text(
        json.dumps({"meta": run_meta, "records": records, "summaries": summaries}, indent=2)
    )

    # CSV
    if records:
        keys = list(records[0].keys())
        with (out_dir / "aci_defect_records.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)

    report = render_report(run_meta, summaries, records, encs)
    (out_dir / "EXPERIMENT_REPORT.md").write_text(report)

    summary_md = render_paper_table(summaries, ceiling)
    (out_dir / "PAPER_TABLE.md").write_text(summary_md)
    return {"n_records": len(records), "out_dir": str(out_dir), "summaries": summaries}


def render_paper_table(summaries: list[dict[str, Any]], ceiling: dict[str, float]) -> str:
    lines = [
        "# ACI-Defect paper table (test1, one plant per note)",
        "",
        "Self-similarity ceiling (unperturbed note vs itself): "
        f"ROUGE-L={ceiling.get('rougeL'):.3f}, MedCon-proxy={ceiling.get('medcon_proxy'):.3f}.",
        "",
        "| ID | Plant | n | Objectified recall | ROUGE-1 | ROUGE-1 drop | ROUGE-L drop | MedCon-proxy drop | Orig FP (test1) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        fp = s.get("orig_fp") or {}
        fp_s = "—"
        if fp.get("fp_rate") is not None:
            fp_s = f"{fp['fp_rate']:.1%} ({fp['n_fail']}/{fp['n_test1']})"
        elif fp.get("source"):
            fp_s = fp["source"]
        rec = s.get("objectified_recall")
        rec_s = "—" if rec is None else f"{rec:.1%}"
        lines.append(
            f"| {s['plant_id']} | {s['title']} | {s['n_applied']} | {rec_s} | "
            f"{s['mean_rouge1']:.3f} | {s['mean_rouge1_drop']:.3f} | "
            f"{s['mean_rougeL_drop']:.3f} | {s['mean_medcon_drop']:.3f} | {fp_s} |"
        )
    lines.append("")
    lines.append(
        "CTRL is a generic 15% token-deletion baseline: reference metrics **should** "
        "drop. P1–P10 should not."
    )
    lines.append("")
    return "\n".join(lines)


def render_report(
    meta: dict[str, Any],
    summaries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    encs: list[dict[str, Any]],
) -> str:
    n = meta["n_test1"]
    ceiling = meta["self_similarity_ceiling"]
    lines = [
        "# ACI-Defect (C2) experiment report",
        "",
        f"_Generated {meta['created_utc']}_",
        "",
        "## 1. What was tested",
        "",
        "ACI-Defect is a **defect-targeted overlay** on ACI-bench (Yim et al., 2023; "
        "CC BY 4.0). We take the **test1 split (n=40)** and plant **one clinically "
        "meaningful defect at a time**, leaving remaining content intact. The claim:",
        "",
        "1. The objectified evaluator for that defect **localizes the planted span**.",
        "2. ROUGE-1/2/L and a MedCon-style concept F1 of the perturbed note vs the "
        "**original gold note** barely move.",
        "3. The same metrics of the unperturbed note vs itself are a ceiling (~1.0).",
        "4. False-positive rate of that evaluator on **unperturbed** test1 is reported "
        "from the existing auto-eval dumps (not re-run here).",
        "",
        "This is **not** a generator leaderboard. We do not score models on ACI-bench "
        "ROUGE. We show that those metrics are the wrong instrument for the defects "
        "clinicians complain about.",
        "",
        f"**Corpus:** `{meta['input']}`  ",
        f"**N test1:** {n}  ",
        f"**Seed:** {meta['seed']}  ",
        f"**Citation:** {meta['citation']}  ",
        f"**License of overlay:** {meta['license']}",
        "",
        "## 2. Metrics (and what they are not)",
        "",
        f"- **ROUGE-1/2/L F1** implemented locally. Self-similarity ceiling ROUGE-L = "
        f"{ceiling['rougeL']:.4f}.",
        f"- **MedCon-proxy:** gazetteer + dose/lab pattern concept F1. **Not** UMLS "
        f"QuickUMLS/MetaMap. Ceiling = {ceiling['medcon_proxy']:.4f}. Treat as a "
        "blindness diagnostic. A true MedCon column can replace this without "
        "changing the plants.",
        f"- **char-3gram cosine** (labeled `tfidf_cosine` in JSON): BERTScore stand-in. "
        f"Ceiling = {ceiling['tfidf_cosine']:.4f}.",
        "- **Drops** are `1 − score` because the reference is the unperturbed gold, "
        "whose self-score is 1.",
        "",
        "## 3. Detection",
        "",
        "Detection in this deterministic runner is the **objectified integrity "
        "check** for each plant. Evaluated clinical-check recall is reported "
        "separately in the accompanying results table:",
        "",
        "| ID | Determinate check |",
        "|---|-------------------|",
        "| P1 | Plan-language span now in HPI and gone from A/P |",
        "| P2 | HPI-narrative span now in A/P and gone from HPI |",
        "| P3 | Inserted symptom sentence sits in PE |",
        "| P4 | Copied RESULTS sentence sits in PE (and remains in RESULTS) |",
        "| P5 | Fabricated PE finding in PE with low transcript support |",
        "| P6 | Case-fact modifier deleted from the note |",
        "| P7 | `history of` + still-active A/P problem |",
        "| P8 | Brand drug present only in lowercase (production style rule) |",
        "| P9 | Two problem headings merged with `and` |",
        "| P10 | CC > 12 words or narrative (`who presents`) |",
        "| CTRL | No evaluator (generic corruption) |",
        "",
        "Recall is **instance-level** (did we flag the planted span?), not note-level "
        "`is_failure`. By construction a correctly applied plant should yield recall "
        "≈ 1.0; anything lower is a planter or parser bug, not a weak judge.",
        "",
        "## 4. Paper table",
        "",
        render_paper_table(summaries, ceiling).split("\n", 2)[-1],
        "## 5. Per-plant notes",
        "",
    ]
    for s in summaries:
        lines.append(f"### {s['plant_id']}. {s['title']}")
        lines.append("")
        lines.append(f"- **Target evaluator:** {s['target_evaluator']}")
        lines.append(f"- **Why ROUGE/MedCon should be weak:** {s['why_reference_blind']}")
        lines.append(
            f"- **Applied / skipped:** {s['n_applied']} / {s['n_skipped']} "
            f"(coverage {s['coverage']:.0%})"
        )
        if s["skip_reasons"]:
            reasons = ", ".join(f"{k} ({v})" for k, v in s["skip_reasons"].items())
            lines.append(f"- **Skip reasons:** {reasons}")
        rec = s["objectified_recall"]
        lines.append(f"- **Objectified recall:** {'—' if rec is None else f'{rec:.1%}'}")
        lines.append(
            f"- **ROUGE-1 / 2 / L:** {s['mean_rouge1']:.3f} / {s['mean_rouge2']:.3f} / "
            f"{s['mean_rougeL']:.3f}  (drops L {s['mean_rougeL_drop']:.3f})"
        )
        lines.append(
            f"- **MedCon-proxy:** {s['mean_medcon_proxy']:.3f} "
            f"(drop {s['mean_medcon_drop']:.3f})"
        )
        lines.append(f"- **char-3gram cosine:** {s['mean_tfidf_cosine']:.3f}")
        lines.append(f"- **Mean Δ words:** {s['mean_delta_words']:+.1f}")
        fp = s.get("orig_fp") or {}
        if fp.get("fp_rate") is not None:
            lines.append(
                f"- **Unperturbed test1 note-fail rate (production judge):** "
                f"{fp['fp_rate']:.1%} ({fp['n_fail']}/{fp['n_test1']}) "
                f"from `{fp['source']}`."
            )
        else:
            lines.append(
                f"- **Unperturbed test1 FP:** {fp.get('source') or 'n/a'} "
                f"({fp.get('evaluator')})"
            )
        # example span
        ex = next((r for r in records if r["plant_id"] == s["plant_id"]), None)
        if ex:
            span = (ex.get("planted_span") or "")[:240]
            lines.append(f"- **Example plant** (`{ex['note_id']}`): `{span}`")
        lines.append("")

    lines += [
        "## 6. How to read the result",
        "",
        "P1–P2 **move** tokens: ROUGE-1 stays at 1.0 (bag of words unchanged). ROUGE-L "
        "drops a few points because LCS is order-sensitive — still far smaller than "
        "CTRL. That is the point: n-gram overlap cannot see section membership.",
        "",
        "P3–P5 insert one sentence (~15–40 tokens) into notes of 300–700 words, so "
        "ROUGE-L drop is a few points at most. P5 can **raise** MedCon-proxy by "
        "adding plausible findings.",
        "",
        "P6 deletes a short modifier; remaining overlap is high — this is the "
        "omission that kills clinical meaning (laterality, dose) without moving "
        "ROUGE.",
        "",
        "P7–P9 are near-paraphrases / structure-only. P8 is case-only (MedCon-blind "
        "by design). P10 lengthens CC; whole-note ROUGE barely notices.",
        "",
        "CTRL (15% random token drop) is the sanity check: if ROUGE-L drop here is "
        "**not** an order of magnitude larger than P1–P10, the metric implementation "
        "is wrong.",
        "",
        "Unperturbed gold FP rates are **not** plant false positives. ACI-bench gold "
        "notes already contain imaginary EHR injections (last names, templated ROS, "
        "vaccine wording). High hallucination/omission fail rates on gold are a "
        "property of the corpus + rubric, reported honestly, and are why C3 "
        "adjudication exists. They are not evidence that plants are dirty.",
        "",
        "## 7. Reproduce",
        "",
        "```bash",
        "python -m aci_defect.run_c2 \\",
        "  --input data/aci_bench_test1_source.json \\",
        "  --out   reproduced_output",
        "```",
        "",
        "Run from the archive root with `code` on `PYTHONPATH`.",
        "",
        "Outputs:",
        "",
        "- `output/plants/P{1-10}_*.json` — perturbed notes + span metadata (the release)",
        "- `output/aci_defect_records.csv` — one row per (plant, note)",
        "- `output/PAPER_TABLE.md` — copy into the paper",
        "- `output/EXPERIMENT_REPORT.md` — this file",
        "",
        "## 8. What this pass does not claim",
        "",
        "- Evaluated clinical-check recall on planted notes.",
        "- Official ACI-bench MedCon (needs UMLS). Replace `medcon_proxy` in "
        "`metrics.py` without touching plants.",
        "- BERTScore-RoBERTa. char-3gram cosine is the stand-in.",
        "- That unperturbed gold is clean. It is not; see C3.",
        "",
        "## 9. Release constraints",
        "",
        "Release perturbation scripts + planted notes under CC BY 4.0, citing Yim et "
        "al. 2023. Do **not** release production notes, real complaints, or judge "
        "prompts.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Run ACI-Defect C2 experiments")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=20260821)
    args = p.parse_args()
    result = run(args.input, args.out, args.seed)
    print(json.dumps({"n_records": result["n_records"], "out_dir": result["out_dir"]}, indent=2))
    for s in result["summaries"]:
        rec = s["objectified_recall"]
        rec_s = "—" if rec is None else f"{rec:.0%}"
        print(
            f"{s['plant_id']:4} n={s['n_applied']:2} rec={rec_s:>4}  "
            f"R-1={s['mean_rouge1']:.3f} drop={s['mean_rouge1_drop']:.3f}  "
            f"R-L drop={s['mean_rougeL_drop']:.3f}  "
            f"Med={s['mean_medcon_proxy']:.3f} drop={s['mean_medcon_drop']:.3f}"
        )


if __name__ == "__main__":
    main()
