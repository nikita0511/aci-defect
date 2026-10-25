#!/usr/bin/env python3
"""Score robustness-battery dumps with the shared planted-span hit rule.

Writes aggregate JSON and a markdown table. Does not print note text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aci_defect.ablation_report import note_flag_rate, planted_spans, score_arm

HERE = Path(__file__).resolve().parent
PLANTS = HERE.parents[1] / "data" / "aci_defect_plants"
ROOT = HERE.parents[1] / "results" / "robustness"
OUT = HERE.parents[1] / "reproduced_output" / "robustness"
PLANT_FILE = {
    "P1": "P1_ap_into_hpi",
    "P2": "P2_hpi_into_ap",
    "P3": "P3_symptom_into_pe",
    "P4": "P4_imaging_into_pe",
    "P5": "P5_fabricated_exam",
    "P7": "P7_history_of_active",
}


def _rows(blob: dict) -> list[dict]:
    return blob.get("results") or []


def _ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("eval_result")]


def recall_for(rows: list[dict], plant_id: str, criterion_id: str | None = None) -> dict[str, Any]:
    spans = planted_spans(PLANTS, PLANT_FILE[plant_id])
    use = [
        r
        for r in rows
        if r.get("split") == "planted"
        and r.get("plant_id") == plant_id
        and (criterion_id is None or r.get("criterion_id") == criterion_id)
    ]
    scored = score_arm(spans, use, "note_sentence", ("violation",))
    scored["criterion_id"] = criterion_id
    return scored


def fpr_for(rows: list[dict], criterion_id: str, plant_id: str | None = None) -> dict[str, Any]:
    use = [
        r
        for r in rows
        if r.get("split") == "gold" and r.get("criterion_id") == criterion_id
        and (plant_id is None or r.get("plant_id") == plant_id)
    ]
    return note_flag_rate(use, "violation")


def youden(recall: float | None, fpr: float | None) -> float | None:
    if recall is None or fpr is None:
        return None
    return recall - fpr


def union_fpr(groups: list[list[dict]]) -> dict[str, Any]:
    """A clean note fails if any arm flags it. Fair baseline for a bundled call."""
    by_note: dict[str, list[bool]] = {}
    for rows in groups:
        for r in rows:
            if r.get("split") != "gold" or not r.get("eval_result"):
                continue
            by_note.setdefault(r["note_id"], []).append(bool(r["eval_result"].get("is_failure")))
    # only notes present in every arm
    n_arms = len(groups)
    flagged = 0
    n = 0
    for flags in by_note.values():
        if len(flags) < n_arms:
            continue
        n += 1
        flagged += any(flags)
    return {"n": n, "fail": flagged, "rate": flagged / n if n else None}


def score_pair(
    rows: list[dict],
    plants: list[str],
    obj_cid,
    cmp_cid,
) -> dict[str, Any]:
    per = {}
    obj_fprs = []
    cmp_fprs = []
    obj_groups = []
    for pid in plants:
        oc = obj_cid(pid) if callable(obj_cid) else obj_cid
        cc = cmp_cid(pid) if callable(cmp_cid) else cmp_cid
        orec = recall_for(rows, pid, oc)
        crec = recall_for(rows, pid, cc)
        ofpr = fpr_for(rows, oc, pid if _gold_is_per_plant(rows, oc) else None)
        cfpr = fpr_for(rows, cc, pid if _gold_is_per_plant(rows, cc) else None)
        obj_fprs.append(ofpr["rate"])
        cmp_fprs.append(cfpr["rate"])
        obj_groups.append([r for r in rows if r.get("criterion_id") == oc and r.get("split") == "gold"])
        per[pid] = {
            "objectified_recall": orec["recall"],
            "objectified_n": orec["n"],
            "objectified_hits": orec["hits"],
            "objectified_fpr": ofpr["rate"],
            "objectified_fpr_n": ofpr["n"],
            "composite_recall": crec["recall"],
            "composite_n": crec["n"],
            "composite_hits": crec["hits"],
            "composite_fpr": cfpr["rate"],
            "objectified_J": youden(orec["recall"], ofpr["rate"]),
            "composite_J": youden(crec["recall"], cfpr["rate"]),
            "parse_fail_objectified": orec["parse_fail"],
            "parse_fail_composite": crec["parse_fail"],
        }
    def _mean(xs: list[float | None]) -> float | None:
        vals = [x for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    def _pool(key_hits: str, key_n: str) -> float | None:
        h = sum(per[p][key_hits] for p in plants)
        n = sum(per[p][key_n] for p in plants)
        return h / n if n else None

    o_rec, c_rec = _pool("objectified_hits", "objectified_n"), _pool("composite_hits", "composite_n")
    o_fpr, c_fpr = _mean(obj_fprs), _mean(cmp_fprs)
    u = union_fpr(obj_groups)
    return {
        "per_plant": per,
        "pooled_objectified_recall": o_rec,
        "pooled_composite_recall": c_rec,
        "mean_objectified_fpr": o_fpr,
        "mean_composite_fpr": c_fpr,
        "union_objectified_fpr": u["rate"],
        "union_objectified_fpr_n": u["n"],
        "objectified_J_mean_fpr": youden(o_rec, o_fpr),
        "composite_J_mean_fpr": youden(c_rec, c_fpr),
        "objectified_J_union_fpr": youden(o_rec, u["rate"]),
        "composite_J_vs_union": youden(c_rec, c_fpr),
    }


def _gold_is_per_plant(rows: list[dict], criterion_id: str) -> bool:
    pids = {r.get("plant_id") for r in rows if r.get("criterion_id") == criterion_id and r.get("split") == "gold"}
    return bool(pids - {"GOLD", criterion_id})


def score_model(model_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"model": model_dir.name, "experiments": {}}
    have = {p.stem: json.loads(p.read_text()) for p in sorted(model_dir.glob("*.json"))}

    if "scaffold_split" in have and "scaffold_bundled" in have:
        rows = _rows(have["scaffold_split"]) + _rows(have["scaffold_bundled"])
        out["experiments"]["matched_effort"] = score_pair(
            rows,
            ["P2", "P3", "P4"],
            lambda pid: f"scaffold_split_{pid}",
            "scaffold_bundled",
        )
    if "bundled_rules" in have:
        out["experiments"]["bundled_rules"] = {
            pid: {
                **recall_for(_rows(have["bundled_rules"]), pid, "bundled_rules"),
                "clean": fpr_for(_rows(have["bundled_rules"]), "bundled_rules"),
            }
            for pid in ("P2", "P3", "P4")
        }
        br = out["experiments"]["bundled_rules"]
        hits = sum(br[p]["hits"] for p in br)
        n = sum(br[p]["n"] for p in br)
        fpr = br["P2"]["clean"]["rate"]
        rec = hits / n if n else None
        out["experiments"]["bundled_rules_pooled"] = {
            "recall": rec,
            "n": n,
            "hits": hits,
            "fpr": fpr,
            "fpr_n": br["P2"]["clean"]["n"],
            "J": youden(rec, fpr),
        }
    if "family_p5" in have:
        out["experiments"]["family_p5"] = score_pair(
            _rows(have["family_p5"]), ["P5"], "p5_objectified", "p5_composite"
        )
    if "family_p7" in have:
        out["experiments"]["family_p7"] = score_pair(
            _rows(have["family_p7"]), ["P7"], "p7_objectified", "p7_composite"
        )
    if "thin_misplacement" in have:
        rows = _rows(have["thin_misplacement"])
        out["experiments"]["thin_misplacement"] = score_pair(
            rows,
            ["P2", "P3", "P4"],
            lambda pid: f"thin_{pid}",
            "thin_complaint",
        )
        p1 = recall_for(rows, "P1", "thin_P1")
        out["experiments"]["thin_misplacement"]["P1_objectified_recall"] = p1["recall"]
        out["experiments"]["thin_misplacement"]["P1_n"] = p1["n"]
        p1c = recall_for(rows, "P1", "thin_complaint")
        out["experiments"]["thin_misplacement"]["P1_composite_recall"] = p1c["recall"]
    if "alt_decomp" in have:
        rows = _rows(have["alt_decomp"])
        per = {}
        for pid, cid in (("P1", "alt_plan"), ("P2", "alt_narr"), ("P3", "alt_narr"), ("P4", "alt_result")):
            rec = recall_for(rows, pid, cid)
            fpr = fpr_for(rows, cid)
            per[pid] = {
                "criterion_id": cid,
                "recall": rec["recall"],
                "n": rec["n"],
                "hits": rec["hits"],
                "parse_fail": rec["parse_fail"],
                "fpr": fpr["rate"],
                "fpr_n": fpr["n"],
                "J": youden(rec["recall"], fpr["rate"]),
            }
        hits = sum(per[p]["hits"] for p in ("P2", "P3", "P4"))
        n = sum(per[p]["n"] for p in ("P2", "P3", "P4"))
        fprs = [per[p]["fpr"] for p in ("P2", "P3", "P4")]
        rec = hits / n if n else None
        mean_fpr = sum(x for x in fprs if x is not None) / len(fprs)
        per["pooled_P2P3P4"] = {"recall": rec, "fpr_mean": mean_fpr, "J": youden(rec, mean_fpr), "n": n, "hits": hits}
        out["experiments"]["alt_decomp"] = per
    return out


def _fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def to_markdown(scores: dict[str, Any]) -> str:
    lines = [
        "# Robustness battery",
        "",
        "Planted-span recall uses the same overlap rule as the main ablation.",
        "Youden J = recall − clean-note flag rate. Mean FPR averages the per-plant",
        "clean rates, matching the main ablation table. Union FPR flags a clean note if",
        "any split judge fires, which is the fair comparison to a single bundled call.",
        "",
    ]
    for model, blob in scores.items():
        lines.append(f"## {model}")
        lines.append("")
        exp = blob.get("experiments") or {}
        me = exp.get("matched_effort")
        if me:
            lines.append("### Matched effort (scaffold on both arms; only question scope differs)")
            lines.append("")
            lines.append("| Plant | Obj recall | Obj FPR | Cmp recall | Cmp FPR | Obj J | Cmp J |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for pid, row in me["per_plant"].items():
                lines.append(
                    f"| {pid} | {_fmt(row['objectified_recall'])} | {_fmt(row['objectified_fpr'])} "
                    f"| {_fmt(row['composite_recall'])} | {_fmt(row['composite_fpr'])} "
                    f"| {_fmt(row['objectified_J'])} | {_fmt(row['composite_J'])} |"
                )
            lines.append("")
            lines.append(
                f"Pooled recall obj/cmp {_fmt(me['pooled_objectified_recall'])} / "
                f"{_fmt(me['pooled_composite_recall'])}. "
                f"Mean FPR obj/cmp {_fmt(me['mean_objectified_fpr'])} / {_fmt(me['mean_composite_fpr'])}. "
                f"J (mean FPR) obj/cmp {_fmt(me['objectified_J_mean_fpr'])} / {_fmt(me['composite_J_mean_fpr'])}. "
                f"Union obj FPR {_fmt(me['union_objectified_fpr'])} "
                f"(J {_fmt(me['objectified_J_union_fpr'])})."
            )
            lines.append("")
        br = exp.get("bundled_rules_pooled")
        if br:
            lines.append("### Information-equal bundled rules (one call, no added examples)")
            lines.append("")
            lines.append(
                f"Recall {_fmt(br['recall'])} on n={br['n']}, clean FPR {_fmt(br['fpr'])} "
                f"(n={br['fpr_n']}), J {_fmt(br['J'])}."
            )
            lines.append("")
        for fam in ("family_p5", "family_p7"):
            block = exp.get(fam)
            if not block:
                continue
            row = next(iter(block["per_plant"].values()))
            lines.append(f"### {fam}")
            lines.append("")
            lines.append(
                f"Objectified recall {_fmt(row['objectified_recall'])} FPR {_fmt(row['objectified_fpr'])} "
                f"J {_fmt(row['objectified_J'])}. "
                f"Composite recall {_fmt(row['composite_recall'])} FPR {_fmt(row['composite_fpr'])} "
                f"J {_fmt(row['composite_J'])}."
            )
            lines.append("")
        thin = exp.get("thin_misplacement")
        if thin:
            lines.append("### Open-weight thin protocol (same wording as the main ablation)")
            lines.append("")
            lines.append(
                f"P2–P4 pooled recall obj/cmp {_fmt(thin['pooled_objectified_recall'])} / "
                f"{_fmt(thin['pooled_composite_recall'])}. "
                f"Mean FPR obj/cmp {_fmt(thin['mean_objectified_fpr'])} / {_fmt(thin['mean_composite_fpr'])}. "
                f"J obj/cmp {_fmt(thin['objectified_J_mean_fpr'])} / {_fmt(thin['composite_J_mean_fpr'])}. "
                f"P1 objectified recall {_fmt(thin.get('P1_objectified_recall'))}."
            )
            lines.append("")
        alt = exp.get("alt_decomp")
        if alt:
            lines.append("### Alternative content-class decomposition")
            lines.append("")
            lines.append("| Plant | Criterion | Recall | FPR | J |")
            lines.append("|---|---|---:|---:|---:|")
            for pid in ("P1", "P2", "P3", "P4"):
                row = alt[pid]
                lines.append(
                    f"| {pid} | {row['criterion_id']} | {_fmt(row['recall'])} | {_fmt(row['fpr'])} | {_fmt(row['J'])} |"
                )
            pool = alt["pooled_P2P3P4"]
            lines.append("")
            lines.append(
                f"P2–P4 pooled recall {_fmt(pool['recall'])}, mean FPR {_fmt(pool['fpr_mean'])}, "
                f"J {_fmt(pool['J'])}."
            )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    scores = {}
    if not ROOT.exists():
        raise SystemExit(f"no results at {ROOT}")
    for model_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and any(p.glob("*.json"))):
        scores[model_dir.name] = score_model(model_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "robustness_scores.json").write_text(json.dumps(scores, indent=2))
    text = to_markdown(scores)
    (OUT / "ROBUSTNESS_RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
