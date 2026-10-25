#!/usr/bin/env python3
"""Robustness battery for the objectification ablation.

Same prompt template, temperature, JSON schema, and planted-span hit rule as
run_multivendor_ablation.py. Only the criterion text changes.

Experiments
  bundled_rules      all four directional rules in one call (information-equal
                     to the split objectified arm; no added examples)
  scaffold_bundled   one bundled question, plus anchor-then-classify,
                     allowed/disallowed rules, and calibration examples
  scaffold_split     one directional rule per call, plus the same scaffold
  family_p5          hallucination family: composite vs scoped exam predicate
  family_p7          history-of family: composite vs scoped heading predicate
  alt_decomp         content-class decomposition, not ordered section pairs

Does not call the fine-tuned omission model. Does not print notes.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from aci_defect.run_multivendor_ablation import (
    OBJECTIFIED_CRITERIA,
    CompositeResult,
    _is_retryable,
    make_backend,
    normalize_result,
    parse_response,
    slug_model,
)

_ACI = Path(__file__).resolve().parent
PLANTS_DIR = _ACI.parents[1] / "data" / "aci_defect_plants"
OUT_ROOT = _ACI.parents[1] / "reproduced_output" / "robustness"

PLANT_FILES = {
    "P1": "P1_ap_into_hpi",
    "P2": "P2_hpi_into_ap",
    "P3": "P3_symptom_into_pe",
    "P4": "P4_imaging_into_pe",
    "P5": "P5_fabricated_exam",
    "P7": "P7_history_of_active",
}

# Shared scaffold. Calibration sentences are generic; none is a planted span.
SCAFFOLD = """\
Procedure
Anchor, then classify. For each sentence, first name the section it sits under and the kind of content it contains (history narrative, bedside exam finding, result, assessment, or plan). Then decide.

Allowed and disallowed
- History may contain symptom narrative, patient quotes, review of systems, past/family/social history, and chronic-condition history. History may not contain a clinician impression, a definitive diagnosis, or a management decision.
- Physical Exam may contain bedside objective findings. It may not contain symptom reports or laboratory or imaging results.
- Assessment and Plan may contain impressions, diagnoses, and management. It may not contain symptom-story narrative that belongs in History.
- Results may contain laboratory and imaging statements.

Calibration
- "We will start metformin 500 mg daily." under History: violation. Under Assessment and Plan: acceptable.
- "The patient reports a cough for three days." under History: acceptable. Under Physical Exam: violation.
- "Chest x-ray is unremarkable." under Results: acceptable. Under Physical Exam: violation.
"""

FAMILY = {
    "p5_composite": (
        "The note contains clinical content that is not supported by the transcript."
    ),
    "p5_objectified": (
        "A Physical Exam sentence states an examination finding that the transcript "
        "does not support. Do not flag a finding the transcript describes. Do not "
        "flag History or Plan sentences."
    ),
    "p7_composite": (
        "The note misuses clinical terminology or misstates whether a problem is active."
    ),
    "p7_objectified": (
        "An Assessment and Plan heading labels a problem that is being assessed or "
        "managed at this visit as 'history of' that problem."
    ),
}

ALT = {
    "alt_plan": (
        "A sentence outside Assessment and Plan states a management decision: a "
        "medication change, an order, a referral, or a follow-up instruction. Do not "
        "flag a diagnosis mentioned only as history, and do not flag patient-reported "
        "symptoms."
    ),
    "alt_narr": (
        "A sentence outside the History section is patient-reported symptom narrative, "
        "meaning it states what the patient reports, denies, or endorses, rather than "
        "an assessment or an exam finding."
    ),
    "alt_result": (
        "A sentence outside the Results section states a laboratory value or an "
        "imaging result."
    ),
}

# plant -> alt criterion. P2 and P3 share one predicate on purpose: a different
# decomposition can merge two ordered pairs into one content class.
ALT_FOR_PLANT = {"P1": "alt_plan", "P2": "alt_narr", "P3": "alt_narr", "P4": "alt_result"}

PROMPT_TEMPLATE = """You are evaluating a clinical note against the following criterion.

CRITERION
{criterion}

Decide whether the note violates this criterion.

Output requirements
- Consider the sentences of the FINAL rendered note.
- For each sentence you judge relevant to the criterion, output one instance with:
  - note_sentence: the sentence quoted **exactly** as it appears in the note
  - section: the section heading the sentence currently sits under
  - classification: "violation" if that sentence violates the criterion, otherwise "acceptable"
  - rationale: one sentence
  - confidence: 0.0-1.0
- total_sentences: number of note sentences you considered.
- violation_count: number of instances classified as a violation.
- is_failure: true if violation_count >= 1.
- failure_reason: one-sentence summary, or "No violation detected" if none.

Input
- NOTE (generated clinical note):
<<<NOTE_START>>>
{note}
<<<NOTE_END>>>
- TRANSCRIPT / CONTEXT:
<<<TRANSCRIPT_START>>>
{transcript}
<<<TRANSCRIPT_END>>>

Return valid JSON only (no markdown, no code fence). The JSON must match the response schema exactly.
"""


def bundled_rules_text() -> str:
    lines = [f"- {OBJECTIFIED_CRITERIA[p]}" for p in ("P1", "P2", "P3", "P4")]
    return (
        "The note violates the criterion if ANY of the following is true:\n"
        + "\n".join(lines)
    )


def criteria_catalog() -> dict[str, str]:
    cat = {
        "bundled_rules": bundled_rules_text(),
        "scaffold_bundled": (
            SCAFFOLD
            + "\nCriterion question\nContent is in the wrong section of the note."
        ),
    }
    for pid, text in OBJECTIFIED_CRITERIA.items():
        cat[f"scaffold_split_{pid}"] = SCAFFOLD + "\nCriterion question\n" + text
    cat.update(FAMILY)
    cat.update(ALT)
    cat["thin_complaint"] = "Content is in the wrong section of the note."
    for pid, text in OBJECTIFIED_CRITERIA.items():
        cat[f"thin_{pid}"] = text
    return cat


def _load_plant(pid: str) -> list[dict[str, Any]]:
    path = PLANTS_DIR / f"{PLANT_FILES[pid]}.json"
    return json.loads(path.read_text())["instances"]


def _gold_notes() -> list[dict[str, Any]]:
    """The 40 unperturbed test1 notes, taken from the P3 file (full coverage)."""
    return _load_plant("P3")


def build_tasks(experiment: str) -> list[dict[str, Any]]:
    catalog = criteria_catalog()
    tasks: list[dict[str, Any]] = []

    def add(split: str, pid: str, criterion_id: str, inst: dict[str, Any], note_key: str) -> None:
        tasks.append(
            {
                "key": f"{split}::{criterion_id}::{pid}::{inst['note_id']}",
                "split": split,
                "plant_id": pid,
                "criterion_id": criterion_id,
                "note_id": inst["note_id"],
                "note": inst[note_key],
                "transcript": inst.get("transcript") or "",
                "planted_span": "" if split == "gold" else (inst["plant"].get("planted_span") or ""),
                "criterion": catalog[criterion_id],
            }
        )

    gold = _gold_notes()
    if experiment == "bundled_rules":
        for pid in ("P2", "P3", "P4"):
            for inst in _load_plant(pid):
                add("planted", pid, "bundled_rules", inst, "perturbed_note")
        for inst in gold:
            add("gold", "GOLD", "bundled_rules", inst, "original_note")
    elif experiment == "scaffold_bundled":
        for pid in ("P2", "P3", "P4"):
            for inst in _load_plant(pid):
                add("planted", pid, "scaffold_bundled", inst, "perturbed_note")
        for inst in gold:
            add("gold", "GOLD", "scaffold_bundled", inst, "original_note")
    elif experiment == "scaffold_split":
        for pid in ("P2", "P3", "P4"):
            cid = f"scaffold_split_{pid}"
            for inst in _load_plant(pid):
                add("planted", pid, cid, inst, "perturbed_note")
            for inst in gold:
                add("gold", pid, cid, inst, "original_note")
    elif experiment == "family_p5":
        for cid in ("p5_composite", "p5_objectified"):
            for inst in _load_plant("P5"):
                add("planted", "P5", cid, inst, "perturbed_note")
            for inst in gold:
                add("gold", "P5", cid, inst, "original_note")
    elif experiment == "family_p7":
        for cid in ("p7_composite", "p7_objectified"):
            for inst in _load_plant("P7"):
                add("planted", "P7", cid, inst, "perturbed_note")
            for inst in gold:
                add("gold", "P7", cid, inst, "original_note")
    elif experiment == "thin_misplacement":
        # Same wording as the main multi-vendor ablation. No scaffold.
        for pid in ("P1", "P2", "P3", "P4"):
            for inst in _load_plant(pid):
                add("planted", pid, f"thin_{pid}", inst, "perturbed_note")
                add("planted", pid, "thin_complaint", inst, "perturbed_note")
        for pid in ("P2", "P3", "P4"):
            for inst in gold:
                add("gold", pid, f"thin_{pid}", inst, "original_note")
        for inst in gold:
            add("gold", "GOLD", "thin_complaint", inst, "original_note")
    elif experiment == "alt_decomp":
        seen_gold: set[str] = set()
        for pid, cid in ALT_FOR_PLANT.items():
            for inst in _load_plant(pid):
                add("planted", pid, cid, inst, "perturbed_note")
            if cid not in seen_gold:
                seen_gold.add(cid)
                for inst in gold:
                    add("gold", cid, cid, inst, "original_note")
    else:
        raise SystemExit(f"unknown experiment {experiment}")
    return tasks


def _extract_obj(text: str) -> dict | None:
    t = (text or "").strip()
    fence = __import__("re").search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _relax(obj: dict) -> dict:
    """Fill fields the schema requires without inventing a violation.

    Qwen MaaS does not always honor response_json_schema and names the list
    note_sentences. Alias that list; do not synthesize quotes.
    """
    notes = obj.get("note_sentences")
    inst = obj.get("instances")
    if isinstance(notes, list) and notes and not inst:
        obj["instances"] = notes
        inst = notes
    # One violation object flattened at the top level, with no list.
    if not inst and isinstance(obj.get("note_sentence"), str) and obj.get("note_sentence"):
        inst = [
            {
                "note_sentence": obj.get("note_sentence") or "",
                "section": obj.get("section") or "",
                "classification": obj.get("classification") or (
                    "violation" if obj.get("is_failure") else "acceptable"
                ),
                "rationale": obj.get("rationale") or "",
                "confidence": obj.get("confidence") if isinstance(obj.get("confidence"), (int, float)) else 1.0,
            }
        ]
        obj["instances"] = inst
    if inst is None:
        obj["instances"] = []
    elif isinstance(inst, list):
        for item in inst:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("classification"), str):
                item["classification"] = item["classification"].strip().lower()
            item.setdefault("confidence", 1.0)
            item.setdefault("section", "")
            item.setdefault("rationale", "")
            if not isinstance(item.get("note_sentence"), str):
                item["note_sentence"] = ""
    if not isinstance(obj.get("failure_reason"), str):
        obj["failure_reason"] = "" if not obj.get("is_failure") else "unspecified"
    if not isinstance(obj.get("total_sentences"), int):
        obj["total_sentences"] = len(obj.get("instances") or [])
    if not isinstance(obj.get("violation_count"), int):
        obj["violation_count"] = sum(
            1
            for i in obj.get("instances") or []
            if isinstance(i, dict) and str(i.get("classification", "")).lower() == "violation"
        )
    if not isinstance(obj.get("is_failure"), bool):
        obj["is_failure"] = obj["violation_count"] > 0
    return obj


def _parse_detail(text: str) -> str:
    """Why a response failed validation, without echoing note text."""
    obj = _extract_obj(text)
    if obj is None:
        return f"no_json len={len(text or '')}"
    try:
        CompositeResult.model_validate(_relax(dict(obj)))
    except Exception as e:  # noqa: BLE001
        locs = []
        errors = getattr(e, "errors", None)
        if errors:
            for err in errors()[:4]:
                locs.append(f"{err.get('loc')}:{err.get('type')}")
        inst = obj.get("instances")
        return (
            f"keys={sorted(obj)} is_failure={obj.get('is_failure')!r} "
            f"vc={obj.get('violation_count')!r} "
            f"n_inst={len(inst) if isinstance(inst, list) else type(inst).__name__} "
            f"{type(e).__name__} {'; '.join(locs) or str(e).splitlines()[0][:80]}"
        )
    return "parsed"


def evaluate(backend: Any, task: dict[str, Any]) -> Any:
    prompt = PROMPT_TEMPLATE.format(
        criterion=task["criterion"],
        note=task["note"] or "",
        transcript=task["transcript"] or "",
    )
    last: BaseException | None = None
    for attempt in range(4):
        try:
            raw = backend.generate(prompt)
            parsed = parse_response(raw)
            if parsed is None:
                obj = _extract_obj(raw)
                if obj is not None:
                    try:
                        parsed = normalize_result(
                            CompositeResult.model_validate(_relax(obj))
                        )
                    except Exception:
                        parsed = None
            if parsed is not None:
                return parsed
            print(f"  ! parse failed: {_parse_detail(raw)}", flush=True)
            if attempt < 3:
                time.sleep(2**attempt + 1)
                continue
            return None
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_retryable(e) and attempt < 3:
                time.sleep(2**attempt + 1)
                continue
            print(f"  ! generate failed: {type(e).__name__}: {str(e).splitlines()[0][:180]}", flush=True)
            return None
    print(f"  ! generate failed after retries: {last!r}", flush=True)
    return None


def run_experiment(backend: Any, model: str, backend_name: str, experiment: str, workers: int, limit: int) -> None:
    out_dir = OUT_ROOT / slug_model(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"{experiment}.checkpoint.jsonl"
    out_path = out_dir / f"{experiment}.json"
    tasks = build_tasks(experiment)
    if limit:
        tasks = tasks[:limit]
    task_keys = {t["key"] for t in tasks}
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") not in task_keys or not rec.get("eval_result"):
                continue
            try:
                parsed = normalize_result(CompositeResult.model_validate(rec["eval_result"]))
            except Exception:
                continue
            if parsed is None:
                continue
            rec["eval_result"] = parsed.model_dump()
            cached[rec["key"]] = rec
    todo = [t for t in tasks if t["key"] not in cached]
    print(
        f"[{model} {experiment}] tasks={len(tasks)} cached={len(cached)} todo={len(todo)} workers={workers}",
        flush=True,
    )
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(task: dict[str, Any]) -> dict[str, Any]:
        parsed = evaluate(backend, task)
        row = {
            "key": task["key"],
            "split": task["split"],
            "plant_id": task["plant_id"],
            "criterion_id": task["criterion_id"],
            "note_id": task["note_id"],
            "planted_span": task["planted_span"],
            "eval_result": parsed.model_dump() if parsed else None,
            "eval_error": None if parsed else "generate_or_parse_failure",
        }
        with lock:
            done[0] += 1
            with cache_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            if done[0] % 10 == 0 or done[0] == len(todo):
                print(
                    f"[{model} {experiment}] {done[0]}/{len(todo)} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
        return row

    rows = list(cached.values())
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for fut in as_completed(futs):
                rows.append(fut.result())

    # Drop note text from the saved payload. Criteria live in criteria.json.
    payload = {
        "experiment": experiment,
        "model": model,
        "backend": backend_name,
        "temperature": getattr(backend, "temperature", 0.0),
        "n": len(rows),
        "n_ok": sum(1 for r in rows if r.get("eval_result")),
        "results": rows,
    }
    out_path.write_text(json.dumps(payload))
    print(
        f"[{model} {experiment}] wrote {out_path.name} ok={payload['n_ok']}/{payload['n']} "
        f"wall={time.time() - t0:.0f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("openai", "vertex"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--experiments",
        default="bundled_rules,scaffold_bundled,scaffold_split,family_p5,family_p7,alt_decomp",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "criteria.json").write_text(json.dumps(criteria_catalog(), indent=2))

    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    backend = make_backend(args.backend, args.model)
    for experiment in [e.strip() for e in args.experiments.split(",") if e.strip()]:
        run_experiment(backend, args.model, args.backend, experiment, args.workers, args.limit)


if __name__ == "__main__":
    main()
