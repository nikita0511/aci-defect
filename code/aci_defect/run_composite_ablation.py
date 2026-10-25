#!/usr/bin/env python3
"""Composite-vs-objectified ablation on ACI-Defect misplacement plants.

Holds the model, the inputs, the decoding config, the output grain, and the
scoring rule fixed, and varies only *how the criterion is posed*:

  arm=zhou      the composite criterion Zhou et al. flag as unenforceable and
                delete, quoted verbatim from their appendix
  arm=complaint the composite as clinicians write it in production feedback

Both arms are asked for instance-level verdicts with an exactly-quoted note
sentence, so the baseline is *not* handicapped by a note-level yes/no output.
Any recall gap is therefore attributable to the criterion wording, which is
the claim under test (objectification, not judge scaffolding).

Usage:
  python -m aci_defect.run_composite_ablation --arm complaint --plants P1 P2 P3 P4
  python -m aci_defect.run_composite_ablation --arm zhou --plants P1 P2
  python -m aci_defect.run_composite_ablation --arm complaint --gold

Requires a Vertex AI project: set GOOGLE_CLOUD_PROJECT (and optionally
GOOGLE_CLOUD_LOCATION, default "global") and authenticate with
`gcloud auth application-default login`.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Literal, Optional

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pydantic import BaseModel, ValidationError, confloat

_ACI = Path(__file__).resolve().parent

MODEL = os.environ.get("ABLATION_MODEL", "gemini-2.5-pro")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

DEFAULT_PLANTS_DIR = _ACI.parents[1] / "data" / "aci_defect_plants"
OUT_DIR = _ACI.parents[1] / "reproduced_output" / "composite_ablation"

PLANT_FILES = {
    "P1": "P1_ap_into_hpi",
    "P2": "P2_hpi_into_ap",
    "P3": "P3_symptom_into_pe",
    "P4": "P4_imaging_into_pe",
}


# ---------------------------------------------------------------- schema
class CompositeInstance(BaseModel):
    note_sentence: str
    section: Optional[str] = None
    classification: Literal["violation", "acceptable"]
    rationale: str
    confidence: confloat(ge=0.0, le=1.0)


class CompositeResult(BaseModel):
    is_failure: bool
    failure_reason: str
    total_sentences: int
    violation_count: int
    instances: List[CompositeInstance] = []


# ---------------------------------------------------------------- prompts
# The composite criterion is stated as posed and NOT decomposed: no ordered
# section pair, no scope restriction, no inclusion/exclusion list. That
# omission is the independent variable.
ARMS = {
    "zhou": {
        "criterion": (
            "Is there a clear separation between the history of present illness "
            "(HPI) and the action plan to ensure there is no overlap?"
        ),
        "source": "Zhou et al. appendix, quoted verbatim; discarded at their enforceability gate",
    },
    "complaint": {
        "criterion": "Content is in the wrong section of the note.",
        "source": "composite complaint as clinicians write it in production feedback",
    },
}

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


def build_prompt(arm: str, note: str, transcript: str) -> str:
    return PROMPT_TEMPLATE.format(
        criterion=ARMS[arm]["criterion"],
        note=note or "",
        transcript=transcript or "",
    )


def parse_response(text: str) -> Optional[CompositeResult]:
    try:
        t = (text or "").strip()
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1:
            t = t[start : end + 1]
        return CompositeResult.model_validate(json.loads(t))
    except (json.JSONDecodeError, ValidationError):
        return None


def evaluate(client, arm: str, note: str, transcript: str) -> Optional[CompositeResult]:
    prompt = build_prompt(arm, note, transcript)
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=CompositeResult.model_json_schema(),
                    temperature=0.0,
                ),
            )
            return parse_response(resp.text or "")
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2**attempt + 1)
            else:
                print(f"  ! generate failed: {e!r}", flush=True)
                return None
    return None


# ---------------------------------------------------------------- tasks
def load_tasks(plants: list[str], gold: bool, plants_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if gold:
        blob = json.loads((plants_dir / "P3_symptom_into_pe.json").read_text())
        for inst in blob["instances"]:
            tasks.append(
                {
                    "key": f"GOLD::{inst['note_id']}",
                    "plant_id": "GOLD",
                    "note_id": inst["note_id"],
                    "note": inst["original_note"],
                    "transcript": inst.get("transcript") or "",
                    "planted_span": "",
                }
            )
        return tasks
    for pid in plants:
        blob = json.loads((plants_dir / f"{PLANT_FILES[pid]}.json").read_text())
        for inst in blob["instances"]:
            tasks.append(
                {
                    "key": f"{pid}::{inst['note_id']}",
                    "plant_id": pid,
                    "note_id": inst["note_id"],
                    "note": inst["perturbed_note"],
                    "transcript": inst.get("transcript") or "",
                    "planted_span": inst["plant"].get("planted_span") or "",
                }
            )
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--plants", nargs="*", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--gold", action="store_true", help="run on unperturbed test1 notes")
    ap.add_argument("--plants-dir", type=Path, default=DEFAULT_PLANTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: cap task count")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_{'GOLD' if args.gold else ''.join(args.plants)}"
    cache_path = out_dir / f"{tag}.checkpoint.jsonl"
    out_path = out_dir / f"{tag}.json"

    tasks = load_tasks(args.plants, args.gold, args.plants_dir)
    if args.limit:
        tasks = tasks[: args.limit]
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") and rec.get("eval_result"):
                cached[rec["key"]] = rec

    todo = [t for t in tasks if t["key"] not in cached]
    print(
        f"[{tag}] tasks={len(tasks)} cached={len(cached)} todo={len(todo)} "
        f"model={MODEL} workers={args.workers}",
        flush=True,
    )

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT to a Vertex AI project.")
    client = genai.Client(
        vertexai=True,
        project=project,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(task: dict[str, Any]) -> dict[str, Any]:
        parsed = evaluate(client, args.arm, task["note"], task["transcript"])
        row = {
            "key": task["key"],
            "plant_id": task["plant_id"],
            "note_id": task["note_id"],
            "planted_span": task["planted_span"],
            "eval_result": parsed.model_dump() if parsed else None,
            "eval_error": None if parsed else "generate_or_parse_failure",
        }
        with lock:
            done[0] += 1
            with cache_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            vc = (row["eval_result"] or {}).get("violation_count")
            print(
                f"[{tag}] {done[0]}/{len(todo)} {task['key']} "
                f"{'ok' if parsed else 'FAIL'} violations={vc} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        return row

    rows = list(cached.values())
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for f in as_completed(futs):
                rows.append(f.result())

    payload = {
        "arm": args.arm,
        "criterion": ARMS[args.arm]["criterion"],
        "criterion_source": ARMS[args.arm]["source"],
        "model": MODEL,
        "temperature": 0.0,
        "backend": "vertex-sync",
        "n": len(rows),
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    ok = sum(1 for r in rows if r.get("eval_result"))
    print(
        f"[{tag}] wrote {out_path} ok={ok}/{len(rows)} wall={time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
