#!/usr/bin/env python3
"""Multi-vendor composite-vs-objectified ablation.

Same protocol as run_composite_ablation.py: temperature 0, instance-level JSON,
planted-span hit rule applied later. The only intended differences across arms
are the criterion text (and, for objectified, one directional pair per plant).

Backends:
  openai     OPENAI_API_KEY
  anthropic  ANTHROPIC_API_KEY
  vertex     ADC + GOOGLE_CLOUD_PROJECT

Example:
  python run_multivendor_ablation.py --backend openai --model gpt-4.1 \\
      --arm complaint --plants P1 P2 P3 P4
  python run_multivendor_ablation.py --backend anthropic --model claude-sonnet-4-6 \\
      --arm objectified --gold --plants P2 P3 P4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ValidationError, confloat

_ACI = Path(__file__).resolve().parent
DEFAULT_PLANTS_DIR = _ACI.parents[1] / "data" / "aci_defect_plants"
OUT_ROOT = _ACI.parents[1] / "reproduced_output" / "composite_ablation_multivendor"

PLANT_FILES = {
    "P1": "P1_ap_into_hpi",
    "P2": "P2_hpi_into_ap",
    "P3": "P3_symptom_into_pe",
    "P4": "P4_imaging_into_pe",
}

# Scoped directional criteria — the objectification under test, not the
# withheld production prompts. Same JSON schema as the composite arms.
OBJECTIFIED_CRITERIA = {
    "P1": (
        "A History-section sentence contains Assessment/Plan content: a clinician "
        "impression, definitive diagnosis, or management decision. Do not flag "
        "symptom narrative, patient quotes, or brief non-decisional history."
    ),
    "P2": (
        "An Assessment-and-Plan sentence contains History-of-present-illness "
        "narrative (symptom story or patient-reported history) rather than "
        "assessment or plan."
    ),
    "P3": (
        "A Physical-Exam sentence reports subjective symptom content that belongs "
        "in History, not objective exam findings."
    ),
    "P4": (
        "A Physical-Exam sentence reports laboratory, imaging, or other Results "
        "content rather than bedside exam findings."
    ),
}

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
    "objectified": {
        "criterion": "DIRECTIONAL_PER_PLANT",
        "source": "scoped directional pair; one ordered section pair per call",
    },
}


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
    instances: List[CompositeInstance]


def normalize_result(parsed: CompositeResult) -> Optional[CompositeResult]:
    """Repair summary fields from quoted instances; never invent instances."""
    actual = sum(i.classification == "violation" for i in parsed.instances)
    if parsed.violation_count > 0 and actual == 0:
        return None
    parsed.violation_count = actual
    parsed.is_failure = actual > 0
    if actual == 0:
        parsed.failure_reason = "No violation detected"
    return parsed


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


def slug_model(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def criterion_for(arm: str, plant_id: str) -> str:
    if arm == "objectified":
        if plant_id not in OBJECTIFIED_CRITERIA:
            raise ValueError(f"objectified arm has no criterion for {plant_id}")
        return OBJECTIFIED_CRITERIA[plant_id]
    return ARMS[arm]["criterion"]


def build_prompt(arm: str, plant_id: str, note: str, transcript: str) -> str:
    return PROMPT_TEMPLATE.format(
        criterion=criterion_for(arm, plant_id),
        note=note or "",
        transcript=transcript or "",
    )


def parse_response(text: str) -> Optional[CompositeResult]:
    try:
        t = (text or "").strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
        if fence:
            t = fence.group(1).strip()
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1:
            t = t[start : end + 1]
        return normalize_result(CompositeResult.model_validate(json.loads(t)))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    s = str(exc).lower()
    return any(
        tok in s
        for tok in ("429", "rate", "timeout", "503", "502", "overloaded", "resource_exhausted")
    )


class Backend:
    temperature: float | str = 0.0

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIBackend(Backend):
    def __init__(self, model: str) -> None:
        from openai import OpenAI

        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not key:
            raise SystemExit("Set OPENAI_API_KEY")
        self.client = OpenAI()
        self.model = model
        self.uses_default_temperature = model.lower().startswith("gpt-5")
        self.temperature = "default" if self.uses_default_temperature else 0.0

    def generate(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return valid JSON only. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": CompositeResult,
            "max_completion_tokens": 4096,
        }
        if self.uses_default_temperature:
            resp = self.client.chat.completions.parse(
                **kwargs,
                reasoning_effort="minimal",
            )
        else:
            resp = self.client.chat.completions.parse(**kwargs, temperature=0)
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            return resp.choices[0].message.content or ""
        return parsed.model_dump_json()


class AnthropicBackend(Backend):
    def __init__(self, model: str) -> None:
        import anthropic

        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise SystemExit("Set ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = 0.0

    def generate(self, prompt: str) -> str:
        # anthropic>=1.7 dropped the temperature kwarg; pass it via extra_body.
        msg = self.client.messages.parse(
            model=self.model,
            max_tokens=8192,
            system="Return valid JSON only. No markdown.",
            messages=[{"role": "user", "content": prompt}],
            output_format=CompositeResult,
            extra_body={"temperature": 0},
        )
        parsed = getattr(msg, "parsed_output", None)
        if parsed is not None:
            return parsed.model_dump_json()
        parts = []
        for block in msg.content:
            if getattr(block, "type", "") == "text":
                parts.append(block.text)
        return "".join(parts)


class VertexBackend(Backend):
    def __init__(self, model: str) -> None:
        from google import genai
        from google.genai import types
        from google.genai.types import HttpOptions

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
            "VERTEX_PROJECT"
        )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        self.types = types
        self.client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=CompositeResult.model_json_schema(),
                temperature=0.0,
            ),
        )
        return resp.text or ""


def make_backend(name: str, model: str) -> Backend:
    if name == "openai":
        return OpenAIBackend(model)
    if name == "anthropic":
        return AnthropicBackend(model)
    if name == "vertex":
        return VertexBackend(model)
    raise SystemExit(f"unknown backend {name}")


def evaluate(backend: Backend, arm: str, plant_id: str, note: str, transcript: str) -> Optional[CompositeResult]:
    prompt = build_prompt(arm, plant_id, note, transcript)
    last: BaseException | None = None
    for attempt in range(4):
        try:
            parsed = parse_response(backend.generate(prompt))
            if parsed is not None:
                return parsed
            if attempt < 3:
                time.sleep(2**attempt + 1)
                continue
            return None
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_retryable(e) and attempt < 3:
                time.sleep(2**attempt + 1)
                continue
            print(f"  ! generate failed: {e!r}", flush=True)
            return None
    print(f"  ! generate failed after retries: {last!r}", flush=True)
    return None


def load_tasks(
    arm: str, plants: list[str], gold: bool, plants_dir: Path
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if gold:
        blob = json.loads((plants_dir / "P3_symptom_into_pe.json").read_text())
        # Objectified clean-note specificity is criterion-specific. Composite
        # arms need only one call per clean note, matching the paper protocol.
        pids = (plants or ["P2", "P3", "P4"]) if arm == "objectified" else ["GOLD"]
        for pid in pids:
            for inst in blob["instances"]:
                tasks.append(
                    {
                        "key": (
                            f"GOLD::{pid}::{inst['note_id']}"
                            if arm == "objectified"
                            else f"GOLD::{inst['note_id']}"
                        ),
                        "plant_id": pid,
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
    ap.add_argument("--backend", choices=("openai", "anthropic", "vertex"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--plants", nargs="*", default=["P1", "P2", "P3", "P4"])
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--plants-dir", type=Path, default=DEFAULT_PLANTS_DIR)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.arm == "zhou" and not args.gold:
        args.plants = [p for p in args.plants if p in ("P1", "P2")] or ["P1", "P2"]
    if args.arm == "objectified" and args.gold and args.plants == ["P1", "P2", "P3", "P4"]:
        args.plants = ["P2", "P3", "P4"]

    out_dir = args.out_root / slug_model(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_{'GOLD' if args.gold else ''.join(args.plants)}"
    cache_path = out_dir / f"{tag}.checkpoint.jsonl"
    out_path = out_dir / f"{tag}.json"

    tasks = load_tasks(args.arm, args.plants, args.gold, args.plants_dir)
    if args.limit:
        tasks = tasks[: args.limit]
    task_keys = {t["key"] for t in tasks}
    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("key") in task_keys
                and rec.get("eval_result")
            ):
                try:
                    parsed = normalize_result(
                        CompositeResult.model_validate(rec["eval_result"])
                    )
                except ValidationError:
                    continue
                if parsed is None:
                    continue
                rec["eval_result"] = parsed.model_dump()
                cached[rec["key"]] = rec

    todo = [t for t in tasks if t["key"] not in cached]
    print(
        f"[{args.model} {tag}] tasks={len(tasks)} cached={len(cached)} "
        f"todo={len(todo)} backend={args.backend} workers={args.workers}",
        flush=True,
    )
    backend = make_backend(args.backend, args.model)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    def work(task: dict[str, Any]) -> dict[str, Any]:
        parsed = evaluate(
            backend, args.arm, task["plant_id"], task["note"], task["transcript"]
        )
        row = {
            "key": task["key"],
            "plant_id": task["plant_id"],
            "note_id": task["note_id"],
            "planted_span": task["planted_span"],
            "eval_result": parsed.model_dump() if parsed else None,
            "eval_error": None if parsed else "generate_or_parse_failure",
            "criterion": criterion_for(args.arm, task["plant_id"]),
        }
        with lock:
            done[0] += 1
            with cache_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            vc = (row["eval_result"] or {}).get("violation_count")
            print(
                f"[{args.model} {tag}] {done[0]}/{len(todo)} {task['key']} "
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

    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            have = {r.get("key") for r in rows}
            for r in prev.get("results") or []:
                if r.get("key") and r["key"] not in have:
                    rows.append(r)
        except json.JSONDecodeError:
            pass

    payload = {
        "arm": args.arm,
        "model": args.model,
        "backend": args.backend,
        "temperature": backend.temperature,
        "criterion": ARMS[args.arm]["criterion"],
        "criterion_source": ARMS[args.arm]["source"],
        "objectified_criteria": OBJECTIFIED_CRITERIA if args.arm == "objectified" else None,
        "n": len(rows),
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    ok = sum(1 for r in rows if r.get("eval_result"))
    print(
        f"[{args.model} {tag}] wrote {out_path} ok={ok}/{len(rows)} "
        f"wall={time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
