#!/usr/bin/env python3
"""BERTScore-F1 of each planted note vs original gold (P1–P11).

Drop = 1 − F1, same convention as ROUGE-1 / MedCon-proxy. Model is the
bert-score default for English (roberta-large). Notes longer than the
model context (512 tokens) are truncated, which is the library default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from bert_score import score

DEFAULT_PLANTS_DIR = Path(__file__).resolve().parent / "output" / "plants"
DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "bertscore_drops.json"

# Paper Table 3 plants (no P12, no CTRL).
PLANT_IDS = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9",
    "P10",
    "P11",
]


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_pairs(plant_id: str, plants_dir: Path) -> tuple[list[str], list[str], list[str]]:
    paths = sorted(plants_dir.glob(f"{plant_id}_*.json"))
    if not paths:
        raise FileNotFoundError(f"no plant file for {plant_id} in {plants_dir}")
    payload = json.loads(paths[0].read_text())
    hyps, refs, ids = [], [], []
    for inst in payload["instances"]:
        hyps.append(inst["perturbed_note"])
        refs.append(inst["original_note"])
        ids.append(inst["note_id"])
    return hyps, refs, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plants-dir", type=Path, default=DEFAULT_PLANTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    device = _device()
    print(f"device={device}")
    results = {}
    for pid in PLANT_IDS:
        hyps, refs, ids = _load_pairs(pid, args.plants_dir)
        print(f"{pid}: scoring {len(hyps)} pairs")
        _p, _r, f1 = score(
            hyps,
            refs,
            lang="en",
            model_type="roberta-large",
            device=device,
            batch_size=4 if device == "cpu" else 8,
            verbose=True,
        )
        f1_list = [float(x) for x in f1]
        drops = [1.0 - x for x in f1_list]
        mean_f1 = sum(f1_list) / len(f1_list)
        mean_drop = sum(drops) / len(drops)
        results[pid] = {
            "n": len(hyps),
            "mean_f1": round(mean_f1, 6),
            "mean_drop": round(mean_drop, 6),
            "min_drop": round(min(drops), 6),
            "max_drop": round(max(drops), 6),
            "per_note": [
                {"note_id": nid, "f1": round(f, 6), "drop": round(d, 6)}
                for nid, f, d in zip(ids, f1_list, drops)
            ],
        }
        print(
            f"{pid}: n={len(hyps)}  BERTScore-F1={mean_f1:.3f}  "
            f"drop={mean_drop:.3f}  (min {min(drops):.3f}, max {max(drops):.3f})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print()
    print("| ID | n | BERTScore-F1 | drop |")
    print("|---|---:|---:|---:|")
    for pid in PLANT_IDS:
        r = results[pid]
        print(f"| {pid} | {r['n']} | {r['mean_f1']:.3f} | {r['mean_drop']:.3f} |")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
