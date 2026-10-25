# ACI-Defect

Data, labels, and code for *Objectify, Don't Delete: Complaint-Derived Evaluation of Ambient Clinical Notes* (GenAI4Health Workshop at NeurIPS 2026).

Repository: https://github.com/nikita0511/aci-defect

ACI-Defect is a derived overlay on ACI-bench (Yim et al., 2023, CC BY 4.0) and is
not affiliated with the ACI-bench authors.

Licenses: data and labels are CC BY 4.0 (`LICENSE_DATA.txt`, attribution in
`NOTICE.md`); code in `code/` is MIT (`LICENSE`).

Everything releasable here derives from ACI-bench, a public research corpus. The
deployed scribe studied in the paper is not represented: no production notes,
transcripts, clinician comments, or identifiers appear in this archive, and the
production analysis is included only as the aggregates in `results/c4_*.csv`.

## Layout

```
data/aci_bench_test1_source.json   40 ACI-bench test1 encounters with merged case facts
data/aci_defect_plants/            ACI-Defect: P1-P12 plus the random-deletion control
data/labels/                       corrected clinical labels and the second-rater pass
results/                           the tables reported in the paper
results/composite_ablation/        raw judge output for the composite arms
results/robustness/                raw judge output for the robustness battery
code/aci_defect/                   planting, section parsing, detection, metrics, scoring
```

## Corrected clinical labels

Three clinicians split ACI-bench test1 between them and labeled every sentence
and every transcript-derived case fact with evaluator output hidden, so each
item carries one first-pass label. A second clinician who had not labeled the
note relabeled 10 of the 40 notes (915 items) under the same blinding; see
Second-rater agreement below. Every judge-labeler disagreement was then
adjudicated (150 hallucination, 280 omission). After adjudication the
hallucination endpoint has 102 positive, 1765 negative, and 11 excluded
sentences; the omission endpoint has 368 positive, 1686 negative, and 8
excluded case facts.

`corrected_binary_label` is the analysis-ready field. Rows marked `excluded`
were grain differences or unresolved cases and sit outside the precision and
recall denominators. Adjudication was not blinded, so the `independent_*` fields
are kept alongside the corrected ones rather than overwritten; every flip can be
traced to its verdict and comment. Note text is byte-identical to the released
corpus, since automated repair of the reference would make gold an output of the
system being evaluated against it.

## ACI-Defect

Each file in `data/aci_defect_plants/` plants one defect at a time into the 40
test1 notes and records the planted span, so localization can be scored against
a known target. A note is skipped when the required section or cue is absent,
which is why coverage is reported as `n_planted/40` rather than 40 everywhere.
`results/c3_plant_summary.csv` pairs planted-span recall with the drop in
ROUGE-1, BERTScore-F1, and the MedCon proxy on the same notes;
`results/reference_metric_records.csv` is the row-level version.

That summary covers P1-P11, the plants with an evaluated target check. P12 and
the random-deletion control ship with their reference-metric rows in
`reference_metric_records.csv` but carry no planted-span recall: P12's
first-person cue already occurs in the reference notes, so a flag on it cannot
be attributed to the plant, and the control has no target check by
construction.

## Objectification ablation

`results/ablation_summary.csv` holds the arm-by-arm result. All arms share the
model, notes, decoding, response schema, and hit rule. The objectified arm is the
deployed multi-step judge, while the composite arms receive only the criterion
sentence, so the arms differ in prompt effort as well as scope. The robustness
battery below removes that difference. The composite arms' raw output is in
`results/composite_ablation/`, so their recall and their flag rate on
unperturbed notes can be recomputed from this archive:

```bash
PYTHONPATH=code python -m aci_defect.verify_ablation
```

The deployed objectified judges' prompts are not released, so that arm is
reported in `ablation_summary.csv` rather than recomputed;
`code/aci_defect/span_hit.py` is the hit rule applied to every arm.

## Robustness battery

Follow-up runs on Gemini 2.5 Pro, GPT-4.1, and Qwen3-Next-80B at temperature 0,
scored with the same hit rule. Every criterion string is in
`results/robustness/criteria.json`.

- Matched effort (`scaffold_split`, `scaffold_bundled`): both arms receive the
  same anchor-then-classify procedure, allowed/disallowed rules, and three
  calibration examples; only the criterion question differs. Plants P2-P4.
- Bundled rules (`bundled_rules`): the four directional rules in one call.
- Content-class decomposition (`alt_decomp`): the same complaints rewritten by
  content class instead of ordered section pair.
- Families (`family_p5`, `family_p7`): composite versus scoped predicate for a
  fabricated exam finding and for "history of" on an active problem.
- Thin protocol (`thin_misplacement`, Qwen only): the main ablation's wording.

`results/robustness_matched_effort.csv`, `robustness_content_class.csv`, and
`robustness_families.csv` are the summary tables. Mean FPR averages the
per-check clean-note rates; union FPR fails a clean note if any objectified
check fires. Under matched effort, objectified checks lower per-check false
positives on all three models, but the single composite question is the better
note-level screen, and HPI-into-A&P shows no advantage.

Rescore the shipped judge output (writes `reproduced_output/robustness/`):

```bash
PYTHONPATH=code python -m aci_defect.score_robustness_battery
```

Rerunning the judges needs API access (`requirements-robustness.txt`):

```bash
PYTHONPATH=code python -m aci_defect.run_robustness_battery --backend openai --model gpt-4.1
PYTHONPATH=code python -m aci_defect.run_robustness_battery --backend vertex --model gemini-2.5-pro
```

The OpenAI backend reads `OPENAI_API_KEY`; the Vertex backend uses application
default credentials and `GOOGLE_CLOUD_PROJECT`.

## Second-rater agreement

`data/labels/second_rater_labels.csv` is an independent second pass over 10 of
the 40 notes (915 items), by a clinician who had not labeled that note, with the
same instrument and with judge output hidden. Rater IDs are pseudonymous and
every item's two passes are by different clinicians. Against the first pass,
note-level agreement is 10/10 for both dimensions; span-level Cohen's kappa is
0.61 for omission (95% bootstrap CI 0.49-0.71) and 0.41 for hallucination
(0.22-0.58), with raw agreement 0.91 and 0.94. The second rater marks more
defects than the first. This pass measures how reproducible the first pass is;
it does not validate the adjudicated labels, which were not modified by it.

```bash
python code/aci_defect/verify_irr.py
```

## Reproducing the plants and metrics

Python 3.10 or newer. Planting, section parsing, ROUGE, and the MedCon proxy use
only the standard library:

```bash
PYTHONPATH=code python -m aci_defect.run_c2 \
  --input data/aci_bench_test1_source.json \
  --out reproduced_output \
  --seed 20260821
```

BERTScore needs `torch` and `bert-score` (`requirements-bertscore.txt`) and uses
`roberta-large` with the library's default 512-token handling:

```bash
python code/aci_defect/score_bertscore.py \
  --plants-dir data/aci_defect_plants \
  --out reproduced_output/bertscore_drops.json
```

Recall in `results/c3_plant_summary.csv` is planted-span localization by the
evaluated checks, not the deterministic plant-integrity assertion that
`run_c2.py` prints.

`DATA_CARD.md` documents the schemas and limitations, `NOTICE.md` the
attribution, and `MANIFEST.sha256` the file hashes.

## Citation

Cite the paper and ACI-bench; both entries are in `CITATION.bib`, and
`CITATION.cff` covers this repository.
