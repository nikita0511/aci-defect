# Data card

## Source corpus

The public source is ACI-bench test1 (40 encounters), keyed by the stable
`aci-D2N###` note IDs. Source text is preserved because the plants are defined
as surgical transformations of those notes. Patient names and encounter details
are inherited from the released research corpus and are not production data.

## Clinical labels

`hallucination_sentence_labels_corrected.csv` has one row per reference-note
sentence. The positive endpoint is any fabrication, transformation, inference,
or misattribution; benign-only labels are negative, and `corrected_types` keeps
the winning subtype where one was recorded.

`omission_case_fact_labels_corrected.csv` has one row per generated case fact.
The positive endpoint is `inappropriately missing` or `inappropriately
partially missing`. Appropriate omissions, and facts actually present in the
note, are negative.

In both files `independent_*` holds the evaluator-hidden review, `adjudication_*`
is populated only for disagreements, `evaluation_inclusion=excluded` marks grain
differences and unresolved cases, and `corrected_binary_label` is the endpoint
used in the reported scoring. `negative_other` means adjudication rejected both
compared labels; it is negative for this endpoint.

`second_rater_labels.csv` has one row per item on the 10 doubly labeled notes.
`first_pass_*` is the evaluator-hidden first pass, `second_pass_*` the
independent second pass, and `corrected_binary_label` the adjudicated endpoint
for reference. Rater IDs (`rater_A`-`rater_C`) are pseudonymous and stable
across rows. `results/irr_summary.json` holds kappa, bootstrap intervals,
prevalence-adjusted agreement, and the cross-tab against adjudication.

## ACI-Defect

Each plant file carries the original note, the perturbed note, the public
transcript, the case facts, the operation metadata, and deterministic integrity
checks. Reference metrics compare each perturbed note with its own unperturbed
source, so the self-similarity ceiling is 1.0 and the reported drop is
`1 - score`. `medcon_proxy` is a UMLS-free concept F1 over a gazetteer plus
dose and lab patterns, not official MedCon. BERTScore is `roberta-large` F1.

## Result tables

- `c2_grounding_summary.csv`: grounding-judge precision and recall against the
  clinical labels, before and after adjudication, with the size of the positive
  class on each side of the adjudication pass.
- `c3_plant_summary.csv`, `reference_metric_records.csv`,
  `bertscore_records.json`: planted-span recall against reference-metric drops.
  `coverage_note` records why a check misses part of its own plant; it is a
  property of that check, and every row is inside the reported median and range.
- `ablation_summary.csv`, `ablation_clean_note_flags.csv`: the objectified and
  composite arms, and where an un-scoped criterion sends content on clean notes.
  Buckets are assigned automatically from the composite judge's own rationale
  text and are not clinician labels. On the unperturbed notes an objectified
  judge abstains where its source section is absent, which is why the
  HPI-into-A&P row is over 39 notes rather than 40.
- `robustness/`, `robustness_*.csv`: matched-effort, bundled-rule,
  content-class, and family runs on three models. Judge payloads keep quoted
  note sentences and rationales but not full note text; the notes are in
  `data/aci_defect_plants/`.
- `c4_rating_bands.csv`, `c4_correlations.csv`, `c4_named_family_rates.csv`:
  de-identified aggregates of the production rating analysis. Rates are per
  opportunity, `mean_z_instance_burden` is the mean standardized instance rate
  across judges, and judges abstain when a section or cue is absent, which is
  why `*_notes_scored` varies by judge.

## Limitations

The clinical labels cover grounding only, hallucination and omission; they do
not certify the leakage or style checks. Adjudication was not blinded, so
corrected precision is an upper bound and the raw figures are retained
alongside it. Agreement was measured on 10 of the 40 notes, and span-level
kappa for hallucination is moderate. Objectification was done by one clinician;
agreement between independent rewriters was not measured, and the content-class
decomposition is a proxy for it. Matched effort covers three models and P2-P4
only. Omission is scored against generated case facts rather than an
independently authored fact list. P2 is bounded by the evaluator's extraction
window and P8 by a finite brand-name gazetteer. Only test1 is covered, not full
ACI-bench. The production material is aggregate-only and cannot be re-derived
from this archive.
