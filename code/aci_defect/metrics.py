"""Reference metrics for ACI-Defect.

ROUGE-1/2/L are implemented locally (no extra deps) so the experiment is
reproducible. MedCon is a UMLS-free *concept F1 proxy*: gazetteer + dose/lab
patterns. It is labeled medcon_proxy in all tables — not QuickUMLS/MetaMap.
TF-IDF cosine stands in for BERTScore when transformers are unavailable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

# Conditions / findings / tests that dominate ACI-bench notes.
_GAZETTEER = frozenset(
    t.lower()
    for t in """
    hypertension diabetes mellitus type depression anxiety gerd reflux arthritis
    gout hypothyroidism hyperlipidemia atrial fibrillation afib copd asthma
    congestive heart failure chf copd pneumonia bronchitis sinusitis migraine
    kidney transplant renal stones nephrolithiasis anemia obesity osa sleep apnea
    cad coronary artery disease cabg stent murmur edema rhonchi wheeze rales
    metformin lisinopril lasix furosemide carvedilol metoprolol amlodipine
    losartan atorvastatin simvastatin omeprazole pantoprazole protonix tylenol
    ibuprofen tramadol ultram lortab hydrocodone oxycodone prednisone gabapentin
    allopurinol methotrexate prozac fluoxetine zoloft sertraline lexapro
    wellbutrin synthroid levothyroxine lantus insulin farxiga jardiance ozempic
    mounjaro wegovy imitrex sumatriptan robitussin digoxin warfarin eliquis
    echocardiogram echocardiogram ecg ekg x-ray radiograph ct mri ultrasound
    hemoglobin a1c hba1c creatinine glucose lipid cholesterol ejection fraction
    mammogram colonoscopy ablation cardiology rheumatology physical therapy
    mammogram vaccine covid shingles
    """.split()
)

_DOSE_RE = re.compile(
    r"\b[a-z][a-z0-9\-]{2,}\s+\d+(?:\.\d+)?\s*(?:mg|mcg|units?|meq)\b",
    re.I,
)
_LAB_RE = re.compile(
    r"\b(?:a1c|hba1c|hemoglobin a1c|ef|ejection fraction|creatinine|glucose|bmi|bp)"
    r"\s*(?:is|of|at|=|:)?\s*\d+(?:\.\d+)?",
    re.I,
)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _f1(overlap: int, pred: int, gold: int) -> float:
    if pred == 0 and gold == 0:
        return 1.0
    if overlap == 0:
        return 0.0
    p = overlap / pred if pred else 0.0
    r = overlap / gold if gold else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def rouge_n(hyp: str, ref: str, n: int) -> float:
    h = ngrams(tokenize(hyp), n)
    r = ngrams(tokenize(ref), n)
    if not h and not r:
        return 1.0
    hc, rc = Counter(h), Counter(r)
    overlap = sum((hc & rc).values())
    return _f1(overlap, sum(hc.values()), sum(rc.values()))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    # DP on shorter axis
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp: str, ref: str) -> float:
    h, r = tokenize(hyp), tokenize(ref)
    if not h and not r:
        return 1.0
    lcs = _lcs_len(h, r)
    return _f1(lcs, len(h), len(r))


def extract_concepts(text: str) -> set[str]:
    toks = tokenize(text)
    concepts = {t for t in toks if t in _GAZETTEER}
    for m in _DOSE_RE.finditer(text or ""):
        concepts.add(re.sub(r"\s+", " ", m.group(0).lower()))
    for m in _LAB_RE.finditer(text or ""):
        concepts.add(re.sub(r"\s+", " ", m.group(0).lower()))
    # bigrams that are both gazetteer-ish medical compounds
    for i in range(len(toks) - 1):
        bi = f"{toks[i]} {toks[i + 1]}"
        if toks[i] in _GAZETTEER and toks[i + 1] in _GAZETTEER:
            concepts.add(bi)
    return concepts


def medcon_proxy_f1(hyp: str, ref: str) -> float:
    h, r = extract_concepts(hyp), extract_concepts(ref)
    if not h and not r:
        return 1.0
    overlap = len(h & r)
    return _f1(overlap, len(h), len(r))


def tfidf_cosine(hyp: str, ref: str) -> float:
    """Character 3-gram cosine — cheap lexical-embedding stand-in."""
    def grams(text: str) -> Counter:
        s = re.sub(r"\s+", " ", (text or "").lower())
        if len(s) < 3:
            return Counter([s])
        return Counter(s[i : i + 3] for i in range(len(s) - 2))

    a, b = grams(hyp), grams(ref)
    if not a and not b:
        return 1.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def word_count(text: str) -> int:
    return len(tokenize(text))


def score_pair(hyp: str, ref: str) -> dict[str, float]:
    return {
        "rouge1": round(rouge_n(hyp, ref, 1), 6),
        "rouge2": round(rouge_n(hyp, ref, 2), 6),
        "rougeL": round(rouge_l(hyp, ref), 6),
        "medcon_proxy": round(medcon_proxy_f1(hyp, ref), 6),
        "tfidf_cosine": round(tfidf_cosine(hyp, ref), 6),
        "hyp_words": float(word_count(hyp)),
        "ref_words": float(word_count(ref)),
        "delta_words": float(word_count(hyp) - word_count(ref)),
    }
