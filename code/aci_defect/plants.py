"""P1–P12 plant generators plus a generic-deletion control.

Each successful plant returns a dict with perturbed note, planted_span, and
span metadata. Failures raise SkipPlant so the runner can record coverage.
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any

from .sections import (
    NoteDoc,
    find_in_sections,
    insert_at,
    move_span,
    parse_note,
    replace_once,
    section_insert_pos,
    split_units,
)

BRAND_DRUGS = {
    "Tylenol": "tylenol",
    "Advil": "advil",
    "Motrin": "motrin",
    "Protonix": "protonix",
    "Lantus": "lantus",
    "Prozac": "prozac",
    "Zoloft": "zoloft",
    "Lexapro": "lexapro",
    "Synthroid": "synthroid",
    "Lipitor": "lipitor",
    "Crestor": "crestor",
    "Nexium": "nexium",
    "Eliquis": "eliquis",
    "Xarelto": "xarelto",
    "Coumadin": "coumadin",
    "Farxiga": "farxiga",
    "Jardiance": "jardiance",
    "Ozempic": "ozempic",
    "Mounjaro": "mounjaro",
    "Wegovy": "wegovy",
    "Robitussin": "robitussin",
    "Ultram": "ultram",
    "Lortab": "lortab",
    "Lasix": "lasix",
    "Imitrex": "imitrex",
    "Wellbutrin": "wellbutrin",
    "Cymbalta": "cymbalta",
    "Benadryl": "benadryl",
    "Zofran": "zofran",
    "Adderall": "adderall",
    "Flexeril": "flexeril",
    "Flomax": "flomax",
    "Neurontin": "neurontin",
    "Glucophage": "glucophage",
    "Norvasc": "norvasc",
    "Cozaar": "cozaar",
    "Toprol": "toprol",
    "Coreg": "coreg",
    "Plavix": "plavix",
    "Prilosec": "prilosec",
    "Pepcid": "pepcid",
    "Spiriva": "spiriva",
    "Ventolin": "ventolin",
    "ProAir": "proair",
    "Singulair": "singulair",
    "Flonase": "flonase",
    "Ambien": "ambien",
    "Xanax": "xanax",
    "Ativan": "ativan",
    "Klonopin": "klonopin",
    "Percocet": "percocet",
    "Vicodin": "vicodin",
    "Colace": "colace",
    "Miralax": "miralax",
    "Cipro": "cipro",
    "Levaquin": "levaquin",
    "Keflex": "keflex",
    "Augmentin": "augmentin",
    "Bactrim": "bactrim",
    "Diflucan": "diflucan",
    "Naprosyn": "naprosyn",
    "Mobic": "mobic",
    "Celebrex": "celebrex",
    "Phenergan": "phenergan",
    "Zofran": "zofran",
    "Imitrex": "imitrex",
    "Humira": "humira",
    "Enbrel": "enbrel",
}


class SkipPlant(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _clone_enc(enc: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(enc)


_PLAN_RE = re.compile(
    r"(?i)\b(we will|i will|i recommend|will increase|will continue|will order|"
    r"will refer|will obtain|will repeat|will start|will initiate|prescribe|"
    r"prescribed|increase (his|her|the)|refer(?:red)? to|order a |refill|"
    r"mg (?:once |twice )?daily|follow up|follow-up)\b"
)
_DIAGNOSIS_RE = re.compile(
    r"(?i)\b(viral syndrome|diagnosis|i believe|consistent with|likely|"
    r"poorly controlled|well controlled|assessment)\b"
)
_YEAR_OLD_SEX_RE = re.compile(
    r"(?i)\b(\d{1,3})[\s-]*year[\s-]*old[\s-]+(male|female|man|woman|gentleman|lady)\b"
)
_SEX_SWAP = {
    "male": "female",
    "female": "male",
    "man": "woman",
    "woman": "man",
    "gentleman": "lady",
    "lady": "gentleman",
}
_FIRST_PERSON_PLANT = (
    "I will personally telephone the patient tonight to review tonight's plan."
)
_HPI_NARRATIVE_RE = re.compile(
    r"(?i)\b(reports?|denies|endorses|states|started to|over the weekend|"
    r"for the past|she has been|he has been|patient (reports|denies|states))\b"
)
_MODIFIER_RE = re.compile(
    r"\b(left|right|bilateral|unilateral|upper|lower|distal|proximal|"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|units?|weeks?|days?|months?|hours?)|"
    r"once daily|twice daily|once weekly|every six hours)\b",
    re.I,
)


def _append_to_section(note: str, doc: NoteDoc, key: str, span: str) -> str:
    pos = section_insert_pos(doc, key)
    payload = "\n" + span.strip() + "\n"
    return insert_at(note, pos, payload)


def _pick_plan_unit(ap: str) -> str:
    units = split_units(ap)
    scored: list[tuple[int, str]] = []
    for u in units:
        if len(u) < 25 or len(u) > 280:
            continue
        if u.lower().startswith("patient agreements"):
            continue
        if re.match(r"^[A-Z][a-z].{8,60} is a \d", u):
            continue  # opening demographics
        score = 0
        if _PLAN_RE.search(u):
            score += 2
        if _DIAGNOSIS_RE.search(u):
            score += 1
        if u.startswith("•") and "Treatment" in "".join(units[max(0, units.index(u) - 2) : units.index(u) + 1]):
            score += 1
        if score:
            scored.append((score, u))
    if not scored:
        # fallback: a Medical Treatment bullet
        for u in units:
            if "mg" in u.lower() and len(u) > 20:
                return u
        raise SkipPlant("no plan/diagnosis sentence in A/P")
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    return scored[0][1]


def _pick_hpi_narrative(hpi: str) -> str:
    units = split_units(hpi)
    for u in units[1:]:  # skip opening identifier
        if len(u) < 40 or len(u) > 320:
            continue
        if _HPI_NARRATIVE_RE.search(u):
            return u
    raise SkipPlant("no HPI narrative sentence")


def plant_p1_ap_into_hpi(enc: dict[str, Any]) -> dict[str, Any]:
    """Move an A/P plan sentence into HPI (tokens preserved, section wrong)."""
    note = enc.get("note") or ""
    doc = parse_note(note)
    if not doc.has("hpi") or not doc.has("ap"):
        raise SkipPlant("missing HPI or A/P")
    unit = _pick_plan_unit(doc.ap_body())
    try:
        src = find_in_sections(doc, ("ap", "assessment", "plan"), unit)
    except ValueError as e:
        raise SkipPlant(str(e)) from e
    dest = section_insert_pos(doc, "hpi")
    new_note = move_span(note, src[0], src[1], dest)
    return {
        "note": new_note,
        "planted_span": unit,
        "source_section": "ap",
        "dest_section": "hpi",
        "operation": "move",
    }


def plant_p2_hpi_into_ap(enc: dict[str, Any]) -> dict[str, Any]:
    note = enc.get("note") or ""
    doc = parse_note(note)
    if not doc.has("hpi") or not doc.has("ap"):
        raise SkipPlant("missing HPI or A/P")
    unit = _pick_hpi_narrative(doc.body("hpi"))
    try:
        src = find_in_sections(doc, ("hpi",), unit)
    except ValueError as e:
        raise SkipPlant(str(e)) from e
    dest_key = "ap" if any(s.key == "ap" for s in doc.sections) else (
        "plan" if any(s.key == "plan" for s in doc.sections) else "assessment"
    )
    dest = section_insert_pos(doc, dest_key)
    new_note = move_span(note, src[0], src[1], dest)
    return {
        "note": new_note,
        "planted_span": unit,
        "source_section": "hpi",
        "dest_section": "ap",
        "operation": "move",
    }


_SYMPTOM_PLANT = (
    "The patient reports dull, aching pain that is worse in the evening and "
    "improves somewhat with rest."
)


def plant_p3_symptom_into_pe(enc: dict[str, Any]) -> dict[str, Any]:
    note = enc.get("note") or ""
    doc = parse_note(note)
    if not doc.has("pe"):
        raise SkipPlant("missing PE")
    span = _SYMPTOM_PLANT
    if span.lower() in note.lower():
        raise SkipPlant("symptom plant already present")
    return {
        "note": _append_to_section(note, doc, "pe", span),
        "planted_span": span,
        "source_section": None,
        "dest_section": "pe",
        "operation": "insert",
    }


def plant_p4_imaging_into_pe(enc: dict[str, Any]) -> dict[str, Any]:
    """Copy an existing RESULTS sentence into PE (tokens preserved, section wrong)."""
    note = enc.get("note") or ""
    doc = parse_note(note)
    if not doc.has("pe"):
        raise SkipPlant("missing PE")
    results = doc.body("results")
    if not results:
        raise SkipPlant("missing RESULTS")
    unit = None
    for u in split_units(results):
        if len(u) < 20 or len(u) > 280:
            continue
        if re.search(r"(?i)\b(x-ray|xr |ct |mri |ultrasound|echo|ekg|ecg|radiograph|monitor|a1c|hemoglobin|panel|unremarkable|normal|elevated)\b", u):
            unit = u
            break
    if unit is None:
        for u in split_units(results):
            if len(u) >= 20:
                unit = u
                break
    if not unit:
        raise SkipPlant("no RESULTS sentence to copy")
    if unit in doc.body("pe"):
        raise SkipPlant("results sentence already in PE")
    return {
        "note": _append_to_section(note, doc, "pe", unit),
        "planted_span": unit,
        "source_section": "results",
        "dest_section": "pe",
        "operation": "copy",
    }


_FABRICATED_NORMALS = [
    "Cranial nerves II-XII are intact and the neurologic exam is nonfocal.",
    "Abdomen is soft, nontender, and nondistended with no rebound or guarding.",
    "Skin is warm and dry with no rash or lesions.",
]


def plant_p5_fabricated_exam(enc: dict[str, Any]) -> dict[str, Any]:
    """Insert a normal exam finding that is not supported by the transcript."""
    note = enc.get("note") or ""
    doc = parse_note(note)
    if not doc.has("pe"):
        raise SkipPlant("missing PE")
    transcript = (enc.get("transcript") or "").lower()
    note_l = note.lower()
    span = None
    for cand in _FABRICATED_NORMALS:
        blob = " ".join(cand.lower().split()[0:6])
        if blob in note_l:
            continue
        anchors = re.findall(r"\b[a-z]{4,}\b", cand.lower())
        spoken = sum(1 for a in anchors if a in transcript)
        if spoken >= 4:
            continue
        span = cand
        break
    if not span:
        raise SkipPlant("no unused fabricated-normal candidate")
    return {
        "note": _append_to_section(note, doc, "pe", span),
        "planted_span": span,
        "source_section": None,
        "dest_section": "pe",
        "operation": "insert",
        "unsupported_in_transcript": True,
    }


_GENERIC_MODS = frozenset(
    {
        "left",
        "right",
        "bilateral",
        "unilateral",
        "upper",
        "lower",
        "distal",
        "proximal",
    }
)
_BARE_DOSE_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:mg|mcg|units?|weeks?|days?|months?|hours?)$", re.I
)
_GROW_STOP = frozenset(
    {
        "patient",
        "clinician",
        "reports",
        "report",
        "states",
        "notes",
        "prescribes",
        "prescribed",
        "recommends",
        "recommended",
        "instructs",
        "increases",
        "increased",
        "currently",
        "plans",
        "taking",
        "takes",
        "take",
        "that",
        "this",
        "from",
        "been",
        "have",
        "has",
        "was",
        "were",
    }
)


def _flex_pat(span: str) -> str:
    return r"\s+".join(re.escape(t) for t in span.split())


def _flex_count(text: str, span: str) -> int:
    if not span.strip():
        return 0
    return len(re.findall(_flex_pat(span), text or "", flags=re.I))


def _flex_delete_all(text: str, span: str) -> str:
    out = re.sub(_flex_pat(span), "", text or "", flags=re.I)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r" \n", "\n", out)


def _p6_span_candidates(fact: str, mod: str, note: str) -> list[str]:
    """Longest fact-substrings that contain the modifier and occur in the note."""
    m = re.search(re.escape(mod), fact, re.I)
    if not m:
        return []
    left_words = re.findall(r"\S+", fact[: m.start()])
    right_words = re.findall(r"\S+", fact[m.end() :])
    found: set[str] = set()
    for i in range(min(5, len(left_words)) + 1):
        for j in range(min(4, len(right_words)) + 1):
            lw = left_words[len(left_words) - i :] if i else []
            if lw and lw[0].lower().strip(".,;:") in _GROW_STOP:
                continue
            rw = right_words[:j]
            chunk = " ".join([*lw, fact[m.start() : m.end()], *rw]).strip(" .,;:")
            if 3 <= len(chunk) <= 80 and _flex_count(note, chunk) > 0:
                found.add(chunk)
    if _flex_count(note, mod) > 0:
        found.add(mod)
    return sorted(found, key=lambda s: (len(s.split()), len(s)), reverse=True)


def _p6_accept_span(span: str, note: str) -> bool:
    n = _flex_count(note, span)
    if n < 1:
        return False
    low = span.strip().lower()
    if low in _GENERIC_MODS:
        return False
    if _BARE_DOSE_RE.match(span) and n > 1:
        return False
    if len(span.split()) == 1 and not re.search(r"\d", span):
        return False
    return True


def plant_p6_drop_modifier(enc: dict[str, Any]) -> dict[str, Any]:
    """Delete a fact-specific dose/laterality span, every copy.

    First-occurrence deletes leave a second '20 mg' in A/P, so the omission
    judge correctly says not_omitted. Expand to drug+dose or laterality+anatomy
    and remove all copies of that phrase.
    """
    note = enc.get("note") or ""
    facts = enc.get("case_facts") or []
    ranked: list[tuple[int, int, str, str]] = []
    for fact in facts:
        for m in _MODIFIER_RE.finditer(fact):
            mod = m.group(0)
            if len(mod) < 3:
                continue
            for span in _p6_span_candidates(fact, mod, note):
                if not _p6_accept_span(span, note):
                    continue
                n = _flex_count(note, span)
                words = span.lower().split()
                specificity = len(words)
                score = specificity * 10
                if re.search(r"\d+\s*(?:mg|mcg|units?)\b", span, re.I):
                    score += 25
                if re.search(r"\b(left|right|bilateral)\b.+\b", span, re.I) and len(words) >= 2:
                    score += 12
                if words and words[0] in {"to", "of", "the", "a", "an", "with", "in", "on", "at"}:
                    score -= 20
                if re.search(r"\bevery\s+\d", span, re.I) and not re.search(
                    r"\d+\s*(?:mg|mcg|units?)\b", span, re.I
                ):
                    score -= 30
                if n == 1:
                    score += 4
                ranked.append((score, n, span, fact))
    if not ranked:
        raise SkipPlant("no unique fact-specific modifier span")
    ranked.sort(key=lambda x: -x[0])
    _, n_before, span, fact = ranked[0]
    new_note = _flex_delete_all(note, span)
    n_after = _flex_count(new_note, span)
    if n_after != 0:
        raise SkipPlant("span still present after delete")
    if new_note.strip() == note.strip():
        raise SkipPlant("delete did not change note")
    return {
        "note": new_note,
        "planted_span": span,
        "source_section": None,
        "dest_section": None,
        "operation": "delete_modifier",
        "case_fact": fact,
        "modifier_count_before": n_before,
        "modifier_count_after": n_after,
    }


_SKIP_LABELS = {
    "patient agreements",
    "follow up",
    "follow-up",
    "medical reasoning",
    "additional testing",
    "medical treatment",
    "specialist referrals",
    "patient education and counseling",
    "patient education",
}

_PROBLEM_HEADER_RE = re.compile(
    r"^(?:#+\s*|\d+[\.\)]\s*)?(?P<label>[A-Z][A-Za-z0-9 \-/]{2,60})\s*[.:]?\s*$",
    re.M,
)


def _ap_problem_labels(ap: str) -> list[str]:
    labels: list[str] = []
    for m in _PROBLEM_HEADER_RE.finditer(ap):
        lab = m.group("label").strip().rstrip(".")
        if lab.lower() in _SKIP_LABELS:
            continue
        if len(lab.split()) > 8:
            continue
        if re.match(r"(?i)^(the patient|he is|she is|this is)\b", lab):
            continue
        labels.append(lab)
    # unique, preserve order
    seen = set()
    out = []
    for lab in labels:
        k = lab.lower()
        if k not in seen:
            seen.add(k)
            out.append(lab)
    return out


def plant_p7_history_of_active(enc: dict[str, Any]) -> dict[str, Any]:
    """Rewrite an active A/P problem as 'history of X'."""
    note = enc.get("note") or ""
    doc = parse_note(note)
    ap = doc.ap_body()
    if not ap:
        raise SkipPlant("missing A/P")
    labels = _ap_problem_labels(ap)
    if not labels:
        raise SkipPlant("no A/P problem label")
    label = labels[0]
    ap_secs = doc.sections_for("ap")
    if not ap_secs:
        raise SkipPlant("missing A/P")
    region_start = min(s.body_start for s in ap_secs)
    region_end = max(s.body_end for s in ap_secs)
    region = note[region_start:region_end]
    heading = re.search(
        r"(?m)^(?:#+\s*|\d+[\.\)]\s*)?" + re.escape(label) + r"\s*[.:]?\s*$",
        region,
    )
    if heading:
        abs_start = region_start + heading.start()
        abs_end = region_start + heading.end()
        new_note = note[:abs_start] + f"History of {label}." + note[abs_end:]
    else:
        i = region.lower().find(label.lower())
        if i < 0:
            raise SkipPlant("could not rewrite problem as history of")
        abs_i = region_start + i
        new_note = note[:abs_i] + f"history of {label}" + note[abs_i + len(label) :]
    return {
        "note": new_note,
        "planted_span": f"History of {label}",
        "source_section": "ap",
        "dest_section": "ap",
        "operation": "rewrite",
        "active_problem": label,
    }


def plant_p8_lowercase_drugs(enc: dict[str, Any]) -> dict[str, Any]:
    note = enc.get("note") or ""
    hits: list[str] = []
    new = note
    for proper, lower in BRAND_DRUGS.items():
        if re.search(r"\b" + re.escape(proper) + r"\b", new):
            new, n = re.subn(r"\b" + re.escape(proper) + r"\b", lower, new)
            if n:
                hits.append(proper)
    if not hits:
        raise SkipPlant("no capitalized brand drug to lowercase")
    return {
        "note": new,
        "planted_span": ", ".join(hits),
        "source_section": None,
        "dest_section": None,
        "operation": "lowercase",
        "drugs": hits,
    }


def plant_p9_bundle_diagnoses(enc: dict[str, Any]) -> dict[str, Any]:
    note = enc.get("note") or ""
    doc = parse_note(note)
    labels = _ap_problem_labels(doc.ap_body())
    if len(labels) < 2:
        raise SkipPlant("need two A/P problem labels")
    a, b = labels[0], labels[1]
    bundled = f"{a} and {b}."
    ap_secs = doc.sections_for("ap")
    region_start = min(s.body_start for s in ap_secs)
    region_end = max(s.body_end for s in ap_secs)
    region = note[region_start:region_end]

    def find_heading(label: str, text: str):
        return re.search(
            r"(?m)^(?:#+\s*|\d+[\.\)]\s*)?" + re.escape(label) + r"\s*[.:]?\s*$",
            text,
        )

    ha = find_heading(a, region)
    hb = find_heading(b, region)
    if not ha or not hb:
        raise SkipPlant("could not locate both problem headings")
    # Work in absolute offsets; edit later span first.
    a0, a1 = region_start + ha.start(), region_start + ha.end()
    b0, b1 = region_start + hb.start(), region_start + hb.end()
    if a0 < b0:
        note = note[:b0] + note[b1:]
        region_end -= b1 - b0
        region = note[region_start:region_end]
        ha = find_heading(a, region)
        if not ha:
            raise SkipPlant("lost first heading after dropping second")
        new_note = (
            note[: region_start + ha.start()]
            + bundled
            + note[region_start + ha.end() :]
        )
    else:
        note = note[:a0] + note[a1:]
        region_end -= a1 - a0
        region = note[region_start:region_end]
        hb = find_heading(b, region)
        if not hb:
            raise SkipPlant("lost second heading after dropping first")
        new_note = (
            note[: region_start + hb.start()]
            + bundled
            + note[region_start + hb.end() :]
        )
    return {
        "note": new_note,
        "planted_span": bundled,
        "source_section": "ap",
        "dest_section": "ap",
        "operation": "merge_headings",
        "labels": [a, b],
    }


def plant_p10_narrative_cc(enc: dict[str, Any]) -> dict[str, Any]:
    note = enc.get("note") or ""
    doc = parse_note(note)
    cc_secs = doc.sections_for("cc")
    if not cc_secs:
        raise SkipPlant("missing CC")
    cc = cc_secs[0].body.strip()
    if not cc:
        raise SkipPlant("missing CC")
    if len(cc.split()) >= 12:
        raise SkipPlant("CC already narrative")
    age = enc.get("patient_age") or "adult"
    sex = {"M": "male", "F": "female"}.get(str(enc.get("patient_sex") or "").upper(), "patient")
    label = re.sub(r"[.\n]+$", "", cc.split("\n")[0]).strip()
    span = (
        f"The patient is a {age}-year-old {sex} who presents today with {label.lower()} "
        f"and would like this issue addressed in detail during the current visit."
    )
    sec = cc_secs[0]
    new_note = note[: sec.body_start] + span + "\n" + note[sec.body_end :]
    return {
        "note": new_note,
        "planted_span": span,
        "source_section": "cc",
        "dest_section": "cc",
        "operation": "rewrite",
        "original_cc": cc,
    }


def plant_p11_swap_sex(enc: dict[str, Any]) -> dict[str, Any]:
    """Swap documented sex in N-year-old lines; chart metadata stays put."""
    note = enc.get("note") or ""
    matches = list(_YEAR_OLD_SEX_RE.finditer(note))
    if not matches:
        raise SkipPlant("no N-year-old sex phrase")
    chart = str(enc.get("patient_sex") or "").strip().upper()
    chart_word = {"M": "male", "F": "female"}.get(chart)
    first_sex = matches[0].group(2).lower()
    if first_sex in {"man", "gentleman"}:
        first_sex = "male"
    elif first_sex in {"woman", "lady"}:
        first_sex = "female"
    # Gold already disagrees with the chart: swapping would *fix* the note.
    if chart_word and first_sex != chart_word:
        raise SkipPlant("documented sex already mismatches chart")
    replacements: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        age, sex = m.group(1), m.group(2)
        swapped = _SEX_SWAP[sex.lower()]
        if sex[0].isupper():
            swapped = swapped.capitalize()
        repl = f"{age}-year-old {swapped}"
        replacements.append(repl)
        return repl

    new = _YEAR_OLD_SEX_RE.sub(_sub, note)
    if new == note or not replacements:
        raise SkipPlant("sex swap did not change the note")
    return {
        "note": new,
        "planted_span": replacements[0],
        "source_section": "hpi",
        "dest_section": "hpi",
        "operation": "swap_sex",
        "n_swaps": len(replacements),
        "swapped_spans": replacements,
    }


def plant_p12_first_person_i(enc: dict[str, Any]) -> dict[str, Any]:
    """Insert a distinctive clinician-I sentence as the first body line.

    Placed immediately after the first section header so the production
    first-person finder (first match in section order) hits the plant, not
    a pre-existing A/P “I recommend”.
    """
    note = enc.get("note") or ""
    span = _FIRST_PERSON_PLANT
    if span.lower() in note.lower():
        raise SkipPlant("first-person plant already present")
    header = re.search(
        r"(?m)^([A-Z][A-Z0-9 /&\-]{1,60})\s*:?\s*$",
        note,
    )
    if not header:
        raise SkipPlant("no section header")
    insert_at = header.end()
    rest = note[insert_at:].lstrip("\n")
    new_note = note[:insert_at] + "\n\n" + span + "\n" + rest
    return {
        "note": new_note,
        "planted_span": span,
        "source_section": None,
        "dest_section": "cc",
        "operation": "insert",
    }


def plant_control_delete_15pct(enc: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Generic corruption: drop ~15% of whitespace-delimited tokens. ROUGE *should* move."""
    note = enc.get("note") or ""
    tokens = note.split(" ")
    if len(tokens) < 40:
        raise SkipPlant("note too short")
    n_drop = max(1, int(round(len(tokens) * 0.15)))
    idx = set(rng.sample(range(len(tokens)), n_drop))
    new = " ".join(t for i, t in enumerate(tokens) if i not in idx)
    out = _clone_enc(enc)
    out["note"] = new
    return {
        "note": new,
        "planted_span": f"deleted_{n_drop}_tokens",
        "source_section": None,
        "dest_section": None,
        "operation": "random_delete",
        "n_deleted": n_drop,
    }


PLANT_SPECS: list[dict[str, Any]] = [
    {
        "id": "P1",
        "name": "ap_into_hpi",
        "title": "Move an A/P sentence into HPI",
        "target_evaluator": "A&P→HPI leakage",
        "why_reference_blind": "Tokens still present; only section membership changes",
        "fn": plant_p1_ap_into_hpi,
    },
    {
        "id": "P2",
        "name": "hpi_into_ap",
        "title": "Move an HPI narrative sentence into A/P",
        "target_evaluator": "HPI→A/P leakage",
        "why_reference_blind": "Tokens still present",
        "fn": plant_p2_hpi_into_ap,
    },
    {
        "id": "P3",
        "name": "symptom_into_pe",
        "title": "Insert a symptom-report sentence into PE",
        "target_evaluator": "subjective→PE leakage",
        "why_reference_blind": "Small n-gram change",
        "fn": plant_p3_symptom_into_pe,
    },
    {
        "id": "P4",
        "name": "imaging_into_pe",
        "title": "Copy a RESULTS sentence into PE",
        "target_evaluator": "results→PE leakage",
        "why_reference_blind": "Tokens duplicated; concept set unchanged",
        "fn": plant_p4_imaging_into_pe,
    },
    {
        "id": "P5",
        "name": "fabricated_exam",
        "title": "Insert a fabricated normal exam finding",
        "target_evaluator": "hallucination",
        "why_reference_blind": "Adds plausible clinical tokens; MedCon may increase",
        "fn": plant_p5_fabricated_exam,
    },
    {
        "id": "P6",
        "name": "drop_modifier",
        "title": "Delete a case-fact modifier (laterality/dose/time)",
        "target_evaluator": "omission",
        "why_reference_blind": "Remaining lexical overlap high",
        "fn": plant_p6_drop_modifier,
    },
    {
        "id": "P7",
        "name": "history_of_active",
        "title": "Rewrite an active A/P problem as history of X",
        "target_evaluator": "history-of misuse",
        "why_reference_blind": "Near-paraphrase",
        "fn": plant_p7_history_of_active,
    },
    {
        "id": "P8",
        "name": "lowercase_drugs",
        "title": "Lowercase brand drug names",
        "target_evaluator": "deterministic style",
        "why_reference_blind": "Case change is invisible to MedCon and nearly to ROUGE-L",
        "fn": plant_p8_lowercase_drugs,
    },
    {
        "id": "P9",
        "name": "bundle_diagnoses",
        "title": "Merge two A/P problem labels onto one line",
        "target_evaluator": "diagnosis structure",
        "why_reference_blind": "Concept set unchanged",
        "fn": plant_p9_bundle_diagnoses,
    },
    {
        "id": "P10",
        "name": "narrative_cc",
        "title": "Rewrite CC as a narrative sentence",
        "target_evaluator": "CC conciseness",
        "why_reference_blind": "Adds tokens; may raise ROUGE vs a longer reference but here vs original CC it moves little globally",
        "fn": plant_p10_narrative_cc,
    },
    {
        "id": "P11",
        "name": "swap_sex",
        "title": "Swap documented sex in N-year-old lines",
        "target_evaluator": "demographics",
        "why_reference_blind": "One-token sex swap; concept F1 unchanged",
        "fn": plant_p11_swap_sex,
    },
    {
        "id": "P12",
        "name": "first_person_i",
        "title": "Insert a clinician first-person I sentence",
        "target_evaluator": "provider voice (first_person_i)",
        "why_reference_blind": "Adds one short sentence; gold already has many I’s",
        "fn": plant_p12_first_person_i,
    },
    {
        "id": "CTRL",
        "name": "random_delete_15pct",
        "title": "Control: delete 15% of tokens at random",
        "target_evaluator": "(none — generic corruption)",
        "why_reference_blind": "Negative control: ROUGE SHOULD drop",
        "fn": None,  # special-cased (needs rng)
        "is_control": True,
    },
]
