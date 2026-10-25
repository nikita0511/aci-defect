"""Section parser for ACI-bench gold notes (mixed virtassist / aci / virtscribe templates).

Edits are surgical on the original string so uninvolved tokens (and therefore
ROUGE) do not move except at the planted span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADER_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^ASSESSMENT AND PLAN\s*:?\s*$", re.I), "ap"),
    (re.compile(r"^ASSESSMENT\s*&\s*PLAN\s*:?\s*$", re.I), "ap"),
    (re.compile(r"^ASSESSMENT\s*/\s*PLAN\s*:?\s*$", re.I), "ap"),
    (re.compile(r"^A\s*/\s*P\s*:?\s*$", re.I), "ap"),
    (re.compile(r"^HISTORY OF PRESENT ILLNESS\s*:?\s*$", re.I), "hpi"),
    (re.compile(r"^INTERVAL HISTORY\s*:?\s*$", re.I), "hpi"),
    (re.compile(r"^PHYSICAL EXAMINATION\s*:?\s*$", re.I), "pe"),
    (re.compile(r"^PHYSICAL EXAM\s*:?\s*$", re.I), "pe"),
    (re.compile(r"^CHIEF COMPLAINT\s*:?\s*$", re.I), "cc"),
    (re.compile(r"^REVIEW OF SYSTEMS\s*:?\s*$", re.I), "ros"),
    (re.compile(r"^RESULTS\s*:?\s*$", re.I), "results"),
    (re.compile(r"^LABORATORY\s*:?\s*$", re.I), "results"),
    (re.compile(r"^IMAGING\s*:?\s*$", re.I), "results"),
    (re.compile(r"^ASSESSMENT\s*:?\s*$", re.I), "assessment"),
    (re.compile(r"^IMPRESSION\s*:?\s*$", re.I), "assessment"),
    (re.compile(r"^PLAN\s*:?\s*$", re.I), "plan"),
    (re.compile(r"^INSTRUCTIONS\s*:?\s*$", re.I), "plan"),
    (re.compile(r"^EXAM\s*:?\s*$", re.I), "pe"),
    (re.compile(r"^SUBJECTIVE\s*:?\s*$", re.I), "hpi"),
    (re.compile(r"^OBJECTIVE\s*:?\s*$", re.I), "pe"),
    (re.compile(r"^VITALS REVIEWED\s*:?\s*$", re.I), "vitals"),
    (re.compile(r"^VITALS\s*:?\s*$", re.I), "vitals"),
    (re.compile(r"^CURRENT MEDICATIONS\s*:?\s*$", re.I), "meds"),
    (re.compile(r"^MEDICATIONS\s*:?\s*$", re.I), "meds"),
]


@dataclass
class Section:
    header: str
    key: str
    body: str
    header_start: int = -1
    body_start: int = -1
    body_end: int = -1


@dataclass
class NoteDoc:
    text: str = ""
    sections: list[Section] = field(default_factory=list)

    def body(self, key: str) -> str:
        return "\n\n".join(s.body for s in self.sections if s.key == key).strip()

    def ap_body(self) -> str:
        ap = self.body("ap")
        if ap:
            return ap
        return "\n\n".join(
            s.body for s in self.sections if s.key in ("assessment", "plan")
        ).strip()

    def has(self, key: str) -> bool:
        if key == "ap":
            return bool(self.ap_body())
        return any(s.key == key and s.body.strip() for s in self.sections)

    def sections_for(self, key: str) -> list[Section]:
        if key == "ap":
            ap = [s for s in self.sections if s.key == "ap"]
            if ap:
                return ap
            return [s for s in self.sections if s.key in ("assessment", "plan")]
        return [s for s in self.sections if s.key == key]


def parse_note(note: str) -> NoteDoc:
    text = note or ""
    lines = text.splitlines(keepends=True)
    sections: list[Section] = []
    current_key = "_preamble"
    current_header = ""
    header_start = 0
    body_start = 0
    buf: list[str] = []
    pos = 0

    def flush(end: int) -> None:
        nonlocal current_key, current_header, header_start, body_start, buf
        body = "".join(buf)
        sections.append(
            Section(
                header=current_header,
                key=current_key,
                body=body.strip("\n"),
                header_start=header_start,
                body_start=body_start,
                body_end=end,
            )
        )
        buf = []

    for line in lines:
        stripped = line.strip()
        matched_key = None
        if stripped and len(stripped) < 80:
            for pat, key in _HEADER_RULES:
                if pat.match(stripped):
                    matched_key = key
                    break
        if matched_key is not None:
            flush(pos)
            current_key = matched_key
            current_header = stripped
            header_start = pos
            body_start = pos + len(line)
            pos += len(line)
            continue
        buf.append(line)
        pos += len(line)
    flush(len(text))
    if sections and sections[0].key == "_preamble" and not sections[0].body.strip():
        sections = sections[1:]
    return NoteDoc(text=text, sections=sections)


def split_units(text: str) -> list[str]:
    units: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("•", "-", "*")) or re.match(r"^[A-Za-z].{0,40}:\s", line):
            units.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•])", line)
        units.extend(p.strip() for p in parts if p.strip())
    return units


def find_in_sections(doc: NoteDoc, keys: tuple[str, ...], unit: str) -> tuple[int, int]:
    for key in keys:
        for sec in doc.sections_for(key):
            chunk = doc.text[sec.body_start : sec.body_end]
            i = chunk.find(unit)
            if i >= 0:
                return sec.body_start + i, sec.body_start + i + len(unit)
    raise ValueError(f"unit not found in {keys}: {unit[:80]!r}")


def section_insert_pos(doc: NoteDoc, key: str) -> int:
    secs = doc.sections_for(key)
    if not secs:
        raise ValueError(f"no section {key}")
    return secs[-1].body_end


def delete_span(text: str, start: int, end: int) -> str:
    return text[:start] + text[end:]


def insert_at(text: str, pos: int, snippet: str) -> str:
    return text[:pos] + snippet + text[pos:]


def move_span(text: str, src_start: int, src_end: int, dest_pos: int, dest_prefix: str = "\n\n") -> str:
    chunk = text[src_start:src_end].strip("\n")
    payload = dest_prefix + chunk + "\n"
    if dest_pos > src_end:
        text = delete_span(text, src_start, src_end)
        dest_pos -= src_end - src_start
        return insert_at(text, dest_pos, payload)
    if dest_pos <= src_start:
        text = insert_at(text, dest_pos, payload)
        src_start += len(payload)
        src_end += len(payload)
        return delete_span(text, src_start, src_end)
    return text


def replace_once(text: str, old: str, new: str) -> str:
    i = text.find(old)
    if i < 0:
        raise ValueError("replace target missing")
    return text[:i] + new + text[i + len(old) :]
