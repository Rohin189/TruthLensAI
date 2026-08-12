"""Detect page boundaries and section headers within document text,
so chunks can be tagged with page_number and section for the Evidence Viewer."""
import re
from bisect import bisect_right

PAGE_MARKER_RE = re.compile(r"<<PAGE:(\d+)>>")

SECTION_HEADER_RE = re.compile(
    r"\n\s*(?:\d+\.?\s+)?"
    r"(Abstract|Introduction|Related Work|Background|Methodology|Method|"
    r"Approach|Experiments?|Results?|Discussion|Conclusion|Limitations|"
    r"Future Work|Ablation Study|Evaluation)\s*\n",
    re.IGNORECASE,
)


def build_marked_text(pages: list[dict]) -> str:
    """Concatenate page texts with inline page markers so we can later
    figure out which page any given character offset belongs to."""
    return "\n".join(f"<<PAGE:{p['page_number']}>>\n{p['text']}" for p in pages)


def get_page_offsets(text: str) -> list[tuple[int, int]]:
    """Returns [(char_offset, page_number), ...] sorted by offset."""
    return [(m.start(), int(m.group(1))) for m in PAGE_MARKER_RE.finditer(text)]


def get_section_offsets(text: str) -> list[tuple[int, str]]:
    """Returns [(char_offset, section_name), ...] sorted by offset."""
    return [(m.start(), m.group(1).title()) for m in SECTION_HEADER_RE.finditer(text)]


def _lookup(offset: int, offsets: list[tuple[int, "T"]], default: "T") -> "T":
    """Binary search: find the most recent (offset, value) at or before `offset`."""
    if not offsets:
        return default
    positions = [o for o, _ in offsets]
    idx = bisect_right(positions, offset) - 1
    if idx < 0:
        return default
    return offsets[idx][1]


def page_for_offset(offset: int, page_offsets: list[tuple[int, int]]) -> int:
    return _lookup(offset, page_offsets, default=-1)


def section_for_offset(offset: int, section_offsets: list[tuple[int, str]]) -> str:
    return _lookup(offset, section_offsets, default="Front Matter")


def strip_page_markers(text: str) -> str:
    return PAGE_MARKER_RE.sub("", text).strip()