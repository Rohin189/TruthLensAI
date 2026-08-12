"""Clean raw extracted PDF text for chunking/embedding."""
import re

LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}


def fix_ligatures(text: str) -> str:
    for lig, replacement in LIGATURE_MAP.items():
        text = text.replace(lig, replacement)
    return text



def clean_text(text: str) -> str:
    text = fix_ligatures(text)
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}v?\d*\s*\[.*?\]\s*\d{1,2}\s\w+\s\d{4}", "", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = text.split("\n")
    cleaned_lines = [
        ln for ln in lines
        if len(ln.strip()) == 0 or sum(c.isalnum() for c in ln) / max(len(ln), 1) > 0.3
    ]
    return "\n".join(cleaned_lines).strip()

def truncate_references_section(text: str) -> str:
    match = re.search(r"\n\s*(references|bibliography)\s*\n", text, re.IGNORECASE)
    if match:
        return text[: match.start()]
    return text


def sanitize_text(text: str) -> str:
    """Remove invalid unicode surrogate characters that break UTF-8 encoding."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")
