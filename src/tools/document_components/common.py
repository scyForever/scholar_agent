from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any, Dict, List

from src.core.models import PaperAsset, ParsedDocument


SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*)?\.?\s*(abstract|introduction|background|related work|method|methods|methodology|approach|experiment|experiments|results|discussion|conclusion|limitations|appendix)\b",
    flags=re.IGNORECASE,
)
FORMULA_LINE_RE = re.compile(r"[=+\-*/^_\\]|(?:\([0-9]{1,3}\)\s*$)")
FIGURE_CAPTION_RE = re.compile(r"^(figure|fig\.)\s*\d+", flags=re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^table\s*\d+", flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text or "").replace("\r", "\n").splitlines()).strip()


def normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return cleaned.strip("-") or "document"


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def normalize_heading(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", str(text or "").strip()).strip().lower()


def looks_like_heading(text: str, font_size: float, median_font_size: float) -> bool:
    cleaned = normalize_inline_whitespace(text)
    if len(cleaned) > 140 or not cleaned:
        return False
    if SECTION_HEADING_RE.match(cleaned):
        return True
    if font_size >= median_font_size + 1.2 and len(cleaned.split()) <= 12:
        return True
    if cleaned.isupper() and len(cleaned.split()) <= 10:
        return True
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z][\w-]+", cleaned):
        return True
    return False


def serialize_document(document: ParsedDocument) -> Dict[str, Any]:
    return asdict(document)


def serialize_asset(asset: PaperAsset) -> Dict[str, Any]:
    return asdict(asset)
