from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass(slots=True)
class ToolExecutionRequest:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiteratureSearchToolRequest:
    query: str
    platforms: Sequence[str] = field(default_factory=tuple)
    max_results: int = 10
    time_range: str = ""
    author: str = ""


@dataclass(slots=True)
class LiteratureSourceSearchRequest:
    source_name: str
    query: str
    max_results: int = 10
    time_range: str = ""
    author: str = ""


@dataclass(slots=True)
class PaperFetchToolRequest:
    identifier: str
    identifier_type: str = "auto"
    prefer: str = "pdf"
    download_dir: str = ""


@dataclass(slots=True)
class PDFParseToolRequest:
    pdf_path: str
    target_section: str = ""


@dataclass(slots=True)
class VisualExtractToolRequest:
    pdf_path: str
    page_numbers: List[int] | None = None
    output_dir: str = ""


@dataclass(slots=True)
class SectionReadToolRequest:
    pdf_path: str
    section_name: str
    max_chunks: int = 5


@dataclass(slots=True)
class WebSearchToolRequest:
    query: str
    max_results: int = 5
