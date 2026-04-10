from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.core.models import Paper


@dataclass(slots=True)
class PreferenceMemoryRequest:
    user_id: str
    preference: str
    metadata: Dict[str, Any] | None = None


@dataclass(slots=True)
class PaperMemoryRequest:
    user_id: str
    paper: Paper
    summary: str
    highlights: List[str] | None = None


@dataclass(slots=True)
class MemoryRecallRequest:
    user_id: str
    query: str
    limit: int = 8


@dataclass(slots=True)
class PaperRankingRequest:
    user_id: str
    papers: Sequence[Paper]
    limit: int


@dataclass(slots=True)
class SearchPreferenceRequest:
    user_id: str
    topic: str
    time_range: str = ""
    sources: Sequence[str] | None = None
    max_results: int | None = None


@dataclass(slots=True)
class LiteratureSkillSearchRequest:
    query: str
    platforms: Sequence[str] = field(default_factory=tuple)
    max_results: int = 10
    time_range: str = ""
    author: str = ""
    user_id: str = ""


@dataclass(slots=True)
class PaperFetchSkillRequest:
    identifier: str
    identifier_type: str = "auto"
    prefer: str = "pdf"
    download_dir: str = ""


@dataclass(slots=True)
class PDFReadSkillRequest:
    pdf_path: str
    target_section: str = ""


@dataclass(slots=True)
class TargetedReadSkillRequest:
    pdf_path: str
    section_name: str
    max_chunks: int = 5


@dataclass(slots=True)
class VisualExtractSkillRequest:
    pdf_path: str
    page_numbers: List[int] | None = None
    output_dir: str = ""


@dataclass(slots=True)
class ResearchPlanSkillRequest:
    topic: str
    intent: str = "generate_survey"
    slots: Dict[str, Any] | None = None
