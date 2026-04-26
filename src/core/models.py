from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionMode(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    FULL = "full"


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    PREFERENCE = "preference"
    FEEDBACK = "feedback"
    PAPER_SUMMARY = "paper_summary"
    RESEARCH_NOTE = "research_note"


@dataclass(slots=True)
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    url: str = ""
    pdf_url: str = ""
    citations: int = 0
    source: str = ""
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    doi: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    pmcid: str = ""
    full_text_url: str = ""
    html_url: str = ""
    open_access: bool = False


@dataclass(slots=True)
class PaperSection:
    heading: str
    level: int = 1
    text: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisualElement:
    element_id: str
    kind: str
    page: int
    caption: str = ""
    text: str = ""
    latex: str = ""
    markdown: str = ""
    image_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    source_path: str
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    page_texts: List[str] = field(default_factory=list)
    sections: List[PaperSection] = field(default_factory=list)
    chunks: List[str] = field(default_factory=list)
    figures: List[VisualElement] = field(default_factory=list)
    tables: List[VisualElement] = field(default_factory=list)
    formulas: List[VisualElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperAsset:
    identifier: str
    source: str
    asset_type: str
    url: str = ""
    local_path: str = ""
    content: str = ""
    content_type: str = ""
    available: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchTask:
    task_id: str
    title: str
    goal: str
    deliverable: str = ""
    dependencies: List[str] = field(default_factory=list)
    suggested_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResearchPlan:
    topic: str
    objective: str
    tasks: List[ResearchTask] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    validation: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    query: str
    papers: List[Paper] = field(default_factory=list)
    total_found: int = 0
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    rewritten_queries: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperAnalysis:
    paper: Paper
    summary: str
    contributions: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    raw_analysis: str = ""


@dataclass(slots=True)
class DebateResult:
    question: str
    thesis: str
    supporting_points: List[str] = field(default_factory=list)
    counter_points: List[str] = field(default_factory=list)
    synthesis: str = ""


@dataclass(slots=True)
class ReviewDraft:
    title: str
    abstract: str
    outline: List[str]
    body: str
    references: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningResult:
    mode: str
    answer: str
    steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MoAResult:
    answer: str
    candidates: List[str] = field(default_factory=list)
    rationale: str = ""
    score: float = 0.0
    errors: List[str] = field(default_factory=list)


@dataclass(slots=True)
class VerificationResult:
    answer: str
    consistency_score: float
    paths: List[str] = field(default_factory=list)
    verdict: str = "unknown"
    suggestions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    content: str
    memory_type: MemoryType
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    score: float = 0.0


@dataclass(slots=True)
class ShortTermMemory:
    raw: List[Dict[str, str]] = field(default_factory=list)
    highlights: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class AgentResponse:
    answer: str
    intent: str
    slots: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    needs_input: bool = False
    whitebox: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DialogueState:
    intent: str = ""
    current_slots: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    short_memory: ShortTermMemory = field(default_factory=ShortTermMemory)
    last_trace_id: str = ""
    last_search_result: Optional[SearchResult] = None


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    source_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
