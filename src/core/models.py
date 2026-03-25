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


@dataclass(slots=True)
class VerificationResult:
    answer: str
    consistency_score: float
    paths: List[str] = field(default_factory=list)
    verdict: str = "unknown"
    suggestions: List[str] = field(default_factory=list)


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
    last_trace_id: str = ""


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    source_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
