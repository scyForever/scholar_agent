from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from src.core.models import Paper, ResearchPlan
from src.skills import ResearchSkillset
from src.whitebox.tracer import WhiteboxTracer


class ResearchPlannerAgent:
    def __init__(self, skills: ResearchSkillset, tracer: WhiteboxTracer) -> None:
        self.skills = skills
        self.tracer = tracer

    def run(self, query: str, intent: str, slots: Dict[str, Any], trace_id: str) -> ResearchPlan:
        plan = self.skills.planning.plan(query, intent=intent, slots=slots)
        self.tracer.trace_step(
            trace_id,
            "plan",
            {"query": query, "intent": intent, "slots": slots},
            asdict(plan),
        )
        return plan


class ResearchSearchAgent:
    def __init__(self, skills: ResearchSkillset, tracer: WhiteboxTracer) -> None:
        self.skills = skills
        self.tracer = tracer

    def run(
        self,
        query: str,
        *,
        time_range: str = "",
        max_results: int = 10,
        user_id: str = "",
        platforms: List[str] | None = None,
        author: str = "",
        trace_id: str = "",
    ) -> Dict[str, Any]:
        result = self.skills.search.search(
            query,
            platforms=platforms,
            max_results=max_results,
            time_range=time_range,
            author=author,
            user_id=user_id,
        )
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                "research_search",
                {
                    "query": query,
                    "time_range": time_range,
                    "max_results": max_results,
                    "platforms": platforms or [],
                },
                {
                    "count": len(result.get("papers") or []),
                    "platforms": result.get("platforms") or [],
                    "memory_trace": result.get("memory_trace") or {},
                },
            )
        return result


class ResearchReadingAgent:
    def __init__(self, skills: ResearchSkillset, tracer: WhiteboxTracer) -> None:
        self.skills = skills
        self.tracer = tracer

    def read_pdf(
        self,
        pdf_path: str,
        *,
        target_section: str = "",
        trace_id: str = "",
    ) -> Dict[str, Any]:
        document = self.skills.reading.parse_pdf(pdf_path, target_section=target_section)
        payload = asdict(document)
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                "deep_read",
                {"pdf_path": pdf_path, "target_section": target_section},
                {
                    "section_count": len(document.sections),
                    "chunk_count": len(document.chunks),
                    "table_count": len(document.tables),
                    "figure_count": len(document.figures),
                },
            )
        return payload

    def build_analysis_context(
        self,
        paper: Paper,
        *,
        pdf_path: str = "",
        focus: str = "",
        trace_id: str = "",
    ) -> Dict[str, Any]:
        if not pdf_path:
            return {
                "paper": asdict(paper),
                "sections": [],
                "chunks": [],
                "focus": focus,
            }
        matched = self.skills.reading.targeted_read(
            pdf_path,
            section_name=focus or "method",
            max_chunks=6,
        )
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                "deep_read_focus",
                {"paper_title": paper.title, "focus": focus or "method"},
                {
                    "matched_sections": len(matched.get("matched_sections") or []),
                    "chunk_count": len(matched.get("chunks") or []),
                },
            )
        return {
            "paper": asdict(paper),
            "focus": focus,
            "sections": matched.get("matched_sections") or [],
            "chunks": matched.get("chunks") or [],
            "abstract": matched.get("abstract") or paper.abstract,
        }


class ResearchMemoryAgent:
    def __init__(self, skills: ResearchSkillset, tracer: WhiteboxTracer) -> None:
        self.skills = skills
        self.tracer = tracer

    def remember_paper(
        self,
        user_id: str,
        paper: Paper,
        summary: str,
        *,
        highlights: List[str] | None = None,
        trace_id: str = "",
    ) -> str:
        memory_id = self.skills.memory.remember_paper(
            user_id,
            paper,
            summary,
            highlights=highlights,
        )
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                "research_memory_store",
                {"user_id": user_id, "paper_title": paper.title},
                {"memory_id": memory_id},
            )
        return memory_id

    def recall(self, user_id: str, query: str, *, trace_id: str = "") -> List[Dict[str, Any]]:
        memories = self.skills.memory.recall_context(user_id, query)
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                "research_memory_recall",
                {"user_id": user_id, "query": query},
                {"count": len(memories)},
            )
        return memories
