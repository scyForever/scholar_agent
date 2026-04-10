from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.core.models import Paper
from src.memory.manager import MemoryManager
from src.skills.components import (
    DeepReadingComponent,
    LiteratureSearchComponent,
    ResearchMemoryComponent,
    ResearchPlanningComponent,
)
from src.skills.contracts import (
    LiteratureSkillSearchRequest,
    MemoryRecallRequest,
    PaperFetchSkillRequest,
    PaperMemoryRequest,
    PaperRankingRequest,
    PDFReadSkillRequest,
    PreferenceMemoryRequest,
    ResearchPlanSkillRequest,
    SearchPreferenceRequest,
    TargetedReadSkillRequest,
    VisualExtractSkillRequest,
)


class ResearchMemorySkillHarness(ResearchMemoryComponent):
    def remember_preference(
        self,
        request: PreferenceMemoryRequest | str,
        preference: str = "",
        *,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        if isinstance(request, PreferenceMemoryRequest):
            return super().remember_preference(
                request.user_id,
                request.preference,
                metadata=request.metadata,
            )
        return super().remember_preference(request, preference, metadata=metadata)

    def remember_paper(
        self,
        request: PaperMemoryRequest | str,
        paper: Paper | None = None,
        summary: str = "",
        *,
        highlights: List[str] | None = None,
    ) -> str:
        if isinstance(request, PaperMemoryRequest):
            return super().remember_paper(
                request.user_id,
                request.paper,
                request.summary,
                highlights=request.highlights,
            )
        if paper is None:
            raise ValueError("paper is required when using legacy remember_paper signature")
        return super().remember_paper(
            request,
            paper,
            summary,
            highlights=highlights,
        )

    def recall_context(
        self,
        request: MemoryRecallRequest | str,
        query: str = "",
        *,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        if isinstance(request, MemoryRecallRequest):
            return super().recall_context(
                request.user_id,
                request.query,
                limit=request.limit,
            )
        return super().recall_context(request, query, limit=limit)

    def rank_unseen_first(
        self,
        request: PaperRankingRequest | str,
        papers: Sequence[Paper] | None = None,
        *,
        limit: int = 10,
    ) -> Dict[str, Any]:
        if isinstance(request, PaperRankingRequest):
            return super().rank_unseen_first(
                request.user_id,
                request.papers,
                limit=request.limit,
            )
        return super().rank_unseen_first(request, papers or [], limit=limit)

    def remember_search_preferences(
        self,
        request: SearchPreferenceRequest | str,
        *,
        topic: str = "",
        time_range: str = "",
        sources: Sequence[str] | None = None,
        max_results: int | None = None,
    ) -> None:
        if isinstance(request, SearchPreferenceRequest):
            return super().remember_search_preferences(
                request.user_id,
                topic=request.topic,
                time_range=request.time_range,
                sources=request.sources,
                max_results=request.max_results,
            )
        return super().remember_search_preferences(
            request,
            topic=topic,
            time_range=time_range,
            sources=sources,
            max_results=max_results,
        )


class LiteratureSearchSkillHarness(LiteratureSearchComponent):
    def __init__(self, memory_harness: ResearchMemorySkillHarness | None = None) -> None:
        super().__init__(memory_component=memory_harness)

    def search(
        self,
        request: LiteratureSkillSearchRequest | str,
        *,
        platforms: Sequence[str] | None = None,
        max_results: int = 10,
        time_range: str = "",
        author: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        if isinstance(request, LiteratureSkillSearchRequest):
            return super().search(
                request.query,
                platforms=request.platforms,
                max_results=request.max_results,
                time_range=request.time_range,
                author=request.author,
                user_id=request.user_id,
            )
        return super().search(
            request,
            platforms=platforms,
            max_results=max_results,
            time_range=time_range,
            author=author,
            user_id=user_id,
        )


class DeepReadingSkillHarness(DeepReadingComponent):
    def fetch_full_text(
        self,
        request: PaperFetchSkillRequest | str,
        *,
        identifier_type: str = "auto",
        prefer: str = "pdf",
        download_dir: str = "",
    ):
        if isinstance(request, PaperFetchSkillRequest):
            return super().fetch_full_text(
                request.identifier,
                identifier_type=request.identifier_type,
                prefer=request.prefer,
                download_dir=request.download_dir,
            )
        return super().fetch_full_text(
            request,
            identifier_type=identifier_type,
            prefer=prefer,
            download_dir=download_dir,
        )

    def parse_pdf(
        self,
        request: PDFReadSkillRequest | str,
        *,
        target_section: str = "",
    ):
        if isinstance(request, PDFReadSkillRequest):
            return super().parse_pdf(
                request.pdf_path,
                target_section=request.target_section,
            )
        return super().parse_pdf(request, target_section=target_section)

    def targeted_read(
        self,
        request: TargetedReadSkillRequest | str,
        *,
        section_name: str = "",
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        if isinstance(request, TargetedReadSkillRequest):
            return super().targeted_read(
                request.pdf_path,
                section_name=request.section_name,
                max_chunks=request.max_chunks,
            )
        return super().targeted_read(
            request,
            section_name=section_name,
            max_chunks=max_chunks,
        )

    def extract_visuals(
        self,
        request: VisualExtractSkillRequest | str,
        *,
        page_numbers: List[int] | None = None,
        output_dir: str = "",
    ) -> Dict[str, Any]:
        if isinstance(request, VisualExtractSkillRequest):
            return super().extract_visuals(
                request.pdf_path,
                page_numbers=request.page_numbers,
                output_dir=request.output_dir,
            )
        return super().extract_visuals(
            request,
            page_numbers=page_numbers,
            output_dir=output_dir,
        )


class ResearchPlanningSkillHarness(ResearchPlanningComponent):
    def plan(
        self,
        request: ResearchPlanSkillRequest | str,
        *,
        intent: str = "generate_survey",
        slots: Dict[str, Any] | None = None,
    ):
        if isinstance(request, ResearchPlanSkillRequest):
            return super().plan(
                request.topic,
                intent=request.intent,
                slots=request.slots,
            )
        return super().plan(request, intent=intent, slots=slots)


class ResearchSkillsHarness:
    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.memory = ResearchMemorySkillHarness(memory_manager)
        self.search = LiteratureSearchSkillHarness(memory_harness=self.memory)
        self.reading = DeepReadingSkillHarness()
        self.planning = ResearchPlanningSkillHarness()
