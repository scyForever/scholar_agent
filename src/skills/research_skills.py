from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.core.models import Paper
from src.memory.manager import MemoryManager
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
from src.skills.harness import (
    DeepReadingSkillHarness,
    LiteratureSearchSkillHarness,
    ResearchMemorySkillHarness,
    ResearchPlanningSkillHarness,
    ResearchSkillsHarness,
)


class ResearchMemorySkill:
    """兼容旧接口的 memory skill 包装，内部统一委托给 harness。"""

    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.harness = ResearchMemorySkillHarness(memory_manager)

    def remember_preference(self, user_id: str, preference: str, *, metadata: Dict[str, Any] | None = None) -> str:
        return self.harness.remember_preference(
            PreferenceMemoryRequest(
                user_id=user_id,
                preference=preference,
                metadata=metadata,
            )
        )

    def remember_paper(
        self,
        user_id: str,
        paper: Paper,
        summary: str,
        *,
        highlights: List[str] | None = None,
    ) -> str:
        return self.harness.remember_paper(
            PaperMemoryRequest(
                user_id=user_id,
                paper=paper,
                summary=summary,
                highlights=highlights,
            )
        )

    def recall_context(self, user_id: str, query: str, *, limit: int = 8) -> List[Dict[str, Any]]:
        return self.harness.recall_context(
            MemoryRecallRequest(
                user_id=user_id,
                query=query,
                limit=limit,
            )
        )

    def rank_unseen_first(
        self,
        user_id: str,
        papers: Sequence[Paper],
        *,
        limit: int,
    ) -> Dict[str, Any]:
        return self.harness.rank_unseen_first(
            PaperRankingRequest(
                user_id=user_id,
                papers=papers,
                limit=limit,
            )
        )

    def remember_search_preferences(
        self,
        user_id: str,
        *,
        topic: str,
        time_range: str = "",
        sources: Sequence[str] | None = None,
        max_results: int | None = None,
    ) -> None:
        self.harness.remember_search_preferences(
            SearchPreferenceRequest(
                user_id=user_id,
                topic=topic,
                time_range=time_range,
                sources=sources,
                max_results=max_results,
            )
        )


class LiteratureSearchSkill:
    """兼容旧接口的 literature skill 包装。"""

    def __init__(self, memory_skill: ResearchMemorySkill | ResearchMemorySkillHarness | None = None) -> None:
        if isinstance(memory_skill, ResearchMemorySkill):
            memory_harness = memory_skill.harness
        else:
            memory_harness = memory_skill
        self.harness = LiteratureSearchSkillHarness(memory_harness=memory_harness)

    def search(
        self,
        query: str,
        *,
        platforms: Sequence[str] | None = None,
        max_results: int = 10,
        time_range: str = "",
        author: str = "",
        user_id: str = "",
    ) -> Dict[str, Any]:
        return self.harness.search(
            LiteratureSkillSearchRequest(
                query=query,
                platforms=platforms or [],
                max_results=max_results,
                time_range=time_range,
                author=author,
                user_id=user_id,
            )
        )


class DeepReadingSkill:
    """兼容旧接口的 deep reading skill 包装。"""

    def __init__(self) -> None:
        self.harness = DeepReadingSkillHarness()

    def fetch_full_text(
        self,
        identifier: str,
        *,
        identifier_type: str = "auto",
        prefer: str = "pdf",
        download_dir: str = "",
    ):
        return self.harness.fetch_full_text(
            PaperFetchSkillRequest(
                identifier=identifier,
                identifier_type=identifier_type,
                prefer=prefer,
                download_dir=download_dir,
            )
        )

    def parse_pdf(self, pdf_path: str, *, target_section: str = ""):
        return self.harness.parse_pdf(
            PDFReadSkillRequest(
                pdf_path=pdf_path,
                target_section=target_section,
            )
        )

    def targeted_read(
        self,
        pdf_path: str,
        *,
        section_name: str,
        max_chunks: int = 5,
    ) -> Dict[str, Any]:
        return self.harness.targeted_read(
            TargetedReadSkillRequest(
                pdf_path=pdf_path,
                section_name=section_name,
                max_chunks=max_chunks,
            )
        )

    def extract_visuals(
        self,
        pdf_path: str,
        *,
        page_numbers: List[int] | None = None,
        output_dir: str = "",
    ) -> Dict[str, Any]:
        return self.harness.extract_visuals(
            VisualExtractSkillRequest(
                pdf_path=pdf_path,
                page_numbers=page_numbers,
                output_dir=output_dir,
            )
        )


class ResearchPlanningSkill:
    """兼容旧接口的 planning skill 包装。"""

    def __init__(self) -> None:
        self.harness = ResearchPlanningSkillHarness()

    def plan(self, topic: str, *, intent: str = "generate_survey", slots: Dict[str, Any] | None = None):
        return self.harness.plan(
            ResearchPlanSkillRequest(
                topic=topic,
                intent=intent,
                slots=slots,
            )
        )


class ResearchSkillset:
    """兼容旧接口的 skillset 包装，内部统一使用 skills harness。"""

    def __init__(self, memory_manager: MemoryManager | None = None) -> None:
        self.harness = ResearchSkillsHarness(memory_manager)
        self.memory = self.harness.memory
        self.search = self.harness.search
        self.reading = self.harness.reading
        self.planning = self.harness.planning
