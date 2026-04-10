from __future__ import annotations

from typing import Callable, Dict

from src.core.models import AgentResponse, ExecutionMode
from src.harness import ScholarAgentHarness


class AgentV2:
    """向后兼容的公开入口，内部执行链路已切换到 harness。"""

    def __init__(self) -> None:
        self.harness = ScholarAgentHarness()
        services = self.harness.services
        self.llm = services.llm
        self.tracer = services.tracer
        self.templates = services.templates
        self.memory = services.memory
        self.feedback = services.feedback
        self.whitelist = services.whitelist
        self.dialogue = services.dialogue
        self.intent_classifier = services.intent_classifier
        self.slot_filler = services.slot_filler
        self.planner = services.planner
        self.retriever = services.retriever
        self.reasoning = services.reasoning
        self.quality = services.quality
        self.multi_agent = self.harness.multi_agent
        self.runtime_graph = self.harness.runtime_graph
        self.research_skills = self.harness.research_skills

    @property
    def execution_mode(self) -> ExecutionMode:
        return self.harness.execution_mode

    @property
    def enable_quality_enhance(self) -> bool:
        return self.harness.enable_quality_enhance

    def set_mode(self, fast_mode: bool = False, enable_quality_enhance: bool = True) -> None:
        self.harness.set_mode(
            fast_mode=fast_mode,
            enable_quality_enhance=enable_quality_enhance,
        )

    def index_pdf(self, pdf_path: str, title: str | None = None, metadata: Dict[str, Any] | None = None) -> str:
        return self.harness.index_pdf(pdf_path, title=title, metadata=metadata)

    def chat(
        self,
        query: str,
        session_id: str = "default",
        on_trace_start: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        return self.harness.chat(
            query,
            session_id=session_id,
            on_trace_start=on_trace_start,
        )

    def submit_feedback(self, session_id: str, query: str, response: str, rating: int, comment: str = "") -> None:
        self.harness.submit_feedback(session_id, query, response, rating, comment)

    def plan_research(self, topic: str, *, intent: str = "generate_survey", slots: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.harness.plan_research(topic, intent=intent, slots=slots)

    def fetch_paper(self, identifier: str, *, identifier_type: str = "auto", prefer: str = "pdf", download_dir: str = "") -> Dict[str, Any]:
        return self.harness.fetch_paper(
            identifier,
            identifier_type=identifier_type,
            prefer=prefer,
            download_dir=download_dir,
        )

    def read_paper(self, pdf_path: str, *, target_section: str = "") -> Dict[str, Any]:
        return self.harness.read_paper(pdf_path, target_section=target_section)

    def extract_paper_visuals(self, pdf_path: str, *, page_numbers: list[int] | None = None, output_dir: str = "") -> Dict[str, Any]:
        return self.harness.extract_paper_visuals(
            pdf_path,
            page_numbers=page_numbers,
            output_dir=output_dir,
        )

    def get_status(self) -> Dict[str, Any]:
        return self.harness.get_status()
