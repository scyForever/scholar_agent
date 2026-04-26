from __future__ import annotations

from dataclasses import asdict

from typing import Any, Callable, Dict

from src.agents.multi_agent import MultiAgentCoordinator
from src.core.llm import LLMManager
from src.core.models import AgentResponse, ExecutionMode, MemoryType, SearchResult
from src.feedback.collector import FeedbackCollector
from src.memory.context_builder import MemoryContextBuilder
from src.memory.manager import MemoryManager
from src.pipeline import AgentRuntimeGraph
from src.planning.task_hierarchy import TaskHierarchyPlanner
from src.preprocessing.dialogue_manager import DialogueManager
from src.preprocessing.intent_classifier import IntentClassifier
from src.preprocessing.slot_filler import SlotFiller
from src.prompt_templates.manager import PromptTemplateManager
from src.quality.enhancer import QualityEnhancer
from src.rag.retriever import HybridRetriever
from src.reasoning.engine import ReasoningEngine
from src.skills import ResearchSkillset
from src.whitelist.manager import WhitelistManager
from src.whitebox.tracer import WhiteboxTracer


class AgentV2:
    def __init__(self) -> None:
        self.llm = LLMManager()
        self.tracer = WhiteboxTracer()
        self.templates = PromptTemplateManager()
        self.templates.ensure_default_templates()
        self.memory = MemoryManager()
        self.memory_context_builder = MemoryContextBuilder()
        self.feedback = FeedbackCollector()
        self.whitelist = WhitelistManager()
        self.dialogue = DialogueManager()
        self.intent_classifier = IntentClassifier(self.llm, self.templates)
        self.slot_filler = SlotFiller()
        self.planner = TaskHierarchyPlanner()
        self.retriever = HybridRetriever(llm=self.llm)
        self.reasoning = ReasoningEngine(
            self.llm,
            self.tracer,
            retriever=self.retriever,
            whitelist=self.whitelist,
        )
        self.quality = QualityEnhancer(self.llm)
        self.multi_agent = MultiAgentCoordinator(
            llm=self.llm,
            retriever=self.retriever,
            whitelist=self.whitelist,
            reasoning=self.reasoning,
            templates=self.templates,
            tracer=self.tracer,
            memory_manager=self.memory,
        )
        self.runtime_graph = AgentRuntimeGraph(
            multi_agent=self.multi_agent,
            reasoning=self.reasoning,
            quality=self.quality,
            tracer=self.tracer,
        )
        self.research_skills = ResearchSkillset(self.memory)
        self.execution_mode = ExecutionMode.STANDARD
        self.enable_quality_enhance = False

    def set_mode(self, fast_mode: bool = False, enable_quality_enhance: bool = True) -> None:
        if fast_mode:
            self.execution_mode = ExecutionMode.FAST
        elif enable_quality_enhance:
            self.execution_mode = ExecutionMode.FULL
        else:
            self.execution_mode = ExecutionMode.STANDARD
        self.enable_quality_enhance = enable_quality_enhance

    def index_pdf(self, pdf_path: str, title: str | None = None, metadata: Dict[str, Any] | None = None) -> str:
        return self.retriever.index_pdf(pdf_path, title=title, metadata=metadata)

    def chat(
        self,
        query: str,
        session_id: str = "default",
        on_trace_start: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        state = self.dialogue.get_state(session_id)
        prior_search_result = state.last_search_result
        self.dialogue.add_user_message(session_id, query)
        trace_id = self.tracer.start_trace(session_id, query, {"mode": self.execution_mode.value})
        if on_trace_start is not None:
            on_trace_start(trace_id)
        trace_tokens = self.llm.bind_trace(self.tracer, trace_id)
        budget_tokens = None
        intent = ""
        slots: Dict[str, Any] = {}

        try:
            recalled = self.memory.recall(query, user_id=session_id, limit=5)
            short_memory = self.dialogue.get_state(session_id).short_memory
            memory_context_result = self.memory_context_builder.build(
                query=query,
                short_memory=short_memory,
                long_records=recalled,
            )
            memory_context = memory_context_result.text
            self.tracer.trace_step(
                trace_id,
                "memory_recall",
                {"query": query},
                {
                    "short_layers": {
                        "raw": len(short_memory.raw),
                        "highlights": len(short_memory.highlights),
                        "summary": bool(short_memory.summary),
                    },
                    "long_count": len(recalled),
                    "context_budget": memory_context_result.stats,
                },
            )

            if state.missing_slots and state.intent:
                intent = state.intent
                current_slots = dict(state.current_slots)
            else:
                classified = self.intent_classifier.classify(query)
                intent = str(classified["intent"])
                current_slots = {}
                self.tracer.trace_step(trace_id, "intent", {"query": query}, classified)

            slot_result = self.slot_filler.fill_slots_once(query, intent, current_slots)
            self.tracer.trace_step(trace_id, "slots", {"query": query, "intent": intent}, slot_result)
            if slot_result["missing"]:
                state = self.dialogue.update_state(
                    session_id,
                    intent=intent,
                    current_slots=slot_result["slots"],
                    missing_slots=slot_result["missing"],
                    last_trace_id=trace_id,
                )
                answer = slot_result["ask"]
                self.dialogue.add_assistant_message(session_id, answer)
                self.tracer.finish_trace(trace_id, {"answer": answer})
                return AgentResponse(
                    answer=answer,
                    intent=intent,
                    slots=slot_result["slots"],
                    trace_id=trace_id,
                    needs_input=True,
                    whitebox=self.tracer.get_trace(trace_id),
                )

            slots = slot_result["slots"]
            self.dialogue.update_state(session_id, intent="", current_slots={}, missing_slots=[], last_trace_id=trace_id)
            level, config = self.planner.classify(query, intent, slots)
            budget_tokens = self.llm.bind_budget(config.max_llm_calls)
            self.tracer.trace_step(
                trace_id,
                "planning",
                {"query": query, "intent": intent, "slots": slots},
                {
                    "task_level": level.value,
                    "task_config": asdict(config),
                    "runtime_constraints": {
                        "enable_multi_agent": config.enable_multi_agent,
                        "max_llm_calls": config.max_llm_calls,
                    },
                },
            )

            runtime_result = self.runtime_graph.execute(
                query=query,
                intent=intent,
                slots=slots,
                session_id=session_id,
                trace_id=trace_id,
                task_config=config,
                history=self.dialogue.get_state(session_id).history,
                memory_context=memory_context,
                prior_search_result=prior_search_result,
                execution_mode=self.execution_mode,
                enable_quality_enhance=self.enable_quality_enhance,
            )
            artifacts = dict(runtime_result.get("artifacts") or {})
            answer = str(runtime_result.get("answer") or "")
            latest_search_result = artifacts.get("search_result")
            if isinstance(latest_search_result, SearchResult):
                self.dialogue.update_state(session_id, last_search_result=latest_search_result)

            self.memory.store(
                session_id,
                self._conversation_memory_content(query, answer, intent),
                memory_type=MemoryType.CONVERSATION,
                metadata={
                    "intent": intent,
                    "task_level": level.value,
                    "short_memory_summary": self.dialogue.get_state(session_id).short_memory.summary,
                },
                importance=0.7,
            )

            self.dialogue.add_assistant_message(session_id, answer)
            self.tracer.finish_trace(trace_id, {"answer": answer, "intent": intent})
            return AgentResponse(
                answer=answer,
                intent=intent,
                slots=slots,
                trace_id=trace_id,
                whitebox=self.tracer.get_trace(trace_id),
                artifacts=artifacts,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            self.tracer.trace_step(
                trace_id,
                "error",
                {"query": query, "intent": intent, "slots": slots},
                {"error": error_message},
            )
            self.tracer.fail_trace(
                trace_id,
                {"error": error_message, "intent": intent, "slots": slots},
            )
            raise
        finally:
            if budget_tokens is not None:
                self.llm.reset_budget(budget_tokens)
            self.llm.reset_trace(trace_tokens)

    def _conversation_memory_content(self, query: str, answer: str, intent: str) -> str:
        compact_answer = answer.strip()
        if len(compact_answer) > 600:
            compact_answer = compact_answer[:597] + "..."
        compact_query = query.strip()
        if len(compact_query) > 240:
            compact_query = compact_query[:237] + "..."
        return "\n".join(
            [
                f"用户问题：{compact_query}",
                f"任务意图：{intent}",
                f"回答摘要：{compact_answer}",
            ]
        )

    def submit_feedback(self, session_id: str, query: str, response: str, rating: int, comment: str = "") -> None:
        self.feedback.record_feedback(session_id, query, response, rating, comment)

    def plan_research(self, topic: str, *, intent: str = "generate_survey", slots: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return asdict(self.research_skills.planning.plan(topic, intent=intent, slots=slots))

    def fetch_paper(self, identifier: str, *, identifier_type: str = "auto", prefer: str = "pdf", download_dir: str = "") -> Dict[str, Any]:
        return asdict(
            self.research_skills.reading.fetch_full_text(
                identifier,
                identifier_type=identifier_type,
                prefer=prefer,
                download_dir=download_dir,
            )
        )

    def read_paper(self, pdf_path: str, *, target_section: str = "") -> Dict[str, Any]:
        return asdict(self.research_skills.reading.parse_pdf(pdf_path, target_section=target_section))

    def extract_paper_visuals(self, pdf_path: str, *, page_numbers: list[int] | None = None, output_dir: str = "") -> Dict[str, Any]:
        return self.research_skills.reading.extract_visuals(pdf_path, page_numbers=page_numbers, output_dir=output_dir)

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.execution_mode.value,
            "quality_enhance": self.enable_quality_enhance,
            "runtime_graph": self.runtime_graph.uses_langgraph(),
            "multi_agent_graph": self.multi_agent.uses_langgraph(),
            "llm_status": self.llm.get_status(),
            "templates": list(self.templates.list_templates()),
        }
