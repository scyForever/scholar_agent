from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict

from src.agents.multi_agent import MultiAgentCoordinator
from src.core.models import AgentResponse, ExecutionMode, MemoryType, SearchResult
from src.harness.contracts import ScholarChatRequest
from src.harness.services import ScholarAgentServices
from src.pipeline.runtime_graph import AgentRuntimeGraph


class ScholarAgentHarness:
    def __init__(self, services: ScholarAgentServices | None = None) -> None:
        self.services = services or ScholarAgentServices.build_default()
        self.multi_agent = MultiAgentCoordinator(
            llm=self.services.llm,
            retriever=self.services.retriever,
            whitelist=self.services.whitelist,
            reasoning=self.services.reasoning,
            templates=self.services.templates,
            tracer=self.services.tracer,
            memory_manager=self.services.memory,
        )
        self.runtime_graph = AgentRuntimeGraph(
            multi_agent=self.multi_agent,
            reasoning=self.services.reasoning,
            quality=self.services.quality,
            tracer=self.services.tracer,
        )
        self.research_skills = self.services.research_skills
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
        return self.services.retriever.index_pdf(pdf_path, title=title, metadata=metadata)

    def execute_chat(self, request: ScholarChatRequest) -> AgentResponse:
        state = self.services.dialogue.get_state(request.session_id)
        prior_search_result = state.last_search_result
        self.services.dialogue.add_user_message(request.session_id, request.query)
        trace_id = self.services.tracer.start_trace(
            request.session_id,
            request.query,
            {"mode": self.execution_mode.value},
        )
        if request.on_trace_start is not None:
            request.on_trace_start(trace_id)
        trace_tokens = self.services.llm.bind_trace(self.services.tracer, trace_id)
        budget_tokens = None
        intent = ""
        slots: Dict[str, Any] = {}

        try:
            recalled = self.services.memory.recall(request.query, user_id=request.session_id, limit=5)
            memory_context = "\n".join(item.content for item in recalled)
            self.services.tracer.trace_step(
                trace_id,
                "memory_recall",
                {"query": request.query},
                {"count": len(recalled)},
            )

            if state.missing_slots and state.intent:
                intent = state.intent
                current_slots = dict(state.current_slots)
            else:
                classified = self.services.intent_classifier.classify(request.query)
                intent = str(classified["intent"])
                current_slots = {}
                self.services.tracer.trace_step(trace_id, "intent", {"query": request.query}, classified)

            slot_result = self.services.slot_filler.fill_slots_once(request.query, intent, current_slots)
            self.services.tracer.trace_step(
                trace_id,
                "slots",
                {"query": request.query, "intent": intent},
                slot_result,
            )
            if slot_result["missing"]:
                self.services.dialogue.update_state(
                    request.session_id,
                    intent=intent,
                    current_slots=slot_result["slots"],
                    missing_slots=slot_result["missing"],
                    last_trace_id=trace_id,
                )
                answer = slot_result["ask"]
                self.services.dialogue.add_assistant_message(request.session_id, answer)
                self.services.tracer.finish_trace(trace_id, {"answer": answer})
                return AgentResponse(
                    answer=answer,
                    intent=intent,
                    slots=slot_result["slots"],
                    trace_id=trace_id,
                    needs_input=True,
                    whitebox=self.services.tracer.get_trace(trace_id),
                )

            slots = slot_result["slots"]
            self.services.dialogue.update_state(
                request.session_id,
                intent="",
                current_slots={},
                missing_slots=[],
                last_trace_id=trace_id,
            )
            level, config = self.services.planner.classify(request.query, intent, slots)
            budget_tokens = self.services.llm.bind_budget(config.max_llm_calls)
            self.services.tracer.trace_step(
                trace_id,
                "planning",
                {"query": request.query, "intent": intent, "slots": slots},
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
                query=request.query,
                intent=intent,
                slots=slots,
                session_id=request.session_id,
                trace_id=trace_id,
                task_config=config,
                history=self.services.dialogue.get_state(request.session_id).history,
                memory_context=memory_context,
                prior_search_result=prior_search_result,
                execution_mode=self.execution_mode,
                enable_quality_enhance=self.enable_quality_enhance,
            )
            artifacts = dict(runtime_result.get("artifacts") or {})
            answer = str(runtime_result.get("answer") or "")
            latest_search_result = artifacts.get("search_result")
            if isinstance(latest_search_result, SearchResult):
                self.services.dialogue.update_state(
                    request.session_id,
                    last_search_result=latest_search_result,
                )

            self.services.memory.store(
                request.session_id,
                f"用户问题：{request.query}\n系统回答：{answer}",
                memory_type=MemoryType.CONVERSATION,
                metadata={"intent": intent, "task_level": level.value},
                importance=0.7,
            )

            self.services.dialogue.add_assistant_message(request.session_id, answer)
            self.services.tracer.finish_trace(trace_id, {"answer": answer, "intent": intent})
            return AgentResponse(
                answer=answer,
                intent=intent,
                slots=slots,
                trace_id=trace_id,
                whitebox=self.services.tracer.get_trace(trace_id),
                artifacts=artifacts,
            )
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            self.services.tracer.trace_step(
                trace_id,
                "error",
                {"query": request.query, "intent": intent, "slots": slots},
                {"error": error_message},
            )
            self.services.tracer.fail_trace(
                trace_id,
                {"error": error_message, "intent": intent, "slots": slots},
            )
            raise
        finally:
            if budget_tokens is not None:
                self.services.llm.reset_budget(budget_tokens)
            self.services.llm.reset_trace(trace_tokens)

    def chat(
        self,
        query: str,
        session_id: str = "default",
        on_trace_start: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        return self.execute_chat(
            ScholarChatRequest(
                query=query,
                session_id=session_id,
                on_trace_start=on_trace_start,
            )
        )

    def submit_feedback(self, session_id: str, query: str, response: str, rating: int, comment: str = "") -> None:
        self.services.feedback.record_feedback(session_id, query, response, rating, comment)

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
        return self.research_skills.reading.extract_visuals(
            pdf_path,
            page_numbers=page_numbers,
            output_dir=output_dir,
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.execution_mode.value,
            "quality_enhance": self.enable_quality_enhance,
            "runtime_graph": self.runtime_graph.uses_langgraph(),
            "multi_agent_graph": self.multi_agent.uses_langgraph(),
            "llm_status": self.services.llm.get_status(),
            "templates": list(self.services.templates.list_templates()),
        }
