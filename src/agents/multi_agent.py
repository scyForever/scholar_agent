from __future__ import annotations

from typing import Any, Dict, List

from src.agents.analyze_agent import AnalyzeAgent
from src.agents.coder_agent import CoderAgent
from src.agents.debate_agent import DebateAgent
from src.agents.research_agents import (
    ResearchMemoryAgent,
    ResearchPlannerAgent,
    ResearchReadingAgent,
    ResearchSearchAgent,
)
from src.agents.search_agent import SearchAgent
from src.agents.write_agent import WriteAgent
from src.core.llm import LLMManager
from src.core.models import ExecutionMode, SearchResult
from src.memory.manager import MemoryManager
from src.pipeline import MultiAgentPipeline
from src.planning.task_hierarchy import TaskConfig
from src.prompt_templates.manager import PromptTemplateManager
from src.rag.retriever import HybridRetriever
from src.reasoning.engine import ReasoningEngine
from src.skills import ResearchSkillset
from src.whitelist.manager import WhitelistManager
from src.whitebox.tracer import WhiteboxTracer


class MultiAgentCoordinator:
    intent_flows_full = {
        "generate_survey": ["plan", "search", "analyze", "debate", "write"],
        "compare_methods": ["plan", "search", "analyze", "debate", "write"],
        "generate_code": ["plan", "search", "analyze", "coder"],
        "search_papers": ["search", "write"],
        "daily_update": ["search", "analyze", "write"],
        "analyze_paper": ["search", "analyze", "write"],
        "explain_concept": ["search", "write"],
    }

    intent_flows_fast = {
        "generate_survey": ["plan", "search", "analyze", "write"],
        "compare_methods": ["plan", "search", "analyze", "write"],
        "generate_code": ["plan", "search", "analyze", "coder"],
        "search_papers": ["search", "write"],
        "daily_update": ["search", "analyze", "write"],
        "analyze_paper": ["search", "analyze", "write"],
        "explain_concept": ["write"],
    }

    def __init__(
        self,
        llm: LLMManager,
        retriever: HybridRetriever,
        whitelist: WhitelistManager,
        reasoning: ReasoningEngine,
        templates: PromptTemplateManager,
        tracer: WhiteboxTracer,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.llm = llm
        self.skills = ResearchSkillset(memory_manager)
        self.planner_agent = ResearchPlannerAgent(self.skills, tracer)
        self.research_search_agent = ResearchSearchAgent(self.skills, tracer)
        self.reading_agent = ResearchReadingAgent(self.skills, tracer)
        self.memory_agent = ResearchMemoryAgent(self.skills, tracer)
        self.search_agent = SearchAgent(
            retriever,
            whitelist,
            tracer,
            research_search_agent=self.research_search_agent,
        )
        self.analyze_agent = AnalyzeAgent(
            llm,
            templates,
            tracer,
            reading_agent=self.reading_agent,
            memory_agent=self.memory_agent,
        )
        self.debate_agent = DebateAgent(reasoning, tracer)
        self.write_agent = WriteAgent(llm, templates, tracer)
        self.coder_agent = CoderAgent(llm, templates, tracer)
        self.pipeline = MultiAgentPipeline(self)
        self.graph = self.pipeline.graph

    def uses_langgraph(self) -> bool:
        return self.pipeline.uses_langgraph()

    def execute(
        self,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        mode: ExecutionMode,
        trace_id: str,
        task_config: TaskConfig | None = None,
        history: List[Dict[str, str]] | None = None,
        session_id: str = "",
        prior_search_result: SearchResult | None = None,
    ) -> Dict[str, Any]:
        flow = self._resolve_flow(
            intent=intent,
            mode=mode,
            task_config=task_config,
            slots=slots,
            prior_search_result=prior_search_result,
        )
        return self.pipeline.execute(
            query=query,
            intent=intent,
            slots=slots,
            mode=mode,
            trace_id=trace_id,
            flow=flow,
            task_config=task_config,
            history=history,
            session_id=session_id,
            prior_search_result=prior_search_result,
        )

    def _resolve_flow(
        self,
        *,
        intent: str,
        mode: ExecutionMode,
        task_config: TaskConfig | None,
        slots: Dict[str, Any],
        prior_search_result: SearchResult | None,
    ) -> List[str]:
        flow_map = (
            self.intent_flows_fast
            if mode == ExecutionMode.FAST
            else self.intent_flows_full
        )
        if task_config is not None and not task_config.enable_multi_agent:
            flow_map = self.intent_flows_fast
        flow = list(flow_map.get(intent, ["search", "write"]))

        if intent != "explain_concept":
            return flow

        rag_mode = str(slots.get("rag_mode") or "auto")
        context_source = str(slots.get("context_source") or "")
        if rag_mode == "off":
            return ["write"]
        if rag_mode == "local_only":
            return ["search", "write"]
        if context_source == "previous_search":
            return ["search", "write"]
        if prior_search_result is not None and any(
            marker in str(slots.get("topic") or "")
            for marker in ("之前查找到的资料", "之前查到的资料", "之前搜索到的资料", "之前的检索结果", "前面的检索结果")
        ):
            return ["search", "write"]
        return flow

    def _flow_index(self, flow: List[str], step_name: str) -> int:
        try:
            return flow.index(step_name)
        except ValueError:
            return 0

    def _resolve_analysis_limit(
        self,
        *,
        flow: List[str],
        current_index: int,
        query: str,
        task_config: TaskConfig | None,
    ) -> int:
        budget = self.llm.get_budget_status()
        remaining = budget.get("remaining")
        if remaining is None:
            return 5
        reserved = self._reserve_future_llm_calls(
            flow=flow,
            current_index=current_index,
            query=query,
            task_config=task_config,
        )
        available = max(int(remaining) - reserved, 0)
        return min(5, available)

    def _reserve_future_llm_calls(
        self,
        *,
        flow: List[str],
        current_index: int,
        query: str,
        task_config: TaskConfig | None,
    ) -> int:
        reserved = 0
        base_modes = task_config.reasoning_modes if task_config is not None else []
        preferred_modes = list(dict.fromkeys(["debate", *base_modes]))
        for step in flow[current_index + 1 :]:
            if step in {"write", "coder"}:
                reserved += 1
            elif step == "debate":
                reserved += self.debate_agent.reasoning.estimate_llm_calls(
                    query,
                    mode="auto",
                    preferred_modes=preferred_modes,
                )
        return reserved
