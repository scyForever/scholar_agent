from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from typing import Any, Dict, List, TypedDict

from config.settings import settings
from src.agents.research_agents import (
    ResearchMemoryAgent,
    ResearchPlannerAgent,
    ResearchReadingAgent,
    ResearchSearchAgent,
)
from src.core.llm import LLMManager
from src.core.models import (
    DebateResult,
    ExecutionMode,
    Paper,
    PaperAnalysis,
    ResearchPlan,
    SearchResult,
)
from src.memory.manager import MemoryManager
from src.core.structured_outputs import SearchAgentExecutionStep, SearchAgentFinalOutput
from src.planning.task_hierarchy import TaskConfig
from src.preprocessing.query_rewriter import QueryRewriter
from src.prompt_templates.manager import PromptTemplateManager
from src.rag.retriever import HybridRetriever
from src.reasoning.engine import ReasoningEngine
from src.skills import ResearchSkillset
from src.tools import TOOL_REGISTRY
from src.whitelist.manager import WhitelistManager
from src.whitebox.tracer import WhiteboxTracer

try:
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
    from langchain_core.tools import StructuredTool
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover
    create_agent = None
    AIMessage = None
    BaseMessage = None
    ToolMessage = None
    StructuredTool = None
    END = None
    START = None
    StateGraph = None


class MultiAgentState(TypedDict, total=False):
    query: str
    intent: str
    slots: Dict[str, Any]
    mode: ExecutionMode
    session_id: str
    trace_id: str
    task_config: TaskConfig | None
    history: List[Dict[str, str]]
    flow: List[str]
    research_plan: ResearchPlan | None
    search_result: SearchResult | None
    analyses: List[PaperAnalysis]
    debate: DebateResult | None
    answer: str
    artifacts: Dict[str, Any]


class SearchAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        whitelist: WhitelistManager,
        tracer: WhiteboxTracer,
        research_search_agent: ResearchSearchAgent | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = retriever.llm
        self.whitelist = whitelist
        self.tracer = tracer
        self.rewriter = QueryRewriter(retriever.llm)
        self.research_search_agent = research_search_agent

    def run(
        self,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        history: List[Dict[str, str]],
        trace_id: str,
        session_id: str = "",
    ) -> SearchResult:
        topic = str(slots.get("topic") or slots.get("paper_title") or query)
        time_range = str(slots.get("time_range") or "")
        max_results = int(slots.get("max_papers") or 12)
        rewrite_plan = self.rewriter.plan(topic, intent=intent)
        rewritten_queries = rewrite_plan.external_queries
        local_context = self.retriever.retrieve(
            topic,
            history,
            top_k=5,
            rewritten_queries=rewrite_plan.local_queries,
            rewrite_plan=rewrite_plan,
        )
        if intent == "analyze_paper" and local_context.get("results"):
            result = SearchResult(
                query=topic,
                papers=[],
                total_found=0,
                source_breakdown={"local_rag": len(local_context["results"])},
                rewritten_queries=rewritten_queries,
                trace={"local_rag": local_context, "search_mode": "local_rag_only"},
            )
            self.tracer.trace_step(trace_id, "search", {"query": topic}, asdict(result))
            return result

        allowed_tool_names = [
            tool
            for tool in self.whitelist.allowed_tools("search_agent")
            if tool != "search_web"
        ]
        prioritized_tool_names = self._prioritize_search_tools(
            topic=topic,
            intent=intent,
            allowed_tool_names=allowed_tool_names,
        )
        search_payload = self._run_external_search(
            topic=topic,
            intent=intent,
            time_range=time_range,
            max_results=max_results,
            rewritten_queries=rewritten_queries,
            allowed_tool_names=prioritized_tool_names,
        )
        aggregated = search_payload["aggregated"]

        papers = sorted(
            aggregated.values(),
            key=lambda item: (item.score, item.citations, item.year or 0),
            reverse=True,
        )[:max_results]
        memory_trace: Dict[str, Any] = {}
        if session_id and self.research_search_agent is not None:
            ranked = self.research_search_agent.skills.memory.rank_unseen_first(
                session_id,
                papers,
                limit=max_results,
            )
            papers = list(ranked["papers"])
            memory_trace = {
                "seen_count": ranked["seen_count"],
                "unseen_count": ranked["unseen_count"],
                "seen_titles": ranked["seen_titles"],
            }
            self.research_search_agent.skills.memory.remember_search_preferences(
                session_id,
                topic=topic,
                time_range=time_range,
                sources=search_payload["selected_tools"],
                max_results=max_results,
            )
        source_breakdown: Dict[str, int] = defaultdict(int)
        for paper in papers:
            source_breakdown[paper.source] += 1
        result = SearchResult(
            query=topic,
            papers=papers,
            total_found=len(aggregated),
            source_breakdown=dict(source_breakdown),
            rewritten_queries=rewritten_queries,
            trace={
                "local_rag": local_context,
                "search_mode": search_payload["search_mode"],
                "tool_strategy": search_payload["tool_strategy"],
                "agent_selected_tools": search_payload["selected_tools"],
                "agent_tool_calls": search_payload["tool_calls"],
                "agent_summary": search_payload["agent_summary"],
                "agent_final_output": search_payload["final_output"],
                "agent_output_source": search_payload["final_output_source"],
                "agent_provider_attempts": search_payload["provider_attempts"],
                "agent_errors": search_payload["agent_errors"],
                "memory_ranking": memory_trace,
            },
        )
        self.tracer.trace_step(trace_id, "search", {"query": topic}, asdict(result))
        return result

    def _paper_score(self, paper: Paper) -> float:
        recency = float((paper.year or 0) / 3000)
        citations = min(paper.citations / 1000.0, 1.0)
        return 0.55 * citations + 0.45 * recency

    def _run_external_search(
        self,
        *,
        topic: str,
        intent: str,
        time_range: str,
        max_results: int,
        rewritten_queries: List[str],
        allowed_tool_names: List[str],
    ) -> Dict[str, Any]:
        planning_provider_names = self._search_planning_provider_names()
        fallback_errors: List[str] = []
        if not allowed_tool_names:
            fallback_errors.append("no_search_tools_allowed")
        elif create_agent is None or StructuredTool is None:
            fallback_errors.append("langchain_agent_runtime_unavailable")
        elif not planning_provider_names:
            fallback_errors.append("zhipu_unavailable_for_search_agent")

        if (
            not self._supports_agentic_search()
            or not allowed_tool_names
            or not planning_provider_names
        ):
            fallback_result = self._deterministic_search(
                rewritten_queries=rewritten_queries,
                max_results=max_results,
                time_range=time_range,
                tool_names=allowed_tool_names,
            )
            return {
                "aggregated": fallback_result["aggregated"],
                "search_mode": "hybrid",
                "tool_strategy": "deterministic_fallback",
                "selected_tools": fallback_result["selected_tools"],
                "tool_calls": fallback_result["tool_calls"],
                "agent_summary": fallback_result["final_output"].aggregation.summary,
                "final_output": fallback_result["final_output"].model_dump(mode="json"),
                "final_output_source": "deterministic_search_fallback",
                "provider_attempts": [],
                "agent_errors": fallback_errors,
            }

        agent_tools = self._build_agent_tools(allowed_tool_names)
        if not agent_tools:
            fallback_result = self._deterministic_search(
                rewritten_queries=rewritten_queries,
                max_results=max_results,
                time_range=time_range,
                tool_names=allowed_tool_names,
            )
            return {
                "aggregated": fallback_result["aggregated"],
                "search_mode": "hybrid",
                "tool_strategy": "deterministic_fallback",
                "selected_tools": fallback_result["selected_tools"],
                "tool_calls": fallback_result["tool_calls"],
                "agent_summary": fallback_result["final_output"].aggregation.summary,
                "final_output": fallback_result["final_output"].model_dump(mode="json"),
                "final_output_source": "deterministic_search_fallback",
                "provider_attempts": [],
                "agent_errors": fallback_errors,
            }

        provider_attempts: List[str] = []
        agent_errors: List[str] = []
        user_prompt = self._search_agent_user_prompt(
            topic=topic,
            intent=intent,
            time_range=time_range,
            max_results=max_results,
            rewritten_queries=rewritten_queries,
        )

        for provider_name in planning_provider_names:
            provider_attempts.append(provider_name)
            try:
                model = self.llm.create_langchain_chat_model(
                    provider_name,
                    purpose="搜索工具规划",
                    temperature=0.0,
                    max_tokens=1200,
                )
                agent = create_agent(
                    model=model,
                    tools=agent_tools,
                    system_prompt=self._search_agent_system_prompt(),
                    name="scholar_search_agent",
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_prompt}]},
                    config={"recursion_limit": 6},
                    interrupt_after=["tools"],
                )
                payload = self._parse_agent_result(
                    result, allowed_tool_names=allowed_tool_names
                )
                final_output, final_output_source = self._resolve_agent_final_output(
                    result=result,
                    provider_name=provider_name,
                    topic=topic,
                    time_range=time_range,
                    rewritten_queries=rewritten_queries,
                    payload=payload,
                    allowed_tool_names=allowed_tool_names,
                )
                payload["final_output"] = final_output.model_dump(mode="json")
                payload["final_output_source"] = final_output_source
                payload["agent_summary"] = (
                    final_output.aggregation.summary or payload["agent_summary"]
                )
                for tool_name in final_output.selected_tools:
                    if (
                        tool_name in allowed_tool_names
                        and tool_name not in payload["selected_tools"]
                    ):
                        payload["selected_tools"].append(tool_name)
                self.llm.record_provider_success(provider_name)
                if payload["aggregated"]:
                    payload["search_mode"] = "hybrid_agentic"
                    payload["tool_strategy"] = "langchain_agent"
                    payload["provider_attempts"] = provider_attempts
                    payload["agent_errors"] = agent_errors
                    return payload
                agent_errors.append(f"{provider_name}: no_tool_results")
            except Exception as exc:
                self.llm.record_provider_failure(provider_name)
                agent_errors.append(f"{provider_name}: {type(exc).__name__}: {exc}")

        fallback_result = self._deterministic_search(
            rewritten_queries=rewritten_queries,
            max_results=max_results,
            time_range=time_range,
            tool_names=allowed_tool_names,
        )
        return {
            "aggregated": fallback_result["aggregated"],
            "search_mode": "hybrid",
            "tool_strategy": "deterministic_fallback_after_agent_failure",
            "selected_tools": fallback_result["selected_tools"],
            "tool_calls": fallback_result["tool_calls"],
            "agent_summary": fallback_result["final_output"].aggregation.summary,
            "final_output": fallback_result["final_output"].model_dump(mode="json"),
            "final_output_source": "deterministic_search_fallback",
            "provider_attempts": provider_attempts,
            "agent_errors": agent_errors,
        }

    def _deterministic_search(
        self,
        *,
        rewritten_queries: List[str],
        max_results: int,
        time_range: str,
        tool_names: List[str],
    ) -> Dict[str, Any]:
        aggregated: Dict[str, Paper] = {}
        tool_calls: List[Dict[str, Any]] = []
        selected_tools: List[str] = []
        candidate_queries = rewritten_queries[:2] if len(tool_names) >= 4 else rewritten_queries[:3]
        for tool_name in tool_names:
            used = False
            for rewritten in candidate_queries:
                papers = self._invoke_search_tool(
                    tool_name,
                    query=rewritten,
                    max_results=max_results // max(len(tool_names), 1) + 2,
                    time_range=time_range,
                    use_langchain=False,
                )
                tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "query": rewritten,
                        "max_results": max_results // max(len(tool_names), 1) + 2,
                        "time_range": time_range,
                        "count": len(papers),
                    }
                )
                if papers:
                    used = True
                self._merge_papers(aggregated, papers)
            if used and tool_name not in selected_tools:
                selected_tools.append(tool_name)
        final_output = self._fallback_agent_final_output(
            selected_tools=selected_tools,
            tool_calls=tool_calls,
            aggregated=aggregated,
            summary_reason="未命中 LangChain agent 时，按白名单顺序执行确定性检索。",
        )
        return {
            "aggregated": aggregated,
            "selected_tools": selected_tools,
            "tool_calls": tool_calls,
            "final_output": final_output,
        }

    def _invoke_search_tool(
        self,
        tool_name: str,
        *,
        query: str,
        max_results: int,
        time_range: str,
        use_langchain: bool,
    ) -> List[Paper]:
        try:
            if use_langchain:
                tool = TOOL_REGISTRY.get_langchain_tool(tool_name)
                result = tool.invoke(
                    {
                        "query": query,
                        "max_results": max_results,
                        "time_range": time_range,
                    }
                )
                return self._normalize_tool_papers(result)
            result = TOOL_REGISTRY.call(
                tool_name,
                query=query,
                max_results=max_results,
                time_range=time_range,
            )
            return self._normalize_tool_papers(result)
        except TypeError:
            if use_langchain:
                tool = TOOL_REGISTRY.get_langchain_tool(tool_name)
                result = tool.invoke({"query": query, "max_results": max_results})
                return self._normalize_tool_papers(result)
            result = TOOL_REGISTRY.call(tool_name, query=query, max_results=max_results)
            return self._normalize_tool_papers(result)
        except Exception:
            return []

    def _normalize_tool_papers(self, result: Any) -> List[Paper]:
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict) and isinstance(result.get("papers"), list):
            items = result.get("papers") or []
        else:
            return []
        papers: List[Paper] = []
        for item in items:
            if isinstance(item, Paper):
                papers.append(item)
            elif isinstance(item, dict):
                try:
                    papers.append(self._paper_from_payload(item))
                except Exception:
                    continue
        return papers

    def _merge_papers(self, aggregated: Dict[str, Paper], papers: List[Paper]) -> None:
        for paper in papers:
            key = (paper.title or "").strip().lower()
            if not key:
                continue
            paper.score = self._paper_score(paper)
            if key not in aggregated or paper.score > aggregated[key].score:
                aggregated[key] = paper

    def _supports_agentic_search(self) -> bool:
        return (
            create_agent is not None
            and StructuredTool is not None
            and bool(self.llm.get_healthy_provider_names())
        )

    def _prioritize_search_tools(
        self,
        *,
        topic: str,
        intent: str,
        allowed_tool_names: List[str],
    ) -> List[str]:
        if not allowed_tool_names:
            return []
        lowered = topic.lower()
        if any(token in lowered for token in ("medical", "biomedical", "clinical", "drug", "patient", "蛋白", "医学", "生物")):
            preferred = ["search_pubmed", "search_semantic_scholar", "search_openalex"]
        elif any(token in lowered for token in ("wireless", "signal", "通信", "robot", "robotics", "control", "芯片", "电气")):
            preferred = ["search_ieee_xplore", "search_openalex", "search_semantic_scholar"]
        else:
            preferred = ["search_arxiv", "search_openalex", "search_semantic_scholar"]
        if intent in {"daily_update", "generate_survey"}:
            preferred.append("search_web_of_science")
        ordered = [name for name in preferred if name in allowed_tool_names]
        ordered.extend(
            name for name in allowed_tool_names if name not in ordered and name != "search_literature"
        )
        if "search_literature" in allowed_tool_names:
            ordered.append("search_literature")
        limit = 3 if len(ordered) >= 3 else len(ordered)
        return ordered[:limit]

    def _search_planning_provider_names(self) -> List[str]:
        # 搜索工具规划阶段固定只使用 zhipu，避免在多个 provider 间轮询。
        return [
            name for name in self.llm.get_verified_provider_names() if name == "zhipu"
        ]

    def _build_agent_tools(self, tool_names: List[str]) -> List[Any]:
        agent_tools: List[Any] = []
        if StructuredTool is None:
            return agent_tools
        for tool_name in tool_names:
            try:
                base_tool = TOOL_REGISTRY.get_langchain_tool(tool_name)
            except KeyError:
                continue

            def _runner(
                _tool_name: str = tool_name,
                **kwargs: Any,
            ) -> tuple[str, Dict[str, Any]]:
                papers = self._invoke_search_tool(
                    _tool_name,
                    query=str(kwargs.get("query") or ""),
                    max_results=int(kwargs.get("max_results") or 10),
                    time_range=str(kwargs.get("time_range") or ""),
                    use_langchain=False,
                )
                artifact = {
                    "tool_name": _tool_name,
                    "query": str(kwargs.get("query") or ""),
                    "max_results": int(kwargs.get("max_results") or 10),
                    "time_range": str(kwargs.get("time_range") or ""),
                    "papers": [asdict(paper) for paper in papers],
                }
                return self._tool_summary(_tool_name, papers), artifact

            agent_tools.append(
                StructuredTool.from_function(
                    func=_runner,
                    name=tool_name,
                    description=str(base_tool.description or ""),
                    args_schema=base_tool.args_schema,
                    infer_schema=False,
                    response_format="content_and_artifact",
                )
            )
        return agent_tools

    def _tool_summary(self, tool_name: str, papers: List[Paper]) -> str:
        if not papers:
            return f"{tool_name} 未返回结果。"
        lines = [f"{tool_name} 返回 {len(papers)} 篇候选论文。"]
        for paper in papers[:3]:
            lines.append(
                f"- {paper.title} ({paper.year or 'N/A'}) | {paper.source} | citations={paper.citations}"
            )
        return "\n".join(lines)

    def _search_agent_system_prompt(self) -> str:
        return (
            "你是 ScholarAgent 的学术搜索代理。\n"
            "你的职责是根据用户研究主题，从给定学术检索工具中选择最合适的 1 到 3 个工具执行检索。\n"
            "规则：\n"
            "1. 你必须至少调用一个工具，必要时可以调用多个工具，但不要无意义重复。\n"
            "2. 优先选择覆盖面互补、与主题匹配度高的工具。\n"
            "3. 调用工具时只能使用给定参数：query、max_results、time_range。\n"
            "4. 优先使用提供的 rewritten_queries，不要自行发明过长查询。\n"
            "5. 工具调用完成后，用一句简短中文总结检索策略与主要发现。\n"
            "6. 不要输出伪造论文，不要跳过工具直接编造结果。"
        )

    def _search_agent_user_prompt(
        self,
        *,
        topic: str,
        intent: str,
        time_range: str,
        max_results: int,
        rewritten_queries: List[str],
    ) -> str:
        candidate_queries = (
            "\n".join(f"- {query}" for query in rewritten_queries[:4]) or "- " + topic
        )
        return (
            f"任务意图：{intent}\n"
            f"核心主题：{topic}\n"
            f"时间范围：{time_range or '不限'}\n"
            f"目标论文数：{max_results}\n"
            "候选检索式：\n"
            f"{candidate_queries}\n\n"
            "请根据主题选择最合适的学术搜索工具并执行检索。若主题较广，可组合多个工具；若主题较窄，优先选择最匹配工具。"
        )

    def _parse_agent_result(
        self, result: Dict[str, Any], *, allowed_tool_names: List[str]
    ) -> Dict[str, Any]:
        aggregated: Dict[str, Paper] = {}
        selected_tools: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        agent_summary = ""
        allowed_set = set(allowed_tool_names)
        messages = result.get("messages") or []
        for message in messages:
            if AIMessage is not None and isinstance(message, AIMessage):
                for tool_call in getattr(message, "tool_calls", []) or []:
                    tool_name = str(tool_call.get("name") or "")
                    if (
                        tool_name
                        and tool_name in allowed_set
                        and tool_name not in selected_tools
                    ):
                        selected_tools.append(tool_name)
                content = self._message_text(message)
                if content:
                    agent_summary = content
            if ToolMessage is not None and isinstance(message, ToolMessage):
                artifact = getattr(message, "artifact", None) or {}
                if not isinstance(artifact, dict):
                    continue
                call_record = {
                    "tool_name": str(
                        artifact.get("tool_name") or getattr(message, "name", "") or ""
                    ),
                    "query": str(artifact.get("query") or ""),
                    "max_results": int(artifact.get("max_results") or 0),
                    "time_range": str(artifact.get("time_range") or ""),
                    "count": len(artifact.get("papers") or []),
                }
                tool_calls.append(call_record)
                papers = [
                    self._paper_from_payload(payload)
                    for payload in artifact.get("papers") or []
                    if isinstance(payload, dict)
                ]
                self._merge_papers(aggregated, papers)

        return {
            "aggregated": aggregated,
            "selected_tools": selected_tools,
            "tool_calls": tool_calls,
            "agent_summary": agent_summary,
        }

    def _resolve_agent_final_output(
        self,
        *,
        result: Dict[str, Any],
        provider_name: str,
        topic: str,
        time_range: str,
        rewritten_queries: List[str],
        payload: Dict[str, Any],
        allowed_tool_names: List[str],
    ) -> tuple[SearchAgentFinalOutput, str]:
        structured = result.get("structured_response")
        validated = self._validate_agent_final_output(
            structured, allowed_tool_names=allowed_tool_names
        )
        if validated is not None:
            return validated, "agent_response_format"
        try:
            followup = self.llm.call_structured(
                self._search_agent_followup_prompt(
                    topic=topic,
                    time_range=time_range,
                    rewritten_queries=rewritten_queries,
                    payload=payload,
                ),
                SearchAgentFinalOutput,
                provider=provider_name,
                temperature=0.0,
                max_tokens=900,
                purpose="搜索结果结构化总结",
                budgeted=True,
            )
            validated = self._validate_agent_final_output(
                followup, allowed_tool_names=allowed_tool_names
            )
            if validated is not None:
                return validated, "followup_structured_call"
        except Exception:
            pass
        return (
            self._fallback_agent_final_output(
                selected_tools=payload["selected_tools"],
                tool_calls=payload["tool_calls"],
                aggregated=payload["aggregated"],
                summary_reason="模型未返回可解析 structured response，已按真实工具调用生成结构化摘要。",
            ),
            "deterministic_structured_fallback",
        )

    def _validate_agent_final_output(
        self,
        structured: Any,
        *,
        allowed_tool_names: List[str],
    ) -> SearchAgentFinalOutput | None:
        if structured is None:
            return None
        try:
            if isinstance(structured, SearchAgentFinalOutput):
                output = structured
            elif hasattr(structured, "model_dump"):
                output = SearchAgentFinalOutput.model_validate(
                    structured.model_dump(mode="json")
                )
            else:
                output = SearchAgentFinalOutput.model_validate(structured)
        except Exception:
            return None
        allowed_set = set(allowed_tool_names)
        output.selected_tools = [
            name for name in output.selected_tools if name in allowed_set
        ]
        output.execution_plan = [
            step
            for step in output.execution_plan
            if not step.tool_name or step.tool_name in allowed_set
        ]
        return output

    def _search_agent_followup_prompt(
        self,
        *,
        topic: str,
        time_range: str,
        rewritten_queries: List[str],
        payload: Dict[str, Any],
    ) -> str:
        top_papers = [
            {
                "title": paper.title,
                "year": paper.year,
                "source": paper.source,
                "citations": paper.citations,
            }
            for paper in sorted(
                payload["aggregated"].values(),
                key=lambda item: (item.score, item.citations, item.year or 0),
                reverse=True,
            )[:8]
        ]
        return (
            "你是 ScholarAgent 的搜索结果结构化汇总器。\n"
            "请基于已经真实执行过的检索调用，输出结构化结果，不要补造论文或工具。\n"
            f"核心主题：{topic}\n"
            f"时间范围：{time_range or '不限'}\n"
            f"候选检索式：{json.dumps(rewritten_queries[:4], ensure_ascii=False)}\n"
            f"已选择工具：{json.dumps(payload['selected_tools'], ensure_ascii=False)}\n"
            f"实际工具调用：{json.dumps(payload['tool_calls'], ensure_ascii=False)}\n"
            f"聚合论文摘要：{json.dumps(top_papers, ensure_ascii=False)}\n"
            "要求：\n"
            "1. selected_tools 只能填写已真实使用过的工具。\n"
            "2. execution_plan 要按执行顺序概括调用步骤，每步包含 tool_name、query、purpose。\n"
            "3. aggregation.summary 要概括覆盖范围与主要发现。\n"
            "4. aggregation.key_findings 提炼 3 到 5 条。\n"
            "5. aggregation.representative_titles 只能写真实返回的论文标题。"
        )

    def _fallback_agent_final_output(
        self,
        *,
        selected_tools: List[str],
        tool_calls: List[Dict[str, Any]],
        aggregated: Dict[str, Paper],
        summary_reason: str,
    ) -> SearchAgentFinalOutput:
        ranked_papers = sorted(
            aggregated.values(),
            key=lambda item: (item.score, item.citations, item.year or 0),
            reverse=True,
        )
        execution_plan = [
            SearchAgentExecutionStep(
                tool_name=str(call.get("tool_name") or ""),
                query=str(call.get("query") or ""),
                purpose=(
                    f"使用 {call.get('tool_name') or '检索工具'} 检索主题相关论文"
                    if call.get("tool_name")
                    else "执行检索"
                ),
            )
            for call in tool_calls[:6]
        ]
        key_findings = [
            f"{paper.title} ({paper.year or 'N/A'}, {paper.source}, citations={paper.citations})"
            for paper in ranked_papers[:5]
        ]
        representative_titles = [
            paper.title for paper in ranked_papers[:8] if paper.title
        ]
        if ranked_papers:
            summary = (
                f"{summary_reason} 共聚合 {len(ranked_papers)} 篇候选论文，"
                f"优先结果主要来自 {', '.join(sorted({paper.source for paper in ranked_papers[:5] if paper.source})) or '多个来源'}。"
            )
        else:
            summary = f"{summary_reason} 当前未检索到有效论文结果。"
        return SearchAgentFinalOutput(
            selected_tools=list(dict.fromkeys(name for name in selected_tools if name)),
            tool_selection_reason=summary_reason,
            execution_plan=execution_plan,
            aggregation={
                "summary": summary,
                "key_findings": key_findings,
                "representative_titles": representative_titles,
            },
        )

    def _message_text(self, message: BaseMessage) -> str:  # type: ignore
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(
                part.strip() for part in parts if str(part).strip()
            ).strip()
        return str(content or "").strip()

    def _paper_from_payload(self, payload: Dict[str, Any]) -> Paper:
        return Paper(
            paper_id=str(payload.get("paper_id") or ""),
            title=str(payload.get("title") or ""),
            abstract=str(payload.get("abstract") or ""),
            authors=[str(item) for item in payload.get("authors") or []],
            year=(
                int(payload["year"]) if payload.get("year") not in (None, "") else None
            ),
            venue=str(payload.get("venue") or ""),
            url=str(payload.get("url") or ""),
            pdf_url=str(payload.get("pdf_url") or ""),
            citations=int(payload.get("citations") or 0),
            source=str(payload.get("source") or ""),
            categories=[str(item) for item in payload.get("categories") or []],
            keywords=[str(item) for item in payload.get("keywords") or []],
            score=float(payload.get("score") or 0.0),
            metadata=dict(payload.get("metadata") or {}),
        )


class AnalyzeAgent:
    def __init__(
        self,
        llm: LLMManager,
        templates: PromptTemplateManager,
        tracer: WhiteboxTracer,
        reading_agent: ResearchReadingAgent | None = None,
        memory_agent: ResearchMemoryAgent | None = None,
    ) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer
        self.reading_agent = reading_agent
        self.memory_agent = memory_agent

    def run(
        self,
        papers: List[Paper],
        trace_id: str,
        max_items: int | None = None,
        slots: Dict[str, Any] | None = None,
        user_id: str = "",
    ) -> List[PaperAnalysis]:
        analyses: List[PaperAnalysis] = []
        capped = 5 if max_items is None else max(0, min(max_items, 5))
        selected_papers = papers[:capped]
        resolved_slots = dict(slots or {})
        pdf_path = str(resolved_slots.get("pdf_path") or "")
        focus = str(resolved_slots.get("focus") or "")
        stage_token = self.llm.bind_stage("analyze")
        try:
            for paper in selected_papers:
                deep_context = ""
                if pdf_path and self.reading_agent is not None:
                    reading_payload = self.reading_agent.build_analysis_context(
                        paper,
                        pdf_path=pdf_path,
                        focus=focus,
                        trace_id=trace_id,
                    )
                    sections = reading_payload.get("sections") or []
                    chunks = reading_payload.get("chunks") or []
                    deep_context = "\n".join(
                        [
                            "深度阅读材料：",
                            *[
                                f"- {section.get('heading', 'section')}: {str(section.get('text') or '')[:800]}"
                                for section in sections[:3]
                            ],
                            *[f"- {str(chunk)[:800]}" for chunk in chunks[:4]],
                        ]
                    ).strip()
                prompt = self.templates.render(
                    "paper_analysis",
                    title=paper.title,
                    abstract=paper.abstract,
                    context=(
                        f"来源：{paper.source}, 年份：{paper.year}, 引用数：{paper.citations}"
                        + (f"\n{deep_context}" if deep_context else "")
                    ),
                )
                raw = self.llm.call(prompt, purpose="论文分析", budgeted=True)
                analysis = PaperAnalysis(
                    paper=paper,
                    summary=raw[:500],
                    contributions=self._extract_lines(raw, "贡献"),
                    methods=self._extract_lines(raw, "方法"),
                    findings=self._extract_lines(raw, "发现"),
                    limitations=self._extract_lines(raw, "局限"),
                    raw_analysis=raw,
                )
                analyses.append(analysis)
                if user_id and self.memory_agent is not None:
                    highlights = [*analysis.contributions[:2], *analysis.findings[:2]]
                    self.memory_agent.remember_paper(
                        user_id,
                        paper,
                        analysis.summary,
                        highlights=highlights,
                        trace_id=trace_id,
                    )
        finally:
            self.llm.reset_stage(stage_token)
        self.tracer.trace_step(
            trace_id,
            "analyze",
            {"papers": [paper.title for paper in selected_papers]},
            {
                "count": len(analyses),
                "analysis_limit": capped,
                "deep_read_used": bool(pdf_path),
                "focus": focus,
            },
        )
        return analyses

    def _extract_lines(self, text: str, keyword: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            clean = line.strip("-* 0123456789.")
            if keyword in clean:
                lines.append(clean)
        return lines[:5]


class DebateAgent:
    def __init__(self, reasoning: ReasoningEngine, tracer: WhiteboxTracer) -> None:
        self.reasoning = reasoning
        self.tracer = tracer

    def run(
        self,
        query: str,
        analyses: List[PaperAnalysis],
        trace_id: str,
        task_config: TaskConfig | None = None,
    ) -> DebateResult:
        materials = "\n\n".join(
            f"论文：{item.paper.title}\n摘要：{item.summary}\n贡献：{'；'.join(item.contributions)}"
            for item in analyses
        )
        base_modes = task_config.reasoning_modes if task_config is not None else []
        preferred_modes = list(dict.fromkeys(["debate", *base_modes]))
        result = self.reasoning.reason(
            query,
            materials,
            mode="debate",
            trace_id=trace_id,
            preferred_modes=preferred_modes,
            stage="debate",
        )
        debate = DebateResult(
            question=query,
            thesis="围绕研究问题的多视角综合判断",
            supporting_points=[item.paper.title for item in analyses[:3]],
            counter_points=[item.paper.title for item in analyses[3:5]],
            synthesis=result.answer,
        )
        self.tracer.trace_step(trace_id, "debate", {"query": query}, asdict(debate))
        return debate


class WriteAgent:
    def __init__(
        self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer
    ) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(
        self,
        intent: str,
        query: str,
        research_plan: ResearchPlan | None,
        search_result: SearchResult | None,
        analyses: List[PaperAnalysis],
        debate: DebateResult | None,
        trace_id: str,
    ) -> str:
        if intent == "search_papers" and search_result is not None:
            lines = [f"共找到 {len(search_result.papers)} 篇候选论文："]
            for index, paper in enumerate(search_result.papers[:10], start=1):
                lines.append(
                    f"{index}. {paper.title} ({paper.year or 'N/A'}) | {paper.source} | citations={paper.citations}"
                )
            answer = "\n".join(lines)
        else:
            materials = self._compose_materials(research_plan, search_result, analyses, debate)
            template_name, purpose = self._writer_profile(intent)
            prompt = self.templates.render(
                template_name, topic=query, materials=materials
            )
            stage_token = self.llm.bind_stage("write")
            try:
                answer = self.llm.call(
                    prompt,
                    max_tokens=settings.llm_long_output_max_tokens,
                    purpose=purpose,
                    budgeted=True,
                )
            finally:
                self.llm.reset_stage(stage_token)
        self.tracer.trace_step(
            trace_id, "write", {"intent": intent}, {"answer_preview": answer[:500]}
        )
        return answer

    def _writer_profile(self, intent: str) -> tuple[str, str]:
        profiles = {
            "generate_survey": ("survey_writer", "综述写作"),
            "compare_methods": ("compare_writer", "方法对比写作"),
            "analyze_paper": ("paper_answer_writer", "单篇论文解读"),
            "daily_update": ("daily_update_writer", "研究动态写作"),
            "explain_concept": ("concept_writer", "概念解释写作"),
        }
        return profiles.get(intent, ("survey_writer", "结构化写作"))

    def _compose_materials(
        self,
        research_plan: ResearchPlan | None,
        search_result: SearchResult | None,
        analyses: List[PaperAnalysis],
        debate: DebateResult | None,
    ) -> str:
        parts: List[str] = []
        if research_plan is not None:
            parts.append("研究计划：")
            parts.append(f"- 目标：{research_plan.objective}")
            parts.extend(
                f"- 任务：{task.title} | 交付：{task.deliverable or '未指定'}"
                for task in research_plan.tasks[:6]
            )
        if search_result is not None:
            local_rag = search_result.trace.get("local_rag", {})
            local_results = local_rag.get("results") or []
            if local_results:
                parts.append("本地论文片段：")
                for chunk in local_results[:5]:
                    content = self._chunk_content(chunk)
                    source_type = self._chunk_source_type(chunk)
                    parts.append(f"- [{source_type}] {content[:600]}")
            supplement = local_rag.get("supplement") or []
            if supplement:
                parts.append("\n补充网页片段：")
                for item in supplement[:3]:
                    title = str(item.get("title") or "")
                    snippet = str(item.get("snippet") or "")
                    parts.append(f"- {title}: {snippet[:300]}")
            parts.append("论文列表：")
            parts.extend(
                f"- {paper.title} ({paper.year or 'N/A'}, {paper.source})"
                for paper in search_result.papers[:10]
            )
        if analyses:
            parts.append("\n分析结果：")
            parts.extend(
                f"- {analysis.paper.title}: {analysis.summary}" for analysis in analyses
            )
        if debate is not None:
            parts.append("\n辩论综合：")
            parts.append(debate.synthesis)
        return "\n".join(parts)

    def _chunk_content(self, chunk: Any) -> str:
        if hasattr(chunk, "content"):
            return str(chunk.content or "")
        if isinstance(chunk, dict):
            return str(chunk.get("content") or "")
        return str(chunk or "")

    def _chunk_source_type(self, chunk: Any) -> str:
        if hasattr(chunk, "source_type"):
            return str(chunk.source_type or "chunk")
        if isinstance(chunk, dict):
            return str(chunk.get("source_type") or "chunk")
        return "chunk"


class CoderAgent:
    def __init__(
        self, llm: LLMManager, templates: PromptTemplateManager, tracer: WhiteboxTracer
    ) -> None:
        self.llm = llm
        self.templates = templates
        self.tracer = tracer

    def run(self, query: str, analyses: List[PaperAnalysis], trace_id: str) -> str:
        materials = "\n".join(
            f"- {analysis.paper.title}: {analysis.summary}" for analysis in analyses
        )
        prompt = self.templates.render(
            "code_generation", topic=query, materials=materials
        )
        answer = self.llm.call(prompt, purpose="代码生成", budgeted=True)
        self.tracer.trace_step(
            trace_id, "coder", {"query": query}, {"answer_preview": answer[:500]}
        )
        return answer


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
        self.graph = self._build_graph()

    def uses_langgraph(self) -> bool:
        return self.graph is not None

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
    ) -> Dict[str, Any]:
        flow = self._resolve_flow(intent=intent, mode=mode, task_config=task_config)
        if self.graph is None:
            return self._execute_sequential(
                query=query,
                intent=intent,
                slots=slots,
                trace_id=trace_id,
                task_config=task_config,
                history=history or [],
                session_id=session_id,
                flow=flow,
            )

        result = self.graph.invoke(
            {
                "query": query,
                "intent": intent,
                "slots": slots,
                "mode": mode,
                "session_id": session_id,
                "trace_id": trace_id,
                "task_config": task_config,
                "history": history or [],
                "flow": flow,
                "artifacts": {},
            }
        )
        return dict(result.get("artifacts") or {})

    def _build_graph(self) -> Any:
        if StateGraph is None:
            return None
        graph = StateGraph(MultiAgentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("search", self._search_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("debate", self._debate_node)
        graph.add_node("write", self._write_node)
        graph.add_node("coder", self._coder_node)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "analyze")
        graph.add_edge("analyze", "debate")
        graph.add_edge("debate", "write")
        graph.add_edge("write", "coder")
        graph.add_edge("coder", END)
        return graph.compile()

    def _plan_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "plan" not in state.get("flow", []):
            return {}
        research_plan = self.planner_agent.run(
            state["query"],
            state["intent"],
            state["slots"],
            state["trace_id"],
        )
        return self._state_update(state, research_plan=research_plan)

    def _search_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "search" not in state.get("flow", []):
            return {}
        search_result = self.search_agent.run(
            state["query"],
            state["intent"],
            state["slots"],
            state.get("history", []),
            state["trace_id"],
            session_id=state.get("session_id", ""),
        )
        return self._state_update(state, search_result=search_result)

    def _analyze_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "analyze" not in state.get("flow", []):
            return {}
        search_result = state.get("search_result")
        if search_result is None:
            return {}
        analysis_limit = self._resolve_analysis_limit(
            flow=state["flow"],
            current_index=self._flow_index(state["flow"], "analyze"),
            query=state["query"],
            task_config=state.get("task_config"),
        )
        analyses = self.analyze_agent.run(
            search_result.papers,
            state["trace_id"],
            max_items=analysis_limit,
            slots=state.get("slots") or {},
            user_id=state.get("session_id", ""),
        )
        return self._state_update(state, analyses=analyses)

    def _debate_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "debate" not in state.get("flow", []):
            return {}
        analyses = state.get("analyses") or []
        debate = self.debate_agent.run(
            state["query"],
            analyses,
            state["trace_id"],
            task_config=state.get("task_config"),
        )
        return self._state_update(state, debate=debate)

    def _write_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "write" not in state.get("flow", []):
            return {}
        answer = self.write_agent.run(
            state["intent"],
            state["query"],
            state.get("research_plan"),
            state.get("search_result"),
            state.get("analyses") or [],
            state.get("debate"),
            state["trace_id"],
        )
        return self._state_update(state, answer=answer)

    def _coder_node(self, state: MultiAgentState) -> Dict[str, Any]:
        if "coder" not in state.get("flow", []):
            return {}
        answer = self.coder_agent.run(
            state["query"],
            state.get("analyses") or [],
            state["trace_id"],
        )
        return self._state_update(state, answer=answer)

    def _state_update(self, state: MultiAgentState, **updates: Any) -> Dict[str, Any]:
        artifacts = dict(state.get("artifacts") or {})
        artifacts.update(updates)
        if "answer" in updates and updates["answer"]:
            artifacts["answer"] = updates["answer"]
        return {**updates, "artifacts": artifacts}

    def _execute_sequential(
        self,
        *,
        query: str,
        intent: str,
        slots: Dict[str, Any],
        trace_id: str,
        task_config: TaskConfig | None,
        history: List[Dict[str, str]],
        session_id: str,
        flow: List[str],
    ) -> Dict[str, Any]:
        artifacts: Dict[str, Any] = {}
        research_plan: ResearchPlan | None = None
        search_result: SearchResult | None = None
        analyses: List[PaperAnalysis] = []
        debate: DebateResult | None = None

        for index, step in enumerate(flow):
            if step == "plan":
                research_plan = self.planner_agent.run(query, intent, slots, trace_id)
                artifacts["research_plan"] = research_plan
            elif step == "search":
                search_result = self.search_agent.run(
                    query, intent, slots, history, trace_id, session_id=session_id
                )
                artifacts["search_result"] = search_result
            elif step == "analyze" and search_result is not None:
                analysis_limit = self._resolve_analysis_limit(
                    flow=flow,
                    current_index=index,
                    query=query,
                    task_config=task_config,
                )
                analyses = self.analyze_agent.run(
                    search_result.papers,
                    trace_id,
                    max_items=analysis_limit,
                    slots=slots,
                    user_id=session_id,
                )
                artifacts["analyses"] = analyses
            elif step == "debate":
                debate = self.debate_agent.run(
                    query, analyses, trace_id, task_config=task_config
                )
                artifacts["debate"] = debate
            elif step == "write":
                artifacts["answer"] = self.write_agent.run(
                    intent,
                    query,
                    research_plan,
                    search_result,
                    analyses,
                    debate,
                    trace_id,
                )
            elif step == "coder":
                artifacts["answer"] = self.coder_agent.run(query, analyses, trace_id)

        return artifacts

    def _resolve_flow(
        self,
        *,
        intent: str,
        mode: ExecutionMode,
        task_config: TaskConfig | None,
    ) -> List[str]:
        flow_map = (
            self.intent_flows_fast
            if mode == ExecutionMode.FAST
            else self.intent_flows_full
        )
        if task_config is not None and not task_config.enable_multi_agent:
            flow_map = self.intent_flows_fast
        return list(flow_map.get(intent, ["search", "write"]))

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
