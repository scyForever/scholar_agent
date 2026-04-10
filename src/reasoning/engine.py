from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List

from src.core.llm import LLMManager
from src.core.models import IndexedChunk, Paper, ReasoningResult
from src.rag.retriever import HybridRetriever
from src.tools import TOOL_REGISTRY, TOOL_REGISTRY_HARNESS
from src.tools.contracts import ToolExecutionRequest
from src.whitelist.manager import WhitelistManager
from src.whitebox.tracer import WhiteboxTracer


class ReasoningEngine:
    REACT_MAX_STEPS = 3
    TOT_MAX_DEPTH = 2
    TOT_BRANCH_FACTOR = 2
    TOT_BEAM_WIDTH = 2

    def __init__(
        self,
        llm: LLMManager | None = None,
        tracer: WhiteboxTracer | None = None,
        retriever: HybridRetriever | None = None,
        whitelist: WhitelistManager | None = None,
    ) -> None:
        self.llm = llm or LLMManager()
        self.tracer = tracer or WhiteboxTracer()
        self.retriever = retriever
        self.whitelist = whitelist

    def reason(
        self,
        query: str,
        context: str,
        mode: str = "auto",
        trace_id: str | None = None,
        preferred_modes: Iterable[str] | None = None,
        stage: str | None = None,
    ) -> ReasoningResult:
        methods = {
            "direct": self._direct_answer,
            "cot": self._chain_of_thought,
            "react": self._react_loop,
            "tot": self._tree_of_thought,
            "debate": self._debate_reasoning,
            "reflection": self._reflection_loop,
            "cove": self._chain_of_verification,
        }
        resolved_mode = self._resolve_mode(query, mode, preferred_modes, methods.keys())
        stage_token = self.llm.bind_stage(stage) if stage else None
        try:
            result = methods[resolved_mode](query, context)
        finally:
            if stage_token is not None:
                self.llm.reset_stage(stage_token)
        if trace_id:
            self.tracer.trace_step(
                trace_id,
                f"reasoning:{resolved_mode}",
                {
                    "query": query,
                    "requested_mode": mode,
                    "preferred_modes": list(self._normalize_modes(preferred_modes, methods.keys())),
                },
                asdict(result),
            )
        return result

    def estimate_llm_calls(
        self,
        query: str,
        mode: str = "auto",
        preferred_modes: Iterable[str] | None = None,
    ) -> int:
        methods = {"direct", "cot", "react", "tot", "debate", "reflection", "cove"}
        resolved_mode = self._resolve_mode(query, mode, preferred_modes, methods)
        estimates = {
            "direct": 1,
            "cot": 1,
            "react": self.REACT_MAX_STEPS + 1,
            "tot": 6,
            "debate": 4,
            "reflection": 2,
            "cove": 3,
        }
        return estimates.get(resolved_mode, 1)

    def _resolve_mode(
        self,
        query: str,
        mode: str,
        preferred_modes: Iterable[str] | None,
        supported_modes: Iterable[str],
    ) -> str:
        normalized_supported = set(supported_modes)
        normalized_preferred = self._normalize_modes(preferred_modes, normalized_supported)
        if mode != "auto":
            if mode not in normalized_supported:
                raise ValueError(f"Unsupported reasoning mode: {mode}")
            if normalized_preferred and mode not in normalized_preferred:
                return normalized_preferred[0]
            return mode
        return self._auto_mode(query, normalized_preferred)

    def _normalize_modes(
        self,
        preferred_modes: Iterable[str] | None,
        supported_modes: Iterable[str],
    ) -> List[str]:
        supported = set(supported_modes)
        normalized: List[str] = []
        for mode in preferred_modes or []:
            if mode in supported and mode not in normalized:
                normalized.append(mode)
        return normalized

    def _auto_mode(self, query: str, preferred_modes: Iterable[str] | None = None) -> str:
        allowed = list(preferred_modes or ["direct", "cot", "react", "tot", "debate", "reflection", "cove"])
        lowered = query.lower()
        candidates: List[str] = []

        if any(token in query for token in ("对比", "比较", "区别")):
            candidates.extend(["debate", "cot", "direct"])
        if any(token in query for token in ("综述", "survey", "路线")):
            candidates.extend(["reflection", "cot", "direct"])
        if any(token in query for token in ("实现", "代码", "步骤", "怎么做", "流程", "方案")):
            candidates.extend(["react", "cot", "direct"])
        if any(token in query for token in ("验证", "核实", "检查", "自洽")):
            candidates.extend(["cove", "cot", "direct"])
        if any(token in lowered for token in ("trade-off", "branch", "branches", "path")):
            candidates.extend(["tot", "cot", "direct"])
        if len(query) > 50:
            candidates.extend(["cot", "direct"])
        else:
            candidates.extend(["direct", "cot"])

        for candidate in candidates:
            if candidate in allowed:
                return candidate
        return allowed[0] if allowed else "direct"

    def _direct_answer(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请直接给出结构化回答。",
            purpose="直接回答",
            budgeted=True,
        )
        return ReasoningResult(mode="direct", answer=answer, steps=["整理上下文", "直接回答"], confidence=0.6)

    def _chain_of_thought(self, query: str, context: str) -> ReasoningResult:
        answer = self.llm.call(
            f"问题：{query}\n上下文：{context}\n请先给出简洁的推理步骤，再给出答案。",
            purpose="链式推理",
            budgeted=True,
        )
        return ReasoningResult(mode="cot", answer=answer, steps=["拆解问题", "逐步推理", "形成结论"], confidence=0.72)

    def _react_loop(self, query: str, context: str) -> ReasoningResult:
        tool_specs = self._reasoning_tool_specs()
        if not tool_specs:
            fallback = self._chain_of_thought(query, context)
            return ReasoningResult(
                mode="react",
                answer=fallback.answer,
                steps=["无可用工具，回退到链式推理"],
                confidence=0.52,
                metadata={"fallback_reason": "no_reasoning_tools_available"},
            )

        scratchpad: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        final_answer = ""
        for step_index in range(1, self.REACT_MAX_STEPS + 1):
            prompt = (
                "你正在执行带真实工具调用的 ReAct 推理。请基于问题、已有材料和历史观察，"
                "决定下一步要调用哪个工具，或者直接结束。\n"
                "只输出 JSON，不要输出额外文本。\n"
                '输出格式：{"thought":"...","action":"finish 或工具名","action_input":{"query":"..."},"answer":"当 action=finish 时填写最终答案"}\n'
                "规则：\n"
                "1. 优先调用工具核实关键事实，避免直接臆断。\n"
                "2. action 只能是 finish 或下方工具名。\n"
                "3. 若工具需要 query 参数，必须显式提供。\n"
                "4. 当已有证据足够时，再使用 finish。\n"
                f"问题：{query}\n"
                f"已有材料：{self._bounded_text(context, 3200)}\n"
                f"可用工具：{json.dumps(tool_specs, ensure_ascii=False)}\n"
                f"历史轨迹：{json.dumps(scratchpad, ensure_ascii=False)}"
            )
            decision = self._call_reasoning_json(
                prompt,
                purpose=f"ReAct推理-决策-{step_index}",
                default={"thought": "", "action": "finish", "action_input": {}, "answer": ""},
            )
            thought = str(decision.get("thought") or "").strip()
            action = str(decision.get("action") or "finish").strip()
            action_input = decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {}
            answer = str(decision.get("answer") or "").strip()

            if action == "finish":
                final_answer = answer or self.llm.call(
                    (
                        f"问题：{query}\n"
                        f"已有材料：{self._bounded_text(context, 3200)}\n"
                        f"工具调用轨迹：{json.dumps(scratchpad, ensure_ascii=False)}\n"
                        "请基于这些真实观察给出最终结论，并明确指出依据。"
                    ),
                    purpose="ReAct推理-总结",
                    budgeted=True,
                )
                scratchpad.append(
                    {
                        "step": step_index,
                        "thought": thought,
                        "action": "finish",
                        "action_input": action_input,
                    }
                )
                break

            observation, call_record = self._invoke_reasoning_tool(action, action_input, query=query)
            scratchpad.append(
                {
                    "step": step_index,
                    "thought": thought,
                    "action": action,
                    "action_input": call_record.get("input", {}),
                    "observation": observation,
                }
            )
            tool_calls.append(call_record)

        if not final_answer:
            final_answer = self.llm.call(
                (
                    f"问题：{query}\n"
                    f"已有材料：{self._bounded_text(context, 3200)}\n"
                    f"工具调用轨迹：{json.dumps(scratchpad, ensure_ascii=False)}\n"
                    "请根据上述观察生成最终答案，并明确不确定性。"
                ),
                purpose="ReAct推理-总结",
                budgeted=True,
            )
        steps = [
            f"第{item['step']}轮：{item['action']}"
            for item in scratchpad
            if str(item.get("action") or "").strip()
        ]
        return ReasoningResult(
            mode="react",
            answer=final_answer,
            steps=steps or ["工具观察", "综合结论"],
            confidence=0.76,
            metadata={
                "tool_specs": tool_specs,
                "scratchpad": scratchpad,
                "tool_calls": tool_calls,
            },
        )

    def _tree_of_thought(self, query: str, context: str) -> ReasoningResult:
        frontier: List[Dict[str, Any]] = [{"id": "root", "path": [], "score": 0.0, "parent_id": None}]
        tree_levels: List[Dict[str, Any]] = []
        candidate_counter = 0
        context_excerpt = self._bounded_text(context, 3600)

        for depth in range(1, self.TOT_MAX_DEPTH + 1):
            expanded: List[Dict[str, Any]] = []
            for node in frontier:
                expand_prompt = (
                    "你正在执行树搜索式推理，请围绕当前路径继续扩展多个候选分支。"
                    "只输出 JSON，不要输出额外文本。\n"
                    '格式：{"branches":[{"title":"分支标题","reasoning":"该分支当前的推理与阶段性结论","next_focus":"下一步关注点"}]}\n'
                    f"问题：{query}\n"
                    f"上下文：{context_excerpt}\n"
                    f"当前路径：{json.dumps(node.get('path', []), ensure_ascii=False)}\n"
                    f"请给出 {self.TOT_BRANCH_FACTOR} 个不同的下一步分支。"
                )
                payload = self._call_reasoning_json(
                    expand_prompt,
                    purpose=f"树搜索-扩展-D{depth}",
                    default={"branches": []},
                )
                branches = self._normalize_tot_branches(payload)
                for branch in branches[: self.TOT_BRANCH_FACTOR]:
                    candidate_counter += 1
                    branch_text = str(branch.get("reasoning") or "").strip()
                    candidate = {
                        "id": f"d{depth}_n{candidate_counter}",
                        "parent_id": node.get("id"),
                        "title": str(branch.get("title") or f"分支{candidate_counter}").strip(),
                        "reasoning": branch_text,
                        "next_focus": str(branch.get("next_focus") or "").strip(),
                        "path": [*list(node.get("path") or []), branch_text],
                        "score": 0.0,
                    }
                    expanded.append(candidate)
            if not expanded:
                break

            score_prompt = (
                "请对下面这些树搜索候选分支做评估与剪枝。"
                "只输出 JSON，不要输出额外文本。\n"
                '格式：{"scores":[{"id":"候选ID","score":0.0,"reason":"评分理由"}],"best_id":"最优候选ID"}\n'
                "score 范围为 0 到 1，越高表示越适合继续保留。\n"
                f"问题：{query}\n"
                f"上下文：{context_excerpt}\n"
                f"候选分支：{json.dumps([{k: item[k] for k in ('id', 'title', 'reasoning', 'next_focus', 'path')} for item in expanded], ensure_ascii=False)}"
            )
            scoring = self._call_reasoning_json(
                score_prompt,
                purpose=f"树搜索-评估-D{depth}",
                default={"scores": [], "best_id": ""},
            )
            score_map = self._normalize_tot_scores(scoring)
            for item in expanded:
                item["score"] = score_map.get(item["id"], 0.0)
            expanded.sort(key=lambda item: (float(item.get("score") or 0.0), len(item.get("path") or [])), reverse=True)
            frontier = expanded[: self.TOT_BEAM_WIDTH]
            tree_levels.append(
                {
                    "depth": depth,
                    "expanded": [
                        {
                            "id": item["id"],
                            "parent_id": item["parent_id"],
                            "title": item["title"],
                            "score": item["score"],
                            "reasoning": self._bounded_text(item["reasoning"], 300),
                        }
                        for item in expanded
                    ],
                    "frontier": [item["id"] for item in frontier],
                }
            )

        best_node = frontier[0] if frontier else {"path": [], "score": 0.0, "reasoning": ""}
        answer = self.llm.call(
            (
                f"问题：{query}\n"
                f"上下文：{context_excerpt}\n"
                f"最佳路径：{json.dumps(best_node.get('path', []), ensure_ascii=False)}\n"
                f"保留分支：{json.dumps([{k: item[k] for k in ('id', 'title', 'score', 'path')} for item in frontier], ensure_ascii=False)}\n"
                "请基于最佳路径给出最终答案，并吸收其他保留分支中有价值的补充。"
            ),
            purpose="树状推理-综合",
            budgeted=True,
        )
        return ReasoningResult(
            mode="tot",
            answer=answer,
            steps=["树扩展", "分支评估剪枝", "最佳路径综合"],
            confidence=max(min(float(best_node.get("score") or 0.0), 0.92), 0.45),
            metadata={
                "max_depth": self.TOT_MAX_DEPTH,
                "branch_factor": self.TOT_BRANCH_FACTOR,
                "beam_width": self.TOT_BEAM_WIDTH,
                "tree_levels": tree_levels,
                "best_path": list(best_node.get("path") or []),
            },
        )

    def _debate_reasoning(self, query: str, context: str) -> ReasoningResult:
        context_excerpt = self._bounded_text(context, 3600)
        affirmative = self.llm.call(
            (
                f"问题：{query}\n"
                f"材料：{context_excerpt}\n"
                "你是正方代理。请提出最强支持论证，要求："
                "1. 只基于给定材料；2. 明确核心主张、证据和适用边界；3. 不要泛泛而谈。"
            ),
            system_prompt="你是学术辩论中的正方代理，目标是提出最有证据支撑的支持观点。",
            purpose="多代理辩论-正方立论",
            budgeted=True,
        )
        negative = self.llm.call(
            (
                f"问题：{query}\n"
                f"材料：{context_excerpt}\n"
                f"正方立论：{affirmative}\n"
                "你是反方代理。请系统指出正方论证的漏洞、反例、前提限制和可能的替代解释。"
            ),
            system_prompt="你是学术辩论中的反方代理，目标是识别漏洞、反例和不成立的前提。",
            purpose="多代理辩论-反方质询",
            budgeted=True,
        )
        rebuttal = self.llm.call(
            (
                f"问题：{query}\n"
                f"材料：{context_excerpt}\n"
                f"正方立论：{affirmative}\n"
                f"反方质询：{negative}\n"
                "你是正方复辩代理。请回应最关键的质疑，保留能成立的结论，放弃站不住脚的部分。"
            ),
            system_prompt="你是学术辩论中的正方复辩代理，目标是做有约束的回应，而不是强行胜利。",
            purpose="多代理辩论-正方复辩",
            budgeted=True,
        )
        answer = self.llm.call(
            (
                f"问题：{query}\n"
                f"材料：{context_excerpt}\n"
                f"正方立论：{affirmative}\n"
                f"反方质询：{negative}\n"
                f"正方复辩：{rebuttal}\n"
                "你是主持裁判代理。请综合双方观点，给出："
                "1. 哪些结论证据最充分；2. 哪些结论仍不确定；3. 最终综合判断。"
            ),
            system_prompt="你是严格的学术辩论主持裁判，只能基于双方给出的论据做综合裁决。",
            purpose="多代理辩论-主持总结",
            budgeted=True,
        )
        return ReasoningResult(
            mode="debate",
            answer=answer,
            steps=["正方立论", "反方质询", "正方复辩", "主持裁决"],
            confidence=0.82,
            metadata={
                "agents": [
                    {"role": "affirmative", "message": affirmative},
                    {"role": "negative", "message": negative},
                    {"role": "rebuttal", "message": rebuttal},
                    {"role": "moderator", "message": answer},
                ]
            },
        )

    def _reflection_loop(self, query: str, context: str) -> ReasoningResult:
        draft = self.llm.call(
            f"请先生成问题“{query}”的初稿，材料：{context}",
            purpose="反思推理-初稿",
            budgeted=True,
        )
        final = self.llm.call(
            f"请审阅并优化以下初稿，使其更严谨、更完整：\n{draft}",
            purpose="反思推理-修订",
            budgeted=True,
        )
        return ReasoningResult(mode="reflection", answer=final, steps=["初稿生成", "反思审阅", "优化输出"], confidence=0.8)

    def _chain_of_verification(self, query: str, context: str) -> ReasoningResult:
        draft = self.llm.call(
            f"问题：{query}\n材料：{context}\n先给出一个答案。",
            purpose="验证链-初稿",
            budgeted=True,
        )
        verification = self.llm.call(
            f"请针对以下答案提出3个验证问题并回答：\n{draft}",
            purpose="验证链-验证",
            budgeted=True,
        )
        final = self.llm.call(
            f"请结合原答案与验证结果给出修正版。\n原答案：{draft}\n验证：{verification}",
            purpose="验证链-修订",
            budgeted=True,
        )
        return ReasoningResult(
            mode="cove",
            answer=final,
            steps=["生成答案", "提出验证问题", "修正结论"],
            confidence=0.82,
        )

    def _reasoning_tool_specs(self) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        if self.retriever is not None:
            specs.append(
                {
                    "name": "search_local_rag",
                    "description": "检索本地 RAG 向量库与本地文档片段。",
                    "parameters": [
                        {"name": "query", "type": "str", "required": True},
                        {"name": "top_k", "type": "int", "required": False},
                    ],
                }
            )
        for tool_name in self._allowed_reasoning_tool_names():
            try:
                definition = TOOL_REGISTRY.get_definition(tool_name)
            except KeyError:
                continue
            specs.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": [
                        {
                            "name": parameter.name,
                            "type": parameter.type_name,
                            "required": parameter.required,
                        }
                        for parameter in definition.parameters
                    ],
                }
            )
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in specs:
            name = str(item.get("name") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(item)
        return deduped

    def _allowed_reasoning_tool_names(self) -> List[str]:
        if self.whitelist is None:
            return []
        configured = self.whitelist.allowed_tools("reasoning_agent")
        if not configured:
            configured = self.whitelist.allowed_tools("search_agent")
        return list(dict.fromkeys(name for name in configured if name))

    def _invoke_reasoning_tool(
        self,
        tool_name: str,
        action_input: Dict[str, Any],
        *,
        query: str,
    ) -> tuple[str, Dict[str, Any]]:
        try:
            if tool_name == "search_local_rag":
                if self.retriever is None:
                    raise RuntimeError("local_rag_unavailable")
                tool_query = str(action_input.get("query") or query).strip()
                top_k = self._coerce_int(action_input.get("top_k"), default=3, minimum=1, maximum=5)
                result = self.retriever.retrieve(tool_query, top_k=top_k)
                observation = self._summarize_local_rag_result(result)
                return observation, {
                    "tool_name": tool_name,
                    "input": {"query": tool_query, "top_k": top_k},
                    "observation_preview": self._bounded_text(observation, 500),
                    "result_count": len(result.get("results") or []),
                }

            definition = TOOL_REGISTRY.get_definition(tool_name)
            kwargs = self._build_tool_kwargs(definition.parameters, action_input, fallback_query=query)
            result = TOOL_REGISTRY_HARNESS.execute(
                ToolExecutionRequest(
                    name=tool_name,
                    kwargs=kwargs,
                )
            )
            observation = self._summarize_tool_result(tool_name, result)
            return observation, {
                "tool_name": tool_name,
                "input": kwargs,
                "observation_preview": self._bounded_text(observation, 500),
                "result_count": self._tool_result_count(result),
            }
        except Exception as exc:
            observation = f"{tool_name} 调用失败：{type(exc).__name__}: {exc}"
            return observation, {
                "tool_name": tool_name,
                "input": dict(action_input),
                "error": f"{type(exc).__name__}: {exc}",
                "observation_preview": observation,
                "result_count": 0,
            }

    def _build_tool_kwargs(
        self,
        parameters: Iterable[Any],
        action_input: Dict[str, Any],
        *,
        fallback_query: str,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for parameter in parameters:
            value = action_input.get(parameter.name)
            if value in (None, "") and parameter.name == "query":
                value = fallback_query
            if value in (None, ""):
                if parameter.required:
                    raise ValueError(f"missing_required_parameter:{parameter.name}")
                continue
            kwargs[parameter.name] = self._coerce_parameter_value(parameter.type_name, value)
        return kwargs

    def _coerce_parameter_value(self, type_name: str, value: Any) -> Any:
        normalized = type_name.strip().lower()
        if normalized in {"int", "integer"}:
            return self._coerce_int(value, default=0)
        if normalized in {"float"}:
            return float(value)
        if normalized in {"bool", "boolean"}:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "on"}
        if normalized == "dict":
            return value if isinstance(value, dict) else {}
        if normalized == "list":
            return value if isinstance(value, list) else [value]
        return str(value)

    def _coerce_int(
        self,
        value: Any,
        *,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        try:
            converted = int(value)
        except Exception:
            converted = default
        if minimum is not None:
            converted = max(converted, minimum)
        if maximum is not None:
            converted = min(converted, maximum)
        return converted

    def _summarize_local_rag_result(self, payload: Dict[str, Any]) -> str:
        results = payload.get("results") if isinstance(payload, dict) else []
        supplement = payload.get("supplement") if isinstance(payload, dict) else []
        lines = [f"本地 RAG 命中 {len(results) if isinstance(results, list) else 0} 条结果。"]
        for chunk in list(results or [])[:3]:
            if not isinstance(chunk, IndexedChunk):
                continue
            title = str(chunk.metadata.get("title") or chunk.document_id or "未命名文档")
            lines.append(
                f"- {title} [{chunk.source_type}] score={chunk.score:.3f}: {self._bounded_text(chunk.content, 160)}"
            )
        if supplement:
            lines.append(f"补充 web 结果 {len(supplement)} 条。")
        return "\n".join(lines)

    def _summarize_tool_result(self, tool_name: str, result: Any) -> str:
        if isinstance(result, list):
            if not result:
                return f"{tool_name} 未返回结果。"
            first = result[0]
            if isinstance(first, Paper):
                lines = [f"{tool_name} 返回 {len(result)} 篇论文。"]
                for paper in result[:3]:
                    lines.append(
                        f"- {paper.title} ({paper.year or 'N/A'}) | {paper.source} | citations={paper.citations}"
                    )
                return "\n".join(lines)
            if isinstance(first, IndexedChunk):
                lines = [f"{tool_name} 返回 {len(result)} 个片段。"]
                for chunk in result[:3]:
                    title = str(chunk.metadata.get("title") or chunk.document_id or "未命名文档")
                    lines.append(
                        f"- {title} [{chunk.source_type}] score={chunk.score:.3f}: {self._bounded_text(chunk.content, 160)}"
                    )
                return "\n".join(lines)
            if isinstance(first, dict):
                lines = [f"{tool_name} 返回 {len(result)} 条结构化结果。"]
                for item in result[:3]:
                    title = str(item.get("title") or item.get("name") or item.get("url") or "结果")
                    snippet = self._bounded_text(
                        str(item.get("snippet") or item.get("abstract") or item.get("content") or ""),
                        140,
                    )
                    lines.append(f"- {title}: {snippet}")
                return "\n".join(lines)
            return f"{tool_name} 返回 {len(result)} 条结果：{self._bounded_text(json.dumps(result[:3], ensure_ascii=False), 300)}"
        if isinstance(result, dict):
            return f"{tool_name} 返回结构化结果：{self._bounded_text(json.dumps(result, ensure_ascii=False), 320)}"
        return f"{tool_name} 返回：{self._bounded_text(str(result), 320)}"

    def _tool_result_count(self, result: Any) -> int:
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict):
            if isinstance(result.get("results"), list):
                return len(result["results"])
            return len(result)
        return 1 if result else 0

    def _normalize_tot_branches(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        branches = payload.get("branches") if isinstance(payload, dict) else []
        normalized: List[Dict[str, str]] = []
        if not isinstance(branches, list):
            return normalized
        for item in branches:
            if not isinstance(item, dict):
                continue
            reasoning = str(item.get("reasoning") or "").strip()
            if not reasoning:
                continue
            normalized.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "reasoning": reasoning,
                    "next_focus": str(item.get("next_focus") or "").strip(),
                }
            )
        return normalized

    def _normalize_tot_scores(self, payload: Dict[str, Any]) -> Dict[str, float]:
        scores = payload.get("scores") if isinstance(payload, dict) else []
        normalized: Dict[str, float] = {}
        if not isinstance(scores, list):
            return normalized
        for item in scores:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id") or "").strip()
            if not node_id:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except Exception:
                score = 0.0
            normalized[node_id] = max(0.0, min(score, 1.0))
        return normalized

    def _call_reasoning_json(
        self,
        prompt: str,
        *,
        purpose: str,
        default: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self.llm.call_json(prompt, purpose=purpose, budgeted=True)
        normalized = self._normalize_json_payload(result)
        if normalized:
            return normalized
        return dict(default)

    def _normalize_json_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict) and "raw" not in payload:
            return payload
        raw_text = ""
        if isinstance(payload, dict):
            raw_text = str(payload.get("raw") or "")
        elif isinstance(payload, str):
            raw_text = payload
        raw_text = raw_text.strip()
        if not raw_text:
            return {}
        try:
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _bounded_text(self, text: str, limit: int) -> str:
        content = str(text or "").strip()
        if len(content) <= limit:
            return content
        return f"{content[:limit]}..."
