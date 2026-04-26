from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, Protocol

from src.core.models import DialogueState, ShortTermMemory


MEMORY_SIGNAL_WORDS = (
    "偏好",
    "关注",
    "需要",
    "不要",
    "研究",
    "论文",
    "方法",
    "数据",
    "结论",
    "格式",
    "时间",
    "范围",
    "主题",
)


class SummaryLLM(Protocol):
    def call(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        response_format: str = "text",
        purpose: str = "",
        budgeted: bool = False,
    ) -> str:
        ...


class DialogueManager:
    def __init__(self, llm: SummaryLLM | None = None) -> None:
        self.llm = llm
        self._states: Dict[str, DialogueState] = defaultdict(DialogueState)
        self.recent_raw_messages = 6
        self.highlight_source_messages = 24
        self.max_highlights = 10
        self.max_summary_chars = 900

    def get_state(self, session_id: str) -> DialogueState:
        return self._states[session_id]

    def update_state(self, session_id: str, **kwargs: object) -> DialogueState:
        state = self._states[session_id]
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    def add_user_message(self, session_id: str, message: str) -> None:
        self._states[session_id].history.append({"role": "user", "content": message})
        self._refresh_short_memory(session_id)

    def add_assistant_message(self, session_id: str, message: str) -> None:
        self._states[session_id].history.append({"role": "assistant", "content": message})
        self._refresh_short_memory(session_id)

    def get_short_memory_context(self, session_id: str) -> str:
        memory = self._states[session_id].short_memory
        sections: list[str] = []
        if memory.raw:
            raw_lines = [f"{item['role']}：{item['content']}" for item in memory.raw]
            sections.append("短期记忆-原文层：\n" + "\n".join(raw_lines))
        if memory.highlights:
            sections.append("短期记忆-重点提炼层：\n" + "\n".join(f"- {item}" for item in memory.highlights))
        if memory.summary:
            sections.append("短期记忆-摘要层：\n" + memory.summary)
        return "\n\n".join(sections)

    def _refresh_short_memory(self, session_id: str) -> None:
        state = self._states[session_id]
        normalized = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or "").strip(),
            }
            for item in state.history
            if str(item.get("content") or "").strip()
        ]
        raw = normalized[-self.recent_raw_messages :]
        older = normalized[: -self.recent_raw_messages]
        highlight_source = normalized[-self.highlight_source_messages :]
        highlights = self._extract_highlights(highlight_source)
        summary_source = older if older else normalized
        summary = self._build_summary(summary_source, highlights)
        metadata = {
            "history_messages": len(normalized),
            "raw_messages": len(raw),
            "older_messages": len(older),
            "highlight_source_messages": len(highlight_source),
        }
        state.short_memory = ShortTermMemory(
            raw=raw,
            highlights=highlights,
            summary=summary,
            metadata=metadata,
        )

    def _extract_highlights(self, raw: list[dict[str, str]]) -> list[str]:
        candidates: list[str] = []
        for item in raw:
            content = item["content"]
            sentences = [part.strip() for part in re.split(r"[\n。！？!?；;]+", content) if part.strip()]
            for sentence in sentences:
                if len(sentence) > 180:
                    sentence = sentence[:177] + "..."
                if any(word in sentence for word in MEMORY_SIGNAL_WORDS):
                    candidates.append(f"{item['role']}：{sentence}")
        deduped: list[str] = []
        seen: set[str] = set()
        for item in reversed(candidates):
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= self.max_highlights:
                break
        return list(reversed(deduped))

    def _build_summary(self, raw: list[dict[str, str]], highlights: list[str]) -> str:
        if not raw:
            return ""
        if self.llm is None:
            return ""

        prompt = self._build_summary_prompt(raw, highlights)
        summary = self.llm.call(
            prompt,
            system_prompt=(
                "你是 ScholarAgent 的短期记忆摘要器。"
                "只基于给定对话生成中文摘要，保留用户目标、约束、偏好、已确认结论和待跟进点。"
                "不要补充外部事实，不要输出标题、JSON、Markdown 代码块或解释。"
            ),
            temperature=0.1,
            max_tokens=320,
            purpose="short_memory_summary",
        ).strip()
        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars - 3] + "..."
        return summary

    def _build_summary_prompt(self, raw: list[dict[str, str]], highlights: list[str]) -> str:
        messages: list[str] = []
        for index, item in enumerate(raw, start=1):
            role = item.get("role", "")
            content = item.get("content", "")
            messages.append(f"{index}. {role}：{content}")

        highlight_text = "\n".join(f"- {item}" for item in highlights) if highlights else "无"
        return (
            "请把下面对话压缩为短期记忆摘要层，供下一轮回答直接使用。\n"
            "输出要求：\n"
            "1. 120 到 300 字，信息密度优先。\n"
            "2. 保留明确约束、用户偏好、任务状态和已经达成的结论。\n"
            "3. 不复述寒暄，不加入原文没有的信息。\n\n"
            f"重点提炼层：\n{highlight_text}\n\n"
            "待摘要对话：\n"
            + "\n".join(messages)
        )
