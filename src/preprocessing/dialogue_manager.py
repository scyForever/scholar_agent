from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict

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


class DialogueManager:
    def __init__(self) -> None:
        self._states: Dict[str, DialogueState] = defaultdict(DialogueState)
        self.max_raw_messages = 12
        self.max_highlights = 8
        self.max_summary_chars = 700

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
            raw_lines = [f"{item['role']}：{item['content']}" for item in memory.raw[-6:]]
            sections.append("短期记忆-原文层：\n" + "\n".join(raw_lines))
        if memory.highlights:
            sections.append("短期记忆-重点提炼层：\n" + "\n".join(f"- {item}" for item in memory.highlights))
        if memory.summary:
            sections.append("短期记忆-摘要层：\n" + memory.summary)
        return "\n\n".join(sections)

    def _refresh_short_memory(self, session_id: str) -> None:
        state = self._states[session_id]
        raw = [
            {
                "role": str(item.get("role") or ""),
                "content": str(item.get("content") or "").strip(),
            }
            for item in state.history[-self.max_raw_messages :]
            if str(item.get("content") or "").strip()
        ]
        highlights = self._extract_highlights(raw)
        state.short_memory = ShortTermMemory(
            raw=raw,
            highlights=highlights,
            summary=self._build_summary(raw, highlights),
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
        if len(candidates) < self.max_highlights:
            for item in raw[-4:]:
                content = item["content"]
                if len(content) > 120:
                    content = content[:117] + "..."
                candidates.append(f"{item['role']}：{content}")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= self.max_highlights:
                break
        return deduped

    def _build_summary(self, raw: list[dict[str, str]], highlights: list[str]) -> str:
        if not raw:
            return ""
        user_messages = [item["content"] for item in raw if item["role"] == "user"]
        assistant_messages = [item["content"] for item in raw if item["role"] == "assistant"]
        parts: list[str] = []
        if user_messages:
            latest_user = user_messages[-1]
            if len(latest_user) > 180:
                latest_user = latest_user[:177] + "..."
            parts.append(f"最近用户诉求：{latest_user}")
        if highlights:
            parts.append("持续关注点：" + "；".join(item.split("：", 1)[-1] for item in highlights[:3]))
        if assistant_messages:
            latest_assistant = assistant_messages[-1]
            if len(latest_assistant) > 220:
                latest_assistant = latest_assistant[:217] + "..."
            parts.append(f"最近系统回应：{latest_assistant}")
        summary = "\n".join(parts)
        if len(summary) > self.max_summary_chars:
            summary = summary[: self.max_summary_chars - 3] + "..."
        return summary
