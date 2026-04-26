import unittest
from typing import Any

from src.preprocessing.dialogue_manager import DialogueManager


class FakeSummaryLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def call(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.response


class DialogueLLMSummaryTests(unittest.TestCase):
    def test_short_memory_summary_uses_llm_output(self) -> None:
        llm = FakeSummaryLLM("用户正在调整 ScholarAgent 的短期记忆，要求摘要层改为 LLM 摘要。")
        dialogue = DialogueManager(llm)

        dialogue.add_user_message("s1", "请将摘要层替换为 LLM 摘要，需要保持原文层和重点提炼层不变。")

        memory = dialogue.get_state("s1").short_memory
        self.assertEqual(memory.summary, llm.response)
        self.assertEqual(len(llm.calls), 1)
        self.assertIn("短期记忆摘要层", llm.calls[0]["prompt"])
        self.assertIn("请将摘要层替换为 LLM 摘要", llm.calls[0]["prompt"])
        self.assertEqual(llm.calls[0]["purpose"], "short_memory_summary")

    def test_summary_is_empty_without_llm(self) -> None:
        dialogue = DialogueManager()

        dialogue.add_user_message("s1", "记录一条对话。")

        self.assertEqual(dialogue.get_state("s1").short_memory.summary, "")


if __name__ == "__main__":
    unittest.main()
