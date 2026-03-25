from __future__ import annotations

from pathlib import Path

from src.core.agent_v2 import AgentV2
from src.core.llm import LLMManager
from src.memory.manager import MemoryManager
from src.prompt_templates.manager import PromptTemplateManager
from src.tools.registry import TOOL_REGISTRY
from src.whitebox.tracer import WhiteboxTracer


def main() -> None:
    print("开始验证 ScholarAgent 核心功能...")

    llm = LLMManager()
    assert "mock" in llm.get_status()

    prompts = PromptTemplateManager()
    prompts.ensure_default_templates()
    assert list(prompts.list_templates())

    tracer = WhiteboxTracer()
    trace_id = tracer.start_trace("verify", "验证流程")
    tracer.trace_step(trace_id, "ping", {"message": "hello"}, {"status": "ok"})
    tracer.finish_trace(trace_id, {"result": "done"})
    assert tracer.get_trace(trace_id)["status"] == "completed"

    memory = MemoryManager()
    memory.store("verify_user", "用户偏好是结构化回答", importance=0.8)
    assert memory.recall("结构化回答", limit=3)

    agent = AgentV2()
    response = agent.chat("搜索近三年关于多智能体强化学习的论文", session_id="verify")
    assert response.answer

    assert TOOL_REGISTRY.list_tools()
    assert Path("data/memory").exists()
    print("验证完成，核心模块可用。")


if __name__ == "__main__":
    main()
