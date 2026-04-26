from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.agent_v2 import AgentV2
from src.core.llm import LLMManager
from src.core.models import MemoryType
from src.memory.context_builder import MemoryContextBuilder
from src.memory.manager import MemoryManager
from src.preprocessing.dialogue_manager import DialogueManager
from src.prompt_templates.manager import PromptTemplateManager
from src.tools.registry import TOOL_REGISTRY
from src.whitelist.manager import WhitelistManager
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
    memory.remember_preference("verify_user", "偏好 2023-2024 年的大模型综述")
    assert memory.recall_research_context("verify_user", "大模型", limit=3)

    dialogue = DialogueManager()
    dialogue.add_user_message("verify_session", "我偏好结构化回答，重点关注大模型幻觉治理。")
    dialogue.add_assistant_message("verify_session", "可以，我会按研究问题、方法、结论组织。")
    short_memory = dialogue.get_state("verify_session").short_memory
    assert short_memory.raw
    assert short_memory.highlights
    assert short_memory.summary
    assert short_memory.metadata["raw_messages"] >= 1
    assert "原文层" in dialogue.get_short_memory_context("verify_session")
    context_result = MemoryContextBuilder(max_chars=1200).build(
        short_memory=short_memory,
        long_records=[],
        query="大模型幻觉治理",
    )
    assert context_result.text
    assert context_result.stats["context_chars"] <= 1200

    with TemporaryDirectory() as tmpdir:
        isolated_memory = MemoryManager(Path(tmpdir) / "memory.db")
        isolated_memory.remember_preference("alice", "用户偏好具身智能、多模态大模型和结构化综述")
        isolated_memory.store(
            "bob",
            "用户偏好图神经网络和推荐系统论文",
            memory_type=MemoryType.PREFERENCE,
        )
        alice_recall = isolated_memory.recall("具身智能综述", user_id="alice", limit=3)
        bob_recall = isolated_memory.recall("具身智能综述", user_id="bob", limit=3)
        assert alice_recall
        assert all(item.user_id == "alice" for item in alice_recall)
        assert all(item.user_id == "bob" for item in bob_recall)
        assert "长期记忆-用户专属召回" in isolated_memory.format_recall_context(alice_recall)

    agent = AgentV2()
    status = agent.get_status()
    assert "runtime_graph" in status
    assert "multi_agent_graph" in status
    research_plan = agent.plan_research("写一篇关于大模型幻觉的综述", slots={"time_range": "2023-2024"})
    assert research_plan["tasks"]
    assert research_plan["metadata"]["time_range"] == "2023-2024"
    fetched = agent.fetch_paper(
        "2401.14805",
        identifier_type="arxiv",
        prefer="html",
        download_dir="cache/test_verify",
    )
    assert fetched["available"]
    assert fetched["source"] == "arXiv"

    assert TOOL_REGISTRY.list_tools()
    assert TOOL_REGISTRY.list_langchain_tools(
        names=["search_arxiv", "search_openalex", "search_pubmed", "search_ieee_xplore"]
    )
    pubmed_results = TOOL_REGISTRY.call(
        "search_pubmed",
        query="multi-agent reinforcement learning",
        max_results=2,
        time_range="2023-2024",
    )
    assert isinstance(pubmed_results, list)
    assert TOOL_REGISTRY.get_definition("fetch_paper_asset")
    assert TOOL_REGISTRY.get_definition("parse_pdf_document")
    whitelist = WhitelistManager()
    assert "search_pubmed" in whitelist.allowed_tools("search_agent")
    assert Path("data/memory").exists()
    print("验证完成，核心模块可用。")


if __name__ == "__main__":
    main()
