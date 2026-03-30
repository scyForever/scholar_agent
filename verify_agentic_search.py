from __future__ import annotations

import argparse
import json
import time

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from src.core.agent_v2 import AgentV2
from src.core.structured_outputs import SearchAgentFinalOutput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 LangChain agent 搜索路径与真实 provider 可用性。")
    parser.add_argument(
        "--query",
        default="搜索近三年关于多智能体强化学习的论文",
        help="用于验证的搜索请求。",
    )
    parser.add_argument(
        "--topic",
        default="多智能体强化学习",
        help="传给搜索节点的核心主题。",
    )
    parser.add_argument(
        "--time-range",
        default="近三年",
        help="传给搜索节点的时间范围。",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=8,
        help="搜索节点聚合的论文上限。",
    )
    parser.add_argument(
        "--tools",
        default="search_arxiv,search_openalex",
        help="用于验证的工具列表，逗号分隔。",
    )
    return parser.parse_args()


class _ProbeToolInput(BaseModel):
    query: str = Field(description="用于验证 tool-calling 的测试查询。")


def _provider_supports_search_agent(agent: AgentV2, provider_name: str) -> bool:
    model = agent.llm.create_langchain_chat_model(
        provider_name,
        purpose="搜索 agent 能力预检",
        temperature=0.0,
        max_tokens=400,
    )

    def _probe_tool(query: str) -> tuple[str, dict[str, object]]:
        return (
            f"已完成本地测试检索：{query}",
            {
                "tool_name": "search_arxiv",
                "query": query,
                "max_results": 1,
                "time_range": "",
                "papers": [
                    {
                        "paper_id": "probe-paper",
                        "title": "Probe Paper",
                        "year": 2024,
                        "source": "probe",
                        "citations": 1,
                    }
                ],
            },
        )

    tool = StructuredTool.from_function(
        func=_probe_tool,
        name="search_arxiv",
        description="用于验证 provider 是否支持 LangChain agent tool-calling 的本地测试工具。",
        args_schema=_ProbeToolInput,
        infer_schema=False,
        response_format="content_and_artifact",
    )
    probe_agent = create_agent(
        model=model,
        tools=[tool],
        system_prompt=(
            "你是搜索 agent 预检器。"
            "你必须调用一次 search_arxiv，然后返回结构化结果。"
        ),
        response_format=SearchAgentFinalOutput,
        name="search_agent_probe",
    )
    result = probe_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请调用 search_arxiv 检索 multi-agent reinforcement learning，并返回 structured response。",
                }
            ]
        },
        config={"recursion_limit": 6},
    )
    structured = result.get("structured_response")
    return bool(structured)


def pick_real_provider(agent: AgentV2) -> str:
    for provider_name in agent.llm.get_healthy_provider_names():
        try:
            started = time.perf_counter()
            if _provider_supports_search_agent(agent, provider_name):
                elapsed = time.perf_counter() - started
                print(f"[provider-check] {provider_name} 通过搜索 agent 能力预检，耗时 {elapsed:.2f}s")
                return provider_name
        except Exception as exc:
            agent.llm.record_provider_failure(provider_name)
            print(f"[provider-check] {provider_name} 失败: {type(exc).__name__}: {exc}")
    raise RuntimeError("没有找到可用的真实 provider。")


def force_provider(agent: AgentV2, provider_name: str) -> None:
    for name, status in agent.llm.provider_status.items():
        if name in {"mock", provider_name}:
            continue
        status.available = False


def main() -> None:
    args = parse_args()
    agent = AgentV2()
    configured = [name for name in agent.llm.providers if name != "mock"]
    print("已配置真实 provider:", configured)
    provider_name = pick_real_provider(agent)
    print("首个可用 provider:", provider_name)
    force_provider(agent, provider_name)
    print("本次验证强制使用 provider:", provider_name)

    trace_id = agent.tracer.start_trace("verify-agentic-search", args.query)
    search_agent = agent.multi_agent.search_agent
    print("开始查询改写...")
    rewrite_plan = search_agent.rewriter.plan(args.topic, intent="search_papers")
    print("查询改写完成。")
    preferred_tools = [item.strip() for item in args.tools.split(",") if item.strip()]
    whitelist_tools = set(agent.whitelist.allowed_tools("search_agent"))
    allowed_tool_names = [tool_name for tool_name in preferred_tools if tool_name in whitelist_tools]
    if not allowed_tool_names:
        raise RuntimeError("验证脚本未找到可用的白名单搜索工具。")
    print("本次验证工具:", allowed_tool_names)
    probe_query = rewrite_plan.external_queries[0] if rewrite_plan.external_queries else args.topic
    for tool_name in allowed_tool_names:
        started = time.perf_counter()
        papers = search_agent._invoke_search_tool(
            tool_name,
            query=probe_query,
            max_results=1,
            time_range=args.time_range,
            use_langchain=False,
        )
        elapsed = time.perf_counter() - started
        print(f"[tool-check] {tool_name} 返回 {len(papers)} 条结果，耗时 {elapsed:.2f}s")
    print("开始执行 LangChain agent 搜索...")
    payload = search_agent._run_external_search(
        topic=args.topic,
        intent="search_papers",
        time_range=args.time_range,
        max_results=args.max_papers,
        rewritten_queries=rewrite_plan.external_queries,
        allowed_tool_names=allowed_tool_names,
    )
    print("LangChain agent 搜索执行完成。")
    papers = sorted(
        payload["aggregated"].values(),
        key=lambda item: (item.score, item.citations, item.year or 0),
        reverse=True,
    )
    summary = {
        "rewritten_queries": rewrite_plan.external_queries,
        "tool_strategy": payload.get("tool_strategy"),
        "agent_selected_tools": payload.get("selected_tools"),
        "agent_tool_calls": payload.get("tool_calls"),
        "agent_output_source": payload.get("final_output_source"),
        "agent_final_output": payload.get("final_output"),
        "agent_provider_attempts": payload.get("provider_attempts"),
        "agent_errors": payload.get("agent_errors"),
        "total_found": len(payload.get("aggregated") or {}),
        "papers": [
            {
                "title": paper.title,
                "year": paper.year,
                "source": paper.source,
                "citations": paper.citations,
            }
            for paper in papers[:5]
        ],
    }
    agent.tracer.finish_trace(trace_id, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if payload.get("tool_strategy") != "langchain_agent":
        raise SystemExit("验证失败：搜索节点未命中 LangChain agent 路径。")
    if not payload.get("selected_tools"):
        raise SystemExit("验证失败：agent_selected_tools 为空。")
    if not payload.get("final_output"):
        raise SystemExit("验证失败：缺少结构化最终输出。")


if __name__ == "__main__":
    main()
