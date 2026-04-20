from __future__ import annotations

import argparse
import json

from src.evaluation import ProjectEvaluationRunner
from src.evaluation.dataset_builder import write_payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ScholarAgent 自建评测集。")
    parser.add_argument(
        "--suite",
        choices=["retrieval", "generation", "rag", "agent", "all"],
        default="all",
        help="选择评测套件。",
    )
    parser.add_argument(
        "--answer-source",
        choices=["auto", "agent", "llm", "oracle"],
        default="auto",
        help="生成评测答案来源。auto 会在有真实 provider 时使用 llm，否则退回 oracle；agent 为兼容旧参数，等价于 llm。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="覆盖 RAG 数据集里的默认 top-k，0 表示按数据集配置执行。",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Agent 评测重复次数，用于稳定性统计。",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="保留本次评测临时索引目录，便于排查问题。",
    )
    parser.add_argument(
        "--enable-vector",
        action="store_true",
        help="启用向量检索评测。默认关闭，以避免在资源紧张环境下触发 OOM。",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="启用 reranker 评测。默认关闭，通常与 --enable-vector 一起使用。",
    )
    parser.add_argument(
        "--agent-llm-mode",
        choices=["mock", "auto"],
        default="mock",
        help="Agent 评测中的写作与分析阶段是否强制使用 mock LLM。默认 mock，保证评测稳定可复现。",
    )
    parser.add_argument(
        "--metric-judge-mode",
        choices=["provider", "rule", "auto"],
        default="provider",
        help="指标评分方式。provider 优先使用配置好的大模型 API 打分，失败时回退规则；rule 只用规则；auto 在存在健康 provider 时使用大模型。",
    )
    parser.add_argument(
        "--rebuild-datasets",
        action="store_true",
        help="在运行评测前重建 data/evaluation 下的 100+ 语料和 case 文件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_datasets:
        write_payloads()
    runner = ProjectEvaluationRunner(
        enable_vector=args.enable_vector,
        enable_reranker=args.enable_reranker,
        agent_llm_mode=args.agent_llm_mode,
        metric_judge_mode=args.metric_judge_mode,
    )
    outputs = []

    if args.suite in {"retrieval", "all"}:
        outputs.append(
            runner.run_retrieval_suite(
                top_k=args.top_k or None,
                keep_workspace=args.keep_workspace,
            )
        )
    if args.suite in {"generation", "all"}:
        outputs.append(
            runner.run_generation_suite(
                answer_source=args.answer_source,
                keep_workspace=args.keep_workspace,
            )
        )
    if args.suite == "rag":
        outputs.append(
            runner.run_rag_suite(
                answer_source=args.answer_source,
                top_k=args.top_k or None,
                keep_workspace=args.keep_workspace,
            )
        )
    if args.suite in {"agent", "all"}:
        outputs.append(
            runner.run_agent_suite(
                repeats=args.repeats,
                keep_workspace=args.keep_workspace,
            )
        )

    if args.suite == "all":
        payload = {"reports": outputs}
    else:
        payload = outputs[0]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
