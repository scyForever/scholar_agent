from __future__ import annotations

import json
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Tuple

from config.settings import settings
from src.core.agent_v2 import AgentV2
from src.core.llm import LLMManager
from src.core.models import IndexedChunk, SearchResult
from src.core.structured_outputs import (
    AgentSemanticEvaluationOutput,
    GenerationEvaluationOutput,
    RetrievalEvaluationOutput,
)
from src.evaluation.dataset_builder import write_payloads
from src.preprocessing.query_rewriter import QueryRewriter
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import LocalChromaVectorStore


@dataclass(slots=True)
class RetrievalEvalCase:
    case_id: str
    query: str
    top_k: int
    relevant_doc_ids: List[str]


@dataclass(slots=True)
class GenerationEvalCase:
    case_id: str
    query: str
    context_doc_ids: List[str]
    reference_points: List[Dict[str, Any]]
    forbidden_keywords: List[str]


@dataclass(slots=True)
class AgentEvalCase:
    case_id: str
    query: str
    expected_intent: str
    expected_needs_input: bool
    required_slots: List[str]
    expected_slot_values: Dict[str, Any]
    required_trace_steps: List[str]
    optional_trace_steps: List[str]
    success_keywords: List[str]
    artifact_expectations: Dict[str, Any]
    required_missing_slots: List[str]
    forbidden_trace_steps: List[str]


class ProjectEvaluationRunner:
    def __init__(
        self,
        *,
        corpus_path: Path | None = None,
        retrieval_dataset_path: Path | None = None,
        generation_dataset_path: Path | None = None,
        agent_dataset_path: Path | None = None,
        workspace_root: Path | None = None,
        enable_vector: bool = False,
        enable_reranker: bool = False,
        agent_llm_mode: str = "mock",
        metric_judge_mode: str = "provider",
    ) -> None:
        self.corpus_path = corpus_path or (settings.evaluation_dir / "rag_eval_corpus.json")
        self.retrieval_dataset_path = retrieval_dataset_path or (
            settings.evaluation_dir / "retrieval_eval_dataset.json"
        )
        self.generation_dataset_path = generation_dataset_path or (
            settings.evaluation_dir / "generation_eval_dataset.json"
        )
        self.agent_dataset_path = agent_dataset_path or (settings.evaluation_dir / "agent_eval_dataset.json")
        self.workspace_root = workspace_root or (settings.cache_dir / "evaluation_runtime")
        self.enable_vector = enable_vector
        self.enable_reranker = enable_reranker
        self.agent_llm_mode = agent_llm_mode
        self.metric_judge_mode = metric_judge_mode
        self.metric_judge_llm = LLMManager()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._ensure_eval_assets()

    def run_rag_suite(
        self,
        *,
        answer_source: str = "auto",
        top_k: int | None = None,
        keep_workspace: bool = False,
    ) -> Dict[str, Any]:
        retrieval_report = self.run_retrieval_suite(
            top_k=top_k,
            keep_workspace=keep_workspace,
        )
        generation_report = self.run_generation_suite(
            answer_source=answer_source,
            keep_workspace=keep_workspace,
        )
        summary = {
            "suite": "rag",
            "corpus": str(self.corpus_path),
            "reports": {
                "retrieval": retrieval_report,
                "generation": generation_report,
            },
            "metric_formulas": {
                "retrieval": self._retrieval_metric_formulas(),
                "generation": self._generation_metric_formulas(),
            },
        }
        self._write_report("rag", summary)
        return summary

    def run_retrieval_suite(
        self,
        *,
        top_k: int | None = None,
        keep_workspace: bool = False,
    ) -> Dict[str, Any]:
        workspace = self._reset_workspace("retrieval")
        retriever = self._build_retriever(workspace)
        corpus = self._load_corpus()
        self._index_corpus(retriever, corpus)

        case_reports: List[Dict[str, Any]] = []
        retrieval_cases = self._load_retrieval_cases()
        for case in retrieval_cases:
            effective_top_k = top_k or case.top_k
            retrieval = self._retrieve_chunks(retriever, case.query, effective_top_k)
            case_reports.append(
                self._score_retrieval_case(
                    case=case,
                    retrieved_chunks=retrieval["results"],
                    trace=retrieval["trace"],
                )
            )

        summary = {
            "suite": "retrieval",
            "dataset": str(self.retrieval_dataset_path),
            "corpus": str(self.corpus_path),
            "metric_judge_mode": self._metric_judge_mode_label(),
            "environment": self._rag_environment(retriever),
            "metric_formulas": self._retrieval_metric_formulas(),
            "aggregate": self._aggregate_retrieval_reports(case_reports),
            "cases": case_reports,
        }
        self._write_report("retrieval", summary)
        if not keep_workspace:
            self._cleanup_workspace(workspace)
        return summary

    def run_generation_suite(
        self,
        *,
        answer_source: str = "auto",
        keep_workspace: bool = False,
    ) -> Dict[str, Any]:
        workspace = self._reset_workspace("generation")
        retriever = self._build_retriever(workspace)
        corpus = self._load_corpus()
        self._index_corpus(retriever, corpus)
        corpus_map = {str(item.get("doc_id") or ""): dict(item) for item in corpus}
        llm_manager = None
        resolved_answer_source = "oracle"
        if answer_source in {"auto", "agent", "llm"}:
            llm_manager = self._build_agent_harness(retriever).llm
            if answer_source in {"agent", "llm"}:
                resolved_answer_source = "llm"
            elif llm_manager.get_verified_provider_names():
                resolved_answer_source = "llm"

        case_reports: List[Dict[str, Any]] = []
        generation_cases = self._load_generation_cases()
        for case in generation_cases:
            contexts = [corpus_map.get(doc_id, {}) for doc_id in case.context_doc_ids]
            answer_text = self._resolve_generation_answer(
                case=case,
                context_documents=contexts,
                answer_source=resolved_answer_source,
                llm_manager=llm_manager,
            )
            case_reports.append(
                self._score_generation_case(
                    case=case,
                    answer_text=answer_text,
                    answer_source=resolved_answer_source,
                    context_documents=contexts,
                )
            )

        summary = {
            "suite": "generation",
            "dataset": str(self.generation_dataset_path),
            "corpus": str(self.corpus_path),
            "answer_source": resolved_answer_source,
            "context_source": "gold_documents",
            "metric_judge_mode": self._metric_judge_mode_label(),
            "environment": self._rag_environment(retriever),
            "metric_formulas": self._generation_metric_formulas(),
            "aggregate": self._aggregate_generation_reports(case_reports),
            "cases": case_reports,
        }
        self._write_report("generation", summary)
        if not keep_workspace:
            self._cleanup_workspace(workspace)
        return summary

    def run_agent_suite(
        self,
        *,
        repeats: int = 2,
        keep_workspace: bool = False,
    ) -> Dict[str, Any]:
        workspace = self._reset_workspace("agent")
        retriever = self._build_retriever(workspace)
        corpus = self._load_corpus()
        self._index_corpus(retriever, corpus)
        agent = self._build_agent_harness(retriever)
        semantic_eval_enabled = self.metric_judge_mode != "rule"

        case_reports: List[Dict[str, Any]] = []
        for case in self._load_agent_cases():
            runs: List[Dict[str, Any]] = []
            for repeat_index in range(max(repeats, 1)):
                session_id = f"eval-{case.case_id}-{repeat_index}"
                started = time.perf_counter()
                response = agent.chat(case.query, session_id=session_id)
                latency_ms = (time.perf_counter() - started) * 1000.0
                runs.append(
                    self._score_agent_case(
                        case=case,
                        response=response,
                        latency_ms=latency_ms,
                        semantic_eval_enabled=semantic_eval_enabled,
                    )
                )
            case_reports.append(self._aggregate_agent_case_runs(case, runs))

        summary = {
            "suite": "agent",
            "dataset": str(self.agent_dataset_path),
            "corpus": str(self.corpus_path),
            "semantic_eval_enabled": semantic_eval_enabled,
            "agent_llm_mode": self.agent_llm_mode,
            "metric_judge_mode": self._metric_judge_mode_label(),
            "environment": self._rag_environment(retriever),
            "metric_formulas": self._agent_metric_formulas(),
            "aggregate": self._aggregate_agent_reports(case_reports),
            "cases": case_reports,
        }
        self._write_report("agent", summary)
        if not keep_workspace:
            self._cleanup_workspace(workspace)
        return summary

    def _reset_workspace(self, suite_name: str) -> Path:
        workspace = self.workspace_root / suite_name
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _cleanup_workspace(self, workspace: Path) -> None:
        if workspace.exists():
            shutil.rmtree(workspace)

    def _build_retriever(self, workspace: Path) -> HybridRetriever:
        db_path = workspace / "rag_eval.sqlite"
        retriever = HybridRetriever(db_path=db_path)
        retriever.vector_store = LocalChromaVectorStore(
            storage_path=workspace / "vector_db",
            collection_name="eval_rag_chunks",
            embedder=retriever.embedder,
        )
        retriever.vector_store.clear()
        retriever._evaluation_disable_vector = not self.enable_vector
        retriever._evaluation_vector_reason = "" if self.enable_vector else "disabled_by_default_for_stable_evaluation"
        retriever._evaluation_disable_reranker = not self.enable_reranker
        retriever._evaluation_reranker_reason = "" if self.enable_reranker else "disabled_by_default_for_stable_evaluation"
        def _patched_retrieve(
            bound_retriever: HybridRetriever,
            query: str,
            chat_history: List[Dict[str, str]] | None = None,
            top_k: int | None = None,
            rewritten_queries: List[str] | None = None,
            rewrite_plan: Any | None = None,
        ) -> Dict[str, Any]:
            return self._safe_retrieve_for_agent(
                bound_retriever,
                query,
                chat_history=chat_history,
                top_k=top_k,
                rewritten_queries=rewritten_queries,
                rewrite_plan=rewrite_plan,
            )

        retriever.retrieve = MethodType(_patched_retrieve, retriever)
        return retriever

    def _build_agent_harness(self, retriever: HybridRetriever) -> AgentV2:
        agent = AgentV2()
        agent.set_mode(fast_mode=True, enable_quality_enhance=False)
        agent.retriever = retriever
        agent.retriever.llm = agent.llm
        agent.reasoning.retriever = retriever
        agent.multi_agent.search_agent.retriever = retriever
        agent.multi_agent.search_agent.llm = agent.llm
        agent.multi_agent.search_agent.rewriter = QueryRewriter(agent.llm)
        if self.agent_llm_mode == "mock":
            self._freeze_agent_llm_to_mock(agent)
        return agent

    def _freeze_agent_llm_to_mock(self, agent: AgentV2) -> None:
        mock_provider = agent.llm.providers.get("mock")
        mock_status = agent.llm.provider_status.get("mock")
        if mock_provider is None or mock_status is None:
            return
        agent.llm.providers = {"mock": mock_provider}
        agent.llm.provider_status = {"mock": mock_status}

    def _ensure_eval_assets(self) -> None:
        expected = (
            (self.corpus_path, "documents", 100),
            (self.retrieval_dataset_path, "cases", 100),
            (self.generation_dataset_path, "cases", 100),
            (self.agent_dataset_path, "cases", 100),
        )
        needs_rebuild = False
        for path, key, minimum in expected:
            if not path.exists():
                needs_rebuild = True
                break
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                needs_rebuild = True
                break
            if len(payload.get(key) or []) < minimum:
                needs_rebuild = True
                break
        if needs_rebuild:
            write_payloads(settings.evaluation_dir)

    def _metric_judge_available(self) -> bool:
        if self.metric_judge_mode == "rule":
            return False
        return bool(self.metric_judge_llm.get_healthy_provider_names())

    def _metric_judge_mode_label(self) -> str:
        if self.metric_judge_mode == "rule":
            return "rule_only"
        if self._metric_judge_available():
            return "provider_llm"
        return "rule_fallback"

    def _load_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_corpus(self) -> List[Dict[str, Any]]:
        payload = self._load_json(self.corpus_path)
        return list(payload.get("documents") or [])

    def _load_retrieval_cases(self) -> List[RetrievalEvalCase]:
        payload = self._load_json(self.retrieval_dataset_path)
        cases: List[RetrievalEvalCase] = []
        for item in payload.get("cases") or []:
            cases.append(
                RetrievalEvalCase(
                    case_id=str(item.get("case_id") or ""),
                    query=str(item.get("query") or ""),
                    top_k=int(item.get("top_k") or 4),
                    relevant_doc_ids=[str(value) for value in item.get("relevant_doc_ids") or []],
                )
            )
        return cases

    def _load_generation_cases(self) -> List[GenerationEvalCase]:
        payload = self._load_json(self.generation_dataset_path)
        cases: List[GenerationEvalCase] = []
        for item in payload.get("cases") or []:
            cases.append(
                GenerationEvalCase(
                    case_id=str(item.get("case_id") or ""),
                    query=str(item.get("query") or ""),
                    context_doc_ids=[str(value) for value in item.get("context_doc_ids") or []],
                    reference_points=[dict(point) for point in item.get("reference_points") or []],
                    forbidden_keywords=[str(value) for value in item.get("forbidden_keywords") or []],
                )
            )
        return cases

    def _load_agent_cases(self) -> List[AgentEvalCase]:
        payload = self._load_json(self.agent_dataset_path)
        cases: List[AgentEvalCase] = []
        for item in payload.get("cases") or []:
            cases.append(
                AgentEvalCase(
                    case_id=str(item.get("case_id") or ""),
                    query=str(item.get("query") or ""),
                    expected_intent=str(item.get("expected_intent") or ""),
                    expected_needs_input=bool(item.get("expected_needs_input")),
                    required_slots=[str(value) for value in item.get("required_slots") or []],
                    expected_slot_values=dict(item.get("expected_slot_values") or {}),
                    required_trace_steps=[str(value) for value in item.get("required_trace_steps") or []],
                    optional_trace_steps=[str(value) for value in item.get("optional_trace_steps") or []],
                    success_keywords=[str(value) for value in item.get("success_keywords") or []],
                    artifact_expectations=dict(item.get("artifact_expectations") or {}),
                    required_missing_slots=[str(value) for value in item.get("required_missing_slots") or []],
                    forbidden_trace_steps=[str(value) for value in item.get("forbidden_trace_steps") or []],
                )
            )
        return cases

    def _index_corpus(self, retriever: HybridRetriever, corpus: Iterable[Dict[str, Any]]) -> None:
        vector_enabled, vector_reason = retriever.vector_store.status()
        for item in corpus:
            document_id = str(item.get("doc_id") or "")
            title = str(item.get("title") or "未命名文档")
            text = str(item.get("text") or "")
            metadata = {
                "eval_doc_id": document_id,
                "category": str(item.get("category") or ""),
                "title": title,
            }
            chunks = retriever._chunk_text(text)
            records = [
                *retriever._build_chunk_records(document_id, chunks, "text_chunk", metadata),
                *retriever._build_chunk_records(document_id, retriever._build_kg_triples(chunks), "kg_chunk", metadata),
            ]
            with retriever._connect() as conn:
                conn.execute(
                    "INSERT INTO documents (id, title, metadata, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (
                        document_id,
                        title,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
                retriever._store_chunk_records(conn, records)
                conn.commit()
            if vector_enabled and not bool(getattr(retriever, "_evaluation_disable_vector", False)):
                try:
                    retriever.vector_store.upsert(records)
                except Exception as exc:
                    retriever._evaluation_disable_vector = True
                    retriever._evaluation_vector_reason = f"{type(exc).__name__}: {exc}"
                    vector_enabled = False
                    vector_reason = retriever._evaluation_vector_reason
        if not vector_enabled:
            retriever._evaluation_vector_reason = vector_reason

    def _safe_retrieve_for_agent(
        self,
        retriever: HybridRetriever,
        query: str,
        chat_history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
        rewritten_queries: List[str] | None = None,
        rewrite_plan: Any | None = None,
    ) -> Dict[str, Any]:
        effective_top_k = top_k or settings.rag_top_k
        payload = ProjectEvaluationRunner._retrieve_chunks(
            self,
            retriever,
            query,
            effective_top_k,
        )
        trace = dict(payload.get("trace") or {})
        results = list(payload.get("results") or [])
        return {
            "query": query,
            "enhanced_query": query,
            "rewritten_queries": list(trace.get("rewritten_queries") or rewritten_queries or [query]),
            "routes": list(trace.get("routes") or retriever._route_sources(query)),
            "results": results,
            "supplement": [],
            "trace": {
                "conversation_enhance": query,
                "rewrites": list(trace.get("rewritten_queries") or rewritten_queries or [query]),
                "rewrite_source": "shared_plan" if rewrite_plan is not None else "evaluation_fallback",
                "shared_core_topic": getattr(rewrite_plan, "core_topic", "") if rewrite_plan is not None else "",
                "shared_english_query": getattr(rewrite_plan, "english_query", "") if rewrite_plan is not None else "",
                "routes": list(trace.get("routes") or retriever._route_sources(query)),
                "indexed_chunk_count": retriever._indexed_chunk_count(),
                "vector_db_enabled": bool(trace.get("vector_enabled")),
                "vector_db_reason": str(trace.get("vector_reason") or ""),
                "vector_collection": retriever.vector_store.collection_name,
                "reranker_enabled": bool(trace.get("reranker_enabled")),
                "reranker_reason": str(trace.get("reranker_reason") or ""),
                "lexical_hit_count": int(trace.get("lexical_hit_count") or 0),
                "vector_hit_count": int(trace.get("vector_hit_count") or 0),
                "before_fusion": int(trace.get("lexical_hit_count") or 0) + int(trace.get("vector_hit_count") or 0),
                "after_rerank": len(results),
                "validated_count": int(trace.get("validated_count") or len(results)),
                "evaluation_mode": str(trace.get("mode") or ""),
            },
        }

    def _retrieve_chunks(
        self,
        retriever: HybridRetriever,
        query: str,
        top_k: int,
    ) -> Dict[str, Any]:
        local_queries = QueryRewriter(retriever.llm).plan(query, intent="search_papers").local_queries[:4]
        candidate_queries = [query, *local_queries]
        candidate_queries = list(dict.fromkeys(item for item in candidate_queries if item))[:4]
        routes = retriever._route_sources(query)
        vector_enabled, vector_reason = retriever.vector_store.status()
        if bool(getattr(retriever, "_evaluation_disable_vector", False)):
            vector_enabled = False
            vector_reason = str(getattr(retriever, "_evaluation_vector_reason", "") or "vector_disabled_by_evaluation")
        reranker_enabled, reranker_reason = retriever.reranker.status()
        if bool(getattr(retriever, "_evaluation_disable_reranker", False)):
            reranker_enabled = False
            reranker_reason = str(getattr(retriever, "_evaluation_reranker_reason", "") or "reranker_disabled_by_evaluation")
        ranked_lists: List[List[IndexedChunk]] = []
        lexical_hit_count = 0
        vector_hit_count = 0
        route_traces: List[Dict[str, Any]] = []

        with retriever._connect() as conn:
            for rewritten_query in candidate_queries:
                for source_type in routes:
                    lexical_hits = retriever._retrieve_from_source(conn, rewritten_query, source_type, top_k * 3)
                    semantic_hits: List[IndexedChunk] = []
                    if vector_enabled:
                        try:
                            semantic_hits = retriever.vector_store.search(rewritten_query, source_type, top_k * 3)
                        except Exception as exc:
                            retriever._evaluation_disable_vector = True
                            retriever._evaluation_vector_reason = f"{type(exc).__name__}: {exc}"
                            vector_enabled = False
                            vector_reason = retriever._evaluation_vector_reason
                    if lexical_hits:
                        ranked_lists.append(lexical_hits)
                        lexical_hit_count += len(lexical_hits)
                    if semantic_hits:
                        ranked_lists.append(semantic_hits)
                        vector_hit_count += len(semantic_hits)
                    route_traces.append(
                        {
                            "rewritten_query": rewritten_query,
                            "source_type": source_type,
                            "lexical_hits": len(lexical_hits),
                            "vector_hits": len(semantic_hits),
                        }
                    )

        if not ranked_lists:
            return {
                "results": [],
                "trace": {
                    "rewritten_queries": candidate_queries,
                    "routes": routes,
                    "vector_enabled": vector_enabled,
                    "vector_reason": vector_reason,
                    "reranker_enabled": reranker_enabled,
                    "reranker_reason": reranker_reason,
                    "lexical_hit_count": 0,
                    "vector_hit_count": 0,
                    "mode": "empty",
                },
            }

        fused = retriever._rrf_fusion(ranked_lists) if len(ranked_lists) > 1 else list(ranked_lists[0])
        if reranker_enabled:
            try:
                reranked = retriever._rerank(query, fused)
                retrieval_mode = "hybrid_full" if vector_enabled else "lexical_with_reranker"
            except Exception as exc:
                retriever._evaluation_disable_reranker = True
                retriever._evaluation_reranker_reason = f"{type(exc).__name__}: {exc}"
                reranked = sorted(fused, key=lambda item: item.score, reverse=True)
                retrieval_mode = "hybrid_without_reranker" if vector_enabled else "lexical_only"
        else:
            reranked = sorted(fused, key=lambda item: item.score, reverse=True)
            retrieval_mode = "hybrid_without_reranker" if vector_enabled else "lexical_only"
        validated = [chunk for chunk in reranked if self._chunk_relevant(query, chunk)]
        results = validated[:top_k] if validated else reranked[:top_k]
        return {
            "results": results,
            "trace": {
                "rewritten_queries": candidate_queries,
                "routes": routes,
                "vector_enabled": vector_enabled,
                "vector_reason": vector_reason,
                "reranker_enabled": reranker_enabled,
                "reranker_reason": reranker_reason,
                "lexical_hit_count": lexical_hit_count,
                "vector_hit_count": vector_hit_count,
                "mode": retrieval_mode,
                "parallel_tasks": route_traces,
                "validated_count": len(validated),
            },
        }

    def _chunk_relevant(self, query: str, chunk: IndexedChunk) -> bool:
        query_tokens = self._normalized_tokens(query)
        content_tokens = self._normalized_tokens(chunk.content)
        overlap = len(set(query_tokens).intersection(content_tokens))
        return overlap > 0 or float(chunk.score or 0.0) >= 0.18

    def _resolve_generation_answer(
        self,
        *,
        case: GenerationEvalCase,
        context_documents: List[Dict[str, Any]],
        answer_source: str,
        llm_manager: Any | None,
    ) -> str:
        contexts = "\n".join(str(item.get("text") or "") for item in context_documents)
        if answer_source == "llm" and llm_manager is not None:
            prompt = (
                "请严格基于以下论文材料回答问题，不要引入材料外信息。\n"
                f"问题：{case.query}\n"
                f"材料：\n{contexts}\n"
                "请用中文给出结构化总结。"
            )
            return str(
                llm_manager.call(
                    prompt,
                    purpose="生成评测回答",
                    max_tokens=1200,
                )
            )
        return self._oracle_generation_answer(case, contexts)

    def _oracle_generation_answer(self, case: GenerationEvalCase, contexts: str) -> str:
        supported_points = [
            point["statement"]
            for point in case.reference_points
            if self._fact_supported(point, contexts)
        ]
        if not supported_points:
            return "未从当前上下文中找到足够证据回答该问题。"
        return " ".join(supported_points)

    def _score_retrieval_case(
        self,
        *,
        case: RetrievalEvalCase,
        retrieved_chunks: List[IndexedChunk],
        trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        retrieved_doc_ids = [self._chunk_doc_id(chunk) for chunk in retrieved_chunks]
        retrieved_doc_set = {doc_id for doc_id in retrieved_doc_ids if doc_id}
        relevant_doc_set = set(case.relevant_doc_ids)
        relevant_hits = len(retrieved_doc_set.intersection(relevant_doc_set))
        recall_at_k = relevant_hits / max(len(relevant_doc_set), 1)
        precision_at_k = relevant_hits / max(len(retrieved_chunks), 1)
        rule_context_relevance = sum(
            1.0 for doc_id in retrieved_doc_ids if doc_id in relevant_doc_set
        ) / max(len(retrieved_doc_ids), 1)
        provider_metrics = self._provider_score_retrieval_case(
            case=case,
            retrieved_chunks=retrieved_chunks,
        )
        if provider_metrics is not None:
            context_relevance = provider_metrics["context_relevance"]
            judge_source = "provider_llm"
            judge_details = provider_metrics["details"]
        else:
            context_relevance = round(rule_context_relevance, 4)
            judge_source = "rule"
            judge_details = {}
        rule_metrics = {
            "recall_at_k": round(recall_at_k, 4),
            "precision_at_k": round(precision_at_k, 4),
            "context_relevance": round(rule_context_relevance, 4),
        }

        return {
            "case_id": case.case_id,
            "query": case.query,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_titles": [str(chunk.metadata.get("title") or "") for chunk in retrieved_chunks],
            "metrics": {
                "recall_at_k": round(recall_at_k, 4),
                "precision_at_k": round(precision_at_k, 4),
                "context_relevance": round(context_relevance, 4),
            },
            "rule_metrics": rule_metrics,
            "judge_source": judge_source,
            "judge_details": judge_details,
            "matched_relevant_doc_count": relevant_hits,
            "retrieval_trace": trace,
        }

    def _score_generation_case(
        self,
        *,
        case: GenerationEvalCase,
        answer_text: str,
        answer_source: str,
        context_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        contexts = "\n".join(str(item.get("text") or "") for item in context_documents)
        mentioned_facts = [
            point["fact_id"]
            for point in case.reference_points
            if self._fact_mentioned(point, answer_text)
        ]
        supported_facts = [
            point["fact_id"]
            for point in case.reference_points
            if self._fact_supported(point, contexts)
        ]
        faithful_facts = sorted(set(mentioned_facts).intersection(supported_facts))
        hallucination_hits = [
            keyword for keyword in case.forbidden_keywords if keyword.lower() in answer_text.lower()
        ]

        if mentioned_facts or hallucination_hits:
            faithfulness = len(faithful_facts) / max(len(set(mentioned_facts)) + len(hallucination_hits), 1)
            answer_truthfulness = len(set(mentioned_facts)) / max(len(set(mentioned_facts)) + len(hallucination_hits), 1)
        else:
            faithfulness = 0.0
            answer_truthfulness = 0.0
        answer_relevance = len(set(mentioned_facts)) / max(len(case.reference_points), 1)
        rule_metrics = {
            "faithfulness": round(faithfulness, 4),
            "answer_truthfulness": round(answer_truthfulness, 4),
            "answer_relevance": round(answer_relevance, 4),
        }
        provider_metrics = self._provider_score_generation_case(
            case=case,
            answer_text=answer_text,
            context_documents=context_documents,
        )
        if provider_metrics is not None:
            metrics = provider_metrics["metrics"]
            judge_source = "provider_llm"
            judge_details = provider_metrics["details"]
        else:
            metrics = rule_metrics
            judge_source = "rule"
            judge_details = {}

        return {
            "case_id": case.case_id,
            "query": case.query,
            "answer_source": answer_source,
            "context_doc_ids": list(case.context_doc_ids),
            "context_titles": [str(item.get("title") or "") for item in context_documents],
            "metrics": metrics,
            "rule_metrics": rule_metrics,
            "judge_source": judge_source,
            "judge_details": judge_details,
            "mentioned_facts": mentioned_facts,
            "supported_facts": supported_facts,
            "faithful_facts": faithful_facts,
            "hallucination_hits": hallucination_hits,
            "answer": answer_text,
        }

    def _fact_supported(self, point: Dict[str, Any], context: str) -> bool:
        keywords = [str(item) for item in point.get("keywords") or [] if str(item).strip()]
        lowered = context.lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if not keywords:
            return False
        threshold = 1 if len(keywords) <= 2 else 2
        return matched >= threshold

    def _fact_mentioned(self, point: Dict[str, Any], answer: str) -> bool:
        keywords = [str(item) for item in point.get("keywords") or [] if str(item).strip()]
        lowered = answer.lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
        if not keywords:
            return False
        threshold = 1 if len(keywords) <= 2 else 2
        return matched >= threshold

    def _aggregate_retrieval_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_names = ["recall_at_k", "precision_at_k", "context_relevance"]
        aggregate_metrics = {
            name: round(
                statistics.mean(report["metrics"][name] for report in reports),
                4,
            )
            for name in metric_names
        }
        return {
            "case_count": len(reports),
            "metrics": aggregate_metrics,
            "judge_sources": self._count_judge_sources(reports),
        }

    def _aggregate_generation_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        metric_names = ["faithfulness", "answer_truthfulness", "answer_relevance"]
        aggregate_metrics = {
            name: round(
                statistics.mean(report["metrics"][name] for report in reports),
                4,
            )
            for name in metric_names
        }
        return {
            "case_count": len(reports),
            "metrics": aggregate_metrics,
            "judge_sources": self._count_judge_sources(reports),
        }

    def _count_judge_sources(self, reports: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for report in reports:
            source = str(report.get("judge_source") or "rule")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def _provider_score_generation_case(
        self,
        *,
        case: GenerationEvalCase,
        answer_text: str,
        context_documents: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if self.metric_judge_mode == "rule":
            return None
        if not self._metric_judge_available():
            return None
        prompt = self._generation_judge_prompt(
            case=case,
            answer_text=answer_text,
            context_documents=context_documents,
        )
        try:
            result = self.metric_judge_llm.call_structured(
                prompt,
                GenerationEvaluationOutput,
                purpose="生成评测打分",
                max_tokens=1200,
            )
        except Exception:
            return None
        return {
            "metrics": {
                "faithfulness": round(float(result.faithfulness.score), 4),
                "answer_truthfulness": round(float(result.answer_truthfulness.score), 4),
                "answer_relevance": round(float(result.answer_relevance.score), 4),
            },
            "details": {
                "summary": result.summary,
                "faithfulness_reason": result.faithfulness.reason,
                "answer_truthfulness_reason": result.answer_truthfulness.reason,
                "answer_relevance_reason": result.answer_relevance.reason,
            },
        }

    def _provider_score_retrieval_case(
        self,
        *,
        case: RetrievalEvalCase,
        retrieved_chunks: List[IndexedChunk],
    ) -> Dict[str, Any] | None:
        if self.metric_judge_mode == "rule":
            return None
        if not self._metric_judge_available():
            return None
        if not retrieved_chunks:
            return None
        prompt = self._retrieval_judge_prompt(
            case=case,
            retrieved_chunks=retrieved_chunks,
        )
        try:
            result = self.metric_judge_llm.call_structured(
                prompt,
                RetrievalEvaluationOutput,
                purpose="检索评测打分",
                max_tokens=900,
            )
        except Exception:
            return None
        return {
            "context_relevance": round(float(result.context_relevance.score), 4),
            "details": {
                "summary": result.summary,
                "context_relevance_reason": result.context_relevance.reason,
            },
        }

    def _provider_score_agent_answer(
        self,
        *,
        case: AgentEvalCase,
        answer_text: str,
        trace_steps: List[str],
        slots: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        if self.metric_judge_mode == "rule":
            return None
        if not self._metric_judge_available():
            return None
        prompt = self._agent_judge_prompt(
            case=case,
            answer_text=answer_text,
            trace_steps=trace_steps,
            slots=slots,
        )
        try:
            result = self.metric_judge_llm.call_structured(
                prompt,
                AgentSemanticEvaluationOutput,
                purpose="Agent 语义评测打分",
                max_tokens=900,
            )
        except Exception:
            return None
        return {
            "score": round(float(result.answer_quality.score), 4),
            "reason": result.answer_quality.reason or result.summary,
            "source": "provider_llm",
        }

    def _score_agent_case(
        self,
        *,
        case: AgentEvalCase,
        response: Any,
        latency_ms: float,
        semantic_eval_enabled: bool,
    ) -> Dict[str, Any]:
        trace = dict(response.whitebox or {})
        steps = [dict(step) for step in trace.get("steps") or []]
        step_types = [str(step.get("type") or "") for step in steps]
        slots = dict(response.slots or {})
        answer = str(response.answer or "")

        intent_match = 1.0 if response.intent == case.expected_intent else 0.0
        needs_input_match = 1.0 if bool(response.needs_input) == bool(case.expected_needs_input) else 0.0
        slot_coverage = self._slot_coverage(case.required_slots, slots)
        slot_value_match = self._slot_value_match(case.expected_slot_values, slots)
        trace_step_coverage = self._trace_step_coverage(case.required_trace_steps, step_types)
        forbidden_step_ok = 1.0 if not any(step in step_types for step in case.forbidden_trace_steps) else 0.0
        missing_slot_match = self._missing_slot_match(case.required_missing_slots, steps)
        answer_keyword_coverage = self._keyword_coverage(case.success_keywords, answer)
        artifact_match, artifact_details = self._agent_artifact_match(case.artifact_expectations, response.artifacts)
        semantic_answer = self._provider_score_agent_answer(
            case=case,
            answer_text=answer,
            trace_steps=step_types,
            slots=slots,
        )
        semantic_answer_score = (
            semantic_answer["score"]
            if semantic_answer is not None
            else answer_keyword_coverage
        )
        semantic_answer_source = (
            semantic_answer["source"]
            if semantic_answer is not None
            else "rule"
        )

        components = [
            needs_input_match,
            intent_match,
            slot_coverage,
            slot_value_match,
            trace_step_coverage,
            forbidden_step_ok,
            missing_slot_match,
            artifact_match,
        ]
        if semantic_eval_enabled and case.success_keywords:
            components.append(semantic_answer_score)
        function_match = sum(components) / max(len(components), 1)
        task_success = (
            needs_input_match == 1.0
            and intent_match == 1.0
            and slot_coverage == 1.0
            and slot_value_match == 1.0
            and trace_step_coverage == 1.0
            and forbidden_step_ok == 1.0
            and missing_slot_match == 1.0
            and artifact_match == 1.0
            and (
                not semantic_eval_enabled
                or not case.success_keywords
                or semantic_answer_score >= 0.5
            )
        )

        process_metrics = {
            "intent_match": intent_match,
            "slot_coverage": round(slot_coverage, 4),
            "slot_value_match": round(slot_value_match, 4),
            "trace_step_coverage": round(trace_step_coverage, 4),
            "forbidden_step_ok": forbidden_step_ok,
            "missing_slot_match": round(missing_slot_match, 4),
            "artifact_match": round(artifact_match, 4),
            "answer_keyword_coverage": round(answer_keyword_coverage, 4),
            "trace_step_count": len(step_types),
            "local_hit_count": self._local_hit_count(response.artifacts),
            "constraint_budget_present": self._constraint_budget_present(response.artifacts),
            "evidence_priority_present": self._evidence_priority_present(trace),
            "semantic_answer_score": round(float(semantic_answer_score), 4),
        }

        return {
            "task_success": task_success,
            "function_match": round(function_match, 4),
            "latency_ms": round(latency_ms, 2),
            "process_metrics": process_metrics,
            "answer": answer,
            "intent": response.intent,
            "needs_input": bool(response.needs_input),
            "slots": slots,
            "trace_steps": step_types,
            "artifact_details": artifact_details,
            "semantic_answer": {
                "score": round(float(semantic_answer_score), 4),
                "source": semantic_answer_source,
                "reason": "" if semantic_answer is None else semantic_answer["reason"],
                "keyword_coverage": round(answer_keyword_coverage, 4),
            },
        }

    def _aggregate_agent_case_runs(
        self,
        case: AgentEvalCase,
        runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        success_values = [1.0 if run["task_success"] else 0.0 for run in runs]
        function_values = [float(run["function_match"]) for run in runs]
        latency_values = [float(run["latency_ms"]) for run in runs]
        process_metrics = {
            key: round(
                statistics.mean(float(run["process_metrics"][key]) for run in runs),
                4,
            )
            for key in runs[0]["process_metrics"].keys()
        }
        return {
            "case_id": case.case_id,
            "query": case.query,
            "task_success_rate": round(statistics.mean(success_values), 4),
            "function_match_score": round(statistics.mean(function_values), 4),
            "performance_stability": {
                "repeats": len(runs),
                "success_std": round(statistics.pstdev(success_values), 4) if len(runs) > 1 else 0.0,
                "function_match_std": round(statistics.pstdev(function_values), 4) if len(runs) > 1 else 0.0,
                "latency_mean_ms": round(statistics.mean(latency_values), 2),
                "latency_std_ms": round(statistics.pstdev(latency_values), 2) if len(runs) > 1 else 0.0,
            },
            "process_metrics": process_metrics,
            "runs": runs,
        }

    def _aggregate_agent_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        process_keys = list(reports[0]["process_metrics"].keys()) if reports else []
        return {
            "case_count": len(reports),
            "task_success_rate": round(statistics.mean(report["task_success_rate"] for report in reports), 4) if reports else 0.0,
            "function_match_score": round(statistics.mean(report["function_match_score"] for report in reports), 4) if reports else 0.0,
            "performance_stability": {
                "latency_mean_ms": round(statistics.mean(report["performance_stability"]["latency_mean_ms"] for report in reports), 2) if reports else 0.0,
                "latency_std_ms": round(statistics.mean(report["performance_stability"]["latency_std_ms"] for report in reports), 2) if reports else 0.0,
                "success_std": round(statistics.mean(report["performance_stability"]["success_std"] for report in reports), 4) if reports else 0.0,
                "function_match_std": round(statistics.mean(report["performance_stability"]["function_match_std"] for report in reports), 4) if reports else 0.0,
            },
            "process_metrics": {
                key: round(statistics.mean(report["process_metrics"][key] for report in reports), 4)
                for key in process_keys
            },
            "judge_sources": self._count_judge_sources(
                [
                    {"judge_source": report["runs"][0].get("semantic_answer", {}).get("source", "rule")}
                    for report in reports
                    if report.get("runs")
                ]
            ),
        }

    def _slot_coverage(self, required_slots: List[str], actual_slots: Dict[str, Any]) -> float:
        if not required_slots:
            return 1.0
        filled = sum(1 for slot in required_slots if actual_slots.get(slot))
        return filled / max(len(required_slots), 1)

    def _slot_value_match(self, expected_values: Dict[str, Any], actual_slots: Dict[str, Any]) -> float:
        if not expected_values:
            return 1.0
        matches = 0
        for key, expected in expected_values.items():
            actual = actual_slots.get(key)
            if isinstance(expected, list):
                if list(actual or []) == list(expected):
                    matches += 1
            else:
                if actual == expected:
                    matches += 1
        return matches / max(len(expected_values), 1)

    def _trace_step_coverage(self, required_steps: List[str], actual_steps: List[str]) -> float:
        if not required_steps:
            return 1.0
        matched = sum(1 for step in required_steps if step in actual_steps)
        return matched / max(len(required_steps), 1)

    def _missing_slot_match(self, required_missing_slots: List[str], steps: List[Dict[str, Any]]) -> float:
        if not required_missing_slots:
            return 1.0
        slot_steps = [step for step in steps if step.get("type") == "slots"]
        if not slot_steps:
            return 0.0
        output = slot_steps[-1].get("output") or {}
        missing = [str(item) for item in output.get("missing") or []]
        matched = sum(1 for slot in required_missing_slots if slot in missing)
        return matched / max(len(required_missing_slots), 1)

    def _keyword_coverage(self, keywords: List[str], text: str) -> float:
        if not keywords:
            return 1.0
        lowered = text.lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in lowered)
        return matched / max(len(keywords), 1)

    def _agent_artifact_match(
        self,
        expectations: Dict[str, Any],
        artifacts: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        if not expectations:
            return 1.0, {}
        search_result = artifacts.get("search_result")
        if not isinstance(search_result, SearchResult):
            return 0.0, {"reason": "missing_search_result"}
        trace = dict(search_result.trace or {})
        local_rag = trace.get("local_rag") or {}
        local_hits = len(local_rag.get("results") or [])
        checks = 0
        passed = 0
        details: Dict[str, Any] = {
            "search_mode": trace.get("search_mode"),
            "local_hits": local_hits,
        }
        if "search_mode" in expectations:
            checks += 1
            if trace.get("search_mode") == expectations["search_mode"]:
                passed += 1
        if "min_local_hits" in expectations:
            checks += 1
            if local_hits >= int(expectations["min_local_hits"] or 0):
                passed += 1
        return (passed / max(checks, 1), details)

    def _local_hit_count(self, artifacts: Dict[str, Any]) -> int:
        search_result = artifacts.get("search_result")
        if not isinstance(search_result, SearchResult):
            return 0
        local_rag = (search_result.trace or {}).get("local_rag") or {}
        return len(local_rag.get("results") or [])

    def _constraint_budget_present(self, artifacts: Dict[str, Any]) -> float:
        search_result = artifacts.get("search_result")
        if not isinstance(search_result, SearchResult):
            return 0.0
        return 1.0 if (search_result.trace or {}).get("constraint_budget") else 0.0

    def _evidence_priority_present(self, trace: Dict[str, Any]) -> float:
        for step in trace.get("steps") or []:
            if step.get("type") == "analyze":
                output = step.get("output") or {}
                return 1.0 if output.get("evidence_priority") else 0.0
        return 0.0

    def _chunk_doc_id(self, chunk: IndexedChunk) -> str:
        return str(chunk.metadata.get("eval_doc_id") or "")

    def _normalized_tokens(self, text: str) -> List[str]:
        return [
            token
            for token in (
                str(text or "")
                .lower()
                .replace("：", " ")
                .replace("，", " ")
                .replace("。", " ")
                .replace(",", " ")
                .replace(".", " ")
                .split()
            )
            if token
        ]

    def _generation_judge_prompt(
        self,
        *,
        case: GenerationEvalCase,
        answer_text: str,
        context_documents: List[Dict[str, Any]],
    ) -> str:
        reference_points = "\n".join(
            f"- {point.get('statement', '')}"
            for point in case.reference_points
        )
        contexts = "\n\n".join(
            f"[{item.get('title', '未命名文档')}]\n{item.get('text', '')}"
            for item in context_documents
        )
        forbidden = "、".join(case.forbidden_keywords) if case.forbidden_keywords else "无"
        return (
            "你是 ScholarAgent 的生成评测裁判。请严格根据给定问题、标准上下文、参考事实点和候选回答，"
            "对回答进行结构化评分。评分范围 0 到 1。\n"
            "faithfulness：回答中的关键断言是否能被上下文支持。\n"
            "answer_truthfulness：回答是否避免明显幻觉、错误事实或禁用关键词。\n"
            "answer_relevance：回答是否覆盖问题重点和参考事实点。\n"
            "如果回答明显脱离上下文，faithfulness 和 truthfulness 都应降低。\n"
            f"问题：{case.query}\n"
            f"参考事实点：\n{reference_points}\n"
            f"禁止关键词：{forbidden}\n"
            f"标准上下文：\n{contexts}\n"
            f"候选回答：\n{answer_text}\n"
            "请返回结构化评分。"
        )

    def _retrieval_judge_prompt(
        self,
        *,
        case: RetrievalEvalCase,
        retrieved_chunks: List[IndexedChunk],
    ) -> str:
        contexts = "\n\n".join(
            f"[{index + 1}] 标题：{chunk.metadata.get('title', '未命名文档')}\n"
            f"内容：{chunk.content}"
            for index, chunk in enumerate(retrieved_chunks)
        )
        return (
            "你是 ScholarAgent 的检索评测裁判。请严格根据用户查询和返回的检索上下文，"
            "评估这些上下文对回答查询的整体相关性。评分范围 0 到 1。\n"
            "context_relevance：既考虑主题匹配程度，也考虑返回结果中的噪声比例。"
            "如果大部分结果都紧扣查询，得分应接近 1；如果混入明显偏题结果，得分应降低。\n"
            "请只根据已经返回的上下文打分，不要因为可能存在未返回的更优文档而扣分。\n"
            f"用户查询：{case.query}\n"
            f"返回上下文：\n{contexts}\n"
            "请返回结构化评分。"
        )

    def _agent_judge_prompt(
        self,
        *,
        case: AgentEvalCase,
        answer_text: str,
        trace_steps: List[str],
        slots: Dict[str, Any],
    ) -> str:
        return (
            "你是 ScholarAgent 的 agent 结果评测裁判。请对最终回答质量做 0 到 1 的结构化评分。\n"
            "answer_quality 关注：回答是否真正响应用户任务、是否与预期意图一致、是否体现出应该完成的学术任务。\n"
            "如果回答只是空泛模板、未回应问题重点或任务类型错误，分数应明显降低。\n"
            f"用户问题：{case.query}\n"
            f"预期意图：{case.expected_intent}\n"
            f"抽取槽位：{json.dumps(slots, ensure_ascii=False)}\n"
            f"执行步骤：{', '.join(trace_steps)}\n"
            f"候选回答：\n{answer_text}\n"
            "请返回结构化评分。"
        )

    def _retrieval_metric_formulas(self) -> Dict[str, Dict[str, str]]:
        return {
            "recall_at_k": {
                "description": "检索到的相关文档覆盖率。",
                "formula": "|retrieved_doc_ids ∩ relevant_doc_ids| / max(|relevant_doc_ids|, 1)",
            },
            "precision_at_k": {
                "description": "前 k 个检索结果中真实相关文档的占比。",
                "formula": "|retrieved_doc_ids ∩ relevant_doc_ids| / max(|retrieved_doc_ids|, 1)",
            },
            "context_relevance": {
                "description": "优先由 provider 大模型基于查询和返回上下文判断整体相关性；若 provider 不可用，则回退规则分。",
                "formula": "provider_judge(query, retrieved_contexts).context_relevance.score; fallback=count(doc_id ∈ relevant_doc_ids for doc_id in retrieved_doc_ids) / max(len(retrieved_doc_ids), 1)",
            },
        }

    def _generation_metric_formulas(self) -> Dict[str, Dict[str, str]]:
        return {
            "faithfulness": {
                "description": "优先由 provider 大模型基于金标准上下文判断回答是否忠于证据；若 provider 不可用，则回退规则分。",
                "formula": "provider_judge(query, context_documents, answer, reference_points).faithfulness.score; fallback=|mentioned_facts ∩ supported_facts| / max(|mentioned_facts| + |hallucination_hits|, 1)",
            },
            "answer_truthfulness": {
                "description": "优先由 provider 大模型判断回答是否包含错误事实或幻觉；若 provider 不可用，则回退规则分。",
                "formula": "provider_judge(query, context_documents, answer, reference_points).answer_truthfulness.score; fallback=|mentioned_facts| / max(|mentioned_facts| + |hallucination_hits|, 1)",
            },
            "answer_relevance": {
                "description": "优先由 provider 大模型判断回答对问题与参考事实点的覆盖程度；若 provider 不可用，则回退规则分。",
                "formula": "provider_judge(query, context_documents, answer, reference_points).answer_relevance.score; fallback=|mentioned_facts| / max(|reference_points|, 1)",
            },
        }

    def _agent_metric_formulas(self) -> Dict[str, Dict[str, str]]:
        return {
            "task_success_rate": {
                "description": "任务是否端到端完成的平均成功率。",
                "formula": "mean(1 if run.task_success else 0 for run in runs)",
            },
            "function_match_score": {
                "description": "意图、槽位、trace、artifact 等组件匹配度的平均分。",
                "formula": "mean([needs_input_match, intent_match, slot_coverage, slot_value_match, trace_step_coverage, forbidden_step_ok, missing_slot_match, artifact_match, optional(semantic_answer_score)])",
            },
            "performance_stability.latency_mean_ms": {
                "description": "重复运行的平均时延。",
                "formula": "mean(run.latency_ms for run in runs)",
            },
            "performance_stability.latency_std_ms": {
                "description": "重复运行时延的总体标准差。",
                "formula": "pstdev(run.latency_ms for run in runs)",
            },
            "process_metrics": {
                "description": "过程指标按 case 求均值，包括意图匹配、槽位覆盖、trace 覆盖、本地命中数和约束/证据卡片是否出现。",
                "formula": "mean(case.process_metrics[key] for case in cases)",
            },
            "process_metrics.semantic_answer_score": {
                "description": "使用 provider 大模型对最终回答质量的 0 到 1 评分，失败时回退规则分。",
                "formula": "provider_judge(answer, query, intent, slots, trace_steps).answer_quality.score",
            },
        }

    def _rag_environment(self, retriever: HybridRetriever) -> Dict[str, Any]:
        vector_enabled, vector_reason = retriever.vector_store.status()
        if bool(getattr(retriever, "_evaluation_disable_vector", False)):
            vector_enabled = False
            vector_reason = str(getattr(retriever, "_evaluation_vector_reason", "") or "vector_disabled_by_evaluation")
        reranker_enabled, reranker_reason = retriever.reranker.status()
        if bool(getattr(retriever, "_evaluation_disable_reranker", False)):
            reranker_enabled = False
            reranker_reason = str(getattr(retriever, "_evaluation_reranker_reason", "") or "reranker_disabled_by_evaluation")
        return {
            "vector_enabled": vector_enabled,
            "vector_reason": vector_reason,
            "reranker_enabled": reranker_enabled,
            "reranker_reason": reranker_reason,
        }

    def _write_report(self, suite_name: str, payload: Dict[str, Any]) -> Path:
        report_dir = settings.report_dir / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{suite_name}_report.json"
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return report_path
