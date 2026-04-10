from __future__ import annotations

from typing import Any, Dict, List


class RAGQueryPreparationComponent:
    def __init__(self, retriever: Any) -> None:
        self.retriever = retriever

    def prepare(
        self,
        query: str,
        *,
        chat_history: List[Dict[str, str]] | None = None,
        rewritten_queries: List[str] | None = None,
    ) -> Dict[str, Any]:
        enhanced_query = self._conversation_enhance(query, chat_history or [])
        rewrites = list(rewritten_queries or self._rewrite_queries(enhanced_query))[:8]
        routes = self._route_sources(enhanced_query)
        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "rewritten_queries": rewrites,
            "routes": routes,
        }

    def _conversation_enhance(self, query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            return query
        recent = " ".join(item["content"] for item in history[-4:] if item["role"] == "user")
        if recent and query not in recent:
            return f"{recent} {query}"
        return query

    def _rewrite_queries(self, query: str) -> List[str]:
        return self.retriever.rewriter.rewrite(query, intent="search_papers", target="local")[:8]

    def _route_sources(self, query: str) -> List[str]:
        routes = ["text_chunk"]
        lowered = query.lower()
        if any(token in lowered for token in ("表", "table", "benchmark", "指标")):
            routes.append("table_chunk")
        if any(token in lowered for token in ("why", "how", "是什么", "如何", "为什么")):
            routes.append("qa_chunk")
        if any(token in lowered for token in ("关系", "演化", "关联", "graph")):
            routes.append("kg_chunk")
        return list(dict.fromkeys(routes))
