import unittest

from src.core.models import Paper
from src.tools.research_search_tool import (
    FUSION_SOURCE_COUNT_KEY,
    compute_fusion_score,
    diversify_ranked_papers,
    merge_paper_records,
    register_source_observation,
    _dedupe_papers,
)


class CrossSourceFusionTests(unittest.TestCase):
    def test_dedupe_papers_prioritizes_relevance_over_raw_citations(self) -> None:
        query = "multi-agent reinforcement learning"
        generic_openalex = Paper(
            paper_id="oa-generic",
            title="Learning Dynamics in Complex Systems",
            abstract="This work studies learning systems and optimization under dynamic noise.",
            year=2026,
            citations=6000,
            source="OpenAlex",
        )
        relevant_arxiv = Paper(
            paper_id="ax-relevant",
            title="Multi-Agent Reinforcement Learning for Cooperative Traffic Control",
            abstract="We study multi-agent reinforcement learning for cooperative decision making.",
            year=2025,
            citations=0,
            source="arXiv",
            arxiv_id="2501.12345",
            pdf_url="https://arxiv.org/pdf/2501.12345.pdf",
            open_access=True,
        )
        register_source_observation(generic_openalex, source_name="OpenAlex", source_rank=0)
        register_source_observation(relevant_arxiv, source_name="arXiv", source_rank=0)

        ranked = _dedupe_papers([generic_openalex, relevant_arxiv], query=query)

        self.assertEqual(ranked[0].paper_id, "ax-relevant")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_merge_paper_records_keeps_more_complete_cross_source_record(self) -> None:
        openalex = Paper(
            paper_id="oa-1",
            title="Shared Paper",
            abstract="Short abstract.",
            year=2024,
            citations=120,
            source="OpenAlex",
            doi="10.1000/shared",
            url="https://openalex.org/W1",
        )
        arxiv = Paper(
            paper_id="ax-1",
            title="Shared Paper",
            abstract="A much longer abstract with additional implementation details.",
            year=2024,
            citations=0,
            source="arXiv",
            arxiv_id="2401.00001",
            pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
            full_text_url="https://arxiv.org/pdf/2401.00001.pdf",
            open_access=True,
        )
        register_source_observation(openalex, source_name="OpenAlex", source_rank=0)
        register_source_observation(arxiv, source_name="arXiv", source_rank=1)

        merged = merge_paper_records(openalex, arxiv)

        self.assertEqual(merged.source, "arXiv")
        self.assertEqual(merged.arxiv_id, "2401.00001")
        self.assertTrue(merged.open_access)
        self.assertIn("implementation details", merged.abstract)
        self.assertEqual(merged.metadata[FUSION_SOURCE_COUNT_KEY], 2)
        self.assertIn("OpenAlex", merged.metadata["_fusion_sources"])
        self.assertIn("arXiv", merged.metadata["_fusion_sources"])

    def test_diversify_ranked_papers_avoids_single_source_domination(self) -> None:
        openalex_1 = Paper(paper_id="oa-1", title="OA-1", source="OpenAlex", score=3.10)
        openalex_2 = Paper(paper_id="oa-2", title="OA-2", source="OpenAlex", score=3.02)
        arxiv = Paper(paper_id="ax-1", title="AX-1", source="arXiv", score=2.95)

        reranked = diversify_ranked_papers([openalex_1, openalex_2, arxiv], limit=3)

        self.assertEqual(reranked[0].paper_id, "oa-1")
        self.assertEqual(reranked[1].paper_id, "ax-1")

    def test_compute_fusion_score_rewards_multi_source_support(self) -> None:
        paper = Paper(
            paper_id="shared",
            title="Multi-Agent Reinforcement Learning for Shared Control",
            abstract="This paper studies multi-agent reinforcement learning for shared control.",
            year=2024,
            citations=20,
            source="OpenAlex",
        )
        register_source_observation(paper, source_name="OpenAlex", source_rank=0)
        single_source_score = compute_fusion_score(paper, query="multi-agent reinforcement learning")
        register_source_observation(paper, source_name="arXiv", source_rank=1)
        multi_source_score = compute_fusion_score(paper, query="multi-agent reinforcement learning")

        self.assertGreater(multi_source_score, single_source_score)


if __name__ == "__main__":
    unittest.main()
