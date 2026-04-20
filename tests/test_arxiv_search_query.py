import unittest
from unittest.mock import Mock, patch

from src.tools.research_search_tool import (
    ArxivAdapter,
    SearchRequest,
    _build_arxiv_search_query,
)


class ArxivSearchQueryTests(unittest.TestCase):
    def test_build_arxiv_search_query_for_plain_text(self) -> None:
        query = "multi-agent reinforcement learning"

        built = _build_arxiv_search_query(query)

        self.assertIn('ti:"multi-agent reinforcement learning"', built)
        self.assertIn('abs:"multi-agent reinforcement learning"', built)
        self.assertIn('all:"multi-agent" AND all:reinforcement AND all:learning', built)
        self.assertNotEqual(built, query)

    def test_build_arxiv_search_query_preserves_advanced_syntax(self) -> None:
        built = _build_arxiv_search_query(
            'cat:cs.LG AND ti:"multi-agent reinforcement learning"',
            author="Yann LeCun",
        )

        self.assertEqual(
            built,
            '(cat:cs.LG AND ti:"multi-agent reinforcement learning") AND au:"Yann LeCun"',
        )

    @patch("src.tools.research_search_tool.requests.get")
    def test_arxiv_adapter_uses_fielded_query_and_relevance_sort(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<feed xmlns='http://www.w3.org/2005/Atom'></feed>"
        )
        mock_get.return_value = mock_response

        adapter = ArxivAdapter()
        adapter.search(SearchRequest(query="multi-agent reinforcement learning", max_results=5))

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["sortBy"], "relevance")
        self.assertEqual(
            kwargs["params"]["search_query"],
            _build_arxiv_search_query("multi-agent reinforcement learning"),
        )


if __name__ == "__main__":
    unittest.main()
