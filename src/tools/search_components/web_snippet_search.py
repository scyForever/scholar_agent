from __future__ import annotations

from typing import Dict, List

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


class WebSnippetSearchComponent:
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        if BeautifulSoup is None:
            return []
        try:
            response = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            response.raise_for_status()
        except Exception:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[Dict[str, str]] = []
        for item in soup.select(".result")[:max_results]:
            title_node = item.select_one(".result__title")
            snippet_node = item.select_one(".result__snippet")
            link_node = item.select_one(".result__url")
            results.append(
                {
                    "title": title_node.get_text(" ", strip=True) if title_node else "",
                    "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    "url": link_node.get_text(" ", strip=True) if link_node else "",
                }
            )
        return results
