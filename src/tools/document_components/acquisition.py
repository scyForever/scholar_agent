from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import requests

from src.core.models import PaperAsset
from src.tools.document_components.common import slug


REQUEST_TIMEOUT = 30


class PaperAcquisitionService:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36 ScholarAgent/1.0"
                )
            }
        )

    def fetch(
        self,
        identifier: str,
        *,
        identifier_type: str = "auto",
        prefer: str = "pdf",
        download_dir: str = "",
    ) -> PaperAsset:
        resolved_type = self._resolve_identifier_type(identifier, identifier_type)
        if resolved_type == "arxiv":
            return self._fetch_arxiv(identifier, prefer=prefer, download_dir=download_dir)
        if resolved_type == "doi":
            return self._fetch_doi(identifier, prefer=prefer, download_dir=download_dir)
        if resolved_type == "pmcid":
            return self._fetch_pmc(identifier, prefer=prefer, download_dir=download_dir)
        if resolved_type == "pmid":
            return self._fetch_pubmed(identifier, prefer=prefer, download_dir=download_dir)
        return PaperAsset(
            identifier=identifier,
            source="unknown",
            asset_type=prefer,
            available=False,
            metadata={"reason": f"unsupported_identifier_type:{resolved_type}"},
        )

    def _resolve_identifier_type(self, identifier: str, identifier_type: str) -> str:
        if identifier_type != "auto":
            return identifier_type.strip().lower()
        normalized = identifier.strip()
        if normalized.lower().startswith("10."):
            return "doi"
        if normalized.lower().startswith("pmc"):
            return "pmcid"
        if normalized.isdigit():
            return "pmid"
        return "arxiv"

    def _fetch_arxiv(self, arxiv_id: str, *, prefer: str, download_dir: str) -> PaperAsset:
        normalized = arxiv_id.replace("arXiv:", "").strip()
        pdf_url = f"https://arxiv.org/pdf/{normalized}.pdf"
        html_url = f"https://arxiv.org/abs/{normalized}"
        if prefer == "html":
            content = self._fetch_text(html_url)
            local_path = self._maybe_write_text(download_dir, f"{slug(normalized)}.html", content)
            return PaperAsset(
                identifier=normalized,
                source="arXiv",
                asset_type="html",
                url=html_url,
                local_path=local_path,
                content=content,
                content_type="text/html",
                available=bool(content),
                metadata={"pdf_url": pdf_url},
            )
        local_path = self._download_binary(pdf_url, download_dir, f"{slug(normalized)}.pdf")
        return PaperAsset(
            identifier=normalized,
            source="arXiv",
            asset_type="pdf",
            url=pdf_url,
            local_path=local_path,
            content_type="application/pdf",
            available=bool(local_path),
            metadata={"html_url": html_url},
        )

    def _fetch_doi(self, doi: str, *, prefer: str, download_dir: str) -> PaperAsset:
        normalized = doi.replace("https://doi.org/", "").strip()
        metadata = self._resolve_doi_metadata(normalized)
        if prefer == "html":
            html_url = str(metadata.get("html_url") or f"https://doi.org/{normalized}")
            content = self._fetch_text(html_url)
            local_path = self._maybe_write_text(download_dir, f"{slug(normalized)}.html", content)
            return PaperAsset(
                identifier=normalized,
                source=str(metadata.get("source") or "DOI"),
                asset_type="html",
                url=html_url,
                local_path=local_path,
                content=content,
                content_type="text/html",
                available=bool(content),
                metadata=metadata,
            )
        pdf_url = str(metadata.get("pdf_url") or "")
        if pdf_url:
            local_path = self._download_binary(pdf_url, download_dir, f"{slug(normalized)}.pdf")
            if local_path:
                return PaperAsset(
                    identifier=normalized,
                    source=str(metadata.get("source") or "DOI"),
                    asset_type="pdf",
                    url=pdf_url,
                    local_path=local_path,
                    content_type="application/pdf",
                    available=True,
                    metadata=metadata,
                )
        return PaperAsset(
            identifier=normalized,
            source=str(metadata.get("source") or "DOI"),
            asset_type="pdf",
            url=str(metadata.get("html_url") or f"https://doi.org/{normalized}"),
            content_type="application/pdf",
            available=False,
            metadata={**metadata, "reason": "pdf_not_resolved"},
        )

    def _fetch_pmc(self, pmcid: str, *, prefer: str, download_dir: str) -> PaperAsset:
        normalized = pmcid.upper().strip()
        html_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/"
        pdf_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/pdf/"
        if prefer == "html":
            content = self._fetch_text(html_url)
            local_path = self._maybe_write_text(download_dir, f"{slug(normalized)}.html", content)
            return PaperAsset(
                identifier=normalized,
                source="PMC",
                asset_type="html",
                url=html_url,
                local_path=local_path,
                content=content,
                content_type="text/html",
                available=bool(content),
                metadata={"pdf_url": pdf_url},
            )
        local_path = self._download_binary(pdf_url, download_dir, f"{slug(normalized)}.pdf")
        return PaperAsset(
            identifier=normalized,
            source="PMC",
            asset_type="pdf",
            url=pdf_url,
            local_path=local_path,
            content_type="application/pdf",
            available=bool(local_path),
            metadata={"html_url": html_url},
        )

    def _fetch_pubmed(self, pmid: str, *, prefer: str, download_dir: str) -> PaperAsset:
        html_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        content = self._fetch_text(html_url)
        local_path = self._maybe_write_text(download_dir, f"pubmed-{pmid}.html", content) if prefer == "html" else ""
        return PaperAsset(
            identifier=pmid,
            source="PubMed",
            asset_type="html" if prefer == "html" else prefer,
            url=html_url,
            local_path=local_path,
            content=content if prefer == "html" else "",
            content_type="text/html" if prefer == "html" else "",
            available=bool(content),
            metadata={"reason": "pubmed_only_html_without_pmc"},
        )

    def _resolve_doi_metadata(self, doi: str) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"doi": doi, "html_url": f"https://doi.org/{doi}"}
        try:
            crossref = self.session.get(
                f"https://doi.org/{doi}",
                headers={"Accept": "application/vnd.citationstyles.csl+json", **self.session.headers},
                timeout=REQUEST_TIMEOUT,
            )
            if crossref.ok:
                payload = crossref.json()
                metadata["title"] = payload.get("title", "")
                metadata["source"] = payload.get("container-title", "")
        except Exception:
            pass
        try:
            openalex = self.session.get(
                "https://api.openalex.org/works",
                params={"filter": f"doi:https://doi.org/{doi}", "per-page": 1},
                timeout=REQUEST_TIMEOUT,
            )
            if openalex.ok:
                payload = openalex.json()
                result = ((payload or {}).get("results") or [{}])[0]
                location = result.get("primary_location") or {}
                metadata["source"] = metadata.get("source") or "OpenAlex"
                metadata["html_url"] = str(location.get("landing_page_url") or metadata["html_url"])
                metadata["pdf_url"] = str(location.get("pdf_url") or "")
                metadata["open_access"] = bool((result.get("open_access") or {}).get("is_oa"))
        except Exception:
            pass
        return metadata

    def _fetch_text(self, url: str) -> str:
        if not url:
            return ""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception:
            return ""
        return response.text

    def _download_binary(self, url: str, download_dir: str, filename: str) -> str:
        if not url:
            return ""
        try:
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception:
            return ""
        target_dir = Path(download_dir).expanduser() if download_dir else Path("cache/papers")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_bytes(response.content)
        return str(target_path)

    def _maybe_write_text(self, download_dir: str, filename: str, content: str) -> str:
        if not download_dir or not content:
            return ""
        target_dir = Path(download_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_text(content, encoding="utf-8")
        return str(target_path)
