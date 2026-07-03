# tools/search_tool.py
"""
PubMed search tool.
Responsibility: Search PubMed for medical literature.
Depends on: tools/base_tool.py, config/settings.py
No API key required — PubMed is free and public.
"""

import asyncio
import logging
import requests
from typing import Any, Dict, List, Optional

from tools.base_tool import BaseTool

logger = logging.getLogger("uvicorn.error")

__all__ = ["SearchTool"]

# -----------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------

_PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "ClinicalTrialMatcher/1.0 (research@eisai.com)",
}

_MAX_RESULTS = 5      # Max papers to fetch per query
_MAX_ABSTRACT_CHARS = 800   # Truncate abstracts for token efficiency


# -----------------------------------------------------------
# SEARCH TOOL
# -----------------------------------------------------------

class SearchTool(BaseTool):
    """
    PubMed medical literature search tool.

    Searches PubMed for relevant papers based on a query.
    Returns paper titles, years, and abstracts.
    No API key required.
    """

    @property
    def name(self) -> str:
        return "pubmed_search"

    @property
    def description(self) -> str:
        return (
            "Searches PubMed for peer-reviewed medical literature. "
            "Use for finding research on cancer treatments, biomarkers, "
            "and clinical trial evidence."
        )

    # -----------------------------------------------------------
    # INTERNAL HELPERS
    # -----------------------------------------------------------

    def _search_pubmed_ids(
        self,
        query: str,
        max_results: int = _MAX_RESULTS,
    ) -> List[str]:
        """
        Step 1 — Search PubMed for paper IDs matching the query.
        Returns list of PubMed IDs (PMIDs).
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }

        try:
            response = requests.get(
                _PUBMED_SEARCH_URL,
                params=params,
                headers=_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(
                "PubMed search | query='%s' | found=%d IDs",
                query, len(ids),
            )
            return ids
        except Exception:
            logger.exception("PubMed ID search failed for query: '%s'", query)
            return []

    def _fetch_paper_summaries(
        self,
        pmids: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Step 2 — Fetch paper summaries (title, year, authors)
        for the given PubMed IDs.
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }

        try:
            response = requests.get(
                _PUBMED_SUMMARY_URL,
                params=params,
                headers=_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            result_data = data.get("result", {})

            papers = []
            for pmid in pmids:
                paper = result_data.get(pmid, {})
                if not paper:
                    continue
                papers.append({
                    "pmid": pmid,
                    "title": paper.get("title", "Unknown Title"),
                    "year": paper.get("pubdate", "")[:4],
                    "authors": ", ".join(
                        a.get("name", "")
                        for a in (paper.get("authors") or [])[:3]
                    ),
                    "journal": paper.get("source", ""),
                })

            logger.info(
                "PubMed summaries fetched | count=%d", len(papers)
            )
            return papers

        except Exception:
            logger.exception("PubMed summary fetch failed for PMIDs: %s", pmids)
            return []

    def _fetch_abstracts(
        self,
        pmids: List[str],
    ) -> Dict[str, str]:
        """
        Step 3 — Fetch full abstracts for the given PubMed IDs.
        Returns dict of {pmid: abstract_text}.
        """
        if not pmids:
            return {}

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text",
        }

        try:
            response = requests.get(
                _PUBMED_FETCH_URL,
                params=params,
                headers=_HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            raw_text = response.text

            # PubMed returns all abstracts concatenated
            # Split by PMID markers
            abstracts: Dict[str, str] = {}
            current_pmid = None
            current_lines = []

            for line in raw_text.split("\n"):
                # PubMed abstract blocks start with PMID
                for pmid in pmids:
                    if line.strip().startswith(f"PMID- {pmid}"):
                        if current_pmid and current_lines:
                            abstracts[current_pmid] = " ".join(
                                current_lines
                            )[:_MAX_ABSTRACT_CHARS]
                        current_pmid = pmid
                        current_lines = []
                        break

                if line.startswith("AB  -") or (
                    current_pmid and line.startswith("      ")
                ):
                    cleaned = line.replace("AB  -", "").strip()
                    if cleaned:
                        current_lines.append(cleaned)

            # Save last paper
            if current_pmid and current_lines:
                abstracts[current_pmid] = " ".join(
                    current_lines
                )[:_MAX_ABSTRACT_CHARS]

            logger.info(
                "PubMed abstracts fetched | count=%d", len(abstracts)
            )
            return abstracts

        except Exception:
            logger.exception("PubMed abstract fetch failed")
            return {}

    # -----------------------------------------------------------
    # PUBLIC EXECUTE
    # -----------------------------------------------------------

    async def execute(self, payload: Any) -> List[Dict[str, Any]]:
        """
        Search PubMed for medical literature.

        Args:
            payload: Search query string or dict with 'query' key.

        Returns:
            List of paper dicts:
            [
                {
                    "pmid": "...",
                    "title": "...",
                    "year": "...",
                    "authors": "...",
                    "journal": "...",
                    "abstract": "...",
                }
            ]
        """
        # Parse payload
        if isinstance(payload, str):
            query = payload.strip()
        elif isinstance(payload, dict):
            query = (payload.get("query") or "").strip()
        else:
            raise ValueError("SearchTool payload must be a string or dict.")

        if not query:
            raise ValueError("Search query cannot be empty.")

        logger.info("SearchTool.execute | query='%s'", query)

        try:
            # Run sync requests in thread pool
            pmids = await asyncio.to_thread(
                self._search_pubmed_ids, query
            )

            if not pmids:
                logger.info("No PubMed results for query: '%s'", query)
                return []

            # Fetch summaries and abstracts in parallel
            summaries, abstracts = await asyncio.gather(
                asyncio.to_thread(self._fetch_paper_summaries, pmids),
                asyncio.to_thread(self._fetch_abstracts, pmids),
            )

            # Merge summaries with abstracts
            results = []
            for paper in summaries:
                pmid = paper["pmid"]
                results.append({
                    **paper,
                    "abstract": abstracts.get(pmid, "Abstract not available."),
                })

            logger.info(
                "SearchTool.execute complete | results=%d", len(results)
            )
            return results

        except Exception:
            logger.exception(
                "SearchTool.execute failed for query: '%s'", query
            )
            return []
