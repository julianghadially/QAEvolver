"""Retrieval module combining Serper search and Firecrawl scraping."""

from dataclasses import dataclass, field
from typing import Optional

from src.services import SerperService, SearchResult, FirecrawlService, ScrapedPage


@dataclass
class RetrievalResult:
    """Result from a search + scrape retrieval step."""
    query: str
    search_results: list[SearchResult] = field(default_factory=list)
    scraped_page: Optional[ScrapedPage] = None
    success: bool = False
    error: Optional[str] = None


_serper = SerperService()
_firecrawl = FirecrawlService()


def retrieve(query: str, num_search_results: int = 10) -> RetrievalResult:
    """Execute a search query and scrape the top result.

    Uses Serper for web search, then Firecrawl to scrape the top-1 result.
    Falls back to search snippets if scraping fails.

    Args:
        query: The search query to execute.
        num_search_results: Number of search results to request from Serper.

    Returns:
        RetrievalResult with search results and scraped content.
    """
    result = RetrievalResult(query=query)

    try:
        search_results = _serper.search(query, num_results=num_search_results)
        result.search_results = search_results
    except Exception as e:
        result.error = f"Search failed: {e}"
        return result

    if not search_results:
        result.error = "No search results returned"
        return result

    top_result = search_results[0]
    scraped = _firecrawl.scrape(top_result.link)
    result.scraped_page = scraped

    if scraped.success and scraped.markdown:
        result.success = True
    else:
        result.success = True
        snippets = "\n\n".join(
            f"**{sr.title}**\n{sr.snippet}"
            for sr in search_results[:5]
        )
        result.scraped_page = ScrapedPage(
            url=top_result.link,
            markdown=snippets,
            title="Search snippets (scrape fallback)",
            success=False,
            error=scraped.error or "Scrape returned empty content"
        )

    return result
