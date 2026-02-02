"""
Semantic Scholar API client for Phase 2 search.

Uses the Semantic Scholar Graph API /paper/search endpoint.
An API key is optional but recommended to increase rate limits.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from paper_novelty_pipeline.config import (
    API_TIMEOUT,
    PHASE2_MAX_QUERY_ATTEMPTS,
    RETRY_DELAY,
    SEMANTIC_SCHOLAR_API_BASE,
    SEMANTIC_SCHOLAR_API_KEY,
)


class SemanticScholarClient:
    """Lightweight client for Semantic Scholar Graph API search."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = SEMANTIC_SCHOLAR_API_BASE,
        timeout: int = API_TIMEOUT,
        max_attempts: int = PHASE2_MAX_QUERY_ATTEMPTS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "paper-novelty-pipeline/phase2",
            }
        )
        key = api_key or SEMANTIC_SCHOLAR_API_KEY
        if key:
            self.session.headers["x-api-key"] = key

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        offset: int = 0,
        fields: str,
    ) -> Dict[str, Any]:
        """Search papers by query string."""
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": int(limit),
            "offset": int(offset),
            "fields": fields,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.RequestException(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # requests.RequestException or JSON error
                last_exc = exc
                if attempt + 1 >= self.max_attempts:
                    break
                sleep_s = min(RETRY_DELAY * (2 ** attempt), 60)
                time.sleep(sleep_s)

        raise RuntimeError(
            f"Semantic Scholar search failed after {self.max_attempts} attempts: {last_exc}"
        )
