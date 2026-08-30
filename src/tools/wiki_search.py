from __future__ import annotations

import re
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class WikiEnv:
    def __init__(self, max_intro_sentences: int = 5):
        self.max_intro_sentences = max_intro_sentences
        self._current_sentences: list[str] = []
        self._lookup_keyword: str | None = None
        self._lookup_pos = 0

    def search(self, entity: str) -> str:
        """Search[entity]"""
        page = self._fetch_extract(entity)

        if page is None:
            similar = self._fetch_search_suggestions(entity)
            self._current_sentences = []
            self._lookup_keyword = None
            if similar:
                return (
                    f"Could not find [{entity}]. "
                    f"Similar: {similar}"
                )
            return f"Could not find [{entity}]."

        title, extract = page
        sentences = [s for s in SENTENCE_SPLIT_RE.split(extract) if s.strip()]
        self._current_sentences = sentences
        self._lookup_keyword = None
        self._lookup_pos = 0

        intro = " ".join(sentences[: self.max_intro_sentences])
        return intro if intro else f"[{title}] page found but has no readable extract."

    def lookup(self, keyword: str) -> str:
        """Lookup[string] - Ctrl+F behavior on the current page."""
        if not self._current_sentences:
            return "No page currently open. Use Search first."

        if keyword != self._lookup_keyword:
            self._lookup_keyword = keyword
            self._lookup_pos = 0

        keyword_lower = keyword.lower()
        matches = [
            i for i, s in enumerate(self._current_sentences)
            if keyword_lower in s.lower()
        ]
        if not matches:
            return f"No mentions of [{keyword}] found."

        remaining = [i for i in matches if i >= self._lookup_pos]
        if not remaining:
            return f"No more results for [{keyword}]."

        idx = remaining[0]
        self._lookup_pos = idx + 1
        result_num = matches.index(idx) + 1
        return f"(Result {result_num}/{len(matches)}) {self._current_sentences[idx]}"

    def _fetch_extract(self, title: str) -> tuple[str, str] | None:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        }
        resp = requests.get(WIKI_API, params=params, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "missing" in page:
                return None
            extract = page.get("extract", "")
            if extract.strip():
                return page.get("title", title), extract
        return None

    def _fetch_search_suggestions(self, query: str, limit: int = 5) -> str:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        resp = requests.get(WIKI_API, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        titles = [r["title"] for r in results]
        return ", ".join(titles)
