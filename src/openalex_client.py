#!/usr/bin/env python3
"""
Thin OpenAlex API client with on-disk caching and polite pool mailto.
All costs are charged against the free or paid OpenAlex credit.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import requests

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_MAILTO = "researcher-mobility-probe@example.org"


class OpenAlexClient:
    def __init__(self, cache_dir=None, mailto=DEFAULT_MAILTO, delay=0.05):
        self.mailto = mailto
        self.delay = delay
        self.session = requests.Session()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url):
        if not self.cache_dir:
            return None
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def _call(self, endpoint, params):
        url = f"{OPENALEX_BASE}/{endpoint}"
        req_params = dict(params)
        req_params["mailto"] = self.mailto
        for attempt in range(8):
            if self.delay:
                time.sleep(self.delay)
            r = self.session.get(url, params=req_params, timeout=90)
            if r.status_code in (429, 503):
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    backoff = max(1, int(retry_after))
                else:
                    backoff = min(60, 2 ** attempt)
                time.sleep(backoff)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    def get(self, endpoint, params=None, use_cache=True):
        params = params or {}
        # Build a stable cache key from the fully qualified URL
        r = requests.Request("GET", f"{OPENALEX_BASE}/{endpoint}", params=params)
        prepared = self.session.prepare_request(r)
        url = prepared.url
        cache_path = self._cache_path(url)
        if use_cache and cache_path and cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        data = self._call(endpoint, params)
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        return data

    def paginate(self, endpoint, params=None, max_pages=1000):
        """Iterate over all pages of a list endpoint using cursor pagination."""
        params = dict(params or {})
        params["cursor"] = "*"
        for _ in range(max_pages):
            data = self.get(endpoint, params)
            yield data
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor
        else:
            raise RuntimeError(f"Hit max_pages ({max_pages}) for {endpoint}")

    def sample_works(self, n, base_filter, select_fields, per_call=200, seed_start=1):
        """Return up to n random works. sample() returns per_call works per call.
        We vary the `seed` query parameter so repeated calls return distinct random
        samples while remaining cacheable.
        """
        results = []
        seen = set()
        calls = 0
        max_calls = (n // per_call) + 2
        while len(results) < n and calls < max_calls:
            calls += 1
            data = self.get(
                "works",
                {
                    "filter": base_filter,
                    "sample": per_call,
                    "per-page": per_call,
                    "select": select_fields,
                    "seed": seed_start + calls - 1,
                },
            )
            for w in data.get("results", []):
                wid = w.get("id")
                if wid and wid not in seen:
                    seen.add(wid)
                    results.append(w)
                    if len(results) >= n:
                        break
        return results[:n]

    def fetch_works_by_authors(self, author_ids, select_fields, per_page=200, subfield_id="subfields/1702"):
        """Return all works in the target subfield for a list of author IDs (<=100 for OR)."""
        if not author_ids:
            return []
        ids = "|".join(sorted(author_ids))
        base_filter = f"authorships.author.id:{ids},topics.subfield.id:{subfield_id}"
        results = []
        for page in self.paginate(
            "works",
            {
                "filter": base_filter,
                "select": select_fields,
                "per-page": per_page,
            },
        ):
            results.extend(page.get("results", []))
        return results
