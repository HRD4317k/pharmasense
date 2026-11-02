"""One polite HTTP layer for every client: per-host rate limiting, retries, and an on-disk cache.

Why a shared layer: the free APIs publish rate limits (openFDA 240/min, PubChem ~5/s, ...). Putting the
throttle and the cache here means a notebook re-run or a Streamlit refresh does not hit the network again.
Clients receive an `Http` object, so tests can pass a fake one and never touch the internet.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import Settings


class ApiError(RuntimeError):
    """A source could not be reached or returned something unusable. Callers record it as a gap."""


# minimum seconds between two requests to the same host (conservative readings of each service's policy)
MIN_INTERVAL = {
    "api.fda.gov": 0.30,                     # 240 requests/minute
    "pubchem.ncbi.nlm.nih.gov": 0.25,        # PubChem asks for <= 5 requests/second
    "www.ebi.ac.uk": 0.15,
    "rest.uniprot.org": 0.15,
    "string-db.org": 1.0,                    # STRING asks for ~1 request/second
    "api.platform.opentargets.org": 0.2,
    "clinicaltrials.gov": 0.2,
}


class Http:
    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None):
        self.settings = settings or Settings.load()
        self.session = session or requests.Session()
        ua = "PharmaSense-research/0.1 (+https://github.com/HRD4317k/pharmasense"
        ua += f"; {self.settings.contact_email})" if self.settings.contact_email else ")"
        self.session.headers.update({"User-Agent": ua, "Accept": "application/json"})
        self._last: dict[str, float] = {}
        self.cache_dir = Path(self.settings.cache_dir)
        self.calls = 0           # network calls actually made (cache hits not counted)
        self.hits = 0

    # ------------------------------------------------------------------ cache
    def _key(self, method, url, params, body) -> str:
        p = {k: v for k, v in (params or {}).items() if k != "api_key"}      # never key the cache on a secret
        raw = json.dumps([method, url, sorted(p.items()), body], sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()

    def _cache_path(self, url, key) -> Path:
        return self.cache_dir / urlparse(url).netloc / f"{key}.json"

    def _read(self, path: Path, ttl_days):
        try:
            if not path.exists():
                return False, None
            obj = json.loads(path.read_text())
            if ttl_days is not None and time.time() - obj["t"] > ttl_days * 86400:
                return False, None
            return True, obj["data"]
        except Exception:
            return False, None

    def _write(self, path: Path, data):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"t": time.time(), "data": data}))
        except Exception:
            pass                                                    # a read-only disk must not break a query

    # ------------------------------------------------------------------ requests
    def _throttle(self, host):
        gap = MIN_INTERVAL.get(host, 0.2)
        wait = self._last.get(host, 0) + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def _request(self, method, url, params=None, json_body=None, headers=None, ttl_days="default",
                 retries=3, not_found_ok=True):
        ttl = self.settings.cache_ttl_days if ttl_days == "default" else ttl_days
        key = self._key(method, url, params, json_body)
        path = self._cache_path(url, key)
        if ttl != 0:
            ok, data = self._read(path, ttl)
            if ok:
                self.hits += 1
                return data
        host = urlparse(url).netloc
        last_err = "unknown error"
        for attempt in range(retries):
            self._throttle(host)
            try:
                self.calls += 1
                r = self.session.request(method, url, params=params, json=json_body, headers=headers,
                                         timeout=self.settings.timeout)
            except requests.RequestException as e:
                last_err = f"network error: {e.__class__.__name__}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 404 and not_found_ok:
                self._write(path, None)                              # remember "no such record"
                return None
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(10, 2.0 * (attempt + 1)))
                continue
            if r.status_code != 200:
                raise ApiError(f"{host}: HTTP {r.status_code} {r.text[:160]!r}")
            try:
                data = r.json()
            except ValueError as e:
                raise ApiError(f"{host}: response was not JSON") from e
            self._write(path, data)
            return data
        raise ApiError(f"{host}: {last_err} after {retries} tries")

    def get_json(self, url, params=None, headers=None, ttl_days="default", not_found_ok=True):
        return self._request("GET", url, params=params, headers=headers, ttl_days=ttl_days, not_found_ok=not_found_ok)

    def post_json(self, url, json_body, headers=None, ttl_days="default"):
        return self._request("POST", url, json_body=json_body, headers=headers, ttl_days=ttl_days)
