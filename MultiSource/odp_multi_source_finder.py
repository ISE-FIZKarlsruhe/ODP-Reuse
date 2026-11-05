#!/usr/bin/env python3
"""
ODP Multi-Source Reuse Finder
-----------------------------
Find ontologies that *reuse* Ontology Design Patterns (ODPs) by searching across:
  - Google Custom Search (web): requires GOOGLE_API_KEY and GOOGLE_CSE_ID
  - GitHub Code Search: requires GITHUB_TOKEN
  - BioPortal (OntoPortal): requires BIOPORTAL_API_KEY
  - MatPortal (OntoPortal): requires MATPORTAL_API_KEY
  - LOV (Linked Open Vocabularies) API: no key required (best-effort)
  - Generic SPARQL endpoints (e.g., LOV SPARQL, Ontobee): optional

It can also auto-discover ODP pattern IRIs by pulling files from the official repo:
  - GitHub repo: odpa/patterns-repository

Output CSV columns:
  source, pattern_label, pattern_iri, match_label, url, details

Install deps:
  python -m pip install requests SPARQLWrapper
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import random
from typing import Dict, Iterable, List, Optional, Tuple, Any

import requests

try:
    from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
except Exception:
    SPARQLWrapper = None
    SPARQL_JSON = None


# ----------------------
# Constants & Regexes
# ----------------------

GITHUB_API = "https://api.github.com"
ODP_OWNER = "odpa"
ODP_REPO = "patterns-repository"
SEARCH_EXTS = ["owl", "ttl", "rdf", "nt", "trig", "xml"]

RE_XML_BASE = re.compile(r'xml:base\s*=\s*"([^"]+)"')
RE_ONTOLOGY_IRI_ATTR = re.compile(r'<owl:Ontology[^>]*?(?:IRI|rdf:about)\s*=\s*"([^"]+)"', re.I)
RE_ONTOLOGY_IRI_TURTLE = re.compile(r'@base\s+<([^>]+)>', re.I)


# ----------------------
# Utilities
# ----------------------

def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def sleep_short():
    time.sleep(0.8)


def get_env(name: str, fallback: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, fallback)
    return v

def _now_ts() -> int:
    return int(time.time())

def _sleep_until(ts: int):
    wait = max(0, ts - _now_ts() + 1)
    log(f"[GitHub] Sleeping {wait}s until reset...")
    time.sleep(wait)

def _jitter_ms(base_ms: int) -> float:
    # ±25% jitter
    spread = int(base_ms * 0.25)
    return (base_ms + random.randint(-spread, spread)) / 1000.0

def as_str(val: Any) -> str:
    """Normalize any value (list/dict/etc.) to a string."""
    if val is None:
        return ""
    if isinstance(val, (str, int, float, bool)):
        return str(val)
    if isinstance(val, list):
        for e in val:
            s = as_str(e)
            if s:
                return s
        return ""
    if isinstance(val, dict):
        for k in ("uri", "@id", "id", "value", "url", "href"):
            if k in val:
                return as_str(val[k])
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    return str(val)

def normalize_url(u: Any) -> str:
    return as_str(u).strip()


# ----------------------
# GitHub helper functions
# ----------------------

def gh_session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    hdrs = {"Accept": "application/vnd.github.v3+json"}
    if token:
        hdrs["Authorization"] = f"token {token}"
    s.headers.update(hdrs)
    return s


def gh_request(session: requests.Session, url: str, params=None, *, max_retries: int = 6) -> requests.Response:
    """
    GitHub GET with rate-limit + abuse-detection backoff.
    - Obeys X-RateLimit-Remaining/Reset
    - Obeys Retry-After when present
    - Exponential backoff with jitter on 403/429/5xx
    """
    attempt = 0
    backoff = 2  # seconds
    while True:
        r = session.get(url, params=params)
        # Happy path
        if r.status_code in (200, 201):
            return r

        # Parse common headers
        remaining = r.headers.get("X-RateLimit-Remaining")
        reset     = r.headers.get("X-RateLimit-Reset")
        retry_after = r.headers.get("Retry-After")
        msg = ""
        try:
            j = r.json()
            if isinstance(j, dict) and "message" in j:
                msg = j["message"]
        except Exception:
            pass

        # Secondary rate limit / abuse detection often returns 403 with message
        if r.status_code in (403, 429):
            # If primary ratelimit exhausted
            if remaining == "0" and reset and reset.isdigit():
                _sleep_until(int(reset))
                attempt += 1
                continue
            # Retry-After header if present
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
                log(f"[GitHub] Retry-After={wait}s. Backing off...")
                time.sleep(wait)
                attempt += 1
                continue
            # Abuse detection message fallback
            if "abuse detection" in (msg or "").lower() or "temporarily blocked" in (msg or "").lower():
                wait = min(120, backoff)
                log(f"[GitHub] Abuse detection: {msg}. Backoff {wait}s...")
                time.sleep(wait + random.random())
                attempt += 1
                backoff = min(240, backoff * 2)
                continue

        # Handle transient server issues
        if 500 <= r.status_code < 600:
            wait = min(60, backoff)
            log(f"[GitHub] {r.status_code} server error. Backoff {wait}s...")
            time.sleep(wait + random.random())
            attempt += 1
            backoff = min(120, backoff * 2)
            continue

        # Any other 4xx is a property of the request, not of the moment: a query
        # GitHub cannot parse (422) fails identically on every retry, and each
        # retry still counts against the search rate limit.
        if 400 <= r.status_code < 500:
            r.raise_for_status()

        # If we reach here and can retry, do so a few times
        attempt += 1
        if attempt <= max_retries:
            wait = min(60, backoff)
            log(f"[GitHub] {r.status_code} {msg if msg else r.reason}. Retry in {wait}s...")
            time.sleep(wait + random.random())
            backoff = min(120, backoff * 2)
            continue

        # Give up
        r.raise_for_status()


def gh_list_dir(session: requests.Session, owner: str, repo: str, path: str="") -> List[Dict]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}".rstrip("/")
    r = gh_request(session, url)
    return r.json()


def gh_read_file_text(session: requests.Session, owner: str, repo: str, path: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    r = gh_request(session, url)
    j = r.json()
    if isinstance(j, dict) and j.get("encoding") == "base64":
        return base64.b64decode(j["content"]).decode("utf-8", errors="replace")
    raw = j.get("download_url") if isinstance(j, dict) else None
    if raw:
        r2 = session.get(raw)
        r2.raise_for_status()
        return r2.text
    return ""


def walk_repo_patterns(session: requests.Session, owner: str, repo: str, start_path: str="") -> List[Tuple[str, str]]:
    results = []
    stack = [start_path]
    while stack:
        path = stack.pop()
        items = gh_list_dir(session, owner, repo, path)
        for it in items:
            t = it.get("type")
            p = it.get("path")
            if t == "dir":
                stack.append(p)
            elif t == "file":
                name = it["name"].lower()
                if name.endswith(".owl") or name.endswith(".ttl"):
                    txt = gh_read_file_text(session, owner, repo, p)
                    results.append((p, txt))
    return results


def extract_pattern_iri(text: str) -> Optional[str]:
    m = RE_XML_BASE.search(text)
    if m:
        return m.group(1).strip()
    m = RE_ONTOLOGY_IRI_ATTR.search(text)
    if m:
        return m.group(1).strip()
    m = RE_ONTOLOGY_IRI_TURTLE.search(text)
    if m:
        return m.group(1).strip()
    return None


def derive_label_from_iri(iri: str) -> str:
    last = iri.rstrip("/").split("/")[-1]
    return re.sub(r'\.owl$','', last, flags=re.I)


# ---------- NEW: safer GitHub query builder + helpers ----------

def build_code_search_queries_for_iri(iri: str) -> List[str]:
    """
    One GitHub Code Search query per file extension, with `in:file` to force
    content search.

    There is deliberately no grouped ``( extension:a OR extension:b )`` query:
    the REST code-search parser rejects it with a 422 on every call, so it only
    ever cost a request per IRI before these per-extension queries ran.
    """
    exts = ["owl", "ttl", "rdf", "xml"]  # concise set; expand if you want
    return [f'"{iri}" extension:{e} in:file' for e in exts]


def is_generic_iri(iri: str) -> bool:
    t = iri.strip().lower().rstrip("/")
    if t.endswith("/template") or "/template/" in t:
        return True
    if t in {
        "http://ontologydesignpatterns.org", "https://ontologydesignpatterns.org",
        "http://www.ontologydesignpatterns.org", "https://www.ontologydesignpatterns.org",
    }:
        return True
    return False


def safe_queries(queries: List[str], max_len: int = 256) -> List[str]:
    """GitHub rejects overly long queries; keep only those under max_len."""
    return [q for q in queries if len(q) <= max_len]


class SearchThrottle:
    """
    Space calls so that at most ``per_minute`` start in any minute.

    GitHub allows 10 code-search requests per minute. Exceeding it does not just
    fail the request: it trips the secondary rate limit, whose Retry-After is
    typically ~12 minutes. Staying under the limit is far cheaper than obeying
    the back-off afterwards.
    """

    def __init__(self, per_minute: float = 9, clock=time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self.clock, self.sleep = clock, sleep
        self._last: Optional[float] = None

    def wait(self) -> None:
        if self._last is not None:
            gap = self.interval - (self.clock() - self._last)
            if gap > 0:
                self.sleep(gap)
        self._last = self.clock()


def github_code_search(session: requests.Session, query: str, *, per_page=50, max_pages=5, delay_ms=1200, max_retries=6,
                       throttle: Optional[SearchThrottle] = None) -> Iterable[Dict]:
    """
    Iterate GitHub code search results with polite pacing.
    """
    for page in range(1, max_pages + 1):
        params = {"q": query, "per_page": min(100, max(1, per_page)), "page": page}
        if throttle is not None:
            throttle.wait()
        r = gh_request(session, f"{GITHUB_API}/search/code", params=params, max_retries=max_retries)
        js = r.json()
        items = js.get("items", [])
        for it in items:
            yield it
        if len(items) < params["per_page"]:
            break
        # polite delay between pages
        time.sleep(_jitter_ms(delay_ms))


def search_github_patterns(patterns: List[Tuple[str, str]], search, *,
                           skip_generic: bool = False,
                           deadline: Optional[float] = None,
                           clock=None,
                           on_pattern_done=None) -> Tuple[List[Dict[str, str]], List[Dict]]:
    """
    Run the per-extension code-search queries for every pattern IRI.

    ``search(query)`` yields GitHub code-search items. Returns ``(rows, status)``:
    one result row per hit, and one status row per pattern IRI saying whether it
    was searched -- so a run that stops early says which IRIs it never reached
    instead of silently reporting them as unreused.

    Once ``clock()`` passes ``deadline`` the remaining IRIs are marked
    ``not_searched`` and not queried.
    """
    now = clock if clock is not None else time.monotonic
    rows: List[Dict[str, str]] = []
    status: List[Dict] = []
    for idx, (label, iri) in enumerate(patterns, 1):
        st = {"source": "github", "pattern_iri": iri, "status": "done",
              "queries": 0, "hits": 0, "note": ""}
        status.append(st)
        if deadline is not None and now() >= deadline:
            st.update(status="not_searched", note="time budget exhausted")
            continue
        log(f"[GitHub {idx}/{len(patterns)}] Searching for IRI: {iri}")
        if skip_generic and is_generic_iri(iri):
            log(f"[GitHub] Skipping generic template/homepage IRI (flagged): {iri}")
            st.update(status="skipped", note="generic IRI")
            continue

        queries = safe_queries(build_code_search_queries_for_iri(iri))
        if not queries:
            log(f"[GitHub] No usable queries (likely too long) for: {iri}")
            st.update(status="skipped", note="query too long")
            continue
        rejected = 0
        for q in queries:
            st["queries"] += 1
            try:
                for item in search(q):
                    rows.append({
                        "source": "github",
                        "pattern_label": as_str(label),
                        "pattern_iri": as_str(iri),
                        "match_label": as_str(item.get("repository", {}).get("full_name")),
                        "url": normalize_url(item.get("html_url")),
                        "details": as_str(item.get("path")),
                    })
                    st["hits"] += 1
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 422:
                    log(f"[GitHub] 422 for query: {q}. Skipping it.")
                    rejected += 1
                    continue
                raise
        if rejected:
            st["note"] = f"{rejected} quer{'y' if rejected == 1 else 'ies'} rejected (422)"
        if on_pattern_done is not None:
            on_pattern_done(rows, status)
    return rows, status


def load_patterns_file(path) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """
    Read pattern IRIs from a CSV with a ``pattern_iri`` column and an optional
    ``pattern_set`` column (e.g. ``release`` / ``new``), or from a plain list
    with one IRI per line. Returns ``(patterns, pattern_set_by_iri)``.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    if not lines:
        return [], {}
    sets: Dict[str, str] = {}
    iris: List[str] = []
    if lines[0].split(",")[0].strip() == "pattern_iri":
        for r in csv.DictReader(lines):
            iri = (r.get("pattern_iri") or "").strip()
            if not iri:
                continue
            iris.append(iri)
            if (r.get("pattern_set") or "").strip():
                sets[iri] = r["pattern_set"].strip()
    else:
        iris = [l.strip() for l in lines]
    return [(derive_label_from_iri(i), i) for i in iris], sets


OUT_COLUMNS = ["source", "pattern_label", "pattern_iri", "match_label", "url", "details",
               "pattern_set"]


class IncrementalCsv:
    """
    Write result rows as they arrive, de-duplicated by (source, pattern_iri,
    url) exactly as the end-of-run write used to be.

    A harvest that is cancelled hours in keeps everything it found so far.
    """

    def __init__(self, path, pattern_sets: Optional[Dict[str, str]] = None):
        self.path = path
        self.pattern_sets = pattern_sets or {}
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._fh, fieldnames=OUT_COLUMNS, extrasaction="ignore")
        self._w.writeheader()
        self._fh.flush()
        self._seen = set()
        self._done = 0
        self.n_written = 0

    def sync(self, rows: List[Dict[str, str]]) -> None:
        for r in rows[self._done:]:
            k = (as_str(r.get("source")), as_str(r.get("pattern_iri")), as_str(r.get("url")))
            if k in self._seen:
                continue
            self._seen.add(k)
            self._w.writerow({**r, "pattern_set": self.pattern_sets.get(as_str(r.get("pattern_iri")), "")})
            self.n_written += 1
        self._done = len(rows)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ----------------------
# Google Custom Search
# ----------------------

def google_cse_search(api_key: str, cse_id: str, q: str, num: int = 10, start: int = 1) -> Dict:
    """
    Returns CSE JSON results for a query (num up to 10 per call).
    Raises HTTPError with JSON message on non-200.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": q, "num": num, "start": start}
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"error": {"message": r.text}}
        raise requests.HTTPError(f"{r.status_code} CSE error: {err}", response=r)
    return r.json()


def google_key_cx_cycle(cli_keys: Optional[str], cli_cxs: Optional[str],
                        env_key: Optional[str], env_cx: Optional[str]) -> List[Tuple[str, str]]:
    """
    Build a list of (key, cx) pairs to try, from CLI or environment.
    """
    keys = [k.strip() for k in (cli_keys.split(",") if cli_keys else []) if k.strip()]
    cxs  = [c.strip() for c in (cli_cxs.split(",") if cli_cxs else []) if c.strip()]
    if not keys and env_key:
        keys = [env_key]
    if not cxs and env_cx:
        cxs = [env_cx]
    pairs = []
    for k in keys or []:
        for cx in cxs or []:
            pairs.append((k, cx))
    return pairs


# ----------------------
# BioPortal / MatPortal (OntoPortal)
# ----------------------

def bioportal_search(api_key: str, q: str, pages: int = 1, page_size: int = 50) -> Iterable[Dict]:
    """
    BioPortal Search API (OntoPortal): https://data.bioontology.org/documentation
    """
    base = "https://data.bioontology.org/search"
    headers = {"Authorization": f"apikey token={api_key}"}
    for p in range(1, pages + 1):
        params = {"q": q, "pagesize": page_size, "page": p}
        r = requests.get(base, params=params, headers=headers, timeout=60)
        if r.status_code == 401:
            log("[BioPortal] Unauthorized. Check BIOPORTAL_API_KEY.")
            return
        r.raise_for_status()
        yield r.json()
        sleep_short()


def matportal_search(api_key: str, q: str, pages: int = 1, page_size: int = 50) -> Iterable[Dict]:
    """
    MatPortal (OntoPortal) Search API: same as BioPortal; different base URL.
    """
    base = "https://rest.matportal.org/search"
    headers = {"Authorization": f"apikey token={api_key}"}
    for p in range(1, pages + 1):
        params = {"q": q, "pagesize": page_size, "page": p}
        r = requests.get(base, params=params, headers=headers, timeout=60)
        if r.status_code == 401:
            log("[MatPortal] Unauthorized. Check MATPORTAL_API_KEY.")
            return
        r.raise_for_status()
        yield r.json()
        sleep_short()


# ----------------------
# LOV API
# ----------------------

def lov_search_terms(q: str, page: int = 1) -> Optional[Dict]:
    # LOV v2 term search (subject to change)
    url = "https://lov.linkeddata.es/dataset/lov/api/v2/term/search"
    params = {"q": q, "page": page}
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"[LOV] term search error: {e}")
        return None


def lov_search_vocab(q: str, page: int = 1) -> Optional[Dict]:
    # LOV v2 vocabulary search (subject to change)
    url = "https://lov.linkeddata.es/dataset/lov/api/v2/vocabulary/search"
    params = {"q": q, "page": page}
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"[LOV] vocabulary search error: {e}")
        return None


# ----------------------
# SPARQL Search
# ----------------------

def sparql_find_imports(endpoint_url: str, iri_prefix: str, limit: int = 200) -> List[Tuple[str, str]]:
    """
    Find ontologies that owl:imports anything from iri_prefix
    Returns list of (ontology, importedIRI)
    """
    if SPARQLWrapper is None:
        log("[SPARQL] SPARQLWrapper not installed. Install with: pip install SPARQLWrapper")
        return []

    query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?ontology ?imported WHERE {{
      ?ontology owl:imports ?imported .
      FILTER (CONTAINS(STR(?imported), "{iri_prefix}"))
    }} LIMIT {limit}
    """
    try:
        sparql = SPARQLWrapper(endpoint_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(SPARQL_JSON)
        res = sparql.query().convert()
        rows = []
        for b in res.get("results", {}).get("bindings", []):
            onto = b.get("ontology", {}).get("value", "")
            imp = b.get("imported", {}).get("value", "")
            if onto and imp:
                rows.append((onto, imp))
        return rows
    except Exception as e:
        log(f"[SPARQL] query error: {e}")
        return []


# ----------------------
# Main runner
# ----------------------

def main():
    ap = argparse.ArgumentParser(description="Find reuse of Ontology Design Patterns across multiple sources.")
    ap.add_argument("--sources", default="github,google,bioportal,lov",
                    help="Comma-separated sources: github,google,bioportal,matportal,lov,sparql")
    ap.add_argument("--patterns", help="Comma-separated list of pattern IRIs. If omitted, auto-discover from odpa/patterns-repository.")
    ap.add_argument("--patterns-file",
                    help="CSV with a pattern_iri column (and optionally pattern_set), or one IRI per line. "
                         "Takes precedence over --patterns and auto-discovery.")
    ap.add_argument("--github-per-page", type=int, default=50, help="GitHub items per page (max 100). Lower to be gentler.")
    ap.add_argument("--github-max-pages", type=int, default=5, help="GitHub pages per query (100 items each).")
    ap.add_argument("--github-delay-ms", type=int, default=1200, help="Delay in milliseconds between GitHub queries (adds jitter).")
    ap.add_argument("--github-max-retries", type=int, default=6, help="Max retries for GitHub requests when rate limited/abuse detected.")
    ap.add_argument("--github-search-per-minute", type=float, default=9,
                    help="Ceiling on code-search calls per minute (GitHub allows 10; going over trips a "
                         "~12-minute secondary-limit back-off). 0 disables pacing.")
    ap.add_argument("--github-max-minutes", type=float, default=0,
                    help="Stop issuing GitHub searches after this many minutes; remaining IRIs are recorded "
                         "as not_searched and the run exits with status 3. 0 = no limit.")
    ap.add_argument("--status-out",
                    help="Per-IRI GitHub search status CSV (default: <out>_github_status.csv).")
    ap.add_argument("--google-pages", type=int, default=2, help="Google pages (10 results per page).")
    ap.add_argument("--google-keys", help="Comma-separated GOOGLE_API_KEYs to try (overrides env).")
    ap.add_argument("--google-cxs",  help="Comma-separated GOOGLE_CSE_IDs to try (overrides env).")
    ap.add_argument("--google-skip-on-403", action="store_true",
                    help="If Google returns 403 once, skip Google for the rest of the run.")
    ap.add_argument("--google-skip-on-429", action="store_true",
                    help="If Google returns 429 (daily cap), skip Google for the rest of the run.")
    ap.add_argument("--google-max-queries", type=int, default=90,
                    help="Hard cap on number of Google API calls in this run (helps avoid hitting 100/day).")
    ap.add_argument("--google-sites", help="Comma-separated domains to restrict search (adds site: filters).")
    ap.add_argument("--google-min-hits", type=int, default=0,
                    help="Stop fetching more Google pages for an IRI once this many hits have been collected.")
    ap.add_argument("--bioportal-pages", type=int, default=2, help="BioPortal pages.")
    ap.add_argument("--matportal-pages", type=int, default=2, help="MatPortal pages.")
    ap.add_argument("--sparql-endpoint", help="SPARQL endpoint URL for 'sparql' source.")
    ap.add_argument("--skip-generic", action="store_true", help="Skip generic/template ODP IRIs (e.g., /template, homepage IRIs).")
    ap.add_argument("--out", default="odp_multi_results.csv", help="Output CSV path.")
    args = ap.parse_args()
    t_start = time.monotonic()

    # Credentials
    gh_token = get_env("GITHUB_TOKEN")
    google_key = get_env("GOOGLE_API_KEY")
    google_cx  = get_env("GOOGLE_CSE_ID")
    bioportal_key = get_env("BIOPORTAL_API_KEY")
    matportal_key = get_env("MATPORTAL_API_KEY")

    # Determine sources
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    # Get pattern IRIs
    patterns: List[Tuple[str, str]] = []  # (label, iri)
    pattern_sets: Dict[str, str] = {}

    if args.patterns_file:
        patterns, pattern_sets = load_patterns_file(args.patterns_file)
        by_set: Dict[str, int] = {}
        for _l, iri in patterns:
            by_set[pattern_sets.get(iri, "-")] = by_set.get(pattern_sets.get(iri, "-"), 0) + 1
        log(f"Using {len(patterns)} pattern IRIs from {args.patterns_file} "
            f"({', '.join(f'{k}: {v}' for k, v in sorted(by_set.items()))}).")
    elif args.patterns:
        for iri in [p.strip() for p in args.patterns.split(",") if p.strip()]:
            label = derive_label_from_iri(iri)
            patterns.append((label, iri))
        log(f"Using {len(patterns)} pattern IRIs from --patterns.")
    else:
        # Auto-discover from odpa/patterns-repository
        log("Auto-discovering ODP IRIs from odpa/patterns-repository ...")
        gh = gh_session(gh_token)
        files = walk_repo_patterns(gh, ODP_OWNER, ODP_REPO, "")
        discovered = []
        for path, text in files:
            iri = extract_pattern_iri(text)
            if iri and "ontologydesignpatterns.org" in iri:
                discovered.append(iri)
        # dedupe
        seen = set()
        for iri in discovered:
            if iri not in seen:
                patterns.append((derive_label_from_iri(iri), iri))
                seen.add(iri)
        log(f"Discovered {len(patterns)} unique ODP pattern IRIs.")

    if not patterns:
        log("No patterns found. Provide --patterns or check repository access.")
        sys.exit(1)

    rows: List[Dict[str, str]] = []
    out = args.out
    writer = IncrementalCsv(out, pattern_sets=pattern_sets)
    status_out = args.status_out or (os.path.splitext(out)[0] + "_github_status.csv")
    gh_status: List[Dict] = []

    def write_status(status: List[Dict]) -> None:
        with open(status_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["source", "pattern_iri", "pattern_set", "status",
                                               "queries", "hits", "note"])
            w.writeheader()
            for s in status:
                w.writerow({**s, "pattern_set": pattern_sets.get(s["pattern_iri"], "")})

    def checkpoint(rows_so_far, status_so_far):
        writer.sync(rows_so_far)
        write_status(status_so_far)

    # ---------------- GitHub ----------------
    if "github" in sources:
        if not gh_token:
            log("[GitHub] Missing GITHUB_TOKEN; skipping GitHub search.")
        else:
            session = gh_session(gh_token)
            throttle = SearchThrottle(per_minute=args.github_search_per_minute)
            deadline = (t_start + 60 * args.github_max_minutes) if args.github_max_minutes > 0 else None

            def search(q):
                return github_code_search(
                    session, q,
                    per_page=args.github_per_page,
                    max_pages=args.github_max_pages,
                    delay_ms=args.github_delay_ms,
                    max_retries=args.github_max_retries,
                    throttle=throttle,
                )

            gh_rows, gh_status = search_github_patterns(
                patterns, search,
                skip_generic=args.skip_generic,
                deadline=deadline,
                on_pattern_done=lambda r, s: checkpoint(rows + r, s),
            )
            rows.extend(gh_rows)
            checkpoint(rows, gh_status)

    # ---------------- Google CSE ----------------
    if "google" in sources:
        # Build (key,cx) candidates
        pairs = google_key_cx_cycle(args.google_keys, args.google_cxs, google_key, google_cx)
        if not pairs:
            log("[Google] Missing GOOGLE_API_KEY/GOOGLE_CSE_ID (or --google-keys/--google-cxs); skipping Google CSE.")
        else:
            google_calls = 0
            hard_fail_google = False
            for idx, (label, iri) in enumerate(patterns, 1):
                if hard_fail_google:
                    break
                log(f"[Google {idx}/{len(patterns)}] Searching for IRI: {iri}")

                sites_clause = ""
                if args.google_sites:
                    doms = [d.strip() for d in args.google_sites.split(",") if d.strip()]
                    if doms:
                        sites_clause = " (" + " OR ".join(f"site:{d}" for d in doms) + ")"

                q = f'("owl:imports" OR "@base" OR "xmlns:" OR "@prefix") "{iri}" (filetype:owl OR filetype:ttl OR filetype:rdf OR filetype:nt OR filetype:trig OR filetype:xml){sites_clause}'
                print(f"[Google] Query: {q}")
                start = 1
                iri_hits = 0
                for _ in range(args.google_pages):
                    if google_calls >= args.google_max_queries:
                        log(f"[Google] Reached google-max-queries={args.google_max_queries}. Skipping remaining Google calls.")
                        hard_fail_google = True
                        break

                    success = False
                    for (gkey, gcx) in pairs:
                        try:
                            res = google_cse_search(gkey, gcx, q, num=10, start=start)
                            google_calls += 1
                            items = res.get("items", [])
                            for item in items:
                                rows.append({
                                    "source": "google",
                                    "pattern_label": as_str(label),
                                    "pattern_iri": as_str(iri),
                                    "match_label": as_str(item.get("title","")),
                                    "url": normalize_url(item.get("link","")),
                                    "details": as_str(item.get("snippet",""))
                                })
                            iri_hits += len(items)
                            success = True
                            break  # this page done with the working pair
                        except requests.HTTPError as e:
                            code = e.response.status_code if e.response is not None else None
                            msg = str(e)
                            if code == 429:
                                log(f"[Google] 429 (daily quota). Details: {msg}")
                                if args.google_skip_on_429:
                                    log("[Google] Skipping Google for the rest of this run (--google-skip-on-429).")
                                    hard_fail_google = True
                                    break
                                else:
                                    success = False
                                    break
                            elif code == 403:
                                log(f"[Google] 403 for this key/cx. Details: {msg}")
                                # try next key/cx
                            else:
                                log(f"[Google] Error: {msg}")
                                # try next key/cx

                    if hard_fail_google or not success:
                        break

                    if args.google_min_hits and iri_hits >= args.google_min_hits:
                        break

                    start += 10
                    sleep_short()

    # ---------------- BioPortal ----------------
    if "bioportal" in sources:
        if not bioportal_key:
            log("[BioPortal] Missing BIOPORTAL_API_KEY; skipping BioPortal search.")
        else:
            for idx, (label, iri) in enumerate(patterns, 1):
                log(f"[BioPortal {idx}/{len(patterns)}] Searching for IRI: {iri}")
                for page in bioportal_search(bioportal_key, iri, pages=args.bioportal_pages):
                    for res in page.get("collection", []):
                        pref = as_str(res.get("prefLabel") or res.get("@id") or res.get("id") or "")
                        url  = normalize_url(res.get("@id") or res.get("links", {}).get("self") or res.get("links", {}).get("ontology"))
                        rows.append({
                            "source": "bioportal",
                            "pattern_label": as_str(label),
                            "pattern_iri": as_str(iri),
                            "match_label": pref,
                            "url": url,
                            "details": as_str(res.get("definition") or res.get("annotatedProperty") or "")
                        })
                sleep_short()

    # ---------------- MatPortal ----------------
    if "matportal" in sources:
        if not matportal_key:
            log("[MatPortal] Missing MATPORTAL_API_KEY; skipping MatPortal search.")
        else:
            for idx, (label, iri) in enumerate(patterns, 1):
                log(f"[MatPortal {idx}/{len(patterns)}] Searching for IRI: {iri}")
                for page in matportal_search(matportal_key, iri, pages=args.matportal_pages):
                    for res in page.get("collection", []):
                        pref = as_str(res.get("prefLabel") or res.get("@id") or res.get("id") or "")
                        url  = normalize_url(res.get("@id") or res.get("links", {}).get("self") or res.get("links", {}).get("ontology"))
                        rows.append({
                            "source": "matportal",
                            "pattern_label": as_str(label),
                            "pattern_iri": as_str(iri),
                            "match_label": pref,
                            "url": url,
                            "details": as_str(res.get("definition") or res.get("annotatedProperty") or "")
                        })
                sleep_short()

    # ---------------- LOV ----------------
    if "lov" in sources:
        for idx, (label, iri) in enumerate(patterns, 1):
            log(f"[LOV {idx}/{len(patterns)}] Searching for IRI: {iri}")

            # Strategy 1: term search
            js = lov_search_terms(iri, page=1)
            if js and isinstance(js, dict):
                terms = js.get("results", [])
                for t in terms:
                    vocab = t.get("vocabulary", {})
                    match_label = as_str(
                        vocab.get("prefix")
                        or t.get("prefixedName")
                        or t.get("type")
                        or t.get("label")
                        or ""
                    )
                    url = normalize_url(t.get("uri") or vocab.get("uri") or vocab.get("uriSpace"))
                    rows.append({
                        "source": "lov-terms",
                        "pattern_label": as_str(label),
                        "pattern_iri": as_str(iri),
                        "match_label": match_label,
                        "url": url,
                        "details": json.dumps({"vocab": vocab}, ensure_ascii=False)
                    })

            # Strategy 2: vocabulary search
            js2 = lov_search_vocab(iri, page=1)
            if js2 and isinstance(js2, dict):
                vocabs = js2.get("results", [])
                for v in vocabs:
                    match_label = as_str(v.get("prefPrefix") or v.get("prefix") or v.get("uri") or v.get("uriSpace"))
                    url = normalize_url(v.get("uri") or v.get("uriSpace"))
                    rows.append({
                        "source": "lov-vocab",
                        "pattern_label": as_str(label),
                        "pattern_iri": as_str(iri),
                        "match_label": match_label,
                        "url": url,
                        "details": as_str(v.get("description") or "")
                    })
            sleep_short()

    # ---------------- SPARQL endpoint ----------------
    if "sparql" in sources:
        if not args.sparql_endpoint:
            log("[SPARQL] No --sparql-endpoint provided; skipping SPARQL source.")
        else:
            iri_prefix = "ontologydesignpatterns"
            log(f"[SPARQL] Querying {args.sparql_endpoint} for imports starting with {iri_prefix}")
            pairs = sparql_find_imports(args.sparql_endpoint, iri_prefix, limit=500)
            for onto, imp in pairs:
                rows.append({
                    "source": "sparql",
                    "pattern_label": derive_label_from_iri(imp),
                    "pattern_iri": imp,
                    "match_label": onto,
                    "url": onto,
                    "details": "owl:imports"
                })

    # Rows are de-duplicated by (source, pattern_iri, url) as they are written.
    writer.sync(rows)
    writer.close()
    log(f"Done. Wrote {writer.n_written} rows to {out}")

    unsearched = [s for s in gh_status if s["status"] == "not_searched"]
    if gh_status:
        log(f"GitHub search status: {status_out}")
    if unsearched:
        log(f"[GitHub] INCOMPLETE: {len(unsearched)} of {len(gh_status)} pattern IRIs were not searched "
            f"(--github-max-minutes={args.github_max_minutes:g}). Their absence from the results is "
            f"not evidence of no reuse.")
        sys.exit(3)


if __name__ == "__main__":
    main()
