#!/usr/bin/env python3
"""
ODP Paper Harvester (arXiv + OpenAlex + Crossref + Unpaywall)
-------------------------------------------------------------
Searches for papers on "ontology design patterns" (ODPs) and related terms,
saves rich metadata (CSV + JSONL), and downloads PDFs when it is legally/technically allowed.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.parse

import requests

# Optional dependency: feedparser (for arXiv Atom)
try:
    import feedparser
except Exception as e:
    feedparser = None

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_BASE = "https://api.openalex.org/works"
CROSSREF_BASE = "https://api.crossref.org/works"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"

ODP_DEFAULT_QUERIES = [
    'ontology design pattern',
    'ontology design patterns',
    '"ontology design pattern"',
    '"ontology design patterns"',
    'ontology pattern reuse',
    'ontology pattern',
    'ODP ontology',
    '"ontology pattern" reuse',
    '"ontology design pattern" reuse',
    'ontology design pattern ODP',
]

PUBLISHER_HINTS = [
    'acm', 'springer', 'elsevier', 'ieee', 'wiley', 'ios press', 'frontiers',
    'mdpi', 'nature', 'springer nature', 'taylor & francis'
]

# Lightweight polite rate limiting
def sleep_a_bit(t=1.0):
    time.sleep(t)

@dataclass
class Work:
    id: str
    doi: Optional[str]
    title: Optional[str]
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    publisher: Optional[str]
    url: Optional[str]
    source: str  # "arxiv" | "openalex" | "crossref"
    oa_url: Optional[str] = None
    pdf_url: Optional[str] = None
    is_oa: Optional[bool] = None
    abstract: Optional[str] = None
    extra: Dict = None

def norm(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return re.sub(r'\s+', ' ', s).strip()

def doi_norm(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '').strip()
    return doi.lower()

def pick_filename(s: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9._-]+', '_', s)[:140]
    return s.strip('_') or 'file'

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def arxiv_search(queries: List[str], max_results=200) -> List[Work]:
    if feedparser is None:
        print("feedparser not installed; skipping arXiv. Run: pip install feedparser", file=sys.stderr)
        return []
    all_works = []
    for q in queries:
        # Build arXiv query: search all fields
        q_enc = urllib.parse.quote_plus(f'all:{q}')
        start = 0
        batch = 100
        while start < max_results:
            url = f"{ARXIV_API}?search_query={q_enc}&start={start}&max_results={batch}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if not feed.entries:
                break
            for e in feed.entries:
                arxiv_id = e.get('id')
                title = norm(e.get('title'))
                abstract = norm(e.get('summary'))
                authors = [norm(a.get('name')) for a in e.get('authors', []) if a.get('name')]
                link = None
                pdf_url = None
                for l in e.get('links', []):
                    if l.get('type') == 'application/pdf':
                        pdf_url = l.get('href')
                    if l.get('rel') == 'alternate':
                        link = l.get('href')
                pub_year = None
                try:
                    if e.get('published'):
                        pub_year = int(e.get('published')[:4])
                except Exception:
                    pass
                w = Work(
                    id=arxiv_id or title or f"arxiv_{start}",
                    doi=None,  # arXiv may include doi in e.arxiv_doi sometimes; skipping here
                    title=title,
                    authors=authors,
                    year=pub_year,
                    venue='arXiv',
                    publisher='arXiv',
                    url=link or arxiv_id,
                    source='arxiv',
                    oa_url=pdf_url,
                    pdf_url=pdf_url,
                    is_oa=True,
                    abstract=abstract,
                    extra={'arxiv_primary_category': e.get('arxiv_primary_category', {}).get('term') if e.get('arxiv_primary_category') else None}
                )
                all_works.append(w)
            start += batch
            sleep_a_bit(0.8)
    return all_works

def openalex_search(queries: List[str], per_page=200, max_pages=5) -> List[Work]:
    results = []
    for q in queries:
        page = 1
        while page <= max_pages:
            params = {
                'search': q,
                'per_page': per_page,
                'page': page,
                'sort': 'cited_by_count:desc'
            }
            r = requests.get(OPENALEX_BASE, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            works = data.get('results', []) or []
            if not works:
                break
            for w in works:
                doi = doi_norm(w.get('doi'))
                title = norm(w.get('title'))
                year = w.get('publication_year')

                host_venue = (w.get('host_venue') or {})
                venue = host_venue.get('display_name')

                prim_loc = (w.get('primary_location') or {})
                src_obj = (prim_loc.get('source') or {})
                publisher = src_obj.get('publisher')

                url = prim_loc.get('landing_page_url') or w.get('id')

                oa = (w.get('open_access') or {})
                is_oa = oa.get('is_oa')
                oa_url = oa.get('oa_url')

                best = (w.get('best_oa_location') or {})
                pdf_url = best.get('url_for_pdf') or best.get('url')

                authors = []
                for auth in (w.get('authorships') or []):
                    author_obj = auth.get('author') or {}
                    dn = author_obj.get('display_name')
                    if dn:
                        authors.append(dn)

                abstract = None  # OpenAlex abstracts come inverted; skipped

                results.append(Work(
                    id=w.get('id') or (doi or title or f"openalex_{page}"),
                    doi=doi,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    publisher=publisher,
                    url=url,
                    source='openalex',
                    oa_url=oa_url,
                    pdf_url=pdf_url,
                    is_oa=is_oa,
                    abstract=abstract,
                    extra={'openalex': w.get('id')}
                ))
            page += 1
            sleep_a_bit(1.0)
    return results

def crossref_search(queries: List[str], rows=100, max_pages=5) -> List[Work]:
    items = []
    for q in queries:
        for page in range(max_pages):
            params = {
                'query': q,
                'rows': rows,
                'offset': page * rows
            }
            r = requests.get(CROSSREF_BASE, params=params, timeout=30, headers={'User-Agent': 'ODP-Harvester/1.0 (mailto:you@example.com)'})
            r.raise_for_status()
            data = r.json()
            message = data.get('message', {})
            works = message.get('items', [])
            if not works:
                break
            for w in works:
                doi = doi_norm(w.get('DOI'))
                title = norm(' '.join(w.get('title', []))) if w.get('title') else None
                year = None
                if w.get('issued', {}).get('date-parts'):
                    try:
                        year = int(w['issued']['date-parts'][0][0])
                    except Exception:
                        pass
                publisher = w.get('publisher')
                venue = None
                if w.get('container-title'):
                    venue = ' '.join(w.get('container-title'))
                url = w.get('URL')
                authors = []
                for a in w.get('author', []):
                    name = ' '.join([p for p in [a.get('given'), a.get('family')] if p])
                    if name:
                        authors.append(name)
                items.append(Work(
                    id=doi or title or url or f"crossref_{page}",
                    doi=doi,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    publisher=publisher,
                    url=url,
                    source='crossref',
                    oa_url=None,
                    pdf_url=None,
                    is_oa=None,
                    abstract=None,
                    extra={'crossref_type': w.get('type')}
                ))
            sleep_a_bit(1.0)
    return items

def unpaywall_enrich(works: List[Work], email: str) -> None:
    for w in works:
        if not w.doi:
            continue
        url = f"{UNPAYWALL_BASE}/{urllib.parse.quote(w.doi)}"
        params = {'email': email}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            d = r.json()
            w.is_oa = d.get('is_oa', w.is_oa)
            best = d.get('best_oa_location') or {}
            if best:
                w.oa_url = best.get('url') or w.oa_url
                w.pdf_url = best.get('url_for_pdf') or w.pdf_url
            # Fall back to any OA location
            if not w.pdf_url and d.get('oa_locations'):
                for loc in d['oa_locations']:
                    if loc.get('url_for_pdf'):
                        w.pdf_url = loc['url_for_pdf']
                        break
                    if loc.get('url'):
                        w.oa_url = w.oa_url or loc['url']
        except Exception as e:
            # Be resilient; continue
            pass
        sleep_a_bit(0.4)

def dedupe_works(works: List[Work]) -> List[Work]:
    seen = {}
    deduped = []
    for w in works:
        key = w.doi or w.id or (w.title or '').lower()
        if key in seen:
            # Merge sparse fields
            prev = seen[key]
            for field in ['title', 'authors', 'year', 'venue', 'publisher', 'url', 'oa_url', 'pdf_url', 'is_oa', 'abstract']:
                pv = getattr(prev, field)
                wv = getattr(w, field)
                if pv in (None, [], False) and wv not in (None, [], False):
                    setattr(prev, field, wv)
        else:
            seen[key] = w
            deduped.append(seen[key])
    return deduped

def looks_like_publisher_of_interest(publisher: Optional[str]) -> bool:
    if not publisher:
        return False
    p = publisher.lower()
    return any(h in p for h in PUBLISHER_HINTS)

def filter_publishers(works: List[Work]) -> List[Work]:
    return [w for w in works if looks_like_publisher_of_interest(w.publisher or '') or (w.venue and looks_like_publisher_of_interest(w.venue)) or (w.url and any(h in (w.url or '').lower() for h in ['acm.org','springer','ieee','elsevier','sciencedirect','wiley']))]

def attempt_institutional_pdf_download(url: str, session: requests.Session, outpath: Path) -> bool:
    """
    Tries to download a PDF from a publisher page (e.g., ACM DL) assuming
    you have institutional/VPN access. Only use this if you have the rights.
    """
    try:
        with session.get(url, timeout=45, allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            ctype = r.headers.get('Content-Type', '')
            if 'pdf' not in ctype.lower() and not url.lower().endswith('.pdf'):
                # Some sites require a special path for the PDF (e.g., dl.acm.org/doi/pdf/DOI)
                # Try a few common patterns opportunistically.
                if 'dl.acm.org/doi/' in url and '/pdf/' not in url:
                    maybe = url.replace('/doi/', '/doi/pdf/')
                    with session.get(maybe, timeout=45, allow_redirects=True, stream=True) as r2:
                        r2.raise_for_status()
                        ctype2 = r2.headers.get('Content-Type', '')
                        if 'pdf' in ctype2.lower() or maybe.lower().endswith('.pdf'):
                            with open(outpath, 'wb') as f:
                                for chunk in r2.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            return True
                return False
            with open(outpath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception:
        return False

def download_pdfs(works: List[Work], outdir: Path, email: str, attempt_institutional: bool = False) -> None:
    pdf_dir = outdir / "pdfs"
    ensure_dir(pdf_dir)
    s = requests.Session()
    headers = {'User-Agent': 'ODP-Harvester/1.0'}
    s.headers.update(headers)

    for w in works:
        filename_hint = pick_filename((w.title or w.doi or w.id or 'paper')) + ".pdf"
        outpath = pdf_dir / filename_hint

        if outpath.exists():
            continue

        # Prefer clearly OA pdf_url
        if w.pdf_url:
            try:
                with s.get(w.pdf_url, timeout=45, allow_redirects=True, stream=True) as r:
                    r.raise_for_status()
                    ctype = r.headers.get('Content-Type', '')
                    if 'pdf' in ctype.lower() or w.pdf_url.lower().endswith('.pdf'):
                        with open(outpath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        continue
            except Exception:
                pass

        # If OA flag true but only oa_url present, try follow
        if (w.is_oa and w.oa_url) and not outpath.exists():
            try:
                with s.get(w.oa_url, timeout=45, allow_redirects=True, stream=True) as r:
                    r.raise_for_status()
                    ctype = r.headers.get('Content-Type', '')
                    if 'pdf' in ctype.lower() or w.oa_url.lower().endswith('.pdf'):
                        with open(outpath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        continue
            except Exception:
                pass

        # Optional institutional attempt (e.g., ACM DL via VPN)
        if attempt_institutional and not outpath.exists() and w.url:
            ok = attempt_institutional_pdf_download(w.url, s, outpath)
            if ok:
                continue

        # arXiv fallback: direct PDF
        if w.source == 'arxiv' and w.pdf_url and not outpath.exists():
            try:
                with s.get(w.pdf_url, timeout=45, allow_redirects=True, stream=True) as r:
                    r.raise_for_status()
                    ctype = r.headers.get('Content-Type', '')
                    if 'pdf' in ctype.lower():
                        with open(outpath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        continue
            except Exception:
                pass

        sleep_a_bit(0.2)

def save_outputs(works: List[Work], outdir: Path) -> Tuple[Path, Path]:
    ensure_dir(outdir)
    # CSV
    csv_path = outdir / "metadata.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','doi','title','authors','year','venue','publisher','url','source','is_oa','oa_url','pdf_url','abstract'])
        for w in works:
            writer.writerow([
                w.id, w.doi, w.title, '; '.join(w.authors or []),
                w.year, w.venue, w.publisher, w.url, w.source,
                w.is_oa, w.oa_url, w.pdf_url, w.abstract
            ])
    # JSONL
    jsonl_path = outdir / "metadata.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for w in works:
            f.write(json.dumps(asdict(w), ensure_ascii=False) + '\n')
    return csv_path, jsonl_path

def main():
    parser = argparse.ArgumentParser(description="Harvest ODP papers (arXiv, OpenAlex, Crossref, Unpaywall).")
    parser.add_argument('--query', action='append', help='Search query (can be given multiple times). Default: ODP presets.', default=None)
    parser.add_argument('--email', required=True, help='Your email for Unpaywall (required by their TOS, e.g., you@kit.edu).')
    parser.add_argument('--outdir', default='./odp_results', help='Output directory.')
    parser.add_argument('--max_arxiv', type=int, default=200, help='Max arXiv results per query (total cap).')
    parser.add_argument('--openalex_pages', type=int, default=3, help='Max OpenAlex pages per query (per_page=200).')
    parser.add_argument('--crossref_pages', type=int, default=2, help='Max Crossref pages per query (rows=100).')
    parser.add_argument('--attempt_institutional_pdf', action='store_true', help='Try publisher PDFs via VPN if you have access rights (e.g., ACM).')
    parser.add_argument('--filter_big_publishers', action='store_true', help='Keep only works from large publishers (ACM, IEEE, Springer, Elsevier, etc.).')
    args = parser.parse_args()

    queries = args.query or ODP_DEFAULT_QUERIES
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"[1/6] Searching arXiv for {len(queries)} queries...")
    arxiv_works = arxiv_search(queries, max_results=args.max_arxiv)

    print(f"[2/6] Searching OpenAlex...")
    openalex_works = openalex_search(queries, max_pages=args.openalex_pages)

    print(f"[3/6] Searching Crossref...")
    crossref_works = crossref_search(queries, max_pages=args.crossref_pages)

    all_works = arxiv_works + openalex_works + crossref_works
    print(f"Found {len(all_works)} total candidate records before dedupe.")

    print(f"[4/6] Enriching with Unpaywall (email={args.email})...")
    unpaywall_enrich(all_works, email=args.email)

    print(f"[5/6] Deduplicating and filtering...")
    deduped = dedupe_works(all_works)
    if args.filter_big_publishers:
        deduped = filter_publishers(deduped)

    print(f"Kept {len(deduped)} unique records.")
    csv_path, jsonl_path = save_outputs(deduped, outdir)
    print(f"Saved metadata to:\n- {csv_path}\n- {jsonl_path}")

    print(f"[6/6] Downloading PDFs (OA only by default {'+ institutional attempt' if args.attempt_institutional_pdf else ''})...")
    download_pdfs(deduped, outdir, email=args.email, attempt_institutional=args.attempt_institutional_pdf)

    print("Done. Check the 'pdfs/' folder for downloaded files.")

if __name__ == '__main__':
    main()
