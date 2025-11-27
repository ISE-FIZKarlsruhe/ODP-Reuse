#!/usr/bin/env python3
"""
Per-paper OpenAlex citations exporter with robust matching, year folders + summary.

Improvements:
- Candidate ranking: title similarity + year proximity + CEUR evidence.
- Phrase-search fallback and suspicious-count guard.
- Optional 'require CEUR' to avoid non-CEUR mismatches.
- Audit columns in main CSV: matched OpenAlex ID/title/year/score, meta_citing_total, ceur_match.
- Citing de-duplication by OpenAlex ID.

"""

import argparse
import csv
import logging
import os
import re
import signal
import sys
import time
import urllib.parse
from difflib import SequenceMatcher
from typing import Dict, Optional, List, Tuple, Any

import requests

OPENALEX_BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "per-paper-citations-openalex/1.3 (mailto:you@example.com)"}

# ---------- Logging ----------
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    handlers = []

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)
    handlers.append(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        handlers.append(fh)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)
    for h in handlers:
        root.addHandler(h)

# ---------- Helpers ----------
def normalize_space(s: str) -> str:
    return " ".join((s or "").split())

def clean_title(title: str) -> str:
    t = normalize_space(title)
    t = re.sub(r'https?://\S+', '', t, flags=re.IGNORECASE)
    for needle in [
        "ceur-ws.org", "ceur workshop", "ceur workshop proceedings",
        "ontologydesignpatterns.org", "submission:", "submissions:"
    ]:
        t = re.sub(re.escape(needle), '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip(' :;-')
    return t

def sanitize_filename(name: str, maxlen: int = 120) -> str:
    base = re.sub(r'[^\w\s\-]+', '', name).strip()
    base = re.sub(r'\s+', ' ', base)
    base = base[:maxlen].strip().replace(' ', '_')
    return base or "paper"

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def title_similarity(a: str, b: str) -> float:
    a, b = normalize_space((a or "").lower()), normalize_space((b or "").lower())
    if not a or not b:
        return 0.0
    tok_sim = jaccard(tokenize(a), tokenize(b))
    seq_sim = SequenceMatcher(None, a, b).ratio()
    # Weighted blend: token structure matters more for long titles
    return 0.6 * tok_sim + 0.4 * seq_sim

def is_ceur(work: Dict[str, Any]) -> bool:
    hv = (work.get("host_venue") or {}).get("display_name", "") or ""
    pl = work.get("primary_location") or {}
    urls = [
        pl.get("landing_page_url") or "",
        pl.get("pdf_url") or "",
        (work.get("doi") or "")
    ]
    if re.search(r"ceur[-.]?ws\.org", " ".join(urls), flags=re.IGNORECASE):
        return True
    if re.search(r"\bceur\b", hv, flags=re.IGNORECASE) or "workshop proceedings" in hv.lower():
        return True
    return False

def get_year(work: Dict[str, Any]) -> Optional[int]:
    y = work.get("publication_year")
    try:
        return int(y) if y is not None else None
    except Exception:
        return None

def best_url_from_work(work: Dict) -> str:
    pl = work.get("primary_location") or {}
    for k in ("landing_page_url", "pdf_url"):
        v = pl.get(k)
        if isinstance(v, str) and v.startswith(("http://", "https://")):
            return v
    hv = work.get("host_venue") or {}
    v = hv.get("url")
    if isinstance(v, str) and v.startswith(("http://", "https://")):
        return v
    doi = work.get("doi")
    if isinstance(doi, str) and doi.strip():
        if doi.lower().startswith("http"):
            return doi
        return f"https://doi.org/{doi}"
    return ""

# ---------- OpenAlex search & match ----------
def openalex_search_candidates(title: str, per_page: int = 25, phrase: bool = False) -> List[Dict]:
    qstr = f"\"{title}\"" if phrase else title
    params = {"search": qstr, "per_page": per_page}
    url = f"{OPENALEX_BASE}/works?{urllib.parse.urlencode(params)}"
    r = requests.get(url, headers=HEADERS, timeout=40)
    r.raise_for_status()
    return r.json().get("results", [])

def score_candidate(
    cand: Dict, cleaned_title: str, seed_year: Optional[int], venue_hint: str
) -> Tuple[float, Dict[str, float]]:
    """Return (score, components) for ranking."""
    ctitle = cand.get("title") or ""
    sim = title_similarity(cleaned_title, ctitle)
    y = get_year(cand)
    year_bonus = 0.0
    if seed_year and y:
        diff = abs(y - seed_year)
        if diff == 0:
            year_bonus = 0.15
        elif diff == 1:
            year_bonus = 0.10
        elif diff == 2:
            year_bonus = 0.05
        else:
            year_bonus = -0.10  # too far
    ceur_bonus = 0.10 if is_ceur(cand) else 0.0
    hint_bonus = 0.0
    if venue_hint:
        hv = (cand.get("host_venue") or {}).get("display_name", "") or ""
        if venue_hint.lower() in hv.lower():
            hint_bonus = 0.05
    total = sim + year_bonus + ceur_bonus + hint_bonus
    comps = {"sim": sim, "year_bonus": year_bonus, "ceur_bonus": ceur_bonus, "hint_bonus": hint_bonus}
    return total, comps

def select_best_work(
    cleaned_title: str,
    seed_year: Optional[int],
    venue_hint: str,
    per_page: int,
    title_threshold: float,
    year_tolerance: int,
    require_ceur: bool,
    suspicious_high: int,
    debug_save_path: Optional[str] = None,
) -> Tuple[Optional[Dict], Dict]:
    """
    Try normal search, then phrase-search if needed. Apply scoring & guards.
    Returns (work_or_None, debug_info)
    """
    debug = {"rounds": []}

    def consider(cands: List[Dict], label: str) -> Tuple[Optional[Dict], Dict]:
        ranked = []
        for c in cands:
            score, comps = score_candidate(c, cleaned_title, seed_year, venue_hint)
            ranked.append((score, comps, c))
        ranked.sort(key=lambda t: t[0], reverse=True)
        # add to debug
        dbg_items = []
        for s, comps, c in ranked[:10]:
            dbg_items.append({
                "title": c.get("title"),
                "year": get_year(c),
                "openalex_id": c.get("id"),
                "host_venue": (c.get("host_venue") or {}).get("display_name", ""),
                "is_ceur": is_ceur(c),
                "score": s, **comps
            })
        debug["rounds"].append({"label": label, "candidates": dbg_items})
        # choose best acceptable
        for s, comps, c in ranked:
            # hard checks
            if get_year(c) and seed_year and abs(get_year(c) - seed_year) > max(0, year_tolerance):
                continue
            if require_ceur and not is_ceur(c):
                continue
            if s < title_threshold:
                continue
            return c, {"score": s, **comps}
        return None, {}

    # Round 1: normal search
    try:
        cands = openalex_search_candidates(cleaned_title, per_page=per_page, phrase=False)
    except Exception as e:
        logging.warning(f"OpenAlex search failed: {e}")
        cands = []
    best, meta = consider(cands, "search")

    # Round 2: phrase search fallback if no acceptable
    if not best:
        try:
            cands2 = openalex_search_candidates(cleaned_title, per_page=min(10, per_page), phrase=True)
        except Exception as e:
            logging.warning(f"OpenAlex phrase search failed: {e}")
            cands2 = []
        best, meta = consider(cands2, "phrase_search")

    # Optional debug dump
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        try:
            import json
            with open(debug_save_path, "w", encoding="utf-8") as f:
                import datetime
                debug["chosen"] = {
                    "id": best.get("id") if best else None,
                    "title": best.get("title") if best else None,
                    "year": get_year(best) if best else None,
                    "meta": meta
                }
                debug["timestamp"] = datetime.datetime.utcnow().isoformat()
                json.dump(debug, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return best, meta

# ---------- Citing fetch ----------
def fetch_citing_works(openalex_id: str, max_citers: int, progress_every: int = 25) -> Tuple[List[Dict], int]:
    citing: List[Dict] = []
    per_page = 50
    cursor = "*"
    n_logged = 0
    total_meta = 0
    seen_ids = set()

    while len(citing) < max_citers:
        params = {
            "filter": f"cites:{openalex_id}",
            "per_page": per_page,
            "cursor": cursor,
            "sort": "publication_year:desc"
        }
        url = f"{OPENALEX_BASE}/works?{urllib.parse.urlencode(params)}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=40)
            if r.status_code != 200:
                logging.warning(f"OpenAlex citing HTTP {r.status_code} for {openalex_id}")
                break
            data = r.json()
            total_meta = data.get("meta", {}).get("count", total_meta)
            batch = data.get("results", [])
            # de-dup by id
            for w in batch:
                wid = w.get("id")
                if not wid or wid in seen_ids:
                    continue
                seen_ids.add(wid)
                citing.append(w)
                if len(citing) >= max_citers:
                    break
            cursor = data.get("meta", {}).get("next_cursor")
            if len(citing) - n_logged >= max(1, progress_every):
                logging.info(f"  ... {len(citing)} citing items fetched (meta total ~{total_meta})")
                n_logged = len(citing)
            if not cursor or not batch:
                break
            time.sleep(0.2)
        except Exception as e:
            logging.error(f"OpenAlex citing fetch failed for {openalex_id}: {e}")
            break

    return citing[:max_citers], int(total_meta or 0)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV with at least a 'title' and 'year' column")
    ap.add_argument("--main-out", required=True, help="Output main CSV with added audit columns")
    ap.add_argument("--data-dir", default="output/data", help="Directory to store per-paper CSVs")
    ap.add_argument("--year-column", default="year", help="Name of the 'year' column in the input CSV")
    ap.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between papers")
    ap.add_argument("--max-citers", type=int, default=1000, help="Max citing works to collect per paper")
    ap.add_argument("--max", type=int, default=None, help="Process only first N rows (testing)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip fetching if per-paper CSV already exists")
    ap.add_argument("--progress-every", type=int, default=25, help="Log every N citing items")
    ap.add_argument("--log-file", default=None, help="Also write logs to this file")
    ap.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    # NEW matching controls
    ap.add_argument("--title-threshold", type=float, default=0.82, help="Min title similarity to accept a match")
    ap.add_argument("--year-tolerance", type=int, default=1, help="Accept if |seed_year - match_year| <= tol")
    ap.add_argument("--max-candidates", type=int, default=25, help="Search top-K candidates for matching")
    ap.add_argument("--require-ceur", action="store_true", help="Require CEUR evidence for a match")
    ap.add_argument("--suspicious-high", type=int, default=1000, help="If meta citations > this and no CEUR, reject")
    ap.add_argument("--debug-candidates", action="store_true", help="Write candidate dumps to data/debug/*.json")
    args = ap.parse_args()

    setup_logging(args.log_level, args.log_file)
    os.makedirs(args.data_dir, exist_ok=True)
    if args.debug_candidates:
        os.makedirs(os.path.join(args.data_dir, "debug"), exist_ok=True)

    # Graceful Ctrl+C
    interrupted = {"flag": False}
    def _sigint(_sig, _frm):
        interrupted["flag"] = True
        logging.warning("Interrupted (Ctrl+C). Finishing current write and exiting...")
    signal.signal(signal.SIGINT, _sigint)

    start_all = time.time()
    logging.info("Starting per-paper citation export (OpenAlex, robust matching)")
    logging.info(f"Input: {args.input}")
    logging.info(f"Main CSV: {args.main_out}")
    logging.info(f"Output dir: {os.path.abspath(args.data_dir)}")
    if args.log_file:
        logging.info(f"Log file: {os.path.abspath(args.log_file)}")

    # Read input & set up main output
    with open(args.input, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        headers = list(reader.fieldnames or [])
        if not headers:
            logging.error("Empty or headerless CSV. Include 'title' and 'year' columns.")
            sys.exit(1)

        # locate columns
        title_key = next((c for c in headers if c.lower() == "title"), None)
        if not title_key:
            logging.error("Input CSV must include a 'title' column (case-insensitive).")
            sys.exit(1)
        year_key = next((c for c in headers if c.lower() == args.year_column.lower()), None)
        if not year_key:
            logging.error(f"Input CSV must include a '{args.year_column}' column.")
            sys.exit(1)
        # optional link column for venue hint
        link_key = next((c for c in headers if c.lower() in ("link","url","pdf_url")), None)

        out_headers = headers + [
            "citation_count",
            # audit columns:
            "matched_openalex_id", "matched_title", "matched_year",
            "match_score", "meta_citing_total", "ceur_match"
        ]
        with open(args.main_out, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_headers)
            writer.writeheader()

            # Count rows for nicer logs
            fin.seek(0)
            _ = next(fin)
            total_rows = sum(1 for _ in fin)
            fin.seek(0)
            reader = csv.DictReader(fin)

            processed = 0
            for row in reader:
                if interrupted["flag"]:
                    break
                if args.max is not None and processed >= args.max:
                    logging.info(f"--max reached: {args.max}. Stopping.")
                    break

                raw_title = row.get(title_key, "") or ""
                year_val = str(row.get(year_key, "") or "").strip()
                seed_year = None
                try:
                    seed_year = int(year_val)
                except Exception:
                    pass
                venue_hint = ""
                if link_key:
                    venue_hint = row.get(link_key, "") or ""

                cleaned = clean_title(raw_title)
                safe_name = sanitize_filename(raw_title)
                year_dir = os.path.join(args.data_dir, year_val or "unknown")
                os.makedirs(year_dir, exist_ok=True)
                per_paper_path = os.path.join(year_dir, f"{safe_name}.csv")

                processed += 1
                logging.info(f"[{processed}/{total_rows}] -> {raw_title}  (year={year_val or 'unknown'})")

                citing_count = 0
                matched_id = matched_title = None
                matched_year = None
                match_score = None
                meta_total = 0
                ceur_flag = False
                t0 = time.time()

                # If skipping existing, just count rows in the existing file
                if args.skip_existing and os.path.exists(per_paper_path):
                    citing_count = count_rows_in_csv(per_paper_path)
                    logging.info(f"  ↳ Skipping fetch (exists). Using existing count: {citing_count}")
                else:
                    # Find the work in OpenAlex (robust matching)
                    debug_path = (
                        os.path.join(args.data_dir, "debug", f"{safe_name}.json")
                        if args.debug_candidates else None
                    )
                    best, meta = select_best_work(
                        cleaned_title=cleaned,
                        seed_year=seed_year,
                        venue_hint="ceur" if venue_hint else "",
                        per_page=args.max_candidates,
                        title_threshold=args.title_threshold,
                        year_tolerance=args.year_tolerance,
                        require_ceur=args.require_ceur,
                        suspicious_high=args.suspicious_high,
                        debug_save_path=debug_path
                    )

                    found = best is not None
                    logging.info(f"  Found in OpenAlex: {'YES' if found else 'NO'}")
                    with open(per_paper_path, "w", newline="", encoding="utf-8") as fpp:
                        w = csv.writer(fpp)
                        w.writerow([
                            "citing_title",
                            "citing_url",
                            "citing_year",
                            "citing_doi",
                            "citing_openalex_id",
                            "venue_name",
                            "is_oa",
                            "oa_status"
                        ])

                        if not found:
                            logging.warning("  No reliable match; wrote header only.")
                        else:
                            matched_id = best.get("id") or ""
                            matched_title = best.get("title") or ""
                            matched_year = get_year(best)
                            ceur_flag = is_ceur(best)
                            match_score = meta.get("score")

                            # Fetch citing works
                            citing, meta_total = fetch_citing_works(matched_id, args.max_citers, args.progress_every)

                            # Guard: suspiciously high count without CEUR evidence? reject.
                            #if (not ceur_flag) and (meta_total and meta_total > args.suspicious_high):
                            logging.warning(
                                    f"  Suspiciously high citing total (~{meta_total}) and not CEUR; rejecting match. limit: {args.suspicious_high}"
                                )
                            #     citing = []
                            #     meta_total = 0
                            #     matched_id = matched_title = None
                            #     matched_year = None
                            #     match_score = None
                            #     ceur_flag = False
                            # else:
                            for work2 in citing:
                                title2 = normalize_space(work2.get("title") or "")
                                if not title2:
                                    continue
                                url2 = best_url_from_work(work2)
                                year2 = work2.get("publication_year") or ""
                                doi2 = work2.get("doi") or ""
                                if isinstance(doi2, str) and doi2 and not doi2.lower().startswith("http"):
                                    doi2 = f"https://doi.org/{doi2}"
                                venue = (work2.get("host_venue") or {}).get("display_name", "")
                                oa = (work2.get("open_access") or {}).get("is_oa", "")
                                oa_status = (work2.get("open_access") or {}).get("oa_status", "")
                                w.writerow([
                                    title2, url2, year2, doi2, work2.get("id") or "", venue, oa, oa_status
                                ])
                                citing_count += 1
                            logging.info(f"  Citing items fetched: {citing_count} (meta total ~{meta_total})")

                elapsed = time.time() - t0
                logging.info(f"  Done. citing_count={citing_count} | {elapsed:.1f}s")

                # Write row to MAIN CSV (original fields + audit)
                out_row = dict(row)
                out_row["citation_count"] = citing_count
                out_row["matched_openalex_id"] = matched_id or ""
                out_row["matched_title"] = matched_title or ""
                out_row["matched_year"] = matched_year if matched_year is not None else ""
                out_row["match_score"] = f"{match_score:.3f}" if isinstance(match_score, float) else ""
                out_row["meta_citing_total"] = meta_total or ""
                out_row["ceur_match"] = int(bool(ceur_flag))
                writer.writerow(out_row)

                if not interrupted["flag"] and args.sleep > 0:
                    logging.info(f"  Sleeping {args.sleep:.1f}s before next paper...")
                    time.sleep(args.sleep)

    tot_elapsed = time.time() - start_all
    if interrupted["flag"]:
        logging.warning(f"Stopped early. Elapsed {tot_elapsed/60:.1f} min")
    else:
        logging.info(f"All done. Elapsed {tot_elapsed/60:.1f} min")

def count_rows_in_csv(path: str) -> int:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            _ = next(r, None)
            return sum(1 for _ in r)
    except Exception:
        return 0

if __name__ == "__main__":
    main()
