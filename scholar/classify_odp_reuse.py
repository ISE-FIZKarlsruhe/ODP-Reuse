#!/usr/bin/env python3
"""
Classify citing papers for Ontology Design Pattern (ODP) reuse.

Now includes:
- --pattern-library-dir (ODP names from folders)
- --log-dir (writes timestamped run logs)
- Simplified rules: only 'Possibly' or 'Unlikely'
- Reuse gate = reuse terms + ontology + pattern(s) in title/abstract
- No window hits or WOP hints
- Writes per-paper, summary, and all-classified CSVs
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
import urllib.parse
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

OPENALEX_BASE = "https://api.openalex.org"
HEADERS = {"User-Agent": "odp-reuse-classifier/2.1 (mailto:you@example.com)"}


# ---------- Console encoding ----------
def _force_utf8_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------- Logging ----------------
def setup_logging(level: str = "INFO", log_dir: Optional[str] = None):
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)
    root.addHandler(sh)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"classify_odp_reuse.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        logging.info(f"Logging to file: {os.path.abspath(log_path)}")


# ---------------- Configurable signals ----------------
ODP_KEYPHRASES = [
    r"ontology design pattern(s)?",
    r"\bodp(s)?\b",
    r"content ontology design pattern(s)?",
    r"pattern-based ontolog(y|ies)",
    r"ontology pattern(s)?",
    r"design pattern(s)? for ontolog(y|ies)",
    r"\bopla\b",
]

NEG_SOFTWARE = [
    r"software design pattern(s)?", r"programming (design )?pattern(s)?", r"goF pattern(s)?"
]

ONTOLOGY_WORD = r"\bontology\b"
PATTERN_WORDS = [r"\bpattern\b", r"\bpatterns\b"]

REUSE_TERMS = [
    r"\breuse\b", r"\bre-used\b", r"\breused\b", r"\breusing\b", r"\buse\b",
    r"\butilize\b", r"\brework\b", r"\brewrite\b", r"\breimplement\b", r"\brecycle\b",
    r"\bharness\b", r"\bmodularize\b", r"\breapply\b", r"\bemploy\b",
    r"\binstantiate\b", r"\binstantiated\b", r"\binstantiation\b",
    r"\badopt\b", r"\badopted\b", r"\badapting\b", r"\badapted\b",
    r"\bapply\b", r"\bapplied\b", r"\bapplying\b",
    r"\bextend\b", r"\bextended\b", r"\bextending\b",
    r"\bbased on\b", r"\bbuild on\b", r"\bbuilt on\b", r"\bleverage\b", r"\bleveraged\b",
]


# ---------------- OpenAlex helpers ----------------
def openalex_get_by_doi(doi: str) -> Optional[Dict]:
    if not doi:
        return None
    doi_norm = doi.strip()
    if doi_norm and not doi_norm.lower().startswith("http"):
        doi_norm = "https://doi.org/" + doi_norm
    url = f"{OPENALEX_BASE}/works/doi:{urllib.parse.quote(doi_norm, safe='')}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def openalex_search_by_title(title: str) -> Optional[Dict]:
    if not title:
        return None
    params = {"search": title, "per_page": 1}
    url = f"{OPENALEX_BASE}/works?{urllib.parse.urlencode(params)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            results = r.json().get("results", [])
            return results[0] if results else None
    except Exception:
        return None
    return None


def extract_text_fields(work: Dict) -> Tuple[str, str]:
    title = work.get("title") or ""
    inv = work.get("abstract_inverted_index") or {}
    if not inv or not isinstance(inv, dict):
        return title, (work.get("abstract") or "")
    pos_to_word = {}
    for w, poss in inv.items():
        for p in poss:
            pos_to_word[p] = w
    if not pos_to_word:
        return title, ""
    words = [pos_to_word.get(i, "") for i in range(max(pos_to_word.keys()) + 1)]
    return title, " ".join(words)


# ---------------- Text utils ----------------
def fold(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.lower()


def normalized(text: str) -> str:
    return " ".join((text or "").split())


# ---------------- Compile robust name patterns ----------------
def _name_tokens(name: str) -> List[str]:
    return [t for t in re.split(r"[\s\-_/]+", name) if t]


def build_name_regex(name: str) -> re.Pattern:
    toks = _name_tokens(name)
    core = r"[\s\-_]*".join(map(re.escape, toks))
    return re.compile(rf"(?<!\w){core}(?!\w)", flags=re.IGNORECASE)


def load_odp_names_from_dir(directory: str) -> List[str]:
    names: List[str] = []
    if directory and os.path.isdir(directory):
        for entry in os.scandir(directory):
            if entry.is_dir():
                names.append(entry.name)
    names.sort()
    return names


# ---------------- Core logic ----------------
def contains_any_regex(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def reuse_gate(title: str, abstract: str) -> bool:
    t = title or ""
    a = abstract or ""
    ta = f"{t}\n{a}"
    return (
        contains_any_regex(REUSE_TERMS, ta)
        and bool(re.search(ONTOLOGY_WORD, ta, flags=re.IGNORECASE))
        and contains_any_regex(PATTERN_WORDS, ta)
    )


def build_name_patterns(names: List[str]) -> List[Tuple[str, re.Pattern]]:
    return [(n, build_name_regex(n)) for n in names]


def classify_label(title: str, abstract: str, url: str, odp_name_patterns: List[Tuple[str, re.Pattern]]):
    text = f"{title}\n{abstract}\n{url or ''}"
    url_l = fold(url or "")

    hit_keyphrase = int(contains_any_regex(ODP_KEYPHRASES, text))
    hit_odp_url = int(("ontologydesignpatterns.org" in url_l) or ("ontologydesignpatterns.org" in fold(text)))

    matched_names = [raw for raw, pat in odp_name_patterns if pat.search(text)]
    hit_named_odp = int(bool(matched_names))

    neg_software = int(contains_any_regex(NEG_SOFTWARE, text) and ("ontology" not in fold(text)))

    label = "Possibly" if reuse_gate(title, abstract) and (hit_keyphrase or hit_odp_url or hit_named_odp) and not neg_software else "Unlikely"

    flags = {
        "hit_keyphrase": hit_keyphrase,
        "hit_odp_url": hit_odp_url,
        "hit_named_odp": hit_named_odp,
        "hit_neg_software": neg_software,
    }
    return label, sorted(set(matched_names)), flags


# ---------------- File discovery ----------------
def list_per_paper_csvs(root: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(root):
        return files
    for entry in os.scandir(root):
        if entry.is_dir():
            for f in os.scandir(entry.path):
                if f.is_file() and f.name.lower().endswith(".csv") and not f.name.endswith("-classified.csv"):
                    files.append(f.path)
        elif entry.is_file() and entry.name.lower().endswith(".csv") and not entry.name.endswith("-classified.csv"):
            files.append(entry.path)
    files.sort()
    return files


# ---------------- Main ----------------
def main():
    _force_utf8_console()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pattern-library-dir", required=True)
    ap.add_argument("--log-dir", default="logs", help="Directory to store run logs")
    ap.add_argument("--out-summary", default="odp_reuse_summary.csv")
    ap.add_argument("--out-all-classified", default="odp_reuse_summary-classified.csv")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--log", default="INFO")
    ap.add_argument("--max-per-paper", type=int, default=None)
    args = ap.parse_args()

    setup_logging(args.log, args.log_dir)

    odp_names = load_odp_names_from_dir(args.pattern_library_dir)
    if not odp_names:
        logging.error(f"No ODP pattern folders found in {args.pattern_library_dir}")
        sys.exit(1)
    odp_patterns = build_name_patterns(odp_names)
    logging.info(f"Loaded {len(odp_names)} ODP names from {args.pattern_library_dir}")

    per_paper_files = list_per_paper_csvs(args.data_dir)
    if not per_paper_files:
        logging.error(f"No CSV files found under {args.data_dir}")
        sys.exit(1)

    summary_rows = []
    all_classified_rows = []

    for path in per_paper_files:
        logging.info(f"Processing {path}")
        out_path = path[:-4] + "-classified.csv"

        with open(path, newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            rows = list(reader)
            headers = reader.fieldnames or []

        out_headers = headers + [
            "abstract_fetched", "venue_name_openalex", "odpreuse_label",
            "hit_keyphrase", "hit_odp_url", "hit_named_odp", "hit_neg_software", "matched_odp_names",
        ]

        num_possibly = num_unlikely = 0
        processed = 0

        with open(out_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_headers)
            writer.writeheader()

            for row in rows:
                doi = (row.get("citing_doi") or "").strip()
                title = normalized(row.get("citing_title") or "")
                url = (row.get("citing_url") or "").strip()

                work = openalex_get_by_doi(doi) if doi else None
                if work is None and title:
                    work = openalex_search_by_title(title)

                oa_title, abstract, venue = "", "", ""
                if work:
                    oa_title, abstract = extract_text_fields(work)
                    venue = (work.get("host_venue") or {}).get("display_name", "")

                title_for_scoring = title or oa_title
                label, names, flags = classify_label(title_for_scoring, abstract, url, odp_patterns)

                if label == "Possibly":
                    num_possibly += 1
                else:
                    num_unlikely += 1

                out = dict(row)
                out.update({
                    "abstract_fetched": int(bool(work and abstract)),
                    "venue_name_openalex": venue,
                    "odpreuse_label": label,
                    "hit_keyphrase": flags["hit_keyphrase"],
                    "hit_odp_url": flags["hit_odp_url"],
                    "hit_named_odp": flags["hit_named_odp"],
                    "hit_neg_software": flags["hit_neg_software"],
                    "matched_odp_names": "; ".join(names),
                })
                writer.writerow(out)
                all_classified_rows.append(out)

                processed += 1
                if args.max_per_paper and processed >= args.max_per_paper:
                    break
                time.sleep(args.sleep)

        summary_rows.append({
            "paper_file": os.path.relpath(path, start=args.data_dir),
            "classified_file": os.path.relpath(out_path, start=args.data_dir),
            "num_citing": len(rows),
            "num_possibly": num_possibly,
            "num_unlikely": num_unlikely,
        })

    # Write summary files
    with open(args.out_summary, "w", newline="", encoding="utf-8") as fsum:
        writer = csv.DictWriter(fsum, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    if all_classified_rows:
        # Build a stable union of all keys across rows
        all_keys = []
        seen = set()
        for r in all_classified_rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        # (optional) put some important columns first, then the rest
        preferred = [
            "paper_file", "classified_file", "citing_doi", "citing_title", "citing_url",
            "venue_name_openalex", "abstract_fetched", "odpreuse_label",
            "hit_keyphrase", "hit_odp_url", "hit_named_odp", "hit_neg_software",
            "matched_odp_names",
        ]
        ordered = [k for k in preferred if k in seen] + [k for k in all_keys if k not in set(preferred)]

        with open(args.out_all_classified, "w", newline="", encoding="utf-8") as fall:
            writer = csv.DictWriter(fall, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            for r in all_classified_rows:
                # Fill missing columns with ""
                row_norm = {k: r.get(k, "") for k in ordered}
                writer.writerow(row_norm)


    logging.info(f"Processed {len(per_paper_files)} files.")
    logging.info(f"Summary written to: {args.out_summary}")
    logging.info(f"All results written to: {args.out_all_classified}")


if __name__ == "__main__":
    main()

# python -u classify_odp_reuse.py --data-dir output/main_papers --pattern-library-dir C:\Users\eno\Documents\PhD\ODPs_Reused\MultiSource\patterns-repository --log-dir logs --out-summary main_papers_reuse_summary.csv --log INFO
