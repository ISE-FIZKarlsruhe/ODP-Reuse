#!/usr/bin/env python3
import argparse, csv, os, re, shutil, sys, time
from pathlib import Path
from typing import Optional, Tuple
import requests

UA = "odp-github-dataset/1.1"
TIMEOUT = 60
RETRY = 3
SLEEP_BETWEEN = 0.8
RAW_HOST = "raw.githubusercontent.com"

def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

def sanitize_label(label: str) -> str:
    s = (label or "").strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._\-+]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_.")
    return s or "unknown"

def guess_ext(url_path: str, details: str) -> str:
    for s in (url_path, details or ""):
        m = re.search(r'\.([A-Za-z0-9]{2,6})(?:$|[?#])', s)
        if m:
            ext = m.group(1).lower()
            if ext in ("owl","ttl","rdf","xml","nt","trig"):
                return "." + ext
    return ".owl"

def github_blob_to_raw(u: str) -> Optional[str]:
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)$", u)
    if m:
        owner, repo, ref, path = m.groups()
        return f"https://{RAW_HOST}/{owner}/{repo}/{ref}/{path}"
    return None

def ensure_unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix, parent = base.stem, base.suffix, base.parent
    i = 2
    while True:
        cand = parent / f"{stem}-{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1

def download_file(url: str, out_path: Path) -> Tuple[bool, str, Optional[str]]:
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    u = github_blob_to_raw(url) or url
    last_err = ""
    for attempt in range(1, RETRY + 1):
        try:
            r = session.get(u, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code in (429, 403):
                last_err = f"{r.status_code} {r.reason}"
                time.sleep(2 * attempt); continue
            if r.status_code != 200:
                last_err = f"{r.status_code} {r.reason}"
                time.sleep(attempt); continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True, f"downloaded ({len(r.content)} bytes)", str(out_path)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(attempt)
    return False, last_err, None

def find_pattern_files_by_iri(patterns_repo: Path, pattern_iri: str):
    hits = []
    if not patterns_repo.is_dir() or not pattern_iri:
        return hits
    for p in patterns_repo.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".owl",".ttl"):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    if pattern_iri in f.read():
                        hits.append(p)
            except Exception:
                pass
    return hits

def main():
    ap = argparse.ArgumentParser(description="Build local dataset from GitHub reuse CSV.")
    ap.add_argument("--csv", required=True, help="Path to report/github.csv")
    ap.add_argument("--out-dir", default="github_dataset", help="Root output directory")
    ap.add_argument("--patterns-repo", required=True, help="Local patterns-repository path")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_root = Path(args.out_dir)
    patt_repo = Path(args.patterns_repo)
    if not csv_path.is_file():
        log(f"[!] CSV not found: {csv_path}"); sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "download_log.csv"
    with open(csv_path, newline="", encoding="utf-8") as fin, \
         open(log_path, "w", newline="", encoding="utf-8") as flog:

        reader = csv.DictReader(fin)
        logwriter = csv.DictWriter(flog, fieldnames=[
            "pattern_label","pattern_iri","match_label","sanitized_label",
            "url","saved_as","pattern_files_copied","status","message"
        ])
        logwriter.writeheader()

        n = 0
        for row in reader:
            if (row.get("source") or "").lower() != "github":
                continue

            pattern_label = (row.get("pattern_label") or "").strip()
            pattern_iri   = (row.get("pattern_iri") or "").strip()
            match_label   = (row.get("match_label") or "").strip()
            url           = (row.get("url") or "").strip()
            details       = (row.get("details") or "").strip()
            if not match_label or not url:
                continue

            repo_s = sanitize_label(match_label)
            repo_dir = out_root / repo_s
            patt_dir = repo_dir / "patterns"
            repo_dir.mkdir(parents=True, exist_ok=True)
            patt_dir.mkdir(parents=True, exist_ok=True)

            ext = guess_ext(url, details)
            target = ensure_unique_path(repo_dir / f"{repo_s}{ext}")

            ok, msg, saved_path = download_file(url, target)

            hits = find_pattern_files_by_iri(patt_repo, pattern_iri)
            for p in hits:
                dst = ensure_unique_path(patt_dir / p.name)
                try:
                    shutil.copy2(p, dst)
                except Exception as e:
                    log(f"[copy] failed {p} -> {dst}: {e}")

            logwriter.writerow({
                "pattern_label": pattern_label,
                "pattern_iri": pattern_iri,
                "match_label": match_label,
                "sanitized_label": repo_s,
                "url": url,
                "saved_as": saved_path or "",
                "pattern_files_copied": len(hits),
                "status": "ok" if ok else "error",
                "message": msg
            })

            n += 1
            if args.limit and n >= args.limit:
                break
            time.sleep(SLEEP_BETWEEN)

    log(f"[✓] Done. Log written to: {log_path}")
    log(f"[i] Dataset root: {out_root}")

if __name__ == "__main__":
    main()
