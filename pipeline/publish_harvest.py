#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_harvest.py -- turn a merged harvest into a new, fully recorded version.

publish-release.yml runs this unattended after every harvest that added
something, so it carries the checks a reviewer would have made:

  check         the update only ADDS: every released MANIFEST row, view pair,
                view ontology and audit record is still there, byte for byte,
                and every new MANIFEST row is one merge_harvest.py reported
  version       the next minor version, from CITATION.cff
  latest        the newest Zenodo version of the concept
  fetch         download that version's corpus.zip and corpus-hashed.zip,
                check their MD5 against Zenodo, unpack them
  add-files     copy the added ontologies into the unpacked corpus; refuses if
                any is already there or does not match its recorded hash
  merge-hashed  add the newly hashed files to corpus-hashed/, appending to
                format_map.csv and hash_failures.csv so earlier rows stay as
                they are
  zip           re-pack a tree with the archive's layout (<prefix>/<portion>/...)
  record        write the version down: CITATION.cff, CHANGELOG.md,
                release-iri/history.csv, release-iri/archive.json,
                release-iri/additions/v<version>.csv and the release notes

history.csv holds one row per version (counts per view, what the version
added, its DOI); the site's changelog page draws the dataset's evolution from
it. archive.json names the Zenodo record and git tag of the newest version;
reproduce.yml checks that pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from odpqual.gate import main_ontology_file, sha256  # noqa: E402

VIEWS = ("github-reuse", "github-complete", "noOBO-reuse", "noOBO-complete",
         "OBO-reuse", "OBO-complete")
HISTORY_COLUMNS = (["version", "date", "tag", "zenodo_record", "doi", "discovered", "kept"]
                   + [f"{v}_{m}" for v in VIEWS for m in ("ontologies", "pairs")]
                   + ["added_ontologies", "added_kept", "added_pairs", "harvest_run", "summary"])
ZENODO = "https://zenodo.org"
ARCHIVE_FILES = ("corpus.zip", "corpus-hashed.zip")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def num(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


# -------------------------------------------------------------------- check

def _table_rows(path: Path) -> Counter:
    return Counter(tuple(sorted(r.items())) for r in read_csv(path)) if path.is_file() else Counter()


def check_additive(old: Path, new: Path, added: set[tuple[str, str]]) -> list[str]:
    """Problems that make the update more than an addition; [] if none."""
    problems: list[str] = []
    o = {(r["dataset"], r["ontology_id"]): r for r in read_csv(old / "MANIFEST.csv")}
    n = {(r["dataset"], r["ontology_id"]): r for r in read_csv(new / "MANIFEST.csv")}
    for k, r in o.items():
        if k not in n:
            problems.append(f"MANIFEST.csv: release row {k[0]}/{k[1]} is missing")
        elif n[k] != r:
            diff = sorted(c for c in r if r[c] != n[k].get(c))
            problems.append(f"MANIFEST.csv: release row {k[0]}/{k[1]} changed ({', '.join(diff)})")
    for k in sorted(set(n) - set(o) - added):
        problems.append(f"MANIFEST.csv: {k[0]}/{k[1]} is new but the merge did not report it")
    for k in sorted(added - set(n)):
        problems.append(f"MANIFEST.csv: the merge reported {k[0]}/{k[1]} but it is absent")
    lower: dict[tuple[str, str], str] = {}
    for ds, oid in sorted(n, key=lambda k: (k not in o, k)):   # released ids first
        first = lower.setdefault((ds, oid.lower()), oid)
        if first != oid:
            problems.append(f"MANIFEST.csv: {ds}/{oid} differs only in letter case from "
                            f"{ds}/{first}; Windows and macOS cannot unpack both directories")

    # The row tables must keep every released row. statistics.csv and
    # top_patterns.csv are summaries recomputed from them and change by design.
    tables = sorted(p.relative_to(old) for p in list(old.glob("views/*/pairs.csv"))
                    + list(old.glob("views/*/ontologies.csv"))
                    + list(old.glob("audit/removed_*.csv")))
    for rel in tables:
        before, after = _table_rows(old / rel), _table_rows(new / rel)
        lost = before - after
        if lost:
            example = dict(next(iter(lost))).get("ontology_id", "?")
            problems.append(f"{rel.as_posix()}: {sum(lost.values())} release row(s) missing or "
                            f"changed (e.g. {example})")
        if rel.parts[0] == "views" and rel.name in ("pairs.csv", "ontologies.csv"):
            ds = rel.parts[1].split("-")[0]
            extra = {dict(t).get("ontology_id") or "" for t in after - before}
            stray = sorted(e for e in extra if (ds, e) not in added)
            if stray:
                problems.append(f"{rel.as_posix()}: rows for {stray[0]} and {len(stray) - 1} other "
                                "ontologies the merge did not add")
    return problems


# ------------------------------------------------------------------ version

def next_minor(version: str) -> str:
    major, minor, _patch = (int(x) for x in version.lstrip("v").split("."))
    return f"{major}.{minor + 1}.0"


def citation_version(text: str) -> str:
    m = re.search(r"^version:\s*['\"]?([0-9.]+)", text, re.M)
    if not m:
        raise ValueError("CITATION.cff has no version")
    return m.group(1)


def set_citation(text: str, version: str, date: str) -> str:
    text = re.sub(r"^version:.*$", f"version: {version}", text, count=1, flags=re.M)
    return re.sub(r"^date-released:.*$", f"date-released: '{date}'", text, count=1, flags=re.M)


# ------------------------------------------------------- history, changelog

def history_row(release: Path, *, version: str, date: str, tag: str, record: str, doi: str,
                previous: dict, added: list[dict], harvest_run: str, summary: str = "") -> dict:
    rows = read_csv(release / "MANIFEST.csv")
    stats = {r["dataset"]: r for r in read_csv(release / "dataset_statistics.csv")}
    row = {"version": version, "date": date, "tag": tag, "zenodo_record": record, "doi": doi,
           "discovered": len(rows), "kept": sum(1 for r in rows if not r["drop_reason"])}
    for v in VIEWS:
        s = stats.get(v, {})
        row[f"{v}_ontologies"] = num(s.get("n_ontologies"))
        row[f"{v}_pairs"] = num(s.get("n_pairs"))
    kept = sum(1 for r in added if not r.get("drop_reason"))
    pairs = sum(row[f"{v}_pairs"] - num(previous[f"{v}_pairs"]) for v in VIEWS
                if v.endswith("-reuse") and previous.get(f"{v}_pairs"))
    row.update({"added_ontologies": len(added), "added_kept": kept, "added_pairs": pairs,
                "harvest_run": harvest_run,
                "summary": summary or f"+{kept} gate-passing ontologies, +{pairs} pairs "
                                      f"from the harvest of {date}"})
    return row


def changelog_entry(row: dict, previous: dict) -> str:
    v, date = row["version"], row["date"]
    added, kept, pairs = (num(row["added_ontologies"]), num(row["added_kept"]),
                          num(row["added_pairs"]))
    L = [f"## v{v} — {date}", "",
         f"**+{kept} ontologies · +{pairs} ontology–pattern pairs** in the quality-gated views, "
         f"from the harvest of {date}"
         + (f" ([workflow run]({row['harvest_run']}))" if row.get("harvest_run") else "") + ". "
         f"{added:,} ontologies were added to the corpus; {kept} pass the quality gate and the "
         f"other {added - kept:,} are kept with the rule that excluded them.", "",
         "No ontology, file or verdict of an earlier version was removed or changed.", "",
         "| view | ontologies | pairs |", "|---|---:|---:|"]
    for view in VIEWS:
        a, b = num(previous.get(f"{view}_ontologies")), num(row.get(f"{view}_ontologies"))
        c, d = num(previous.get(f"{view}_pairs")), num(row.get(f"{view}_pairs"))
        if previous.get(f"{view}_ontologies") and (a, c) != (b, d):
            L.append(f"| `{view}` | {a} → {b} ({b - a:+d}) | {c} → {d} ({d - c:+d}) |")
    L += ["", f"Zenodo: [{row['doi']}](https://doi.org/{row['doi']}) · added ontologies: "
              f"`release-iri/additions/v{v}.csv`", ""]
    return "\n".join(L)


def prepend_changelog(path: Path, entry: str) -> None:
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Changelog\n"
    i = text.find("\n## ")
    text = (text[:i + 1] + entry + "\n" + text[i + 1:]) if i >= 0 else text.rstrip("\n") + "\n\n" + entry
    path.write_text(text, encoding="utf-8", newline="\n")


def _n(n: int, one: str, many: str) -> str:
    return f"{n:,} {one if n == 1 else many}"


def release_notes(row: dict) -> str:
    """Plain-text version notes, for Zenodo and the commit and tag messages."""
    v = row["version"]
    return (f"Version {v} adds {_n(num(row['added_ontologies']), 'ontology', 'ontologies')} "
            f"to the corpus from the harvest of {row['date']} ({row['harvest_run']}); "
            f"{num(row['added_kept'])} pass the quality gate, adding "
            f"{_n(num(row['added_pairs']), 'ontology-pattern pair', 'ontology-pattern pairs')} "
            "to the views. No ontology, file or verdict of an earlier version was removed or "
            f"changed. The added ontologies are listed in release-iri/additions/v{v}.csv.")


# ------------------------------------------------------------------- corpus

def add_files(additions: Path, corpus: Path, rows: list[dict]) -> list[str]:
    """Copy the added ontologies into the corpus; nothing is copied on a problem."""
    problems = []
    for r in rows:
        ds, oid = r["dataset"], r["ontology_id"]
        src, dst = additions / ds / oid, corpus / ds / oid
        if not src.is_dir():
            problems.append(f"{ds}/{oid}: missing from the additions")
            continue
        if dst.exists():
            problems.append(f"{ds}/{oid}: already in the archived corpus")
            continue
        main = main_ontology_file(src)
        got = sha256(main) if main is not None else ""
        if got != (r.get("content_sha256") or ""):
            problems.append(f"{ds}/{oid}: file hash {got[:12] or 'none'} does not match the "
                            f"recorded {r.get('content_sha256', '')[:12]}")
    if not problems:
        for r in rows:
            shutil.copytree(additions / r["dataset"] / r["ontology_id"],
                            corpus / r["dataset"] / r["ontology_id"])
    return problems


SIDE_FILES = ("format_map.csv", "hash_failures.csv", "README.md")


def _append_rows(target: Path, new_csv: Path, backslash: bool) -> None:
    rows = read_csv(new_csv)
    if not rows:
        return
    fix = (lambda s: s.replace("/", "\\")) if backslash else (lambda s: s)
    if target.is_file():
        text = target.read_bytes().decode("utf-8")
        eol = "\r\n" if "\r\n" in text else "\n"
        cols = next(csv.reader(io.StringIO(text)))
    else:
        text, eol, cols = "", "\r\n" if backslash else "\n", list(rows[0])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=eol)
    if not text:
        w.writerow(cols)
    for r in rows:
        w.writerow([fix(r.get(c, "")) if c.startswith("file") else r.get(c, "") for c in cols])
    if text and not text.endswith(eol):
        text += eol
    target.write_bytes((text + buf.getvalue()).encode("utf-8"))


def merge_hashed(new: Path, hashed: Path) -> None:
    """Add a hash_corpus.py output for the added ontologies to corpus-hashed/.

    The archive's format_map.csv was written on Windows (backslashes, CRLF); new
    rows follow whatever style the file already has, so earlier rows stay as
    they are.
    """
    fmap = hashed / "format_map.csv"
    first = read_csv(fmap)[:1] if fmap.is_file() else []
    backslash = bool(first) and "\\" in first[0].get("file_plain", "")
    for src in sorted(p for p in new.rglob("*") if p.is_file()):
        rel = src.relative_to(new)
        if len(rel.parts) == 1 and rel.name in SIDE_FILES:
            continue
        dst = hashed / rel
        if dst.exists():
            raise FileExistsError(f"{rel.as_posix()} is already in {hashed}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in ("format_map.csv", "hash_failures.csv"):
        if (new / name).is_file():
            _append_rows(hashed / name, new / name, backslash)


def zip_tree(root: Path, prefix: str, out: Path) -> None:
    """Pack ``root`` as ``prefix/...``: each directory entry, then its files,
    then its subdirectories, all sorted -- the layout of the archived zips."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        def walk(d: Path, arc: str) -> None:
            zf.write(d, arc + "/")
            kids = sorted(d.iterdir(), key=lambda p: p.name)
            for f in (k for k in kids if k.is_file()):
                zf.write(f, f"{arc}/{f.name}")
            for sub in (k for k in kids if k.is_dir()):
                walk(sub, f"{arc}/{sub.name}")
        walk(root, prefix)


def fetch(record: str, out: Path, get=None) -> dict[str, Path]:
    """Download the record's archive files, check their MD5, unpack them."""
    import requests
    get = get or requests.get
    meta = get(f"{ZENODO}/api/records/{record}", timeout=60)
    meta.raise_for_status()
    files = {f["key"]: f for f in meta.json()["files"]}
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for key in ARCHIVE_FILES:
        if key not in files:
            raise FileNotFoundError(f"record {record} has no {key}")
        dest = out / key
        md5 = hashlib.md5()
        with get(files[key]["links"]["self"], stream=True, timeout=600) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    md5.update(chunk)
        want = files[key]["checksum"].split(":", 1)[1]
        if md5.hexdigest() != want:
            raise ValueError(f"{key}: md5 {md5.hexdigest()} != Zenodo's {want}")
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(out)
        got[key] = dest
        print(f"[OK] {key} md5 {want} verified and unpacked")
    return got


# -------------------------------------------------------------------- main

def cmd_notes(a) -> None:
    """The version notes, before the Zenodo draft (and so the DOI) exists."""
    history = read_csv(a.release / "history.csv")
    row = history_row(a.release, version=a.version, date=a.date, tag=f"v{a.version}",
                      record="", doi="", previous=history[-1] if history else {},
                      added=read_csv(a.added), harvest_run=a.harvest_run)
    a.out.write_text(release_notes(row) + "\n", encoding="utf-8")
    print(release_notes(row))


def cmd_record(a) -> None:
    rel = a.release
    history = read_csv(rel / "history.csv")
    previous = history[-1] if history else {}
    draft = json.loads(a.draft.read_text(encoding="utf-8"))
    added = read_csv(a.added)
    row = history_row(rel, version=a.version, date=a.date, tag=f"v{a.version}",
                      record=str(draft["draft_id"]), doi=draft["doi"], previous=previous,
                      added=added, harvest_run=a.harvest_run)
    entry = changelog_entry(row, previous)
    write_csv(rel / "history.csv", history + [row], HISTORY_COLUMNS)
    write_csv(rel / "additions" / f"v{a.version}.csv", added, list(added[0]) if added else [])
    archive = {"zenodo_record": str(draft["draft_id"]), "doi": draft["doi"],
               "release_tag": f"v{a.version}", "concept": str(draft.get("concept", ""))}
    (rel / "archive.json").write_text(json.dumps(archive, indent=1) + "\n", encoding="utf-8",
                                      newline="\n")
    a.citation.write_text(set_citation(a.citation.read_text(encoding="utf-8"), a.version, a.date),
                          encoding="utf-8", newline="\n")
    prepend_changelog(a.changelog, entry)
    a.notes_out.write_text(entry, encoding="utf-8")       # the GitHub Release body
    print(f"[OK] v{a.version} recorded: +{row['added_kept']} ontologies, +{row['added_pairs']} pairs")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--added", type=Path, required=True)
    p = sub.add_parser("version")
    p.add_argument("--citation", type=Path, required=True)
    p = sub.add_parser("latest")
    p.add_argument("--concept", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--record", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("add-files")
    p.add_argument("--additions", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--added", type=Path, required=True)
    p = sub.add_parser("merge-hashed")
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--into", type=Path, required=True)
    p = sub.add_parser("zip")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("notes")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--added", type=Path, required=True)
    p.add_argument("--harvest-run", default="")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("record")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--citation", type=Path, required=True)
    p.add_argument("--changelog", type=Path, required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--added", type=Path, required=True)
    p.add_argument("--harvest-run", default="")
    p.add_argument("--notes-out", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "check":
        added = {(r["dataset"], r["ontology_id"]) for r in read_csv(a.added)}
        problems = check_additive(a.old, a.new, added)
        for x in problems:
            print(f"::error::{x}")
        if problems:
            return 1
        print(f"[OK] the update only adds: {len(added)} ontologies, every release row unchanged")
    elif a.cmd == "version":
        print(next_minor(citation_version(a.citation.read_text(encoding="utf-8"))))
    elif a.cmd == "latest":
        import zenodo_publish
        print(zenodo_publish.latest_version_id(ZENODO, "", int(a.concept)))
    elif a.cmd == "fetch":
        fetch(a.record, a.out)
    elif a.cmd == "add-files":
        problems = add_files(a.additions, a.corpus, read_csv(a.added))
        for x in problems:
            print(f"::error::{x}")
        if problems:
            return 1
        print(f"[OK] added files copied into {a.corpus}")
    elif a.cmd == "merge-hashed":
        merge_hashed(a.new, a.into)
        print(f"[OK] hashed additions merged into {a.into}")
    elif a.cmd == "zip":
        zip_tree(a.root, a.prefix, a.out)
        print(f"[OK] {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    elif a.cmd == "notes":
        cmd_notes(a)
    elif a.cmd == "record":
        cmd_record(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
