#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_reproduction.py -- check that the archived corpus reproduces MANIFEST.csv.

This is the reproducibility claim the resource can actually test: not that a
re-harvest finds the same ontologies (it queries live APIs, so it cannot), but
that the archived corpus, run through the released gate, yields the released
verdicts. ``.github/workflows/reproduce.yml`` runs both checks below against the
corpus downloaded from Zenodo.

  hashes   every ontology file in the archive is byte-identical to the file the
           gate hashed (content_sha256). Cheap, covers all ontologies,
           including the few too large to re-parse on a hosted runner.

               python verify_reproduction.py hashes \\
                   --manifest ../release-iri/MANIFEST.csv --corpus ../release-iri/corpus

  compare  a re-run of build_manifest.py gives the released row for every
           ontology. ``path`` and ``error_detail`` (which embeds the path) are
           machine-specific and not compared; ``parsed`` still carries the error
           type. Rows the re-run skipped with --max-mb are listed, not failed:
           their hash check stands in for them.

               python verify_reproduction.py compare \\
                   --release ../release-iri/MANIFEST.csv --regated out/MANIFEST.csv

Both exit non-zero on any discrepancy and can append a Markdown summary to
--summary (e.g. $GITHUB_STEP_SUMMARY).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from odpqual.gate import main_ontology_file, sha256  # noqa: E402

MACHINE_SPECIFIC = {"path", "error_detail"}
SIZE_SKIP = "G1_parse:SizeLimitExceeded"

Key = tuple[str, str]


def load(p: Path) -> tuple[list[str], dict[Key, dict]]:
    with p.open(encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        rows = {(r["dataset"], r["ontology_id"]): r for r in rd}
        return list(rd.fieldnames or []), rows


@dataclass
class CompareResult:
    datasets: list[str]
    n_identical: int = 0
    mismatches: list[tuple[Key, list[str]]] = field(default_factory=list)
    missing: list[Key] = field(default_factory=list)
    extra: list[Key] = field(default_factory=list)
    size_skipped: list[Key] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.mismatches or self.missing or self.extra)


def compare_manifests(release: Path, regated: Path) -> CompareResult:
    rel_cols, rel = load(release)
    new_cols, new = load(regated)
    datasets = sorted({ds for ds, _ in new})
    rel = {k: r for k, r in rel.items() if k[0] in set(datasets)}
    cols = [c for c in rel_cols if c in set(new_cols) and c not in MACHINE_SPECIFIC]

    res = CompareResult(datasets=datasets)
    res.missing = sorted(set(rel) - set(new))
    res.extra = sorted(set(new) - set(rel))
    for k in sorted(set(rel) & set(new)):
        if new[k].get("drop_reason", "").startswith(SIZE_SKIP):
            res.size_skipped.append(k)
            continue
        diff = [c for c in cols if (rel[k].get(c) or "") != (new[k].get(c) or "")]
        if diff:
            res.mismatches.append((k, diff))
        else:
            res.n_identical += 1
    return res


@dataclass
class HashResult:
    n_verified: int = 0
    mismatched: list[Key] = field(default_factory=list)
    missing: list[Key] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.mismatched or self.missing)


def verify_hashes(manifest: Path, corpus: Path) -> HashResult:
    _cols, rows = load(manifest)
    res = HashResult()
    for (ds, oid), r in sorted(rows.items()):
        d = corpus / ds / oid
        if not d.is_dir():
            res.missing.append((ds, oid))
            continue
        main = main_ontology_file(d)
        want = r.get("content_sha256") or ""
        got = sha256(main) if main is not None else ""
        if got == want:
            res.n_verified += 1
        else:
            res.mismatched.append((ds, oid))
    return res


# ---------------------------------------------------------------- reporting

def _list(title: str, keys, limit: int = 30) -> list[str]:
    keys = list(keys)
    if not keys:
        return []
    out = [f"**{title}** ({len(keys)}):", ""]
    out += [f"- `{ds}/{oid}`" for ds, oid in keys[:limit]]
    if len(keys) > limit:
        out.append(f"- ... and {len(keys) - limit} more")
    return out + [""]


def report_compare(res: CompareResult) -> str:
    total = res.n_identical + len(res.mismatches) + len(res.size_skipped)
    L = [f"### Gate re-run vs released MANIFEST.csv ({', '.join(res.datasets)})", "",
         f"- identical rows: **{res.n_identical} of {total}**",
         f"- skipped for size (hash-verified only): {len(res.size_skipped)}",
         f"- differing: {len(res.mismatches)} | missing from re-run: {len(res.missing)} | "
         f"not in release: {len(res.extra)}", "",
         "Not compared: `path`, `error_detail` (machine-specific).", ""]
    if res.mismatches:
        L += ["**Differing rows** (first 30):", ""]
        L += [f"- `{ds}/{oid}`: {', '.join(cols)}" for (ds, oid), cols in res.mismatches[:30]]
        L.append("")
    L += _list("Skipped for size", res.size_skipped)
    L += _list("Missing from re-run", res.missing)
    L += _list("Not in release", res.extra)
    L.append("**PASS**" if res.ok else "**FAIL**")
    return "\n".join(L) + "\n"


def report_hashes(res: HashResult, n_rows: int) -> str:
    L = ["### Archived corpus vs content_sha256", "",
         f"- byte-identical ontology files: **{res.n_verified} of {n_rows}**",
         f"- altered: {len(res.mismatched)} | missing from archive: {len(res.missing)}", ""]
    L += _list("Altered", res.mismatched)
    L += _list("Missing from archive", res.missing)
    L.append("**PASS**" if res.ok else "**FAIL**")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hashes")
    h.add_argument("--manifest", required=True, type=Path)
    h.add_argument("--corpus", required=True, type=Path,
                   help="release layout: <corpus>/<dataset>/<ontology_id>/")
    c = sub.add_parser("compare")
    c.add_argument("--release", required=True, type=Path)
    c.add_argument("--regated", required=True, type=Path)
    for p in (h, c):
        p.add_argument("--summary", type=Path, help="append the Markdown report here")
    a = ap.parse_args(argv)

    if a.cmd == "hashes":
        res = verify_hashes(a.manifest, a.corpus)
        text = report_hashes(res, len(load(a.manifest)[1]))
    else:
        res = compare_manifests(a.release, a.regated)
        text = report_compare(res)
    print(text)
    if a.summary:
        with a.summary.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    sys.exit(0 if res.ok else 1)


if __name__ == "__main__":
    main()
