#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_harvest.py -- add a harvest's new ontologies to the release. Never remove.

A re-harvest is a partial sample of GitHub, not a census: code search keeps at
most 1,000 results per query and ranks them differently on every run. Run
42239149156 did not return 525 release ontologies, and in a random sample of 100
of them 88 files were still at the same path. Replacing the release with a
harvest would therefore drop ontologies for no reason. This script only adds.

RULES
-----
  * Every release row of MANIFEST.csv is kept exactly as it is, in place:
    its archived file, its verdict and its pairs. Release ontologies the harvest
    did not return stay; so do those whose file has since changed upstream (the
    release keeps the archived version).
  * An ontology is added when its (dataset, ontology_id) is not in the release.
    Every added ontology ships, gate-passing or not, as the corpus always has.
  * Added ontologies are gated against the RELEASE, not against the harvest:
    a new copy of a released file is a G3 duplicate of the released row, even
    when its id sorts first (the released row is already published and cited).
    G1 and G2 depend on the file alone and are taken from the harvest; G3-G6
    are recomputed.
  * Every new removal is re-verified from the files, as audit_filters.py does.
    A G3 duplicate of a released row is verified against the hash the release
    recorded for that row. If any re-check is DISPUTED, nothing is written.
  * Pairs a harvest finds on a RELEASED file (a newly published pattern, say)
    are reported as candidates, not merged: adding one means re-gating the
    archived file with that pattern, and the archive lives on Zenodo.

WHAT IT WRITES
--------------
  <release>/MANIFEST.csv              release rows + added rows
  <release>/views/<portion>-*/        pairs and ontologies appended; statistics,
                                      top patterns and README recomputed
  <release>/dataset_statistics.csv    the changed views' rows replaced
  <release>/gate_summary.csv|md       recomputed from the merged manifest
  <release>/audit/                    new removals appended; summary recomputed
  <additions>/corpus/<portion>/<id>/  the added ontologies' files, to be added to
                                      the Zenodo corpus with the next version
  <report>/REPORT.md                  what was added and what was not, and why

Tables are written with LF line endings, so a pull request shows added lines
only. Afterwards, regenerate the release README:

    python write_release_readme.py --release <release> --keep-hashed-note

Usage
-----
    python merge_harvest.py --release ../release-iri \\
        --harvest-manifest ../harvest/out/MANIFEST.csv --harvest-root ../harvest \\
        --additions ../harvest/corpus-additions --report ../harvest/merge

Exit status: 0 merged (or nothing to add), 1 a new removal is DISPUTED.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_filters  # noqa: E402
import build_manifest  # noqa: E402
import build_release  # noqa: E402
from build_manifest import DATASETS  # noqa: E402
from odpqual.gate import apply_gate, gate_summary, main_ontology_file, sha256  # noqa: E402

EOL = "\n"
AUDIT_RULES = ("G1_parse", "G2_pattern_copy", "G3_duplicate", "G4_substantive", "G5_has_gt")
GATED_INT_FIELDS = ("n_own_entities", "n_gt_patterns", "n_complete_patterns")


# ---------------------------------------------------------------- csv helpers

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh), [])


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", lineterminator=EOL)
        w.writeheader()
        w.writerows(rows)


def key(r: dict) -> tuple[str, str]:
    return r["dataset"], r["ontology_id"]


def gt(r: dict) -> int:
    return int(r.get("n_gt_patterns") or 0)


def kept(r: dict) -> bool:
    return not r.get("drop_reason")


# ------------------------------------------------------------------ manifest

@dataclass
class Merge:
    rows: list[dict]                                   # the merged manifest
    added: list[dict]                                  # new rows, gated
    not_returned: list[tuple[str, str]] = field(default_factory=list)
    changed_upstream: list[tuple[str, str]] = field(default_factory=list)
    gained: list[tuple[str, str, int, int]] = field(default_factory=list)


def _record(row: dict):
    d = dict(row)
    for f in GATED_INT_FIELDS:
        d[f] = int(d.get(f) or 0)
    return build_manifest.record_from_cache(d)


def _row(rec) -> dict:
    return {k: "" if v is None else str(v) for k, v in rec.csv_row().items()}


def regate(new_rows: list[dict], release_rows: list[dict],
           min_own_entities: int = 1) -> list[dict]:
    """Gate the added rows against the release (see RULES)."""
    keeper: dict[tuple[str, str], str] = {}
    for r in release_rows:
        if r["drop_reason"].startswith(("G1", "G2")) or not r["content_sha256"]:
            continue
        keeper.setdefault((r["dataset"], r["content_sha256"]),
                          r["duplicate_of"] or r["ontology_id"])

    by_ds: dict[str, list] = defaultdict(list)
    for r in sorted(new_rows, key=key):
        rec = _record(r)
        if not rec.drop_reason.startswith(("G1", "G2")):
            rec.drop_reason, rec.duplicate_of = "", ""
            k = keeper.get((rec.dataset, rec.content_sha256)) if rec.content_sha256 else None
            if k:
                rec.duplicate_of = k
                rec.drop_reason = f"G3_duplicate:of({k})"
        by_ds[rec.dataset].append(rec)
    out = []
    for recs in by_ds.values():
        # G3 among the added rows, then G4-G6, exactly as build_manifest does.
        apply_gate(recs, min_own_entities=min_own_entities)
        out.extend(_row(rec) for rec in recs)
    return out


def merge_manifest(release_rows: list[dict], harvest_rows: list[dict], *,
                   min_own_entities: int = 1) -> Merge:
    rel = {key(r): r for r in release_rows}
    hv = {key(r): r for r in harvest_rows}
    harvested = {r["dataset"] for r in harvest_rows}

    added = regate([r for r in harvest_rows if key(r) not in rel], release_rows,
                   min_own_entities)

    # Each portion's new rows go right after its last release row, so the
    # release rows keep their positions and a diff shows insertions only.
    pending: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(added, key=key):
        pending[r["dataset"]].append(r)
    last = {r["dataset"]: i for i, r in enumerate(release_rows)}
    rows: list[dict] = []
    for i, r in enumerate(release_rows):
        rows.append(r)
        if last[r["dataset"]] == i:
            rows.extend(pending.pop(r["dataset"], []))
    for ds in list(pending):
        rows.extend(pending.pop(ds))

    common = [k for k in rel if k in hv]
    same = [k for k in common if rel[k]["content_sha256"] == hv[k]["content_sha256"]]
    return Merge(
        rows=rows,
        added=added,
        not_returned=[k for k in rel if k[0] in harvested and k not in hv],
        changed_upstream=[k for k in common if k not in same
                          and rel[k]["content_sha256"] and hv[k]["content_sha256"]],
        gained=[(k[0], k[1], gt(rel[k]), gt(hv[k])) for k in same if gt(hv[k]) > gt(rel[k])],
    )


# --------------------------------------------------------------------- views

def load_view(vdir: Path) -> tuple[list[dict], list[dict]]:
    pairs = read_csv(vdir / "pairs.csv")
    for p in pairs:
        p["coverage_pct"] = float(p["coverage_pct"])
        p["complete_reuse"] = int(p["complete_reuse"])
    return pairs, read_csv(vdir / "ontologies.csv")


def rewrite_view(vdir: Path, extra: list[tuple[dict, list[dict]]] | None = None) -> dict:
    """Rewrite a view from its own tables plus ``extra`` (row, pairs) entries.

    Ontologies and pairs are ordered by ontology id, as build_release.py
    orders them, so the result equals a full rebuild on the merged corpus.
    """
    source, level = vdir.name.rsplit("-", 1)
    pairs, ontos = load_view(vdir)
    for row, prs in extra or []:
        ontos.append(row)
        pairs.extend(prs)
    ontos.sort(key=lambda r: r["ontology_id"])
    pairs.sort(key=lambda p: p["ontology_id"])
    return build_release.write_view(vdir, source, level, ontos, pairs, eol=EOL)


def extend_views(release: Path, added: list[dict], data_root: Path) -> dict[str, tuple[dict, dict]]:
    """Append the added gate-passing ontologies to their views.

    Returns {view: (statistics before, statistics after)} for changed views,
    and rewrites those rows of dataset_statistics.csv.
    """
    entries: dict[str, list[tuple[dict, list[dict]]]] = defaultdict(list)
    for r in sorted((r for r in added if kept(r)), key=key):
        source, oid = key(r)
        prs = build_release.pair_rows(data_root / DATASETS[source] / oid, oid)
        for p in prs:
            p["pattern_file"] = (f"../../corpus/{source}/{oid}/patterns/"
                                 + Path(p["pattern_file"]).name)
            p["ontology_dir"] = f"../../corpus/{source}/{oid}"
        if prs:
            entries[source].append((r, prs))
        else:
            print(f"[WARN] {source}/{oid} passes the gate but yields no pair; not added to a view")

    changed: dict[str, tuple[dict, dict]] = {}
    for source, items in entries.items():
        for level in build_release.LEVELS:
            vdir = release / "views" / f"{source}-{level}"
            if not vdir.is_dir():
                continue
            take = [(r, prs) for r, prs in items
                    if level == "reuse" or any(p["complete_reuse"] for p in prs)]
            if not take:
                continue
            before = read_csv(vdir / "statistics.csv")[0]
            after = rewrite_view(vdir, take)
            changed[vdir.name] = (before, after)

    ds_path = release / "dataset_statistics.csv"
    if changed and ds_path.is_file():
        cols = read_header(ds_path)
        rows = read_csv(ds_path)
        for r in rows:
            if r["dataset"] in changed:
                r.update({k: v for k, v in changed[r["dataset"]][1].items()})
        write_csv(ds_path, rows, cols)
    return changed


# ---------------------------------------------------------------- gate, audit

def write_gate_summary(rows: list[dict], out: Path, min_own_entities: int = 1) -> None:
    per_ds: dict[str, list] = {}
    for r in rows:
        per_ds.setdefault(r["dataset"], []).append(SimpleNamespace(
            drop_reason=r["drop_reason"], n_complete_patterns=int(r["n_complete_patterns"] or 0)))
    summary = [{"dataset": ds, **gate_summary(recs)} for ds, recs in per_ds.items()]
    build_manifest.write_gate_summary(out, summary, min_own_entities, eol=EOL)


def verify_against_release(d: Path, keeper: dict) -> tuple[str, dict]:
    """A new copy of a released file: re-hash it and compare with the hash the
    release recorded for the kept row (reproduce.yml checks that recorded hash
    against the archived file)."""
    main = main_ontology_file(d)
    if main is None:
        return "DISPUTED", {"evidence": "no ontology file", "kept_instead": keeper["ontology_id"]}
    h = sha256(main)
    size = f"{main.stat().st_size / 1e6:.2f}"
    if h == keeper["content_sha256"]:
        return "ok", {"evidence": f"sha256 matches the hash recorded for release row "
                                  f"{keeper['ontology_id']} ({h[:16]}...)",
                      "kept_instead": keeper["ontology_id"], "size_mb": size}
    return "DISPUTED", {"evidence": f"hashes DIFFER {h[:12]} vs recorded "
                                    f"{keeper['content_sha256'][:12]}",
                        "kept_instead": keeper["ontology_id"], "size_mb": size}


def audit_additions(added: list[dict], release_rows: list[dict], data_root: Path,
                    min_own_entities: int = 1) -> dict[str, list[dict]]:
    rel = {key(r): r for r in release_rows}
    new = {key(r): r for r in added}
    out: dict[str, list[dict]] = defaultdict(list)
    for r in added:
        if kept(r):
            continue
        rule = r["drop_reason"].split(":", 1)[0]
        d = data_root / DATASETS[r["dataset"]] / r["ontology_id"]
        base = {k: r[k] for k in ("dataset", "ontology_id", "drop_reason", "n_entities_wide",
                                  "n_pattern_files", "n_own_entities", "path")}
        try:
            k = (r["dataset"], r["duplicate_of"])
            if rule == "G3_duplicate" and k in rel:
                v, ev = verify_against_release(d, rel[k])
            else:
                kd = (data_root / DATASETS[k[0]] / k[1]) if rule == "G3_duplicate" and k in new else None
                v, ev = audit_filters.verify_removal(rule, d, r, kd, min_own_entities)
        except Exception as exc:  # noqa: BLE001
            v, ev = "DISPUTED", {"evidence": f"audit error: {type(exc).__name__}: {exc}"}
        out[rule].append({**base, "verified": v, **ev})
    return out


def append_audit(audit_dir: Path, new: dict[str, list[dict]]) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for rule, rows in new.items():
        p = audit_dir / f"removed_{rule}.csv"
        old = read_csv(p) if p.is_file() else []
        cols = read_header(p) if p.is_file() else sorted(
            {c for r in rows for c in r},
            key=lambda c: (c not in ("dataset", "ontology_id", "verified", "evidence"), c))
        write_csv(p, old + rows, cols)

    summary = []
    for rule in AUDIT_RULES:
        p = audit_dir / f"removed_{rule}.csv"
        if not p.is_file():
            continue
        verdicts = Counter(r["verified"] for r in read_csv(p))
        if verdicts:
            summary.append({"rule": rule, "n_removed": sum(verdicts.values()), **verdicts})
    if summary:
        cols = sorted({c for r in summary for c in r},
                      key=lambda c: (c not in ("rule", "n_removed"), c))
        write_csv(audit_dir / "audit_summary.csv", summary, cols)


# -------------------------------------------------------------------- report

def plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def write_report(path: Path, m: Merge, n_release: int, changed: dict, views_before: dict,
                 new_pairs: int, run_url: str) -> None:
    n_kept = sum(1 for r in m.added if kept(r))
    verdicts = Counter((r["drop_reason"].split(":", 1)[0] or "kept") for r in m.added)
    L = [f"# Harvest update: {plural(len(m.added), 'ontology', 'ontologies')} added, "
         f"{n_kept} {'passes' if n_kept == 1 else 'pass'} the gate", "",
         f"**No release row was removed or changed.** The release's {n_release:,} ontologies "
         f"keep their rows, archived files, verdicts and pairs, including the "
         f"{len(m.not_returned):,} this harvest did not return.", "",
         f"- {len(m.added)} ontologies added, {n_kept} "
         f"{'passes' if n_kept == 1 else 'pass'} the gate",
         f"- {new_pairs} ontology–pattern pairs added to the views",
         "- verdicts of the added ontologies: "
         + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())), "",
         "| view | ontologies before | after | pairs before | after |",
         "|---|---:|---:|---:|---:|"]
    for name, before in views_before.items():
        after = changed.get(name, (before, before))[1]
        L.append(f"| `{name}` | {before['n_ontologies']} | {after['n_ontologies']} "
                 f"| {before['n_pairs']} | {after['n_pairs']} |")
    L += ["", "## Not merged", "",
          f"- **{len(m.gained)} released ontologies have more pairs in this harvest** "
          f"({sum(b - a for _d, _o, a, b in m.gained)} pairs). Adding them means re-gating the "
          "archived file with the new patterns, which needs the archived corpus; they are "
          "listed in `more_pairs_candidates.csv`.",
          f"- **{len(m.changed_upstream)} released ontologies changed upstream.** The release "
          "keeps the archived version; see `changed_upstream.csv`.",
          f"- **{len(m.not_returned)} released ontologies were not returned** by this harvest's "
          "searches. They stay in the release; see `not_returned.csv`.", "",
          "## Before tagging the new version", "",
          "The corpus is archived on Zenodo, not in git, so this pull request carries the "
          "tables only. The added ontologies' files are in the run's `corpus-additions` "
          "artifact" + (f" ({run_url})" if run_url else "") + ".", "",
          "1. Unpack the artifact over `release-iri/` (it holds `corpus/github/<id>/`) and "
          "check every file against the manifest: `python pipeline/verify_reproduction.py "
          "hashes --manifest release-iri/MANIFEST.csv --corpus release-iri/corpus`.",
          "2. Hash the new files (existing ones are skipped): `python pipeline/hash_corpus.py "
          "--corpus release-iri/corpus`, then `python pipeline/write_release_readme.py "
          "--release release-iri` so the README counts them.",
          "3. Set `date-released` in `CITATION.cff`, publish the Zenodo version with the full "
          "corpus (`pipeline/zenodo_publish.py`), and tag the version `CITATION.cff` names.",
          "4. Point `ZENODO_RECORD` and `RELEASE_TAG` in `.github/workflows/reproduce.yml` at "
          "the new record and tag.", "",
          "`FILTERS.md` states the v1.2.0 counts by hand and is not updated here.", ""]
    path.write_text("\n".join(L), encoding="utf-8", newline=EOL)


def write_disputed(path: Path, disputed: list[dict]) -> None:
    L = [f"# Harvest update stopped: {len(disputed)} new removals are DISPUTED", "",
         "Re-checking these removals from the files did not reproduce the gate's verdict, "
         "so nothing was written to the release.", "",
         "| ontology | rule | evidence |", "|---|---|---|"]
    L += [f"| `{r['ontology_id']}` | {r['drop_reason']} | {r.get('evidence', '')} |"
          for r in disputed]
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline=EOL)


# ---------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--harvest-manifest", required=True, type=Path)
    ap.add_argument("--harvest-root", required=True, type=Path,
                    help="directory holding github_dataset/ as gated by the harvest")
    ap.add_argument("--additions", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--min-own-entities", type=int, default=1)
    ap.add_argument("--run-url", default="")
    args = ap.parse_args(argv)

    man_path = args.release / "MANIFEST.csv"
    release_rows = read_csv(man_path)
    m = merge_manifest(release_rows, read_csv(args.harvest_manifest),
                       min_own_entities=args.min_own_entities)
    args.report.mkdir(parents=True, exist_ok=True)

    def dump(name, rows, cols):
        write_csv(args.report / name, rows, cols)

    dump("not_returned.csv", [{"dataset": d, "ontology_id": o} for d, o in m.not_returned],
         ["dataset", "ontology_id"])
    dump("changed_upstream.csv", [{"dataset": d, "ontology_id": o} for d, o in m.changed_upstream],
         ["dataset", "ontology_id"])
    dump("more_pairs_candidates.csv",
         [{"dataset": d, "ontology_id": o, "pairs_in_release": a, "pairs_in_harvest": b}
          for d, o, a, b in m.gained],
         ["dataset", "ontology_id", "pairs_in_release", "pairs_in_harvest"])

    outputs = {"added": 0, "added_kept": 0, "new_pairs": 0}
    if not m.added:
        (args.report / "REPORT.md").write_text(
            "# Harvest update: nothing to add\n\nEvery ontology this harvest found is already "
            "in the release. No file was changed.\n", encoding="utf-8", newline=EOL)
        print("nothing to add; the release is unchanged")
    else:
        audit = audit_additions(m.added, release_rows, args.harvest_root, args.min_own_entities)
        disputed = [r for rows in audit.values() for r in rows if r["verified"] != "ok"]
        if disputed:
            write_disputed(args.report / "REPORT.md", disputed)
            for r in disputed:
                print(f"[DISPUTED] {r['ontology_id']} {r['drop_reason']}: {r.get('evidence', '')}")
            return 1

        views_before = {r["dataset"]: r for r in read_csv(args.release / "dataset_statistics.csv")}
        write_csv(man_path, m.rows, read_header(man_path))
        changed = extend_views(args.release, m.added, args.harvest_root)
        write_gate_summary(m.rows, args.release, args.min_own_entities)
        append_audit(args.release / "audit", audit)
        dump("added.csv", m.added, build_manifest.CSV_COLUMNS)
        build_release.publish_corpus(m.added, args.harvest_root, args.additions / "corpus",
                                     sorted({r["dataset"] for r in m.added}))
        new_pairs = sum(int(after["n_pairs"]) - int(before["n_pairs"])
                        for name, (before, after) in changed.items() if name.endswith("-reuse"))
        write_report(args.report / "REPORT.md", m, len(release_rows), changed, views_before,
                     new_pairs, args.run_url)
        outputs = {"added": len(m.added), "added_kept": sum(1 for r in m.added if kept(r)),
                   "new_pairs": new_pairs}
        print(f"added {outputs['added']} ontologies ({outputs['added_kept']} pass the gate), "
              f"{new_pairs} pairs; release rows unchanged: {len(release_rows)}")

    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            for k, v in outputs.items():
                fh.write(f"{k}={v}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
