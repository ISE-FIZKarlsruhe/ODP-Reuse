#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_manifest.py -- apply the quality gate to every dataset and write
MANIFEST.csv, the single source of every ontology count reported in the papers.

Rule: no count appears in any manuscript that is not a documented filter over
MANIFEST.csv. See odpqual/gate.py for the rules and their justifications.

Usage
-----
    python build_manifest.py --data-root /path/to/MultiSource --out out/

    # resume an interrupted run (per-ontology results are cached)
    python build_manifest.py --data-root ... --out out/ --resume

Outputs (under --out):
    MANIFEST.csv          one row per ontology, all gate columns
    gate_summary.csv      per-dataset attrition, one row per rule
    gate_summary.md       the same as a readable table
    cache/<dataset>.jsonl per-ontology cache for --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from odpqual.gate import (  # noqa: E402
    RULES,
    OntologyRecord,
    analyse_ontology,
    apply_gate,
    gate_summary,
)

logging.getLogger("rdflib").setLevel(logging.ERROR)

# The datasets. Keys are the names used throughout both papers.
DATASETS = {
    "github": "github_dataset",   # GitHub Code Search portion
    "noOBO": "noOBO",             # Paper/Repo portion (non-OBO)
    "OBO": "OBO",                 # DOS-DP portion
}

CSV_COLUMNS = [
    "ontology_id", "dataset", "path", "content_sha256", "n_pattern_files",
    "parsed", "error_detail", "n_triples", "n_triples_annotation",
    "n_entities_narrow", "n_entities_wide", "declared_iris",
    "self_match_reason", "duplicate_of",
    "n_own_entities", "n_gt_patterns", "n_complete_patterns", "max_reuse_pct",
    "n_label_confirmed", "n_label_entities",
    "stage_reached", "drop_reason",
    "n_pattern_errors", "pattern_errors",
]


def load_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                out[d["ontology_id"]] = d
            except Exception:  # noqa: BLE001
                continue
    return out


def record_from_cache(d: dict) -> OntologyRecord:
    rec = OntologyRecord(ontology_id=d["ontology_id"], dataset=d["dataset"],
                         path=d["path"])
    for k, v in d.items():
        if hasattr(rec, k) and k not in ("patterns", "entities"):
            setattr(rec, k, v)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="directory containing github_dataset/, noOBO/, OBO/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--min-own-entities", type=int, default=1,
                    help="G4 threshold: minimum entities an ontology must "
                         "contribute beyond its patterns. Default 1 -- the "
                         "released value, and the minimal non-vacuous "
                         "requirement. See FILTERS.md for the cost curve.")
    ap.add_argument("--max-mb", type=float, default=0.0,
                    help="skip ontology files larger than this (0 = no limit, the "
                         "default). A large ontology is a legitimate member of the "
                         "corpus, so the DATASET is never size-limited; size only "
                         "constrains the embedding stage (see run_rq5_extraction.py).")
    ap.add_argument("--keep-annotations", action="store_true",
                    help="do NOT strip annotation/dcterms/skos triples")
    ap.add_argument("--approved-labels", type=Path, default=None,
                    help="CSV of human-adjudicated label matches (needs columns "
                         "dataset, ontology, pattern, human_verdict). Only rows "
                         "with human_verdict=reuse extend the ground truth; this "
                         "is what distinguishes the iri+label track from iri.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    approved: set[tuple[str, str, str]] = set()
    if args.approved_labels and args.approved_labels.is_file():
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.add((r["dataset"], r["ontology"], r["pattern"]))
        print(f"[iri+label] {len(approved)} human-confirmed label matches loaded")

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "cache"
    cache_dir.mkdir(exist_ok=True)

    all_records: list[OntologyRecord] = []
    per_dataset: dict[str, list[OntologyRecord]] = {}

    for ds in args.datasets:
        root = args.data_root / DATASETS[ds]
        if not root.is_dir():
            print(f"[WARN] {ds}: {root} not found, skipping")
            continue
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if args.limit:
            dirs = dirs[:args.limit]

        cache_path = cache_dir / f"{ds}.jsonl"
        cached = load_cache(cache_path) if args.resume else {}
        if cached:
            print(f"[{ds}] resuming: {len(cached)} cached of {len(dirs)}")

        recs: list[OntologyRecord] = []
        t0 = time.time()
        with cache_path.open("a" if args.resume else "w", encoding="utf-8") as cf:
            for i, d in enumerate(dirs, 1):
                if d.name in cached:
                    recs.append(record_from_cache(cached[d.name]))
                    continue
                try:
                    rec = analyse_ontology(
                        d, ds,
                        strip_annots=not args.keep_annotations,
                        max_mb=args.max_mb,
                        approved_labels=approved or None,
                    )
                except Exception as exc:  # noqa: BLE001
                    rec = OntologyRecord(ontology_id=d.name, dataset=ds, path=str(d))
                    rec.parsed = type(exc).__name__
                    rec.drop_reason = f"G1_parse:{type(exc).__name__}"
                recs.append(rec)
                cf.write(json.dumps(rec.csv_row(), ensure_ascii=False) + "\n")
                cf.flush()
                if i % 25 == 0 or i == len(dirs):
                    el = time.time() - t0
                    print(f"[{ds}] {i}/{len(dirs)}  {el:6.0f}s  "
                          f"({el / max(i - len(cached), 1):.2f}s/onto)", flush=True)

        per_dataset[ds] = recs
        all_records.extend(recs)

    # The gate's duplicate rule is applied WITHIN a dataset, because the three
    # portions are reported separately and a cross-portion duplicate is a
    # finding rather than a defect.
    for ds, recs in per_dataset.items():
        apply_gate(recs, min_own_entities=args.min_own_entities)

    # ---- MANIFEST.csv
    man = args.out / "MANIFEST.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for rec in all_records:
            w.writerow(rec.csv_row())
    print(f"\n[OK] {man}  ({len(all_records)} rows)")

    # ---- summaries
    rows = []
    for ds, recs in per_dataset.items():
        s = gate_summary(recs)
        s = {"dataset": ds, **s}
        rows.append(s)
    if rows:
        keys = list(rows[0])
        with (args.out / "gate_summary.csv").open("w", newline="",
                                                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        md = ["# Quality-gate attrition", "",
              "Counts are per dataset; each ontology is attributed to the FIRST rule",
              "it fails, so the columns partition `discovered`.", "",
              "| " + " | ".join(keys) + " |",
              "|" + "|".join(["---"] * len(keys)) + "|"]
        for r in rows:
            md.append("| " + " | ".join(str(r[k]) for k in keys) + " |")
        md.append("")
        md.append("Rules: G1 parseable, G2 not a pattern copy, G3 not a duplicate,")
        md.append("G4 substantive (>=%d own entities), G5 has ground truth,"
                  % args.min_own_entities)
        md.append("G6 complete reuse (variant membership). See odpqual/gate.py.")
        (args.out / "gate_summary.md").write_text("\n".join(md) + "\n",
                                                  encoding="utf-8")
        print("\n".join(md))


if __name__ == "__main__":
    main()
