#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rq_stats.py -- compute RQ1-RQ4 over the quality-gated corpus and emit both
tidy CSVs and the LaTeX tables used in the paper.

Population is defined by MANIFEST.csv: only rows with an empty drop_reason are
analysed. No count is produced by any other route.

Usage
-----
    python run_rq_stats.py --manifest out/MANIFEST.csv \
        --data-root /path/to/MultiSource --out out/rq
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from odpstats.rq_stats import (  # noqa: E402
    ROLES,
    THRESHOLDS,
    analyse,
    rq1,
    rq1_top_patterns,
    rq2,
    rq3,
    rq4,
    write_csv,
)

BS = "\\"
EOL = BS + BS
DS_LABEL = {"noOBO": "Paper \\& Repo", "OBO": "DOS--DP", "github": "GitHub"}


def tex_table(caption: str, label: str, header: list[str],
              rows: list[list[str]], colspec: str) -> str:
    out = [BS + "begin{table}[t]", BS + "centering", BS + "small",
           BS + "caption{" + caption + "}", BS + "label{" + label + "}",
           BS + r"setlength{" + BS + r"tabcolsep}{4pt}",
           BS + r"begin{tabular}{" + colspec + "}", BS + "toprule",
           " & ".join(BS + "textbf{" + h + "}" for h in header) + " " + EOL,
           BS + "midrule"]
    out += [" & ".join(r) + " " + EOL for r in rows]
    out += [BS + "bottomrule", BS + r"end{tabular}", BS + r"end{table}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-mb", type=float, default=0.0)
    ap.add_argument("--population", choices=("all", "gated"), default="all",
                    help="'all' (default) reports the REAL corpus: every "
                         "ontology that parses, gate-passing or not. 'gated' "
                         "restricts to those passing the quality gate. Rows "
                         "carry gate_passed either way, so either population "
                         "can be recovered from the output.")
    ap.add_argument("--keep-annotations", action="store_true")
    ap.add_argument("--approved-labels", type=Path, default=None,
                    help="CSV of human-adjudicated label matches (columns "
                         "dataset, ontology, pattern, human_verdict). Only rows "
                         "with human_verdict=reuse extend the ground truth. "
                         "MUST match the --approved-labels given to "
                         "build_manifest.py: pass it for an iri+label manifest "
                         "and omit it for an iri manifest, or the statistics "
                         "will describe a different ground truth from the "
                         "manifest they are computed over.")
    args = ap.parse_args()

    approved: set[tuple[str, str, str]] = set()
    if args.approved_labels and args.approved_labels.is_file():
        import csv as _csv
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.add((r["dataset"], r["ontology"], r["pattern"]))
        print(f"[iri+label] {len(approved)} human-confirmed label matches loaded")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"analysing ontologies (population={args.population}) ...")
    res = analyse(args.manifest, args.data_root, DATASETS,
                  strip_annots=not args.keep_annotations, max_mb=args.max_mb,
                  population=args.population,
                  approved_labels=approved or None)
    pairs, per_onto = res["pairs"], res["per_onto"]
    print(f"pairs: {len(pairs)}   ontologies: {len(per_onto)}")

    write_csv(args.out / "pairs.csv", pairs)
    write_csv(args.out / "per_ontology.csv", per_onto)

    t1 = rq1(pairs, per_onto)
    t1top = rq1_top_patterns(pairs)
    t2 = rq2(pairs)
    t3 = rq3(pairs)
    t4 = rq4(per_onto)
    write_csv(args.out / "rq1_extent.csv", t1)
    write_csv(args.out / "rq1_top_patterns.csv", t1top)
    write_csv(args.out / "rq2_reuse_forms.csv", t2)
    write_csv(args.out / "rq3_complexity.csv", t3)
    write_csv(args.out / "rq4_expansion.csv", t4)

    tex = args.out / "tables"
    tex.mkdir(exist_ok=True)

    # ---- RQ1
    hdr = ["Dataset", "Onto.", "Patt.", "Onto.", "Patt.", "Mean p/o"] + \
          [f"$\\geq${t}\\%" for t in THRESHOLDS]
    rows = [[DS_LABEL.get(r["dataset"], r["dataset"]),
             str(r["onto_reuse"]), str(r["patterns_reuse"]),
             str(r["onto_complete"]), str(r["patterns_complete"]),
             f"{r['mean_patterns_per_onto']:.2f}"] +
            [f"{r[f'ge_{t}pct']} ({r[f'ge_{t}pct_share']:.1f}\\%)"
             for t in THRESHOLDS]
            for r in t1]
    (tex / "rq1_extent.tex").write_text(tex_table(
        "RQ1: extent of ODP reuse on the quality-gated corpus. Columns 2--3 give "
        "ontologies and distinct patterns with any verified reuse; columns 4--5 the "
        "complete-reuse subset; then mean patterns reused per ontology and the "
        "distribution over reuse thresholds. Population: MANIFEST.csv, rows with no "
        "drop reason.",
        "tab:rq1_extent", hdr, rows,
        "l|rr|rr|r|" + "r" * len(THRESHOLDS)), encoding="utf-8")

    # ---- RQ2
    hdr2 = ["Dataset"] + [r.replace("_", " ") for r in ROLES]
    rows2 = [[DS_LABEL.get(r["dataset"], r["dataset"])] +
             [f"{r[f'{k}_pct']:.1f}\\%" for k in ROLES] for r in t2]
    (tex / "rq2_reuse_forms.tex").write_text(tex_table(
        "RQ2: structural forms of ODP reuse, as a share of role occurrences. An "
        "entity may occupy several roles, so shares are over occurrences rather "
        "than entities.",
        "tab:rq2_forms", hdr2, rows2, "l|" + "r" * len(ROLES)), encoding="utf-8")

    # ---- RQ3
    hdr3 = ["Dataset", "Pairs", "Pattern DL", "Ontology DL",
            "Onto.\\ more expr.", "Top constructs only in ontology"]
    rows3 = [[DS_LABEL.get(r["dataset"], r["dataset"]), str(r["n_pairs"]),
              r["pattern_dl_mode"], r["ontology_dl_mode"],
              f"{r['pct_ontology_more_expressive']:.1f}\\%",
              r["top_constructs_only_in_ontology"].replace("_", " ")[:60]]
             for r in t3]
    (tex / "rq3_complexity.tex").write_text(tex_table(
        "RQ3: sources of logical complexity. DL columns give the modal "
        "expressivity label on each side; the last two columns quantify how often "
        "the ontology uses constructs its reused pattern does not. DL labels are "
        "approximate (construct-based, not reasoner-certified).",
        "tab:rq3_complexity", hdr3, rows3, "l|r|ll|r|l"), encoding="utf-8")

    # ---- RQ4
    hdr4 = ["Dataset", "Onto.", "C", "OP", "DP",
            "added C", "added OP", "added DP", "\\% added"]
    rows4 = [[DS_LABEL.get(r["dataset"], r["dataset"]), str(r["n_ontologies"]),
              f"{r['mean_n_C']:.1f}", f"{r['mean_n_OP']:.1f}", f"{r['mean_n_DP']:.1f}",
              f"{r['mean_added_C']:.1f}", f"{r['mean_added_OP']:.1f}",
              f"{r['mean_added_DP']:.1f}", f"{r['pct_entities_added']:.1f}\\%"]
             for r in t4]
    (tex / "rq4_expansion.tex").write_text(tex_table(
        "RQ4: ontology expansion beyond the reused patterns. Mean declared "
        "classes (C), object properties (OP) and datatype properties (DP) per "
        "ontology, and how many of them are not supplied by any reused pattern.",
        "tab:rq4_expansion", hdr4, rows4, "l|r|rrr|rrr|r"), encoding="utf-8")

    print(f"\n[OK] LaTeX tables in {tex}")
    for t in sorted(tex.glob("*.tex")):
        print("   ", t.name)


if __name__ == "__main__":
    main()
