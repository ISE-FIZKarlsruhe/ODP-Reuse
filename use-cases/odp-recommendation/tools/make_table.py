#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_table.py -- render the paper's recommendation results table from the CSVs
in ``stats/``, so the table can never drift from the underlying data.

Run ``tools/make_stats.py`` first, then:

    python tools/make_table.py --dataset noOBO

Writes ``stats/uc_reco_results.tex``. Copy it next to the paper sources (the
paper input path is ``sections/tables/uc_reco_results.tex``).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

BS = "\\"          # a single backslash
EOL = BS + BS      # LaTeX end-of-row

# (group, backend key, display label)
ROWS = [
    ("data-free", "random", "Random ranking (floor)"),
    ("data-free", "embed_item_FIXEDcatalog[st_all-MiniLM-L6-v2]",
     "Content retrieval, fixed catalogue"),
    ("data-free", "doc2vec", "Doc2Vec content matching$^{" + BS + "dagger}$"),
    ("data-informed", "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]",
     "Content retrieval, dataset catalogue"),
    ("data-informed", "popularity", "Popularity prior"),
    ("data-informed", "embed_ontology_knn_cf", "Collaborative filtering (embedding)"),
    ("data-informed", "ontology_knn_cf", "Collaborative filtering (Jaccard)"),
]

BEST = "ontology_knn_cf"

GROUP_TAG = {
    "data-free": BS + r"multirow{3}{*}{" + BS + r"rotatebox{90}{" + BS
                 + r"scriptsize data-free}}",
    "data-informed": BS + r"multirow{4}{*}{" + BS + r"rotatebox{90}{" + BS
                     + r"scriptsize data-informed}}",
}


def read_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["recommender"]: r for r in csv.DictReader(fh)}


def bold(text: str, on: bool) -> str:
    return (BS + "textbf{" + text + "}") if on else text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="noOBO")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    stats = HERE / "stats"
    ci = read_csv(stats / f"reco_ci_map_{args.dataset}.csv")
    head = read_csv(stats / f"reco_headline_{args.dataset}.csv")

    # The CI CSV is sorted by MAP, and the LLM backends were scored on a smaller
    # sample -- so take the test-set size from the rows actually in this table,
    # which are all full-roster methods.
    table_keys = [k for _g, k, _l in ROWS if k in ci]
    n = max(int(ci[k]["n"]) for k in table_keys)
    if len({int(ci[k]["n"]) for k in table_keys}) > 1:
        raise SystemExit(
            "[ERROR] table rows have differing n: "
            + ", ".join(f"{k}={ci[k]['n']}" for k in table_keys)
            + " -- a single caption cannot describe them."
        )

    lines: list[str] = []
    add = lines.append
    add(BS + r"begin{table}[t]")
    add(BS + "centering")
    add(BS + r"caption{Pattern recommendation on the Paper/Repo portion ($70/30$ split,")
    add(f"${n}$ test ontologies). MAP with a $95" + BS + r"%$ percentile-bootstrap")
    add(r"confidence interval; P@1 for context. Methods are grouped by whether they")
    add(r"consume the ODP-Reuse records. $^{" + BS + r"dagger}$A faithful")
    add(r"re-implementation of ODPReco's content-matching approach, not ODPReco itself.}")
    add(BS + r"label{tab:uc_reco}")
    add(BS + "small")
    add(BS + r"setlength{" + BS + r"tabcolsep}{5pt}")
    add(BS + r"begin{tabular}{ll|ccc}")
    add(BS + "toprule")
    add(" & ".join(["", BS + r"textbf{Recommender}", BS + r"textbf{MAP}",
                    BS + r"textbf{95" + BS + r"% CI}", BS + r"textbf{P@1}"]) + " " + EOL)
    add(BS + "midrule")

    prev = None
    for group, key, label in ROWS:
        if key not in ci:
            print(f"[SKIP] {key}: not in stats CSV")
            continue
        if group != prev:
            if prev is not None:
                add(BS + "midrule")
            tag = GROUP_TAG[group]
            prev = group
        else:
            tag = ""
        row, hrow = ci[key], head[key]
        is_best = key == BEST
        cells = [
            tag,
            bold(label, is_best),
            bold(f"{float(row['mean_map']):.3f}", is_best),
            f"$[{float(row['ci_lo_95']):.3f}, {float(row['ci_hi_95']):.3f}]$",
            f"{float(hrow['P@1']):.3f}",
        ]
        add(" & ".join(cells) + " " + EOL)

    add(BS + "bottomrule")
    add(BS + r"end{tabular}")
    add(BS + r"end{table}")

    out = args.out or (stats / "uc_reco_results.tex")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out.relative_to(HERE)}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
