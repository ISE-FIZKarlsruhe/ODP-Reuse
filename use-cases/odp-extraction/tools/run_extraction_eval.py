#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extraction_eval.py -- use case: automatic ODP extraction.

Reproduces the paper's baseline (OWL2Vec* and RDF2Vec, each with K-Means and
HDBSCAN) on the quality-gated corpus, and reports each configuration under BOTH
model-selection regimes:

  oracle        hyper-parameters chosen by argmax entity-F1 against the ground
                truth -- comparable to the previously published numbers, and an
                upper bound rather than an achievable score.
  unsupervised  hyper-parameters chosen by silhouette, never touching the GT.

The gap between them quantifies how much of the published performance came from
oracle model selection. See extractbench/evaluate.py for why both are reported.

Only the two embedders named in the paper are run. The 45-method sweep lives in
PatternMiner and is out of scope here.

Usage
-----
    python run_rq5_extraction.py --manifest out/MANIFEST.csv \
        --data-root /path/to/MultiSource --out out/rq5 \
        --datasets noOBO OBO github --subset complete
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_USECASE = Path(__file__).resolve().parent.parent      # use-cases/odp-extraction
_REPO = _USECASE.parent.parent                        # repository root
_PIPELINE = _REPO / "pipeline"

# The embedders and the evaluator live in this use case; the quality gate and the
# dataset registry live in the shared pipeline, because both use cases and the
# RQ statistics must agree on them.
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_PIPELINE))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from extractbench import owl2vec, rdf2vec  # noqa: E402
from extractbench.evaluate import evaluate_ontology  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file, pattern_files  # noqa: E402

EMBEDDERS = {"owl2vec": owl2vec.embed, "rdf2vec": rdf2vec.embed}
CLUSTERERS = ("kmeans", "hdbscan")

# EMBEDDING-stage limit, not a dataset limit.
#
# OWL2Vec* and RDF2Vec generate n_walks (=100) walks per entity, so both runtime
# and peak memory scale linearly with the entity count. The gated corpus has a
# median of 68 entities per ontology (GitHub) and 60 (Paper/Repo), but the DOS-DP
# portion ranges from 7,257 to 193,638 entities: `upheno` alone would imply ~19M
# walks, hours of CPU, and likely MemoryError on commodity hardware.
#
# 5,000 admits every GitHub and Paper/Repo ontology while excluding the handful of
# OBO-scale ones. Those ontologies REMAIN IN THE DATASET -- only this stage skips
# them, and every skip is written to rq5_skipped.csv with its reason. Pass
# --max-entities 0 on a machine with the memory to attempt them.
DEFAULT_MAX_ENTITIES = 5000
DS_LABEL = {"noOBO": "Paper \\& Repo", "OBO": "DOS--DP", "github": "GitHub"}
BS = "\\"
EOL = BS + BS


def load_population(manifest: Path, datasets: list[str], subset: str) -> list[dict]:
    rows = [r for r in csv.DictReader(manifest.open(encoding="utf-8"))
            if not r["drop_reason"] and r["dataset"] in datasets]
    if subset == "complete":
        rows = [r for r in rows if int(r.get("n_complete_patterns") or 0) > 0]
    return rows


def ground_truth(onto_dir: Path, host: set[str], strip_annots: bool,
                 complete_only: bool, host_graph=None, dataset: str = "",
                 approved_labels: set[tuple[str, str, str]] | None = None
                 ) -> dict[str, set[str]]:
    """GT pattern instances: the pattern entities present in the host.

    With complete_only, a pattern contributes only if ALL of its entities are
    present -- the complete-reuse criterion, evaluated after the quality gate has
    already removed pattern copies and duplicates.

    ``approved_labels`` holds (dataset, ontology, pattern) triples whose
    label-only matches a human adjudicated as genuine reuse; for those pairs the
    ground truth is the union of IRI-matched and label-matched entities. Same
    call and same guards as odpqual/gate.py, so the manifest, the RQ statistics
    and this evaluation cannot describe three different ground truths.

    Without it, evaluating an iri+label manifest would score against IRI-only
    ground truth -- and the ontologies admitted PURELY by a confirmed label
    match would return {} here and be silently dropped as "no_ground_truth".
    """
    gt: dict[str, set[str]] = {}
    for pf in pattern_files(onto_dir):
        pg, _ = parse_graph(pf)
        if pg is None:
            continue
        if strip_annots:
            pg, _ = strip_annotations(pg)
        pents = set(collect_entities(pg, wide=True))
        if not pents:
            continue
        inter = host & pents
        if (approved_labels and host_graph is not None
                and (dataset, onto_dir.name, pf.stem) in approved_labels):
            from odpqual.gtmodes import gt_by_label
            inter = inter | (gt_by_label(host_graph, host, pg) - inter)
        if not inter:
            continue
        if complete_only and not pents <= host:
            continue
        gt[pf.stem] = inter
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--datasets", nargs="*", default=["noOBO", "OBO", "github"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="complete",
                    help="complete-reuse subset (paper default) or any-reuse")
    ap.add_argument("--max-mb", type=float, default=0.0,
                    help="skip files larger than this (0 = no limit, default)")
    ap.add_argument("--max-entities", type=int, default=DEFAULT_MAX_ENTITIES,
                    help=f"skip ontologies with more entities than this "
                         f"(default {DEFAULT_MAX_ENTITIES}; 0 = no limit). This is "
                         f"an EMBEDDING-stage limit only: walk count and memory "
                         f"grow linearly with entity count. The ontology stays in "
                         f"the dataset; the exclusion is recorded in "
                         f"rq5_skipped.csv, never silent.")
    ap.add_argument("--keep-annotations", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--approved-labels", type=Path, default=None,
                    help="CSV of human-adjudicated label matches (columns "
                         "dataset, ontology, pattern, human_verdict). Only rows "
                         "with human_verdict=reuse extend the ground truth. "
                         "MUST match the --approved-labels given to "
                         "build_manifest.py: without it, an iri+label manifest "
                         "is scored against IRI-only ground truth and the "
                         "ontologies admitted purely by a confirmed label "
                         "match are dropped as 'no_ground_truth'.")
    args = ap.parse_args()

    approved: set[tuple[str, str, str]] = set()
    if args.approved_labels and args.approved_labels.is_file():
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.add((r["dataset"], r["ontology"], r["pattern"]))
        print(f"[iri+label] {len(approved)} human-confirmed label matches loaded")

    args.out.mkdir(parents=True, exist_ok=True)
    strip_annots = not args.keep_annotations
    per_path = args.out / "rq5_per_ontology.jsonl"
    done: set[tuple[str, str]] = set()
    if args.resume and per_path.is_file():
        for line in per_path.open(encoding="utf-8"):
            try:
                d = json.loads(line)
                done.add((d["dataset"], d["ontology"]))
            except Exception:  # noqa: BLE001
                continue
        print(f"resuming: {len(done)} already done")

    pop = load_population(args.manifest, args.datasets, args.subset)
    if args.limit:
        pop = pop[:args.limit]
    print(f"population ({args.subset}): {len(pop)} ontologies "
          f"across {sorted({r['dataset'] for r in pop})}")

    # Every exclusion is recorded with its reason. A silently dropped ontology is
    # how count inconsistencies get created, so skips are data, not noise.
    skipped: list[tuple[str, str, str, str]] = []

    t0 = time.time()
    with per_path.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for i, r in enumerate(pop, 1):
            ds, oid = r["dataset"], r["ontology_id"]
            if (ds, oid) in done:
                continue
            onto_dir = args.data_root / DATASETS[ds] / oid
            main_f = main_ontology_file(onto_dir)
            if main_f is None:
                skipped.append((ds, oid, "no_ontology_file", ""))
                continue
            if args.max_mb > 0 and main_f.stat().st_size > args.max_mb * 1024 * 1024:
                skipped.append((ds, oid, "file_too_large",
                                f"{main_f.stat().st_size / 1e6:.0f}MB>{args.max_mb}MB"))
                continue
            g, err = parse_graph(main_f)
            if g is None:
                skipped.append((ds, oid, "parse_error", err))
                continue
            if strip_annots:
                g, _ = strip_annotations(g)
            entities = collect_entities(g, wide=True)
            if not entities:
                skipped.append((ds, oid, "no_entities", ""))
                continue
            if args.max_entities > 0 and len(entities) > args.max_entities:
                skipped.append((ds, oid, "too_many_entities",
                                f"{len(entities)}>{args.max_entities}"))
                continue
            gt = ground_truth(onto_dir, set(entities), strip_annots,
                              complete_only=(args.subset == "complete"),
                              host_graph=g, dataset=ds,
                              approved_labels=approved or None)
            if not gt:
                skipped.append((ds, oid, "no_ground_truth",
                                f"subset={args.subset}"))
                continue

            labels = {e: e.rsplit("/", 1)[-1] for e in entities}
            for mname, embed in EMBEDDERS.items():
                try:
                    emb = embed(g, entities, labels)
                except MemoryError:
                    # Expected on the largest ontologies: OWL2Vec*/RDF2Vec build
                    # n_walks walks per entity, so memory grows with |entities|.
                    # Recorded, never silent -- the dataset keeps the ontology even
                    # when this machine cannot embed it.
                    skipped.append((ds, oid, "embed_out_of_memory",
                                    f"{mname}, {len(entities)} entities"))
                    print(f"  [OOM] {oid}/{mname} ({len(entities)} entities)",
                          flush=True)
                    continue
                except Exception as exc:  # noqa: BLE001
                    skipped.append((ds, oid, "embed_error",
                                    f"{mname}: {type(exc).__name__}"))
                    print(f"  [skip] {oid}/{mname}: {type(exc).__name__}")
                    continue
                emb = np.nan_to_num(np.asarray(emb, dtype=np.float32))
                for cl in CLUSTERERS:
                    res = evaluate_ontology(emb, entities, gt, cl)
                    rec = {
                        "dataset": ds, "ontology": oid, "method": mname,
                        "clusterer": cl, "n_entities": len(entities),
                        "n_gt": len(gt),
                    }
                    for regime in ("oracle", "unsup"):
                        s = res[regime]
                        rec[f"{regime}_precision"] = round(s["precision"], 4)
                        rec[f"{regime}_recall"] = round(s["recall"], 4)
                        rec[f"{regime}_f1"] = round(s["f1"], 4)
                        rec[f"{regime}_config"] = s["config"]
                    fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(pop):
                el = time.time() - t0
                print(f"[{i}/{len(pop)}] {el:6.0f}s "
                      f"({el / max(i - len(done), 1):.1f}s/onto)", flush=True)

    # ---------------- skip report (never silent)
    if skipped:
        sp = args.out / "rq5_skipped.csv"
        with sp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "ontology", "reason", "detail"])
            w.writerows(skipped)
        from collections import Counter
        tally = Counter(f"{d}:{r}" for d, _o, r, _x in skipped)
        print(f"\n[SKIPPED] {len(skipped)} ontologies excluded from RQ5 "
              f"-> {sp.name}")
        for k, v in tally.most_common():
            print(f"    {k}: {v}")

    # ---------------- aggregate
    recs = [json.loads(l) for l in per_path.open(encoding="utf-8")]
    print(f"\n{len(recs)} (ontology, method, clusterer) results")
    if not recs:
        print("[ERROR] no results produced -- see the skip report above.")
        return
    agg: list[dict] = []
    for ds in sorted({r["dataset"] for r in recs}):
        for m in EMBEDDERS:
            for cl in CLUSTERERS:
                sub = [r for r in recs
                       if r["dataset"] == ds and r["method"] == m
                       and r["clusterer"] == cl]
                if not sub:
                    continue
                row = {"dataset": ds, "method": m, "clusterer": cl,
                       "n_ontologies": len(sub)}
                for regime in ("oracle", "unsup"):
                    for k in ("precision", "recall", "f1"):
                        row[f"{regime}_{k}"] = round(
                            float(np.mean([r[f"{regime}_{k}"] for r in sub])), 4)
                row["oracle_gap_f1"] = round(row["oracle_f1"] - row["unsup_f1"], 4)
                agg.append(row)

    if agg:
        with (args.out / "rq5_results.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg[0]))
            w.writeheader()
            w.writerows(agg)
        print(f"[OK] {args.out / 'rq5_results.csv'}")

        # LaTeX
        lines = [BS + "begin{table}[t]", BS + "centering", BS + "small",
                 BS + r"caption{RQ5: automatic ODP extraction on the "
                 r"quality-gated complete-reuse subset. Entity-level "
                 r"precision/recall/$F_1$, macro-averaged over ontologies. "
                 r"\textsc{oracle} selects clustering hyper-parameters by argmax "
                 r"$F_1$ against the ground truth (an upper bound, and what "
                 r"prior work reported); \textsc{unsup.} selects them by "
                 r"silhouette without the ground truth. The gap is the "
                 r"contribution of oracle model selection.}",
                 BS + r"label{tab:rq5_extraction}",
                 BS + r"setlength{" + BS + r"tabcolsep}{4pt}",
                 BS + r"begin{tabular}{ll|l|rrr|rrr|r}", BS + "toprule",
                 r"\textbf{Dataset} & \textbf{Model} & \textbf{Clust.} & "
                 r"\multicolumn{3}{c|}{\textsc{oracle}} & "
                 r"\multicolumn{3}{c|}{\textsc{unsup.}} & "
                 r"$\Delta F_1$ " + EOL,
                 r" & & & P & R & $F_1$ & P & R & $F_1$ & " + EOL,
                 BS + "midrule"]
        for row in agg:
            lines.append(" & ".join([
                DS_LABEL.get(row["dataset"], row["dataset"]),
                {"owl2vec": "OWL2Vec*", "rdf2vec": "RDF2Vec"}[row["method"]],
                {"kmeans": "K-Means", "hdbscan": "HDBSCAN"}[row["clusterer"]],
                f"{row['oracle_precision']:.3f}", f"{row['oracle_recall']:.3f}",
                f"{row['oracle_f1']:.3f}",
                f"{row['unsup_precision']:.3f}", f"{row['unsup_recall']:.3f}",
                f"{row['unsup_f1']:.3f}",
                f"{row['oracle_gap_f1']:.3f}",
            ]) + " " + EOL)
        lines += [BS + "bottomrule", BS + r"end{tabular}", BS + r"end{table}"]
        tex = args.out / "tables"
        tex.mkdir(exist_ok=True)
        (tex / "rq5_extraction.tex").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
        print(f"[OK] {tex / 'rq5_extraction.tex'}")
        for row in agg:
            print(f"  {row['dataset']:8s} {row['method']:8s} {row['clusterer']:8s} "
                  f"oracle_F1={row['oracle_f1']:.3f}  unsup_F1={row['unsup_f1']:.3f} "
                  f"(gap {row['oracle_gap_f1']:+.3f})  n={row['n_ontologies']}")


if __name__ == "__main__":
    main()
