#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emit_asprino_predictions.py -- recover predictions.jsonl for a finished
conceptual-components run, without re-running the pipeline.

WHY THIS EXISTS
---------------
``run_asprino_full.py`` writes ``predictions.jsonl`` -- the per-ontology cluster
memberships -- in the same pass that scores them. The ODP-Reuse run in
``results/asprino-full/`` was produced before that writer existed, so it kept
the scores (``asprino_full_per_ontology.jsonl``, ``asprino_full_results.csv``)
and the community-level artefacts, but not the per-ontology groups in the
runner's cluster format. The MSE run has all three.

Re-running ``--stage score --resume`` would fix it, but it re-runs step 6
(TF-IDF + K-Means over the synset/frame column) and so needs
``topLevelAnchestors-prefix.csv`` from the upstream clone, and would pick its
own ``k`` -- producing clusters that need not be the ones the published scores
were computed from.

This script instead reads ``clustered_communities.json``, which already carries
the step-6 ``cluster`` assignment for every community of the finished run. The
clusters it emits are therefore *the* clusters, not a re-derivation.

SELF-VERIFICATION
-----------------
The script re-scores what it emits and compares against
``asprino_full_results.csv`` from the same directory. If every published
precision / recall / F1 / cluster-count figure is not reproduced exactly, it
writes nothing and exits non-zero: an emitted file that disagrees with the
published scores would be worse than no file at all.

Usage
-----
    python tools/emit_asprino_predictions.py \\
        --run results/asprino-full \\
        --manifest ../../pipeline/out/MANIFEST.csv \\
        --data-root C:/Users/eno/Documents/PhD/MultiSource_server \\
        --datasets github noOBO --subset reuse

For a ``--corpus``-based run, pass ``--corpus corpora/<name>`` instead of
``--manifest``/``--data-root``, exactly as for ``run_asprino_full.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_USECASE / "tools"))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from extractbench.evaluate import match_and_score  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from run_extraction_eval import ground_truth, load_population  # noqa: E402

RELEASE_DIRS = {"github": "github", "noOBO": "noOBO", "OBO": "OBO"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="a finished run directory holding "
                         "clustered_communities.json")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--datasets", nargs="*", default=["noOBO", "github"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--corpus-layout", choices=("source", "release"),
                    default="source")
    ap.add_argument("--corpus-name", default="plain")
    ap.add_argument("--approved-labels", type=Path, default=None)
    ap.add_argument("--out-name", default="predictions.jsonl")
    ap.add_argument("--force", action="store_true",
                    help="write even if the published scores are not reproduced")
    args = ap.parse_args()

    for _attr in ("run", "manifest", "data_root", "corpus", "approved_labels"):
        v = getattr(args, _attr)
        if v is not None:
            setattr(args, _attr, v.resolve())

    cc_path = args.run / "clustered_communities.json"
    if not cc_path.is_file():
        ap.error(f"no clustered_communities.json in {args.run}")
    records = json.loads(cc_path.read_text(encoding="utf-8"))
    if not records or "cluster" not in records[0]:
        ap.error(f"{cc_path} carries no step-6 'cluster' field; "
                 "this run did not reach step 6")
    k_clusters = len({r["cluster"] for r in records})
    print(f"[in] {len(records)} communities, {k_clusters} step-6 clusters, "
          f"{len({(r['dataset'], r['ontology']) for r in records})} ontologies")

    dirs = RELEASE_DIRS if args.corpus_layout == "release" else DATASETS
    if args.corpus:
        from run_method_eval import walk_corpus
        pop = walk_corpus(args.corpus)
        print(f"corpus {args.corpus}: {len(pop)} ontologies")
    else:
        if not (args.manifest and args.data_root):
            ap.error("give either --corpus DIR, or --manifest and --data-root")
        pop = load_population(args.manifest, args.datasets, args.subset)
        print(f"population ({args.subset}): {len(pop)} ontologies")
    by_id = {(r["dataset"], r["ontology_id"]): r for r in pop}

    approved = None
    if args.approved_labels:
        from odpqual.gtmodes import load_approved_labels
        approved = load_approved_labels(args.approved_labels)

    by_onto: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        by_onto[(r["dataset"], r["ontology"])].append(r)

    out_path = args.run / args.out_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    scored_rows: list[dict] = []
    n_emitted = 0
    t0 = time.time()

    with tmp_path.open("w", encoding="utf-8") as pfh:
        for i, ((ds, oid), rs) in enumerate(by_onto.items(), 1):
            src = by_id.get((ds, oid), {})
            onto_dir = (Path(src["dir"]) if src.get("dir")
                        else args.data_root / dirs[ds] / oid)
            mf = (Path(src["file"]) if src.get("file")
                  else main_ontology_file(onto_dir))
            g, _err = parse_graph(mf)
            if g is None:
                print(f"  [skip] {ds}/{oid}: parse failed ({_err})")
                continue
            gs, _ = strip_annotations(g)
            ents = set(collect_entities(gs, wide=True))
            kw = {}
            if approved is not None:
                kw["approved_labels"] = approved
            gt = ground_truth(onto_dir, ents, True,
                              complete_only=(args.subset == "complete"),
                              host_graph=gs, dataset=ds, **kw)

            # Exactly run_asprino_full.py's two readings: step 1 keeps each
            # community apart; step 6 merges the communities of one ontology
            # that landed in the same cross-corpus cluster.
            step1 = {j: set(r["entities"]) & ents for j, r in enumerate(rs)}
            step1 = {j: v for j, v in step1.items() if v}
            merged: dict[int, set[str]] = {}
            for r in rs:
                merged.setdefault(r["cluster"], set()).update(
                    set(r["entities"]) & ents)
            merged = {j: v for j, v in merged.items() if v}

            for method, clusters in (("asprino_cc", step1),
                                     ("asprino_cc_full", merged)):
                pfh.write(json.dumps({
                    "corpus": args.corpus_name, "dataset": ds,
                    "ontology": oid, "method": method,
                    "n_entities": len(ents), "scored": bool(gt),
                    "n_clusters": len(clusters),
                    "clusters": [sorted(c) for c in clusters.values()],
                }) + "\n")
                n_emitted += 1
                if not gt:
                    continue
                s = match_and_score(clusters, gt)
                scored_rows.append({
                    "dataset": ds, "method": method,
                    "precision": s["precision"], "recall": s["recall"],
                    "f1": s["f1"], "n_predicted": s["n_predicted"]})
            if i % 50 == 0:
                print(f"  [{i}/{len(by_onto)}] {n_emitted} records, "
                      f"{time.time()-t0:.0f}s", flush=True)

    print(f"[emitted] {n_emitted} records in {time.time()-t0:.0f}s")

    # ---------------- self-verification against the published scores --------
    published = args.run / "asprino_full_results.csv"
    if not published.is_file():
        print(f"[warn] no {published.name}; nothing to verify against")
        tmp_path.replace(out_path)
        print(f"[OK] {out_path}")
        return

    import numpy as np
    got: dict[tuple[str, str], dict[str, float]] = {}
    for ds in sorted({r["dataset"] for r in scored_rows}):
        for m in sorted({r["method"] for r in scored_rows}):
            sub = [r for r in scored_rows
                   if r["dataset"] == ds and r["method"] == m]
            if not sub:
                continue
            got[(ds, m)] = {
                "n_ontologies": float(len(sub)),
                "precision": round(float(np.mean([r["precision"] for r in sub])), 4),
                "recall": round(float(np.mean([r["recall"] for r in sub])), 4),
                "f1": round(float(np.mean([r["f1"] for r in sub])), 4),
                "mean_n_predicted": round(
                    float(np.mean([r["n_predicted"] for r in sub])), 2),
            }

    bad: list[str] = []
    seen = set()
    with published.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], row["method"])
            seen.add(key)
            if key not in got:
                bad.append(f"{key}: published but not reproduced")
                continue
            g = got[key]
            if int(row["k_clusters"]) != k_clusters:
                bad.append(f"{key}: k_clusters {row['k_clusters']} "
                           f"!= {k_clusters}")
            for field, cast in (("n_ontologies", int), ("precision", float),
                                ("recall", float), ("f1", float),
                                ("mean_n_predicted", float)):
                exp, act = cast(row[field]), g[field]
                if abs(exp - act) > 1e-9:
                    bad.append(f"{key}.{field}: published {exp} != {act}")
    for key in sorted(set(got) - seen):
        bad.append(f"{key}: reproduced but not published")

    print()
    for key in sorted(got):
        g = got[key]
        print(f"  {key[0]:<8} {key[1]:<18} n={int(g['n_ontologies'])} "
              f"P={g['precision']:.4f} R={g['recall']:.4f} F1={g['f1']:.4f} "
              f"clusters={g['mean_n_predicted']:.2f}")

    if bad:
        print("\n[FAIL] emitted clusters do not reproduce the published scores:")
        for b in bad:
            print("  " + b)
        if not args.force:
            tmp_path.unlink(missing_ok=True)
            print(f"\nNothing written. Re-run with --force to keep "
                  f"{out_path.name} anyway.")
            sys.exit(1)
        print("\n[--force] writing anyway.")
    else:
        print(f"\n[verified] every figure in {published.name} reproduced "
              f"exactly from the emitted clusters.")

    tmp_path.replace(out_path)
    print(f"[OK] {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
