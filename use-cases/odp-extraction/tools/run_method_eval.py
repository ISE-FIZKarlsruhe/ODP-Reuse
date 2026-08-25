#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_method_eval.py -- UC1 for the methods the benchmark was built for.

``run_extraction_eval.py`` evaluates embedding + clustering baselines. This
runner evaluates, on the same population, with the same ground truth and the
same Hungarian entity-level metric:

  asprino_cc      Asprino, Carriero & Presutti (K-CAP 2021), step 1, run from
                  the authors' own code (third_party/conceptual_components/).
  law_cfp         Lawrynowicz et al. (SWJ 2018), class-frame-pattern level.
  law_ap          the same, axiom-pattern level.
  namespace       entities grouped by IRI namespace -- the shortcut baseline.
  connected_components / all_in_one / singletons
                  structural and degenerate reference points.

Both cited methods are deterministic and select no hyper-parameter against the
ground truth, so each gets one number, not an oracle/unsupervised pair.

CORPUS
------
``--corpus-layout source`` reads the harvest tree (``github_dataset/``,
``noOBO/``, ``OBO/``); ``--corpus-layout release`` reads a release corpus
(``github/``, ``noOBO/``, ``OBO/``), which is how the IRI-anonymised twin
``release-iri/corpus-hashed`` is evaluated. Running both answers "which corpus
was UC1 run on" with two numbers instead of a claim.

Usage
-----
    python tools/run_method_eval.py \
        --manifest ../../pipeline/out/MANIFEST.csv \
        --data-root /path/to/MultiSource --out results/methods-iri \
        --datasets github noOBO --subset reuse
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
_PIPELINE = _REPO / "pipeline"
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_PIPELINE))
sys.path.insert(0, str(_USECASE / "tools"))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from extractbench import asprino, structural  # noqa: E402
from extractbench.evaluate import match_and_score  # noqa: E402
from extractbench.lawrynowicz import extract as law_extract, fit_corpus  # noqa: E402
from extractbench.lawrynowicz.axiom_trees import forest as _axiom_forest  # noqa: E402
from extractbench.lawrynowicz.treemine import FlatTree  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from run_extraction_eval import ground_truth, load_population  # noqa: E402

RELEASE_DIRS = {"github": "github", "noOBO": "noOBO", "OBO": "OBO"}
DS_LABEL = {"noOBO": "Paper \\& Repo", "OBO": "DOS--DP", "github": "GitHub"}

METHOD_LABEL = {
    "asprino_cc": "Asprino et al.\\ (step 1)",
    "law_cfp": "{\\L}awrynowicz et al.\\ (CFP)",
    "law_ap": "{\\L}awrynowicz et al.\\ (axiom pattern)",
    "namespace": "IRI namespace grouping",
    "connected_components": "Connected components",
    "all_in_one": "One cluster (all entities)",
    "singletons": "Singletons",
}
METHOD_ORDER = ["all_in_one", "singletons", "connected_components", "namespace",
                "asprino_cc", "law_ap", "law_cfp"]


def law_forest(graph):
    """One ontology's axiom forest, flattened, so the graph can be released."""
    return [FlatTree(t) for t in _axiom_forest(graph)]


def walk_corpus(root: Path) -> list[dict]:
    """Every ontology under ``root``, without a manifest.

    Two layouts are accepted, which between them cover the ODP-Reuse release
    and an arbitrary collection of ontology files:

      root/<name>/<name>.owl [+ root/<name>/patterns/*]   one directory each
      root/<name>.owl                                     a flat directory

    Ground truth is whatever ``patterns/`` yields; a directory without one is
    still processed, and its predictions are written unscored.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    # Both layouts are collected, not one or the other: a collection can hold a
    # few ontologies that came as directories beside many that came as single
    # files, and taking only the first kind found would silently drop the rest.
    for d in sorted(root.iterdir()):
        if d.is_dir() and main_ontology_file(d) is not None:
            rows.append({"dataset": root.name, "ontology_id": d.name,
                         "dir": str(d)})
            seen.add(d.name)
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".owl", ".ttl", ".rdf", ".xml", ".n3",
                                    ".nt", ".trig", ".jsonld"):
            continue
        name = f.stem
        while name in seen:
            name += "_file"
        seen.add(name)
        rows.append({"dataset": root.name, "ontology_id": name,
                     "dir": str(root), "file": str(f)})
    return rows


def _load(onto_dir: Path, strip_annots: bool, main_f: Path | None = None):
    """Parse one ontology; return (raw_graph, stripped_graph, entities) or a reason.

    ``main_f`` names the file explicitly, which a flat corpus directory needs:
    there, every ontology shares one directory and ``main_ontology_file`` would
    return the same (largest) file for all of them.
    """
    if main_f is None:
        main_f = main_ontology_file(onto_dir)
    if main_f is None:
        return None, "no_ontology_file", ""
    g, err = parse_graph(main_f)
    if g is None:
        return None, "parse_error", err
    gs = strip_annotations(g)[0] if strip_annots else g
    ents = collect_entities(gs, wide=True)
    if not ents:
        return None, "no_entities", ""
    return (g, gs, ents), "", ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="MANIFEST.csv; with --data-root this evaluates the "
                         "quality-gated ODP-Reuse population. Omit both and "
                         "pass --corpus instead to run over any directory of "
                         "ontologies, with or without ground truth.")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="a directory of ontologies: either one subdirectory "
                         "per ontology (with an optional patterns/ subdirectory "
                         "holding the reference patterns) or a flat directory "
                         "of RDF files. Ground truth is used where patterns/ "
                         "yields any overlap and simply absent otherwise.")
    ap.add_argument("--emit-predictions", action="store_true",
                    help="write every predicted cluster to predictions.jsonl, "
                         "whether or not it can be scored. Implied by --corpus.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--datasets", nargs="*", default=["noOBO", "github"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--corpus-layout", choices=("source", "release"),
                    default="source",
                    help="'source' = the harvest tree (github_dataset/...); "
                         "'release' = a release corpus (github/...), which is "
                         "how the IRI-anonymised twin is evaluated")
    ap.add_argument("--corpus-name", default="",
                    help="free-text tag written into every result row, e.g. "
                         "'plain' or 'anonymised'")
    ap.add_argument("--methods", nargs="*", default=list(METHOD_ORDER))
    ap.add_argument("--max-entities", type=int, default=0,
                    help="0 = no limit. Unlike the embedding runner, none of "
                         "these methods generates walks per entity, so the "
                         "DOS-DP portion is reachable.")
    ap.add_argument("--max-mb", type=float, default=0.0)
    ap.add_argument("--asprino-max-entities", type=int, default=0,
                    help="skip conceptual-component extraction on ontologies "
                         "with more entities than this (0 = no limit). The "
                         "method builds an intensional graph by SPARQL "
                         "construction and then detects communities in it "
                         "recursively, so its cost grows steeply: a 377k-triple "
                         "ontology does not finish in ten minutes. Every skip is "
                         "recorded in methods_skipped.csv, and the other methods "
                         "still run on that ontology.")
    ap.add_argument("--keep-annotations", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--law-min-support", type=int, default=4,
                    help="sigma_F: ontologies a subtree must occur in (paper: 4)")
    ap.add_argument("--law-sigma-cf", type=int, default=4,
                    help="sigma_CF: classes an itemset must hold for (paper: 4)")
    ap.add_argument("--law-support-unit", choices=("forest", "tree"),
                    default="forest",
                    help="'forest' = the paper's cross-corpus run (support "
                         "counted in ontologies); 'tree' = its per-ontology run "
                         "(support counted in axioms, relative threshold 1%%)")
    ap.add_argument("--law-max-size", type=int, default=12,
                    help="subtree size cap; the paper's observed cross-corpus "
                         "range is 2-12")
    ap.add_argument("--approved-labels", type=Path, default=None)
    args = ap.parse_args()

    approved: set[tuple[str, str, str]] = set()
    if args.approved_labels and args.approved_labels.is_file():
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.add((r["dataset"], r["ontology"], r["pattern"]))

    if not args.corpus and not (args.manifest and args.data_root):
        ap.error("give either --corpus DIR, or --manifest and --data-root")

    # Resolve every input path before it is used. A relative --data-root turns
    # each ontology path into cwd + "\..\..\" + the relative tail, and Windows
    # applies its 260-character limit to that un-normalised string: directory
    # listings then come back missing files that plainly exist, and the ontology
    # is recorded as "no_ontology_file". Eighteen ontologies of the
    # IRI-anonymised corpus were lost this way before this line existed.
    for _attr in ("data_root", "corpus", "manifest", "out", "top_level",
                  "approved_labels"):
        _v = getattr(args, _attr, None)
        if isinstance(_v, Path):
            setattr(args, _attr, _v.resolve())

    args.out.mkdir(parents=True, exist_ok=True)
    strip_annots = not args.keep_annotations
    dirs = RELEASE_DIRS if args.corpus_layout == "release" else DATASETS
    corpus = args.corpus_name or (args.corpus.name if args.corpus
                                  else args.corpus_layout)
    emit = args.emit_predictions or bool(args.corpus)

    if args.corpus:
        pop = walk_corpus(args.corpus)
        print(f"corpus {args.corpus}: {len(pop)} ontologies", flush=True)
    else:
        pop = load_population(args.manifest, args.datasets, args.subset)
        print(f"population ({args.subset}): {len(pop)} ontologies across "
              f"{sorted({r['dataset'] for r in pop})}; corpus={corpus}",
              flush=True)
    if args.limit:
        pop = pop[:args.limit]

    # ---------------- stage 1: load every ontology once ----------------
    #
    # ONE parse per ontology, and the graph is dropped as soon as the methods
    # that need it have run. Holding every parsed graph until the corpus-level
    # Lawrynowicz fit needs them costs tens of GB on a corpus of large
    # ontologies -- the materials-science collection peaks above 14 GB that way
    # -- so the graph-consuming methods run inside this loop and only their
    # output, a few sets of IRIs, is carried forward.
    skipped: list[tuple[str, str, str, str]] = []
    loaded: dict[tuple[str, str], dict] = {}
    graph_methods = [m for m in args.methods
                     if m == "asprino_cc" or m in structural.METHODS]
    need_law = any(m.startswith("law_") for m in args.methods)
    forests: dict[str, object] = {}
    t0 = time.time()
    for i, r in enumerate(pop, 1):
        ds, oid = r["dataset"], r["ontology_id"]
        onto_dir = (Path(r["dir"]) if r.get("dir")
                    else args.data_root / dirs[ds] / oid)
        if not onto_dir.is_dir():
            skipped.append((ds, oid, "no_ontology_dir", str(onto_dir)))
            continue
        # In a flat corpus every ontology shares one directory, so the file has
        # to be named explicitly: main_ontology_file() would return the largest
        # file in that directory for all of them.
        mf = Path(r["file"]) if r.get("file") else main_ontology_file(onto_dir)
        if mf is not None and args.max_mb > 0 and \
                mf.stat().st_size > args.max_mb * 1024 * 1024:
            skipped.append((ds, oid, "file_too_large",
                            f"{mf.stat().st_size / 1e6:.0f}MB>{args.max_mb}MB"))
            continue
        got, reason, detail = _load(onto_dir, strip_annots, mf)
        if got is None:
            skipped.append((ds, oid, reason, detail))
            continue
        g, gs, ents = got
        if args.max_entities > 0 and len(ents) > args.max_entities:
            skipped.append((ds, oid, "too_many_entities",
                            f"{len(ents)}>{args.max_entities}"))
            continue
        gt = ground_truth(onto_dir, set(ents), strip_annots,
                          complete_only=(args.subset == "complete"),
                          host_graph=gs, dataset=ds,
                          approved_labels=approved or None)
        if not gt and not emit:
            skipped.append((ds, oid, "no_ground_truth", f"subset={args.subset}"))
            continue
        if not gt:
            # No answer key for this ontology. Extraction still runs and its
            # output is written to predictions.jsonl; only the scoring is
            # skipped, and the ontology is recorded as unscored rather than
            # dropped. This is the mode a corpus without ground truth uses.
            skipped.append((ds, oid, "no_ground_truth_predict_only",
                            f"subset={args.subset}"))
        # Run every graph-consuming method now, while the graph is in hand.
        clusters_by_method: dict[str, dict[int, set[str]]] = {}
        timings: dict[str, float] = {}
        for m in graph_methods:
            tm = time.time()
            try:
                if m == "asprino_cc":
                    if (args.asprino_max_entities > 0
                            and len(ents) > args.asprino_max_entities):
                        skipped.append((ds, oid, "asprino_too_many_entities",
                                        f"{len(ents)}>{args.asprino_max_entities}"))
                        continue
                    clusters_by_method[m] = asprino.extract(g, ents)
                else:
                    clusters_by_method[m] = structural.METHODS[m](gs, ents)
            except MemoryError:
                skipped.append((ds, oid, "method_out_of_memory", m))
            except Exception as exc:  # noqa: BLE001
                skipped.append((ds, oid, "method_error",
                                f"{m}: {type(exc).__name__}: {exc}"))
            timings[m] = time.time() - tm
        if need_law:
            forests[f"{ds}/{oid}"] = law_forest(gs)

        loaded[(ds, oid)] = {"entities": ents, "gt": gt, "dir": onto_dir,
                             "file": mf, "clusters": clusters_by_method,
                             "timings": timings}
        del g, gs                      # the graph is not needed again
        # A corpus of large ontologies spends minutes per file; per-ontology
        # progress is what makes such a run legible while it is happening.
        every = 1 if len(pop) <= 200 else 25
        if i % every == 0:
            print(f"  loaded {i}/{len(pop)} {oid[:48]} "
                  f"({len(ents)} ents, {time.time() - t0:.0f}s)", flush=True)
    print(f"[load] {len(loaded)} usable, {len(skipped)} skipped, "
          f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- stage 2: corpus-level fit (Lawrynowicz) ----------
    fitted = None
    law_stats: dict = {}
    if any(m.startswith("law_") for m in args.methods) and loaded:
        t = time.time()
        print("[lawrynowicz] mining the corpus", flush=True)
        fitted = fit_corpus(
            forests, min_support=args.law_min_support,
            sigma_cf=args.law_sigma_cf, support_unit=args.law_support_unit,
            max_size=args.law_max_size, pre_encoded=True,
            progress=lambda m: print(m, flush=True))
        law_stats = dict(fitted.stats)
        law_stats["fit_seconds"] = round(time.time() - t, 1)
        (args.out / "lawrynowicz_mining_stats.json").write_text(
            json.dumps(law_stats, indent=2), encoding="utf-8")
        print(f"[lawrynowicz] fit in {law_stats['fit_seconds']}s", flush=True)

    # ---------------- stage 3: score every method ----------------------
    per_path = args.out / "methods_per_ontology.jsonl"
    pred_path = args.out / "predictions.jsonl"
    recs: list[dict] = []
    unscored = 0
    pf = pred_path.open("w", encoding="utf-8") if emit else None
    with per_path.open("w", encoding="utf-8") as fh:
        for n, ((ds, oid), v) in enumerate(loaded.items(), 1):
            ents, gt = v["entities"], v["gt"]
            for m in args.methods:
                t = time.time()
                try:
                    if m == "law_cfp":
                        clusters = law_extract(fitted, f"{ds}/{oid}", ents,
                                               "class_frame")
                    elif m == "law_ap":
                        clusters = law_extract(fitted, f"{ds}/{oid}", ents,
                                               "axiom_pattern")
                    elif m in v["clusters"]:
                        clusters = v["clusters"][m]
                        t = time.time() - v["timings"].get(m, 0.0)
                    else:
                        continue          # already recorded as a skip above
                except Exception as exc:  # noqa: BLE001
                    skipped.append((ds, oid, "method_error",
                                    f"{m}: {type(exc).__name__}: {exc}"))
                    continue
                if pf is not None:
                    pf.write(json.dumps({
                        "corpus": corpus, "dataset": ds, "ontology": oid,
                        "method": m, "n_entities": len(ents),
                        "n_clusters": len(clusters),
                        "scored": bool(gt),
                        "clusters": [sorted(c) for c in clusters.values()],
                    }) + "\n")
                if not gt:
                    # No answer key: the prediction is kept, the score is not
                    # invented. Reporting 0.000 here would be a measurement of
                    # nothing.
                    unscored += 1
                    continue
                s = match_and_score(clusters, gt)
                rec = {"corpus": corpus, "dataset": ds, "ontology": oid,
                       "method": m, "n_entities": len(ents), "n_gt": len(gt),
                       "precision": round(s["precision"], 4),
                       "recall": round(s["recall"], 4),
                       "f1": round(s["f1"], 4),
                       "n_predicted": s["n_predicted"],
                       "n_matched": s["n_matched"],
                       "seconds": round(time.time() - t, 3)}
                fh.write(json.dumps(rec) + "\n")
                recs.append(rec)
            if n % 25 == 0 or n == len(loaded):
                print(f"[{n}/{len(loaded)}] {time.time() - t0:.0f}s", flush=True)

    # ---------------- skip report (never silent) -----------------------
    if skipped:
        sp = args.out / "methods_skipped.csv"
        with sp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "ontology", "reason", "detail"])
            w.writerows(skipped)
        from collections import Counter
        tally = Counter(f"{d}:{r}" for d, _o, r, _x in skipped)
        print(f"\n[SKIPPED] {len(skipped)} -> {sp.name}")
        for k, v2 in tally.most_common():
            print(f"    {k}: {v2}")

    if pf is not None:
        pf.close()
        print(f"[OK] {pred_path} ({unscored} (ontology, method) predictions "
              f"written without a score)")

    if not recs:
        print("No ontology in this run had ground truth, so nothing was scored. "
              "The extraction output is in predictions.jsonl.")
        return

    # ---------------- aggregate ----------------------------------------
    import numpy as np
    agg: list[dict] = []
    for ds in sorted({r["dataset"] for r in recs}):
        for m in [x for x in METHOD_ORDER if x in args.methods]:
            sub = [r for r in recs if r["dataset"] == ds and r["method"] == m]
            if not sub:
                continue
            agg.append({
                "corpus": corpus, "dataset": ds, "method": m,
                "n_ontologies": len(sub),
                "precision": round(float(np.mean([r["precision"] for r in sub])), 4),
                "recall": round(float(np.mean([r["recall"] for r in sub])), 4),
                "f1": round(float(np.mean([r["f1"] for r in sub])), 4),
                "mean_n_predicted": round(
                    float(np.mean([r["n_predicted"] for r in sub])), 2),
                "mean_n_gt": round(float(np.mean([r["n_gt"] for r in sub])), 2),
                "median_seconds": round(
                    float(np.median([r["seconds"] for r in sub])), 3),
            })

    with (args.out / "methods_results.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    print(f"\n[OK] {args.out / 'methods_results.csv'}")
    for row in agg:
        print(f"  {row['dataset']:8s} {row['method']:22s} "
              f"P={row['precision']:.3f} R={row['recall']:.3f} "
              f"F1={row['f1']:.3f}  clusters={row['mean_n_predicted']:.2f} "
              f"(gt {row['mean_n_gt']:.2f})  n={row['n_ontologies']}")


if __name__ == "__main__":
    main()
