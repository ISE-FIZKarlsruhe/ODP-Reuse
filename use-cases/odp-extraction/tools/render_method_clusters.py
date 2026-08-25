#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_method_clusters.py -- a figure and an OWL module per predicted cluster,
for every extraction method, not only conceptual components.

``render_cc_figures.py`` draws the conceptual-components communities, which it
can because step 1 hands it the relations directly. Every other method in this
use case -- the embedding baselines, the Lawrynowicz tree-mining variants,
the structural reference points -- returns a cluster as a bare set of entity
IRIs and no relations at all. To draw one, or to write it as RDF, the axioms
that hold it together have to be recovered from the host ontology.

That recovery is module extraction, and it is done the way
``PatternMiner/rq_11_pattern_induction/src/robot_extract.py`` does it: ROBOT
Syntactic Locality Module Extraction where ROBOT can be run, an indexed rdflib
approximation otherwise. See ``extractbench/modules.py`` for what each backend
does and where they differ.

The module is then the single source of truth for both outputs: the ``.ttl`` is
that module, and the ``.png`` draws the module's entity-level relations. A
figure and its RDF can therefore never disagree.

OUTPUT LAYOUT
-------------
As for conceptual components, with the method between the dataset and the
ontology so the methods stay apart and stay comparable:

    <out>/<dataset>/<method>/<ontology>/Cluster001.png
                                       /Cluster001.ttl
                                       /clusters.json
                                       /INDEX.txt

A NOTE ON `singletons`
----------------------
``singletons`` returns one cluster per entity -- 54,344 of them on the MSE
corpus, each a module for a single term. They are a lower-bound reference point
rather than an extraction, and drawing 54,344 one-node figures is not
informative, so the method is skipped unless it is named explicitly in
``--methods``.

Usage
-----
    python tools/render_method_clusters.py \\
        --predictions results/mseo/predictions.jsonl \\
                      results/mseo-embed/predictions.jsonl \\
        --corpus corpora/mseo --out results/method-figures

    python tools/render_method_clusters.py \\
        --predictions results/asprino-full/predictions.jsonl \\
        --manifest ../../pipeline/out/MANIFEST.csv \\
        --data-root C:/Users/eno/Documents/PhD/MultiSource_server \\
        --out results/method-figures

    # authoritative modules for one method, via ROBOT
    python tools/render_method_clusters.py ... --backend robot \\
        --robot-jar C:/path/to/robot.jar --methods "owl2vec|kmeans"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_USECASE / "tools"))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from run_extraction_eval import load_population  # noqa: E402

RELEASE_DIRS = {"github": "github", "noOBO": "noOBO", "OBO": "OBO"}
SKIP_BY_DEFAULT = {"singletons"}
MAX_PATH = 240
BAD = '<>:"/\\|?*'


def safe_name(name: str) -> str:
    out = "".join("_" if c in BAD or ord(c) < 32 else c for c in name)
    return out.rstrip(" .") or "_"


def verify(root: Path) -> None:
    """Report what a rendered method tree holds, and what is unfinished."""
    per = {}
    incomplete, strays = [], 0
    for manifest in sorted(root.glob("*/*/*/clusters.json")):
        d = manifest.parent
        ds, method = d.parent.parent.name, d.parent.name
        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            incomplete.append(f"{ds}/{method}/{d.name} (unreadable manifest)")
            continue
        s = per.setdefault((ds, method), {"ontologies": 0, "clusters": 0,
                                          "figures": 0, "rdf": 0,
                                          "skipped": 0, "missing": 0})
        s["ontologies"] += 1
        for c in meta.get("clusters", ()):
            s["clusters"] += 1
            if (d / c["rdf"]).is_file():
                s["rdf"] += 1
            else:
                s["missing"] += 1
            if c.get("figure"):
                if (d / c["figure"]).is_file():
                    s["figures"] += 1
                else:
                    s["missing"] += 1
            elif c.get("figure_skipped"):
                s["skipped"] += 1
            else:
                s["missing"] += 1
        if not _method_complete(d):
            incomplete.append(f"{ds}/{method}/{d.name}")
        strays += len([q for q in d.glob("Cluster*")
                       if q.is_file() and q.suffix == ""])

    print(f"root: {root}")
    print(f"  {'dataset':<8}{'method':<26}{'onto':>6}{'clusters':>9}"
          f"{'figures':>9}{'rdf':>8}{'no-fig':>8}{'missing':>9}")
    tot = {"ontologies": 0, "clusters": 0, "figures": 0, "rdf": 0,
           "skipped": 0, "missing": 0}
    for (ds, m), s in sorted(per.items()):
        for k in tot:
            tot[k] += s[k]
        print(f"  {ds:<8}{m:<26}{s['ontologies']:>6}{s['clusters']:>9}"
              f"{s['figures']:>9}{s['rdf']:>8}{s['skipped']:>8}"
              f"{s['missing']:>9}")
    print(f"  {'TOTAL':<34}{tot['ontologies']:>6}{tot['clusters']:>9}"
          f"{tot['figures']:>9}{tot['rdf']:>8}{tot['skipped']:>8}"
          f"{tot['missing']:>9}")
    if strays:
        print(f"  {strays} Graphviz source files left behind")
    if incomplete:
        print(f"  {len(incomplete)} ontology-methods incomplete "
              f"(re-run with --skip-existing to finish them):")
        for i in incomplete[:12]:
            print(f"      {i}")
    elif not tot["missing"]:
        print("  [OK] every cluster has its module, and its figure or a "
              "recorded reason for not having one")


def _method_complete(d: Path) -> bool:
    """Has this ontology-method directory really been finished?

    A killed run leaves `clusters.json` behind: the worker's Graphviz children
    die with it, every render raises, the loop records `figure: null` and writes
    the manifest anyway. Treating the manifest's existence as "done" then makes
    --skip-existing preserve exactly the output that needs redoing. So the
    manifest is checked against the files it claims.
    """
    manifest = d / "clusters.json"
    if not manifest.is_file():
        return False
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    for c in meta.get("clusters", ()):
        if not (d / c["rdf"]).is_file():
            return False
        # A figure is legitimately absent only when it was deliberately skipped.
        if not c.get("figure"):
            if not c.get("figure_skipped"):
                return False
        elif not (d / c["figure"]).is_file():
            return False
    return True


def render_ontology(job: dict) -> dict:
    """Every cluster of every method for one ontology. Worker process."""
    sys.path[:0] = [job["usecase"], job["pipeline"], job["tools"]]
    from odpqual.entities import parse_graph
    from extractbench.modules import (TermIndex, drawable, extract_via_robot,
                                      robot_command)
    from extractbench.render_cc import render_graph

    t0 = time.time()
    res = {"dataset": job["dataset"], "ontology": job["ontology"],
           "methods": 0, "clusters": 0, "figures": 0, "modules": 0,
           "robot": 0, "python": 0, "truncated": 0, "too_big": 0,
           "error": ""}
    try:
        g, err = parse_graph(Path(job["file"]))
        if g is None:
            res["error"] = f"parse_error: {err}"
            return res
        index = TermIndex(g)
        cmd = (robot_command(job["robot_jar"])
               if job["backend"] == "robot" else None)
        if job["backend"] == "robot" and cmd is None:
            res["error"] = "robot_unavailable"
            return res

        for method, clusters in job["by_method"].items():
            out = Path(job["out"]) / safe_name(method) / job["dirname"]
            out.mkdir(parents=True, exist_ok=True)
            res["methods"] += 1
            shown = clusters
            if job["max_clusters"] and len(clusters) > job["max_clusters"]:
                shown = clusters[: job["max_clusters"]]
                res["truncated"] += len(clusters) - len(shown)
            width = max(3, len(str(len(shown))))
            records = []

            for i, seeds in enumerate(shown, 1):
                stem = out / f"Cluster{i:0{width}d}"
                ttl = stem.with_suffix(".ttl")
                backend = "python"
                if cmd is not None and extract_via_robot(
                        cmd, Path(job["file"]), seeds, ttl, job["robot_method"]):
                    backend = "robot"
                    res["robot"] += 1
                    from rdflib import Graph
                    module = Graph()
                    try:
                        module.parse(str(ttl))
                    except Exception:              # noqa: BLE001
                        module = index.module(seeds)
                else:
                    module = index.module(seeds)
                    header = (f"# Predicted cluster {i} of {len(clusters)}\n"
                              f"# dataset   : {job['dataset']}\n"
                              f"# ontology  : {job['ontology']}\n"
                              f"# method    : {method}\n"
                              f"# source    : {job['file']}\n"
                              f"# seeds     : {len(seeds)}\n"
                              f"# axioms    : {len(module)}\n"
                              f"# backend   : rdflib module approximation "
                              f"(mentions of a seed, blank nodes closed)\n\n")
                    ttl.write_text(header + module.serialize(format="turtle"),
                                   encoding="utf-8")
                    res["python"] += 1
                res["modules"] += 1

                triples, singles = drawable(module, seeds)
                figure = None
                # A cluster of thousands of entities -- `all_in_one` is the
                # whole ontology, and `namespace` reaches 8,832 -- has no
                # legible drawing, and laying one out costs far more than it
                # returns. The module is still written; only the figure is
                # skipped, and every skip is counted and recorded so the
                # absence is never mistaken for a failure.
                n_nodes = len({s for s, _, _, _ in triples}
                              | {o for _, _, o, _ in triples} | set(singles))
                if job["max_figure_nodes"] and n_nodes > job["max_figure_nodes"]:
                    res["too_big"] += 1
                    records.append({"cluster": i, "figure": None,
                                    "figure_skipped": "too_many_nodes",
                                    "n_figure_nodes": n_nodes,
                                    "rdf": ttl.name, "backend": backend,
                                    "n_seeds": len(seeds),
                                    "n_axioms": len(module),
                                    "n_drawn_relations": len(triples),
                                    "seeds": sorted(seeds)})
                    res["clusters"] += 1
                    continue
                try:
                    produced = render_graph(
                        triples, singles, stem,
                        title=(f"{job['dataset']} / {job['ontology']}\\n"
                               f"{method} — cluster {i} of {len(clusters)}\\n"
                                              f"{len(seeds)} entities (solid), "
                               f"{len(module)} axioms"),
                        fmt=job["format"], dpi=job["dpi"],
                        highlight=set(seeds))
                    if produced is not None:
                        figure = Path(produced).name
                        res["figures"] += 1
                except Exception as exc:           # noqa: BLE001
                    res["error"] = (res["error"]
                                    + f"; {method}/c{i}: {exc}").strip("; ")
                records.append({"cluster": i, "figure": figure,
                                "rdf": ttl.name, "backend": backend,
                                "n_seeds": len(seeds), "n_axioms": len(module),
                                "n_drawn_relations": len(triples),
                                "seeds": sorted(seeds)})
                res["clusters"] += 1

            (out / "clusters.json").write_text(json.dumps({
                "dataset": job["dataset"], "ontology": job["ontology"],
                "method": method, "source_file": job["file"],
                "n_clusters_predicted": len(clusters),
                "n_clusters_written": len(shown),
                "clusters": records}, indent=1), encoding="utf-8")
            (out / "INDEX.txt").write_text(
                f"dataset   : {job['dataset']}\n"
                f"ontology  : {job['ontology']}\n"
                f"method    : {method}\n"
                f"source    : {job['file']}\n"
                f"clusters  : {len(shown)} written"
                + (f" of {len(clusters)} predicted (truncated by --max-clusters)"
                   if len(shown) != len(clusters) else "") + "\n\n"
                "Each Cluster<n>.ttl is an OWL module extracted from the source\n"
                "ontology for that cluster's entities, and Cluster<n>."
                f"{job['format']}\n"
                "draws that module: arrows are relations between two named\n"
                "entities, hollow arrowheads are subClassOf/subPropertyOf, and\n"
                "dashed lines are equivalence. rdf:type and axioms pointing at\n"
                "OWL/RDFS vocabulary are omitted from the figure but are in the\n"
                "module. Seed IRIs are listed in clusters.json.\n",
                encoding="utf-8")
    except Exception:                              # noqa: BLE001
        res["error"] = traceback.format_exc(limit=3)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--datasets", nargs="*", default=["noOBO", "github"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--corpus-layout", choices=("source", "release"),
                    default="source")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="only these methods (default: all but "
                         + ", ".join(sorted(SKIP_BY_DEFAULT)) + ")")
    ap.add_argument("--backend", choices=("python", "robot"), default="python")
    ap.add_argument("--robot-jar", default=None,
                    help="path to robot.jar; omit to use `robot` on PATH")
    ap.add_argument("--robot-method", default="STAR",
                    choices=("STAR", "BOT", "TOP", "MIREOT"))
    ap.add_argument("--max-clusters", type=int, default=0,
                    help="cap the clusters written per ontology-method; "
                         "0 means no cap. What is dropped is logged and "
                         "recorded, never silently omitted")
    ap.add_argument("--max-figure-nodes", type=int, default=300,
                    help="skip the figure (never the module) for a cluster "
                         "that would draw more nodes than this; 0 disables")
    ap.add_argument("--format", default="png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="render nothing; report what --out already holds and "
                         "which ontology-methods are incomplete")
    args = ap.parse_args()

    for _a in ("manifest", "data_root", "corpus", "out"):
        v = getattr(args, _a)
        if v is not None:
            setattr(args, _a, v.resolve())

    if args.verify_only:
        verify(args.out)
        return

    # ---- predicted clusters, grouped by ontology so each parses once -------
    wanted = set(args.methods) if args.methods else None
    grouped: dict[tuple[str, str], dict[str, list]] = defaultdict(dict)
    skipped_methods: dict[str, int] = defaultdict(int)
    collisions: dict[str, int] = defaultdict(int)

    # The same method can appear in two prediction files -- asprino_cc is in
    # both the step-1 sweep and the full 1-7 run, with different cluster counts
    # because the runs differ. Letting the second silently replace the first
    # would publish one run's figures under a name that claims both. Which
    # methods clash is settled in one pass FIRST, so a clashing method is
    # qualified by its run in every directory rather than only where the two
    # runs happen to cover the same ontology -- otherwise one method ends up
    # split across `asprino_cc`, `asprino_cc@mseo` and `asprino_cc@mseo-cc-full`
    # and no directory holds all of it.
    runs_per_method: dict[str, set[str]] = defaultdict(set)
    for pf in args.predictions:
        for line in pf.open(encoding="utf-8"):
            if line.strip():
                runs_per_method[json.loads(line)["method"]].add(pf.parent.name)
    clashing = {m for m, runs in runs_per_method.items() if len(runs) > 1}

    for pf in args.predictions:
        run = pf.parent.name          # e.g. "mseo-cc-full"
        for line in pf.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["method"]
            if wanted is not None and m not in wanted:
                continue
            if wanted is None and m in SKIP_BY_DEFAULT:
                skipped_methods[m] += r["n_clusters"]
                continue
            key = (r["dataset"], r["ontology"])
            if m in clashing:
                collisions[m] += 1
                m = f"{m}@{run}"
            grouped[key][m] = r["clusters"]
    if not grouped:
        print("no predictions matched")
        return
    for m, n in sorted(skipped_methods.items()):
        print(f"[skip] method {m!r}: {n} clusters "
              f"(name it in --methods to include it)")
    for m in sorted(clashing):
        print(f"[note] method {m!r} appears in "
              f"{len(runs_per_method[m])} prediction files "
              f"({', '.join(sorted(runs_per_method[m]))}); every ontology is "
              f"filed under {m}@<run> so no run is lost and none is mixed")

    # ---- resolve each ontology to a file ----------------------------------
    dirs = RELEASE_DIRS if args.corpus_layout == "release" else DATASETS
    if args.corpus:
        from run_method_eval import walk_corpus
        pop = walk_corpus(args.corpus)
    elif args.manifest and args.data_root:
        pop = load_population(args.manifest, args.datasets, args.subset)
    else:
        ap.error("give either --corpus DIR, or --manifest and --data-root")
    by_id = {(r["dataset"], r["ontology_id"]): r for r in pop}

    jobs, missing = [], []
    for (ds, oid), by_method in sorted(grouped.items()):
        src = by_id.get((ds, oid))
        if src is None:
            missing.append((ds, oid))
            continue
        onto_dir = (Path(src["dir"]) if src.get("dir")
                    else args.data_root / dirs[ds] / oid)
        mf = (Path(src["file"]) if src.get("file")
              else (main_ontology_file(onto_dir) if onto_dir.is_dir() else None))
        if mf is None or not Path(mf).is_file():
            missing.append((ds, oid))
            continue
        ds_dir = args.out / safe_name(ds)
        dirname = safe_name(oid)
        budget = MAX_PATH - len(str(ds_dir)) - 40
        if len(dirname) > budget > 8:
            import hashlib
            dirname = (dirname[: budget - 9] + "_"
                       + hashlib.sha1(oid.encode()).hexdigest()[:8])
        if args.skip_existing and all(
                _method_complete(ds_dir / safe_name(m) / dirname)
                for m in by_method):
            continue
        jobs.append({"dataset": ds, "ontology": oid, "file": str(mf),
                     "out": str(ds_dir), "dirname": dirname,
                     "by_method": by_method,
                     "backend": args.backend, "robot_jar": args.robot_jar,
                     "robot_method": args.robot_method,
                     "max_clusters": args.max_clusters,
                     "max_figure_nodes": args.max_figure_nodes,
                     "format": args.format, "dpi": args.dpi,
                     "usecase": str(_USECASE),
                     "pipeline": str(_REPO / "pipeline"),
                     "tools": str(_USECASE / "tools")})
    if args.limit:
        jobs = jobs[: args.limit]
    if missing:
        print(f"[skip] {len(missing)} ontologies have no readable source file")

    n_clusters = sum(len(c) for j in jobs for c in j["by_method"].values())
    print(f"{len(jobs)} ontologies, "
          f"{len({m for j in jobs for m in j['by_method']})} methods, "
          f"{n_clusters} clusters, backend={args.backend}, "
          f"{args.workers} workers -> {args.out}")
    if not jobs:
        return

    t0 = time.time()
    done = 0
    agg = {"clusters": 0, "figures": 0, "modules": 0, "robot": 0,
           "python": 0, "truncated": 0, "too_big": 0}
    failures, results = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(render_ontology, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            for k in agg:
                agg[k] += r.get(k, 0)
            if r["error"]:
                failures.append(r)
            if done % 10 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {agg['clusters']} clusters, "
                      f"{agg['figures']} figures, {agg['modules']} modules, "
                      f"{time.time()-t0:.0f}s", flush=True)

    report = {"out": str(args.out), "backend": args.backend,
              "robot_method": args.robot_method if args.backend == "robot" else None,
              "ontologies": len(jobs), **agg,
              "clusters_dropped_by_max_clusters": agg["truncated"],
              "max_figure_nodes": args.max_figure_nodes,
              "clusters_with_module_but_no_figure": agg["too_big"],
              "methods_skipped": dict(skipped_methods),
              "methods_split_by_run": dict(collisions),
              "seconds": round(time.time() - t0, 1),
              "no_source_file": [{"dataset": d, "ontology": o}
                                 for d, o in missing],
              "problems": [{k: r[k] for k in ("dataset", "ontology", "error")}
                           for r in failures]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "METHOD_RENDER_REPORT.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n[done] {agg['modules']} modules and {agg['figures']} figures for "
          f"{agg['clusters']} clusters in {report['seconds']:.0f}s")
    print(f"    backends: robot={agg['robot']}, python={agg['python']}")
    if agg["truncated"]:
        print(f"    {agg['truncated']} clusters dropped by --max-clusters")
    if agg["too_big"]:
        print(f"    {agg['too_big']} clusters got a module but no figure "
              f"(over --max-figure-nodes {args.max_figure_nodes})")
    if failures:
        print(f"[{len(failures)} ontologies had a problem; see the report]")
        for r in failures[:6]:
            print(f"    {r['dataset']}/{r['ontology']}: "
                  f"{r['error'].splitlines()[-1][:110]}")
    print(f"[OK] {args.out / 'METHOD_RENDER_REPORT.json'}")


if __name__ == "__main__":
    main()
