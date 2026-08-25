#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_cc_figures.py -- a figure per conceptual-components community, for every
ontology of every dataset.

Upstream's step 1 can draw each community it finds; the two ``*_as_images``
functions that do it were dropped when the code was vendored (PROVENANCE.md,
PATCH 2). ``extractbench/render_cc.py`` restores that, and this runs it over a
whole corpus.

WHY STEP 1 IS RE-RUN RATHER THAN READ BACK
------------------------------------------
``documents.json`` and ``clustered_communities.json`` hold each community's
entities, but only *after* the reified edge nodes have been expanded into the
three IRIs they stand for. Which triple an entity came from is gone by then, so
those files cannot say where to draw an arrow. Step 1 is therefore re-run here
with the edge nodes kept. It is the cheap stage -- around five minutes for 493
ontologies -- and it is deterministic, so the communities drawn are the
communities scored.

OUTPUT LAYOUT
-------------
One directory per dataset, one per ontology beneath it, and each ontology's
figures and its extracted communities together in that directory:

    <out>/<dataset>/<ontology>/Community001.png
                              /Community002.png
                              /<ontology>_greedy_mod_communities.txt
                              /communities.json
                              /INDEX.txt

``<ontology>_greedy_mod_communities.txt`` keeps upstream's name and block
structure; ``communities.json`` is the same content machine-readable, with each
community's triples, standalone entities and figure filename.

Usage
-----
    # ODP-Reuse, both human-authored datasets
    python tools/render_cc_figures.py \\
        --manifest ../../pipeline/out/MANIFEST.csv \\
        --data-root C:/Users/eno/Documents/PhD/MultiSource_server \\
        --datasets github noOBO --subset reuse \\
        --out results/cc-figures

    # MSE
    python tools/render_cc_figures.py --corpus corpora/mseo \\
        --out results/cc-figures

Both write into the same tree; the dataset directories keep them apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_USECASE / "tools"))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from odpqual.entities import parse_graph  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from run_extraction_eval import load_population  # noqa: E402

RELEASE_DIRS = {"github": "github", "noOBO": "noOBO", "OBO": "OBO"}

# Windows still refuses paths past 260 characters unless long paths are enabled
# for the machine, and several ontology ids are already 80 characters. Any
# directory name that would take the deepest figure path past this is shortened
# and the original recorded in NAME_MAP.txt beside it.
MAX_PATH = 240
BAD = '<>:"/\\|?*'


def safe_name(name: str) -> str:
    out = "".join("_" if c in BAD or ord(c) < 32 else c for c in name)
    return out.rstrip(" .") or "_"


def fit_dir(parent: Path, name: str, reserve: int) -> tuple[str, bool]:
    """A directory name for `name` under `parent` that leaves `reserve` chars."""
    safe = safe_name(name)
    budget = MAX_PATH - len(str(parent)) - 1 - reserve
    if len(safe) <= budget:
        return safe, False
    keep = max(8, budget - 9)
    return f"{safe[:keep]}_{hashlib.sha1(name.encode()).hexdigest()[:8]}", True


def render_one(job: dict) -> dict:
    """Render every community of one ontology. Runs in a worker process."""
    sys.path[:0] = [job["usecase"], job["pipeline"], job["tools"]]
    from extractbench.render_cc import (community_content, community_listing,
                                        communities_with_edges,
                                        render_community)

    t0 = time.time()
    res = {"dataset": job["dataset"], "ontology": job["ontology"],
           "dir": job["dir"], "n_communities": 0, "n_figures": 0,
           "renamed": job["renamed"], "error": ""}
    try:
        g, err = parse_graph(Path(job["file"]))
        if g is None:
            res["error"] = f"parse_error: {err}"
            return res
        comms, pred_map = communities_with_edges(g)
        if not comms:
            res["error"] = "no_communities"
            return res

        out = Path(job["dir"])
        out.mkdir(parents=True, exist_ok=True)
        width = max(3, len(str(len(comms))))
        listing, records = [], []
        for i, uris in enumerate(comms, 1):
            triples, singles = community_content(uris, pred_map)
            stem = out / f"Community{i:0{width}d}"
            title = (f"{job['dataset']} / {job['ontology']}\\n"
                     f"Community {i} of {len(comms)} — "
                     f"{len(triples)} relations, "
                     f"{len(set([s for s,_,_,_ in triples]) | set([o for _,_,o,_ in triples]) | set(singles))} entities")
            figure = None
            try:
                produced = render_community(
                    uris, pred_map, stem, title=title, fmt=job["format"],
                    dpi=job["dpi"], keep_source=job["keep_source"])
                if produced is not None:
                    figure = Path(produced).name
                    res["n_figures"] += 1
            except Exception as exc:                       # noqa: BLE001
                res["error"] = (res["error"] + f"; fig{i}: {exc}").strip("; ")

            listing.append(f"Community {i} : \n" + community_listing(triples, singles))
            records.append({
                "community": i,
                "figure": figure,
                "n_relations": len(triples),
                "triples": [{"s": s, "p": p, "o": o, "kind": k}
                            for s, p, o, k in triples],
                "standalone": singles,
                "entities": sorted(
                    {s for s, _, _, _ in triples} | {o for _, _, o, _ in triples}
                    | set(singles)),
            })
        res["n_communities"] = len(comms)

        (out / f"{safe_name(job['ontology'])}_greedy_mod_communities.txt").write_text(
            "\n".join(listing) + "\n", encoding="utf-8")
        (out / "communities.json").write_text(json.dumps({
            "dataset": job["dataset"], "ontology": job["ontology"],
            "source_file": job["file"], "n_communities": len(comms),
            "communities": records}, indent=1), encoding="utf-8")
        (out / "INDEX.txt").write_text(
            f"dataset      : {job['dataset']}\n"
            f"ontology     : {job['ontology']}\n"
            f"source file  : {job['file']}\n"
            f"communities  : {len(comms)}\n"
            f"figures      : {res['n_figures']} ({job['format']})\n"
            f"\nEach Community<n>.{job['format']} draws one community of the\n"
            f"step-1 conceptual-components extraction. Arrows are the\n"
            f"ontology's own relations, labelled with the property; boxes are\n"
            f"classes and properties. owl:Thing is drawn dashed and grey\n"
            f"because the pipeline injects it wherever a property declares no\n"
            f"domain or range -- it is scaffolding, not a modelling choice.\n"
            f"Full IRIs are in communities.json and in the _greedy_mod_\n"
            f"communities.txt listing.\n", encoding="utf-8")
    except Exception:                                       # noqa: BLE001
        res["error"] = traceback.format_exc(limit=3)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--datasets", nargs="*", default=["noOBO", "github"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--corpus-layout", choices=("source", "release"),
                    default="source")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--format", default="png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--keep-source", action="store_true",
                    help="keep the Graphviz .gv beside each figure")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave an ontology alone if its communities.json exists")
    args = ap.parse_args()

    for _attr in ("manifest", "data_root", "corpus", "out"):
        v = getattr(args, _attr)
        if v is not None:
            setattr(args, _attr, v.resolve())

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
    if args.limit:
        pop = pop[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    jobs, renames, missing = [], [], []
    for r in pop:
        ds, oid = r["dataset"], r["ontology_id"]
        onto_dir = (Path(r["dir"]) if r.get("dir")
                    else args.data_root / dirs[ds] / oid)
        mf = (Path(r["file"]) if r.get("file")
              else (main_ontology_file(onto_dir) if onto_dir.is_dir() else None))
        if mf is None or not Path(mf).is_file():
            missing.append((ds, oid))
            continue
        ds_dir = args.out / safe_name(ds)
        # Reserve room for the longest child: "Community9999.png" and the
        # upstream-named listing, whichever is longer.
        reserve = max(len("Community99999.png"),
                      len(safe_name(oid)) + len("_greedy_mod_communities.txt")) + 2
        name, was_renamed = fit_dir(ds_dir, oid, reserve)
        target = ds_dir / name
        if was_renamed:
            renames.append((ds, oid, name))
        if args.skip_existing and (target / "communities.json").is_file():
            continue
        jobs.append({"dataset": ds, "ontology": oid, "file": str(mf),
                     "dir": str(target), "renamed": was_renamed,
                     "format": args.format, "dpi": args.dpi,
                     "keep_source": args.keep_source,
                     "usecase": str(_USECASE), "pipeline": str(_REPO / "pipeline"),
                     "tools": str(_USECASE / "tools")})

    if missing:
        print(f"[skip] {len(missing)} ontologies have no readable file")
    print(f"rendering {len(jobs)} ontologies with {args.workers} workers "
          f"into {args.out}")
    if not jobs:
        return

    t0 = time.time()
    done = comms = figs = 0
    failures: list[dict] = []
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_one, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            comms += r["n_communities"]
            figs += r["n_figures"]
            if r["error"]:
                failures.append(r)
            if done % 25 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {comms} communities, "
                      f"{figs} figures, {time.time()-t0:.0f}s", flush=True)

    # A run report beside the tree, so the next reader does not have to
    # re-derive what was drawn and what was not.
    report = {
        "out": str(args.out), "format": args.format, "dpi": args.dpi,
        "ontologies_attempted": len(jobs),
        "ontologies_with_figures": sum(1 for r in results if r["n_figures"]),
        "communities": comms, "figures": figs,
        "seconds": round(time.time() - t0, 1),
        "no_file": [{"dataset": d, "ontology": o} for d, o in missing],
        "renamed_directories": [
            {"dataset": d, "ontology": o, "directory": n} for d, o, n in renames],
        "problems": [{k: r[k] for k in ("dataset", "ontology", "error")}
                     for r in failures],
    }
    stamp = "_".join(sorted({j["dataset"] for j in jobs}))
    (args.out / f"RENDER_REPORT_{safe_name(stamp)}.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    if renames:
        with (args.out / "NAME_MAP.txt").open("a", encoding="utf-8") as fh:
            for d, o, n in renames:
                fh.write(f"{d}\t{n}\t{o}\n")

    print(f"\n[done] {figs} figures for {comms} communities over "
          f"{report['ontologies_with_figures']} ontologies in "
          f"{report['seconds']:.0f}s")
    if failures:
        print(f"[{len(failures)} ontologies had a problem; see the report]")
        for r in failures[:8]:
            print(f"    {r['dataset']}/{r['ontology']}: "
                  f"{r['error'].splitlines()[-1][:110]}")
    print(f"[OK] {args.out / f'RENDER_REPORT_{safe_name(stamp)}.json'}")


if __name__ == "__main__":
    main()
