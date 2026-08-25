#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_asprino_full.py -- the COMPLETE conceptual-components pipeline, steps 1-7.

``run_method_eval.py`` runs step 1 only, because step 1 is what produces a
grouping of one ontology's entities. This runner adds steps 2-7 and answers the
question step 1 alone cannot: does clustering the communities across the corpus
change the per-ontology grouping, and does it change the score?

It can. Step 6 assigns every community of every ontology to one cross-corpus
cluster, and two communities of the SAME ontology can land in the same cluster.
Reading a conceptual component's instance in ontology O as the union of O's
communities assigned to that cluster therefore merges communities that step 1
kept apart. That merged reading is what this runner scores, beside the
unmerged step-1 reading, on the same population and metric.

THE PIPELINE
------------
  1  intensional graph -> greedy modularity communities   (asprino.py)
  2  communities -> virtual documents                     (virtual_documents.py)
  3  Framester + UKB server                               EXTERNAL, see below
  4  virtual documents -> synsets + frames                (synset_frames.py)
  5  drop be/have, keep max-scoring synset, add frames
     and their top-level ancestors                        (synset_frames.py)
  6  TF-IDF over the synset/frame column, K-Means         (synset_frames.py)
  7  per-cluster frame analysis                           (reported, not scored)

STEP 3 IS A PREREQUISITE AND IS NOT AUTOMATED
---------------------------------------------
The disambiguator is a 591 MB jar published on Google Drive plus a UKB binary
the user must build for their own platform. To stand it up:

    # 1. build UKB (needs g++ and Boost; on Windows use WSL)
    git clone --depth 1 https://github.com/asoroa/ukb.git
    cd ukb/src && ./configure && make            # produces ukb_wsd

    # 2. fetch the server jar (591 MB)
    #    https://drive.google.com/file/d/1ZCjtxiL80vALaQXWscIwYOJa0LMJwGtN

    # 3. put the built binary where the shipped Mach-O one was, and run the jar
    #    FROM the disambiguation_framedetection directory, with stdin held open
    #    (its main() stops on EOF):
    cd conceptual-components-extraction/project_root/disambiguation_framedetection
    cp <ukb>/src/ukb_wsd ukb/bin/ukb_wsd && chmod +x ukb/bin/ukb_wsd
    tail -f /dev/null | java -jar lgu.commons.server.jar

    # 4. check it
    curl -sG http://localhost:8080/cd --data-urlencode "text=person plays music" \
         --data-urlencode senseInventory=fn17

Pass ``--server`` if it is not on http://localhost:8080.

Usage
-----
    python tools/run_asprino_full.py \
        --manifest ../../pipeline/out/MANIFEST.csv \
        --data-root <MultiSource> --out results/asprino-full \
        --datasets github noOBO --subset reuse \
        --top-level ../../<clone>/project_root/data/topLevelAnchestors-prefix.csv
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
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_USECASE / "tools"))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from extractbench.asprino import communities as step1_communities  # noqa: E402
from extractbench.evaluate import match_and_score  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from run_extraction_eval import ground_truth, load_population  # noqa: E402
from third_party.conceptual_components import synset_frames as sf  # noqa: E402
from third_party.conceptual_components.virtual_documents import (  # noqa: E402
    community_documents,
)

RELEASE_DIRS = {"github": "github", "noOBO": "noOBO", "OBO": "OBO"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--corpus", type=Path, default=None,
                    help="a directory of ontologies instead of the gated "
                         "ODP-Reuse population. Where an ontology has no "
                         "ground truth its groups are written to "
                         "predictions.jsonl and no score is invented.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--datasets", nargs="*", default=["noOBO", "github"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--corpus-layout", choices=("source", "release"),
                    default="source")
    ap.add_argument("--corpus-name", default="plain")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--top-level", type=Path, required=True,
                    help="project_root/data/topLevelAnchestors-prefix.csv")
    ap.add_argument("--k", type=int, default=0,
                    help="number of clusters for step 6; 0 = elbow (step 6b)")
    ap.add_argument("--also-comments", action="store_true",
                    help="upstream step 2's 'True' for rdfs:comment; the "
                         "README's own invocation passes False")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="reuse communities.json / documents.json / synsets.json")
    ap.add_argument("--stage", choices=("all", "documents", "score"),
                    default="all",
                    help="'all' runs every step in one process, which needs the "
                         "step-3 server to be reachable from here. When the "
                         "server runs somewhere this process cannot reach it "
                         "(the jar hard-codes BASE_URI to http://localhost:8080/, "
                         "so a WSL-hosted server is invisible from Windows), run "
                         "'documents' here, then tools/framester_client.py "
                         "beside the server, then 'score' here.")
    args = ap.parse_args()

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
    dirs = RELEASE_DIRS if args.corpus_layout == "release" else DATASETS
    top_level = sf.load_top_level_ancestors(args.top_level)
    print(f"[step 5] {len(top_level)} frame -> top-level-ancestor entries")

    if args.corpus:
        sys.path.insert(0, str(_USECASE / "tools"))
        from run_method_eval import walk_corpus
        pop = walk_corpus(args.corpus)
        print(f"corpus {args.corpus}: {len(pop)} ontologies", flush=True)
    else:
        if not (args.manifest and args.data_root):
            ap.error("give either --corpus DIR, or --manifest and --data-root")
        pop = load_population(args.manifest, args.datasets, args.subset)
        print(f"population ({args.subset}): {len(pop)} ontologies", flush=True)
    if args.limit:
        pop = pop[:args.limit]

    # ---------------- steps 1 + 2 --------------------------------------
    comm_path = args.out / "communities.json"
    doc_path = args.out / "documents.json"
    skipped: list[tuple[str, str, str, str]] = []
    records: list[dict] = []          # one per community, in a global order

    if args.resume and doc_path.is_file():
        records = json.loads(doc_path.read_text(encoding="utf-8"))
        print(f"[resume] {len(records)} community documents loaded")
    else:
        t0 = time.time()
        for i, r in enumerate(pop, 1):
            ds, oid = r["dataset"], r["ontology_id"]
            onto_dir = (Path(r["dir"]) if r.get("dir")
                        else args.data_root / dirs[ds] / oid)
            mf = (Path(r["file"]) if r.get("file")
                  else (main_ontology_file(onto_dir) if onto_dir.is_dir() else None))
            if mf is None:
                skipped.append((ds, oid, "no_ontology_file", ""))
                continue
            g, err = parse_graph(mf)
            if g is None:
                skipped.append((ds, oid, "parse_error", err))
                continue
            gs, _ = strip_annotations(g)
            ents = collect_entities(gs, wide=True)
            if not ents:
                skipped.append((ds, oid, "no_entities", ""))
                continue
            gt = ground_truth(onto_dir, set(ents), True,
                              complete_only=(args.subset == "complete"),
                              host_graph=gs, dataset=ds)
            if not gt and not args.corpus:
                skipped.append((ds, oid, "no_ground_truth", args.subset))
                continue
            comms = step1_communities(g)              # STEP 1
            comms = [sorted(c) for c in comms if c]
            if not comms:
                skipped.append((ds, oid, "no_communities", ""))
                continue
            docs = community_documents(g, comms,      # STEP 2
                                       also_comments=args.also_comments)
            for j, (text, annotated) in enumerate(docs):
                records.append({
                    "dataset": ds, "ontology": oid, "community": j,
                    "entities": comms[j], "text": text,
                    "annotated_text": annotated,
                    "n_entities": len(ents), "n_gt": len(gt),
                })
            if i % 25 == 0:
                print(f"  [1-2] {i}/{len(pop)} ontologies, "
                      f"{len(records)} communities, {time.time()-t0:.0f}s",
                      flush=True)
        comm_path.write_text(json.dumps(
            {f'{r["dataset"]}/{r["ontology"]}': None for r in records}),
            encoding="utf-8")
        doc_path.write_text(json.dumps(records), encoding="utf-8")
    print(f"[steps 1-2] {len(records)} communities over "
          f"{len({(r['dataset'], r['ontology']) for r in records})} ontologies",
          flush=True)
    if args.stage == "documents":
        print(f"[OK] {doc_path}")
        print("Next: run tools/framester_client.py beside the step-3 server:")
        print(f"  python3 tools/framester_client.py --in {doc_path} "
              f"--out {args.out / 'synsets.json'} --workers {args.workers}")
        print("then re-run this with --stage score --resume.")
        return

    # ---------------- steps 3 + 4 --------------------------------------
    syn_path = args.out / "synsets.json"
    if args.stage == "score" and not syn_path.is_file():
        sys.exit(f"--stage score needs {syn_path}; run framester_client.py first")
    if (args.resume or args.stage == "score") and syn_path.is_file():
        syn = json.loads(syn_path.read_text(encoding="utf-8"))
        print(f"[resume] {len(syn)} disambiguations loaded")
    else:
        t0 = time.time()
        print(f"[step 4] disambiguating {len(records)} documents against "
              f"{args.server} with {args.workers} workers", flush=True)
        syn = sf.disambiguate_many([r["text"] for r in records], args.server,
                                   workers=args.workers,
                                   progress=lambda m: print(m, flush=True))
        syn_path.write_text(json.dumps(syn), encoding="utf-8")
        print(f"[step 4] done in {time.time()-t0:.0f}s", flush=True)
    n_err = sum(1 for s in syn if "error" in s)
    if n_err:
        print(f"[step 4] {n_err} documents fell back to their raw text")

    # ---------------- step 5 -------------------------------------------
    filtered = [sf.filter_and_enrich(s.get("annotated synset frame", ""),
                                     records[i]["text"], top_level)
                for i, s in enumerate(syn)]
    n_fallback = sum(1 for i, f in enumerate(filtered)
                     if f == records[i]["text"])
    print(f"[step 5] {len(filtered)} rows, {n_fallback} without any "
          f"synset/frame (fell back to raw text)")

    # ---------------- step 6 -------------------------------------------
    k = args.k
    if k <= 0:
        k = sf.elbow_n_clusters(filtered, k_min=2,
                                k_max=min(200, max(3, len(filtered) // 5)))
        print(f"[step 6b] elbow selected k = {k}")
    labels, _ = sf.cluster_documents(filtered, k)
    print(f"[step 6] {k} clusters over {len(labels)} communities")
    for i, lab in enumerate(labels):
        records[i]["cluster"] = int(lab)
        records[i]["filtered_synset_frame"] = filtered[i]
    (args.out / "clustered_communities.json").write_text(
        json.dumps(records), encoding="utf-8")

    # ---------------- step 7 + scoring ---------------------------------
    # Per ontology: the FULL pipeline's groups are the unions of that
    # ontology's communities that share a cluster; step 1's groups are the
    # communities themselves.
    by_onto: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_onto.setdefault((r["dataset"], r["ontology"]), []).append(r)

    per_path = args.out / "asprino_full_per_ontology.jsonl"
    pred_path = args.out / "predictions.jsonl"
    pfh = pred_path.open("w", encoding="utf-8")
    rows: list[dict] = []
    with per_path.open("w", encoding="utf-8") as fh:
        for (ds, oid), rs in by_onto.items():
            src = next((x for x in pop if x["dataset"] == ds
                        and x["ontology_id"] == oid), {})
            onto_dir = (Path(src["dir"]) if src.get("dir")
                        else args.data_root / dirs[ds] / oid)
            mf = (Path(src["file"]) if src.get("file")
                  else main_ontology_file(onto_dir))
            g, _err = parse_graph(mf)
            gs, _ = strip_annotations(g)
            ents = set(collect_entities(gs, wide=True))
            gt = ground_truth(onto_dir, ents, True,
                              complete_only=(args.subset == "complete"),
                              host_graph=gs, dataset=ds)
            step1 = {i: set(r["entities"]) & ents for i, r in enumerate(rs)}
            step1 = {i: v for i, v in step1.items() if v}
            merged: dict[int, set[str]] = {}
            for r in rs:
                merged.setdefault(r["cluster"], set()).update(
                    set(r["entities"]) & ents)
            merged = {i: v for i, v in merged.items() if v}
            for method, clusters in (("asprino_cc", step1),
                                     ("asprino_cc_full", merged)):
                if pfh is not None:
                    pfh.write(json.dumps({
                        "corpus": args.corpus_name, "dataset": ds,
                        "ontology": oid, "method": method,
                        "n_entities": len(ents), "scored": bool(gt),
                        "n_clusters": len(clusters),
                        "clusters": [sorted(c) for c in clusters.values()],
                    }) + chr(10))
                if not gt:
                    continue
                s = match_and_score(clusters, gt)
                rec = {"corpus": args.corpus_name, "dataset": ds,
                       "ontology": oid, "method": method,
                       "n_entities": len(ents), "n_gt": len(gt),
                       "precision": round(s["precision"], 4),
                       "recall": round(s["recall"], 4),
                       "f1": round(s["f1"], 4),
                       "n_predicted": s["n_predicted"],
                       "n_matched": s["n_matched"]}
                fh.write(json.dumps(rec) + "\n")
                rows.append(rec)

    pfh.close()
    print(f"[OK] {pred_path}")

    import numpy as np
    if not rows:
        print("No ontology had ground truth; groups are in predictions.jsonl.")
        return
    agg = []
    for ds in sorted({r["dataset"] for r in rows}):
        for m in ("asprino_cc", "asprino_cc_full"):
            sub = [r for r in rows if r["dataset"] == ds and r["method"] == m]
            if not sub:
                continue
            agg.append({
                "corpus": args.corpus_name, "dataset": ds, "method": m,
                "n_ontologies": len(sub), "k_clusters": k,
                "precision": round(float(np.mean([r["precision"] for r in sub])), 4),
                "recall": round(float(np.mean([r["recall"] for r in sub])), 4),
                "f1": round(float(np.mean([r["f1"] for r in sub])), 4),
                "mean_n_predicted": round(
                    float(np.mean([r["n_predicted"] for r in sub])), 2),
                "mean_n_gt": round(float(np.mean([r["n_gt"] for r in sub])), 2),
            })
    with (args.out / "asprino_full_results.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0]))
        w.writeheader()
        w.writerows(agg)
    if skipped:
        with (args.out / "asprino_full_skipped.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w2 = csv.writer(fh)
            w2.writerow(["dataset", "ontology", "reason", "detail"])
            w2.writerows(skipped)
    print(f"\n[OK] {args.out / 'asprino_full_results.csv'}")
    for r in agg:
        print(f"  {r['dataset']:8s} {r['method']:18s} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} F1={r['f1']:.3f}  "
              f"clusters={r['mean_n_predicted']:.2f} (gt {r['mean_n_gt']:.2f}) "
              f"n={r['n_ontologies']}")


if __name__ == "__main__":
    main()
