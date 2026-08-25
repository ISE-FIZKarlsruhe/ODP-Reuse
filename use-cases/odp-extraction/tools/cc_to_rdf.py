#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cc_to_rdf.py -- write each conceptual-components community as an RDF file.

Upstream ships this as
``rebuild_intensional_graphs_from_greedy_modularity_communities``, called from
``rdf_greedy_modularity_communities_with_rdf_from_folder`` to fill an
``intensional_communities/`` directory with one Turtle file per community. It
was not vendored, because it produces no quantity the benchmark scores.

WHICH GRAPH UPSTREAM ACTUALLY SERIALISES
----------------------------------------
Worth being precise about, because the name is misleading. Upstream passes the
community together with ``owlthing_nodisjoint_class_distinctproperty`` -- the
*reified* intensional graph, the one that was clustered -- and keeps the triples
running between two members of the community. Every triple in the result is
therefore of the form

    <s>  framester:edge  <http://edge.example/s--p--o>
    <http://edge.example/s--p--o>  framester:edge  <o>

so the file records the machinery the clustering ran on, not the modelling the
ontology states. It is faithful and it is not what a reader wanting "the
component as RDF" expects.

Two files are therefore written per community, and neither invents a triple:

``Community001.ttl``
    The component as an ontology fragment: the triples ``s p o`` that the
    reified pairs stand for. This is the de-reified form, and the default.

``Community001.intensional.ttl`` (with ``--intensional``)
    Upstream's output exactly -- the reified subgraph, ``framester:edge``
    predicates and ``edge.example`` nodes included.

Both are built from the ``communities.json`` that ``render_cc_figures.py``
writes, so step 1 is not re-run and the RDF describes precisely the community
the figure beside it draws.

Usage
-----
    python tools/cc_to_rdf.py --root results/cc-figures
    python tools/cc_to_rdf.py --root results/cc-figures --intensional
    python tools/cc_to_rdf.py --root results/cc-figures --format nt --ext nt
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EDGE_PREFIX = "http://edge.example/"
FRAMESTER_EDGE = "https://w3id.org/framester/schema/edge"

# Bound so the Turtle reads as Turtle rather than as a wall of angle brackets.
# rdflib auto-binds whatever else appears as ns1:, ns2: ...
COMMON = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "prov": "http://www.w3.org/ns/prov#",
    "dul": "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#",
    "cpannotationschema":
        "http://www.ontologydesignpatterns.org/schemas/cpannotationschema.owl#",
    "obo": "http://purl.obolibrary.org/obo/",
    "framester": "https://w3id.org/framester/schema/",
    "edge": EDGE_PREFIX,
}


def _local(iri: str) -> str:
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return iri[cut + 1:] if cut >= 0 else iri


def convert_one(job: dict) -> dict:
    """Write the RDF files for one ontology's communities. Worker process."""
    from rdflib import Graph, URIRef

    res = {"dataset": job["dataset"], "ontology": job["ontology"],
           "n_communities": 0, "n_files": 0, "n_empty": 0, "repaired": 0,
           "error": ""}
    try:
        meta = json.loads(Path(job["json"]).read_text(encoding="utf-8"))
        out = Path(job["json"]).parent
        res["n_communities"] = len(meta["communities"])
        width = max(3, len(str(len(meta["communities"]))))
        index = []

        for c in meta["communities"]:
            n = c["community"]
            stem = out / f"Community{n:0{width}d}"

            g = Graph()
            for pfx, ns in COMMON.items():
                g.bind(pfx, ns, override=False)
            for t in c["triples"]:
                g.add((URIRef(t["s"]), URIRef(t["p"]), URIRef(t["o"])))

            # An entity that occurs in no relation contributes no triple. It is
            # part of the community and would otherwise vanish from the file, so
            # it is named in the header rather than given an invented triple.
            #
            # Recomputed rather than trusted: a community holds an entity node
            # as well as the edge nodes it sits between, so early runs of
            # render_cc_figures.py recorded endpoints under "standalone" too and
            # the header then claimed an entity took part in nothing while its
            # own triple sat below. Anything that is an endpoint is dropped
            # here, and the JSON is corrected in place so the two agree.
            endpoints = ({t["s"] for t in c["triples"]}
                         | {t["o"] for t in c["triples"]})
            orphans = [e for e in c["standalone"] if e not in endpoints]
            if len(orphans) != len(c["standalone"]):
                c["standalone"] = orphans
                res["repaired"] += 1
            header = [
                f"# Conceptual component {n} of {len(meta['communities'])}",
                f"# dataset      : {meta['dataset']}",
                f"# ontology     : {meta['ontology']}",
                f"# source       : {meta['source_file']}",
                f"# relations    : {len(c['triples'])}",
                f"# entities     : {len(c['entities'])}",
            ]
            if orphans:
                header.append(f"# {len(orphans)} entit"
                              f"{'ies' if len(orphans) > 1 else 'y'} of this "
                              f"component occur in no relation and so appear "
                              f"in no triple below:")
                header += [f"#     <{o}>" for o in orphans]
            header.append("#")
            header.append("# De-reified form: the triples the clustered graph's"
                          " edge nodes stand for.")

            body = g.serialize(format=job["format"])
            if not c["triples"]:
                res["n_empty"] += 1
            stem.with_suffix("." + job["ext"]).write_text(
                "\n".join(header) + "\n\n" + body, encoding="utf-8")
            res["n_files"] += 1

            if job["intensional"]:
                gi = Graph()
                for pfx, ns in COMMON.items():
                    gi.bind(pfx, ns, override=False)
                edge = URIRef(FRAMESTER_EDGE)
                for t in c["triples"]:
                    node = URIRef(EDGE_PREFIX + _local(t["s"]) + "--"
                                  + _local(t["p"]) + "--" + _local(t["o"]))
                    gi.add((URIRef(t["s"]), edge, node))
                    gi.add((node, edge, URIRef(t["o"])))
                ihead = header[:6] + [
                    "#",
                    "# Reified (intensional) form, as upstream's",
                    "# rebuild_intensional_graphs_from_greedy_modularity_communities",
                    "# writes it: every triple s p o becomes an edge.example node",
                    "# between s and o, joined by framester:edge. This is the graph",
                    "# the community detection ran on.",
                ]
                (out / f"{stem.name}.intensional.{job['ext']}").write_text(
                    "\n".join(ihead) + "\n\n" + gi.serialize(format=job["format"]),
                    encoding="utf-8")
                res["n_files"] += 1

            index.append({"community": n,
                          "rdf": stem.with_suffix("." + job["ext"]).name,
                          "figure": c.get("figure"),
                          "n_relations": len(c["triples"]),
                          "n_entities": len(c["entities"])})

        if res["repaired"]:
            Path(job["json"]).write_text(json.dumps(meta, indent=1),
                                         encoding="utf-8")
        (out / "RDF_INDEX.json").write_text(json.dumps({
            "dataset": meta["dataset"], "ontology": meta["ontology"],
            "format": job["format"], "intensional": job["intensional"],
            "communities": index}, indent=1), encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        res["error"] = traceback.format_exc(limit=3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="the tree render_cc_figures.py wrote")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="limit to these dataset directories")
    ap.add_argument("--format", default="turtle",
                    help="rdflib serialisation (turtle, nt, xml, json-ld)")
    ap.add_argument("--ext", default="ttl")
    ap.add_argument("--intensional", action="store_true",
                    help="also write upstream's reified form per community")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave an ontology alone if its RDF_INDEX.json exists")
    args = ap.parse_args()
    args.root = args.root.resolve()

    jobs = []
    for js in sorted(args.root.glob("*/*/communities.json")):
        ds = js.parent.parent.name
        if args.datasets and ds not in args.datasets:
            continue
        if args.skip_existing and (js.parent / "RDF_INDEX.json").is_file():
            continue
        jobs.append({"json": str(js), "dataset": ds, "ontology": js.parent.name,
                     "format": args.format, "ext": args.ext,
                     "intensional": args.intensional})

    if not jobs:
        print(f"nothing to do under {args.root}")
        return
    print(f"{len(jobs)} ontologies, {args.workers} workers, "
          f"format={args.format}"
          f"{' + intensional' if args.intensional else ''}")

    t0 = time.time()
    done = files = comms = empty = repaired = 0
    failures = []
    by_ds: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(convert_one, j) for j in jobs]
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            files += r["n_files"]
            comms += r["n_communities"]
            empty += r["n_empty"]
            repaired += r["repaired"]
            by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + r["n_files"]
            if r["error"]:
                failures.append(r)
            if done % 50 == 0 or done == len(jobs):
                print(f"  [{done}/{len(jobs)}] {files} files, "
                      f"{time.time()-t0:.0f}s", flush=True)

    report = {"root": str(args.root), "format": args.format,
              "ext": args.ext, "intensional": args.intensional,
              "ontologies": len(jobs), "communities": comms,
              "files": files, "files_by_dataset": by_ds,
              "components_with_no_relation": empty,
              "components_with_standalone_list_corrected": repaired,
              "seconds": round(time.time() - t0, 1),
              "problems": [{k: r[k] for k in ("dataset", "ontology", "error")}
                           for r in failures]}
    (args.root / "RDF_REPORT.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n[done] {files} RDF files for {comms} components over "
          f"{len(jobs)} ontologies in {report['seconds']:.0f}s")
    for ds, n in sorted(by_ds.items()):
        print(f"    {ds:<10} {n} files")
    if empty:
        print(f"    {empty} components hold no relation (entities only); "
              f"their file lists them in the header")
    if repaired:
        print(f"    {repaired} components had endpoints listed as standalone; "
              f"corrected in communities.json")
    if failures:
        print(f"[{len(failures)} ontologies failed; see RDF_REPORT.json]")
        for r in failures[:5]:
            print(f"    {r['dataset']}/{r['ontology']}: "
                  f"{r['error'].splitlines()[-1][:110]}")
    print(f"[OK] {args.root / 'RDF_REPORT.json'}")


if __name__ == "__main__":
    main()
