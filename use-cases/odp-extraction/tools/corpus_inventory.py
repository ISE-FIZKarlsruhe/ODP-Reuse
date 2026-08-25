#!/usr/bin/env python3
"""corpus_inventory.py -- what version of each ontology is in a corpus directory.

Reads every RDF file and extracts the version metadata an ontology declares
about itself: owl:versionIRI, owl:versionInfo, dcterms:modified / issued /
created, and the declared ontology IRI. That IRI is what identifies the upstream
artefact, so it is what a currency check has to start from.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "pipeline"))
from rdflib import OWL, RDF, RDFS, URIRef  # noqa: E402
from odpqual.entities import parse_graph  # noqa: E402

DCT = "http://purl.org/dc/terms/"
DC = "http://purl.org/dc/elements/1.1/"

FIELDS = {
    "versionIRI": [OWL.versionIRI],
    "versionInfo": [OWL.versionInfo],
    "modified": [URIRef(DCT + "modified")],
    "issued": [URIRef(DCT + "issued")],
    "created": [URIRef(DCT + "created"), URIRef(DC + "date")],
    "title": [URIRef(DCT + "title"), URIRef(DC + "title"), RDFS.label],
    "seeAlso": [RDFS.seeAlso],
}


def main() -> None:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = []
    files = [f for f in sorted(root.rglob("*"))
             if f.is_file() and f.suffix.lower() in
             (".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt")]
    for i, f in enumerate(files, 1):
        g, err = parse_graph(f)
        rec = {"file": f.relative_to(root).as_posix(),
               "bytes": f.stat().st_size, "parsed": "ok" if g else "fail",
               "error": err[:120] if not g else ""}
        if g is not None:
            onts = [s for s in g.subjects(RDF.type, OWL.Ontology)
                    if isinstance(s, URIRef)]
            rec["ontology_iri"] = str(onts[0]) if onts else ""
            rec["n_ontology_iris"] = len(onts)
            rec["n_triples"] = len(g)
            subj = onts[0] if onts else None
            for name, preds in FIELDS.items():
                vals = []
                for p in preds:
                    if subj is not None:
                        vals += [str(o) for o in g.objects(subj, p)]
                    if not vals:
                        vals += [str(o) for o in g.objects(None, p)][:1]
                rec[name] = " | ".join(dict.fromkeys(vals))[:200]
            rec["imports"] = " | ".join(sorted(
                {str(o) for o in g.objects(None, OWL.imports)}))[:300]
        rows.append(rec)
        if i % 20 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    fields = ["file", "bytes", "parsed", "error", "ontology_iri",
              "n_ontology_iris", "n_triples", "title", "versionIRI",
              "versionInfo", "modified", "issued", "created", "seeAlso",
              "imports"]
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {out}")
    print(json.dumps({"n_files": len(rows),
                      "n_parsed": sum(1 for r in rows if r["parsed"] == "ok"),
                      "n_with_versionIRI": sum(1 for r in rows if r.get("versionIRI")),
                      "n_with_versionInfo": sum(1 for r in rows if r.get("versionInfo"))},
                     indent=1))


if __name__ == "__main__":
    main()
