"""One parse of every pattern file; everything the four analyses need, cached.

The libraries are ~147 MB of RDF (135 MB of it DOS-DP), so they are parsed once
into a JSONL cache and every analysis reads that.  ``--resume`` (default) skips
files already in the cache whose sha256 still matches.

Wide entity IRI sets are written to ``<cache>/entities/<sha1(key)>.txt.gz``
rather than inlined, because the DOS-DP files declare tens of thousands of
entities each.

Entity-set semantics come from ``pipeline/odpqual/entities.py`` (the same code
that produced MANIFEST.csv), so ``n_entities_wide`` here and
``n_pattern_entities`` in ``pairs.csv`` are the same measurement.

Declaration counts (``n_classes`` / ``n_object_properties`` /
``n_datatype_properties``) deliberately mirror
``MultiSource/odp_stats.py::collect_defined_entities`` -- ``owl:Class`` union
``rdfs:Class`` subjects, ``owl:ObjectProperty`` subjects,
``owl:DatatypeProperty`` subjects, URIRefs only -- so that the numbers extend
rather than contradict Table "Pattern libraries" in the manuscript.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

# pipeline/ on the path so that odpqual -- the gate's own entity code, reused
# here so the entity counts are the same measurement -- imports unchanged.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef  # noqa: E402

from odpqual.entities import (  # noqa: E402
    collect_entities,
    parse_graph,
    strip_annotations,
)

from . import libraries as libmod  # noqa: E402
from . import vocab  # noqa: E402

logging.getLogger("rdflib").setLevel(logging.ERROR)

BUILTIN_NS = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)

#: at most this many labels / local names are tokenised per file.  Bounds the
#: influence of physically merged imports (a 17 MB DOS-DP file carries every
#: UBERON label).  Selection is deterministic: sorted by IRI, take the first N.
MAX_NAMING_TERMS = 2000

#: at most this many namespace-signature keys recorded per file, most frequent
#: first.  Observed maximum in the four libraries is well under this.
MAX_SIGNATURE_KEYS = 400

STRUCTURAL_SKIP_PREDICATES = frozenset({RDF.type})


# --------------------------------------------------------------------------
# union-find, for connectivity
# --------------------------------------------------------------------------

class _UF:
    __slots__ = ("p",)

    def __init__(self) -> None:
        self.p: dict[object, object] = {}

    def add(self, x) -> None:
        if x not in self.p:
            self.p[x] = x

    def find(self, x):
        p = self.p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _is_builtin(iri: str) -> bool:
    return iri.startswith(BUILTIN_NS)


# --------------------------------------------------------------------------
# per-file feature extraction
# --------------------------------------------------------------------------

def _catalogued_root(ns: str) -> bool:
    """True when ``ns`` *is* a catalogued vocabulary root, not a file under one.

    Distinguishes the two cases that otherwise look the same:

    * ODPA's ``.../cp/owl/agentrole.owl#`` is a namespace the pattern minted
      under the catalogued root ``.../cp/owl/`` -- the pattern owns it;
    * DOS-DP declares everything directly in ``http://purl.obolibrary.org/obo/``,
      which *is* the catalogued root -- the file owns nothing, it materialises
      somebody else's ontology.

    Without the distinction, DOS-DP would count OBO as "its own namespace" and
    the entire OBO/BFO alignment of that library would be invisible.
    """
    n = ns.rstrip("#/")
    return any(p.rstrip("#/") == n for p, _nm, _t in vocab.NAMESPACES)


def _own_namespace(declared: set[str], ontology_iris: list[str]
                   ) -> tuple[str, dict[str, int], str]:
    """``(own_ns, {namespace: n_declared_entities}, source)``.

    Preference order:
      1. the modal namespace among *declared* entities -- data-driven, works even
         when the file has no ``owl:Ontology`` header -- unless that namespace is
         a catalogued vocabulary root (see ``_catalogued_root``);
      2. the declared ontology IRI plus ``#``.
      3. nothing.
    """
    counts: dict[str, int] = {}
    for iri in declared:
        if _is_builtin(iri):
            continue
        ns = vocab.namespace_of(iri)
        counts[ns] = counts.get(ns, 0) + 1
    if counts:
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if not _catalogued_root(best):
            return best, counts, "modal_declared_namespace"
    for oi in ontology_iris:
        src = ("ontology_iri_no_own_declarations" if counts else "ontology_iri")
        return (oi if oi.endswith(("#", "/")) else oi + "#"), counts, src
    return "", counts, "none"


def _labels_for(g: Graph, iris: list[str]) -> list[str]:
    out: list[str] = []
    for iri in iris:
        for o in g.objects(URIRef(iri), RDFS.label):
            if isinstance(o, Literal):
                out.append(str(o))
    return out


def _connectivity(g: Graph, own_declared: list[str], all_declared: list[str],
                  own_core: list[str]) -> dict:
    """Component structure of the pattern's structural graph.

    Nodes are URIRefs and blank nodes; an edge joins the subject and object of
    every triple that is *structural*: object is not a literal, the predicate is
    not ``rdf:type``, the subject is not an ``owl:Ontology`` node, and neither
    endpoint sits in the RDF/RDFS/OWL/XSD namespaces.  Blank nodes are kept as
    nodes, so a class connected to another only through an anonymous
    ``owl:Restriction`` still counts as connected.  ``rdf:type`` is excluded
    because otherwise every class in the file would be joined through
    ``owl:Class``.
    """
    onto_nodes = set(g.subjects(RDF.type, OWL.Ontology))
    uf = _UF()
    n_edges = 0
    for s, p, o in g:
        if isinstance(o, Literal):
            continue
        if p in STRUCTURAL_SKIP_PREDICATES:
            continue
        if s in onto_nodes:
            continue
        if isinstance(s, URIRef) and _is_builtin(str(s)):
            continue
        if isinstance(o, URIRef) and _is_builtin(str(o)):
            continue
        uf.add(s)
        uf.add(o)
        uf.union(s, o)
        n_edges += 1

    def _components(iris: list[str]) -> tuple[int, int, int]:
        """(n_components, size_of_largest_group, n_isolated) over ``iris``."""
        groups: dict[object, int] = {}
        isolated = 0
        for iri in iris:
            node = URIRef(iri)
            if node not in uf.p:
                isolated += 1
                continue
            r = uf.find(node)
            groups[r] = groups.get(r, 0) + 1
        n_comp = len(groups) + isolated
        largest = max(groups.values()) if groups else (1 if isolated else 0)
        return n_comp, largest, isolated

    own_comp, own_largest, own_iso = _components(own_declared)
    all_comp, all_largest, all_iso = _components(all_declared)
    core_comp, core_largest, core_iso = _components(own_core)

    return {
        "n_struct_nodes": len(uf.p),
        "n_struct_edges": n_edges,
        "n_components_own": own_comp,
        "largest_component_own": own_largest,
        "n_isolated_own": own_iso,
        "connected_own": bool(own_comp <= 1),
        "n_components_declared": all_comp,
        "largest_component_declared": all_largest,
        "n_isolated_declared": all_iso,
        "connected_declared": bool(all_comp <= 1),
        # the modelling core: own classes + own object properties only.  The ODP
        # submission template declares its metadata fields (aim, also_known_as,
        # additional_information ...) as owl:DatatypeProperty, and those are by
        # construction isolated; counting them makes almost every pattern
        # "disconnected" for a reason that has nothing to do with its modelling.
        "n_own_core": len(own_core),
        "n_components_core": core_comp,
        "largest_component_core": core_largest,
        "n_isolated_core": core_iso,
        "connected_core": bool(core_comp <= 1),
    }


def extract_one(pf: libmod.PatternFile, ent_dir: Path) -> dict:
    rec: dict = {
        "key": pf.key,
        "library": pf.library,
        "rel_path": pf.rel_path,
        "stem": pf.stem,
        "suffix": pf.suffix,
        "n_bytes": pf.n_bytes,
        "sha256": pf.sha256,
        "group": pf.group,
        "role": pf.role,
        "content_kind": pf.content_kind,
        "parsed": False,
        "error": "",
    }

    g, err = parse_graph(Path(pf.abs_path))
    if g is None:
        rec["error"] = err
        return rec
    rec["parsed"] = True

    # ---- headers -----------------------------------------------------
    onto_iris = sorted(str(s) for s in g.subjects(RDF.type, OWL.Ontology)
                       if isinstance(s, URIRef))
    rec["ontology_iris"] = onto_iris
    rec["imports"] = sorted({str(o) for o in g.objects(None, OWL.imports)
                             if isinstance(o, URIRef)})

    # ---- declaration counts (odp_stats.py-compatible) ----------------
    classes = {str(s) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    classes |= {str(s) for s in g.subjects(RDF.type, RDFS.Class) if isinstance(s, URIRef)}
    obj_props = {str(s) for s in g.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
    dat_props = {str(s) for s in g.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)}
    ann_props = {str(s) for s in g.subjects(RDF.type, OWL.AnnotationProperty) if isinstance(s, URIRef)}
    individuals = {str(s) for s in g.subjects(RDF.type, OWL.NamedIndividual) if isinstance(s, URIRef)}

    rec["n_classes"] = len(classes)
    # Every rdf:type owl:Class / rdfs:Class triple, blank nodes included -- i.e.
    # named classes PLUS anonymous class expressions.  This is what the
    # manuscript's Table "Pattern libraries" counts in its Avg C column
    # (verified: it reproduces Manchester 18.76 and IS-ODP 55.50 exactly), so
    # both numbers are kept and the difference is reported rather than silently
    # resolved one way.
    rec["n_class_expressions"] = (
        sum(1 for _ in g.triples((None, RDF.type, OWL.Class)))
        + sum(1 for _ in g.triples((None, RDF.type, RDFS.Class)))
    )
    rec["n_anon_class_expressions"] = rec["n_class_expressions"] - len(classes)
    rec["n_object_properties"] = len(obj_props)
    rec["n_datatype_properties"] = len(dat_props)
    rec["n_annotation_properties"] = len(ann_props)
    rec["n_individuals"] = len(individuals)

    declared = classes | obj_props | dat_props
    declared_all = declared | ann_props | individuals

    # ---- axiom-shape proxies -----------------------------------------
    rec["n_triples"] = len(g)
    rec["n_subclassof"] = sum(1 for _ in g.triples((None, RDFS.subClassOf, None)))
    rec["n_subpropertyof"] = sum(1 for _ in g.triples((None, RDFS.subPropertyOf, None)))
    rec["n_equivalentclass"] = sum(1 for _ in g.triples((None, OWL.equivalentClass, None)))
    rec["n_disjointwith"] = sum(1 for _ in g.triples((None, OWL.disjointWith, None)))
    rec["n_restrictions"] = sum(1 for _ in g.triples((None, RDF.type, OWL.Restriction)))
    rec["n_somevaluesfrom"] = sum(1 for _ in g.triples((None, OWL.someValuesFrom, None)))
    rec["n_allvaluesfrom"] = sum(1 for _ in g.triples((None, OWL.allValuesFrom, None)))
    rec["n_cardinality"] = sum(
        sum(1 for _ in g.triples((None, p, None)))
        for p in (OWL.cardinality, OWL.minCardinality, OWL.maxCardinality,
                  OWL.qualifiedCardinality, OWL.minQualifiedCardinality,
                  OWL.maxQualifiedCardinality)
    )
    rec["n_inverseof"] = sum(1 for _ in g.triples((None, OWL.inverseOf, None)))
    rec["n_domain"] = sum(1 for _ in g.triples((None, RDFS.domain, None)))
    rec["n_range"] = sum(1 for _ in g.triples((None, RDFS.range, None)))
    rec["n_intersection"] = sum(1 for _ in g.triples((None, OWL.intersectionOf, None)))
    rec["n_union"] = sum(1 for _ in g.triples((None, OWL.unionOf, None)))
    rec["n_propertychain"] = sum(1 for _ in g.triples((None, OWL.propertyChainAxiom, None)))
    rec["n_labels"] = sum(1 for _ in g.triples((None, RDFS.label, None)))
    rec["n_comments"] = sum(1 for _ in g.triples((None, RDFS.comment, None)))

    # ---- vocabulary usage over every distinct URIRef -----------------
    seen: set[str] = set()
    for s, p, o in g:
        if isinstance(s, URIRef):
            seen.add(str(s))
        if isinstance(p, URIRef):
            seen.add(str(p))
        if isinstance(o, URIRef):
            seen.add(str(o))
    rec["n_distinct_iris"] = len(seen)

    own_ns, ns_declared_counts, own_ns_source = _own_namespace(declared, onto_iris)
    rec["own_ns"] = own_ns
    rec["own_ns_source"] = own_ns_source
    rec["n_declared_namespaces"] = len(ns_declared_counts)
    rec["modal_declared_namespace"] = (
        max(ns_declared_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if ns_declared_counts else "")

    # Stored as raw prefix signatures, NOT as resolved vocabulary names, so the
    # catalogue in vocab.py can be revised without reparsing the libraries.
    sig: dict[str, int] = {}
    n_own_ns_iris = 0
    for iri in seen:
        if own_ns and iri.startswith(own_ns):
            n_own_ns_iris += 1
            continue
        k = vocab.signature_of(iri)
        sig[k] = sig.get(k, 0) + 1
    rec["n_own_ns_iris"] = n_own_ns_iris
    rec["ns_signature"] = dict(sorted(sig.items(), key=lambda kv: (-kv[1], kv[0]))
                               [:MAX_SIGNATURE_KEYS])
    rec["n_signature_keys_total"] = len(sig)
    rec["import_signature"] = sorted({vocab.signature_of(i) for i in rec["imports"]})

    # ---- entity sets (pipeline-compatible) ---------------------------
    stripped, n_removed = strip_annotations(g)
    rec["n_triples_logical"] = len(stripped)
    rec["n_triples_annotation"] = n_removed
    ents_wide = collect_entities(stripped, wide=True)
    ents_narrow = collect_entities(stripped, wide=False)
    rec["n_entities_wide"] = len(ents_wide)
    rec["n_entities_narrow"] = len(ents_narrow)

    own_declared = sorted(i for i in declared if own_ns and i.startswith(own_ns))
    own_core = sorted(i for i in (classes | obj_props) if own_ns and i.startswith(own_ns))
    rec["n_own_declared"] = len(own_declared)
    rec["own_declared"] = own_declared if len(own_declared) <= 5000 else own_declared[:5000]

    # ---- naming evidence for the domain classifier --------------------
    if own_declared:
        naming_src = "own_namespace_entities"
        naming_iris = own_declared[:MAX_NAMING_TERMS]
    else:
        naming_src = "all_declared_entities"
        naming_iris = sorted(declared_all)[:MAX_NAMING_TERMS]
    rec["naming_evidence_source"] = naming_src
    rec["n_naming_iris"] = len(naming_iris)
    rec["naming_localnames"] = [vocab.local_name(i) for i in naming_iris]
    rec["naming_labels"] = _labels_for(g, naming_iris)[:MAX_NAMING_TERMS]

    # ---- connectivity -------------------------------------------------
    rec.update(_connectivity(g, own_declared, sorted(declared), own_core))

    # ---- entity set on disk -------------------------------------------
    ent_path = ent_dir / (hashlib.sha1(pf.key.encode()).hexdigest() + ".txt.gz")
    with gzip.open(ent_path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(ents_wide))
    rec["entities_file"] = ent_path.name

    del g, stripped
    return rec


# --------------------------------------------------------------------------
# cache driver
# --------------------------------------------------------------------------

def load_records(cache_dir: Path) -> dict[str, dict]:
    p = cache_dir / "patterns.jsonl"
    out: dict[str, dict] = {}
    if not p.is_file():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            out[d["key"]] = d
    return out


def entity_set(cache_dir: Path, rec: dict) -> set[str]:
    """Read back the wide entity IRI set of one record."""
    fn = rec.get("entities_file")
    if not fn:
        return set()
    p = cache_dir / "entities" / fn
    if not p.is_file():
        return set()
    with gzip.open(p, "rt", encoding="utf-8") as fh:
        return {ln for ln in fh.read().split("\n") if ln}


def build(data_root: Path, cache_dir: Path, resume: bool = True,
          libs: list[str] | None = None, limit: int = 0,
          verbose: bool = True) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ent_dir = cache_dir / "entities"
    ent_dir.mkdir(exist_ok=True)

    files = libmod.discover(data_root, libs)
    if limit:
        files = files[:limit]
    have = load_records(cache_dir) if resume else {}

    jsonl = cache_dir / "patterns.jsonl"
    todo = [f for f in files
            if not (resume and f.key in have and have[f.key].get("sha256") == f.sha256)]
    if verbose:
        print(f"[extract] {len(files)} files, {len(files) - len(todo)} cached, "
              f"{len(todo)} to parse", flush=True)

    mode = "a" if (resume and jsonl.is_file()) else "w"
    t0 = time.time()
    with jsonl.open(mode, encoding="utf-8") as fh:
        for i, pf in enumerate(todo, 1):
            try:
                rec = extract_one(pf, ent_dir)
            except Exception as exc:  # noqa: BLE001
                rec = {
                    "key": pf.key, "library": pf.library, "rel_path": pf.rel_path,
                    "stem": pf.stem, "suffix": pf.suffix, "n_bytes": pf.n_bytes,
                    "sha256": pf.sha256, "group": pf.group, "role": pf.role,
                    "parsed": False,
                    "error": f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}",
                }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            have[pf.key] = rec
            if verbose and (i % 20 == 0 or i == len(todo)):
                print(f"[extract] {i}/{len(todo)}  {time.time() - t0:6.1f}s  "
                      f"{pf.key}", flush=True)

    # Keep only records for files that still exist, and refresh the fields that
    # come from the filesystem rather than from parsing (``role`` in particular
    # depends on the whole-library scan, so a cached record written during a
    # single-library run would otherwise carry a stale role).
    by_key = {f.key: f for f in files}
    out: dict[str, dict] = {}
    for k, v in have.items():
        pf = by_key.get(k)
        if pf is None:
            continue
        v["role"] = pf.role
        v["group"] = pf.group
        v["n_bytes"] = pf.n_bytes
        v["content_kind"] = pf.content_kind
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--cache", type=Path, default=None,
                    help="cache directory (default <out>/cache)")
    ap.add_argument("--out", type=Path, default=Path("pipeline/out/patternstats"))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--libs", nargs="*", default=None, choices=list(libmod.LIB_ORDER))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cache = args.cache or (args.out / "cache")
    recs = build(args.data_root, cache, resume=not args.no_resume,
                 libs=args.libs, limit=args.limit)
    ok = sum(1 for r in recs.values() if r.get("parsed"))
    print(f"[extract] done: {len(recs)} records, {ok} parsed, {len(recs) - ok} failed")


if __name__ == "__main__":
    main()
