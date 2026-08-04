"""
RQ1-RQ4 statistics over the quality-gated corpus.

Every ontology considered here has passed the gate in ``odpqual/gate.py``; the
population is defined by MANIFEST.csv and nothing else. Each RQ writes one tidy
CSV plus the LaTeX table used in the paper.

RQ1  Extent of reuse
     Per dataset: ontologies and distinct patterns in the reuse set and the
     complete-reuse subset; distribution over reuse thresholds (>=10..100% of a
     pattern's entities present); mean patterns reused per ontology; top-k most
     reused patterns.

RQ2  Forms of reuse
     For each reused pattern entity, the structural role it plays in the host:
     direct occurrence, subclass / superclass of a pattern class, sub- /
     superproperty, usage as a property in a triple, inside an OWL restriction,
     or in an equivalence axiom. One entity can occupy several roles, so
     percentages are over role occurrences, not entities.

RQ3  Sources of complexity
     Presence of expressive OWL constructs on the pattern side vs the ontology
     side, and an approximate DL expressivity label per side. The question is
     whether the additional complexity of an ODP-reusing ontology comes from the
     pattern or from the ontology.

RQ4  Ontology expansion
     How many classes / object properties / datatype properties an ontology adds
     beyond the patterns it reuses.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import OWL, RDF, RDFS, Graph, URIRef

from odpqual.entities import collect_entities, parse_graph, strip_annotations
from odpqual.gate import main_ontology_file, pattern_files

THRESHOLDS = (10, 30, 50, 70, 90, 100)

# ---------------------------------------------------------------- RQ2 roles
ROLES = ("direct", "subclass_of_pattern", "superclass_of_pattern",
         "subproperty_of_pattern", "superproperty_of_pattern",
         "used_as_property", "in_restriction", "equivalence")

# ---------------------------------------------------------------- RQ3 constructs
CONSTRUCTS = {
    "someValuesFrom": OWL.someValuesFrom,
    "allValuesFrom": OWL.allValuesFrom,
    "cardinality": OWL.cardinality,
    "minCardinality": OWL.minCardinality,
    "maxCardinality": OWL.maxCardinality,
    "qualifiedCardinality": OWL.qualifiedCardinality,
    "complementOf": OWL.complementOf,
    "unionOf": OWL.unionOf,
    "intersectionOf": OWL.intersectionOf,
    "oneOf": OWL.oneOf,
    "disjointWith": OWL.disjointWith,
    "equivalentClass": OWL.equivalentClass,
    "inverseOf": OWL.inverseOf,
    "TransitiveProperty": OWL.TransitiveProperty,
    "SymmetricProperty": OWL.SymmetricProperty,
    "FunctionalProperty": OWL.FunctionalProperty,
    "hasValue": OWL.hasValue,
}


def dl_label(present: set[str]) -> str:
    """Coarse DL expressivity label from the constructs observed.

    Deliberately approximate and documented as such: a full DL classifier needs
    a reasoner. The label orders corpora, it does not certify them.
    """
    has_neg = bool(present & {"complementOf", "disjointWith"})
    has_union = "unionOf" in present
    has_card = bool(present & {"cardinality", "minCardinality", "maxCardinality",
                               "qualifiedCardinality"})
    has_inv = "inverseOf" in present
    has_role_ax = bool(present & {"TransitiveProperty", "SymmetricProperty"})
    has_func = "FunctionalProperty" in present
    has_exist = bool(present & {"someValuesFrom", "allValuesFrom"})

    if not present:
        return "RDFS"
    if not (has_neg or has_union or has_card or has_inv):
        return "EL" if has_exist else "RDFS"
    base = "ALC" if (has_neg or has_union) else "AL"
    s = base
    if has_role_ax:
        s += "H"
    if has_inv:
        s += "I"
    if has_card or has_func:
        s += "Q"
    return s


def constructs_in(g: Graph) -> set[str]:
    out = set()
    for name, pred in CONSTRUCTS.items():
        if name in ("TransitiveProperty", "SymmetricProperty", "FunctionalProperty"):
            if next(g.triples((None, RDF.type, pred)), None) is not None:
                out.add(name)
        elif next(g.triples((None, pred, None)), None) is not None:
            out.add(name)
    return out


def reuse_roles(g: Graph, host_ents: set[str],
                pattern_ents: set[str]) -> Counter:
    """Count the structural roles reused pattern entities play in the host."""
    roles: Counter = Counter()
    reused = host_ents & pattern_ents
    if not reused:
        return roles
    roles["direct"] = len(reused)

    for s, o in g.subject_objects(RDFS.subClassOf):
        ss, os_ = str(s), str(o)
        if os_ in pattern_ents and ss not in pattern_ents:
            roles["subclass_of_pattern"] += 1
        if ss in pattern_ents and os_ not in pattern_ents:
            roles["superclass_of_pattern"] += 1
    for s, o in g.subject_objects(RDFS.subPropertyOf):
        ss, os_ = str(s), str(o)
        if os_ in pattern_ents and ss not in pattern_ents:
            roles["subproperty_of_pattern"] += 1
        if ss in pattern_ents and os_ not in pattern_ents:
            roles["superproperty_of_pattern"] += 1
    for _s, p, _o in g:
        if isinstance(p, URIRef) and str(p) in pattern_ents:
            roles["used_as_property"] += 1
    for pred in (OWL.someValuesFrom, OWL.allValuesFrom, OWL.onClass,
                 OWL.onProperty, OWL.hasValue):
        for o in g.objects(None, pred):
            if isinstance(o, URIRef) and str(o) in pattern_ents:
                roles["in_restriction"] += 1
    for pred in (OWL.equivalentClass, OWL.equivalentProperty, OWL.sameAs):
        for s, o in g.subject_objects(pred):
            if str(s) in pattern_ents or str(o) in pattern_ents:
                roles["equivalence"] += 1
    return roles


def entity_kinds(g: Graph) -> dict[str, set[str]]:
    """Split declared entities into classes / object props / datatype props."""
    out = {"C": set(), "OP": set(), "DP": set()}
    for t, key in ((OWL.Class, "C"), (RDFS.Class, "C"),
                   (OWL.ObjectProperty, "OP"), (OWL.DatatypeProperty, "DP")):
        for s, _, _ in g.triples((None, RDF.type, t)):
            if isinstance(s, URIRef):
                out[key].add(str(s))
    return out


# ----------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------

def analyse(manifest: Path, data_root: Path, datasets: dict[str, str],
            strip_annots: bool = True, max_mb: float = 0.0,
            population: str = "all",
            approved_labels: set[tuple[str, str, str]] | None = None) -> dict:
    """Re-parse ontologies and compute per-pair RQ measures.

    population="all"    every ontology that parses, whether or not it passed the
                        quality gate. This is what the RQ statistics report:
                        the real corpus as it exists, not a curated view of it.
                        Each row carries `gate_passed` and `drop_reason`, so any
                        table can be recomputed for either population.
    population="gated"  only ontologies that passed the gate.

    Reporting the real data matters: "37% of the corpus is stubs" is a finding
    about ODP reuse in the wild, and it disappears if the stubs are filtered out
    before the statistics are computed.

    ``approved_labels`` holds (dataset, ontology, pattern) triples whose
    label-only matches a human adjudicated as genuine reuse. For those pairs the
    reused set becomes the union of IRI-matched and label-matched entities;
    every other pair stays exact-IRI. This mirrors ``gate.py`` exactly -- it
    calls the same ``gt_by_label`` with the same guards -- so the manifest and
    these statistics cannot describe different ground truths. Without it, running
    the RQ statistics over an iri+label manifest would silently report IRI-only
    measures for a label-extended population.
    """
    allrows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    if population == "gated":
        rows = [r for r in allrows if not r["drop_reason"]]
    else:
        # everything that parsed -- an unparseable file yields no measures at all
        rows = [r for r in allrows
                if not r["drop_reason"].startswith("G1_parse")]
    pairs: list[dict] = []
    per_onto: list[dict] = []

    for i, r in enumerate(rows, 1):
        ds, oid = r["dataset"], r["ontology_id"]
        onto_dir = data_root / datasets[ds] / oid
        if not onto_dir.is_dir():
            continue
        main = main_ontology_file(onto_dir)
        if main is None:
            continue
        if max_mb > 0 and main.stat().st_size > max_mb * 1024 * 1024:
            continue
        g, err = parse_graph(main)
        if g is None:
            continue
        if strip_annots:
            g, _ = strip_annotations(g)
        host = set(collect_entities(g, wide=True))
        hk = entity_kinds(g)
        h_constructs = constructs_in(g)

        p_union: set[str] = set()
        n_pat_reused = 0
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
            # Human-confirmed label matches extend this pair only. Same call and
            # same guards as odpqual/gate.py, so the two agree by construction.
            n_label_extra = 0
            if approved_labels and (ds, oid, pf.stem) in approved_labels:
                from odpqual.gtmodes import gt_by_label
                extra = gt_by_label(g, host, pg) - inter
                if extra:
                    n_label_extra = len(extra)
                    pents = pents | extra
                    inter = inter | extra
            cov = 100.0 * len(inter) / len(pents)
            roles = reuse_roles(g, host, pents)
            pk = entity_kinds(pg)
            p_constructs = constructs_in(pg)
            if inter:
                n_pat_reused += 1
                p_union |= pents
            pairs.append({
                "dataset": ds, "ontology": oid, "pattern": pf.stem,
                "gate_passed": int(not r["drop_reason"]),
                "drop_reason": r["drop_reason"],
                "n_pattern_entities": len(pents), "n_reused": len(inter),
                # how much of n_reused came from a human-confirmed label match
                # rather than an IRI match; 0 for every pair under iri mode, so
                # the strict subset stays recoverable from the released CSV.
                "n_label_confirmed": n_label_extra,
                "coverage_pct": round(cov, 2), "complete": int(cov >= 100.0),
                **{f"role_{k}": roles.get(k, 0) for k in ROLES},
                "pattern_dl": dl_label(p_constructs),
                "ontology_dl": dl_label(h_constructs),
                "pattern_constructs": "|".join(sorted(p_constructs)),
                "ontology_constructs": "|".join(sorted(h_constructs)),
                "constructs_only_in_ontology": "|".join(
                    sorted(h_constructs - p_constructs)),
                "n_pattern_C": len(pk["C"]), "n_pattern_OP": len(pk["OP"]),
                "n_pattern_DP": len(pk["DP"]),
            })
        per_onto.append({
            "dataset": ds, "ontology": oid,
            "gate_passed": int(not r["drop_reason"]),
            "drop_reason": r["drop_reason"],
            "n_entities_wide": len(host),
            "n_patterns_reused": n_pat_reused,
            "n_C": len(hk["C"]), "n_OP": len(hk["OP"]), "n_DP": len(hk["DP"]),
            "added_C": len(hk["C"] - p_union),
            "added_OP": len(hk["OP"] - p_union),
            "added_DP": len(hk["DP"] - p_union),
            "ontology_dl": dl_label(h_constructs),
        })
        if i % 50 == 0:
            print(f"  analysed {i}/{len(rows)}", flush=True)

    return {"pairs": pairs, "per_onto": per_onto}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[WARN] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {path.name}  ({len(rows)} rows)")


def rq1(pairs, per_onto) -> list[dict]:
    out = []
    by_ds = defaultdict(list)
    for p in pairs:
        by_ds[p["dataset"]].append(p)
    onto_by_ds = defaultdict(list)
    for o in per_onto:
        onto_by_ds[o["dataset"]].append(o)

    for ds, ps in sorted(by_ds.items()):
        ontos = onto_by_ds[ds]
        reusing = {p["ontology"] for p in ps if p["n_reused"] > 0}
        complete = {p["ontology"] for p in ps if p["complete"]}
        gated = [p for p in ps if p.get("gate_passed")]
        row = {
            "dataset": ds,
            "onto_analysed": len(ontos),
            "onto_gate_passed": sum(1 for o in ontos if o.get("gate_passed")),
            "onto_reuse_gated": len({p["ontology"] for p in gated
                                     if p["n_reused"] > 0}),
            "onto_complete_gated": len({p["ontology"] for p in gated
                                        if p["complete"]}),
            "onto_reuse": len(reusing),
            "patterns_reuse": len({p["pattern"] for p in ps if p["n_reused"] > 0}),
            "onto_complete": len(complete),
            "patterns_complete": len({p["pattern"] for p in ps if p["complete"]}),
            "mean_patterns_per_onto": round(
                sum(o["n_patterns_reused"] for o in ontos) / max(len(ontos), 1), 2),
        }
        for t in THRESHOLDS:
            n = len({p["ontology"] for p in ps if p["coverage_pct"] >= t})
            row[f"ge_{t}pct"] = n
            row[f"ge_{t}pct_share"] = round(100.0 * n / max(len(ontos), 1), 2)
        out.append(row)
    return out


def rq1_top_patterns(pairs, k: int = 10) -> list[dict]:
    out = []
    by_ds = defaultdict(Counter)
    for p in pairs:
        if p["n_reused"] > 0:
            by_ds[p["dataset"]][p["pattern"]] += 1
    for ds, c in sorted(by_ds.items()):
        for rank, (pat, n) in enumerate(c.most_common(k), 1):
            out.append({"dataset": ds, "rank": rank, "pattern": pat,
                        "n_ontologies": n})
    return out


def rq2(pairs) -> list[dict]:
    out = []
    by_ds = defaultdict(Counter)
    for p in pairs:
        if p["n_reused"] == 0:
            continue
        for r in ROLES:
            by_ds[p["dataset"]][r] += p[f"role_{r}"]
    for ds, c in sorted(by_ds.items()):
        total = sum(c.values()) or 1
        row = {"dataset": ds, "total_role_occurrences": total}
        for r in ROLES:
            row[r] = c[r]
            row[f"{r}_pct"] = round(100.0 * c[r] / total, 2)
        out.append(row)
    return out


def rq3(pairs) -> list[dict]:
    out = []
    by_ds = defaultdict(list)
    for p in pairs:
        if p["n_reused"] > 0:
            by_ds[p["dataset"]].append(p)
    for ds, ps in sorted(by_ds.items()):
        n = len(ps) or 1
        p_dl = Counter(p["pattern_dl"] for p in ps)
        o_dl = Counter(p["ontology_dl"] for p in ps)
        only_o = Counter()
        for p in ps:
            for c in filter(None, p["constructs_only_in_ontology"].split("|")):
                only_o[c] += 1
        out.append({
            "dataset": ds, "n_pairs": n,
            "pattern_dl_mode": p_dl.most_common(1)[0][0] if p_dl else "",
            "ontology_dl_mode": o_dl.most_common(1)[0][0] if o_dl else "",
            "pattern_dl_dist": ";".join(f"{k}={v}" for k, v in p_dl.most_common()),
            "ontology_dl_dist": ";".join(f"{k}={v}" for k, v in o_dl.most_common()),
            "pairs_ontology_more_expressive": sum(
                1 for p in ps if p["constructs_only_in_ontology"]),
            "pct_ontology_more_expressive": round(100.0 * sum(
                1 for p in ps if p["constructs_only_in_ontology"]) / n, 2),
            "top_constructs_only_in_ontology": ";".join(
                f"{k}={v}" for k, v in only_o.most_common(6)),
        })
    return out


def rq4(per_onto) -> list[dict]:
    out = []
    by_ds = defaultdict(list)
    for o in per_onto:
        by_ds[o["dataset"]].append(o)
    for ds, os_ in sorted(by_ds.items()):
        n = len(os_) or 1
        out.append({
            "dataset": ds, "n_ontologies": n,
            **{f"mean_{k}": round(sum(o[k] for o in os_) / n, 2)
               for k in ("n_C", "n_OP", "n_DP", "added_C", "added_OP", "added_DP")},
            "pct_entities_added": round(100.0 * sum(
                o["added_C"] + o["added_OP"] + o["added_DP"] for o in os_) / max(sum(
                    o["n_C"] + o["n_OP"] + o["n_DP"] for o in os_), 1), 2),
        })
    return out
