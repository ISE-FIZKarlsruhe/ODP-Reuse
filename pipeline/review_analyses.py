#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
review_analyses.py -- the numbers the pre-submission review asked for.

Five questions, each answered from files already in the repository, none needing
a new harvest:

A. WHAT IS A PATTERN'S ENTITY SET?
   Per library: patterns, mean/median entities, and the same with the
   foundational-ontology files separated out. The review's objection is that
   21.73 classes per pattern is implausible for a design pattern and suggests an
   imports closure is being counted. Answer both parts: state the definition
   (WIDE entity set of the pattern file as distributed; rdflib does not follow
   owl:imports, so no closure is ever materialised) and show the distribution
   with and without the files that are whole foundational ontologies.

B. HOW MUCH OF RQ1 IS DUL?
   Reuse and coverage recomputed with DUL removed from the pattern library.

C. IS COMPLETE REUSE AN IMPORT DETECTOR?
   For every ontology-pattern pair marked complete, does the ontology declare
   owl:imports on the pattern's own ontology IRI?

D. HOW BIG IS A GROUND-TRUTH PATTERN INSTANCE?
   The distribution of |g| -- what UC1's metrics are actually asked to recover,
   and the reason a one-entity-per-cluster baseline scores as high as it does.

E. DOES TABLE 4's 45 = 45 SURVIVE?
   The review flags that both portions report 45 distinct patterns reused and
   asks whether the identity is genuine. Recount and list both sets.

Usage
-----
    python pipeline/review_analyses.py --pairs out/rq_iri_gated/pairs.csv \
        --manifest out/MANIFEST.csv --data-root /path/to/MultiSource \
        --out out/review
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdflib import OWL, RDF  # noqa: E402

from build_manifest import DATASETS  # noqa: E402
from odpqual.entities import parse_graph  # noqa: E402
from odpqual.gate import main_ontology_file, pattern_files  # noqa: E402

# Pattern files that are whole foundational or upper ontologies rather than
# content patterns. Identified by inspecting every pattern file with more than
# 50 entities and checking its declared ontology IRI and documentation.
FOUNDATIONAL = {"DUL", "dul", "DOLCE-Lite", "IOLite", "iolite", "cco",
                "CommonCoreOntologies", "bfo", "BFO", "Dolce", "DUL-owl"}


def q(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "mean": round(st.mean(xs), 2),
            "median": round(st.median(xs), 2),
            "p25": xs[len(xs) // 4], "p75": xs[(3 * len(xs)) // 4],
            "min": xs[0], "max": xs[-1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-import-test", nargs="*", default=["OBO"],
                    help="datasets excluded from the owl:imports test")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = [r for r in csv.DictReader(args.pairs.open(encoding="utf-8"))
             if r["gate_passed"] == "1"]
    gated = {(r["dataset"], r["ontology_id"])
             for r in csv.DictReader(args.manifest.open(encoding="utf-8"))
             if not r["drop_reason"]}
    print(f"{len(pairs)} gated pairs, {len(gated)} gated ontologies")

    out: dict = {}

    # ---------------- A: pattern entity sets --------------------------
    by_pattern: dict[str, dict] = {}
    for r in pairs:
        p = r["pattern"]
        if p in by_pattern:
            continue
        by_pattern[p] = {
            "n_entities": int(r["n_pattern_entities"]),
            "n_C": int(r["n_pattern_C"]), "n_OP": int(r["n_pattern_OP"]),
            "n_DP": int(r["n_pattern_DP"]),
        }
    found = {p: v for p, v in by_pattern.items() if p in FOUNDATIONAL}
    content = {p: v for p, v in by_pattern.items() if p not in FOUNDATIONAL}
    out["A_entity_sets"] = {
        "definition": ("WIDE entity set of the pattern FILE as distributed; "
                       "owl:imports is never resolved, so no import closure is "
                       "materialised. Size comes from the file's own content."),
        "all": {"patterns": len(by_pattern),
                "entities": q([v["n_entities"] for v in by_pattern.values()]),
                "classes": q([v["n_C"] for v in by_pattern.values()]),
                "object_properties": q([v["n_OP"] for v in by_pattern.values()])},
        "foundational_files": sorted(found),
        "foundational": {"patterns": len(found),
                         "entities": q([v["n_entities"] for v in found.values()]),
                         "classes": q([v["n_C"] for v in found.values()]),
                         "object_properties": q([v["n_OP"] for v in found.values()])},
        "content_only": {"patterns": len(content),
                         "entities": q([v["n_entities"] for v in content.values()]),
                         "classes": q([v["n_C"] for v in content.values()]),
                         "object_properties": q([v["n_OP"] for v in content.values()])},
        "largest_20": sorted(((p, v["n_entities"], v["n_C"], v["n_OP"])
                              for p, v in by_pattern.items()),
                             key=lambda t: -t[1])[:20],
    }

    # ---------------- B: RQ1 with and without DUL ---------------------
    b: dict = {}
    for ds in sorted({r["dataset"] for r in pairs}):
        sub = [r for r in pairs if r["dataset"] == ds]
        nodul = [r for r in sub if r["pattern"] != "DUL"]
        onts = {r["ontology"] for r in sub}
        onts_nodul = {r["ontology"] for r in nodul}
        b[ds] = {
            "with_DUL": {
                "ontologies_with_reuse": len(onts),
                "pairs": len(sub),
                "patterns_per_ontology": round(len(sub) / max(len(onts), 1), 2),
                "mean_coverage_pct": round(
                    st.mean([float(r["coverage_pct"]) for r in sub]), 2),
                "complete_ontologies": len({r["ontology"] for r in sub
                                            if r["complete"] == "1"}),
                "distinct_patterns": len({r["pattern"] for r in sub}),
            },
            "without_DUL": {
                "ontologies_with_reuse": len(onts_nodul),
                "pairs": len(nodul),
                "patterns_per_ontology": round(
                    len(nodul) / max(len(onts_nodul), 1), 2),
                "mean_coverage_pct": round(
                    st.mean([float(r["coverage_pct"]) for r in nodul]), 2)
                if nodul else None,
                "complete_ontologies": len({r["ontology"] for r in nodul
                                            if r["complete"] == "1"}),
                "distinct_patterns": len({r["pattern"] for r in nodul}),
            },
            "ontologies_reusing_only_DUL": len(onts - onts_nodul),
            "DUL_share_of_pairs_pct": round(
                100 * (len(sub) - len(nodul)) / max(len(sub), 1), 2),
        }
    out["B_dul"] = b

    # ---------------- C/D/E: need the corpus --------------------------
    complete_pairs = [r for r in pairs if r["complete"] == "1"]
    imports_yes, imports_no, unresolved = [], [], []
    gt_sizes: list[int] = []
    per_ds_patterns: dict[str, set[str]] = defaultdict(set)

    for r in pairs:
        per_ds_patterns[r["dataset"]].add(r["pattern"])
        gt_sizes.append(int(r["n_reused"]))

    # C: import test, over the complete-reuse pairs only.
    by_onto: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in complete_pairs:
        by_onto[(r["dataset"], r["ontology"])].append(r["pattern"])

    for (ds, oid), pats in sorted(by_onto.items()):
        # DOS-DP is excluded by construction, not by cost: a DOS-DP pattern file
        # is a materialised OWL subset with no ontology IRI of its own (see the
        # library characterisation in the Methodology), so there is no IRI an
        # ontology could import and the import test is vacuous. Its files are
        # also multi-GB, which the test does not need to pay for.
        if ds in args.skip_import_test:
            unresolved.append((ds, oid, "skipped:no_pattern_ontology_iri"))
            continue
        onto_dir = args.data_root / DATASETS[ds] / oid
        mf = main_ontology_file(onto_dir)
        if mf is None:
            unresolved.append((ds, oid, "no_ontology_file"))
            continue
        g, err = parse_graph(mf)
        if g is None:
            unresolved.append((ds, oid, f"parse_error: {err[:60]}"))
            continue
        declared_imports = {str(o) for o in g.objects(None, OWL.imports)}
        for pf in pattern_files(onto_dir):
            if pf.stem not in pats:
                continue
            pg, _ = parse_graph(pf)
            if pg is None:
                unresolved.append((ds, oid, f"pattern_parse:{pf.stem}"))
                continue
            piris = {str(s) for s in pg.subjects(RDF.type, OWL.Ontology)}
            hit = bool(piris & declared_imports)
            # An imports IRI often differs from the declared ontology IRI only by
            # a trailing slash or a version segment; accept a prefix match too.
            if not hit:
                hit = any(any(i.rstrip("/#").startswith(p.rstrip("/#"))
                              or p.rstrip("/#").startswith(i.rstrip("/#"))
                              for i in declared_imports) for p in piris if p)
            (imports_yes if hit else imports_no).append((ds, oid, pf.stem))

    out["C_complete_by_import"] = {
        "complete_pairs": len(complete_pairs),
        "complete_ontologies": len(by_onto),
        "pairs_with_owl_imports": len(imports_yes),
        "pairs_without_owl_imports": len(imports_no),
        "unresolved": len(unresolved),
        "pct_by_import": round(100 * len(imports_yes)
                               / max(len(imports_yes) + len(imports_no), 1), 1),
        "ontologies_with_at_least_one_imported_complete_pattern": len(
            {(d, o) for d, o, _p in imports_yes}),
        "examples_import": imports_yes[:15],
        "examples_materialised": imports_no[:15],
        "unresolved_detail": unresolved[:15],
    }

    # D: ground-truth instance sizes
    out["D_gt_instance_size"] = {
        "all": q(gt_sizes),
        "by_dataset": {ds: q([int(r["n_reused"]) for r in pairs
                              if r["dataset"] == ds])
                       for ds in sorted({r["dataset"] for r in pairs})},
        "note": ("|g| is the number of pattern entities actually present in the "
                 "host, i.e. what UC1 must recover. A one-entity-per-cluster "
                 "baseline scores F1 = 2/(|g|+1) on every matched pair."),
    }

    # E: distinct patterns per portion
    out["E_distinct_patterns"] = {
        ds: {"n": len(s), "patterns": sorted(s)}
        for ds, s in sorted(per_ds_patterns.items())
    }
    if "github" in per_ds_patterns and "noOBO" in per_ds_patterns:
        a, bb = per_ds_patterns["github"], per_ds_patterns["noOBO"]
        out["E_distinct_patterns"]["github_vs_noOBO"] = {
            "n_github": len(a), "n_noOBO": len(bb),
            "identical_sets": sorted(a) == sorted(bb),
            "shared": sorted(a & bb), "github_only": sorted(a - bb),
            "noOBO_only": sorted(bb - a),
        }

    (args.out / "review_analyses.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if k != "E_distinct_patterns" else "...")
                      for k, v in out.items()}, indent=2)[:6000])
    print(f"\n[OK] {args.out / 'review_analyses.json'}")


if __name__ == "__main__":
    main()
