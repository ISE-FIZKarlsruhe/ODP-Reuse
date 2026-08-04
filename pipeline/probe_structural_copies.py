#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_structural_copies.py -- feasibility probe for a candidate gate rule G7:
"structure-preserving, IRI-perturbed copy of a pattern".

WHY
---
The shipped gate (odpqual/gate.py) catches two degenerate cases:

    G2  the ontology IS a pattern      -- declared IRI in a pattern namespace,
                                          byte-identical file, or identical WIDE
                                          entity set (exact IRIs)
    G3  the ontology is a byte-duplicate of another corpus member

Both are *exact-IRI* tests. A reviewer asked about a subtler case they cannot
see: a file that reproduces a pattern's structure but PERTURBS the IRIs --
``...#Role`` becomes ``...#Role_`` or ``...#role``, or the whole fragment is
rehosted under the author's own namespace with mechanically-derived local names.
Under exact-IRI matching such a file has an EMPTY intersection with its pattern,
so it is scored as *no reuse* and removed by G5, when in truth it is either

    (i)  a pattern copy that G2 should have removed for the reason G2 exists, or
    (ii) genuine re-modelled reuse that the corpus is currently blind to.

Either way it is invisible. This script measures how many such files exist.

THE DETECTOR (S1/S2), AND WHY IT HAS NO THRESHOLD
-------------------------------------------------
Let E_H be the host's WIDE entity set and E_P a pattern's, both computed by the
same ``odpqual.entities`` code the gate uses, after the same annotation
stripping. Define the IRI-blind projection

    N(S) = { norm_label(local_name(iri)) : iri in S }

reusing ``odpqual.gtmodes.norm_label`` and ``local_name`` verbatim, so
``Role``, ``Role_``, ``role``, ``AgentRole``/``agent_role`` collapse exactly as
they do in the existing label mode. Then:

    S1  IRI-PERTURBED COPY          N(E_H) == N(E_P)  and  |E_H & E_P| == 0
    S2  IRI-PERTURBED FULL REUSE    N(E_P) <= N(E_H)  and  |E_H & E_P| == 0
    S3  partial                     |N(E_P) & N(E_H)| / |N(E_P)|, reported as a
                                    distribution, never thresholded

S1 is the set-level analogue of G2(c) with IRIs erased; S2 is the analogue of
G6. Both are set relations on normalised strings -- equality and inclusion --
so, exactly as argued in ``odpqual/gtmodes.py`` and
``out/adjudication/METHODS.md`` s2, there is NO similarity measure and NO
cut-off. The ``|E_H & E_P| == 0`` clause makes the rule strictly complementary
to G2: it can only fire where exact-IRI matching sees nothing at all.

The one quantity that could become a threshold -- how big a pattern must be
before set equality is convincing -- is not fixed here either. ``report`` emits
the count as a function of minimum pattern size, in the style of s D2's G4
sensitivity table, so the reader picks the operating point.

CONFIRMATION: A CANONICAL STRUCTURAL FINGERPRINT
------------------------------------------------
S1/S2 are name-based and therefore blind to renamings that survive
normalisation (``Role`` -> ``Rol``, ``C1``). As an independent check on the
flagged shortlist, ``wl_fingerprint`` computes a 1-dimensional Weisfeiler-Leman
colour-refinement invariant of the annotation-stripped graph with EVERY
non-standard IRI erased (all user IRIs get the same initial colour ``_U``; only
RDF/RDFS/OWL/XSD terms keep their identity, so a class stays distinguishable
from a property). Refinement runs until the colour partition stops changing, so
the iteration count is not a parameter either. Equal fingerprints mean the two
graphs are WL-indistinguishable, hence isomorphic modulo IRI renaming for all
practical purposes; unequal fingerprints prove they are not isomorphic. It is
sound-but-incomplete in the standard way (a necessary condition for
isomorphism), and it is *exact*: adding one triple to a copy breaks it. That is
why it confirms rather than detects.

WHAT IS DELIBERATELY NOT IMPLEMENTED
------------------------------------
Embedding cosine (RDF2Vec/OWL2Vec*), which the reviewer also suggested, is not
implemented as a gate. It needs a similarity cut-off plus walk depth, walks per
entity, dimension, window and epochs; word2vec training is not bit-reproducible
across threads; and cross-graph comparison requires either training on the
union graph (which contaminates the very distinction being measured) or an
alignment step. Adopting a tuned cosine threshold would contradict the
reproducibility argument the dataset rests on. See REPORT.md s1(c).

STAGE FILTER (never a dataset exclusion)
----------------------------------------
Files above ``--max-mb`` are not parsed by this probe and are written to
``skipped.csv`` with reason ``file_too_large``, following the convention in
FILTERS.md sB. They remain corpus members.

USAGE
-----
    python probe_structural_copies.py scan \
        --data-root "C:/Users/eno/Documents/PhD/MultiSource_server" \
        --manifest out/MANIFEST.csv --out out/structprobe --max-mb 20 --resume

    python probe_structural_copies.py report --out out/structprobe

OUTPUTS (under --out)
    cache/pairs.jsonl     per-ontology scan cache (--resume reads this)
    pairs.csv             every (ontology, pattern) pair with all metrics
    flagged.csv           S1/S2 pairs with the perturbation evidence
    by_pattern_size.csv   flagged counts as a function of min |N(E_P)|
    coverage_hist.csv     S3 distribution
    skipped.csv           stage exclusions, with reason
    summary.json          every number quoted in REPORT.md
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdflib import BNode, Graph, Literal, URIRef  # noqa: E402

from odpqual.entities import (  # noqa: E402
    collect_entities,
    parse_graph,
    strip_annotations,
)
from odpqual.gate import main_ontology_file, pattern_files, sha256  # noqa: E402
from odpqual.gtmodes import GENERIC_LABELS, local_name, norm_label  # noqa: E402

logging.getLogger("rdflib").setLevel(logging.ERROR)

DATASETS = {"github": "github_dataset", "noOBO": "noOBO", "OBO": "OBO"}

# Vocabularies whose IRIs are NOT perturbable and must keep their identity in the
# structural fingerprint: erasing them would make an owl:Class indistinguishable
# from an owl:ObjectProperty and the invariant would be worthless.
STD_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)

COVERAGE_BINS = (0, 10, 30, 50, 70, 90, 100)


# --------------------------------------------------------------------------- #
# the IRI-blind projection
# --------------------------------------------------------------------------- #
def nln(iri: str) -> str:
    """Normalised local name. `.../onto#Role_` -> `role`."""
    return norm_label(local_name(iri))


def nln_set(iris) -> set[str]:
    return {k for k in (nln(i) for i in iris) if k}


# --------------------------------------------------------------------------- #
# the confirmatory structural invariant
# --------------------------------------------------------------------------- #
def is_std(iri: str) -> bool:
    return any(iri.startswith(ns) for ns in STD_NAMESPACES)


def wl_fingerprint(g: Graph, max_triples: int = 200_000) -> str:
    """IRI-blind 1-WL canonical fingerprint of ``g``.

    Every non-standard IRI is replaced by the single colour ``_U`` and every
    non-standard predicate by ``_P``; RDF/RDFS/OWL/XSD terms keep their identity.
    Colours are refined until the partition stops growing -- no iteration
    parameter. The returned digest is invariant under any bijective renaming of
    the user IRIs and blank nodes, so two graphs that differ only by such a
    renaming get the same digest.

    Returns "" when the graph exceeds ``max_triples`` (recorded, not silent).
    """
    edges: list[tuple] = []
    nodes: set = set()
    n = 0
    for s, p, o in g:
        n += 1
        if n > max_triples:
            return ""
        if isinstance(o, Literal):
            continue
        ps = str(p)
        edges.append((s, ps if is_std(ps) else "_P", o))
        nodes.add(s)
        nodes.add(o)
    if not nodes:
        return "empty"

    col: dict = {}
    for nd in nodes:
        if isinstance(nd, BNode):
            col[nd] = "_B"
        elif isinstance(nd, URIRef) and is_std(str(nd)):
            col[nd] = str(nd)
        else:
            col[nd] = "_U"

    out_adj: dict = collections.defaultdict(list)
    in_adj: dict = collections.defaultdict(list)
    for s, pl, o in edges:
        out_adj[s].append((pl, o))
        in_adj[o].append((pl, s))

    prev = -1
    for _ in range(len(nodes) + 1):
        new = {}
        for nd in nodes:
            sig = (
                col[nd],
                tuple(sorted((pl, col[m]) for pl, m in out_adj.get(nd, ()))),
                tuple(sorted((pl, col[m]) for pl, m in in_adj.get(nd, ()))),
            )
            new[nd] = hashlib.sha1(repr(sig).encode()).hexdigest()[:16]
        parts = len(set(new.values()))
        col = new
        if parts == prev:
            break
        prev = parts

    ms = sorted((col[s], pl, col[o]) for s, pl, o in edges)
    return hashlib.sha256(
        repr((len(nodes), len(edges), ms)).encode()
    ).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# what kind of perturbation is it?
# --------------------------------------------------------------------------- #
def classify_perturbation(pat_names: list[str], host_names: list[str]) -> str:
    """Weakest perturbation class explaining one normalised-name match."""
    ps, hs = set(pat_names), set(host_names)
    if ps & hs:
        return "namespace_only"          # local name byte-identical
    if {p.lower() for p in ps} & {h.lower() for h in hs}:
        return "case"                    # Role vs role
    strip = lambda s: "".join(c for c in s if c.isalnum())  # noqa: E731
    if {strip(p) for p in ps} & {strip(h) for h in hs}:
        return "punctuation"             # Role vs Role_ , has_part vs hasPart
    return "camel_or_other"              # only the full normalisation matches


def perturbation_evidence(pat_iris: set[str], host_iris: set[str],
                          keys: set[str], limit: int = 6) -> tuple[str, str]:
    """(profile, examples) for the normalised keys shared by the two sides."""
    pidx: dict[str, list[str]] = collections.defaultdict(list)
    hidx: dict[str, list[str]] = collections.defaultdict(list)
    for i in pat_iris:
        pidx[nln(i)].append(local_name(i))
    for i in host_iris:
        hidx[nln(i)].append(local_name(i))
    prof: collections.Counter = collections.Counter()
    ex: list[str] = []
    for k in sorted(keys):
        if k not in pidx or k not in hidx:
            continue
        cls = classify_perturbation(pidx[k], hidx[k])
        prof[cls] += 1
        if len(ex) < limit:
            ex.append(f"{sorted(pidx[k])[0]} -> {sorted(hidx[k])[0]} [{cls}]")
    profile = ",".join(f"{c}={n}" for c, n in prof.most_common())
    return profile, " ; ".join(ex)


def ns_of(iri: str) -> str:
    for sep in ("#", "/"):
        if sep in iri:
            return iri.rsplit(sep, 1)[0] + sep
    return iri


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def scan(args: argparse.Namespace) -> None:
    out = args.out
    (out / "cache").mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache" / "pairs.jsonl"
    skip_path = out / "cache" / "skipped.jsonl"

    manifest = {r["ontology_id"]: r
                for r in csv.DictReader(args.manifest.open(encoding="utf-8"))}

    done: set[str] = set()
    if args.resume and cache_path.is_file():
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["ontology_id"])
                except Exception:  # noqa: BLE001
                    continue
        print(f"[resume] {len(done)} ontologies already scanned")

    # population -------------------------------------------------------------
    todo = []
    for oid, r in manifest.items():
        if r["parsed"] != "ok":
            continue                       # G1: cannot be analysed at all
        if args.population == "g5" and not r["drop_reason"].startswith("G5"):
            continue
        if args.population == "lowown":
            w = int(r["n_entities_wide"] or 0)
            o = int(r["n_own_entities"] or 0)
            if not (w >= 20 and o <= 0.5 * w):
                continue
        todo.append(r)
    todo.sort(key=lambda r: r["ontology_id"])
    if args.limit:
        todo = todo[: args.limit]
    print(f"[population={args.population}] {len(todo)} ontologies "
          f"({len(todo) - len(done & {r['ontology_id'] for r in todo})} to do)")

    # pattern entity sets are memoised by content hash: 4,015 pattern files in
    # the corpus carry only 888 distinct contents.
    pmemo: dict[str, tuple] = {}

    fh_pairs = cache_path.open("a", encoding="utf-8")
    fh_skip = skip_path.open("a", encoding="utf-8")
    t0 = time.time()
    n_done = 0
    for r in todo:
        oid = r["ontology_id"]
        if oid in done:
            continue
        n_done += 1
        d = Path(r["path"])
        if not d.is_dir():
            fh_skip.write(json.dumps({"ontology_id": oid,
                                      "reason": "missing_dir"}) + "\n")
            continue
        main = main_ontology_file(d)
        if main is None:
            fh_skip.write(json.dumps({"ontology_id": oid,
                                      "reason": "no_ontology_file"}) + "\n")
            continue
        if args.max_mb > 0 and main.stat().st_size > args.max_mb * 1024 * 1024:
            fh_skip.write(json.dumps({
                "ontology_id": oid, "reason": "file_too_large",
                "mb": round(main.stat().st_size / 1048576, 1)}) + "\n")
            continue

        hg, err = parse_graph(main)
        if hg is None:
            fh_skip.write(json.dumps({"ontology_id": oid, "reason": "parse_error",
                                      "detail": err[:200]}) + "\n")
            continue
        hg, _ = strip_annotations(hg)
        he = set(collect_entities(hg, wide=True))
        hn = nln_set(he)

        rec = {
            "ontology_id": oid, "dataset": r["dataset"],
            "drop_reason": r["drop_reason"], "gate_passed": not r["drop_reason"],
            "n_entities_wide": len(he), "n_nln_host": len(hn),
            "n_own_entities": int(r["n_own_entities"] or 0),
            "host_fp": "", "pairs": [],
        }
        host_fp_needed = False

        for pf in pattern_files(d):
            if args.max_mb > 0 and pf.stat().st_size > args.max_mb * 1024 * 1024:
                fh_skip.write(json.dumps({
                    "ontology_id": oid, "pattern": pf.stem,
                    "reason": "pattern_too_large",
                    "mb": round(pf.stat().st_size / 1048576, 1)}) + "\n")
                continue
            h = sha256(pf)
            if h not in pmemo:
                pg, perr = parse_graph(pf)
                if pg is None:
                    pmemo[h] = (None, None, perr)
                else:
                    pg, _ = strip_annotations(pg)
                    pe = set(collect_entities(pg, wide=True))
                    pmemo[h] = (pe, nln_set(pe), "")
            pe, pn, perr = pmemo[h]
            if pe is None:
                fh_skip.write(json.dumps({"ontology_id": oid, "pattern": pf.stem,
                                          "reason": "pattern_parse_error",
                                          "detail": (perr or "")[:200]}) + "\n")
                continue
            if not pe:
                fh_skip.write(json.dumps({"ontology_id": oid, "pattern": pf.stem,
                                          "reason": "pattern_no_entities"}) + "\n")
                continue

            iri_ov = he & pe
            nln_ov = hn & pn
            shared_nonblock = {k for k in nln_ov if k not in GENERIC_LABELS}
            s1 = (hn == pn) and not iri_ov and bool(pn)
            s2 = pn <= hn and not iri_ov and bool(pn)
            prow = {
                "pattern": pf.stem,
                "n_pattern_entities": len(pe), "n_nln_pattern": len(pn),
                "n_iri_overlap": len(iri_ov),
                "iri_cov_pct": round(100.0 * len(iri_ov) / len(pe), 2),
                "n_nln_overlap": len(nln_ov),
                "nln_cov_pct": round(100.0 * len(nln_ov) / len(pn), 2) if pn else 0.0,
                "n_nln_overlap_nonblocked": len(shared_nonblock),
                "S1_iri_perturbed_copy": s1,
                "S2_iri_perturbed_full": s2,
                "pattern_fp": "", "fp_equal": "",
                "perturbation": "", "examples": "",
                "pattern_ns": "", "host_ns": "",
                "pattern_localnames": "", "host_matched_iris": "",
            }
            if s2:  # S1 implies S2; evidence + fingerprint only for the shortlist
                prow["perturbation"], prow["examples"] = perturbation_evidence(
                    pe, he, nln_ov)
                prow["pattern_ns"] = ";".join(sorted(
                    collections.Counter(ns_of(i) for i in pe))[:2])
                hmatched = {i for i in he if nln(i) in nln_ov}
                prow["host_ns"] = ";".join(sorted(
                    collections.Counter(ns_of(i) for i in hmatched))[:2])
                prow["pattern_localnames"] = " ".join(
                    sorted(local_name(i) for i in pe))[:600]
                prow["host_matched_iris"] = " ".join(sorted(hmatched)[:3])[:600]
                pg2, _ = parse_graph(pf)
                if pg2 is not None:
                    pg2, _ = strip_annotations(pg2)
                    prow["pattern_fp"] = wl_fingerprint(pg2)
                host_fp_needed = True
            rec["pairs"].append(prow)

        if host_fp_needed:
            rec["host_fp"] = wl_fingerprint(hg)
            for prow in rec["pairs"]:
                if prow["pattern_fp"] and rec["host_fp"]:
                    prow["fp_equal"] = str(prow["pattern_fp"] == rec["host_fp"])

        fh_pairs.write(json.dumps(rec) + "\n")
        if n_done % 100 == 0:
            fh_pairs.flush()
            fh_skip.flush()
            print(f"  {n_done} scanned  ({time.time() - t0:.0f}s)")
    fh_pairs.close()
    fh_skip.close()
    print(f"[scan] {n_done} ontologies in {time.time() - t0:.0f}s -> {cache_path}")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
PAIR_COLS = [
    "ontology_id", "dataset", "drop_reason", "gate_passed", "n_entities_wide",
    "n_own_entities", "n_nln_host", "pattern", "n_pattern_entities",
    "n_nln_pattern", "n_iri_overlap", "iri_cov_pct", "n_nln_overlap",
    "nln_cov_pct", "n_nln_overlap_nonblocked", "S1_iri_perturbed_copy",
    "S2_iri_perturbed_full", "host_fp", "pattern_fp", "fp_equal",
    "perturbation", "examples", "pattern_ns", "host_ns",
    "pattern_localnames", "host_matched_iris",
]


def _keyclass_counts(rows: list[dict]) -> dict[str, int]:
    """Total normalised keys per perturbation class over ``rows``.

    ``perturbation`` is stored as "namespace_only=8,case=1"; this sums the
    right-hand sides so the report can say how many INDIVIDUAL renamings of each
    kind were observed, not merely how many pairs contained one.
    """
    tot: collections.Counter = collections.Counter()
    for r in rows:
        for part in r["perturbation"].split(","):
            if "=" in part:
                cls, n = part.split("=", 1)
                try:
                    tot[cls] += int(n)
                except ValueError:
                    pass
    return dict(tot)


def report(args: argparse.Namespace) -> None:
    out = args.out
    recs = []
    with (out / "cache" / "pairs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                recs.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    # last write wins, so a re-scan supersedes an earlier partial one
    byid = {r["ontology_id"]: r for r in recs}
    recs = [byid[k] for k in sorted(byid)]

    rows = []
    for r in recs:
        for p in r["pairs"]:
            row = {c: "" for c in PAIR_COLS}
            row.update({k: v for k, v in r.items() if k != "pairs"})
            row.update(p)
            rows.append(row)

    with (out / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    s1 = [r for r in rows if r["S1_iri_perturbed_copy"]]
    s2 = [r for r in rows if r["S2_iri_perturbed_full"]]
    s2_only = [r for r in s2 if not r["S1_iri_perturbed_copy"]]

    with (out / "flagged.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(s2, key=lambda r: (-int(r["n_nln_pattern"]),
                                              r["ontology_id"])))

    # count as a function of minimum pattern size -- no threshold is chosen
    with (out / "by_pattern_size.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["min_nln_pattern", "S1_pairs", "S1_ontologies",
                    "S2_pairs", "S2_ontologies"])
        for t in (1, 2, 3, 5, 8, 10, 15, 20, 30):
            a = [r for r in s1 if int(r["n_nln_pattern"]) >= t]
            b = [r for r in s2 if int(r["n_nln_pattern"]) >= t]
            w.writerow([t, len(a), len({r["ontology_id"] for r in a}),
                        len(b), len({r["ontology_id"] for r in b})])

    # S3: how the IRI-blind coverage is distributed among pairs IRI sees nothing of
    zero_iri = [r for r in rows if int(r["n_iri_overlap"]) == 0]
    with (out / "coverage_hist.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nln_coverage_at_least_pct", "pairs", "ontologies"])
        for b in COVERAGE_BINS:
            sel = [r for r in zero_iri if float(r["nln_cov_pct"]) >= b]
            w.writerow([b, len(sel), len({r["ontology_id"] for r in sel})])

    skipped = []
    sp = out / "cache" / "skipped.jsonl"
    if sp.is_file():
        with sp.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    skipped.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    with (out / "skipped.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ontology_id", "pattern", "reason",
                                           "mb", "detail"],
                           extrasaction="ignore")
        w.writeheader()
        for s in skipped:
            w.writerow(s)

    # ---- validation against the label-mode human adjudication, if available.
    # The S2 pairs are a subset of the space that adjudication already covered,
    # so its human verdicts give S2 a MEASURED precision instead of an asserted
    # one -- and directly comparable to label mode's 0.215.
    adj_rows: list[dict] = []
    adj_summary: dict = {}
    if args.adjudication and args.adjudication.is_file():
        verdicts = {}
        with args.adjudication.open(encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                verdicts[(r["ontology"], r["pattern"])] = \
                    r.get("human_verdict", "").strip().lower()
        cnt: collections.Counter = collections.Counter()
        for r in s2:
            k = (r["ontology_id"], r["pattern"])
            v = verdicts.get(k)
            label = ("not_an_adjudication_item" if v is None
                     else (v or "deferred_unanimous_not_reuse"))
            cnt[label] += 1
            adj_rows.append({"ontology_id": r["ontology_id"],
                             "pattern": r["pattern"],
                             "n_nln_pattern": r["n_nln_pattern"],
                             "human_verdict": label})
        with (out / "s2_vs_human_adjudication.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["ontology_id", "pattern",
                                               "n_nln_pattern", "human_verdict"])
            w.writeheader()
            w.writerows(adj_rows)
        judged = cnt["reuse"] + cnt["not_reuse"]
        with_deferred = judged + cnt["deferred_unanimous_not_reuse"]
        adj_summary = {
            "counts": dict(cnt),
            "precision_on_explicitly_judged": (
                round(cnt["reuse"] / judged, 4) if judged else None),
            "n_explicitly_judged": judged,
            "precision_counting_deferred_as_not_reuse": (
                round(cnt["reuse"] / with_deferred, 4) if with_deferred else None),
            "n_including_deferred": with_deferred,
            "label_mode_precision_for_comparison": 0.215,
        }

    s1_ont = {r["ontology_id"] for r in s1}
    s2_ont = {r["ontology_id"] for r in s2}
    summary = {
        "ontologies_scanned": len(recs),
        "pairs_evaluated": len(rows),
        "ontologies_with_zero_iri_overlap_on_every_pattern": sum(
            1 for r in recs
            if r["pairs"] and all(p["n_iri_overlap"] == 0 for p in r["pairs"])),
        "S1_pairs": len(s1), "S1_ontologies": len(s1_ont),
        "S2_pairs": len(s2), "S2_ontologies": len(s2_ont),
        "S2_only_pairs": len(s2_only),
        "S1_rate_of_scanned": round(len(s1_ont) / max(1, len(recs)), 5),
        "S2_rate_of_scanned": round(len(s2_ont) / max(1, len(recs)), 5),
        "S1_fp_confirmed_pairs": sum(1 for r in s1 if r["fp_equal"] == "True"),
        "S1_fp_rejected_pairs": sum(1 for r in s1 if r["fp_equal"] == "False"),
        "S1_fp_unavailable_pairs": sum(1 for r in s1 if r["fp_equal"] == ""),
        "S1_by_drop_reason": dict(collections.Counter(
            (byid[o]["drop_reason"] or "(kept)").split(":")[0] for o in s1_ont)),
        "S2_by_drop_reason": dict(collections.Counter(
            (byid[o]["drop_reason"] or "(kept)").split(":")[0] for o in s2_ont)),
        "S1_by_dataset": dict(collections.Counter(
            byid[o]["dataset"] for o in s1_ont)),
        "S1_perturbation_profile": dict(collections.Counter(
            c.split("=")[0] for r in s1 for c in r["perturbation"].split(",") if c)),
        "S1_patterns": dict(collections.Counter(r["pattern"] for r in s1)),
        "S2_by_dataset": dict(collections.Counter(
            byid[o]["dataset"] for o in s2_ont)),
        "S2_perturbation_profile": dict(collections.Counter(
            c.split("=")[0] for r in s2 for c in r["perturbation"].split(",") if c)),
        "S2_perturbed_keys_by_class": dict(collections.Counter({
            k: v for k, v in _keyclass_counts(s2).items()})),
        "S2_patterns": dict(collections.Counter(r["pattern"] for r in s2)),
        "S2_fp_confirmed_pairs": sum(1 for r in s2 if r["fp_equal"] == "True"),
        "S2_fp_rejected_pairs": sum(1 for r in s2 if r["fp_equal"] == "False"),
        "S2_max_pattern_size": max((int(r["n_nln_pattern"]) for r in s2),
                                   default=0),
        "S2_vs_human_adjudication": adj_summary,
        "skipped": dict(collections.Counter(s["reason"] for s in skipped)),
        "distinct_pattern_stems_yielding_no_entities": len(
            {s.get("pattern", "") for s in skipped
             if s["reason"] == "pattern_no_entities"}),
        "ontologies_affected_by_empty_pattern_entities": len(
            {s["ontology_id"] for s in skipped
             if s["reason"] == "pattern_no_entities"}),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# validation of the fingerprint -- without this, "fp_equal=False everywhere" is
# not evidence of anything, because a broken invariant also says False everywhere
# --------------------------------------------------------------------------- #
def _rename_graph(g: Graph, salt: str = "x") -> Graph:
    """Bijectively rename every non-standard IRI and every blank node."""
    out = Graph()
    m: dict = {}

    def sub(t):
        if isinstance(t, URIRef) and not is_std(str(t)):
            return m.setdefault(t, URIRef(
                "http://renamed.example/" + salt + "#"
                + hashlib.sha1(str(t).encode()).hexdigest()[:10]))
        if isinstance(t, BNode):
            return m.setdefault(t, BNode())
        return t

    for s, p, o in g:
        out.add((sub(s), sub(p), sub(o)))
    return out


def selftest(args: argparse.Namespace) -> None:
    """Two positive controls and one negative control on wl_fingerprint.

    P1  rename invariance     -- a pattern and a bijective renaming of it must
                                 receive the same fingerprint.
    P2  real copies           -- the ontologies the shipped gate removed as
                                 G2 `identical_entities` ARE pattern copies, so
                                 their fingerprint should equal the pattern's.
                                 This is the case S1 is designed to catch, and it
                                 must be caught before "S1 = 0" can be believed.
    N1  discrimination        -- distinct patterns must receive distinct
                                 fingerprints (no collisions from the erasure).
    """
    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    # ---- P1 / N1 over distinct pattern files drawn from the whole corpus
    fps: dict[str, str] = {}
    p1_ok = p1_n = 0
    collisions: list[tuple[str, str]] = []
    seen_sha: set[str] = set()
    for r in manifest:
        for pf in pattern_files(Path(r["path"])):
            if p1_n >= args.n_patterns:
                break
            if pf.stat().st_size > 2 * 1024 * 1024:
                continue
            h = sha256(pf)
            if h in seen_sha:
                continue
            seen_sha.add(h)
            g, _ = parse_graph(pf)
            if g is None:
                continue
            g, _ = strip_annotations(g)
            fp = wl_fingerprint(g)
            if not fp or fp == "empty":
                continue
            if fp in fps and fps[fp] != pf.stem:
                collisions.append((fps[fp], pf.stem))
            fps.setdefault(fp, pf.stem)
            p1_n += 1
            p1_ok += int(fp == wl_fingerprint(_rename_graph(g)))
        if p1_n >= args.n_patterns:
            break
    res["P1_rename_invariance"] = {"tested": p1_n, "invariant": p1_ok}
    res["N1_distinct_fingerprints"] = {
        "distinct_contents": p1_n, "distinct_fingerprints": len(fps),
        "same_fp_different_stem": collisions[:20]}

    # ---- P2 the gate's own confirmed pattern copies
    copies = [r for r in manifest if r["self_match_reason"] == "identical_entities"]
    hits = []
    for r in copies:
        d = Path(r["path"])
        m = main_ontology_file(d)
        if m is None or m.stat().st_size > 20 * 1024 * 1024:
            continue
        hg, _ = parse_graph(m)
        if hg is None:
            continue
        hg, _ = strip_annotations(hg)
        he = set(collect_entities(hg, wide=True))
        hfp = wl_fingerprint(hg)
        best = ("", "", 0, 0)
        for pf in pattern_files(d):
            pg, _ = parse_graph(pf)
            if pg is None:
                continue
            pg, _ = strip_annotations(pg)
            if set(collect_entities(pg, wide=True)) != he:
                continue
            pfp = wl_fingerprint(pg)
            best = (pf.stem, pfp, int(pfp == hfp and bool(hfp)), len(pg))
        hits.append({"ontology_id": r["ontology_id"], "pattern": best[0],
                     "host_fp": hfp, "pattern_fp": best[1],
                     "fp_equal": bool(best[2]), "n_entities": len(he),
                     "n_triples_host": len(hg), "n_triples_pattern": best[3]})
    res["P2_known_pattern_copies"] = {
        "tested": len(hits),
        "fp_equal": sum(1 for h in hits if h["fp_equal"]),
        "fp_differ": sum(1 for h in hits if not h["fp_equal"]),
    }
    with (out / "selftest_known_copies.csv").open("w", newline="",
                                                  encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ontology_id", "pattern", "n_entities",
                                           "n_triples_host", "n_triples_pattern",
                                           "fp_equal", "host_fp", "pattern_fp"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(hits)
    (out / "selftest.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("selftest")
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--n-patterns", type=int, default=200,
                   help="distinct pattern files used for the P1/N1 controls")

    s = sub.add_parser("scan")
    s.add_argument("--data-root", type=Path, default=None)
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--population", choices=["all", "g5", "lowown"], default="all",
                   help="all = every ontology that parses (G1 survivors); "
                        "g5 = the no-pattern-overlap drops; "
                        "lowown = n_entities_wide>=20 and own<=50%% of wide")
    s.add_argument("--max-mb", type=float, default=20.0,
                   help="STAGE filter only; larger files go to skipped.csv")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--resume", action="store_true")

    r = sub.add_parser("report")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--adjudication", type=Path,
                   default=Path("out/adjudication/human_sample.csv"),
                   help="human_sample.csv from the label-mode adjudication; used "
                        "to give S2 a measured precision (columns: ontology, "
                        "pattern, human_verdict)")

    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args)
    elif args.cmd == "selftest":
        selftest(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
