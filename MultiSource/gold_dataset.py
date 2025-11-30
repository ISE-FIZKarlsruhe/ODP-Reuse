#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gold_dataset.py — Build reuse-percentage filtered datasets with rich usage statistics.

What it does
------------
1) Scans repos under --base. Each repo has one main ontology file and patterns/ with *.owl|*.ttl.
2) For each (ontology, pattern) pair, computes:
   - reused_total_graph      (exact IRI occurrences)
   - reused_total_label      (per --sim-backend)
   - label_reuse_pct         (reused_total_label / pattern_total * 100)
   - connectivity + usage details (direct/sub/super/subproperty/superproperty/via_property/restriction/equivalent/import-only/isolated)
   - connected property role stats (functional, inverse-functional, transitive, symmetric, asymmetric, reflexive, irreflexive)
3) For each P in --dataset-percents, materializes all pairs with label_reuse_pct ≥ P into:
   /out-dir/<P>_<backend>/{ontology,patterns}/   (symlink or copy)
4) Writes per-dataset statistics CSVs under /out-dir/<P>_<backend>/statistics/:
   - ontologies_overall_counts.csv
   - additional_entities_per_pair.csv
   - constructs_presence_per_ontology.csv (+ totals + DL complexity)
   - constructs_presence_per_pattern.csv (+ DL complexity)
   - pattern_usage_stats.csv
   - pattern_usage_stats_avg.csv  (column-wise means)
   - usage_forms_distribution.csv (counts + % of reuse forms)
   - totals_summary.csv (+ LaTeX)
   - histograms of "additional_*" if requested via --hist-bins

Plus global statistics (over all pairs, independent of thresholds) in:
   /out-dir/global_statistics/
   - reuse_extent_per_ontology.csv
   - reuse_extent_summary.csv (+ LaTeX)
   - pattern_reuse_stats.csv
   - pattern_reuse_top10.tex
   - dl_complexity_allpairs/statistics/* (same as per-dataset stats but for all pairs)

Similarity backends:
  --sim-backend {hybrid,char,word,sbert,exact}
    - exact  : normalized-label equality only
    - char   : cosine over char n-grams (n set by --ngram)
    - word   : cosine over word tokens
    - sbert  : Sentence-Transformers cosine (normalized embeddings)
    - hybrid : char fast path + (optional) SBERT refine near threshold
"""

import os
import sys
import re
import glob
import json
import shutil
import math
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional, Iterable
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal, Namespace

# SBERT (optional)
_SBERT_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _SBERT_AVAILABLE = False

# Quiet rdflib's noisy literal-casting warnings (dateTime/decimal, etc.)
logging.getLogger("rdflib").setLevel(logging.ERROR)
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

# ----------------------------
# Defaults
# ----------------------------

ONTOLOGY_EXTS = ("*.owl", "*.ttl", "*.rdf", "*.xml", "*.nt", "*.trig")
PATTERN_EXTS  = ("*.owl", "*.ttl")

DEFAULT_MAX_FILE_BYTES = 1_000_000_000
DEFAULT_SKIP_SUBSTRS = {""}

DEFAULT_OUT_CSV = "all_pairs_metrics.csv"
DEFAULT_OUT_DIR = "./gold_out"

DEFAULT_LABEL_SIM = 0.90
DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SBERT_DEVICE = "cuda"
DEFAULT_SBERT_PREFILTER = 0.85

DEFAULT_DATASET_PERCENTS = "100,95,90,85,80"
DEFAULT_HIST_BINS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

RDFNS  = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFSNS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWLNS  = Namespace("http://www.w3.org/2002/07/owl#")

# ----------------------------
# Simple LaTeX table helper
# ----------------------------

def write_latex_table(df: pd.DataFrame,
                      path: Path,
                      caption: str = "",
                      label: str = "") -> None:
    """
    Write a simple LaTeX table (table+tabular) from a DataFrame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(df.to_latex(index=False, escape=True))
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")

# ----------------------------
# File discovery & parsing
# ----------------------------

def should_skip(path: str, max_bytes: int, skip_subs: Set[str]) -> bool:
    try:
        if os.path.getsize(path) > max_bytes:
            return True
    except OSError:
        pass
    name = Path(path).name.lower()
    return any(s in name for s in skip_subs)

def try_parse_any(path: str) -> Optional[Graph]:
    g = Graph()
    for fmt in (None, "xml", "turtle", "n3", "nt", "trig", "json-ld"):
        try:
            g.parse(path, format=fmt)
            return g
        except Exception:
            continue
    return None

def get_main_ontology_file(repo_dir: str,
                           max_bytes: int,
                           skip_subs: Set[str]) -> Optional[str]:
    candidates: List[str] = []
    for ext in ONTOLOGY_EXTS:
        candidates.extend(glob.glob(os.path.join(repo_dir, ext)))
    candidates = [p for p in candidates if not should_skip(p, max_bytes, skip_subs)]
    if not candidates:
        return None

    def ext_rank(p: str) -> int:
        e = Path(p).suffix.lower()
        if e == ".owl": return 0
        if e == ".ttl": return 1
        if e in (".rdf", ".xml"): return 2
        return 3

    # choose by (rank asc, size desc)
    return max(candidates, key=lambda p: (- (3 - ext_rank(p)), os.path.getsize(p)))

def list_pattern_files(repo_dir: str,
                       max_bytes: int,
                       skip_subs: Set[str]) -> List[str]:
    patt_dir = os.path.join(repo_dir, "patterns")
    if not os.path.isdir(patt_dir):
        return []
    files: List[str] = []
    for ext in PATTERN_EXTS:
        files.extend(glob.glob(os.path.join(patt_dir, ext)))
    return sorted([p for p in files if not should_skip(p, max_bytes, skip_subs)])

# ----------------------------
# Entity extraction & labels
# ----------------------------

def collect_entities(g: Graph) -> Tuple[Set[str], Set[str], Set[str]]:
    classes = {str(s) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    classes |= {str(s) for s in g.subjects(RDF.type, RDFS.Class) if isinstance(s, URIRef)}
    obj_props = {str(s) for s in g.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
    data_props = {str(s) for s in g.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)}
    return classes, obj_props, data_props

def any_occurs(g: Graph, iri: str) -> bool:
    u = URIRef(iri)
    return (u, None, None) in g or (None, u, None) in g or (None, None, u) in g

def localname(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]

def normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def extract_labels(g: Graph, iris: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for i in iris:
        labels: Set[str] = set()
        subj = URIRef(i)
        for _, _, lit in g.triples((subj, RDFS.label, None)):
            if isinstance(lit, Literal):
                labels.add(normalize_label(str(lit)))
        labels.add(normalize_label(localname(i)))
        labels.discard("")
        out[i] = labels or {normalize_label(localname(i))}
    return out

# ----------------------------
# String / token similarities
# ----------------------------

def char_ngrams(s: str, n: int = 3) -> Counter:
    s = normalize_label(s)
    if len(s) < n:
        return Counter({s: 1}) if s else Counter()
    grams = [s[i:i+n] for i in range(len(s) - n + 1)]
    return Counter(grams)

def word_tokens(s: str) -> Counter:
    s = normalize_label(s)
    toks = [t for t in re.split(r"\W+", s) if t]
    return Counter(toks)

def cosine_from_counters(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def char_label_sim(a: str, b: str, ngram: int = 3) -> float:
    return cosine_from_counters(char_ngrams(a, n=ngram), char_ngrams(b, n=ngram))

def word_label_sim(a: str, b: str) -> float:
    return cosine_from_counters(word_tokens(a), word_tokens(b))

# ----------------------------
# SBERT backend
# ----------------------------

class SbertBackend:
    def __init__(self, model_name: str, device: str = "cpu"):
        if not _SBERT_AVAILABLE:
            raise RuntimeError("sentence-transformers is not installed. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, texts: List[str]) -> np.ndarray:
        normed = [normalize_label(t) for t in texts]
        vecs: List[Optional[np.ndarray]] = []
        to_idx, to_txt = [], []
        for i, t in enumerate(normed):
            if t in self.cache:
                vecs.append(self.cache[t])
            else:
                vecs.append(None); to_idx.append(i); to_txt.append(t)
        if to_txt:
            enc = self.model.encode(to_txt, convert_to_numpy=True, normalize_embeddings=True)
            for j, i in enumerate(to_idx):
                self.cache[normed[i]] = enc[j]
                vecs[i] = enc[j]
        return np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)

    def max_cosine(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        A = self.embed(a); B = self.embed(b)
        sims = A @ B.T
        return float(np.max(sims))

# ----------------------------
# DL Expressivity (construct checks)
# ----------------------------

OWL = OWLNS
RDFS_ = RDFSNS
RDF_ = RDFNS

def _any(g, s=None, p=None, o=None, cond=None):
    for (ss, pp, oo) in g.triples((s, p, o)):
        if cond is None or cond(ss, pp, oo):
            return True
    return False

def _is_uri(x): return isinstance(x, URIRef)
def _is_lit(x): return isinstance(x, Literal)

def has_construct_ROLE_INVERSE(g: Graph) -> bool:
    return (
        _any(g, None, OWL.inverseOf, None) or
        _any(g, None, RDF_.type, OWL.SymmetricProperty) or
        _any(g, None, RDF_.type, OWL.InverseFunctionalProperty)
    )

def has_construct_D(g: Graph) -> bool:
    if _any(g, None, RDF_.type, OWL.DatatypeProperty): return True
    if _any(g, None, OWL.datatypeComplementOf, None):  return True
    if _any(g, None, OWL.oneOf, None):
        for _x, _p, coll in g.triples((None, OWL.oneOf, None)):
            if _any(g, coll, RDF_.first, None, cond=lambda s,p,o: _is_lit(o)):
                return True
    if _any(g, None, OWL.onDatatype, None) and _any(g, None, OWL.withRestrictions, None): return True
    if _any(g, None, OWL.hasValue, None, cond=lambda s,p,o: _is_lit(o)): return True
    return False

def has_construct_CONCEPT_INTERSECTION(g: Graph) -> bool:
    return _any(g, None, OWL.intersectionOf, None)

def has_construct_CONCEPT_UNION(g: Graph) -> bool:
    if _any(g, None, OWL.unionOf, None): return True
    for _x, _p, coll in g.triples((None, OWL.oneOf, None)):
        if _any(g, coll, RDF_.first, None, cond=lambda s,p,o: _is_uri(o)):
            return True
    if _any(g, None, OWL.differentFrom, None): return True
    if _any(g, None, RDF_.type, OWL.AllDifferent): return True
    if _any(g, None, OWL.disjointUnionOf, None): return True
    return False

def has_construct_CONCEPT_COMPLEX_NEGATION(g: Graph) -> bool:
    return (
        _any(g, None, OWL.complementOf, None) or
        _any(g, None, OWL.disjointWith, None) or
        _any(g, None, RDF_.type, OWL.AllDisjointClasses) or
        _any(g, None, OWL.differentFrom, None) or
        _any(g, None, RDF_.type, OWL.AllDifferent) or
        _any(g, None, OWL.disjointUnionOf, None)
    )

def has_construct_FULL_EXISTENTIAL(g: Graph) -> bool:
    if _any(g, None, OWL.someValuesFrom, None, cond=lambda s,p,o: (o != OWL.Thing)): return True
    if _any(g, None, OWL.hasValue, None, cond=lambda s,p,o: _is_uri(o)): return True
    return False

def has_construct_LIMITED_EXISTENTIAL(g: Graph) -> bool:
    return _any(g, None, OWL.someValuesFrom, OWL.Thing)

def has_construct_UNIVERSAL_RESTRICTION(g: Graph) -> bool:
    return _any(g, None, OWL.allValuesFrom, None)

def has_construct_NOMINALS(g: Graph) -> bool:
    if _any(g, None, OWL.hasValue, None, cond=lambda s,p,o: _is_uri(o)): return True
    for _x, _p, coll in g.triples((None, OWL.oneOf, None)):
        if _any(g, coll, RDF_.first, None, cond=lambda s,p,o: _is_uri(o)):
            return True
    if _any(g, None, OWL.differentFrom, None): return True
    if _any(g, None, RDF_.type, OWL.AllDifferent): return True
    if _any(g, None, OWL.sameAs, None): return True
    return False

def has_construct_Q(g: Graph) -> bool:
    if _any(g, None, OWL.minQualifiedCardinality, None) and (
        _any(g, None, OWL.onClass, None, cond=lambda s,p,o: o != OWL.Thing) or
        _any(g, None, OWL.onDataRange, None, cond=lambda s,p,o: o != RDFS.Literal)
    ): return True
    if _any(g, None, OWL.qualifiedCardinality, None) and (
        _any(g, None, OWL.onClass, None, cond=lambda s,p,o: o != OWL.Thing) or
        _any(g, None, OWL.onDataRange, None, cond=lambda s,p,o: o != RDFS.Literal)
    ): return True
    if _any(g, None, OWL.maxQualifiedCardinality, None) and (
        _any(g, None, OWL.onClass, None, cond=lambda s,p,o: o != OWL.Thing) or
        _any(g, None, OWL.onDataRange, None, cond=lambda s,p,o: o != RDFS.Literal)
    ): return True
    return False

def has_construct_N(g: Graph) -> bool:
    return (
        _any(g, None, OWL.minCardinality, None) or
        _any(g, None, OWL.cardinality, None) or
        _any(g, None, OWL.maxCardinality, None) or
        (_any(g, None, OWL.minQualifiedCardinality, None) and (_any(g, None, OWL.onClass, OWL.Thing) or _any(g, None, OWL.onDataRange, RDFS.Literal))) or
        (_any(g, None, OWL.qualifiedCardinality, None) and (_any(g, None, OWL.onClass, OWL.Thing) or _any(g, None, OWL.onDataRange, RDFS.Literal))) or
        (_any(g, None, OWL.maxQualifiedCardinality, None) and (_any(g, None, OWL.onClass, OWL.Thing) or _any(g, None, OWL.onDataRange, RDFS.Literal)))
    )

def has_construct_ROLE_COMPLEX(g: Graph) -> bool:
    return (
        _any(g, None, OWL.hasSelf, None) or
        _any(g, None, RDF_.type, OWL.AsymmetricProperty) or
        _any(g, None, OWL.propertyDisjointWith, None) or
        _any(g, None, RDF_.type, OWL.AllDisjointProperties) or
        _any(g, None, RDF_.type, OWL.IrreflexiveProperty)
    )

def has_construct_ROLE_REFLEXIVITY_CHAINS(g: Graph) -> bool:
    return _any(g, None, RDF_.type, OWL.ReflexiveProperty) or _any(g, None, OWL.propertyChainAxiom, None)

def has_construct_ROLE_DOMAIN_RANGE(g: Graph) -> bool:
    return _any(g, None, RDFS.domain, None) or _any(g, None, RDFS.range, None)

def has_construct_ROLE_HIERARCHY(g: Graph) -> bool:
    return _any(g, None, OWL.equivalentProperty, None) or _any(g, None, RDFS.subPropertyOf, None)

def has_construct_F(g: Graph) -> bool:
    return _any(g, None, RDF_.type, OWL.FunctionalProperty) or _any(g, None, RDF_.type, OWL.InverseFunctionalProperty)

def has_construct_ROLE_TRANSITIVE(g: Graph) -> bool:
    return _any(g, None, RDF_.type, OWL.TransitiveProperty)

_CONSTRUCT_FUNCS = {
    "ROLE_INVERSE": has_construct_ROLE_INVERSE,
    "D": has_construct_D,
    "CONCEPT_INTERSECTION": has_construct_CONCEPT_INTERSECTION,
    "CONCEPT_UNION": has_construct_CONCEPT_UNION,
    "CONCEPT_COMPLEX_NEGATION": has_construct_CONCEPT_COMPLEX_NEGATION,
    "FULL_EXISTENTIAL": has_construct_FULL_EXISTENTIAL,
    "LIMITED_EXISTENTIAL": has_construct_LIMITED_EXISTENTIAL,
    "UNIVERSAL_RESTRICTION": has_construct_UNIVERSAL_RESTRICTION,
    "NOMINALS": has_construct_NOMINALS,
    "Q": has_construct_Q,
    "N": has_construct_N,
    "ROLE_COMPLEX": has_construct_ROLE_COMPLEX,
    "ROLE_REFLEXIVITY_CHAINS": has_construct_ROLE_REFLEXIVITY_CHAINS,
    "ROLE_DOMAIN_RANGE": has_construct_ROLE_DOMAIN_RANGE,
    "ROLE_HIERARCHY": has_construct_ROLE_HIERARCHY,
    "F": has_construct_F,
    "ROLE_TRANSITIVE": has_construct_ROLE_TRANSITIVE,
}
CONSTRUCT_NAMES = list(_CONSTRUCT_FUNCS.keys())

def has_construct(g: Graph, construct: str) -> bool:
    fn = _CONSTRUCT_FUNCS.get(construct)
    return bool(fn and fn(g))

def infer_expressivity_from_graph(g: Graph) -> str:
    """
    Map the detected OWL constructs to a *small, fixed* set of
    well-known DL names:

        EL, ALC, ALCHIQ, SHOIN, SROIQ

    This is a heuristic and deliberately coarse: we only care about
    putting each ontology into one of these buckets.
    """
    flags = {c: has_construct(g, c) for c in CONSTRUCT_NAMES}

    # "Heavy" boolean stuff: full negation, disjunction, or nominals
    has_heavy_bool = (
        flags.get("CONCEPT_COMPLEX_NEGATION", False)
        or flags.get("CONCEPT_UNION", False)
        or flags.get("NOMINALS", False)
    )

    # --- 1) EL vs "ALC and above" ---
    # If we don't see heavy boolean constructs, treat this as EL.
    if not has_heavy_bool:
        return "EL"

    # From here on we are at least ALC

    # Feature flags (very coarse approximations)
    hasH = flags.get("ROLE_HIERARCHY", False)
    hasS = flags.get("ROLE_TRANSITIVE", False)  # S = transitive roles
    hasR = flags.get("ROLE_COMPLEX", False) or flags.get("ROLE_REFLEXIVITY_CHAINS", False)
    hasI = flags.get("ROLE_INVERSE", False)
    hasQ = flags.get("Q", False) or flags.get("N", False)  # any kind of cardinality restriction
    hasO = flags.get("NOMINALS", False)

    # --- 2) Strongest profile first: SROIQ ---
    # SROIQ ≈ S + R + (O or I or Q)
    if hasS and hasR and (hasO or hasI or hasQ):
        return "SROIQ"

    # --- 3) Next: SHOIN ---
    # SHOIN ≈ S + H + O + (N/Q or I)
    if hasS and hasH and hasO and (flags.get("N", False) or hasQ or hasI):
        return "SHOIN"

    # --- 4) Next: ALCHIQ ---
    # ALCHIQ ≈ ALC + H + I + Q
    if hasH and hasI and hasQ:
        return "ALCHIQ"

    # --- 5) Otherwise: plain ALC ---
    return "ALC"

# ----------------------------
# Histogram helper
# ----------------------------

def histogram_counts(values: Iterable[int], bins: List[int]) -> Dict[str, int]:
    counts = Counter()
    vs = list(values)
    if not vs:
        return {}
    for v in vs:
        placed = False
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                key = f"{bins[i]}–{bins[i+1]-1}"
                counts[key] += 1
                placed = True
                break
        if not placed:
            counts[f"{bins[-1]}+"] += 1
    return dict(counts)

# ----------------------------
# Coverage + connectivity + usage (with DIRECT = structure-preserving)
# ----------------------------

def coverage_and_connectivity(onto_g: Optional[Graph],
                              pat_g: Optional[Graph],
                              label_thresh: float,
                              sim_backend: str,
                              ngram: int,
                              sbert: Optional[SbertBackend],
                              sbert_prefilter: float) -> Dict[str, float | int | bool | dict]:
    if onto_g is None or pat_g is None:
        return dict(
            pattern_total=0,
            reused_total_graph=0,
            reused_total_label=0,
            coverage_graph=0.0,
            coverage_label=0.0,
            connectivity_ratio=0.0,
            usage_details={
                "direct": 0, "as_subclass": 0, "as_superclass": 0,
                "as_subproperty": 0, "as_superproperty": 0,
                "via_property": 0, "via_restriction": 0, "equivalent": 0,
                "import_only": False, "isolated": True,
                "connected_props_total": 0,
                "connected_props_functional": 0,
                "connected_props_inverse_functional": 0,
                "connected_props_transitive": 0,
                "connected_props_symmetric": 0,
                "connected_props_asymmetric": 0,
                "connected_props_reflexive": 0,
                "connected_props_irreflexive": 0,
            }
        )

    p_cls, p_obj, p_data = collect_entities(pat_g)
    pattern_entities: Set[str] = p_cls | p_obj | p_data
    patt_total = len(pattern_entities)

    def base_ns(i: str) -> str:
        return i.rsplit("#", 1)[0] + ("#" if "#" in i else "")

    patt_namespaces = {base_ns(i) for i in pattern_entities}

    def is_pattern_iri(u: URIRef) -> bool:
        su = str(u)
        for ns in patt_namespaces:
            if su.startswith(ns):
                return True
        return False

    # GRAPH reuse set
    reused_graph = {i for i in pattern_entities if any_occurs(onto_g, i)}

    # Label reuse
    o_cls, o_obj, o_data = collect_entities(onto_g)
    onto_entities = o_cls | o_obj | o_data
    o_labels_map = extract_labels(onto_g, onto_entities)
    p_labels_map = extract_labels(pat_g, pattern_entities)
    onto_label_flat = sorted({lab for labs in o_labels_map.values() for lab in labs})

    reused_label: Set[str] = set()
    for i, pls in p_labels_map.items():
        if i in reused_graph:
            reused_label.add(i)
            continue
        if set(pls) & set(onto_label_flat):
            reused_label.add(i)
            continue

        matched = False
        if sim_backend == "char":
            for pl in pls:
                if any(char_label_sim(pl, ol, ngram) >= label_thresh for ol in onto_label_flat):
                    matched = True; break
        elif sim_backend == "word":
            for pl in pls:
                if any(word_label_sim(pl, ol) >= label_thresh for ol in onto_label_flat):
                    matched = True; break
        elif sim_backend == "sbert":
            assert sbert is not None, "SBERT backend requested but not initialized"
            if sbert.max_cosine(list(pls), onto_label_flat) >= label_thresh:
                matched = True
        elif sim_backend == "exact":
            matched = False
        else:  # hybrid
            for pl in pls:
                if any(char_label_sim(pl, ol, ngram) >= label_thresh for ol in onto_label_flat):
                    matched = True; break
            if not matched and sbert is not None:
                hi_char = 0.0
                for pl in pls:
                    for ol in onto_label_flat:
                        hi_char = max(hi_char, char_label_sim(pl, ol, ngram))
                if hi_char >= (sbert_prefilter - 0.05) and hi_char < label_thresh:
                    if sbert.max_cosine(list(pls), onto_label_flat) >= label_thresh:
                        matched = True
        if matched:
            reused_label.add(i)

    # Pattern-internal triples for DIRECT (structure-preserving)
    pattern_internal_triples: List[Tuple[URIRef, URIRef, URIRef]] = []
    for s, p, o in pat_g.triples((None, None, None)):
        if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
            if is_pattern_iri(s) and is_pattern_iri(p) and is_pattern_iri(o):
                pattern_internal_triples.append((s, p, o))

    triples_by_entity: Dict[str, List[Tuple[URIRef, URIRef, URIRef]]] = defaultdict(list)
    for (s, p, o) in pattern_internal_triples:
        triples_by_entity[str(s)].append((s, p, o))
        triples_by_entity[str(p)].append((s, p, o))
        triples_by_entity[str(o)].append((s, p, o))

    direct_structural: Set[str] = set()
    for e in pattern_entities:
        its = triples_by_entity.get(e, [])
        if not its:
            if e in reused_graph:
                direct_structural.add(e)
            continue
        ok = True
        for (s, p, o) in its:
            if (s, p, o) not in onto_g:
                ok = False
                break
        if ok:
            direct_structural.add(e)

    # Connectivity ratio
    reused_connected = 0
    for ent in reused_graph:
        u = URIRef(ent)
        connected = False
        for s, p, o in onto_g.triples((u, None, None)):
            if isinstance(p, URIRef) and not is_pattern_iri(p): connected = True
            if isinstance(o, URIRef) and not is_pattern_iri(o): connected = True
            if connected: break
        if not connected:
            for s, p, o in onto_g.triples((None, u, None)):
                if isinstance(s, URIRef) and not is_pattern_iri(s): connected = True
                if isinstance(o, URIRef) and not is_pattern_iri(o): connected = True
                if connected: break
        if not connected:
            for s, p, o in onto_g.triples((None, None, u)):
                if isinstance(s, URIRef) and not is_pattern_iri(s): connected = True
                if isinstance(p, URIRef) and not is_pattern_iri(p): connected = True
                if connected: break
        if connected:
            reused_connected += 1
    connectivity_ratio = (reused_connected / len(reused_graph)) if reused_graph else 0.0

    # Sub/Superclass
    as_sub_set, as_sup_set = set(), set()
    for s, _, o in onto_g.triples((None, RDFS.subClassOf, None)):
        if isinstance(o, URIRef) and str(o) in p_cls:
            as_sub_set.add(str(o))
        if isinstance(s, URIRef) and str(s) in p_cls:
            if isinstance(o, URIRef) and not is_pattern_iri(o):
                as_sup_set.add(str(s))

    # Sub/Superproperty
    as_subproperty_set, as_superproperty_set = set(), set()
    patt_props = p_obj | p_data
    for s, _, o in onto_g.triples((None, RDFS.subPropertyOf, None)):
        if isinstance(o, URIRef) and str(o) in patt_props:
            as_subproperty_set.add(str(o))  # pattern property reused as superproperty
        if isinstance(s, URIRef) and str(s) in patt_props:
            if isinstance(o, URIRef) and not is_pattern_iri(o):
                as_superproperty_set.add(str(s))  # pattern property reused as subproperty of ext

    # Via property (pattern property connecting to external IRIs)
    via_prop_set = set()
    for s, p, o in onto_g.triples((None, None, None)):
        if isinstance(p, URIRef) and str(p) in patt_props:
            if (isinstance(s, URIRef) and not is_pattern_iri(s)) or (isinstance(o, URIRef) and not is_pattern_iri(o)):
                via_prop_set.add(str(p))

    # Via restriction (pattern entities used in OWL restrictions)
    via_restrict_set = set()
    RESTR_PREDICATES = {
        OWL.onProperty, OWL.someValuesFrom, OWL.allValuesFrom,
        OWL.hasValue, OWL.onClass, OWL.onDataRange
    }
    for s, p, o in onto_g.triples((None, None, None)):
        if p in RESTR_PREDICATES and isinstance(o, URIRef):
            so = str(o)
            if so in pattern_entities:
                via_restrict_set.add(so)

    # Equivalent (class or property)
    equiv_set = set()
    for s, _, o in onto_g.triples((None, OWL.equivalentClass, None)):
        if isinstance(s, URIRef) and is_pattern_iri(s) and isinstance(o, URIRef) and not is_pattern_iri(o):
            equiv_set.add(str(s))
        if isinstance(o, URIRef) and is_pattern_iri(o) and isinstance(s, URIRef) and not is_pattern_iri(s):
            equiv_set.add(str(o))
    for s, _, o in onto_g.triples((None, OWL.equivalentProperty, None)):
        if isinstance(s, URIRef) and is_pattern_iri(s) and isinstance(o, URIRef) and not is_pattern_iri(o):
            equiv_set.add(str(s))
        if isinstance(o, URIRef) and is_pattern_iri(o) and isinstance(s, URIRef) and not is_pattern_iri(s):
            equiv_set.add(str(o))

    # Imports-only vs any use
    imports_pattern = False
    for _, _, imported in onto_g.triples((None, OWL.imports, None)):
        if isinstance(imported, URIRef):
            uri = str(imported)
            if any(uri.startswith(ns.rstrip("#/")) for ns in patt_namespaces):
                imports_pattern = True
                break
    any_use = (
        bool(direct_structural) or bool(as_sub_set) or bool(as_sup_set) or
        bool(as_subproperty_set) or bool(as_superproperty_set) or
        bool(via_prop_set) or bool(via_restrict_set) or bool(equiv_set)
    )
    import_only = imports_pattern and not any_use

    # Isolated?
    isolated = True
    for s, p, o in onto_g.triples((None, None, None)):
        sp = isinstance(s, URIRef) and is_pattern_iri(s)
        pp = isinstance(p, URIRef) and is_pattern_iri(p)
        op = isinstance(o, URIRef) and is_pattern_iri(o)
        if sp and ((isinstance(p, URIRef) and not is_pattern_iri(p)) or (isinstance(o, URIRef) and not is_pattern_iri(o))):
            isolated = False; break
        if pp and ((isinstance(s, URIRef) and not is_pattern_iri(s)) or (isinstance(o, URIRef) and not is_pattern_iri(o))):
            isolated = False; break
        if op and ((isinstance(s, URIRef) and not is_pattern_iri(s)) or (isinstance(p, URIRef) and not is_pattern_iri(p))):
            isolated = False; break

    # Connected property role stats (on actually-connected pattern properties)
    def _count_prop_type(props: Set[str], type_iri: URIRef) -> int:
        c = 0
        for p in props:
            u = URIRef(p)
            if (u, RDF.type, type_iri) in onto_g:
                c += 1
        return c

    connected_props_total = len(via_prop_set)
    connected_props_functional           = _count_prop_type(via_prop_set, OWL.FunctionalProperty)
    connected_props_inverse_functional   = _count_prop_type(via_prop_set, OWL.InverseFunctionalProperty)
    connected_props_transitive           = _count_prop_type(via_prop_set, OWL.TransitiveProperty)
    connected_props_symmetric            = _count_prop_type(via_prop_set, OWL.SymmetricProperty)
    connected_props_asymmetric           = _count_prop_type(via_prop_set, OWL.AsymmetricProperty)
    connected_props_reflexive            = _count_prop_type(via_prop_set, OWL.ReflexiveProperty)
    connected_props_irreflexive          = _count_prop_type(via_prop_set, OWL.IrreflexiveProperty)

    usage_details = {
        "direct":          int(len(direct_structural)),
        "as_subclass":     int(len(as_sub_set)),
        "as_superclass":   int(len(as_sup_set)),
        "as_subproperty":  int(len(as_subproperty_set)),
        "as_superproperty":int(len(as_superproperty_set)),
        "via_property":    int(len(via_prop_set)),
        "via_restriction": int(len(via_restrict_set)),
        "equivalent":      int(len(equiv_set)),
        "import_only":     bool(import_only),
        "isolated":        bool(isolated),
        "connected_props_total":                int(connected_props_total),
        "connected_props_functional":           int(connected_props_functional),
        "connected_props_inverse_functional":   int(connected_props_inverse_functional),
        "connected_props_transitive":           int(connected_props_transitive),
        "connected_props_symmetric":            int(connected_props_symmetric),
        "connected_props_asymmetric":           int(connected_props_asymmetric),
        "connected_props_reflexive":            int(connected_props_reflexive),
        "connected_props_irreflexive":          int(connected_props_irreflexive),
    }

    return dict(
        pattern_total=patt_total,
        reused_total_graph=len(reused_graph),
        reused_total_label=len(reused_label),
        coverage_graph=(len(reused_graph) / patt_total) if patt_total else 0.0,
        coverage_label=(len(reused_label) / patt_total) if patt_total else 0.0,
        connectivity_ratio=connectivity_ratio,
        usage_details=usage_details
    )

# ----------------------------
# Stats helpers
# ----------------------------

def count_ontology_entities(g: Graph) -> Tuple[int, int, int]:
    c, o, d = collect_entities(g)
    return len(c), len(o), len(d)

def additional_counts(onto_g: Graph, patt_g: Graph) -> Tuple[int, int, int]:
    oc, oo, od = collect_entities(onto_g)
    pc, po, pd = collect_entities(patt_g)
    return max(len(oc - pc), 0), max(len(oo - po), 0), max(len(od - pd), 0)

# ----------------------------
# FS helpers
# ----------------------------

def _safe_symlink(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.symlink_to(src)
        return True
    except Exception:
        return False

def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# ----------------------------
# Write per-dataset statistics (including pattern + ontology DL complexity, usage %, LaTeX)
# ----------------------------

def _write_dataset_stats(subset_df: pd.DataFrame,
                         ds_root: Path,
                         label_thresh: float,
                         sim_backend: str,
                         ngram: int,
                         sbert: Optional[SbertBackend],
                         sbert_prefilter: float,
                         hist_bins: List[int]):
    stats_dir = ds_root / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    onto_counts_rows: List[Dict[str, object]] = []
    additional_rows: List[Dict[str, object]] = []
    construct_rows: List[Dict[str, object]] = []
    usage_rows: List[Dict[str, object]] = []
    pattern_construct_rows: List[Dict[str, object]] = []
    seen_patterns: Set[str] = set()

    # Caches to avoid reparsing graphs
    onto_graphs: Dict[str, Graph] = {}
    pattern_graphs: Dict[str, Graph] = {}

    # --------------------------
    # Ontology-level statistics
    # --------------------------
    unique_onts = subset_df[["repo", "ontology_file"]].drop_duplicates()

    for _, row in unique_onts.iterrows():
        repo = row["repo"]
        onto_file = row["ontology_file"]
        og = try_parse_any(onto_file)
        if og is None:
            continue
        onto_graphs[onto_file] = og

        oc, oo, od = count_ontology_entities(og)
        n_axioms = len(og)  # triples ≈ axioms

        # Compute construct flags once
        construct_flags = {c: bool(has_construct(og, c)) for c in CONSTRUCT_NAMES}
        expr = infer_expressivity_from_graph(og)

        onto_counts_rows.append({
            "repo": repo,
            "ontology_file": onto_file,
            "n_classes": oc,
            "n_object_properties": oo,
            "n_datatype_properties": od,
            "n_axioms": n_axioms,
            "expressivity": expr,
        })

        row_construct = {
            "repo": repo,
            "ontology_file": onto_file,
            "n_axioms": n_axioms,
            "expressivity": expr,
        }
        row_construct.update(construct_flags)
        construct_rows.append(row_construct)

    # --------------------------------------------
    # Pair-level stats: additional entities + usage
    # --------------------------------------------
    for rr in subset_df.itertuples():
        repo = rr.repo
        onto_file = rr.ontology_file
        patt_file = rr.pattern_file
        patt_name = rr.pattern_name

        og = onto_graphs.get(onto_file)
        if og is None:
            # Ontology failed to parse earlier
            continue

        # Load pattern graph with caching
        pg = pattern_graphs.get(patt_file)
        if pg is None:
            pg = try_parse_any(patt_file)
            if pg is None:
                continue
            pattern_graphs[patt_file] = pg

        # Additional entities (ontology minus pattern)
        add_c, add_o, add_d = additional_counts(og, pg)
        additional_rows.append({
            "repo": repo,
            "pattern_name": patt_name,
            "ontology_file": onto_file,
            "pattern_file": patt_file,
            "additional_classes": add_c,
            "additional_object_properties": add_o,
            "additional_datatype_properties": add_d
        })

        # Pattern-level constructs (once per unique pattern file)
        if patt_file not in seen_patterns:
            seen_patterns.add(patt_file)
            p_construct_flags = {c: bool(has_construct(pg, c)) for c in CONSTRUCT_NAMES}
            p_expr = infer_expressivity_from_graph(pg)

            prow = {
                "repo": repo,
                "pattern_name": patt_name,
                "pattern_file": patt_file,
                "expressivity": p_expr,
            }
            prow.update(p_construct_flags)
            pattern_construct_rows.append(prow)

        # Usage rows (reuse forms and connectivity) – these come from subset_df columns
        usage_rows.append({
            "repo": repo,
            "pattern_name": patt_name,
            "usage_direct": rr.usage_direct,
            "usage_as_subclass": rr.usage_as_subclass,
            "usage_as_superclass": rr.usage_as_superclass,
            "usage_as_subproperty": rr.usage_as_subproperty,
            "usage_as_superproperty": rr.usage_as_superproperty,
            "usage_via_property": rr.usage_via_property,
            "usage_via_restriction": rr.usage_via_restriction,
            "usage_equivalent": rr.usage_equivalent,
            "usage_import_only": int(getattr(rr, "usage_import_only", 0)),
            "usage_isolated": int(getattr(rr, "usage_isolated", 0)),
            "connected_props_total": rr.connected_props_total,
            "connected_props_functional": rr.connected_props_functional,
            "connected_props_inverse_functional": rr.connected_props_inverse_functional,
            "connected_props_transitive": rr.connected_props_transitive,
            "connected_props_symmetric": rr.connected_props_symmetric,
            "connected_props_asymmetric": rr.connected_props_asymmetric,
            "connected_props_reflexive": rr.connected_props_reflexive,
            "connected_props_irreflexive": rr.connected_props_irreflexive,
            "connectivity_ratio": rr.connectivity_ratio
        })

    # --------------------------
    # Build DataFrames
    # --------------------------
    df_onto_counts = pd.DataFrame(
        onto_counts_rows,
        columns=[
            "repo","ontology_file",
            "n_classes","n_object_properties","n_datatype_properties",
            "n_axioms","expressivity"
        ]
    )
    df_additional = pd.DataFrame(
        additional_rows,
        columns=["repo","pattern_name","ontology_file","pattern_file",
                 "additional_classes","additional_object_properties","additional_datatype_properties"]
    )
    df_constructs = (
        pd.DataFrame(construct_rows)
        if construct_rows else
        pd.DataFrame(columns=["repo","ontology_file","n_axioms","expressivity", *CONSTRUCT_NAMES])
    )
    df_usage = pd.DataFrame(
        usage_rows,
        columns=[
            "repo","pattern_name",
            "usage_direct","usage_as_subclass","usage_as_superclass",
            "usage_as_subproperty","usage_as_superproperty",
            "usage_via_property","usage_via_restriction","usage_equivalent",
            "usage_import_only","usage_isolated",
            "connected_props_total","connected_props_functional","connected_props_inverse_functional",
            "connected_props_transitive","connected_props_symmetric","connected_props_asymmetric",
            "connected_props_reflexive","connected_props_irreflexive",
            "connectivity_ratio"
        ]
    )
    df_pattern_constructs = (
        pd.DataFrame(pattern_construct_rows)
        if pattern_construct_rows else
        pd.DataFrame(columns=["repo","pattern_name","pattern_file","expressivity", *CONSTRUCT_NAMES])
    )

    df_onto_counts.to_csv(stats_dir / "ontologies_overall_counts.csv", index=False)
    df_additional.to_csv(stats_dir / "additional_entities_per_pair.csv", index=False)
    df_usage.to_csv(stats_dir / "pattern_usage_stats.csv", index=False)

    # --------------------------
    # DL construct complexity
    # --------------------------

    # Ontology-level constructs + complexity score
    if not df_constructs.empty:
        df_constructs["dl_complexity_score"] = df_constructs[CONSTRUCT_NAMES].sum(axis=1)
        df_constructs.to_csv(stats_dir / "constructs_presence_per_ontology.csv", index=False)

        # Summary for ontologies (counts + percentages)
        num_onts = len(df_constructs)
        construct_summary_onts = []
        for c in CONSTRUCT_NAMES:
            count_c = int(df_constructs[c].sum())
            pct_c = (100.0 * count_c / num_onts) if num_onts else 0.0
            construct_summary_onts.append({
                "construct": c,
                "num_ontologies": count_c,
                "pct_ontologies": pct_c
            })
        df_constructs_summary_onts = pd.DataFrame(construct_summary_onts)
        df_constructs_summary_onts.to_csv(
            stats_dir / "constructs_presence_summary_ontologies.csv", index=False
        )

        write_latex_table(
            df_constructs_summary_onts,
            stats_dir / "constructs_presence_summary_ontologies.tex",
            caption="DL construct presence across ontologies",
            label="tab:dl-constructs-ontologies"
        )

        # Expressivity distribution for ontologies
        if "expressivity" in df_constructs.columns:
            expr_counts = df_constructs["expressivity"].fillna("UNKNOWN").value_counts().reset_index()
            expr_counts.columns = ["expressivity", "num_ontologies"]
            total_onts = expr_counts["num_ontologies"].sum()
            expr_counts["pct_ontologies"] = 100.0 * expr_counts["num_ontologies"] / total_onts
            expr_counts.to_csv(stats_dir / "expressivity_distribution_ontologies.csv", index=False)

            write_latex_table(
                expr_counts,
                stats_dir / "expressivity_distribution_ontologies.tex",
                caption="Distribution of DL expressivity across ontologies in this dataset",
                label="tab:expressivity-ontologies"
            )
    else:
        pd.DataFrame(columns=["repo","ontology_file","n_axioms","expressivity","dl_complexity_score", *CONSTRUCT_NAMES]) \
          .to_csv(stats_dir / "constructs_presence_per_ontology.csv", index=False)

    # Pattern-level constructs + complexity score
    if not df_pattern_constructs.empty:
        # Save the per-file pattern construct table (unchanged)
        df_pattern_constructs["dl_complexity_score"] = df_pattern_constructs[CONSTRUCT_NAMES].sum(axis=1)
        df_pattern_constructs.to_csv(
            stats_dir / "constructs_presence_per_pattern.csv", index=False
        )

        # === AGGREGATE BY pattern_name FOR CONSTRUCT SUMMARY (already fixed) ===
        grp = df_pattern_constructs.groupby("pattern_name")
        num_patterns = grp.ngroups  # number of unique pattern names

        construct_summary_patts = []
        for c in CONSTRUCT_NAMES:
            pattern_has_c = grp[c].max()    # Bool per pattern
            count_c = int(pattern_has_c.sum())
            pct_c = (100.0 * count_c / num_patterns) if num_patterns else 0.0
            construct_summary_patts.append({
                "construct": c,
                "num_patterns": count_c,
                "pct_patterns": pct_c
            })

        df_constructs_summary_patts = pd.DataFrame(construct_summary_patts)
        df_constructs_summary_patts.to_csv(
            stats_dir / "constructs_presence_summary_patterns.csv", index=False
        )

        write_latex_table(
            df_constructs_summary_patts,
            stats_dir / "constructs_presence_summary_patterns.tex",
            caption="DL construct presence across patterns",
            label="tab:dl-constructs-patterns"
        )

        # === NEW: EXPRESSIVITY DISTRIBUTION OVER UNIQUE PATTERNS ===
        if "expressivity" in df_pattern_constructs.columns:
            # For each pattern_name, keep the row with the highest DL complexity score
            patt_expr = (
                df_pattern_constructs
                .sort_values("dl_complexity_score", ascending=False)
                .groupby("pattern_name", as_index=False)
                .first()[["pattern_name", "expressivity"]]
            )

            expr_counts_p = (
                patt_expr["expressivity"]
                .fillna("UNKNOWN")
                .value_counts()
                .reset_index()
            )
            expr_counts_p.columns = ["expressivity", "num_patterns"]
            total_patts = expr_counts_p["num_patterns"].sum()
            expr_counts_p["pct_patterns"] = 100.0 * expr_counts_p["num_patterns"] / total_patts

            expr_counts_p.to_csv(
                stats_dir / "expressivity_distribution_patterns.csv", index=False
            )

            write_latex_table(
                expr_counts_p,
                stats_dir / "expressivity_distribution_patterns.tex",
                caption="Distribution of DL expressivity across patterns in this dataset",
                label="tab:expressivity-patterns"
            )

    # --------------------------
    # Pattern usage averages + percentages
    # --------------------------
    if not df_usage.empty:
        num_cols = [c for c in df_usage.columns if c not in {"repo","pattern_name"}]
        avg_row = {c: float(pd.to_numeric(df_usage[c], errors="coerce").dropna().mean()) for c in num_cols}

        # usage modes we care about for percentages
        usage_mode_cols = [
            "usage_direct","usage_as_subclass","usage_as_superclass",
            "usage_as_subproperty","usage_as_superproperty",
            "usage_via_property","usage_via_restriction","usage_equivalent"
        ]

        # total usage events over the whole dataset (not averaged)
        mode_totals = {c: float(pd.to_numeric(df_usage[c], errors="coerce").dropna().sum()) for c in usage_mode_cols}
        total_usage_events = sum(mode_totals.values()) or 1.0

        # add percentage columns to the avg row: how much each form contributes overall
        for c in usage_mode_cols:
            avg_row[f"{c}_pct"] = 100.0 * mode_totals[c] / total_usage_events

        df_usage_avg = pd.DataFrame([avg_row])
        df_usage_avg.to_csv(stats_dir / "pattern_usage_stats_avg.csv", index=False)

        write_latex_table(
            df_usage_avg,
            stats_dir / "pattern_usage_stats_avg.tex",
            caption="Average usage statistics for patterns (with percentages of reuse forms)",
            label="tab:pattern-usage-avg"
        )

        # Overall distribution of usage modes (counts + percentages)
        dist_rows = []
        for c in usage_mode_cols:
            pretty = c.replace("usage_", "").replace("_", " ")
            cnt = mode_totals[c]
            pct = (100.0 * cnt / total_usage_events) if total_usage_events else 0.0
            dist_rows.append({
                "mode": pretty,
                "count": int(cnt),
                "percent": pct
            })
        df_usage_dist = pd.DataFrame(dist_rows)
        df_usage_dist.to_csv(stats_dir / "usage_forms_distribution.csv", index=False)

        # ----------------------------------------------------
        # Pattern-level distribution of reuse forms
        # (For each pattern: does this reuse form occur at least once?)
        # ----------------------------------------------------

        # usage modes to check
        usage_mode_cols = [
            "usage_direct","usage_as_subclass","usage_as_superclass",
            "usage_as_subproperty","usage_as_superproperty",
            "usage_via_property","usage_via_restriction","usage_equivalent"
        ]

        # Group by pattern, detect presence of each reuse form (>0)
        pattern_groups = df_usage.groupby("pattern_name")

        pattern_level_rows = []
        num_patterns = pattern_groups.ngroups if pattern_groups.ngroups > 0 else 1

        for c in usage_mode_cols:
            pretty = c.replace("usage_", "").replace("_", " ")
            
            # For each pattern, check if ANY pair shows positive usage
            pattern_has_mode = pattern_groups[c].sum() > 0   # returns series indexed by pattern_name
            
            pattern_count = int(pattern_has_mode.sum())  # number of patterns that exhibited this mode
            
            pattern_percent = (100.0 * pattern_count / num_patterns) if num_patterns else 0.0
            
            pattern_level_rows.append({
                "mode": pretty,
                "pattern_count": pattern_count,
                "pattern_percent": pattern_percent
            })

        df_usage_pattern_level = pd.DataFrame(pattern_level_rows)
        df_usage_pattern_level.to_csv(stats_dir / "usage_forms_distribution_pattern_level.csv", index=False)

        write_latex_table(
            df_usage_pattern_level,
            stats_dir / "usage_forms_distribution_pattern_level.tex",
            caption="Pattern-level distribution of reuse forms",
            label="tab:reuse-forms-pattern"
        )

        write_latex_table(
            df_usage_dist,
            stats_dir / "usage_forms_distribution.tex",
            caption="Distribution of pattern reuse forms",
            label="tab:reuse-forms-distribution"
        )
    else:
        pd.DataFrame([{}]).to_csv(stats_dir / "pattern_usage_stats_avg.csv", index=False)

    # --------------------------
    # Totals + histograms
    # --------------------------
    totals = {
        "total_classes": int(df_onto_counts["n_classes"].sum()) if not df_onto_counts.empty else 0,
        "total_object_properties": int(df_onto_counts["n_object_properties"].sum()) if not df_onto_counts.empty else 0,
        "total_datatype_properties": int(df_onto_counts["n_datatype_properties"].sum()) if not df_onto_counts.empty else 0,
        "total_axioms": int(df_onto_counts["n_axioms"].sum()) if not df_onto_counts.empty else 0,
        "num_ontologies": int(len(df_onto_counts)),
        "num_pairs": int(len(df_additional))
    }
    df_totals = pd.DataFrame([totals])
    df_totals.to_csv(stats_dir / "totals_summary.csv", index=False)

    write_latex_table(
        df_totals,
        stats_dir / "totals_summary.tex",
        caption="Overall totals for ontologies and ODP reuse pairs",
        label="tab:totals-summary"
    )

    if not df_additional.empty:
        h_c = histogram_counts(df_additional["additional_classes"].tolist(), hist_bins)
        h_o = histogram_counts(df_additional["additional_object_properties"].tolist(), hist_bins)
        h_d = histogram_counts(df_additional["additional_datatype_properties"].tolist(), hist_bins)
        pd.DataFrame([h_c]).to_csv(stats_dir / "hist_additional_classes.csv", index=False)
        pd.DataFrame([h_o]).to_csv(stats_dir / "hist_additional_object_properties.csv", index=False)
        pd.DataFrame([h_d]).to_csv(stats_dir / "hist_additional_datatype_properties.csv", index=False)

    return stats_dir

def build_global_statistics_for_df(df_source: pd.DataFrame,
                                   out_root: Path,
                                   label_thresh: float,
                                   sim_backend: str,
                                   ngram: int,
                                   sbert: Optional[SbertBackend],
                                   sbert_prefilter: float,
                                   hist_bins: List[int],
                                   caption_suffix: str = "") -> None:
    """
    Build a full global_statistics-style folder for a given DataFrame of pairs.

    It will create:
      - reuse_extent_per_ontology.csv + reuse_extent_summary(.tex)
      - pattern_reuse_stats.csv + top10(.tex)
      - dl_complexity_allpairs/statistics/*  (ontology + pattern constructs, DL complexity)
      - RQ3 global complexity table + expressivity vs reuse
      - per-pair complexity tables and construct breakdowns
    in: out_root/
    """

    out_root.mkdir(parents=True, exist_ok=True)

    # --------------------
    # RQ1: Extent of reuse
    # --------------------
    reuse_rows = []
    for repo, grp in df_source.groupby("repo"):
        onto_file = grp.iloc[0]["ontology_file"]
        num_patterns = grp["pattern_name"].nunique()
        num_patterns_reused_label = grp[grp["reused_total_label"] > 0]["pattern_name"].nunique()
        num_patterns_reused_graph = grp[grp["reused_total_graph"] > 0]["pattern_name"].nunique()
        max_label_reuse = float(grp["label_reuse_pct"].max())
        avg_label_reuse = float(grp["label_reuse_pct"].mean())
        any_reuse_label = bool(num_patterns_reused_graph > 0)

        reuse_rows.append({
            "repo": repo,
            "ontology_file": onto_file,
            "num_patterns": num_patterns,
            "num_patterns_with_label_reuse": num_patterns_reused_label,
            "num_patterns_with_graph_reuse": num_patterns_reused_graph,
            "any_reuse_label": int(any_reuse_label),
            "max_label_reuse_pct": max_label_reuse,
            "avg_label_reuse_pct": avg_label_reuse,
        })

    df_reuse_onts = pd.DataFrame(reuse_rows)
    df_reuse_onts.to_csv(out_root / "reuse_extent_per_ontology.csv", index=False)

    num_onts_global = len(df_reuse_onts)
    num_onts_with_reuse = int(df_reuse_onts["any_reuse_label"].sum()) if num_onts_global else 0
    pct_onts_with_reuse = (100.0 * num_onts_with_reuse / num_onts_global) if num_onts_global else 0.0
    avg_patterns_per_ont = float(df_reuse_onts["num_patterns"].mean()) if num_onts_global else 0.0
    avg_patterns_with_reuse = float(df_reuse_onts["num_patterns_with_label_reuse"].mean()) if num_onts_global else 0.0

    df_reuse_summary = pd.DataFrame([{
        "num_ontologies": num_onts_global,
        "num_ontologies_with_any_pattern_reuse": num_onts_with_reuse,
        "pct_ontologies_with_any_pattern_reuse": pct_onts_with_reuse,
        "avg_patterns_per_ontology": avg_patterns_per_ont,
        "avg_patterns_with_reuse_per_ontology": avg_patterns_with_reuse
    }])
    df_reuse_summary.to_csv(out_root / "reuse_extent_summary.csv", index=False)

    write_latex_table(
        df_reuse_summary,
        out_root / "reuse_extent_summary.tex",
        caption=f"Extent of ODP reuse across ontologies {caption_suffix}".strip(),
        label=f"tab:extent-reuse-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
    )

    # --------------------
    # RQ2: Pattern-centric
    # --------------------
    usage_cols = [
        "usage_direct","usage_as_subclass","usage_as_superclass",
        "usage_as_subproperty","usage_as_superproperty",
        "usage_via_property","usage_via_restriction","usage_equivalent"
    ]

    pattern_rows = []
    for pattern_name, grp in df_source.groupby("pattern_name"):
        num_pairs = len(grp)
        num_ontologies_using_pattern = grp[grp["reused_total_label"] > 0]["repo"].nunique()
        avg_label_reuse = float(grp["label_reuse_pct"].mean())
        max_label_reuse = float(grp["label_reuse_pct"].max())

        mode_counts = {c: int(grp[c].sum()) for c in usage_cols}
        total_mode = sum(mode_counts.values()) or 1  # avoid div-by-zero
        row = {
            "pattern_name": pattern_name,
            "num_pairs": num_pairs,
            "num_ontologies_using_pattern": num_ontologies_using_pattern,
            "avg_label_reuse_pct": avg_label_reuse,
            "max_label_reuse_pct": max_label_reuse,
        }
        for c in usage_cols:
            pretty = c.replace("usage_", "")
            row[f"count_{pretty}"] = mode_counts[c]
            row[f"pct_{pretty}"] = 100.0 * mode_counts[c] / total_mode
        pattern_rows.append(row)

    df_pattern_global = pd.DataFrame(pattern_rows)
    if not df_pattern_global.empty:
        df_pattern_global.sort_values(
            ["num_ontologies_using_pattern","avg_label_reuse_pct"],
            ascending=[False, False],
            inplace=True
        )
    df_pattern_global.to_csv(out_root / "pattern_reuse_stats.csv", index=False)

    if not df_pattern_global.empty:
        top_k = df_pattern_global.head(10)[[
            "pattern_name",
            "num_ontologies_using_pattern",
            "avg_label_reuse_pct",
            "max_label_reuse_pct"
        ]]
        write_latex_table(
            top_k,
            out_root / "pattern_reuse_top10.tex",
            caption=f"Top 10 most frequently reused patterns {caption_suffix}".strip(),
            label=f"tab:top-patterns-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
        )

    # --------------------
    # DL complexity (all pairs for this df_source)
    # --------------------
    dl_root = out_root / "dl_complexity_allpairs"
    _write_dataset_stats(
        subset_df=df_source,
        ds_root=dl_root,
        label_thresh=label_thresh,
        sim_backend=sim_backend,
        ngram=ngram,
        sbert=sbert,
        sbert_prefilter=sbert_prefilter,
        hist_bins=hist_bins
    )

    # --------------------
    # RQ3 + per-pair complexity / construct breakdown
    # --------------------
    try:
        constructs_path = dl_root / "statistics" / "constructs_presence_per_ontology.csv"
        if constructs_path.exists():
            df_constructs = pd.read_csv(constructs_path)

            # Attach reuse info (any_reuse_label) to each ontology
            df_complex_reuse = df_constructs.merge(
                df_reuse_onts[["repo", "ontology_file", "any_reuse_label"]],
                on=["repo", "ontology_file"],
                how="left"
            )
            df_complex_reuse["any_reuse_label"] = df_complex_reuse["any_reuse_label"].fillna(0).astype(int)

            no_reuse = df_complex_reuse[df_complex_reuse["any_reuse_label"] == 0]
            with_reuse = df_complex_reuse[df_complex_reuse["any_reuse_label"] == 1]

            def _safe_mean(series):
                return float(pd.to_numeric(series, errors="coerce").dropna().mean()) if not series.empty else 0.0

            avg_dl_no   = _safe_mean(no_reuse["dl_complexity_score"])
            avg_dl_with = _safe_mean(with_reuse["dl_complexity_score"])
            avg_ax_no   = _safe_mean(no_reuse["n_axioms"])
            avg_ax_with = _safe_mean(with_reuse["n_axioms"])

            def _delta_pct(base, new):
                return (100.0 * (new - base) / base) if base > 0 else 0.0

            delta_dl = _delta_pct(avg_dl_no,   avg_dl_with)
            delta_ax = _delta_pct(avg_ax_no,   avg_ax_with)

            expr_no   = no_reuse["expressivity"].mode().iloc[0] if not no_reuse.empty else "N/A"
            expr_with = with_reuse["expressivity"].mode().iloc[0] if not with_reuse.empty else "N/A"
            expr_shift = f"{expr_no} → {expr_with}"

            rq3_rows = [
                {
                    "Complexity metric": "Avg. DL constructors",
                    "Ontologies w/o reuse": round(avg_dl_no, 2),
                    "Ontologies w/ reuse": round(avg_dl_with, 2),
                    "Δ (%)": f"{delta_dl:+.1f}%",
                },
                {
                    "Complexity metric": "Avg. axioms",
                    "Ontologies w/o reuse": round(avg_ax_no, 1),
                    "Ontologies w/ reuse": round(avg_ax_with, 1),
                    "Δ (%)": f"{delta_ax:+.1f}%",
                },
                {
                    "Complexity metric": "Expressivity level",
                    "Ontologies w/o reuse": expr_no,
                    "Ontologies w/ reuse": expr_with,
                    "Δ (%)": expr_shift,
                },
            ]
            df_rq3 = pd.DataFrame(rq3_rows)
            df_rq3.to_csv(out_root / "rq3_effect_pattern_reuse_complexity.csv", index=False)

            write_latex_table(
                df_rq3,
                out_root / "rq3_effect_pattern_reuse_complexity.tex",
                caption=f"Effect of pattern reuse on ontology complexity {caption_suffix}".strip(),
                label=f"tab:rq3-complexity-effect-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
            )

            # Expressivity vs reuse
            if "expressivity" in df_complex_reuse.columns:
                expr_counts = []
                for expr_val, grp in df_complex_reuse.groupby("expressivity"):
                    expr_val = expr_val if isinstance(expr_val, str) else "UNKNOWN"
                    total = len(grp)
                    with_r = int(grp["any_reuse_label"].sum())
                    without_r = total - with_r
                    pct_with = 100.0 * with_r / total if total else 0.0
                    pct_without = 100.0 * without_r / total if total else 0.0
                    expr_counts.append({
                        "expressivity": expr_val,
                        "num_ontologies": total,
                        "num_with_reuse": with_r,
                        "num_without_reuse": without_r,
                        "pct_with_reuse": pct_with,
                        "pct_without_reuse": pct_without,
                    })
                df_expr_reuse = pd.DataFrame(expr_counts)
                df_expr_reuse.to_csv(out_root / "rq3_expressivity_vs_reuse.csv", index=False)

                write_latex_table(
                    df_expr_reuse,
                    out_root / "rq3_expressivity_vs_reuse.tex",
                    caption=f"Distribution of DL expressivity for ontologies with and without pattern reuse {caption_suffix}".strip(),
                    label=f"tab:rq3-expressivity-vs-reuse-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
                )

            # Construct-level differences
            construct_diff_rows = []
            for c in CONSTRUCT_NAMES:
                if c not in df_complex_reuse.columns:
                    continue
                p_no = float(no_reuse[c].mean()) if not no_reuse.empty else 0.0
                p_with = float(with_reuse[c].mean()) if not with_reuse.empty else 0.0
                construct_diff_rows.append({
                    "construct": c,
                    "pct_ontologies_without_reuse": 100.0 * p_no,
                    "pct_ontologies_with_reuse": 100.0 * p_with,
                    "delta_percentage_points": 100.0 * (p_with - p_no),
                })

            df_construct_diff = pd.DataFrame(construct_diff_rows)
            df_construct_diff.sort_values("delta_percentage_points", ascending=False, inplace=True)
            df_construct_diff.to_csv(out_root / "rq3_construct_differences.csv", index=False)

            top_k_constructs = df_construct_diff.head(10)
            write_latex_table(
                top_k_constructs,
                out_root / "rq3_construct_differences_top10.tex",
                caption=f"Top DL constructs whose presence increases most in ontologies with pattern reuse {caption_suffix}".strip(),
                label=f"tab:rq3-construct-differences-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
            )

        # -----------------------------
        # Per-pair complexity + breakdown
        # -----------------------------
        stats_root = dl_root / "statistics"
        ont_path = stats_root / "constructs_presence_per_ontology.csv"
        pat_path = stats_root / "constructs_presence_per_pattern.csv"

        if ont_path.exists() and pat_path.exists():
            df_ont_full = pd.read_csv(ont_path)
            df_pat_full = pd.read_csv(pat_path)

            ont_fixed = {"repo", "ontology_file", "n_axioms", "expressivity", "dl_complexity_score"}
            pat_fixed = {"repo", "pattern_file", "pattern_name", "expressivity", "dl_complexity_score"}

            ont_construct_cols = [
                c for c in df_ont_full.columns
                if c not in ont_fixed and df_ont_full[c].dtype == bool
            ]
            pat_construct_cols = [
                c for c in df_pat_full.columns
                if c not in pat_fixed and df_pat_full[c].dtype == bool
            ]
            constructs = sorted(set(ont_construct_cols) & set(pat_construct_cols))

            onto_cols = {
                "n_axioms": "onto_n_axioms",
                "expressivity": "onto_expressivity",
                "dl_complexity_score": "onto_dl_complexity",
            }
            onto_cols.update({c: f"onto_{c}" for c in constructs})
            df_ont = df_ont_full[
                ["repo", "ontology_file", "n_axioms", "expressivity", "dl_complexity_score"] + constructs
            ].rename(columns=onto_cols)

            pat_cols = {
                "expressivity": "pattern_expressivity",
                "dl_complexity_score": "pattern_dl_complexity",
            }
            pat_cols.update({c: f"pattern_{c}" for c in constructs})
            df_pat = df_pat_full[
                ["repo", "pattern_file", "pattern_name", "expressivity", "dl_complexity_score"] + constructs
            ].rename(columns=pat_cols)

            df_pairs = df_source.merge(
                df_ont, on=["repo", "ontology_file"], how="left"
            ).merge(
                df_pat, on=["repo", "pattern_file", "pattern_name"], how="left"
            )

            out_pairs = out_root / "complexity_expressivity_per_pair.csv"
            df_pairs.to_csv(out_pairs, index=False)

            agg_rows = []

            def _safe_mean2(series):
                return float(pd.to_numeric(series, errors="coerce").dropna().mean()) if not series.empty else 0.0

            avg_onto_dl    = _safe_mean2(df_pairs["onto_dl_complexity"])
            avg_pattern_dl = _safe_mean2(df_pairs["pattern_dl_complexity"])
            avg_onto_ax    = _safe_mean2(df_pairs["onto_n_axioms"])

            agg_rows.append({
                "Metric": "Avg. ontology DL constructors (per pair)",
                "Value": avg_onto_dl,
            })
            agg_rows.append({
                "Metric": "Avg. pattern DL constructors (per pair)",
                "Value": avg_pattern_dl,
            })
            agg_rows.append({
                "Metric": "Avg. ontology axioms (per pair)",
                "Value": avg_onto_ax,
            })

            for c in constructs:
                onto_col = f"onto_{c}"
                patt_col = f"pattern_{c}"

                if onto_col in df_pairs.columns:
                    onto_frac = df_pairs[onto_col].fillna(False).astype(bool).mean()
                    agg_rows.append({
                        "Metric": f"{c} (ontology side, % pairs)",
                        "Value": 100.0 * onto_frac,
                    })

                if patt_col in df_pairs.columns:
                    patt_frac = df_pairs[patt_col].fillna(False).astype(bool).mean()
                    agg_rows.append({
                        "Metric": f"{c} (pattern side, % pairs)",
                        "Value": 100.0 * patt_frac,
                    })

            df_pairs_summary = pd.DataFrame(agg_rows)
            df_pairs_summary.to_csv(
                out_root / "complexity_expressivity_per_pair_summary.csv",
                index=False
            )

            write_latex_table(
                df_pairs_summary,
                out_root / "complexity_expressivity_per_pair_summary.tex",
                caption=(
                    f"Average logical complexity of ontologies and patterns per ontology–pattern pair {caption_suffix}, "
                    "including per-construct percentages of pairs where each DL construct is used."
                ).strip(),
                label=f"tab:complexity-per-pair-summary-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
            )

            # Construct-level breakdown (per pair)
            summary_rows = []
            total_pairs = len(df_pairs)

            for c in constructs:
                onto_col = f"onto_{c}"
                patt_col = f"pattern_{c}"
                if onto_col not in df_pairs.columns or patt_col not in df_pairs.columns:
                    continue

                onto_val = df_pairs[onto_col].fillna(False).astype(bool)
                patt_val = df_pairs[patt_col].fillna(False).astype(bool)

                has_onto = int(onto_val.sum())
                has_pattern = int(patt_val.sum())
                only_pattern = int((patt_val & ~onto_val).sum())
                only_onto = int((onto_val & ~patt_val).sum())
                both = int((onto_val & patt_val).sum())

                summary_rows.append({
                    "construct": c,
                    "num_pairs_total": total_pairs,
                    "num_pairs_onto_has": has_onto,
                    "num_pairs_pattern_has": has_pattern,
                    "num_pairs_pattern_only": only_pattern,
                    "num_pairs_onto_only": only_onto,
                    "num_pairs_both": both,
                    "pct_pairs_onto_has": 100.0 * has_onto / total_pairs if total_pairs else 0.0,
                    "pct_pairs_pattern_has": 100.0 * has_pattern / total_pairs if total_pairs else 0.0,
                    "pct_pairs_pattern_only": 100.0 * only_pattern / total_pairs if total_pairs else 0.0,
                    "pct_pairs_onto_only": 100.0 * only_onto / total_pairs if total_pairs else 0.0,
                    "pct_pairs_both": 100.0 * both / total_pairs if total_pairs else 0.0,
                })

            df_construct_breakdown = pd.DataFrame(summary_rows)
            df_construct_breakdown.to_csv(
                out_root / "complexity_expressivity_construct_breakdown_per_pair.csv",
                index=False
            )

            if not df_construct_breakdown.empty:
                cols_for_tex = [
                    "construct",
                    "pct_pairs_onto_has",
                    "pct_pairs_pattern_has",
                    "pct_pairs_pattern_only",
                    "pct_pairs_onto_only",
                    "pct_pairs_both",
                ]
                df_tex = df_construct_breakdown[cols_for_tex]
                write_latex_table(
                    df_tex,
                    out_root / "complexity_expressivity_construct_breakdown_per_pair.tex",
                    caption=(
                        f"Per-construct breakdown of logical features in ontology–pattern pairs {caption_suffix}: "
                        "how often a construct appears only in the ontology, only in the pattern, or in both."
                    ).strip(),
                    label=f"tab:construct-breakdown-per-pair-{sim_backend}{caption_suffix.replace(' ', '-').replace('≥', 'ge')}"
                )

                # Wide CSV: one row, columns like ROLE_INVERSE_pct_pairs_pattern_only, etc.
                wide_data = {}
                for _, row in df_construct_breakdown.iterrows():
                    c = row["construct"]
                    for k, v in row.items():
                        if k == "construct":
                            continue
                        col_name = f"{c}_{k}"
                        wide_data[col_name] = v
                df_wide = pd.DataFrame([wide_data])
                df_wide.to_csv(
                    out_root / "complexity_expressivity_construct_breakdown_per_pair_wide.csv",
                    index=False
                )

    except Exception as e:
        print(f"[WARN] Could not compute global-style statistics for {out_root}: {e}", file=sys.stderr)

# ----------------------------
# Main
# ----------------------------

def main(base_dir: str,
         out_dir: Path,
         out_csv: Path,
         max_bytes: int,
         skip_subs: Set[str],
         label_thresh: float,
         sim_backend: str,
         ngram: int,
         sbert_model: str,
         sbert_device: str,
         sbert_prefilter: float,
         materialize_mode: str,
         dataset_percents: List[int],
         hist_bins: List[int],
         make_graph_100: bool):

    out_dir.mkdir(parents=True, exist_ok=True)

    # SBERT if needed
    sbert = None
    if sim_backend in {"sbert", "hybrid"}:
        if not _SBERT_AVAILABLE:
            print("[WARN] sentence-transformers not installed; falling back to char backend.", file=sys.stderr)
            if sim_backend == "sbert":
                sys.exit(1)
            sim_backend = "char"
        else:
            sbert = SbertBackend(model_name=sbert_model, device=sbert_device)

    rows = []

    for sub in sorted(os.listdir(base_dir)):
        repo_dir = os.path.join(base_dir, sub)
        if not os.path.isdir(repo_dir):
            continue

        onto_file = get_main_ontology_file(repo_dir, max_bytes, skip_subs)
        if not onto_file:
            print(f"[WARN] No ontology in: {repo_dir}")
            continue

        og = try_parse_any(onto_file)
        if og is None:
            print(f"[WARN] Could not parse ontology: {onto_file}")
            continue

        patt_files = list_pattern_files(repo_dir, max_bytes, skip_subs)
        if not patt_files:
            continue

        for p in patt_files:
            pg = try_parse_any(p)
            if pg is None:
                print(f"[WARN] Could not parse pattern: {p}")
                continue

            m = coverage_and_connectivity(
                onto_g=og,
                pat_g=pg,
                label_thresh=label_thresh,
                sim_backend=sim_backend,
                ngram=ngram,
                sbert=sbert,
                sbert_prefilter=sbert_prefilter
            )

            rows.append({
                "repo": sub,
                "ontology_file": onto_file,
                "pattern_file": p,
                "pattern_name": Path(p).stem,
                "pattern_total": m["pattern_total"],
                "reused_total_graph": m["reused_total_graph"],
                "reused_total_label": m["reused_total_label"],
                "coverage_graph": m["coverage_graph"],
                "coverage_label": m["coverage_label"],
                "connectivity_ratio": m["connectivity_ratio"],
                # usage — all modes
                "usage_direct": m["usage_details"]["direct"],
                "usage_as_subclass": m["usage_details"]["as_subclass"],
                "usage_as_superclass": m["usage_details"]["as_superclass"],
                "usage_as_subproperty": m["usage_details"]["as_subproperty"],
                "usage_as_superproperty": m["usage_details"]["as_superproperty"],
                "usage_via_property": m["usage_details"]["via_property"],
                "usage_via_restriction": m["usage_details"]["via_restriction"],
                "usage_equivalent": m["usage_details"]["equivalent"],
                "usage_import_only": int(m["usage_details"]["import_only"]),
                "usage_isolated": int(m["usage_details"]["isolated"]),
                # connected property role stats
                "connected_props_total": m["usage_details"]["connected_props_total"],
                "connected_props_functional": m["usage_details"]["connected_props_functional"],
                "connected_props_inverse_functional": m["usage_details"]["connected_props_inverse_functional"],
                "connected_props_transitive": m["usage_details"]["connected_props_transitive"],
                "connected_props_symmetric": m["usage_details"]["connected_props_symmetric"],
                "connected_props_asymmetric": m["usage_details"]["connected_props_asymmetric"],
                "connected_props_reflexive": m["usage_details"]["connected_props_reflexive"],
                "connected_props_irreflexive": m["usage_details"]["connected_props_irreflexive"],
            })

    cols = [
        "repo","pattern_name","ontology_file","pattern_file","pattern_total",
        "reused_total_graph","reused_total_label","coverage_graph","coverage_label",
        "connectivity_ratio",
        "usage_direct","usage_as_subclass","usage_as_superclass",
        "usage_as_subproperty","usage_as_superproperty",
        "usage_via_property","usage_via_restriction","usage_equivalent",
        "usage_import_only","usage_isolated",
        "connected_props_total","connected_props_functional","connected_props_inverse_functional",
        "connected_props_transitive","connected_props_symmetric","connected_props_asymmetric",
        "connected_props_reflexive","connected_props_irreflexive"
    ]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["label_reuse_pct"] = df.apply(
            lambda r: (r["reused_total_label"] / r["pattern_total"] * 100.0) if r["pattern_total"] else 0.0, axis=1
        )
        df = df.sort_values(["repo","pattern_name"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} ({len(df)} rows)")

    if df.empty:
        print("[WARN] No pairs found; stopping.")
        return

    # ------------------------------------------------------------------
    # Global statistics over ALL pairs (top-level)
    # ------------------------------------------------------------------
    build_global_statistics_for_df(
        df_source=df,
        out_root=out_dir / "global_statistics",
        label_thresh=label_thresh,
        sim_backend=sim_backend,
        ngram=ngram,
        sbert=sbert,
        sbert_prefilter=sbert_prefilter,
        hist_bins=hist_bins,
        caption_suffix="(global)"
    )

    # ---------------------------
    # Build datasets by thresholds
    # ---------------------------
    for P in dataset_percents:
        subset = df[df["label_reuse_pct"] >= P].copy()
        ds_root = out_dir / f"{P}_{sim_backend}"
        (ds_root / "ontology").mkdir(parents=True, exist_ok=True)
        (ds_root / "patterns").mkdir(parents=True, exist_ok=True)

        # materialize flat files
        for r in subset.itertuples():
            onto_src = Path(r.ontology_file).resolve()
            patt_src = Path(r.pattern_file).resolve()
            onto_dst = ds_root / "ontology" / f"{Path(r.ontology_file).name}"
            patt_dst = ds_root / "patterns" / f"{Path(r.pattern_file).name}"

            if materialize_mode == "copy":
                _copy_file(onto_src, onto_dst)
                _copy_file(patt_src, patt_dst)
            else:  # symlink default
                if not _safe_symlink(onto_src, onto_dst):
                    _copy_file(onto_src, onto_dst)
                if not _safe_symlink(patt_src, patt_dst):
                    _copy_file(patt_src, patt_dst)

        # manifest for this dataset
        subset.to_csv(ds_root / "dataset_manifest.csv", index=False)

        # per-dataset statistics (+ DL complexity etc.)
        _write_dataset_stats(
            subset_df=subset,
            ds_root=ds_root,
            label_thresh=label_thresh,
            sim_backend=sim_backend,
            ngram=ngram,
            sbert=sbert,
            sbert_prefilter=sbert_prefilter,
            hist_bins=hist_bins
        )

        # Subset-specific global_statistics, e.g. 100_exact/global_statistics
        build_global_statistics_for_df(
            df_source=subset,
            out_root=ds_root / "global_statistics",
            label_thresh=label_thresh,
            sim_backend=sim_backend,
            ngram=ngram,
            sbert=sbert,
            sbert_prefilter=sbert_prefilter,
            hist_bins=hist_bins,
            caption_suffix=f"(subset ≥{P}%)"
        )

        print(f"[OK] Built dataset ≥{P}% ({sim_backend}) at: {ds_root}")


    # Optional: build a graph-only 100% reuse dataset
    if make_graph_100:
        # Only patterns with at least one entity and complete graph reuse
        mask_full_graph_reuse = (
            (df["pattern_total"] > 0) &
            (df["reused_total_graph"] == df["pattern_total"])
            # optionally also require real usage (not just import):
            # & (df["usage_import_only"] == 0)
        )
        subset_g100 = df[mask_full_graph_reuse].copy()

        ds_root_g = out_dir / "graph_100"

        # --- IMPORTANT: reset graph_100 so ontology/ and patterns/ match the new subset ---
        if ds_root_g.exists():
            shutil.rmtree(ds_root_g)

        (ds_root_g / "ontology").mkdir(parents=True, exist_ok=True)
        (ds_root_g / "patterns").mkdir(parents=True, exist_ok=True)

        for r in subset_g100.itertuples():
            onto_src = Path(r.ontology_file).resolve()
            patt_src = Path(r.pattern_file).resolve()
            onto_dst = ds_root_g / "ontology" / Path(r.ontology_file).name
            patt_dst = ds_root_g / "patterns" / Path(r.pattern_file).name

            if materialize_mode == "copy":
                _copy_file(onto_src, onto_dst)
                _copy_file(patt_src, patt_dst)
            else:
                if not _safe_symlink(onto_src, onto_dst):
                    _copy_file(onto_src, onto_dst)
                if not _safe_symlink(patt_src, patt_dst):
                    _copy_file(patt_src, patt_dst)

        # manifest for the graph_100 dataset
        subset_g100.to_csv(ds_root_g / "dataset_manifest.csv", index=False)

        # stats for graph_100
        _write_dataset_stats(
            subset_df=subset_g100,
            ds_root=ds_root_g,
            label_thresh=label_thresh,
            sim_backend=sim_backend,
            ngram=ngram,
            sbert=sbert,
            sbert_prefilter=sbert_prefilter,
            hist_bins=hist_bins
        )

        build_global_statistics_for_df(
            df_source=subset_g100,
            out_root=ds_root_g / "global_statistics",
            label_thresh=label_thresh,
            sim_backend=sim_backend,
            ngram=ngram,
            sbert=sbert,
            sbert_prefilter=sbert_prefilter,
            hist_bins=hist_bins,
            caption_suffix="(graph_100)"
        )

        print(f"[OK] Built dataset graph_100 (graph-only 100% reuse) at: {ds_root_g}")

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Create reuse-percentage filtered datasets with statistics.")
    ap.add_argument("--base", required=True, help="Base directory containing repos (each with an ontology and patterns/).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory.")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Master CSV of all (ontology,pattern) metrics.")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES, help="Skip files larger than this many bytes.")
    ap.add_argument("--skip-name", default="", help="Comma-separated substrings to skip in filenames (case-insensitive).")

    # Similarity configuration
    ap.add_argument("--sim-backend", choices=["hybrid","char","word","sbert","exact"], default="hybrid",
                    help="Similarity backend for labels.")
    ap.add_argument("--label-threshold", type=float, default=DEFAULT_LABEL_SIM,
                    help="Label similarity threshold (0..1).")
    ap.add_argument("--ngram", type=int, default=3, help="n for character n-grams (char/hybrid).")
    ap.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL, help="SentenceTransformer model name.")
    ap.add_argument("--sbert-device", default=DEFAULT_SBERT_DEVICE, choices=["cpu","cuda"], help="SBERT device.")
    ap.add_argument("--sbert-prefilter", type=float, default=DEFAULT_SBERT_PREFILTER,
                    help="If char-cosine ≥ this, skip SBERT (hybrid mode).")

    # Dataset creation
    ap.add_argument("--dataset-percents", default=DEFAULT_DATASET_PERCENTS,
                    help="Comma-separated label-reuse % thresholds, e.g. '100,95,90'.")
    ap.add_argument("--materialize", choices=["symlink","copy"], default="symlink",
                    help="How to place files in each dataset folder.")
    ap.add_argument("--hist-bins", default=",".join(map(str, DEFAULT_HIST_BINS)),
                    help="Comma-separated integer bin edges for histogram CSVs, e.g., '0,1,2,5,10,20,50'.")
    ap.add_argument("--make-graph-100", action="store_true",
                    help="Also materialize a graph-only dataset where coverage_graph == 1.0 into <out-dir>/graph_100/.")

    args = ap.parse_args()

    base_dir = args.base
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Base directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = out_dir / out_csv

    skip_set = {s.strip().lower() for s in args.skip_name.split(",") if s.strip()}
    dataset_percents = [int(p.strip()) for p in args.dataset_percents.split(",") if p.strip()]
    hist_bins = [int(b.strip()) for b in args.hist_bins.split(",") if b.strip()]
    if len(hist_bins) < 2:
        hist_bins = DEFAULT_HIST_BINS

    main(
        base_dir=base_dir,
        out_dir=out_dir,
        out_csv=out_csv,
        max_bytes=args.max_bytes,
        skip_subs=skip_set,
        label_thresh=float(args.label_threshold),
        sim_backend=args.sim_backend,
        ngram=int(args.ngram),
        sbert_model=args.sbert_model,
        sbert_device=args.sbert_device,
        sbert_prefilter=float(args.sbert_prefilter),
        materialize_mode=args.materialize,
        dataset_percents=dataset_percents,
        hist_bins=hist_bins,
        make_graph_100=bool(args.make_graph_100),
    )
# 
