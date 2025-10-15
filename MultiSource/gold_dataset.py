#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gold_dataset.py — Build a gold-standard style evaluation of ODP reuse,
and optionally materialize examples into class folders for experiments.

Labels per (ontology, pattern) pair:
  Correctly-Reused | Partially-Reused | Not-Reused

Similarity backends:
  --sim-backend {hybrid,char,word,sbert}
    - hybrid (default): char 3-gram cosine fast path, SBERT refine
    - char/word: cosine over char n-grams or word tokens only
    - sbert: SBERT cosine only

SBERT options:
  --sbert-model sentence-transformers/all-MiniLM-L6-v2
  --sbert-device cpu|cuda
  --sbert-prefilter 0.85   (skip SBERT if char-cosine already ≥ prefilter)
"""

import os
import sys
import re
import glob
import json
import shutil
import math
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from rdflib import Graph, URIRef, RDF, RDFS, OWL, Literal

# Try to import SBERT lazily
_SBERT_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    _SBERT_AVAILABLE = False

# ----------------------------
# Config (defaults; override via CLI)
# ----------------------------

ONTOLOGY_EXTS = ("*.owl", "*.ttl", "*.rdf", "*.xml", "*.nt", "*.trig")
PATTERN_EXTS  = ("*.owl", "*.ttl")

DEFAULT_MAX_FILE_BYTES = 3_000_000_000  # 1GB
DEFAULT_SKIP_SUBSTRS = {""}          # case-insensitive name substrings

DEFAULT_OUT_CSV = "gold_labels.csv"
DEFAULT_OUT_DIR = "./gold_out"

# Label-similarity threshold
DEFAULT_LABEL_SIM = 0.80

# Labeling thresholds
DEFAULT_MIN_REUSED_ENTITIES = 3         # if small patterns, ensure ≥ N reused
DEFAULT_COVERAGE_GOOD = 0.30            # 30% graph coverage threshold
DEFAULT_CONNECTIVITY_GOOD = 0.60        # 60% reused entities connected to ontology
DEFAULT_CONNECTIVITY_WEAK = 0.20        # boundary for "partial"
DEFAULT_NONREUSED_REL_MAX = 0.20        # ≤20% of non-reused look label-relevant
DEFAULT_IMPORT_CONNECTIVITY_BAD = 0.10  # imports present but no actual usage

# SBERT defaults
DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SBERT_DEVICE = "cpu"
DEFAULT_SBERT_PREFILTER = 0.85

# ----------------------------
# Utilities: parsing & discovery
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
# Entity extraction & reuse
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
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s or "")
    s = s.replace("_", " ").replace("-", " ")
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
# Cosine similarity utilities (char/word)
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
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name, device=device)
        self.cache: Dict[str, np.ndarray] = {}

    def embed(self, texts: List[str]) -> np.ndarray:
        # normalize and dedupe to keep cache small
        normed = [normalize_label(t) for t in texts]
        vecs = []
        to_encode_idx = []
        to_encode_texts = []
        for i, t in enumerate(normed):
            if t in self.cache:
                vecs.append(self.cache[t])
            else:
                vecs.append(None)  # placeholder
                to_encode_idx.append(i)
                to_encode_texts.append(t)
        if to_encode_texts:
            enc = self.model.encode(to_encode_texts, convert_to_numpy=True, normalize_embeddings=True)
            for j, i in enumerate(to_encode_idx):
                self.cache[normed[i]] = enc[j]
                vecs[i] = enc[j]
        return np.vstack(vecs) if vecs else np.zeros((0, 384), dtype=np.float32)  # dimension depends on model

    def max_cosine(self, a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        A = self.embed(a)  # shape [m, d]
        B = self.embed(b)  # shape [n, d]
        # since normalized embeddings, cosine = dot
        sims = A @ B.T  # [m, n]
        return float(np.max(sims))

# ----------------------------
# Metrics computation (now supports SBERT/hybrid)
# ----------------------------

def coverage_and_connectivity(onto_g: Optional[Graph],
                              pat_g: Optional[Graph],
                              label_thresh: float,
                              sim_backend: str,
                              ngram: int,
                              sbert: Optional[SbertBackend],
                              sbert_prefilter: float) -> Dict[str, float | int | bool]:
    """
    Compute:
      - reused sets (graph & label),
      - connectivity ratio of reused entities to non-pattern IRIs,
      - nonreused relevance proxy (fraction label-similar),
      - has_import flag
    """
    if onto_g is None or pat_g is None:
        return dict(
            pattern_total=0,
            reused_total_graph=0,
            reused_total_label=0,
            coverage_graph=0.0,
            coverage_label=0.0,
            connectivity_ratio=0.0,
            nonreused_relevance=0.0,
            has_import=False
        )

    p_cls, p_obj, p_data = collect_entities(pat_g)
    pattern_entities: Set[str] = p_cls | p_obj | p_data
    patt_total = len(pattern_entities)

    def base_ns(i: str) -> str:
        return i.rsplit("#", 1)[0] + ("#" if "#" in i else "")

    patt_namespaces = {base_ns(i) for i in pattern_entities}

    # Graph reuse (exact IRI)
    reused_graph = {i for i in pattern_entities if any_occurs(onto_g, i)}

    # Prepare labels
    o_cls, o_obj, o_data = collect_entities(onto_g)
    onto_entities = o_cls | o_obj | o_data
    o_labels_map = extract_labels(onto_g, onto_entities)
    p_labels_map = extract_labels(pat_g, pattern_entities)

    onto_label_flat = sorted({lab for labs in o_labels_map.values() for lab in labs})

    # Optional: pre-embed ontology label set once per call for SBERT speed (handled by cache)
    reused_label = set()
    for i, pls in p_labels_map.items():
        if i in reused_graph:
            reused_label.add(i)
            continue

        # quick exact label overlap
        if set(pls) & set(onto_label_flat):
            reused_label.add(i)
            continue

        # score via selected backend
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
            # SBERT max over label sets
            if sbert.max_cosine(list(pls), onto_label_flat) >= label_thresh:
                matched = True
        else:  # hybrid: char prefilter, then SBERT refine for near-misses
            # 1) char prefilter: accept if strong
            for pl in pls:
                if any(char_label_sim(pl, ol, ngram) >= label_thresh for ol in onto_label_flat):
                    matched = True; break
            # 2) SBERT for near-misses (char between [sbert_prefilter-0.05, label_thresh))
            if not matched and sbert is not None:
                # check if any char match is high-ish
                hi_char = 0.0
                for pl in pls:
                    for ol in onto_label_flat:
                        hi_char = max(hi_char, char_label_sim(pl, ol, ngram))
                if hi_char >= (sbert_prefilter - 0.05) and hi_char < label_thresh:
                    if sbert.max_cosine(list(pls), onto_label_flat) >= label_thresh:
                        matched = True

        if matched:
            reused_label.add(i)

    # Connectivity of graph-reused entities to non-pattern IRIs
    def is_pattern_iri(u: URIRef) -> bool:
        su = str(u)
        for ns in patt_namespaces:
            if su.startswith(ns):
                return True
        return False

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

    # Non-reused relevance (how many "unused" pattern entities look label-similar to ontology)
    non_reused = pattern_entities - reused_graph
    nonrel_hits = 0
    for i in non_reused:
        pls = p_labels_map.get(i, set())
        if not pls:
            continue
        # exact overlap
        if set(pls) & set(onto_label_flat):
            nonrel_hits += 1
            continue
        # backend
        hit = False
        if sim_backend == "char":
            for pl in pls:
                if any(char_label_sim(pl, ol, ngram) >= label_thresh for ol in onto_label_flat):
                    hit = True; break
        elif sim_backend == "word":
            for pl in pls:
                if any(word_label_sim(pl, ol) >= label_thresh for ol in onto_label_flat):
                    hit = True; break
        elif sim_backend == "sbert":
            if sbert is not None and sbert.max_cosine(list(pls), onto_label_flat) >= label_thresh:
                hit = True
        else:  # hybrid
            # try char quickly
            for pl in pls:
                if any(char_label_sim(pl, ol, ngram) >= label_thresh for ol in onto_label_flat):
                    hit = True; break
            if not hit and sbert is not None:
                # only escalate if close-ish
                hi_char = 0.0
                for pl in pls:
                    for ol in onto_label_flat:
                        hi_char = max(hi_char, char_label_sim(pl, ol, ngram))
                if hi_char >= (sbert_prefilter - 0.05) and hi_char < label_thresh:
                    if sbert.max_cosine(list(pls), onto_label_flat) >= label_thresh:
                        hit = True
        if hit:
            nonrel_hits += 1

    nonreused_relevance = (nonrel_hits / len(non_reused)) if non_reused else 0.0

    # owl:imports into the pattern namespace?
    has_import = False
    for ns in patt_namespaces:
        for _, _, o in onto_g.triples((None, OWL.imports, None)):
            if isinstance(o, URIRef) and str(o).startswith(ns):
                has_import = True; break
        if has_import: break

    return dict(
        pattern_total=patt_total,
        reused_total_graph=len(reused_graph),
        reused_total_label=len(reused_label),
        coverage_graph=(len(reused_graph) / patt_total) if patt_total else 0.0,
        coverage_label=(len(reused_label) / patt_total) if patt_total else 0.0,
        connectivity_ratio=connectivity_ratio,
        nonreused_relevance=nonreused_relevance,
        has_import=has_import
    )

# ----------------------------
# Labeling logic (unchanged)
# ----------------------------

def classify_pair(metrics: Dict[str, float | int | bool],
                  min_reused_entities: int,
                  cov_good: float,
                  conn_good: float,
                  conn_weak: float,
                  nonrel_max: float) -> Tuple[str, str]:
    reused_g   = metrics["reused_total_graph"]
    reused_l   = metrics["reused_total_label"]
    cov_g      = metrics["coverage_graph"]
    conn       = metrics["connectivity_ratio"]
    nonrel     = metrics["nonreused_relevance"]
    has_imp    = metrics["has_import"]

    if reused_g == 0 and reused_l < 2:
        return "Not-Reused", "No graph reuse and <2 label matches"

    if has_imp and conn < DEFAULT_IMPORT_CONNECTIVITY_BAD:
        return "Not-Reused", "Imports present but no meaningful connectivity (<0.10)"

    if ((cov_g >= cov_good) or (reused_g >= min_reused_entities)) and (conn >= conn_good) and (nonrel <= nonrel_max):
        return "Correctly-Reused", f"Coverage ok (graph={cov_g:.2f} or reused≥{min_reused_entities}), connectivity {conn:.2f}, nonreused_relevance {nonrel:.2f}"

    if (reused_g > 0 or reused_l > 0) and ((cov_g < cov_good) or (conn_weak <= conn < conn_good)):
        return "Partially-Reused", f"Some reuse but limited coverage (graph={cov_g:.2f}) or weak connectivity ({conn:.2f})"

    return "Not-Reused", "Metrics fall below reuse thresholds"

# ----------------------------
# Materialization helpers
# ----------------------------

def _safe_symlink(src: Path, dst: Path) -> bool:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return True
    except Exception:
        return False

def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def materialize_pair(row, dest_root: Path, mode: str) -> None:
    label_dir = dest_root / row.gold_label.replace("-", "_").lower()
    bundle = f"{row.repo}__{row.pattern_name}"
    dst_dir = label_dir / bundle
    dst_dir.mkdir(parents=True, exist_ok=True)

    onto_src = Path(row.ontology_file).resolve()
    patt_src = Path(row.pattern_file).resolve()

    onto_dst = dst_dir / Path(row.ontology_file).name
    patt_dst = dst_dir / Path(row.pattern_file).name

    ok = False
    if mode == "symlink":
        ok1 = _safe_symlink(onto_src, onto_dst)
        ok2 = _safe_symlink(patt_src, patt_dst)
        ok = ok1 and ok2
        if not ok:
            _copy_file(onto_src, onto_dst)
            _copy_file(patt_src, patt_dst)
    elif mode == "copy":
        _copy_file(onto_src, onto_dst)
        _copy_file(patt_src, patt_dst)
    elif mode == "none":
        return

    meta = {
        "repo": row.repo,
        "gold_label": row.gold_label,
        "pattern_name": row.pattern_name,
        "ontology_file": str(onto_src),
        "pattern_file": str(patt_src),
        "metrics": {
            "pattern_total": int(row.pattern_total),
            "reused_total_graph": int(row.reused_total_graph),
            "reused_total_label": int(row.reused_total_label),
            "coverage_graph": float(row.coverage_graph),
            "coverage_label": float(row.coverage_label),
            "connectivity_ratio": float(row.connectivity_ratio),
            "nonreused_relevance": float(row.nonreused_relevance),
            "has_import": bool(row.has_import),
        },
        "rationale": row.rationale
    }
    (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def materialize_repo(repo: str, rows: pd.DataFrame, dest_root: Path, mode: str) -> None:
    priorities = {"Correctly-Reused": 2, "Partially-Reused": 1, "Not-Reused": 0}
    best = max(rows["gold_label"], key=lambda L: priorities.get(L, -1))
    label_dir = dest_root / best.replace("-", "_").lower()
    dst_dir = label_dir / repo
    src_dir = Path(rows.iloc[0]["ontology_file"]).resolve().parent

    if mode == "none":
        return

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    if mode == "symlink":
        try:
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            dst_dir.symlink_to(src_dir, target_is_directory=True)
            return
        except Exception:
            pass
    shutil.copytree(src_dir, dst_dir)

# ----------------------------
# Main
# ----------------------------

def main(base_dir: str,
         out_dir: Path,
         out_csv: Path,
         max_bytes: int,
         skip_subs: Set[str],
         label_thresh: float,
         min_reused_entities: int,
         cov_good: float,
         conn_good: float,
         conn_weak: float,
         nonrel_max: float,
         materialize: str,
         grouping: str,
         sim_backend: str,
         ngram: int,
         sbert_model: str,
         sbert_device: str,
         sbert_prefilter: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)

    # Initialize SBERT if needed
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

    # iterate repos (subdirectories of base)
    for sub in sorted(os.listdir(base_dir)):
        repo_dir = os.path.join(base_dir, sub)
        if not os.path.isdir(repo_dir):
            continue

        main_onto = get_main_ontology_file(repo_dir, max_bytes, skip_subs)
        if not main_onto:
            print(f"[WARN] No ontology in: {repo_dir}")
            continue

        og = try_parse_any(main_onto)
        if og is None:
            print(f"[WARN] Could not parse ontology: {main_onto}")
            continue

        pat_files = list_pattern_files(repo_dir, max_bytes, skip_subs)
        if not pat_files:
            continue

        for p in pat_files:
            pg = try_parse_any(p)
            if pg is None:
                print(f"[WARN] Could not parse pattern: {p}")
                continue

            metrics = coverage_and_connectivity(
                onto_g=og,
                pat_g=pg,
                label_thresh=label_thresh,
                sim_backend=sim_backend,
                ngram=ngram,
                sbert=sbert,
                sbert_prefilter=sbert_prefilter
            )
            label, why = classify_pair(metrics,
                                       min_reused_entities,
                                       cov_good,
                                       conn_good,
                                       conn_weak,
                                       nonrel_max)

            rows.append({
                "repo": sub,
                "ontology_file": main_onto,
                "pattern_file": p,
                "pattern_name": Path(p).stem,
                # metrics
                "pattern_total": metrics["pattern_total"],
                "reused_total_graph": metrics["reused_total_graph"],
                "reused_total_label": metrics["reused_total_label"],
                "coverage_graph": metrics["coverage_graph"],
                "coverage_label": metrics["coverage_label"],
                "connectivity_ratio": metrics["connectivity_ratio"],
                "nonreused_relevance": metrics["nonreused_relevance"],
                "has_import": int(metrics["has_import"]),
                # decision
                "gold_label": label,
                "rationale": why
            })

    # Build a well-formed DataFrame even if rows == []
    cols = [
        "repo","pattern_name","ontology_file","pattern_file","pattern_total",
        "reused_total_graph","reused_total_label","coverage_graph","coverage_label",
        "connectivity_ratio","nonreused_relevance","has_import","gold_label","rationale"
    ]
    df = pd.DataFrame(rows, columns=cols)

    # Only sort if columns exist and DataFrame is non-empty
    if not df.empty and {"repo","pattern_name"}.issubset(df.columns):
        df = df.sort_values(["repo","pattern_name"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} ({len(df)} rows)")

    # Manifests
    mf = out_dir / "manifests"
    for name, group in df.groupby("gold_label"):
        (mf / f"{name.replace('-','_').lower()}.txt").write_text(
            "\n".join(f"{r.repo} :: {r.pattern_name}" for r in group.itertuples()),
            encoding="utf-8"
        )
    print(f"[OK] Wrote manifests under {mf}")

    # Materialization
    if materialize not in {"symlink", "copy", "none"}:
        print(f"[WARN] Unknown --materialize={materialize}; skipping materialization.")
        return

    if materialize == "none":
        return

    for lab in ("Correctly-Reused", "Partially-Reused", "Not-Reused"):
        (out_dir / lab.replace("-", "_").lower()).mkdir(parents=True, exist_ok=True)

    if grouping == "pair":
        for r in df.itertuples():
            materialize_pair(r, out_dir, materialize)
        print(f"[OK] Materialized pairs into {out_dir}/<label>/<repo>__<pattern>/")
    elif grouping == "repo":
        for repo, group in df.groupby("repo"):
            materialize_repo(repo, group, out_dir, materialize)
        print(f"[OK] Materialized repos into {out_dir}/<label>/<repo>/")
    else:
        print(f"[WARN] Unknown --grouping={grouping}; skipping materialization.")

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Create a gold-standard style dataset labeling ODP reuse quality, optionally materialized for experiments.")
    ap.add_argument("--base", required=True, help="Base directory containing repos (each with an ontology and patterns/).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory (CSV + manifests + materialized examples).")
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV filename (inside --out-dir unless absolute).")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES, help="Skip files larger than this many bytes.")
    ap.add_argument("--skip-name", default="", help="Comma-separated substrings to skip in filenames (case-insensitive).")
    ap.add_argument("--label-threshold", type=float, default=DEFAULT_LABEL_SIM, help="Label/localname similarity threshold (0..1).")

    # Labeling thresholds
    ap.add_argument("--min-reused-entities", type=int, default=DEFAULT_MIN_REUSED_ENTITIES)
    ap.add_argument("--coverage-good", type=float, default=DEFAULT_COVERAGE_GOOD, help="Graph coverage threshold for 'good' reuse (0..1).")
    ap.add_argument("--connectivity-good", type=float, default=DEFAULT_CONNECTIVITY_GOOD, help="Connectivity threshold for 'good' reuse (0..1).")
    ap.add_argument("--connectivity-weak", type=float, default=DEFAULT_CONNECTIVITY_WEAK, help="Lower bound for partial reuse (0..1).")
    ap.add_argument("--nonreused-relevance-max", type=float, default=DEFAULT_NONREUSED_REL_MAX, help="Max allowed fraction of non-reused entities that look label-relevant (0..1).")

    # Materialization controls
    ap.add_argument("--materialize", choices=["symlink","copy","none"], default="symlink",
                    help="Place selected examples under out_dir/<label>/*")
    ap.add_argument("--grouping", choices=["pair","repo"], default="pair",
                    help="Materialize by (ontology,pattern) pair or entire repo")

    # Similarity configuration
    ap.add_argument("--sim-backend", choices=["hybrid","char","word","sbert"], default="hybrid",
                    help="Similarity backend for labels.")
    ap.add_argument("--ngram", type=int, default=3,
                    help="n for character n-grams when using char/hybrid (default: 3).")

    # SBERT configuration
    ap.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL,
                    help="SentenceTransformer model name.")
    ap.add_argument("--sbert-device", default=DEFAULT_SBERT_DEVICE, choices=["cpu","cuda"],
                    help="Device for SBERT encoding.")
    ap.add_argument("--sbert-prefilter", type=float, default=DEFAULT_SBERT_PREFILTER,
                    help="If char-cosine ≥ this, skip SBERT (hybrid mode).")

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

    main(
        base_dir=base_dir,
        out_dir=out_dir,
        out_csv=out_csv,
        max_bytes=args.max_bytes,
        skip_subs=skip_set,
        label_thresh=float(args.label_threshold),
        min_reused_entities=int(args.min_reused_entities),
        cov_good=float(args.coverage_good),
        conn_good=float(args.connectivity_good),
        conn_weak=float(args.connectivity_weak),
        nonrel_max=float(args.nonreused_relevance_max),
        materialize=args.materialize,
        grouping=args.grouping,
        sim_backend=args.sim_backend,
        ngram=int(args.ngram),
        sbert_model=args.sbert_model,
        sbert_device=args.sbert_device,
        sbert_prefilter=float(args.sbert_prefilter),
    )
