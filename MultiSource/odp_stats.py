#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ODP reuse statistics generator (graph + SBERT/label-based) with distributions & publication figures.

What’s new in this version
--------------------------
- SBERT-based label similarity when available (cosine over embeddings).
- CLI:
  * --sim-backend {auto,sbert,char} (default: auto)
  * --sbert-model "sentence-transformers/all-MiniLM-L6-v2" (default: all-MiniLM-L6-v2)
  * --label-threshold <float> (default: 0.80)
- Falls back to character/word similarity (difflib+Jaccard) with a clear warning.
- Defensive DataFrame creation + plotting/LaTeX guards to avoid crashes on sparse data.
- FIX: robust cosine similarity — no casting arrays to float; always returns scalar.

Usage
-----
python odp_stats.py \
  --base "/path/to/github_dataset" \
  --out-dir "./github_dataset_stats" \
  --max-bytes 200000000 \
  --skip-name "cco" \
  --sim-backend auto \
  --sbert-model all-MiniLM-L6-v2 \
  --label-threshold 0.80
"""

import os
import sys
import re
import glob
import logging
import difflib
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdflib import Graph, URIRef, RDF, RDFS, Literal
from rdflib.namespace import OWL
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------

DEFAULT_BASE_DIR = "/home/ebrahim/odps-extraction/Ontologies_reuseODP/"
DEFAULT_OUT_DIR = "./github_dataset_stats"

# Ontology scan (broader); patterns limited to owl/ttl
ONTOLOGY_EXTS = ("*.owl", "*.ttl", "*.rdf", "*.xml", "*.nt", "*.trig")
PATTERN_EXTS  = ("*.owl", "*.ttl")

# Skip rules
DEFAULT_MAX_FILE_BYTES = 1_000_000_000  # 1 GB
DEFAULT_SKIP_NAME_SUBSTRINGS = {"cco"}  # case-insensitive substrings

# Label similarity threshold (can override via --label-threshold)
LABEL_SIM_THRESHOLD = 0.80

# Similarity backend defaults
DEFAULT_SIM_BACKEND = "auto"  # auto | sbert | char
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"

# Quiet rdflib warnings
logging.getLogger("rdflib").setLevel(logging.ERROR)

# ----------------------------
# Helpers: skip, parsing, IRIs, counts
# ----------------------------

def should_skip(path: str,
                max_bytes: int,
                skip_name_substrings: Set[str]) -> bool:
    try:
        size_ok = os.path.getsize(path) <= max_bytes
    except OSError:
        size_ok = True
    name = Path(path).name.lower()
    name_ok = not any(sub in name for sub in skip_name_substrings)
    return (not size_ok) or (not name_ok)


def try_parse_any(path: str) -> Optional[Graph]:
    g = Graph()
    tries = [None, "xml", "turtle", "n3", "nt", "trig", "json-ld"]
    for fmt in tries:
        try:
            g.parse(path, format=fmt)
            return g
        except Exception:
            continue
    return None


def get_ontology_iri(g: Graph) -> Optional[str]:
    for s in g.subjects(RDF.type, OWL.Ontology):
        if isinstance(s, URIRef):
            return str(s)
    return None


IRI_RE = re.compile(r'https?://[^\s"\'<>]+')

def lightweight_iri_probe(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            head = fh.read(100_000).decode(errors="ignore")
    except Exception:
        return None
    m = re.search(r'^\s*Ontology:\s*[<]?([^>\s]+)[>]?', head,
                  flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = IRI_RE.search(head)
    return m.group(0) if m else None


def safe_parse_graph(path: str) -> Tuple[Optional[Graph], Optional[str], bool]:
    g = try_parse_any(path)
    if g is not None:
        return g, get_ontology_iri(g), False
    iri = lightweight_iri_probe(path)
    return None, iri, True


def collect_defined_entities(g: Graph) -> Tuple[Set[str], Set[str], Set[str]]:
    classes = {str(s) for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    classes |= {str(s) for s in g.subjects(RDF.type, RDFS.Class) if isinstance(s, URIRef)}
    obj_props = {str(s) for s in g.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
    data_props = {str(s) for s in g.subjects(RDF.type, OWL.DatatypeProperty) if isinstance(s, URIRef)}
    return classes, obj_props, data_props


def any_occurs_in_graph(g: Graph, iri: URIRef) -> bool:
    return (iri, None, None) in g or (None, iri, None) in g or (None, None, iri) in g


def count_reuse_in_graph(g: Graph,
                         cls_iris: Set[str],
                         obj_iris: Set[str],
                         data_iris: Set[str]) -> Tuple[int, int, int]:
    reused_cls = sum(1 for iri in cls_iris if any_occurs_in_graph(g, URIRef(iri)))
    reused_obj = sum(1 for iri in obj_iris if any_occurs_in_graph(g, URIRef(iri)))
    reused_data = sum(1 for iri in data_iris if any_occurs_in_graph(g, URIRef(iri)))
    return reused_cls, reused_obj, reused_data


# ----------------------------
# Label extraction & similarity (SBERT backend with char fallback)
# ----------------------------

import string as _string
_PUNCT_TABLE = str.maketrans({c: " " for c in _string.punctuation})

def _camel_to_space(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = s.replace("_", " ").replace("-", " ")
    return s

def _localname(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]

def _normalize_label(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    t = _camel_to_space(t)
    t = t.translate(_PUNCT_TABLE)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def extract_entity_labels(g: Graph, iris: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for iri in iris:
        lbls: Set[str] = set()
        s = URIRef(iri)
        for _, _, lit in g.triples((s, RDFS.label, None)):
            if isinstance(lit, Literal):
                lbls.add(_normalize_label(str(lit)))
        ln = _normalize_label(_localname(iri))
        if ln:
            lbls.add(ln)
        lbls = {l for l in lbls if l}
        out[iri] = lbls or set()
    return out

# ----- Char backend (legacy) -----

def _tokenize(text: str) -> List[str]:
    return [tok for tok in _normalize_label(text).split() if tok]

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _string_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b).ratio()

def char_label_similarity(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    jac = _jaccard(ta, tb)
    rat = _string_ratio(" ".join(ta), " ".join(tb))
    return max(jac, rat)

# ----- SBERT backend -----

class _SBERTBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False  # normalize explicitly below
        )
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return emb / norms

def cosine_max_against_set(vec: np.ndarray, mat: np.ndarray) -> float:
    """
    vec: (D,), mat: (N,D), both L2-normalized.
    Returns scalar max cosine similarity (float).
    """
    if mat is None or mat.size == 0:
        return 0.0
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    sims = mat @ vec  # (N,)
    return float(np.max(sims))

class LabelIndex:
    """
    Holds unique normalized labels and (optionally) their SBERT embeddings for fast cosine search.
    Falls back to char similarity if no SBERT backend is provided.
    """
    def __init__(self, labels: Set[str], backend: Optional[_SBERTBackend]):
        self.labels: List[str] = sorted(l for l in labels if l)
        self.backend = backend
        if backend and self.labels:
            self.emb = backend.encode(self.labels)  # (N, D) normalized
        else:
            self.emb = np.zeros((0, 1), dtype=np.float32)

    def max_similarity(self, query_label: str, threshold: float) -> float:
        q = _normalize_label(query_label)
        if not q:
            return 0.0
        if q in self.labels:
            return 1.0
        if self.backend and self.emb.size > 0:
            qv = self.backend.encode([q])[0]  # (D,)
            return cosine_max_against_set(qv, self.emb)
        # fallback char similarity
        return max((char_label_similarity(q, l) for l in self.labels), default=0.0)

# ----------------------------
# Discovery (pick one main ontology per subfolder)
# ----------------------------

def pick_one_main_ontology(files: List[str]) -> Optional[str]:
    if not files:
        return None

    def ext_rank(p: str) -> int:
        e = Path(p).suffix.lower()
        if e == ".owl": return 0
        if e == ".ttl": return 1
        if e in (".rdf", ".xml"): return 2
        return 3

    return max(files, key=lambda p: (- (3 - ext_rank(p)), os.path.getsize(p)))

def find_ontology_files(base_dir: str,
                        max_bytes: int,
                        skip_name_substrings: Set[str]) -> List[str]:
    results: List[str] = []
    for sub in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        candidates: List[str] = []
        for ext in ONTOLOGY_EXTS:
            for p in glob.glob(os.path.join(sub_path, ext)):
                if should_skip(p, max_bytes, skip_name_substrings):
                    print(f"[SKIP] {p} (too large or matches skip name rule)")
                    continue
                candidates.append(p)
        main = pick_one_main_ontology(candidates)
        if main:
            print(f"[PICK] {sub} \u2192 {Path(main).name}")
            results.append(main)
    return sorted(results)

def find_pattern_files_for(ontology_file: str,
                           max_bytes: int,
                           skip_name_substrings: Set[str]) -> List[str]:
    ontology_dir = os.path.dirname(ontology_file)
    patterns_dir = os.path.join(ontology_dir, "patterns")
    if not (os.path.exists(patterns_dir) and os.path.isdir(patterns_dir)):
        return []
    files: List[str] = []
    for ext in PATTERN_EXTS:
        for p in glob.glob(os.path.join(patterns_dir, ext)):
            if should_skip(p, max_bytes, skip_name_substrings):
                print(f"[SKIP] {p} (too large or matches skip name rule)")
                continue
            files.append(p)
    return sorted(files)


# ----------------------------
# Plot helpers (publication-friendly)
# ----------------------------

def _save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins=20):
    series = series.dropna()
    if series.empty:
        return
    plt.figure(figsize=(10, 6), dpi=220)
    plt.hist(series, bins=bins)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _save_cdf(series: pd.Series, title: str, xlabel: str, path: Path):
    s = series.dropna().sort_values()
    if s.empty:
        return
    y = (pd.Series(range(1, len(s)+1)) / len(s)).values
    plt.figure(figsize=(10, 6), dpi=220)
    plt.plot(s.values, y)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel("Cumulative fraction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _save_box(series: pd.Series, title: str, ylabel: str, path: Path):
    series = series.dropna()
    if series.empty:
        return
    plt.figure(figsize=(7, 6), dpi=220)
    plt.boxplot(series.values, vert=True, showfliers=False)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _save_bar(labels: List[str], values: List[float], title: str, xlabel: str, ylabel: str, path: Path, rotate=75):
    if not labels:
        return
    plt.figure(figsize=(12, 6), dpi=220)
    plt.bar(labels, values)
    plt.xticks(rotation=rotate, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ----------------------------
# Label-based reuse counting using backend
# ----------------------------

def build_label_index_from_graph(g: Graph, entities: Tuple[Set[str], Set[str], Set[str]], backend: Optional[_SBERTBackend]) -> Dict[str, LabelIndex]:
    cls, obj, data = entities
    maps = {
        "cls": extract_entity_labels(g, cls),
        "obj": extract_entity_labels(g, obj),
        "data": extract_entity_labels(g, data),
    }
    pools = {k: set().union(*maps[k].values()) if maps[k] else set() for k in ["cls","obj","data"]}
    return {k: LabelIndex(pools[k], backend) for k in pools}

def count_reuse_by_label_backend(onto_idx: Dict[str, LabelIndex],
                                 pat_g: Optional[Graph],
                                 pat_entities: Tuple[Set[str], Set[str], Set[str]],
                                 threshold: float) -> Tuple[int, int, int]:
    if pat_g is None:
        return 0, 0, 0
    p_cls, p_obj, p_data = pat_entities
    p_cls_labels  = extract_entity_labels(pat_g, p_cls)
    p_obj_labels  = extract_entity_labels(pat_g, p_obj)
    p_data_labels = extract_entity_labels(pat_g, p_data)

    def _count(p_map: Dict[str, Set[str]], idx: LabelIndex) -> int:
        reused = 0
        for p_lbls in p_map.values():
            hit = False
            for pl in p_lbls:
                if idx.max_similarity(pl, threshold) >= threshold:
                    hit = True
                    break
            if hit:
                reused += 1
        return reused

    return (
        _count(p_cls_labels, onto_idx["cls"]),
        _count(p_obj_labels, onto_idx["obj"]),
        _count(p_data_labels, onto_idx["data"]),
    )


# ----------------------------
# Main computation
# ----------------------------

def pct(n, d):
    return (n / d * 100.0) if (d is not None and d and d != 0) else None


def main(base_dir: str,
         out_dir: Path,
         max_bytes: int,
         skip_name_substrings: Set[str],
         sim_backend: str,
         sbert_model: str,
         label_threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # Initialize backend
    backend_obj: Optional[_SBERTBackend] = None
    backend_used = "char"
    if sim_backend in ("sbert", "auto"):
        try:
            backend_obj = _SBERTBackend(sbert_model)
            backend_used = "sbert"
            print(f"[INFO] Using SBERT backend ({sbert_model}) for label similarity.")
        except Exception as e:
            if sim_backend == "sbert":
                print(f"[ERROR] Could not initialize SBERT model '{sbert_model}': {e}", file=sys.stderr)
                sys.exit(1)
            else:
                print("[WARN] sentence-transformers not available or model load failed; falling back to char backend.")
                backend_obj = None
                backend_used = "char"
    else:
        print("[INFO] Using character/word backend for label similarity.")

    print(f"[INFO] Scanning ontologies in: {base_dir}")
    ontology_files = find_ontology_files(base_dir, max_bytes, skip_name_substrings)
    if not ontology_files:
        print("[ERROR] No ontology files found. Check your base directory.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(ontology_files)} ontology files.")

    ont_rows = []     # per ontology
    pat_rows = []     # per pattern file
    reuse_rows = []   # per (ontology_file, pattern_file)
    union_rows = []   # per (ontology_name, pattern_name), union-based

    for onto_path in tqdm(ontology_files, desc="Ontologies"):
        onto_name = Path(onto_path).stem

        og, onto_iri, onto_fallback = safe_parse_graph(onto_path)
        if og is None and onto_iri is None:
            print(f"[WARN] Could not parse or probe IRI for ontology: {onto_path}")
            continue

        o_classes, o_obj, o_data = (set(), set(), set()) if og is None else collect_defined_entities(og)
        pattern_files = find_pattern_files_for(onto_path, max_bytes, skip_name_substrings)
        n_patterns = len(pattern_files)

        ont_rows.append({
            "ontology_file": onto_path,
            "ontology_name": onto_name,
            "ontology_iri": onto_iri or "",
            "used_fallback_for_ontology": onto_fallback,
            "n_classes": len(o_classes),
            "n_object_properties": len(o_obj),
            "n_datatype_properties": len(o_data),
            "n_patterns": n_patterns
        })

        # Build ontology label index once per ontology for SBERT/char
        if og is not None:
            onto_idx = build_label_index_from_graph(og, (o_classes, o_obj, o_data), backend_obj)
        else:
            onto_idx = {"cls": LabelIndex(set(), backend_obj),
                        "obj": LabelIndex(set(), backend_obj),
                        "data": LabelIndex(set(), backend_obj)}

        union_map = defaultdict(lambda: {"cls": set(), "obj": set(), "data": set(), "files": []})

        for p_path in pattern_files:
            p_name = Path(p_path).stem
            pg, p_iri, p_fallback = safe_parse_graph(p_path)

            if pg is not None:
                p_classes, p_obj, p_data = collect_defined_entities(pg)
            else:
                p_classes, p_obj, p_data = set(), set(), set()

            pat_rows.append({
                "pattern_file": p_path,
                "pattern_name": p_name,
                "pattern_iri": p_iri or "",
                "used_fallback_for_pattern": p_fallback,
                "n_classes": len(p_classes),
                "n_object_properties": len(p_obj),
                "n_datatype_properties": len(p_data)
            })

            # Graph-based file-level reuse
            if og is not None and pg is not None:
                r_cls, r_obj, r_data = count_reuse_in_graph(og, p_classes, p_obj, p_data)
            elif og is not None and pg is None:
                r_cls, r_obj, r_data = 0, 0, 0
            else:
                r_cls, r_obj, r_data = 0, 0, 0

            # Label-based file-level reuse (SBERT or fallback)
            lr_cls, lr_obj, lr_data = count_reuse_by_label_backend(
                onto_idx, pg, (p_classes, p_obj, p_data), threshold=label_threshold
            )

            reuse_rows.append({
                "ontology_name": onto_name,
                "ontology_file": onto_path,
                "pattern_name": p_name,
                "pattern_file": p_path,
                # Graph-based
                "reused_classes": r_cls,
                "reused_object_properties": r_obj,
                "reused_datatype_properties": r_data,
                "reused_total": r_cls + r_obj + r_data,
                # Label-based
                "label_reused_classes": lr_cls,
                "label_reused_object_properties": lr_obj,
                "label_reused_datatype_properties": lr_data,
                "label_reused_total": lr_cls + lr_obj + lr_data,
            })

            # Unions payload
            union_map[p_name]["cls"].update(p_classes)
            union_map[p_name]["obj"].update(p_obj)
            union_map[p_name]["data"].update(p_data)
            union_map[p_name]["files"].append(p_path)

        # Per-pattern unions against ontology (graph + label approx via localnames)
        for p_name, payload in union_map.items():
            if og is not None:
                u_cls_reused = sum(1 for iri in payload["cls"] if any_occurs_in_graph(og, URIRef(iri)))
                u_obj_reused = sum(1 for iri in payload["obj"] if any_occurs_in_graph(og, URIRef(iri)))
                u_data_reused = sum(1 for iri in payload["data"] if any_occurs_in_graph(og, URIRef(iri)))

                # Build pattern union label indices (localname-only for speed)
                def _approx_labels(iris: Set[str]) -> Set[str]:
                    return {_normalize_label(_localname(i)) for i in iris if _normalize_label(_localname(i))}

                u_p_cls_labels = LabelIndex(_approx_labels(payload["cls"]), backend_obj)
                u_p_obj_labels = LabelIndex(_approx_labels(payload["obj"]), backend_obj)
                u_p_data_labels = LabelIndex(_approx_labels(payload["data"]), backend_obj)

                # Reuse ontology indices built earlier (onto_idx)
                def _label_union_count(p_idx: LabelIndex, o_idx: LabelIndex) -> int:
                    count = 0
                    for lbl in p_idx.labels:
                        if o_idx.max_similarity(lbl, label_threshold) >= label_threshold:
                            count += 1
                    return count

                u_lr_cls  = _label_union_count(u_p_cls_labels,  onto_idx["cls"])
                u_lr_obj  = _label_union_count(u_p_obj_labels,  onto_idx["obj"])
                u_lr_data = _label_union_count(u_p_data_labels, onto_idx["data"])
            else:
                u_cls_reused = u_obj_reused = u_data_reused = 0
                u_lr_cls = u_lr_obj = u_lr_data = 0

            u_cls = len(payload["cls"])
            u_obj = len(payload["obj"])
            u_data = len(payload["data"])
            u_total = u_cls + u_obj + u_data
            u_reused_total = u_cls_reused + u_obj_reused + u_data_reused
            u_label_reused_total = u_lr_cls + u_lr_obj + u_lr_data

            union_rows.append({
                "ontology_name": onto_name,
                "pattern_name": p_name,
                "pattern_files": ";".join(payload["files"]),
                # Graph-based unions
                "union_n_classes": u_cls,
                "union_n_object_properties": u_obj,
                "union_n_datatype_properties": u_data,
                "union_n_total": u_total,
                "union_reused_classes": u_cls_reused,
                "union_reused_object_properties": u_obj_reused,
                "union_reused_datatype_properties": u_data_reused,
                "union_reused_total": u_reused_total,
                "pct_union_reused_classes": pct(u_cls_reused, u_cls),
                "pct_union_reused_object_properties": pct(u_obj_reused, u_obj),
                "pct_union_reused_datatype_properties": pct(u_data_reused, u_data),
                "pct_union_reused_total": pct(u_reused_total, u_total),
                # Label-based unions
                "union_label_reused_classes": u_lr_cls,
                "union_label_reused_object_properties": u_lr_obj,
                "union_label_reused_datatype_properties": u_lr_data,
                "union_label_reused_total": u_label_reused_total,
                "pct_union_label_reused_total": pct(u_label_reused_total, u_total),
            })

    # ----------------------------
    # DataFrames & base CSVs (defensive)
    # ----------------------------
    ONT_COLS = [
        "ontology_file","ontology_name","ontology_iri","used_fallback_for_ontology",
        "n_classes","n_object_properties","n_datatype_properties","n_patterns"
    ]
    PAT_COLS = [
        "pattern_file","pattern_name","pattern_iri","used_fallback_for_pattern",
        "n_classes","n_object_properties","n_datatype_properties"
    ]
    REUSE_COLS = [
        "ontology_name","ontology_file","pattern_name","pattern_file",
        "reused_classes","reused_object_properties","reused_datatype_properties","reused_total",
        "label_reused_classes","label_reused_object_properties","label_reused_datatype_properties","label_reused_total"
    ]
    UNION_COLS = [
        "ontology_name","pattern_name","pattern_files",
        "union_n_classes","union_n_object_properties","union_n_datatype_properties","union_n_total",
        "union_reused_classes","union_reused_object_properties","union_reused_datatype_properties","union_reused_total",
        "pct_union_reused_classes","pct_union_reused_object_properties","pct_union_reused_datatype_properties","pct_union_reused_total",
        "union_label_reused_classes","union_label_reused_object_properties","union_label_reused_datatype_properties","union_label_reused_total",
        "pct_union_label_reused_total"
    ]

    df_ont   = pd.DataFrame(ont_rows, columns=ONT_COLS)
    df_pat   = pd.DataFrame(pat_rows, columns=PAT_COLS)
    df_reuse = pd.DataFrame(reuse_rows, columns=REUSE_COLS)
    df_union = pd.DataFrame(union_rows, columns=UNION_COLS)

    if not df_ont.empty:   df_ont = df_ont.sort_values(["ontology_name"])
    if not df_pat.empty:   df_pat = df_pat.sort_values(["pattern_name"])
    if not df_reuse.empty: df_reuse = df_reuse.sort_values(["ontology_name","pattern_name"])
    if not df_union.empty: df_union = df_union.sort_values(["ontology_name","pattern_name"])

    if df_pat.empty:
        print("[WARN] No pattern files discovered in any repo (expected under <repo>/patterns/*.owl|*.ttl).")

    # Enrich file-level reuse with ontology & pattern stats
    if not df_reuse.empty:
        df_enriched = df_reuse.merge(
            df_ont[["ontology_name", "n_classes", "n_object_properties", "n_datatype_properties"]],
            on="ontology_name", how="left"
        ).merge(
            df_pat.rename(columns={
                "n_classes": "pattern_n_classes",
                "n_object_properties": "pattern_n_object_properties",
                "n_datatype_properties": "pattern_n_datatype_properties"
            })[["pattern_file", "pattern_name", "pattern_n_classes",
                "pattern_n_object_properties", "pattern_n_datatype_properties"]],
            on=["pattern_file", "pattern_name"], how="left"
        )

        # File-level percentages
        df_enriched["pattern_n_total"] = df_enriched[
            ["pattern_n_classes", "pattern_n_object_properties", "pattern_n_datatype_properties"]
        ].sum(axis=1, min_count=1)

        def _p(a, b): return (a / b * 100.0) if (b and b != 0) else None
        # Graph
        df_enriched["pct_reused_classes"] = df_enriched.apply(lambda r: _p(r["reused_classes"], r["pattern_n_classes"]), axis=1)
        df_enriched["pct_reused_object_properties"] = df_enriched.apply(lambda r: _p(r["reused_object_properties"], r["pattern_n_object_properties"]), axis=1)
        df_enriched["pct_reused_datatype_properties"] = df_enriched.apply(lambda r: _p(r["reused_datatype_properties"], r["pattern_n_datatype_properties"]), axis=1)
        df_enriched["pct_reused_total"] = df_enriched.apply(lambda r: _p(r["reused_total"], r["pattern_n_total"]), axis=1)
        # Labels
        df_enriched["pct_label_reused_classes"] = df_enriched.apply(lambda r: _p(r["label_reused_classes"], r["pattern_n_classes"]), axis=1)
        df_enriched["pct_label_reused_object_properties"] = df_enriched.apply(lambda r: _p(r["label_reused_object_properties"], r["pattern_n_object_properties"]), axis=1)
        df_enriched["pct_label_reused_datatype_properties"] = df_enriched.apply(lambda r: _p(r["label_reused_datatype_properties"], r["pattern_n_datatype_properties"]), axis=1)
        df_enriched["pct_label_reused_total"] = df_enriched.apply(lambda r: _p(r["label_reused_total"], r["pattern_n_total"]), axis=1)
    else:
        df_enriched = pd.DataFrame(columns=[
            "ontology_name","ontology_file","pattern_name","pattern_file",
            "reused_classes","reused_object_properties","reused_datatype_properties","reused_total",
            "label_reused_classes","label_reused_object_properties","label_reused_datatype_properties","label_reused_total",
            "n_classes","n_object_properties","n_datatype_properties",
            "pattern_n_classes","pattern_n_object_properties","pattern_n_datatype_properties",
            "pattern_n_total","pct_reused_classes","pct_reused_object_properties",
            "pct_reused_datatype_properties","pct_reused_total",
            "pct_label_reused_classes","pct_label_reused_object_properties",
            "pct_label_reused_datatype_properties","pct_label_reused_total"
        ])

    # Collapsed-by-name (sum variant)
    if not df_enriched.empty:
        agg_sum = {
            "reused_classes": "sum",
            "reused_object_properties": "sum",
            "reused_datatype_properties": "sum",
            "reused_total": "sum",
            "label_reused_classes": "sum",
            "label_reused_object_properties": "sum",
            "label_reused_datatype_properties": "sum",
            "label_reused_total": "sum",
            "pattern_n_classes": "sum",
            "pattern_n_object_properties": "sum",
            "pattern_n_datatype_properties": "sum"
        }
        collapsed_sum = (df_enriched.groupby(["ontology_name", "pattern_name"], as_index=False).agg(agg_sum))
        collapsed_sum["pattern_n_total"] = collapsed_sum[
            ["pattern_n_classes", "pattern_n_object_properties", "pattern_n_datatype_properties"]
        ].sum(axis=1, min_count=1)

        def _p(a, b): return (a / b * 100.0) if (b and b != 0) else None
        collapsed_sum["pct_reused_total"] = collapsed_sum.apply(lambda r: _p(r["reused_total"], r["pattern_n_total"]), axis=1)
        collapsed_sum["pct_label_reused_total"] = collapsed_sum.apply(lambda r: _p(r["label_reused_total"], r["pattern_n_total"]), axis=1)
    else:
        collapsed_sum = pd.DataFrame(columns=[
            "ontology_name","pattern_name","reused_total","label_reused_total",
            "pattern_n_total","pct_reused_total","pct_label_reused_total"
        ])

    # Write base CSVs
    ont_csv = out_dir / "ontologies_summary.csv"
    pat_csv = out_dir / "patterns_summary.csv"
    reuse_csv = out_dir / "reuse_by_ontology_and_pattern.csv"
    enriched_csv = out_dir / "reuse_by_ontology_and_pattern_enriched.csv"
    collapsed_sum_csv = out_dir / "reuse_collapsed_by_name_sum.csv"
    union_csv = out_dir / "reuse_union_by_name.csv"

    df_ont.to_csv(ont_csv, index=False)
    df_pat.to_csv(pat_csv, index=False)
    df_reuse.to_csv(reuse_csv, index=False)
    df_enriched.to_csv(enriched_csv, index=False)
    collapsed_sum.to_csv(collapsed_sum_csv, index=False)
    df_union.to_csv(union_csv, index=False)

    print(f"[OK] Wrote {ont_csv}")
    print(f"[OK] Wrote {pat_csv}")
    print(f"[OK] Wrote {reuse_csv}")
    print(f"[OK] Wrote {enriched_csv}")
    print(f"[OK] Wrote {collapsed_sum_csv}")
    print(f"[OK] Wrote {union_csv}")

    # ----------------------------
    # Distributions & threshold stats
    # ----------------------------
    chart_dir = out_dir / "charts"
    dist_dir = out_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    def threshold_counts(series: pd.Series, thresholds=(100,90,80,70,60,50,25,10)) -> pd.DataFrame:
        s = series.dropna()
        rows = [{"threshold": t, "count_ge": int((s >= t).sum())} for t in thresholds]
        return pd.DataFrame(rows)

    # Collect series
    s_file_graph  = df_enriched.get("pct_reused_total", pd.Series(dtype=float))
    s_file_label  = df_enriched.get("pct_label_reused_total", pd.Series(dtype=float))
    s_union_graph = df_union.get("pct_union_reused_total", pd.Series(dtype=float))
    s_union_label = df_union.get("pct_union_label_reused_total", pd.Series(dtype=float))

    # Threshold CSVs
    threshold_counts(s_file_graph).to_csv(dist_dir / "thresholds_file_graph.csv", index=False)
    threshold_counts(s_file_label).to_csv(dist_dir / "thresholds_file_labels.csv", index=False)
    threshold_counts(s_union_graph).to_csv(dist_dir / "thresholds_union_graph.csv", index=False)
    threshold_counts(s_union_label).to_csv(dist_dir / "thresholds_union_labels.csv", index=False)

    # Histograms / CDF / Boxplots
    _save_hist(s_file_graph,  "File-level % reused (Graph)",  "% reused (file, graph)",  chart_dir / "hist_file_graph.png")
    _save_hist(s_file_label,  "File-level % reused (Labels)", "% reused (file, labels)", chart_dir / "hist_file_labels.png")
    _save_hist(s_union_graph, "Union % reused (Graph)",       "% reused (union, graph)", chart_dir / "hist_union_graph.png")
    _save_hist(s_union_label, "Union % reused (Labels)",      "% reused (union, labels)",chart_dir / "hist_union_labels.png")

    _save_cdf(s_file_graph,  "CDF: File-level % reused (Graph)",  "% reused (file, graph)",  chart_dir / "cdf_file_graph.png")
    _save_cdf(s_file_label,  "CDF: File-level % reused (Labels)", "% reused (file, labels)", chart_dir / "cdf_file_labels.png")
    _save_cdf(s_union_graph, "CDF: Union % reused (Graph)",       "% reused (union, graph)", chart_dir / "cdf_union_graph.png")
    _save_cdf(s_union_label, "CDF: Union % reused (Labels)",      "% reused (union, labels)",chart_dir / "cdf_union_labels.png")

    _save_box(s_file_graph,  "Boxplot: File-level % reused (Graph)",  "% reused", chart_dir / "box_file_graph.png")
    _save_box(s_file_label,  "Boxplot: File-level % reused (Labels)", "% reused", chart_dir / "box_file_labels.png")
    _save_box(s_union_graph, "Boxplot: Union % reused (Graph)",       "% reused", chart_dir / "box_union_graph.png")
    _save_box(s_union_label, "Boxplot: Union % reused (Labels)",      "% reused", chart_dir / "box_union_labels.png")

    # ----------------------------
    # Per-ontology averages and counts
    # ----------------------------
    if not df_enriched.empty:
        avg_graph_by_onto = (df_enriched.groupby("ontology_name", as_index=False)["pct_reused_total"].mean()
                             .rename(columns={"pct_reused_total": "avg_pct_reused_graph"})
                             .sort_values("avg_pct_reused_graph", ascending=False))
        avg_label_by_onto = (df_enriched.groupby("ontology_name", as_index=False)["pct_label_reused_total"].mean()
                             .rename(columns={"pct_label_reused_total": "avg_pct_reused_labels"})
                             .sort_values("avg_pct_reused_labels", ascending=False))

        avg_graph_by_onto.to_csv(dist_dir / "avg_pct_reused_by_ontology_graph.csv", index=False)
        avg_label_by_onto.to_csv(dist_dir / "avg_pct_reused_by_ontology_labels.csv", index=False)

        top_n = 20
        _save_bar(
            avg_graph_by_onto["ontology_name"].head(top_n).tolist(),
            avg_graph_by_onto["avg_pct_reused_graph"].head(top_n).tolist(),
            f"Top {top_n} Ontologies by Avg % Reused (Graph)",
            "Ontology", "Avg % reused",
            chart_dir / "top_ontologies_avg_pct_reused_graph.png"
        )
        _save_bar(
            avg_label_by_onto["ontology_name"].head(top_n).tolist(),
            avg_label_by_onto["avg_pct_reused_labels"].head(top_n).tolist(),
            f"Top {top_n} Ontologies by Avg % Reused (Labels)",
            "Ontology", "Avg % reused",
            chart_dir / "top_ontologies_avg_pct_reused_labels.png"
        )

    if not df_ont.empty:
        _save_hist(df_ont["n_patterns"], "Distribution of #Patterns per Ontology",
                   "#Patterns", chart_dir / "hist_patterns_per_ontology.png")
        df_ont[["ontology_name","n_patterns"]].to_csv(dist_dir / "patterns_per_ontology.csv", index=False)

    # ----------------------------
    # LaTeX tables (guarded)
    # ----------------------------
    ont_tex = out_dir / "ontologies_summary.tex"
    reuse_tex = out_dir / "reuse_enriched.tex"
    collapsed_sum_tex = out_dir / "reuse_collapsed_by_name_sum.tex"
    union_tex = out_dir / "reuse_union_by_name.tex"

    if not df_ont.empty:
        with open(ont_tex, "w", encoding="utf-8") as f:
            f.write(df_ont.to_latex(index=False, escape=True))
        print(f"[OK] Wrote {ont_tex}")
    else:
        print("[WARN] Skipping LaTeX ontologies table (empty).")

    if not df_enriched.empty:
        latex_cols = [
            "ontology_name","pattern_name",
            # Graph reuse
            "reused_classes","pattern_n_classes","pct_reused_classes",
            "reused_object_properties","pattern_n_object_properties","pct_reused_object_properties",
            "reused_datatype_properties","pattern_n_datatype_properties","pct_reused_datatype_properties",
            "reused_total","pattern_n_total","pct_reused_total",
            # Label reuse
            "label_reused_classes","pct_label_reused_classes",
            "label_reused_object_properties","pct_label_reused_object_properties",
            "label_reused_datatype_properties","pct_label_reused_datatype_properties",
            "label_reused_total","pct_label_reused_total",
        ]
        with open(reuse_tex, "w", encoding="utf-8") as f:
            cols = [c for c in latex_cols if c in df_enriched.columns]
            f.write(df_enriched[cols].to_latex(index=False, float_format="%.1f"))
        print(f"[OK] Wrote {reuse_tex}")
    else:
        print("[WARN] Skipping LaTeX reuse_enriched table (empty).")

    if not collapsed_sum.empty:
        with open(collapsed_sum_tex, "w", encoding="utf-8") as f:
            f.write(collapsed_sum[["ontology_name", "pattern_name",
                                   "reused_total", "pattern_n_total", "pct_reused_total",
                                   "label_reused_total", "pct_label_reused_total"]] \
                    .to_latex(index=False, float_format="%.1f"))
        print(f"[OK] Wrote {collapsed_sum_tex}")
    else:
        print("[WARN] Skipping LaTeX collapsed_by_name_sum table (empty).")

    if not df_union.empty:
        with open(union_tex, "w", encoding="utf-8") as f:
            f.write(df_union[["ontology_name", "pattern_name",
                              "union_reused_total", "union_n_total", "pct_union_reused_total",
                              "union_label_reused_total", "pct_union_label_reused_total"]] \
                    .to_latex(index=False, float_format="%.1f"))
        print(f"[OK] Wrote {union_tex}")
    else:
        print("[WARN] Skipping LaTeX union_by_name table (empty).")

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n=== Summary ===")
    print(f"Similarity backend: {backend_used}")
    print(f"Label threshold: {label_threshold:.2f}")
    print(f"Ontologies analyzed: {len(df_ont)}")
    if not df_ont.empty:
        print(f"Total classes: {df_ont['n_classes'].sum()}")
        print(f"Total object properties: {df_ont['n_object_properties'].sum()}")
        print(f"Total datatype properties: {df_ont['n_datatype_properties'].sum()}")
        print(f"Total patterns (files): {len(df_pat)}")
    if not df_enriched.empty:
        print(f"Ontology–pattern pairs (file-level): {len(df_enriched)}")
        print(f"Total reused entities (graph): {df_enriched['reused_total'].sum()}")
        print(f"Total reused entities (labels): {df_enriched['label_reused_total'].sum()}")
    print(f"[OK] Charts saved under {chart_dir}")
    print(f"[OK] Distributions & thresholds saved under {dist_dir}")


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ODP reuse statistics (graph + SBERT/label-based) with distributions.")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE_DIR,
                        help="Base directory with ontology subfolders.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory for CSVs, LaTeX and charts.")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES,
                        help="Skip files larger than this many bytes (default: 1GB).")
    parser.add_argument("--skip-name", type=str, default="cco",
                        help="Comma-separated substrings; skip files whose names contain any (case-insensitive).")
    parser.add_argument("--label-threshold", type=float, default=LABEL_SIM_THRESHOLD,
                        help="Similarity threshold (0..1) for label-based reuse (default: 0.80).")
    parser.add_argument("--sim-backend", choices=["auto","sbert","char"], default=DEFAULT_SIM_BACKEND,
                        help="Similarity backend for labels: use SBERT, char fallback, or auto-detect.")
    parser.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL,
                        help="Sentence-Transformers model name (e.g., 'all-MiniLM-L6-v2').")
    args = parser.parse_args()

    base_dir = args.base
    out_dir = Path(args.out_dir)
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    skip_set = {s.strip().lower() for s in args.skip_name.split(",") if s.strip()}

    main(
        base_dir=base_dir,
        out_dir=out_dir,
        max_bytes=args.max_bytes,
        skip_name_substrings=skip_set,
        sim_backend=args.sim_backend,
        sbert_model=args.sbert_model,
        label_threshold=float(args.label_threshold),
    )
