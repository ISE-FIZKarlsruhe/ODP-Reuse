#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ODP reuse statistics generator (graph + label-based) with distributions & publication figures.

What this version does
----------------------
- One authoritative chosen_mode from --sim-backend:
    * auto: try SBERT, else char
    * sbert: SBERT or exit on failure
    * char: character/word similarity (with optional edit distance tolerance)
    * exact: exact label equality only
- NEW: --char-max-edits to allow partial matches (Levenshtein <= k edits).
- Output directory is now: <out-dir>/<backend>_<thr10>_<edits>/ e.g. sbert_08_0, char_09_3, exact_10_1
  where <thr10> is threshold * 10 (rounded) with two digits (0–10 → 00–10).
- Graph metrics are always computed.
- Columns/LaTeX/charts adapt to the chosen label method; optional exact columns via --include-exact-columns.
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

import warnings
logging.getLogger("rdflib").setLevel(logging.ERROR)
logging.getLogger("rdflib.term").setLevel(logging.ERROR)
logging.getLogger("rdflib.plugins.parsers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="rdflib")

DEFAULT_BASE_DIR = "/home/ebrahim/odps-extraction/Ontologies_reuseODP/"
DEFAULT_OUT_DIR = "./github_dataset_stats"

ONTOLOGY_EXTS = ("*.owl", "*.ttl", "*.rdf", "*.xml", "*.nt", "*.trig")
PATTERN_EXTS  = ("*.owl", "*.ttl")

DEFAULT_MAX_FILE_BYTES = 3_000_000_000_000  # 1 GB
LABEL_SIM_THRESHOLD = 0.90
DEFAULT_SIM_BACKEND = "auto"  # auto | sbert | char | exact
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"

# ----------------------------
# Helpers: skip, parsing, IRIs
# ----------------------------

def should_skip(path: str, max_bytes: int, skip_name_substrings: Set[str]) -> bool:
    try:
        size_ok = os.path.getsize(path) <= max_bytes
    except OSError:
        size_ok = True
    name = Path(path).name.lower()
    name_ok = not any(sub in name for sub in skip_name_substrings)
    return (not size_ok) or (not name_ok)

def _read_head(path: str, n: int = 120_000) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(n).decode(errors="ignore")
    except Exception:
        return ""

def looks_like_rdf(path: str) -> bool:
    head = _read_head(path)
    if not head:
        return False
    ext = Path(path).suffix.lower()
    if ext in (".owl", ".rdf", ".xml"):
        return any(tok in head for tok in ("<rdf:RDF", "xmlns:rdf=", "xmlns:owl=", "owl:Ontology", ":Ontology"))
    if ext in (".ttl", ".n3"):
        return any(tok in head for tok in ("@prefix", "@base", " a ", " rdf:type "))
    if ext in (".nt", ".trig"):
        return True
    return False

from contextlib import redirect_stderr
import io

def try_parse_any(path: str) -> Optional[Graph]:
    if not looks_like_rdf(path):
        return None
    g = Graph()
    tries = [None, "xml", "turtle", "n3", "nt", "trig", "json-ld"]
    silent = io.StringIO()
    for fmt in tries:
        try:
            with redirect_stderr(silent):
                g.parse(path, format=fmt)
            return g
        except Exception:
            continue
    return None

def get_ontology_iri(g: Graph) -> Optional[str]:
    for s in g.subjects(RDF.type, OWL.Ontology):
        if isinstance(s, URIRef): return str(s)
    return None

IRI_RE = re.compile(r'\b(?:https?|ftp)://[^\s"\'<>]+|\burn:[^\s"\'<>]+', re.I)

def lightweight_iri_probe(path: str) -> Optional[str]:
    head = _read_head(path)
    if not head:
        return None
    m = re.search(r'^\s*Ontology:\s*[<]([^>\s]+)[>]', head, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        iri = m.group(1).strip()
        if iri.startswith(("http://","https://","urn:","ftp://")):
            return iri
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
# Label extraction & similarity
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

# ----- Char backend (+ Levenshtein edits) -----

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

def _levenshtein_bound(a: str, b: str, max_edits: Optional[int] = None) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    if max_edits is not None and abs(la - lb) > max_edits:
        return max_edits + 1
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cj = b[j - 1]
        cur = [j]
        if max_edits is not None:
            i_start = max(1, j - max_edits)
            i_end   = min(la, j + max_edits)
            if i_start > 1:
                cur.extend([max_edits + 1] * (i_start - 1))
        else:
            i_start, i_end = 1, la
        row_min = cur[-1]
        for i in range(i_start, i_end + 1):
            cost = 0 if a[i - 1] == cj else 1
            ins  = cur[-1] + 1
            dele = prev[i] + 1
            sub  = prev[i - 1] + cost
            v = min(ins, dele, sub)
            cur.append(v)
            if v < row_min: row_min = v
        if max_edits is not None and i_end < la:
            cur.extend([max_edits + 1] * (la - i_end))
        if max_edits is not None and row_min > max_edits:
            return max_edits + 1
        prev = cur
    return prev[-1]

def char_label_similarity(a: str, b: str, max_edits: int = 0) -> float:
    na, nb = _normalize_label(a), _normalize_label(b)
    if not na or not nb:
        return 0.0
    if max_edits and _levenshtein_bound(na, nb, max_edits) <= max_edits:
        return 1.0
    ta, tb = _tokenize(na), _tokenize(nb)
    jac = _jaccard(ta, tb)
    rat = _string_ratio(" ".join(ta), " ".join(tb))
    ed = _levenshtein_bound(na, nb, None)
    edit_sim = 1.0 - (ed / max(len(na), len(nb)))
    return max(jac, rat, edit_sim)

# ----- SBERT backend -----

class _SBERTBackend:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return emb / norms

def cosine_max_against_set(vec: np.ndarray, mat: np.ndarray) -> float:
    if mat is None or mat.size == 0:
        return 0.0
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    sims = mat @ vec
    return float(np.max(sims))

class LabelIndex:
    """Holds labels + embedding cache; modes: 'sbert' | 'char' | 'exact'."""
    def __init__(self, labels: Set[str], backend: Optional[_SBERTBackend],
                 mode: str = "char", char_max_edits: int = 0):
        self.labels: List[str] = sorted(l for l in labels if l)
        self.backend = backend if mode == "sbert" else None
        self.mode = mode
        self.char_max_edits = max(0, int(char_max_edits))
        if self.backend and self.labels:
            self.emb = self.backend.encode(self.labels)
        else:
            self.emb = np.zeros((0, 1), dtype=np.float32)

    def max_similarity(self, query_label: str, threshold: float) -> float:
        q = _normalize_label(query_label)
        if not q:
            return 0.0
        if q in self.labels:
            return 1.0
        if self.mode == "sbert" and self.emb.size > 0:
            qv = self.backend.encode([q])[0]
            return cosine_max_against_set(qv, self.emb)
        elif self.mode == "char":
            return max((char_label_similarity(q, l, max_edits=self.char_max_edits) for l in self.labels), default=0.0)
        elif self.mode == "exact":
            return 0.0
        else:
            return 0.0

# ----------------------------
# Discovery
# ----------------------------

def pick_one_main_ontology(files: List[str]) -> Optional[str]:
    if not files:
        return None
    def ext_rank(p: str) -> int:
        e = Path(p).suffix.lower()
        if e == ".owl": return 0
        if e == ".ttl": return 1
        if e in (".rdf",".xml"): return 2
        return 3
    return max(files, key=lambda p: (- (3 - ext_rank(p)), os.path.getsize(p)))

def find_ontology_files(base_dir: str, max_bytes: int, skip_name_substrings: Set[str]) -> List[str]:
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
                if not looks_like_rdf(p):
                    continue
                candidates.append(p)
        main = pick_one_main_ontology(candidates)
        if main:
            print(f"[PICK] {sub} \u2192 {Path(main).name}")
            results.append(main)
    return sorted(results)

def find_pattern_files_for(ontology_file: str, max_bytes: int, skip_name_substrings: Set[str]) -> List[str]:
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
            if not looks_like_rdf(p):
                continue
            files.append(p)
    return sorted(files)

# ----------------------------
# Plot helpers
# ----------------------------

def _save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins=20):
    series = series.dropna()
    if series.empty: return
    plt.figure(figsize=(10, 6), dpi=220)
    plt.hist(series, bins=bins)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def _save_cdf(series: pd.Series, title: str, xlabel: str, path: Path):
    s = series.dropna().sort_values()
    if s.empty: return
    y = (pd.Series(range(1, len(s)+1)) / len(s)).values
    plt.figure(figsize=(10, 6), dpi=220)
    plt.plot(s.values, y)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel); plt.ylabel("Cumulative fraction"); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def _save_box(series: pd.Series, title: str, ylabel: str, path: Path):
    series = series.dropna()
    if series.empty: return
    plt.figure(figsize=(7, 6), dpi=220)
    plt.boxplot(series.values, vert=True, showfliers=False)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def _save_bar(labels: List[str], values: List[float], title: str, xlabel: str, ylabel: str, path: Path, rotate=75):
    if not labels: return
    plt.figure(figsize=(12, 6), dpi=220)
    plt.bar(labels, values)
    plt.xticks(rotation=rotate, ha="right")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

# ----------------------------
# Label backend counting
# ----------------------------

def build_label_index_from_graph(g: Graph,
                                 entities: Tuple[Set[str], Set[str], Set[str]],
                                 backend: Optional[_SBERTBackend],
                                 mode: str,
                                 char_max_edits: int = 0) -> Dict[str, LabelIndex]:
    cls, obj, data = entities
    maps = {
        "cls": extract_entity_labels(g, cls),
        "obj": extract_entity_labels(g, obj),
        "data": extract_entity_labels(g, data),
    }
    pools = {k: set().union(*maps[k].values()) if maps[k] else set() for k in ["cls","obj","data"]}
    return {k: LabelIndex(pools[k], backend, mode=mode, char_max_edits=char_max_edits) for k in pools}

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
            if any(idx.max_similarity(pl, threshold) >= threshold for pl in p_lbls):
                reused += 1
        return reused

    return (
        _count(p_cls_labels, onto_idx["cls"]),
        _count(p_obj_labels, onto_idx["obj"]),
        _count(p_data_labels, onto_idx["data"]),
    )

def count_reuse_by_label_exact(onto_idx_exact: Dict[str, LabelIndex],
                               pat_g: Optional[Graph],
                               pat_entities: Tuple[Set[str], Set[str], Set[str]]) -> Tuple[int, int, int]:
    return count_reuse_by_label_backend(onto_idx_exact, pat_g, pat_entities, threshold=1.0)

# ----------------------------
# Backend resolution + folder token
# ----------------------------

def resolve_backend(sim_backend: str, sbert_model: str):
    if sim_backend == "exact":
        print("[INFO] Using exact label equality only.")
        return "exact", None, "exact"
    if sim_backend == "char":
        print("[INFO] Using character/word backend for label similarity.")
        return "char", None, "char"
    if sim_backend == "sbert":
        try:
            backend_obj = _SBERTBackend(sbert_model)
            print(f"[INFO] Using SBERT backend ({sbert_model}) for label similarity.")
            return "sbert", backend_obj, "sbert"
        except Exception as e:
            print(f"[ERROR] Could not initialize SBERT model '{sbert_model}': {e}", file=sys.stderr)
            sys.exit(1)
    # auto
    try:
        backend_obj = _SBERTBackend(sbert_model)
        print(f"[INFO] Using SBERT backend ({sbert_model}) for label similarity (auto).")
        return "sbert", backend_obj, "sbert"
    except Exception:
        print("[WARN] SBERT unavailable; falling back to char backend (auto).")
        return "char", None, "char"

def _thr_token(x: float) -> str:
    """Map 0.0..1.0 → '00'..'10' (rounded), e.g. 0.90→'09', 1.0→'10'."""
    try:
        v = float(x)
    except Exception:
        v = 0.0
    t = int(round(v * 10))
    t = min(max(t, 0), 10)
    return f"{t:02d}"

def pct(n, d):
    return (n / d * 100.0) if (d is not None and d and d != 0) else None

# ----------------------------
# One backend run (writes to out_dir/<backend>_<thr10>_<edits>/...)
# ----------------------------

def run_one_backend(
    base_dir: str,
    out_dir: Path,
    max_bytes: int,
    skip_name_substrings: Set[str],
    sim_backend: str,
    sbert_model: str,
    label_threshold: float,
    include_exact: bool = False,
    char_max_edits: int = 0
) -> None:
    backend_used, backend_obj, chosen_mode = resolve_backend(sim_backend, sbert_model)

    # Build parameterized subdirectory name like "char_09_3" or "exact_10_1"
    subdir_name = f"{backend_used}_{_thr_token(label_threshold)}_{int(char_max_edits)}"
    backend_subdir = out_dir / subdir_name
    backend_subdir.mkdir(parents=True, exist_ok=True)
    chart_dir = backend_subdir / "charts"; chart_dir.mkdir(parents=True, exist_ok=True)
    dist_dir  = backend_subdir / "distributions"; dist_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning ontologies in: {base_dir}")
    ontology_files = find_ontology_files(base_dir, max_bytes, skip_name_substrings)
    if not ontology_files:
        print("[ERROR] No ontology files found. Check your base directory.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(ontology_files)} ontology files.")

    ont_rows, pat_rows, reuse_rows, union_rows = [], [], [], []

    for onto_path in tqdm(ontology_files, desc=f"Ontologies ({subdir_name})"):
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

        if og is not None:
            onto_idx_fuzzy = build_label_index_from_graph(og, (o_classes, o_obj, o_data), backend_obj, mode=chosen_mode, char_max_edits=char_max_edits)
            onto_idx_exact = build_label_index_from_graph(og, (o_classes, o_obj, o_data), backend=None, mode="exact", char_max_edits=0) if include_exact else None
        else:
            onto_idx_fuzzy = {"cls": LabelIndex(set(), None, mode=chosen_mode, char_max_edits=char_max_edits),
                              "obj": LabelIndex(set(), None, mode=chosen_mode, char_max_edits=char_max_edits),
                              "data": LabelIndex(set(), None, mode=chosen_mode, char_max_edits=char_max_edits)}
            onto_idx_exact = {"cls": LabelIndex(set(), None, mode="exact", char_max_edits=0),
                              "obj": LabelIndex(set(), None, mode="exact", char_max_edits=0),
                              "data": LabelIndex(set(), None, mode="exact", char_max_edits=0)} if include_exact else None

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

            if og is not None and pg is not None:
                r_cls, r_obj, r_data = count_reuse_in_graph(og, p_classes, p_obj, p_data)
            else:
                r_cls = r_obj = r_data = 0

            thr = 1.0 if chosen_mode == "exact" else float(label_threshold)
            lr_cls, lr_obj, lr_data = count_reuse_by_label_backend(
                onto_idx_fuzzy, pg, (p_classes, p_obj, p_data), threshold=thr
            )

            if include_exact and onto_idx_exact is not None:
                ex_cls, ex_obj, ex_data = count_reuse_by_label_exact(
                    onto_idx_exact, pg, (p_classes, p_obj, p_data)
                )

            row = {
                "backend": backend_used,
                "ontology_name": onto_name,
                "ontology_file": onto_path,
                "pattern_name": p_name,
                "pattern_file": p_path,
                "reused_classes": r_cls,
                "reused_object_properties": r_obj,
                "reused_datatype_properties": r_data,
                "reused_total": r_cls + r_obj + r_data,
                "label_reused_classes": lr_cls,
                "label_reused_object_properties": lr_obj,
                "label_reused_datatype_properties": lr_data,
                "label_reused_total": lr_cls + lr_obj + lr_data,
            }
            if include_exact:
                row.update({
                    "exact_label_reused_classes": ex_cls,
                    "exact_label_reused_object_properties": ex_obj,
                    "exact_label_reused_datatype_properties": ex_data,
                    "exact_label_reused_total": ex_cls + ex_obj + ex_data,
                })
            reuse_rows.append(row)

            union_map[p_name]["cls"].update(p_classes)
            union_map[p_name]["obj"].update(p_obj)
            union_map[p_name]["data"].update(p_data)
            union_map[p_name]["files"].append(p_path)

        # per-pattern unions (respect chosen_mode)
        for p_name, payload in union_map.items():
            if og is not None:
                u_cls_reused = sum(1 for iri in payload["cls"] if any_occurs_in_graph(og, URIRef(iri)))
                u_obj_reused = sum(1 for iri in payload["obj"] if any_occurs_in_graph(og, URIRef(iri)))
                u_data_reused = sum(1 for iri in payload["data"] if any_occurs_in_graph(og, URIRef(iri)))

                def _approx_labels(iris: Set[str]) -> Set[str]:
                    return {_normalize_label(_localname(i)) for i in iris if _normalize_label(_localname(i))}

                u_p_cls_labels  = LabelIndex(_approx_labels(payload["cls"]),  backend_obj, mode=chosen_mode, char_max_edits=char_max_edits)
                u_p_obj_labels  = LabelIndex(_approx_labels(payload["obj"]),  backend_obj, mode=chosen_mode, char_max_edits=char_max_edits)
                u_p_data_labels = LabelIndex(_approx_labels(payload["data"]), backend_obj, mode=chosen_mode, char_max_edits=char_max_edits)

                if include_exact:
                    u_p_cls_labels_exact  = LabelIndex(u_p_cls_labels.labels,  None, mode="exact", char_max_edits=0)
                    u_p_obj_labels_exact  = LabelIndex(u_p_obj_labels.labels,  None, mode="exact", char_max_edits=0)
                    u_p_data_labels_exact = LabelIndex(u_p_data_labels.labels, None, mode="exact", char_max_edits=0)

                def _count_thr(p_idx: LabelIndex, o_idx: LabelIndex, thr: float) -> int:
                    return sum(1 for lbl in p_idx.labels if o_idx.max_similarity(lbl, thr) >= thr)

                thr = 1.0 if chosen_mode == "exact" else float(label_threshold)
                u_lr_cls  = _count_thr(u_p_cls_labels,  onto_idx_fuzzy["cls"],  thr)
                u_lr_obj  = _count_thr(u_p_obj_labels,  onto_idx_fuzzy["obj"],  thr)
                u_lr_data = _count_thr(u_p_data_labels, onto_idx_fuzzy["data"], thr)

                if include_exact:
                    u_ex_cls  = _count_thr(u_p_cls_labels_exact,  onto_idx_exact["cls"],  1.0)
                    u_ex_obj  = _count_thr(u_p_obj_labels_exact,  onto_idx_exact["obj"],  1.0)
                    u_ex_data = _count_thr(u_p_data_labels_exact, onto_idx_exact["data"], 1.0)
            else:
                u_cls_reused = u_obj_reused = u_data_reused = 0
                u_lr_cls = u_lr_obj = u_lr_data = 0
                if include_exact:
                    u_ex_cls = u_ex_obj = u_ex_data = 0

            u_cls, u_obj, u_data = len(payload["cls"]), len(payload["obj"]), len(payload["data"])
            u_total = u_cls + u_obj + u_data
            u_reused_total = u_cls_reused + u_obj_reused + u_data_reused
            u_label_reused_total = u_lr_cls + u_lr_obj + u_lr_data

            union_row = {
                "backend": backend_used,
                "ontology_name": onto_name,
                "pattern_name": p_name,
                "pattern_files": ";".join(payload["files"]),
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
                "union_label_reused_classes": u_lr_cls,
                "union_label_reused_object_properties": u_lr_obj,
                "union_label_reused_datatype_properties": u_lr_data,
                "union_label_reused_total": u_label_reused_total,
                "pct_union_label_reused_total": pct(u_label_reused_total, u_total),
            }
            if include_exact:
                u_exact_total = u_ex_cls + u_ex_obj + u_ex_data
                union_row.update({
                    "union_exact_label_reused_classes": u_ex_cls,
                    "union_exact_label_reused_object_properties": u_ex_obj,
                    "union_exact_label_reused_datatype_properties": u_ex_data,
                    "union_exact_label_reused_total": u_exact_total,
                    "pct_union_exact_label_reused_total": pct(u_exact_total, u_total),
                })
            union_rows.append(union_row)

    # ===== DataFrames & outputs =====

    ONT_COLS = [
        "ontology_file","ontology_name","ontology_iri","used_fallback_for_ontology",
        "n_classes","n_object_properties","n_datatype_properties","n_patterns"
    ]
    PAT_COLS = [
        "pattern_file","pattern_name","pattern_iri","used_fallback_for_pattern",
        "n_classes","n_object_properties","n_datatype_properties"
    ]
    REUSE_COLS = [
        "backend","ontology_name","ontology_file","pattern_name","pattern_file",
        "reused_classes","reused_object_properties","reused_datatype_properties","reused_total",
        "label_reused_classes","label_reused_object_properties","label_reused_datatype_properties","label_reused_total",
    ]
    if include_exact:
        REUSE_COLS += [
            "exact_label_reused_classes","exact_label_reused_object_properties","exact_label_reused_datatype_properties","exact_label_reused_total"
        ]
    UNION_COLS = [
        "backend","ontology_name","pattern_name","pattern_files",
        "union_n_classes","union_n_object_properties","union_n_datatype_properties","union_n_total",
        "union_reused_classes","union_reused_object_properties","union_reused_datatype_properties","union_reused_total",
        "pct_union_reused_classes","pct_union_reused_object_properties","pct_union_reused_datatype_properties","pct_union_reused_total",
        "union_label_reused_classes","union_label_reused_object_properties","union_label_reused_datatype_properties","union_label_reused_total",
        "pct_union_label_reused_total",
    ]
    if include_exact:
        UNION_COLS += [
            "union_exact_label_reused_classes","union_exact_label_reused_object_properties","union_exact_label_reused_datatype_properties","union_exact_label_reused_total",
            "pct_union_exact_label_reused_total",
        ]

    df_ont   = pd.DataFrame(ont_rows, columns=ONT_COLS).sort_values(["ontology_name"]) if ont_rows else pd.DataFrame(columns=ONT_COLS)
    df_pat   = pd.DataFrame(pat_rows, columns=PAT_COLS).sort_values(["pattern_name"]) if pat_rows else pd.DataFrame(columns=PAT_COLS)
    df_reuse = pd.DataFrame(reuse_rows, columns=REUSE_COLS).sort_values(["ontology_name","pattern_name"]) if reuse_rows else pd.DataFrame(columns=REUSE_COLS)
    df_union = pd.DataFrame(union_rows, columns=UNION_COLS).sort_values(["ontology_name","pattern_name"]) if union_rows else pd.DataFrame(columns=UNION_COLS)

    # Enriched join for percentages (file-level)
    if not df_reuse.empty:
        df_enriched = df_reuse.merge(
            df_ont[["ontology_name","n_classes","n_object_properties","n_datatype_properties"]],
            on="ontology_name", how="left"
        ).merge(
            df_pat.rename(columns={
                "n_classes":"pattern_n_classes",
                "n_object_properties":"pattern_n_object_properties",
                "n_datatype_properties":"pattern_n_datatype_properties"
            })[["pattern_file","pattern_name","pattern_n_classes","pattern_n_object_properties","pattern_n_datatype_properties"]],
            on=["pattern_file","pattern_name"], how="left"
        )
        df_enriched["pattern_n_total"] = df_enriched[["pattern_n_classes","pattern_n_object_properties","pattern_n_datatype_properties"]].sum(axis=1, min_count=1)

        def _p(a,b): return (a/b*100.0) if (b and b!=0) else None
        # Graph %
        df_enriched["pct_reused_classes"] = df_enriched.apply(lambda r: _p(r["reused_classes"], r["pattern_n_classes"]), axis=1)
        df_enriched["pct_reused_object_properties"] = df_enriched.apply(lambda r: _p(r["reused_object_properties"], r["pattern_n_object_properties"]), axis=1)
        df_enriched["pct_reused_datatype_properties"] = df_enriched.apply(lambda r: _p(r["reused_datatype_properties"], r["pattern_n_datatype_properties"]), axis=1)
        df_enriched["pct_reused_total"] = df_enriched.apply(lambda r: _p(r["reused_total"], r["pattern_n_total"]), axis=1)
        # Label (chosen backend) %
        df_enriched["pct_label_reused_classes"] = df_enriched.apply(lambda r: _p(r["label_reused_classes"], r["pattern_n_classes"]), axis=1)
        df_enriched["pct_label_reused_object_properties"] = df_enriched.apply(lambda r: _p(r["label_reused_object_properties"], r["pattern_n_object_properties"]), axis=1)
        df_enriched["pct_label_reused_datatype_properties"] = df_enriched.apply(lambda r: _p(r["label_reused_datatype_properties"], r["pattern_n_datatype_properties"]), axis=1)
        df_enriched["pct_label_reused_total"] = df_enriched.apply(lambda r: _p(r["label_reused_total"], r["pattern_n_total"]), axis=1)
        # Optional exact %
        if include_exact:
            df_enriched["pct_exact_label_reused_classes"] = df_enriched.apply(lambda r: _p(r["exact_label_reused_classes"], r["pattern_n_classes"]), axis=1)
            df_enriched["pct_exact_label_reused_object_properties"] = df_enriched.apply(lambda r: _p(r["exact_label_reused_object_properties"], r["pattern_n_object_properties"]), axis=1)
            df_enriched["pct_exact_label_reused_datatype_properties"] = df_enriched.apply(lambda r: _p(r["exact_label_reused_datatype_properties"], r["pattern_n_datatype_properties"]), axis=1)
            df_enriched["pct_exact_label_reused_total"] = df_enriched.apply(lambda r: _p(r["exact_label_reused_total"], r["pattern_n_total"]), axis=1)
    else:
        df_enriched = pd.DataFrame()

    # Collapsed-by-name (sum)
    if not df_enriched.empty:
        agg_sum = {
            "reused_total":"sum","label_reused_total":"sum",
            "pattern_n_classes":"sum","pattern_n_object_properties":"sum","pattern_n_datatype_properties":"sum"
        }
        if include_exact:
            agg_sum["exact_label_reused_total"] = "sum"

        collapsed_sum = (df_enriched.groupby(["backend","ontology_name","pattern_name"], as_index=False).agg(agg_sum))
        collapsed_sum["pattern_n_total"] = collapsed_sum[["pattern_n_classes","pattern_n_object_properties","pattern_n_datatype_properties"]].sum(axis=1, min_count=1)
        def _p(a,b): return (a/b*100.0) if (b and b!=0) else None
        collapsed_sum["pct_reused_total"] = collapsed_sum.apply(lambda r: _p(r["reused_total"], r["pattern_n_total"]), axis=1)
        collapsed_sum["pct_label_reused_total"] = collapsed_sum.apply(lambda r: _p(r["label_reused_total"], r["pattern_n_total"]), axis=1)
        if include_exact:
            collapsed_sum["pct_exact_label_reused_total"] = collapsed_sum.apply(lambda r: _p(r["exact_label_reused_total"], r["pattern_n_total"]), axis=1)
    else:
        collapsed_sum = pd.DataFrame()

    # ---- write CSVs ----
    ont_csv          = backend_subdir / "ontologies_summary.csv"
    pat_csv          = backend_subdir / "patterns_summary.csv"
    reuse_csv        = backend_subdir / "reuse_by_ontology_and_pattern.csv"
    enriched_csv     = backend_subdir / "reuse_by_ontology_and_pattern_enriched.csv"
    collapsed_sum_csv= backend_subdir / "reuse_collapsed_by_name_sum.csv"
    union_csv        = backend_subdir / "reuse_union_by_name.csv"

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

    # ---- distributions & charts (GLOBAL unique patterns) ----
    if not df_union.empty:
        agg = {
            "union_n_classes":"max","union_n_object_properties":"max","union_n_datatype_properties":"max","union_n_total":"max",
            "union_reused_classes":"max","union_reused_object_properties":"max","union_reused_datatype_properties":"max","union_reused_total":"max",
            "pct_union_reused_total":"max",
            "union_label_reused_classes":"max","union_label_reused_object_properties":"max","union_label_reused_datatype_properties":"max","union_label_reused_total":"max",
            "pct_union_label_reused_total":"max",
        }
        if include_exact:
            agg.update({"union_exact_label_reused_total":"max","pct_union_exact_label_reused_total":"max"})
        df_union_unique = df_union.groupby("pattern_name", as_index=False).agg(agg).sort_values("pattern_name")
    else:
        df_union_unique = pd.DataFrame()

    (dist_dir / "global_unique_patterns_union.csv").write_text(
        df_union_unique.to_csv(index=False) if not df_union_unique.empty else "pattern_name\n"
    )

    def _safe_float(x):
        try: return float(x)
        except Exception: return None

    s_unique_graph = df_union_unique.get("pct_union_reused_total", pd.Series(dtype=float)).map(_safe_float)
    s_unique_label = df_union_unique.get("pct_union_label_reused_total", pd.Series(dtype=float)).map(_safe_float)

    def threshold_counts(series: pd.Series, thresholds=(100,90,80,70,60,50,25,10)) -> pd.DataFrame:
        s = series.dropna()
        return pd.DataFrame([{"threshold": t, "count_ge": int((s >= t).sum())} for t in thresholds])

    threshold_counts(s_unique_graph).to_csv(dist_dir / "thresholds_global_unique_patterns_graph.csv", index=False)
    threshold_counts(s_unique_label).to_csv(dist_dir / "thresholds_global_unique_patterns_labels.csv", index=False)

    nice = subdir_name  # stamp on titles
    _save_hist(s_unique_graph, f"Unique-pattern % reused (Graph, Global) — {nice}",  "% reused (unique pattern, graph)",  chart_dir / "hist_unique_patterns_graph_global.png")
    _save_hist(s_unique_label, f"Unique-pattern % reused (Labels, Global) — {nice}", "% reused (unique pattern, labels)", chart_dir / "hist_unique_patterns_labels_global.png")
    _save_cdf(s_unique_graph, f"CDF: Unique-pattern % reused (Graph, Global) — {nice}",  "% reused (unique pattern, graph)",  chart_dir / "cdf_unique_patterns_graph_global.png")
    _save_cdf(s_unique_label, f"CDF: Unique-pattern % reused (Labels, Global) — {nice}", "% reused (unique pattern, labels)", chart_dir / "cdf_unique_patterns_labels_global.png")
    _save_box(s_unique_graph, f"Boxplot: Unique-pattern % reused (Graph, Global) — {nice}",  "% reused", chart_dir / "box_unique_patterns_graph_global.png")
    _save_box(s_unique_label, f"Boxplot: Unique-pattern % reused (Labels, Global) — {nice}", "% reused", chart_dir / "box_unique_patterns_labels_global.png")

    def _bin_edges_and_labels():
        return [
            ("eq_100",     lambda v: v == 100.0),
            ("ge95_lt100", lambda v: (v is not None) and (v < 100.0) and (v >= 95.0)),
            ("ge90_lt95",  lambda v: (v is not None) and (v < 95.0)  and (v >= 90.0)),
            ("ge85_lt90",  lambda v: (v is not None) and (v < 90.0)  and (v >= 85.0)),
            ("ge80_lt85",  lambda v: (v is not None) and (v < 85.0)  and (v >= 80.0)),
        ]
    def _global_counts_by_bin(series: pd.Series) -> pd.DataFrame:
        s = series.dropna()
        rows = []
        for bname, pred in _bin_edges_and_labels():
            rows.append({"bin": bname, "count": int(sum(1 for v in s if pred(v)))})
        return pd.DataFrame(rows).sort_values("count", ascending=False)

    graph_bins = _global_counts_by_bin(s_unique_graph)
    graph_bins.to_csv(dist_dir / "global_unique_patterns_bins_graph.csv", index=False)
    plt.figure(figsize=(10, 6), dpi=220); plt.bar(graph_bins["bin"], graph_bins["count"])
    plt.xticks(rotation=0); plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xlabel("Reuse % bin"); plt.ylabel("# unique patterns")
    plt.title(f"Unique Patterns by Reuse % Bins (Graph, Global) — {nice}")
    plt.tight_layout(); plt.savefig(chart_dir / "global_unique_patterns_bins_graph.png"); plt.close()

    label_bins = _global_counts_by_bin(s_unique_label)
    label_bins.to_csv(dist_dir / "global_unique_patterns_bins_labels.csv", index=False)
    plt.figure(figsize=(10, 6), dpi=220); plt.bar(label_bins["bin"], label_bins["count"])
    plt.xticks(rotation=0); plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xlabel("Reuse % bin"); plt.ylabel("# unique patterns")
    plt.title(f"Unique Patterns by Reuse % Bins (Labels, Global) — {nice}")
    plt.tight_layout(); plt.savefig(chart_dir / "global_unique_patterns_bins_labels.png"); plt.close()

    # per-ontology averages
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
        _save_bar(avg_graph_by_onto["ontology_name"].head(top_n).tolist(),
                  avg_graph_by_onto["avg_pct_reused_graph"].head(top_n).tolist(),
                  f"Top {top_n} Ontologies by Avg % Reused (Graph) — {nice}",
                  "Ontology","Avg % reused",
                  chart_dir / "top_ontologies_avg_pct_reused_graph.png")
        _save_bar(avg_label_by_onto["ontology_name"].head(top_n).tolist(),
                  avg_label_by_onto["avg_pct_reused_labels"].head(top_n).tolist(),
                  f"Top {top_n} Ontologies by Avg % Reused (Labels) — {nice}",
                  "Ontology","Avg % reused",
                  chart_dir / "top_ontologies_avg_pct_reused_labels.png")

    if not df_ont.empty:
        _save_hist(df_ont["n_patterns"], "Distribution of #Patterns per Ontology", "#Patterns", chart_dir / "hist_patterns_per_ontology.png")
        df_ont[["ontology_name","n_patterns"]].to_csv(dist_dir / "patterns_per_ontology.csv", index=False)

    # LaTeX tables
    ont_tex          = backend_subdir / "ontologies_summary.tex"
    reuse_tex        = backend_subdir / "reuse_enriched.tex"
    collapsed_sum_tex= backend_subdir / "reuse_collapsed_by_name_sum.tex"
    union_tex        = backend_subdir / "reuse_union_by_name.tex"

    if not df_ont.empty:
        with open(ont_tex, "w", encoding="utf-8") as f:
            f.write(df_ont.to_latex(index=False, escape=True))
        print(f"[OK] Wrote {ont_tex}")
    else:
        print("[WARN] Skipping LaTeX ontologies table (empty).")

    if not df_enriched.empty:
        latex_cols = [c for c in [
            "backend","ontology_name","pattern_name",
            "reused_classes","pattern_n_classes","pct_reused_classes",
            "reused_object_properties","pattern_n_object_properties","pct_reused_object_properties",
            "reused_datatype_properties","pattern_n_datatype_properties","pct_reused_datatype_properties",
            "reused_total","pattern_n_total","pct_reused_total",
            "label_reused_classes","pct_label_reused_classes",
            "label_reused_object_properties","pct_label_reused_object_properties",
            "label_reused_datatype_properties","pct_label_reused_datatype_properties",
            "label_reused_total","pct_label_reused_total",
        ] if c in df_enriched.columns]
        if include_exact:
            latex_cols += [c for c in [
                "exact_label_reused_total","pct_exact_label_reused_total"
            ] if c in df_enriched.columns]
        with open(reuse_tex, "w", encoding="utf-8") as f:
            f.write(df_enriched[latex_cols].to_latex(index=False, float_format="%.1f"))
        print(f"[OK] Wrote {reuse_tex}")
    else:
        print("[WARN] Skipping LaTeX reuse_enriched table (empty).")

    if not collapsed_sum.empty:
        base_cols = [
            "backend","ontology_name","pattern_name",
            "reused_total","pattern_n_total","pct_reused_total",
            "label_reused_total","pct_label_reused_total",
        ]
        if include_exact and "pct_exact_label_reused_total" in collapsed_sum.columns:
            base_cols += ["exact_label_reused_total","pct_exact_label_reused_total"]
        with open(collapsed_sum_tex, "w", encoding="utf-8") as f:
            f.write(collapsed_sum[base_cols].to_latex(index=False, float_format="%.1f"))
        print(f"[OK] Wrote {collapsed_sum_tex}")
    else:
        print("[WARN] Skipping LaTeX collapsed_by_name_sum table (empty).")

    if not df_union.empty:
        base_cols = [
            "backend","ontology_name","pattern_name",
            "union_reused_total","union_n_total","pct_union_reused_total",
            "union_label_reused_total","pct_union_label_reused_total",
        ]
        if include_exact and "pct_union_exact_label_reused_total" in df_union.columns:
            base_cols += ["union_exact_label_reused_total","pct_union_exact_label_reused_total"]
        with open(union_tex, "w", encoding="utf-8") as f:
            f.write(df_union[base_cols].to_latex(index=False, float_format="%.1f"))
        print(f"[OK] Wrote {union_tex}")
    else:
        print("[WARN] Skipping LaTeX union_by_name table (empty).")

    # summary
    print("\n=== Summary ===")
    print(f"Output subdir: {backend_subdir}")
    print(f"Similarity backend: {backend_used} (mode={chosen_mode})")
    print(f"Label threshold: {label_threshold:.2f} (ignored for exact)")
    print(f"Char max edits: {char_max_edits}")
    print(f"Ontologies analyzed: {len(df_ont)}")
    if not df_ont.empty:
        print(f"Total classes: {df_ont['n_classes'].sum()}")
        print(f"Total object properties: {df_ont['n_object_properties'].sum()}")
        print(f"Total datatype properties: {df_ont['n_datatype_properties'].sum()}")
        print(f"Total patterns (files): {len(df_pat)}")
    if not df_enriched.empty:
        print(f"Ontology–pattern pairs (file-level): {len(df_enriched)}")
        print(f"Total reused entities (graph): {df_enriched['reused_total'].sum()}")
        print(f"Total reused entities (labels/{backend_used}): {df_enriched['label_reused_total'].sum()}")
        if include_exact and "exact_label_reused_total" in df_enriched:
            print(f"Total reused entities (labels/exact): {df_enriched['exact_label_reused_total'].sum()}")
    print(f"[OK] Charts saved under {chart_dir}")
    print(f"[OK] Distributions & thresholds saved under {dist_dir}")

# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute ODP reuse statistics (graph + label-based) with distributions.")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE_DIR,
                        help="Base directory with ontology subfolders.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Output directory for CSVs, LaTeX and charts (subfolder is parameterized).")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES,
                        help="Skip files larger than this many bytes (default: 1GB).")
    parser.add_argument("--skip-name", type=str, default="",
                        help="Comma-separated substrings; skip files whose names contain any (case-insensitive).")

    parser.add_argument("--label-threshold", type=float, default=LABEL_SIM_THRESHOLD,
                        help="Similarity threshold (0..1) for label-based reuse (default: 0.90).")
    parser.add_argument("--sim-backend", choices=["auto","sbert","char","exact"], default=DEFAULT_SIM_BACKEND,
                        help="Similarity backend for labels.")
    parser.add_argument("--all-backends", action="store_true",
                        help="Run auto, sbert, char, exact (writes under separate parameterized subfolders).")
    parser.add_argument("--include-exact-columns", action="store_true",
                        help="Also compute and emit exact label reuse columns alongside the selected backend.")
    parser.add_argument("--sbert-model", type=str, default=DEFAULT_SBERT_MODEL,
                        help="Sentence-Transformers model name (e.g., 'all-MiniLM-L6-v2').")
    parser.add_argument("--char-max-edits", type=int, default=0,
                        help="For char/auto fallback, allow up to this many Levenshtein edits to count as a perfect match (partial match tolerance).")

    args = parser.parse_args()

    base_dir = args.base
    out_dir = Path(args.out_dir)
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Base directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    skip_set = {s.strip().lower() for s in args.skip_name.split(",") if s.strip()}

    if args.all_backends:
        for backend in ["auto","sbert","char","exact"]:
            run_one_backend(
                base_dir=base_dir,
                out_dir=out_dir,
                max_bytes=args.max_bytes,
                skip_name_substrings=skip_set,
                sim_backend=backend,
                sbert_model=args.sbert_model,
                label_threshold=float(args.label_threshold),
                include_exact=args.include_exact_columns,
                char_max_edits=int(args.char_max_edits),
            )
    else:
        run_one_backend(
            base_dir=base_dir,
            out_dir=out_dir,
            max_bytes=args.max_bytes,
            skip_name_substrings=skip_set,
            sim_backend=args.sim_backend,
            sbert_model=args.sbert_model,
            label_threshold=float(args.label_threshold),
            include_exact=args.include_exact_columns,
            char_max_edits=int(args.char_max_edits),
        )
