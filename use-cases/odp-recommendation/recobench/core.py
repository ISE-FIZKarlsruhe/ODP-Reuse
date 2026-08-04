"""
recobench.core
==============

Shared spine for the ODP-recommender benchmark used in two studies:

  Thread 1 ("does using the ground-truth datasets help?"): a held-out ablation
  that compares recommenders which DO use the datasets (popularity, ontology-kNN
  collaborative filtering, content/embedding retrieval, LLM) against ones that do
  NOT (random; ODPReco's fixed 73-ODP catalog, run separately via the Java jar).

  Thread 2 ("a new open-source-LLM ODP recommender"): the embedding / Qwen
  backends, evaluated on the same protocol so they are directly comparable.

This module owns everything the backends share so they cannot drift apart:

  * data loading + the ontology->reused-pattern REUSE MATRIX (via the repo's
    canonical rq_0_cache/shared_utils.load_ground_truth, keyed on pattern stem),
  * per-ontology TEXT (labels + local names of classes/properties) for content
    methods, extracted with rdflib exactly like rq_0_cache/build_cache.py,
  * the item universe / CATALOG (which pattern stems a recommender may return),
  * the held-out evaluation PROTOCOL (train/test split or leave-one-out),
  * ranking METRICS (P/R/F1@k, MRR, MAP, nDCG@k, hits@k, R-precision),
  * the Recommender base class and the result/CSV writers.

A "recommendation" is a ranked list of pattern STEMS (e.g. "dul", "agentrole",
"partof"). Stems are the only trustworthy GT identity (github pattern files carry
corrupted internal IRIs; see ../README.md). Ground truth for a test ontology is
the set of stems it actually reused.

Dependency-light: stdlib + numpy + rdflib (already required by the repo). Heavy
backends pull their own deps (sentence-transformers, gensim, ollama) lazily.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random as _random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- repo wiring: reuse the canonical loaders so GT is identical to baselines --
#
# This package runs from two places:
#   (a) the ODP-Reuse benchmark repository (use-cases/odp-recommendation/), where
#       the canonical loaders are vendored as `_shared_utils.py`; and
#   (b) a PatternMiner checkout (ODPReco_useCase/), where the upstream
#       `rq_0_cache/shared_utils.py` is authoritative.
# Prefer upstream when it is importable so the two stay in sync; otherwise fall
# back to the vendored copy. Either way the ground truth is identical to the one
# used by every other baseline.
_THIS = Path(__file__).resolve()
_USECASE = _THIS.parent.parent                 # use-cases/odp-recommendation | ODPReco_useCase
_PATTERNMINER_ROOT = _USECASE.parent           # PatternMiner, when running in place
_RQ0 = _PATTERNMINER_ROOT / "rq_0_cache"
if _RQ0.is_dir() and str(_RQ0) not in sys.path:
    sys.path.insert(0, str(_RQ0))

try:
    from shared_utils import (                  # type: ignore  # noqa: E402
        discover_ontologies,
        load_ground_truth,
    )
except ImportError:                             # standalone benchmark repository
    from ._shared_utils import (                # type: ignore  # noqa: E402
        discover_ontologies,
        load_ground_truth,
    )

# rdflib is chatty about malformed date literals in some pattern files; quiet it.
logging.getLogger("rdflib").setLevel(logging.ERROR)
log = logging.getLogger("recobench")

def _resolve_dataset_root(key: str, *candidates: Path) -> Path:
    """Locate a dataset root.

    Order of precedence:
      1. the per-dataset environment variable (ODP_NOOBO_ROOT / ODP_GITHUB_ROOT),
      2. ODP_DATA_ROOT/<key>, if ODP_DATA_ROOT is set,
      3. the first candidate path that exists,
      4. the first candidate (so error messages name a concrete path).
    """
    env = os.environ.get(f"ODP_{key.upper()}_ROOT")
    if env:
        return Path(env)
    data_root = os.environ.get("ODP_DATA_ROOT")
    if data_root:
        p = Path(data_root) / key
        if p.is_dir():
            return p
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


# Repository root when this package lives at use-cases/odp-recommendation/.
_BENCHMARK_ROOT = _USECASE.parent.parent
_REUSING = _BENCHMARK_ROOT / "Ontologies_reusingODPs"

DEFAULT_DATASETS = {
    # Paper/Repo portion: ontologies that are not from the OBO family.
    "noOBO": _resolve_dataset_root(
        "noOBO",
        _REUSING / "notOBO",
        _REUSING / "noOBO",
        _PATTERNMINER_ROOT / "noOBO",
    ),
    # GitHub portion.
    "github": _resolve_dataset_root(
        "github",
        _BENCHMARK_ROOT / "github_dataset",
        _REUSING / "github_dataset",
        _PATTERNMINER_ROOT / "github_dataset",
    ),
}

_STOP = {
    "is", "of", "the", "a", "an", "to", "for", "and", "or", "in", "on", "by",
    "has", "have", "with", "as", "at", "be", "this", "that", "owl", "rdf",
    "rdfs", "xsd", "thing", "ontology", "pattern", "odp",
}


# ======================================================================
# Ontology record: identity, reused-pattern stems, and a text bag
# ======================================================================

@dataclass
class Ontology:
    name: str
    owl_path: Path
    gt_stems: set[str]                 # ground-truth reused pattern stems
    text: str = ""                     # labels + local names (for content/LLM)
    tokens: set[str] = field(default_factory=set)   # normalised term bag (CF)


def _localname(iri: str) -> str:
    """Tail of an IRI: after the last '#' or '/'."""
    s = iri.rstrip("/#")
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_\-]+")


def tokenize(text: str) -> set[str]:
    """Split camelCase / snake_case / kebab, lowercase, drop stopwords + short."""
    parts = _CAMEL.sub(" ", text)
    toks = re.findall(r"[A-Za-z][A-Za-z0-9]+", parts.lower())
    return {t for t in toks if len(t) > 2 and t not in _STOP}


_RE_LABEL = re.compile(
    r"(?:rdfs:label|skos:prefLabel|<label>)[^>]*>([^<]{1,200})<"
    r"|(?:rdfs:label|skos:prefLabel)\s+[\"']([^\"']{1,200})[\"']",
    re.IGNORECASE)
_RE_IRIATTR = re.compile(
    r'(?:rdf:about|rdf:ID|rdf:resource|rdf:datatype)\s*=\s*"([^"]+)"',
    re.IGNORECASE)
_RE_ANGLE = re.compile(r"<https?://[^>\s]+#?([A-Za-z][A-Za-z0-9_]{1,60})>")
_RE_COMMENT = re.compile(r"<rdfs:comment[^>]*>([^<]{1,300})<", re.IGNORECASE)


def _ontology_text(owl_path: Path, max_read_bytes: int = 8_000_000
                   ) -> tuple[str, set[str]]:
    """Fast, format-agnostic term extraction (regex over raw text, capped).

    Pulls rdfs:label / skos:prefLabel text, rdfs:comment text, and the local
    names of every IRI referenced (rdf:about/ID/resource + <...#Name> forms).
    Good enough for content similarity (we need the vocabulary, not the graph)
    and orders of magnitude faster than rdflib on multi-MB OWL files.
    """
    try:
        raw = owl_path.read_bytes()[:max_read_bytes]
        text_blob = raw.decode("utf-8", errors="ignore")
    except OSError:
        return owl_path.stem, tokenize(owl_path.stem)

    words: set[str] = set(tokenize(owl_path.stem))
    for m in _RE_LABEL.finditer(text_blob):
        words |= tokenize(m.group(1) or m.group(2) or "")
    for m in _RE_COMMENT.finditer(text_blob):
        words |= tokenize(m.group(1))
    n = 0
    for m in _RE_IRIATTR.finditer(text_blob):
        words |= tokenize(_localname(m.group(1)))
        n += 1
        if n > 20000:
            break
    for m in _RE_ANGLE.finditer(text_blob):
        words |= tokenize(m.group(1))

    text = " ".join(sorted(words)) if words else owl_path.stem
    return text, set(words)


# ======================================================================
# Dataset loading -> list[Ontology] + reuse matrix
# ======================================================================

_CACHE_DIR = _THIS.parent / ".cache"


def load_approved_labels(path: Path | str | None) -> dict[str, set[str]]:
    """ontology -> pattern stems whose LABEL match a human confirmed as reuse.

    This is what separates the `iri+label` track from `iri`. Only rows a human
    adjudicated as genuine reuse are admitted: automated label matching has a
    measured precision of 0.215, so accepting it unadjudicated would corrupt the
    ground truth rather than extend it.
    """
    if not path:
        return {}
    import csv as _csv
    p = Path(path)
    if not p.is_file():
        log.warning("approved-labels file not found: %s", p)
        return {}
    out: dict[str, set[str]] = {}
    with p.open(encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r.get("human_verdict", "").strip().lower() == "reuse":
                out.setdefault(r["ontology"], set()).add(r["pattern"])
    log.info("approved labels: %d ontologies, %d pairs",
             len(out), sum(len(v) for v in out.values()))
    return out


_RE_ENTITY = re.compile(
    r'(?:rdf:about|rdf:ID|rdf:resource)\s*=\s*"([^"]+)"'
    r'|<(https?://[^>\s]+)>')


def _host_entities(owl_path: Path, max_read_bytes: int = 24_000_000) -> set[str]:
    """IRIs the host ontology mentions, for ground-truth filtering.

    Regex over the raw text rather than rdflib: this is called once per
    ontology and the corpus contains multi-hundred-MB files (cco alone is 1 GB),
    where a full parse costs minutes each. The GT filter only needs to know
    WHICH IRIs appear, not the graph structure, so an over-approximation from
    the serialisation is both sufficient and two orders of magnitude cheaper.

    Being an over-approximation is the safe direction here: it can only admit a
    pattern the strict parse would also admit, never invent one.
    """
    try:
        blob = owl_path.read_bytes()[:max_read_bytes].decode("utf-8", "ignore")
    except OSError:
        return set()
    out: set[str] = set()
    for m in _RE_ENTITY.finditer(blob):
        iri = m.group(1) or m.group(2)
        if iri and iri.startswith("http"):
            out.add(iri)
            if len(out) > 400_000:      # pathological files: enough is enough
                break
    return out


def load_dataset(root: Path, limit: int = 0, with_text: bool = True,
                 log_every: int = 200, use_cache: bool = True,
                 approved_labels: dict[str, set[str]] | None = None,
                 require_overlap: bool = True) -> list[Ontology]:
    """Discover ontologies, load GT reuse stems, and (optionally) text.

    Skips ontologies with no patterns/ ground truth (nothing to evaluate).
    Extracted text is cached on disk keyed by (size,mtime) so re-runs are fast.

    `approved_labels` extends the ground truth with human-confirmed label
    matches (the iri+label track); omit it for the exact-IRI track.

    `require_overlap` (default True) restricts the ground truth to patterns whose
    entities actually occur in the host. Set it False only to reproduce the older,
    weaker ground truth, which counted a pattern as reused merely because the
    harvester placed its file in patterns/.
    """
    pairs = discover_ontologies(root)
    if limit:
        pairs = pairs[:limit]

    cache: dict[str, dict] = {}
    cache_path = _CACHE_DIR / f"{root.name}_text.json"
    if with_text and use_cache and cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    out: list[Ontology] = []
    dirty = False
    for i, (owl, name) in enumerate(pairs):
        # VERIFIED reuse, not harvester association.
        #
        # load_ground_truth() without an entity_allowlist returns one entry per
        # file in patterns/ regardless of whether any entity actually occurs in
        # the host -- it warns about this itself. That makes the task "predict
        # which pattern files the harvester attached", which is easier than, and
        # not comparable to, the gate's verified-reuse criterion. Supplying the
        # host's entity set restricts the ground truth to patterns whose entities
        # are genuinely present.
        allow = _host_entities(owl) if require_overlap else None
        gt = load_ground_truth(owl.parent, entity_allowlist=allow)
        stems = {s.strip().lower() for s, ents in gt.items() if ents}
        # iri+label track: add pattern stems whose label match a human confirmed.
        # These are reuse the exact-IRI ground truth cannot see, because the host
        # re-minted the pattern under its own namespace.
        if approved_labels:
            stems |= {s.strip().lower()
                      for s in approved_labels.get(name, set())}
        if not stems:
            continue
        text, toks = ("", set())
        if with_text:
            try:
                st = owl.stat()
                sig = f"{st.st_size}:{int(st.st_mtime)}"
            except OSError:
                sig = "0:0"
            hit = cache.get(name)
            if hit and hit.get("sig") == sig:
                text = hit["text"]
                toks = set(hit["tokens"])
            else:
                text, toks = _ontology_text(owl)
                cache[name] = {"sig": sig, "text": text, "tokens": sorted(toks)}
                dirty = True
        out.append(Ontology(name=name, owl_path=owl, gt_stems=stems,
                             text=text, tokens=toks))
        if log_every and (i + 1) % log_every == 0:
            log.info("loaded %d/%d ontologies", i + 1, len(pairs))

    if with_text and use_cache and dirty:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    log.info("dataset %s: %d ontologies with GT (of %d discovered)",
             root.name, len(out), len(pairs))
    return out


def reuse_matrix(onts: list[Ontology]) -> dict[str, set[str]]:
    """{ontology_name: set(reused stems)} -- the recommender 'interaction' data."""
    return {o.name: set(o.gt_stems) for o in onts}


def catalog_from(onts: list[Ontology]) -> list[str]:
    """Sorted item universe = every pattern stem seen across `onts` (train)."""
    cat: set[str] = set()
    for o in onts:
        cat |= o.gt_stems
    return sorted(cat)


def pattern_popularity(onts: list[Ontology]) -> dict[str, int]:
    """Train-set document frequency of each stem (how many ontologies reuse it)."""
    pop: dict[str, int] = {}
    for o in onts:
        for s in o.gt_stems:
            pop[s] = pop.get(s, 0) + 1
    return pop


# ======================================================================
# Ranking metrics (one ontology) + aggregation
# ======================================================================

KS = (1, 3, 5, 10)


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ranking_metrics(ranked: list[str], truth: set[str],
                    ks: tuple[int, ...] = KS) -> dict[str, float]:
    """Standard top-k ranking metrics for one ontology.

    ranked: predicted stems, best first (may be shorter/longer than catalog).
    truth:  set of stems actually reused by this ontology (>=1).
    """
    truth = {t.lower() for t in truth}
    ranked = [r.lower() for r in ranked]
    n_truth = len(truth)
    rels = [1 if r in truth else 0 for r in ranked]

    m: dict[str, float] = {}
    for k in ks:
        topk = ranked[:k]
        hit = sum(1 for r in topk if r in truth)
        m[f"precision@{k}"] = hit / k
        m[f"recall@{k}"] = hit / n_truth if n_truth else 0.0
        p, r = m[f"precision@{k}"], m[f"recall@{k}"]
        m[f"f1@{k}"] = (2 * p * r / (p + r)) if (p + r) else 0.0
        m[f"hits@{k}"] = 1.0 if hit else 0.0
        idcg = _dcg([1] * min(k, n_truth))
        m[f"ndcg@{k}"] = (_dcg(rels[:k]) / idcg) if idcg else 0.0

    # MRR: reciprocal rank of the first correct hit
    rr = 0.0
    for i, r in enumerate(rels):
        if r:
            rr = 1.0 / (i + 1)
            break
    m["mrr"] = rr

    # MAP: average precision over the recall points
    hits = 0
    ap = 0.0
    for i, r in enumerate(rels):
        if r:
            hits += 1
            ap += hits / (i + 1)
    m["map"] = ap / n_truth if n_truth else 0.0

    # R-precision: precision at k = |truth|
    rp_hit = sum(1 for r in ranked[:n_truth] if r in truth)
    m["r_precision"] = rp_hit / n_truth if n_truth else 0.0

    m["n_truth"] = float(n_truth)
    m["n_recommended"] = float(len(ranked))
    return m


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Macro-mean each metric over ontologies."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("n_truth", "n_recommended")]
    agg = {f"mean_{k}": round(float(np.mean([r.get(k, 0.0) for r in rows])), 4)
           for k in keys}
    agg["n_ontologies"] = len(rows)
    agg["mean_n_truth"] = round(float(np.mean([r.get("n_truth", 0) for r in rows])), 2)
    return agg


# ======================================================================
# Held-out evaluation protocol
# ======================================================================

@dataclass
class Split:
    train: list[Ontology]
    test: list[Ontology]


def split_train_test(onts: list[Ontology], test_frac: float = 0.3,
                     seed: int = 13) -> Split:
    idx = list(range(len(onts)))
    _random.Random(seed).shuffle(idx)
    n_test = max(1, int(round(len(onts) * test_frac)))
    test_ids = set(idx[:n_test])
    train = [o for i, o in enumerate(onts) if i not in test_ids]
    test = [o for i, o in enumerate(onts) if i in test_ids]
    return Split(train=train, test=test)


class Recommender:
    """Backend interface. Subclasses implement fit() and recommend().

    Cold-start by construction: recommend() sees only the test ontology's
    content (text/tokens/owl_path), never its gt_stems.
    """

    name = "base"
    uses_dataset = True          # False only for random / fixed-catalog baselines

    def fit(self, train: list[Ontology], catalog: list[str]) -> None:
        self.catalog = list(catalog)

    def recommend(self, onto: Ontology, top_n: int = 10) -> list[str]:
        raise NotImplementedError


def evaluate_recommender(rec: Recommender, split: Split, catalog: list[str],
                         top_n: int = 10, ks: tuple[int, ...] = KS
                         ) -> tuple[dict, list[dict]]:
    """Fit on train, predict+score every test ontology. Returns (agg, per_onto)."""
    rec.fit(split.train, catalog)
    per: list[dict] = []
    for o in split.test:
        ranked = [s for s in rec.recommend(o, top_n=top_n) if s]
        m = ranking_metrics(ranked, o.gt_stems, ks=ks)
        m["ontology"] = o.name
        m["top"] = ";".join(ranked[:top_n])
        per.append(m)
    agg = aggregate([{k: v for k, v in r.items()
                      if isinstance(v, (int, float))} for r in per])
    agg["recommender"] = rec.name
    agg["uses_dataset"] = rec.uses_dataset
    return agg, per


def evaluate_loo(make_rec, onts: list[Ontology], catalog: list[str],
                 top_n: int = 10, ks: tuple[int, ...] = KS) -> tuple[dict, list[dict]]:
    """Leave-one-ontology-out. `make_rec` is a 0-arg factory (fresh per fold).

    Used for small datasets (noOBO) where a single split is high-variance.
    The catalog is fixed (passed in) so folds are comparable; train = all but one.
    """
    per: list[dict] = []
    name = make_rec().name
    uses = make_rec().uses_dataset
    for i, test_o in enumerate(onts):
        train = [o for j, o in enumerate(onts) if j != i]
        rec = make_rec()
        rec.fit(train, catalog)
        ranked = [s for s in rec.recommend(test_o, top_n=top_n) if s]
        m = ranking_metrics(ranked, test_o.gt_stems, ks=ks)
        m["ontology"] = test_o.name
        m["top"] = ";".join(ranked[:top_n])
        per.append(m)
    agg = aggregate([{k: v for k, v in r.items()
                      if isinstance(v, (int, float))} for r in per])
    agg["recommender"] = name
    agg["uses_dataset"] = uses
    return agg, per


# ======================================================================
# Output
# ======================================================================

def write_result(out_dir: Path, tag: str, agg: dict, per: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"summary": agg, "per_ontology": per}, indent=2),
        encoding="utf-8")


def write_comparison(out_dir: Path, aggs: list[dict],
                     headline: str = "mean_map") -> Path:
    """Master comparison table (CSV) across recommenders, sorted by `headline`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["recommender", "uses_dataset", "n_ontologies", "mean_n_truth",
            "mean_precision@1", "mean_precision@5", "mean_recall@5",
            "mean_recall@10", "mean_f1@5", "mean_map", "mean_mrr",
            "mean_ndcg@5", "mean_ndcg@10", "mean_hits@1", "mean_hits@5",
            "mean_hits@10", "mean_r_precision"]
    rows = sorted(aggs, key=lambda a: a.get(headline, 0.0), reverse=True)
    import csv
    path = out_dir / "comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path
