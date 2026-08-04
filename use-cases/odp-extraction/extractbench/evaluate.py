"""
Clustering and entity-level evaluation for automatic ODP extraction.

METRICS
-------
Identical to the published definition. For a predicted cluster p and a
ground-truth pattern instance g:

    precision = |p & g| / |p|,   recall = |p & g| / |g|,
    F1        = 2PR / (P + R)

Predicted clusters are matched to ground-truth patterns by Hungarian assignment
on the cost matrix 1 - F1, then precision/recall/F1 are macro-averaged over the
matched pairs of one ontology, and finally over ontologies.

MODEL SELECTION -- why two numbers are reported
-----------------------------------------------
PatternMiner selects the clustering hyper-parameters (k, linkage, HDBSCAN
min_cluster_size, ...) per ontology by argmax of macro entity F1 **against the
ground truth** (`rq_0_cache/rich_eval.py:312/392`). That is an oracle: it uses
the labels it is being scored against, so the resulting F1 is an upper bound, not
an achievable score: a method cannot be said to reach an F1 that was obtained by
consulting the answer key.

This module therefore reports both, side by side, for every configuration:

  ORACLE        the best F1 over the hyper-parameter grid, chosen using the GT.
                Comparable to the previously published numbers. Label it as an
                upper bound wherever it appears.
  UNSUPERVISED  hyper-parameters chosen without the GT -- k by silhouette score
                over the same grid, HDBSCAN by its own density criterion. This
                is the honest, achievable number.

Reporting the gap between them is itself informative: it quantifies how much of
the published performance came from oracle model selection.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning)

K_GRID = tuple(range(2, 11))          # k = 2..10, as in PatternMiner's grid
MIN_CLUSTER_SIZES = (2, 3, 5)         # HDBSCAN grid


# ----------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------

def _labels_to_clusters(entities: list[str], labels) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for ent, lab in zip(entities, labels):
        lab = int(lab)
        if lab < 0:            # HDBSCAN noise
            continue
        out.setdefault(lab, set()).add(ent)
    return out


def kmeans_candidates(emb: np.ndarray, entities: list[str]):
    """Yield (config_name, clusters, silhouette) for each k in the grid."""
    n = len(entities)
    for k in K_GRID:
        if k >= n:
            continue
        try:
            lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)
        except Exception:  # noqa: BLE001
            continue
        sil = _safe_silhouette(emb, lab)
        yield f"kmeans_k{k}", _labels_to_clusters(entities, lab), sil


def hdbscan_candidates(emb: np.ndarray, entities: list[str]):
    """Yield (config_name, clusters, silhouette) for the HDBSCAN grid.

    Uses sklearn's HDBSCAN when available, else falls back to agglomerative
    clustering with a distance threshold (recorded in the config name so the
    fallback is never silently mistaken for HDBSCAN).
    """
    n = len(entities)
    try:
        from sklearn.cluster import HDBSCAN
        for mcs in MIN_CLUSTER_SIZES:
            if mcs >= n:
                continue
            try:
                lab = HDBSCAN(min_cluster_size=mcs).fit_predict(emb)
            except Exception:  # noqa: BLE001
                continue
            yield (f"hdbscan_mcs{mcs}", _labels_to_clusters(entities, lab),
                   _safe_silhouette(emb, lab))
    except ImportError:
        for thr in (0.25, 0.5, 0.75):
            try:
                lab = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=thr,
                    metric="cosine", linkage="average").fit_predict(emb)
            except Exception:  # noqa: BLE001
                continue
            yield (f"agglom_thr{thr}", _labels_to_clusters(entities, lab),
                   _safe_silhouette(emb, lab))


def _safe_silhouette(emb: np.ndarray, labels) -> float:
    """Silhouette over non-noise points; -1.0 when undefined."""
    mask = np.asarray(labels) >= 0
    lab = np.asarray(labels)[mask]
    if lab.size < 3 or len(set(lab.tolist())) < 2:
        return -1.0
    try:
        return float(silhouette_score(emb[mask], lab))
    except Exception:  # noqa: BLE001
        return -1.0


CLUSTERERS = {"kmeans": kmeans_candidates, "hdbscan": hdbscan_candidates}


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------

def match_and_score(clusters: dict[int, set[str]],
                    gt: dict[str, set[str]]) -> dict:
    """Hungarian-match clusters to GT patterns; macro-average over pairs."""
    gt_keys = [k for k, v in gt.items() if v]
    cl_keys = [k for k, v in clusters.items() if v]
    if not gt_keys or not cl_keys:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_predicted": len(cl_keys), "n_ground_truth": len(gt_keys),
                "n_matched": 0}

    f1m = np.zeros((len(gt_keys), len(cl_keys)), dtype=float)
    pm = np.zeros_like(f1m)
    rm = np.zeros_like(f1m)
    for i, gk in enumerate(gt_keys):
        g = gt[gk]
        for j, ck in enumerate(cl_keys):
            c = clusters[ck]
            inter = len(c & g)
            if not inter:
                continue
            p = inter / len(c)
            r = inter / len(g)
            pm[i, j], rm[i, j] = p, r
            f1m[i, j] = 2 * p * r / (p + r)

    rows, cols = linear_sum_assignment(1.0 - f1m)
    ps, rs, fs = [], [], []
    for i, j in zip(rows, cols):
        ps.append(pm[i, j])
        rs.append(rm[i, j])
        fs.append(f1m[i, j])
    return {
        "precision": float(np.mean(ps)) if ps else 0.0,
        "recall": float(np.mean(rs)) if rs else 0.0,
        "f1": float(np.mean(fs)) if fs else 0.0,
        "n_predicted": len(cl_keys),
        "n_ground_truth": len(gt_keys),
        "n_matched": int(sum(1 for f in fs if f > 0)),
    }


def evaluate_ontology(emb: np.ndarray, entities: list[str],
                      gt: dict[str, set[str]], clusterer: str) -> dict:
    """Score one ontology under both selection regimes.

    Returns a flat dict with oracle_* and unsup_* metrics plus the chosen configs.
    """
    cands = list(CLUSTERERS[clusterer](emb, entities))
    if not cands:
        zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_predicted": 0, "n_ground_truth": len(gt), "n_matched": 0}
        return {"oracle": dict(zero, config=""), "unsup": dict(zero, config="")}

    scored = [(name, match_and_score(cl, gt), sil) for name, cl, sil in cands]

    # ORACLE: pick the config with the best F1 -- uses the ground truth.
    o_name, o_score, _ = max(scored, key=lambda t: t[1]["f1"])
    # UNSUPERVISED: pick by silhouette only -- never touches the ground truth.
    u_name, u_score, _ = max(scored, key=lambda t: t[2])

    return {"oracle": dict(o_score, config=o_name),
            "unsup": dict(u_score, config=u_name),
            "n_configs": len(scored)}
