"""
recobench.stats  --  Phase 0: trustworthy evaluation.

The main results were computed on a single 70/30 split and the LLM rows on only
8 ontologies, so raw point estimates cannot be told apart from noise. This module
adds:

  * bootstrap_ci    -- a percentile bootstrap confidence interval for the mean of
                       any per-ontology metric (resamples ontologies with
                       replacement).
  * paired_delta    -- for two recommenders scored on the SAME ontologies, the
                       mean per-ontology difference of a metric, its bootstrap CI,
                       a two-sided bootstrap p-value, and an exact sign-test p.

Everything operates on the per-ontology dictionaries that `evaluate_*` already
returns / that are stored in `results_reco/<dataset>/<backend>.json`. No new runs
are needed to obtain CIs for results already on disk.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _values(per: list[dict], metric: str) -> np.ndarray:
    return np.array([float(r.get(metric, 0.0) or 0.0) for r in per], dtype=float)


def bootstrap_ci(per: list[dict], metric: str, n_boot: int = 5000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean of `metric` over ontologies."""
    x = _values(per, metric)
    n = len(x)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(x.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": n}


def paired_delta(per_a: list[dict], per_b: list[dict], metric: str,
                 key: str = "ontology", n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired comparison of A vs B on the ontologies both were scored on.

    Returns mean(A-B), a bootstrap CI for that mean, a two-sided bootstrap
    p-value (share of resamples whose mean delta has the opposite sign, doubled),
    and an exact two-sided sign-test p-value over the non-zero pairs.
    """
    a = {r[key]: float(r.get(metric, 0.0) or 0.0) for r in per_a if key in r}
    b = {r[key]: float(r.get(metric, 0.0) or 0.0) for r in per_b if key in r}
    common = sorted(set(a) & set(b))
    d = np.array([a[o] - b[o] for o in common], dtype=float)
    n = len(d)
    if n == 0:
        return {"metric": metric, "n": 0, "mean_delta": 0.0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    md = float(d.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled mean lands on the far side of 0
    if md >= 0:
        p_boot = 2.0 * float((boots <= 0).mean())
    else:
        p_boot = 2.0 * float((boots >= 0).mean())
    p_boot = min(1.0, p_boot)
    # exact sign test over non-zero pairs
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    m = pos + neg
    if m == 0:
        p_sign = 1.0
    else:
        k = min(pos, neg)
        # two-sided: 2 * P(X <= k) under Binomial(m, 0.5)
        cum = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
        p_sign = min(1.0, 2.0 * cum)
    return {"metric": metric, "n": n,
            "mean_delta": round(md, 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "p_bootstrap": round(p_boot, 4),
            "p_sign_test": round(p_sign, 4),
            "wins": pos, "losses": neg, "ties": n - pos - neg}


def load_per(results_dir: Path, backend_file: str) -> list[dict]:
    """Load the per_ontology list from a stored <backend>.json."""
    d = json.loads((results_dir / backend_file).read_text(encoding="utf-8"))
    return d.get("per_ontology", d.get("evaluation", []))


def ci_table(results_dir: Path, backends: list[str], metric: str = "map",
             **kw) -> list[dict]:
    """Bootstrap CI of `metric` for each backend json in `backends`."""
    rows = []
    for bf in backends:
        p = results_dir / bf
        if not p.is_file():
            continue
        per = load_per(results_dir, bf)
        ci = bootstrap_ci(per, metric, **kw)
        ci["backend"] = bf[:-5]
        rows.append(ci)
    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows
