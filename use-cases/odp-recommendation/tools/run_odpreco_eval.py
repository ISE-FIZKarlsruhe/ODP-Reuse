#!/usr/bin/env python3
"""
run_odpreco_eval.py
===================

Integration + evaluation harness that runs the Java **ODPReco** ontology-design-
pattern recommender on the PatternMiner datasets (noOBO / github_dataset) and
scores its recommendations against the per-ontology `patterns/` ground truth,
using the SAME Hungarian-matching + entity-set P/R/F1 machinery as every other
baseline in this repo (so the numbers are apples-to-apples with the
`backend/.../alternatives/altNN_*` family and the RQ stages).

Why a harness at all?
---------------------
ODPReco is a research prototype with hostile I/O:
  * It trains 8 Doc2Vec models FROM SCRATCH on every single invocation (no
    persisted model on disk) and hardcodes every path to
    `System.getProperty("user.dir") + "/resources/..."`, OVERWRITING ~50 fixed
    filenames in place. Two runs in the same cwd corrupt each other.
  * It recommends ONLY from a FIXED 73-ODP catalog (resources/odp_list +
    resources/home_files), whereas the datasets' ground truth is an arbitrary
    set of reused patterns. See pattern_catalog_map.json and the README for the
    resulting recall ceiling (~49% github / ~33% noOBO, instance-weighted).

So for each ontology we give ODPReco its OWN private copy of `resources/`
(a fresh working directory), run it, parse `resources/final_reco`, then map the
recommended catalog ODPs into the same IRI identity space as the ground truth
and evaluate.

Two modes (``--mode``):

  A. ``blackbox``  (default) -- run the real ODPReco jar with its fixed catalog.
     Recommended catalog ODPs are translated to GT-pattern stems via
     pattern_catalog_map.json; the predicted cluster for a matched ODP is the
     GT IRI set of the stem(s) that map to it. Fast, but capped by catalog
     coverage (documented ceiling).

  B. ``custom``    -- rebuild ODPReco's ODP-SIDE catalog resource files from the
     dataset's OWN pool of pattern OWLs so recommendations and GT live in one
     namespace. Full rebuild is large (8 per-dimension line files + graph +
     lists + home_files + every hardcoded '73'); see README "Mode B". If the
     custom catalog has not been built, this mode either aborts with
     instructions or (with --allow-stub) falls back to a clearly-labelled,
     jar-free offline retrieval so the harness still emits the comparable
     output triple. THIS STUB IS NOT ODPReco's Doc2Vec; it is a placeholder.

Metrics
-------
We DO NOT invent formulas. We import and call the repo-canonical
``rq_0_cache/shared_utils.match_and_evaluate(clusters, gt, f1_threshold=0.5)``
which returns macro/micro/instance entity P/R/F1 + total_tp/fp/fn +
n_predicted/n_ground_truth/n_matched. We add hits@k (k in {1,3,5,10}) computed
on the ranked recommendation list against the set of GT stems that are
*reachable* by the catalog (so hits@k measures retrieval quality, not the
catalog-coverage ceiling). Aggregation uses
``aggregate_dual_metrics`` from the same module.

Output (mirrors the alternatives' schema)
------------------------------------------
``<out>/results_<dataset>/``:
  * result.json            -- {clusters, evaluation, summary}
  * detailed_evaluation.csv-- per-ontology table (alt24 header)
  * summary_table.csv      -- one 'mean_*' row (consumed by aggregate_per_rq.py)
  * summary_table.tex      -- the standard 'Mean' LaTeX row
  * skipped.csv            -- ontologies that failed to load/run, with reason

Dependency-light: stdlib + numpy + scipy (already used by the repo's
shared_utils). rdflib only if you ask it to parse GT directly (the repo's
load_ground_truth uses it). Java 8 + Maven are required only for mode A / a real
mode B run.

Author: integration harness for the PatternMiner ODPReco use-case.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Repo wiring: import the CANONICAL metric helpers so results are byte-comparable
# --------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_PATTERNMINER_ROOT = _THIS.parent.parent           # .../PatternMiner
_RQ0 = _PATTERNMINER_ROOT / "rq_0_cache"
if str(_RQ0) not in sys.path:
    sys.path.insert(0, str(_RQ0))

try:
    from shared_utils import (                     # type: ignore
        match_and_evaluate,
        aggregate_dual_metrics,
        load_ground_truth,
        discover_ontologies,
    )
except Exception as exc:  # pragma: no cover
    print(f"FATAL: could not import rq_0_cache/shared_utils.py: {exc}\n"
          f"Expected at: {_RQ0}", file=sys.stderr)
    raise

log = logging.getLogger("odpreco_eval")

# --------------------------------------------------------------------------
# Default locations (all overridable via CLI)
# --------------------------------------------------------------------------
DEFAULT_ODPRECO_DIR = _PATTERNMINER_ROOT / "ODPReco"
DEFAULT_DATASETS = {
    "noOBO": _PATTERNMINER_ROOT / "noOBO",
    "github": _PATTERNMINER_ROOT / "github_dataset",
}
DEFAULT_MAP = _THIS.parent / "pattern_catalog_map.json"
ODPRECO_JAR_REL = Path("target") / "odpreco.jar"
ODPRECO_MAINCLASS = "com.recommender.StartODPReco"

_ONTO_EXTS = (".owl", ".ttl", ".rdf", ".xml", ".nt")


# ==========================================================================
# Build the jar if missing
# ==========================================================================

def ensure_jar(odpreco_dir: Path, force_build: bool = False,
               skip_build: bool = False) -> Path | None:
    """Return path to odpreco.jar, building with `mvn package` if needed.

    Returns None if no jar and build not possible -- caller decides whether
    that is fatal (mode A) or acceptable (mode B stub).
    """
    jar = odpreco_dir / ODPRECO_JAR_REL
    if jar.is_file() and not force_build:
        log.info("Using existing jar: %s", jar)
        return jar
    if skip_build:
        log.warning("Jar missing and --skip-build set: %s", jar)
        return jar if jar.is_file() else None

    mvn = shutil.which("mvn") or shutil.which("mvn.cmd")
    if mvn is None:
        log.error("Maven (mvn) not found on PATH; cannot build ODPReco. "
                  "Install Maven + JDK 8, or pre-build target/odpreco.jar.")
        return jar if jar.is_file() else None

    log.info("Building ODPReco jar with `mvn -q package` in %s ...", odpreco_dir)
    try:
        proc = subprocess.run(
            [mvn, "-q", "-DskipTests", "package"],
            cwd=str(odpreco_dir), capture_output=True, text=True, timeout=3600)
    except Exception as exc:
        log.error("mvn package failed to start: %s", exc)
        return jar if jar.is_file() else None
    if proc.returncode != 0:
        log.error("mvn package returned %d.\nSTDOUT tail:\n%s\nSTDERR tail:\n%s",
                  proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
        return jar if jar.is_file() else None
    if jar.is_file():
        log.info("Built jar: %s", jar)
        return jar
    log.error("mvn succeeded but %s not found.", jar)
    return None


# ==========================================================================
# Ground truth + entity universe for one ontology
# ==========================================================================

def load_gt_for_onto(onto_dir: Path) -> dict[str, set[str]]:
    """{pattern_stem: set(IRIs)} via the repo-canonical loader.

    This is byte-identical to what build_cache.py persists into meta.json and
    what every other baseline matches against. Keys on filename stem; values are
    the URIRef subjects+objects of each pattern file (owl:Ontology discarded).
    """
    return load_ground_truth(onto_dir)


def stem_key(name: str) -> str:
    """Normalise a pattern stem / filename to the map key convention."""
    s = Path(name).stem if ("/" in name or "\\" in name or "." in name) else name
    return s.strip().lower()


# ==========================================================================
# Catalog map (mode A)
# ==========================================================================

def load_catalog_map(map_path: Path) -> tuple[dict[str, str], set[str]]:
    """Return (stem_to_odpname, set_of_reachable_stems).

    Reads pattern_catalog_map.json. The 'mapped' block gives stem->ODP-name;
    null / 'unmapped' entries are NOT reachable. Underscore-prefixed keys are
    comments and are ignored.
    """
    data = json.loads(map_path.read_text(encoding="utf-8"))
    stem_to_odp: dict[str, str] = {}
    for block_name in ("mapped", "unmapped"):
        block = data.get(block_name, {})
        for k, v in block.items():
            if k.startswith("_"):
                continue
            if v is None:
                continue
            stem_to_odp[stem_key(k)] = str(v)
    reachable = set(stem_to_odp.keys())
    return stem_to_odp, reachable


def odp_name_norm(name: str) -> str:
    """Normalise an ODP display name for matching ('AgentRole ODP' -> 'agentrole')."""
    n = name.lower()
    n = re.sub(r"\bodp\b", " ", n)
    n = re.sub(r"[^a-z0-9]+", "", n)
    return n


# ==========================================================================
# Run the ODPReco jar for ONE ontology in an isolated resources copy
# ==========================================================================

def run_odpreco_once(jar: Path, odpreco_dir: Path, onto_owl: Path,
                     java_bin: str, timeout_s: int,
                     keep_workdir: bool = False) -> tuple[list[tuple[float, str, str]], str | None]:
    """Run ODPReco on a single ontology in a private working directory.

    ODPReco hardcodes user.dir/resources/*, so we:
      1. mkdtemp a private workdir,
      2. copy ODPReco/resources/ into it (the full catalog must be present),
      3. write empty -ontdes / -ontcq files (both flags are de-facto required;
         empty files make the tool skip description/CQ Doc2Vec -- see README),
      4. run `java -jar <jar> -ontowl <owl> -ontdes <empty> -ontcq <empty>`
         with cwd = workdir,
      5. parse <workdir>/resources/final_reco.

    Returns (recs, error). recs = [(score, odp_name, iri), ...] in file order
    (final_reco is already reverse-sorted by score). error is None on success.
    """
    src_resources = odpreco_dir / "resources"
    if not src_resources.is_dir():
        return [], f"ODPReco resources/ not found at {src_resources}"

    workdir = Path(tempfile.mkdtemp(prefix="odpreco_"))
    try:
        dst_resources = workdir / "resources"
        # copytree is the safest isolation: each ontology gets a pristine catalog.
        shutil.copytree(src_resources, dst_resources)
        # Clean any leftover lucene index lock from the template (defensive).
        idx = dst_resources / "home_index"
        if idx.is_dir():
            for f in idx.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass

        empty_des = workdir / "empty_description.txt"
        empty_cq = workdir / "empty_cq.txt"
        empty_des.write_text("\n", encoding="utf-8")
        empty_cq.write_text("\n", encoding="utf-8")

        cmd = [java_bin, "-jar", str(jar),
               "-ontowl", str(onto_owl),
               "-ontdes", str(empty_des),
               "-ontcq", str(empty_cq)]
        try:
            proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True,
                                  text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return [], f"timeout after {timeout_s}s"
        except Exception as exc:
            return [], f"java launch failed: {exc}"

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-600:]
            return [], f"java exit {proc.returncode}: {tail.strip()}"

        recs = parse_final_reco(dst_resources)
        return recs, None
    finally:
        if keep_workdir:
            log.info("kept workdir: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


_RECO_LINE = re.compile(
    r"""^\s*
        (?P<score>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)?   # optional leading score
        \s*
        (?P<name>.+?)\s+ODP                              # '<Name> ODP'
        \s*\(?<?                                          # opening of IRI part
        (?P<iri>https?://[^>\s)]+)?                       # optional IRI
    """,
    re.VERBOSE,
)


def parse_final_reco(resources: Path) -> list[tuple[float, str, str]]:
    """Parse recommendations, preferring resources/final_reco (has score+name+IRI).

    Falls back to resources/recommender, then resources/final_sorted_reco.
    Returns [(score, odp_name, iri)] in file order (already score-desc).
    """
    for fname in ("final_reco", "recommender", "final_sorted_reco"):
        f = resources / fname
        if not f.is_file():
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        out: list[tuple[float, str, str]] = []
        for ln in lines:
            if not ln.strip():
                continue
            m = _RECO_LINE.match(ln)
            if not m:
                continue
            score = float(m.group("score")) if m.group("score") else 0.0
            name = (m.group("name") or "").strip()
            iri = (m.group("iri") or "").strip()
            if name:
                out.append((score, name, iri))
        if out:
            return out
    return []


# ==========================================================================
# Mode A: turn recommendations into predicted clusters in GT's IRI space
# ==========================================================================

def predict_clusters_blackbox(recs: list[tuple[float, str, str]],
                              gt: dict[str, set[str]],
                              stem_to_odp: dict[str, str]
                              ) -> tuple[dict[int, set[str]], list[str]]:
    """Map recommended catalog ODPs -> predicted clusters in the GT IRI space.

    A recommended ODP (by display name) is matched to every GT pattern STEM that
    maps to that ODP (via pattern_catalog_map.json). The predicted cluster for
    that recommendation is the union of the GT IRI sets of those stems -- i.e.
    'ODPReco says this ontology reuses ODP X; the entities of X (as they appear
    in THIS ontology's GT) are the prediction'. This keeps recommendations and
    GT in one identity space so the Hungarian matcher can intersect them.

    Returns (clusters {cid:set(iri)}, ranked_pred_stems) where ranked_pred_stems
    is the list of GT stems predicted, in recommendation rank order (for hits@k).
    """
    # invert stem_to_odp: normalised ODP name -> [stems...]
    odp_to_stems: dict[str, list[str]] = {}
    for stem, odp in stem_to_odp.items():
        odp_to_stems.setdefault(odp_name_norm(odp), []).append(stem)

    clusters: dict[int, set[str]] = {}
    ranked_pred_stems: list[str] = []
    seen_stems: set[str] = set()
    cid = 0
    for _score, name, _iri in recs:
        key = odp_name_norm(name)
        stems = odp_to_stems.get(key, [])
        for stem in stems:
            if stem in gt and stem not in seen_stems:
                clusters[cid] = set(gt[stem])
                cid += 1
                seen_stems.add(stem)
                ranked_pred_stems.append(stem)
    return clusters, ranked_pred_stems


# ==========================================================================
# Mode B: custom catalog
# ==========================================================================

CUSTOM_CATALOG_MARKER = "CUSTOM_CATALOG_README.txt"


def predict_clusters_custom_stub(gt: dict[str, set[str]],
                                 pattern_pool: dict[str, set[str]],
                                 top_k: int, sim_threshold: float
                                 ) -> tuple[dict[int, set[str]], list[str]]:
    """JAR-FREE offline placeholder for mode B (clearly NOT ODPReco's Doc2Vec).

    Given a dataset-wide pool {pattern_stem: set(IRIs)} (built from ALL pattern
    OWLs in the dataset), and this ontology's GT entity universe, retrieve the
    top-k pool patterns by Jaccard overlap of their entity-IRI bag with this
    ontology's entity universe, and emit each retrieved pattern's GT IRI set as a
    predicted cluster. This mimics 'catalog retrieval over a custom catalog' so
    the harness produces a comparable triple, but it is a STUB: a faithful mode B
    must regenerate ODPReco's resource files and run the real jar (see README).
    """
    universe: set[str] = set()
    for s in gt.values():
        universe |= s
    if not universe or not pattern_pool:
        return {}, []

    scored: list[tuple[float, str]] = []
    for stem, bag in pattern_pool.items():
        if not bag:
            continue
        inter = len(universe & bag)
        if inter == 0:
            continue
        union = len(universe | bag)
        jac = inter / union if union else 0.0
        if jac >= sim_threshold:
            scored.append((jac, stem))
    scored.sort(reverse=True)
    scored = scored[:top_k]

    clusters: dict[int, set[str]] = {}
    ranked: list[str] = []
    for cid, (_j, stem) in enumerate(scored):
        # Predicted cluster = the pool pattern's entities intersected with this
        # ontology (so it can plausibly hit this ontology's GT set).
        pred = pattern_pool[stem] & universe
        if not pred:
            pred = set(pattern_pool[stem])
        clusters[cid] = pred
        ranked.append(stem)
    return clusters, ranked


def build_pattern_pool(ontos: list[tuple[Path, str]],
                       limit: int = 0) -> dict[str, set[str]]:
    """Aggregate ALL pattern files across the dataset into {stem: set(IRIs)}.

    Used only by the mode-B stub. Reuses load_ground_truth per ontology and
    unions IRIs per stem (the same stem recurs across many ontologies).
    """
    pool: dict[str, set[str]] = {}
    n = 0
    for owl, name in ontos:
        gt = load_ground_truth(owl.parent)
        for stem, iris in gt.items():
            pool.setdefault(stem.lower(), set()).update(iris)
        n += 1
        if limit and n >= limit:
            break
    return pool


# ==========================================================================
# hits@k (ranked retrieval quality vs reachable GT stems)
# ==========================================================================

def hits_at_k(ranked_pred_stems: list[str], gt_stems: set[str],
              ks=(1, 3, 5, 10)) -> dict[str, float]:
    """hits@k = 1 if any of the top-k ranked predicted stems is a true GT stem.

    Computed only over GT stems that exist for this ontology. This measures the
    ranking quality of the recommender among reachable patterns, separate from
    the catalog-coverage ceiling baked into precision/recall.
    """
    out: dict[str, float] = {}
    truth = {s.lower() for s in gt_stems}
    pred = [s.lower() for s in ranked_pred_stems]
    for k in ks:
        topk = set(pred[:k])
        out[f"hits@{k}"] = 1.0 if (topk & truth) else 0.0
    return out


# ==========================================================================
# Per-ontology evaluation row
# ==========================================================================

def evaluate_ontology(onto_name: str, mode: str,
                      clusters: dict[int, set[str]],
                      gt: dict[str, set[str]],
                      ranked_pred_stems: list[str],
                      f1_threshold: float) -> dict:
    """Call the canonical matcher and attach hits@k + bookkeeping."""
    base = match_and_evaluate(clusters, gt, f1_threshold=f1_threshold)
    # alt24-compatible aliases
    base["instance_tp"] = base.get("n_matched", 0)
    base["instance_fp"] = max(base.get("n_predicted", 0) - base.get("n_matched", 0), 0)
    base["instance_fn"] = max(base.get("n_ground_truth", 0) - base.get("n_matched", 0), 0)
    base["entity_precision"] = base.get("macro_entity_precision", 0.0)
    base["entity_recall"] = base.get("macro_entity_recall", 0.0)
    base["entity_f1"] = base.get("macro_entity_f1", 0.0)
    base.update(hits_at_k(ranked_pred_stems, set(gt.keys())))
    base["ontology"] = onto_name
    base["mode"] = mode
    return base


# ==========================================================================
# Output writers (mirror the alternatives' triple)
# ==========================================================================

DETAILED_COLS = [
    "ontology", "mode",
    "instance_precision", "instance_recall", "instance_f1",
    "instance_tp", "instance_fp", "instance_fn",
    "micro_entity_precision", "micro_entity_recall", "micro_entity_f1",
    "macro_entity_precision", "macro_entity_recall", "macro_entity_f1",
    "total_tp", "total_fp", "total_fn",
    "n_predicted", "n_ground_truth", "n_matched",
    "hits@1", "hits@3", "hits@5", "hits@10",
]


def write_outputs(out_dir: Path, dataset: str, mode: str,
                  evals: list[dict], clusters_dump: list[dict],
                  skipped: list[tuple[str, str]], run_params: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"odpreco_{mode}"
    agg = aggregate_dual_metrics(tag, evals)

    # hits@k means (not in aggregate_dual_metrics)
    import numpy as np
    for k in (1, 3, 5, 10):
        vals = [float(e.get(f"hits@{k}", 0) or 0) for e in evals]
        agg[f"mean_hits@{k}"] = round(float(np.mean(vals)), 4) if vals else 0.0

    summary = {
        "dataset": dataset,
        "n_ontologies": len(evals),
        "n_clusters": sum(int(e.get("n_predicted", 0)) for e in evals),
        "mean_instance_precision": agg.get("mean_instance_precision", 0.0),
        "mean_instance_recall": agg.get("mean_instance_recall", 0.0),
        "mean_instance_f1": agg.get("mean_instance_f1", 0.0),
        "mean_micro_entity_precision": agg.get("mean_micro_entity_precision", 0.0),
        "mean_micro_entity_recall": agg.get("mean_micro_entity_recall", 0.0),
        "mean_micro_entity_f1": agg.get("mean_micro_entity_f1", 0.0),
        "mean_macro_entity_precision": agg.get("mean_macro_entity_precision", 0.0),
        "mean_macro_entity_recall": agg.get("mean_macro_entity_recall", 0.0),
        "mean_macro_entity_f1": agg.get("mean_macro_entity_f1", 0.0),
        # alt-style aliases (entity == macro)
        "mean_entity_precision": agg.get("mean_macro_entity_precision", 0.0),
        "mean_entity_recall": agg.get("mean_macro_entity_recall", 0.0),
        "mean_entity_f1": agg.get("mean_macro_entity_f1", 0.0),
        "mean_hits@1": agg.get("mean_hits@1", 0.0),
        "mean_hits@3": agg.get("mean_hits@3", 0.0),
        "mean_hits@5": agg.get("mean_hits@5", 0.0),
        "mean_hits@10": agg.get("mean_hits@10", 0.0),
        "alternative": tag,
        "method": "ODPReco (Doc2Vec + Lucene ODP recommender)",
        "approach": "ODPReco",
        "approach_detail": f"mode={mode}",
        "n_skipped": len(skipped),
    }
    summary.update(run_params)

    # result.json -------------------------------------------------------
    result = {
        "clusters": clusters_dump,
        "evaluation": evals,
        "summary": summary,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    # detailed_evaluation.csv ------------------------------------------
    with open(out_dir / "detailed_evaluation.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DETAILED_COLS, extrasaction="ignore")
        w.writeheader()
        for e in evals:
            w.writerow({k: e.get(k, "") for k in DETAILED_COLS})

    # summary_table.csv (the canonical mean_* row consumed by aggregate_per_rq)
    with open(out_dir / "summary_table.csv", "w", newline="",
              encoding="utf-8") as f:
        row = dict(agg)
        row["tag"] = tag
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        w.writeheader()
        w.writerow(row)

    # summary_table.tex ------------------------------------------------
    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "0.0000"
    tex = (
        "\\begin{table}[ht]\n\\centering\n\\small\n"
        f"\\caption{{Evaluation results for {tag}}}\n"
        f"\\label{{tab:{tag}}}\n"
        "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
        "Metric & \\multicolumn{3}{c}{Instance-Level} & "
        "\\multicolumn{3}{c}{Micro Entity} & \\multicolumn{3}{c}{Macro Entity} \\\\\n"
        " & P & R & F1 & P & R & F1 & P & R & F1 \\\\\n\\midrule\n"
        "Mean & "
        f"{fmt(agg.get('mean_instance_precision'))} & {fmt(agg.get('mean_instance_recall'))} & {fmt(agg.get('mean_instance_f1'))} & "
        f"{fmt(agg.get('mean_micro_entity_precision'))} & {fmt(agg.get('mean_micro_entity_recall'))} & {fmt(agg.get('mean_micro_entity_f1'))} & "
        f"{fmt(agg.get('mean_macro_entity_precision'))} & {fmt(agg.get('mean_macro_entity_recall'))} & {fmt(agg.get('mean_macro_entity_f1'))} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out_dir / "summary_table.tex").write_text(tex, encoding="utf-8")

    # skipped.csv ------------------------------------------------------
    with open(out_dir / "skipped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ontology", "reason"])
        for name, reason in skipped:
            w.writerow([name, reason])

    log.info("wrote outputs to %s", out_dir)


# ==========================================================================
# Main driver
# ==========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run + evaluate ODPReco on a PatternMiner dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DEFAULT_DATASETS), default="noOBO",
                    help="which built-in dataset to run")
    ap.add_argument("--ontologies-dir", default=None,
                    help="override dataset root (a dir of <onto>/ folders)")
    ap.add_argument("--odpreco-dir", default=str(DEFAULT_ODPRECO_DIR),
                    help="path to the ODPReco Java project (holds resources/, pom.xml)")
    ap.add_argument("--out", default=str(_THIS.parent / "results"),
                    help="output root; results land in <out>/results_<dataset>/")
    ap.add_argument("--mode", choices=("blackbox", "custom"), default="blackbox",
                    help="A: blackbox (real jar + fixed catalog); B: custom catalog")
    ap.add_argument("--map", default=str(DEFAULT_MAP),
                    help="pattern->catalog map JSON (mode A)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N ontologies (0 = all)")
    ap.add_argument("--f1-threshold", type=float, default=0.5,
                    help="instance-match F1 cutoff (repo default 0.5)")
    ap.add_argument("--java", default="java", help="java 8 binary")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-ontology ODPReco timeout (seconds)")
    ap.add_argument("--force-build", action="store_true",
                    help="rebuild the jar even if present")
    ap.add_argument("--skip-build", action="store_true",
                    help="never invoke mvn; require a prebuilt jar")
    ap.add_argument("--keep-workdir", action="store_true",
                    help="do not delete per-ontology temp resources (debug)")
    ap.add_argument("--allow-stub", action="store_true",
                    help="mode B only: fall back to the jar-free retrieval STUB "
                         "if the custom catalog has not been built")
    ap.add_argument("--custom-catalog-built", action="store_true",
                    help="mode B only: assert that ODPReco/resources has already "
                         "been regenerated for this dataset, so run the real jar")
    ap.add_argument("--stub-top-k", type=int, default=5,
                    help="mode B stub: number of retrieved patterns per ontology")
    ap.add_argument("--stub-sim-threshold", type=float, default=0.0,
                    help="mode B stub: min Jaccard to retrieve a pattern")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")

    ds_root = Path(args.ontologies_dir) if args.ontologies_dir \
        else DEFAULT_DATASETS[args.dataset]
    odpreco_dir = Path(args.odpreco_dir)
    out_dir = Path(args.out) / f"results_{args.dataset}"

    if not ds_root.is_dir():
        log.error("dataset dir not found: %s", ds_root)
        return 2

    ontos = discover_ontologies(ds_root)
    if args.limit:
        ontos = ontos[:args.limit]
    log.info("dataset=%s  ontologies=%d  mode=%s  out=%s",
             args.dataset, len(ontos), args.mode, out_dir)

    run_params: dict = {
        "f1_threshold": args.f1_threshold,
        "ontologies_dir": str(ds_root),
        "odpreco_dir": str(odpreco_dir),
        "limit": args.limit,
    }

    evals: list[dict] = []
    clusters_dump: list[dict] = []
    skipped: list[tuple[str, str]] = []
    t0 = time.time()

    # --------------------- MODE A: blackbox (real jar) ---------------------
    if args.mode == "blackbox":
        jar = ensure_jar(odpreco_dir, force_build=args.force_build,
                         skip_build=args.skip_build)
        if jar is None or not jar.is_file():
            log.error("No ODPReco jar available; cannot run mode A. "
                      "Build it (mvn package in %s) or use --mode custom --allow-stub.",
                      odpreco_dir)
            return 3
        stem_to_odp, reachable = load_catalog_map(Path(args.map))
        run_params["catalog_map"] = str(args.map)
        run_params["n_reachable_stems"] = len(reachable)

        for owl, name in ontos:
            gt = load_gt_for_onto(owl.parent)
            if not gt:
                skipped.append((name, "no ground-truth patterns/"))
                continue
            recs, err = run_odpreco_once(
                jar, odpreco_dir, owl, args.java, args.timeout,
                keep_workdir=args.keep_workdir)
            if err is not None:
                skipped.append((name, f"odpreco run failed: {err}"))
                log.warning("[skip] %s: %s", name, err)
                continue
            clusters, ranked = predict_clusters_blackbox(recs, gt, stem_to_odp)
            ev = evaluate_ontology(name, "blackbox", clusters, gt, ranked,
                                   args.f1_threshold)
            ev["n_recommended"] = len(recs)
            evals.append(ev)
            for cid, ents in clusters.items():
                clusters_dump.append({"cluster_id": cid, "ontology": name,
                                      "entities": sorted(ents)})
            log.info("[ok] %s  recs=%d  pred=%d  gt=%d  macroF1=%.4f",
                     name, len(recs), ev["n_predicted"], ev["n_ground_truth"],
                     ev.get("macro_entity_f1", 0.0))

    # --------------------- MODE B: custom catalog -------------------------
    else:
        marker = odpreco_dir / "resources" / CUSTOM_CATALOG_MARKER
        if args.custom_catalog_built or marker.is_file():
            # Real run against a regenerated catalog: identical to mode A flow
            # but WITHOUT the catalog map -- recommendations are already in the
            # dataset's namespace, so the recommended IRI tails ARE GT stems.
            jar = ensure_jar(odpreco_dir, force_build=args.force_build,
                             skip_build=args.skip_build)
            if jar is None or not jar.is_file():
                log.error("custom catalog present but no jar; build it first.")
                return 3
            run_params["custom_catalog"] = "regenerated"
            for owl, name in ontos:
                gt = load_gt_for_onto(owl.parent)
                if not gt:
                    skipped.append((name, "no ground-truth patterns/"))
                    continue
                recs, err = run_odpreco_once(
                    jar, odpreco_dir, owl, args.java, args.timeout,
                    keep_workdir=args.keep_workdir)
                if err is not None:
                    skipped.append((name, f"odpreco run failed: {err}"))
                    continue
                # In a regenerated catalog each recommended ODP's name/IRI tail
                # equals a GT stem; map recommended names directly to GT stems.
                clusters, ranked = predict_clusters_blackbox(
                    recs, gt, identity_from_recs(recs, gt))
                ev = evaluate_ontology(name, "custom", clusters, gt, ranked,
                                       args.f1_threshold)
                evals.append(ev)
                for cid, ents in clusters.items():
                    clusters_dump.append({"cluster_id": cid, "ontology": name,
                                          "entities": sorted(ents)})
        elif args.allow_stub:
            log.warning("MODE B STUB: custom ODPReco catalog NOT built; using a "
                        "jar-free Jaccard retrieval placeholder. Numbers are NOT "
                        "ODPReco's Doc2Vec output. See README 'Mode B'.")
            run_params["custom_catalog"] = "STUB(jaccard-retrieval, not ODPReco)"
            pool = build_pattern_pool(ontos)
            run_params["pool_size"] = len(pool)
            for owl, name in ontos:
                gt = load_gt_for_onto(owl.parent)
                if not gt:
                    skipped.append((name, "no ground-truth patterns/"))
                    continue
                # exclude this ontology's own stems? No -- the pool is the
                # catalog; a real catalog contains the pattern definitions.
                clusters, ranked = predict_clusters_custom_stub(
                    gt, pool, args.stub_top_k, args.stub_sim_threshold)
                ev = evaluate_ontology(name, "custom-stub", clusters, gt, ranked,
                                       args.f1_threshold)
                evals.append(ev)
                for cid, ents in clusters.items():
                    clusters_dump.append({"cluster_id": cid, "ontology": name,
                                          "entities": sorted(ents)})
        else:
            log.error(
                "Mode B (custom catalog) requested but the ODPReco catalog has "
                "NOT been regenerated for this dataset.\n"
                "  * To run the REAL custom-catalog pipeline, first regenerate "
                "the ODP-side resource files (see README 'Mode B: how to rebuild "
                "the catalog'), drop a marker file at\n      %s\n"
                "    or pass --custom-catalog-built, then re-run.\n"
                "  * For a quick, clearly-labelled placeholder (NOT ODPReco), "
                "re-run with --allow-stub.",
                marker)
            return 4

    elapsed = time.time() - t0
    run_params["elapsed_seconds"] = round(elapsed, 1)
    write_outputs(out_dir, args.dataset, args.mode, evals, clusters_dump,
                  skipped, run_params)

    log.info("DONE  evaluated=%d  skipped=%d  elapsed=%.1fs",
             len(evals), len(skipped), elapsed)
    if evals:
        import numpy as np
        mf = float(np.mean([e.get("macro_entity_f1", 0.0) for e in evals]))
        log.info("HEADLINE mean_macro_entity_f1 = %.4f", mf)
    return 0


def identity_from_recs(recs, gt):
    """Build a stem->stem map for the custom (regenerated) catalog case.

    A recommended ODP's name (normalised) is matched to a GT stem when they are
    equal after normalisation; the predicted cluster is that GT pattern's set.
    """
    norm_gt = {odp_name_norm(s): s for s in gt}
    mapping: dict[str, str] = {}
    for _score, name, _iri in recs:
        k = odp_name_norm(name)
        if k in norm_gt:
            mapping[norm_gt[k].lower()] = norm_gt[k]
    return mapping


if __name__ == "__main__":
    raise SystemExit(main())
