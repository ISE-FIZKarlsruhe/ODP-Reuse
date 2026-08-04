"""
recobench.run
=============

Driver for the held-out ODP-recommender benchmark.

It runs a roster of recommender backends on one dataset under a chosen protocol
(leave-one-out for small noOBO; train/test split otherwise) and writes:

  results_reco/<dataset>/<backend>.json    per-backend summary + per-ontology rows
  results_reco/<dataset>/comparison.csv    master table (sorted by MAP)
  results_reco/<dataset>/REPORT.md         human-readable ablation, Thread 1 + 2

Thread 1 ("does using the datasets help?") falls straight out of the table:
random (uses_dataset=False) vs popularity / CF / embedding / Doc2Vec / LLM
(uses_dataset=True), plus the SAME embedding model run over ODPReco's FIXED
73-ODP catalog vs the dataset catalog (the fixed-vs-dataset comparison).

Usage:
  python -m recobench.run --dataset noOBO --protocol loo
  python -m recobench.run --dataset noOBO --protocol split --with-llm --llm-sample 20
  python -m recobench.run --dataset github --protocol split --limit 400 --embspec st:all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .core import (
    load_approved_labels,
    DEFAULT_DATASETS, load_dataset, catalog_from, split_train_test,
    evaluate_recommender, evaluate_loo, write_result, write_comparison, Split,
)
from .baselines import (
    RandomRecommender, PopularityRecommender, OntologyKNNRecommender,
)
from .content import (
    EmbeddingItemRecommender, EmbeddingOntologyKNN, Doc2VecRecommender,
)

log = logging.getLogger("recobench.run")
_USECASE = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# ODPReco FIXED catalog (for the fixed-vs-dataset embedding comparison)
# ----------------------------------------------------------------------

def load_fixed_catalog(odpreco_dir: Path, map_path: Path
                       ) -> tuple[dict[str, str], dict[str, str]]:
    """Return (item_docs, item_to_stem) for ODPReco's fixed 73-ODP catalog.

    item id = ODP display name (e.g. 'AgentRole'); doc = its odps_description.txt
    line; stem = the GT stem it maps to via pattern_catalog_map.json (or None).
    Lets EmbeddingItemRecommender run the SAME model over the fixed catalog.
    """
    res = odpreco_dir / "resources"
    names_raw = (res / "odp_list").read_text(encoding="utf-8",
                                             errors="ignore").splitlines()
    descs = (res / "odps_description.txt").read_text(encoding="utf-8",
                                                     errors="ignore").splitlines()
    names = []
    for ln in names_raw:
        ln = ln.strip()
        if not ln:
            continue
        nm = ln.split("ODP", 1)[0].strip()
        names.append(nm or ln)
    item_docs: dict[str, str] = {}
    for i, nm in enumerate(names):
        d = descs[i] if i < len(descs) else ""
        item_docs[nm] = f"{nm}. {d}".strip()

    # map ODP display name -> GT stem via pattern_catalog_map.json (invert it)
    mp = json.loads(map_path.read_text(encoding="utf-8"))
    stem_to_odp = {k.lower(): v for k, v in mp.get("mapped", {}).items()
                   if not k.startswith("_") and v}

    def norm(n: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "", re.sub(r"\bodp\b", " ", n.lower()))

    odp_norm_to_stem: dict[str, str] = {}
    for stem, odp in stem_to_odp.items():
        odp_norm_to_stem[norm(odp)] = stem
    item_to_stem: dict[str, str] = {}
    for nm in item_docs:
        item_to_stem[nm] = odp_norm_to_stem.get(norm(nm))   # may be None
    return item_docs, item_to_stem


# ----------------------------------------------------------------------
# Ingest the Java ODPReco Mode-A result (if present) as an external row
# ----------------------------------------------------------------------

def ingest_odpreco_modeA(results_dir: Path) -> dict | None:
    """Convert run_odpreco_eval.py's result.json into a comparison row, if any."""
    f = results_dir / "result.json"
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    s = d.get("summary", {})
    return {
        "recommender": "ODPReco_jar_modeA(fixed catalog)",
        "uses_dataset": False,
        "n_ontologies": s.get("n_ontologies", 0),
        "mean_precision@1": "", "mean_precision@5": "",
        "mean_recall@5": "", "mean_recall@10": "",
        "mean_map": s.get("mean_macro_entity_f1", ""),  # not directly MAP; noted
        "mean_mrr": "", "_note": "entity-F1 metric, not ranking MAP; see README",
    }


# ----------------------------------------------------------------------
# Roster
# ----------------------------------------------------------------------

def build_roster(args, catalog):
    """List of (factory, heavy) where heavy backends may be sampled for LLM."""
    roster = [
        (lambda: RandomRecommender(), False),
        (lambda: PopularityRecommender(), False),
        (lambda: OntologyKNNRecommender(k=args.knn_k), False),
        (lambda: Doc2VecRecommender(epochs=args.doc2vec_epochs), False),
        (lambda: EmbeddingOntologyKNN(args.embspec, k=args.knn_k), False),
        (lambda: EmbeddingItemRecommender(
            args.embspec, label=f"embed_item_datasetcatalog[{args.embspec}]"), False),
    ]
    # fixed-vs-dataset catalog comparison (same embedding model)
    if args.fixed_catalog:
        try:
            fdocs, fmap = load_fixed_catalog(
                Path(args.odpreco_dir), Path(args.map))
            roster.append((lambda: EmbeddingItemRecommender(
                args.embspec, item_docs=fdocs, item_to_stem=fmap,
                label=f"embed_item_FIXEDcatalog[{args.embspec}]"), False))
        except Exception as e:
            log.warning("could not build fixed catalog: %s", e)
    return roster


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DEFAULT_DATASETS), default="noOBO")
    ap.add_argument("--no-require-overlap", action="store_true",
                    help="reproduce the older ground truth, which counted a "
                         "pattern as reused whenever its file was present in "
                         "patterns/ regardless of entity overlap")
    ap.add_argument("--approved-labels", default=None,
                    help="CSV of human-adjudicated label matches. Supplying it "
                         "selects the iri+label ground truth; omitting it keeps "
                         "the exact-IRI track. Results should be written to "
                         "separate --out directories for the two tracks.")
    ap.add_argument("--ontologies-dir", default=None)
    ap.add_argument("--protocol", choices=("loo", "split"), default="loo")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--knn-k", type=int, default=15)
    ap.add_argument("--doc2vec-epochs", type=int, default=40)
    ap.add_argument("--embspec", default="st:all-MiniLM-L6-v2",
                    help="embedding backend: st:<model> or ollama:<model>")
    ap.add_argument("--fixed-catalog", action="store_true",
                    help="also run the embedding model over ODPReco's fixed catalog")
    ap.add_argument("--odpreco-dir", default=str(_USECASE.parent / "ODPReco"))
    ap.add_argument("--map", default=str(_USECASE / "pattern_catalog_map.json"))
    ap.add_argument("--with-llm", action="store_true",
                    help="also run Qwen rerank + generative (split protocol, sampled)")
    ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-sample", type=int, default=20,
                    help="evaluate LLM backends on at most N test ontologies")
    ap.add_argument("--out", default=str(_USECASE / "results_reco"))
    ap.add_argument("--manifest", default=None,
                    help="MANIFEST.csv; restricts the population to "
                         "gate-passing ontologies of the chosen dataset. "
                         "Omit to run over the raw harvest (not what the "
                         "paper reports).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    root = Path(args.ontologies_dir) if args.ontologies_dir \
        else DEFAULT_DATASETS[args.dataset]
    onts = load_dataset(root, limit=args.limit, approved_labels=load_approved_labels(getattr(args,'approved_labels',None)),
                        require_overlap=not getattr(args,'no_require_overlap',False))

    # Restrict to the quality-gated population, if a manifest is supplied.
    # Without this the benchmark runs over the RAW harvest, which includes
    # unparseable files, byte-identical duplicates (a direct train/test leak
    # for a similarity-based recommender) and pattern copies whose token bag
    # *is* the pattern being predicted. The paper reports the gated figure, so
    # the gated population is what it must be computed on.
    mf = getattr(args, "manifest", None)
    if mf:
        import csv as _csv
        with open(mf, encoding="utf-8", newline="") as fh:
            keep = {r["ontology_id"] for r in _csv.DictReader(fh)
                    if r["dataset"] == args.dataset and not r["drop_reason"]}
        before = len(onts)
        onts = [o for o in onts if o.name in keep]
        log.info("manifest filter: %d -> %d ontologies (gate-passing only)",
                 before, len(onts))

    catalog = catalog_from(onts)
    out_dir = Path(args.out) / args.dataset
    log.info("dataset=%s ontologies=%d catalog=%d protocol=%s",
             args.dataset, len(onts), len(catalog), args.protocol)

    aggs: list[dict] = []
    roster = build_roster(args, catalog)

    for factory, _heavy in roster:
        name = factory().name
        log.info("running %s ...", name)
        try:
            if args.protocol == "loo":
                agg, per = evaluate_loo(factory, onts, catalog, top_n=args.top_n)
            else:
                sp = split_train_test(onts, args.test_frac, args.seed)
                agg, per = evaluate_recommender(factory(), sp, catalog,
                                                top_n=args.top_n)
            write_result(out_dir, name.replace("/", "_").replace(":", "_"),
                         agg, per)
            aggs.append(agg)
            log.info("  %s: MAP=%.3f R@10=%.3f hit@5=%.3f", name,
                     agg.get("mean_map", 0), agg.get("mean_recall@10", 0),
                     agg.get("mean_hits@5", 0))
        except Exception as e:
            log.warning("  %s FAILED: %s", name, e)

    # LLM backends: split protocol, on a sample of the test set (CPU cost)
    if args.with_llm:
        from .llm import LLMRerankRecommender, LLMGenerativeRecommender
        sp = split_train_test(onts, args.test_frac, args.seed)
        sp_sample = Split(train=sp.train, test=sp.test[: args.llm_sample])
        # rerank a STRONG shortlist (embedding ontology-kNN CF), not the weak
        # item retriever, so the LLM adds precision on good candidates.
        base = EmbeddingOntologyKNN(args.embspec, k=args.knn_k)
        for rec in (LLMRerankRecommender(base, model=args.llm_model),
                    LLMGenerativeRecommender(model=args.llm_model)):
            log.info("running %s on %d test ontologies ...", rec.name,
                     len(sp_sample.test))
            try:
                agg, per = evaluate_recommender(rec, sp_sample, catalog,
                                                top_n=args.top_n)
                agg["recommender"] = rec.name + f" (n={len(sp_sample.test)})"
                write_result(out_dir, rec.name.replace("/", "_").replace(":", "_"),
                             agg, per)
                aggs.append(agg)
                log.info("  %s: MAP=%.3f R@10=%.3f", rec.name,
                         agg.get("mean_map", 0), agg.get("mean_recall@10", 0))
            except Exception as e:
                log.warning("  %s FAILED: %s", rec.name, e)

    # external: Java ODPReco Mode A, if its results are present
    ext = ingest_odpreco_modeA(_USECASE / "results" / f"results_{args.dataset}")
    if ext:
        aggs.append(ext)

    path = write_comparison(out_dir, aggs)
    log.info("wrote comparison -> %s", path)
    print(f"\nComparison written to {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
