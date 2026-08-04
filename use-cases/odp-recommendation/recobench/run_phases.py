"""
recobench.run_phases  --  driver for the Phase 0/1/2/4/5 improvements.

Runs the requested phases on a dataset and writes per-backend result JSONs plus
a consolidated ``phases_report.md`` under ``results_reco/phases_<dataset>/``.

  Phase 0  trustworthy eval : bootstrap CIs for the stored roster + paired tests.
  Phase 1  better embedder  : the roster re-run with alternative embedding models
                              (--embspecs), plus the LLM pointwise scorer (sampled).
  Phase 2  hybrid LLM       : few-shot (in-context CF) rerank + self-consistency
                              (sampled, CPU cost).
  Phase 4  fusion           : learning-to-rank over CF/embedding/popularity signals.
  Phase 5  fine-tuning      : contrastive domain-adaptation of the embedder, then
                              re-evaluation of the embedding backends.

Usage:
  python -m recobench.run_phases --phases 0 4 5
  python -m recobench.run_phases --phases 1 --embspecs st:all-MiniLM-L6-v2 ollama:embeddinggemma
  python -m recobench.run_phases --phases 1 2 --llm-sample 8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .core import (DEFAULT_DATASETS, load_dataset, catalog_from, Split,
                   split_train_test, evaluate_recommender, evaluate_loo,
                   write_result)
from .baselines import (RandomRecommender, PopularityRecommender,
                        OntologyKNNRecommender)
from .content import (EmbeddingItemRecommender, EmbeddingOntologyKNN,
                      Doc2VecRecommender)

log = logging.getLogger("recobench.run_phases")
_USECASE = Path(__file__).resolve().parent.parent

METRICS = ["mean_precision@1", "mean_recall@10", "mean_map", "mean_mrr",
           "mean_ndcg@10", "mean_hits@10"]


def _fname(name: str) -> str:
    return name.replace("/", "_").replace(":", "_").replace("\\", "_") + ".json"


def _ev(rec, sp, cat, out_dir, protocol="split", onts=None, factory=None):
    if protocol == "loo":
        agg, per = evaluate_loo(factory, onts, cat, top_n=10)
    else:
        agg, per = evaluate_recommender(rec, sp, cat, top_n=10)
    write_result(out_dir, _fname(agg["recommender"])[:-5], agg, per)
    return agg


# ----------------------------------------------------------------------
def phase0(out_dir, results_dir):
    from .stats import ci_table, load_per, paired_delta
    roster = ["ontology_knn_cf", "embed_ontology_knn_cf", "popularity",
              "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]",
              "embed_item_FIXEDcatalog[st_all-MiniLM-L6-v2]", "doc2vec", "random"]
    files = [r + ".json" for r in roster if (results_dir / (r + ".json")).is_file()]
    cis = ci_table(results_dir, files, "map")
    pairs = []
    def pd(a, b, label):
        fa, fb = results_dir / (a + ".json"), results_dir / (b + ".json")
        if fa.is_file() and fb.is_file():
            pa, pb = load_per(results_dir, a + ".json"), load_per(results_dir, b + ".json")
            names = {x["ontology"] for x in pa} & {x["ontology"] for x in pb}
            pa = [x for x in pa if x["ontology"] in names]
            pb = [x for x in pb if x["ontology"] in names]
            r = paired_delta(pa, pb, "map"); r["comparison"] = label
            pairs.append(r)
    pd("ontology_knn_cf", "popularity", "CF vs popularity")
    pd("ontology_knn_cf", "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]", "CF vs content")
    lr = "llm_rerank[qwen2.5_7b-instruct]"
    if (results_dir / (lr + ".json")).is_file():
        pa = load_per(results_dir, lr + ".json")
        names8 = {x["ontology"] for x in pa}
        pb = [x for x in load_per(results_dir, "embed_ontology_knn_cf.json")
              if x["ontology"] in names8]
        r = paired_delta(pa, pb, "map"); r["comparison"] = "llm_rerank vs base (n=8)"
        pairs.append(r)
    (out_dir / "phase0_ci.json").write_text(
        json.dumps({"ci": cis, "paired": pairs}, indent=2), encoding="utf-8")
    return {"ci": cis, "paired": pairs}


def phase1_embedders(sp, cat, out_dir, embspecs):
    rows = []
    for spec in embspecs:
        tag = spec.replace(":", "_")
        for rec in (EmbeddingOntologyKNN(spec, label=f"embed_cf[{tag}]"),
                    EmbeddingItemRecommender(spec, label=f"embed_item[{tag}]")):
            try:
                rows.append(_ev(rec, sp, cat, out_dir))
                log.info("  %s: MAP=%.3f", rows[-1]["recommender"], rows[-1]["mean_map"])
            except Exception as e:
                log.warning("  %s failed: %s", spec, e)
    return rows


def phase4_fusion(sp, cat, out_dir, onts, protocol, embspec):
    from .fusion import FusionRecommender
    rows = []
    for kind in ("pairwise", "gb"):
        rec = FusionRecommender(embspec, model=kind, label=f"fusion_{kind}")
        rows.append(_ev(rec, sp, cat, out_dir, protocol, onts,
                        factory=lambda k=kind: FusionRecommender(
                            embspec, model=k, label=f"fusion_{k}")))
        log.info("  fusion_%s: MAP=%.3f", kind, rows[-1]["mean_map"])
    return rows


def phase5_finetune(sp, cat, out_dir, base="all-MiniLM-L6-v2", epochs=3):
    from .finetune import finetune_embedder
    ft = _USECASE / "recobench" / ".cache" / "ft_minilm"
    finetune_embedder(sp.train, ft, base=base, epochs=epochs)
    tuned = f"st:{ft}"
    rows = []
    for mk, spec in (("BASE", f"st:{base}"), ("TUNED", tuned)):
        for rec in (EmbeddingOntologyKNN(spec, label=f"embed_cf[{mk}]"),
                    EmbeddingItemRecommender(spec, label=f"embed_item[{mk}]")):
            rows.append(_ev(rec, sp, cat, out_dir))
            log.info("  %s: MAP=%.3f", rows[-1]["recommender"], rows[-1]["mean_map"])
    return rows


def phase12_llm(sp, cat, out_dir, model, embspec, sample):
    from .llm import (LLMRerankRecommender, LLMFewShotRerank, LLMPointwiseScorer)
    sub = Split(train=sp.train, test=sp.test[:sample])
    base = lambda: EmbeddingOntologyKNN(embspec)
    rows = []
    for rec in (LLMRerankRecommender(base(), model=model,
                                     label="llm_rerank_plain"),
                LLMPointwiseScorer(base(), model=model),
                LLMFewShotRerank(base(), model=model)):
        try:
            agg, per = evaluate_recommender(rec, sub, cat, top_n=10)
            agg["recommender"] = rec.name + f" (n={len(sub.test)})"
            write_result(out_dir, _fname(rec.name)[:-5], agg, per)
            rows.append(agg)
            log.info("  %s: MAP=%.3f R@10=%.3f", rec.name,
                     agg["mean_map"], agg["mean_recall@10"])
        except Exception as e:
            log.warning("  %s failed: %s", rec.name, e)
    return rows


def write_report(out_dir, dataset, results):
    L = [f"# Phase 0/1/2/4/5 results --- `{dataset}`\n",
         "All numbers are held-out (70/30 split, seed 13, 23 test ontologies) "
         "unless marked LOO or (n=k). Headline metric: MAP.\n"]
    if "phase0" in results:
        L.append("## Phase 0 --- trustworthy evaluation\n")
        L.append("| Backend | MAP | 95% CI |")
        L.append("|---|---|---|")
        for r in results["phase0"]["ci"]:
            L.append(f"| {r['backend'][:40]} | {r['mean']:.3f} | "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}] |")
        L.append("\n**Paired comparisons (mean MAP delta, same ontologies):**\n")
        L.append("| Comparison | ΔMAP | 95% CI | p (sign) | W/L/T |")
        L.append("|---|---|---|---|---|")
        for r in results["phase0"]["paired"]:
            L.append(f"| {r['comparison']} | {r['mean_delta']:+.3f} | "
                     f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_sign_test']:.3f} | "
                     f"{r['wins']}/{r['losses']}/{r['ties']} |")
        L.append("")
    for pkey, title in (("phase1", "Phase 1 --- better embedding models"),
                        ("phase5", "Phase 5 --- fine-tuned embedder"),
                        ("phase4", "Phase 4 --- learning-to-rank fusion"),
                        ("phase12", "Phase 1/2 --- LLM methods (sampled)")):
        if pkey not in results:
            continue
        L.append(f"## {title}\n")
        L.append("| Method | " + " | ".join(m.replace("mean_", "") for m in METRICS) + " |")
        L.append("|" + "|".join(["---"] * (len(METRICS) + 1)) + "|")
        for r in sorted(results[pkey], key=lambda a: a.get("mean_map", 0), reverse=True):
            L.append("| " + r["recommender"][:40] + " | " +
                     " | ".join(f"{r.get(m, 0):.3f}" for m in METRICS) + " |")
        L.append("")
    (out_dir / "phases_report.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", out_dir / "phases_report.md")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="noOBO", choices=sorted(DEFAULT_DATASETS))
    ap.add_argument("--phases", nargs="+", default=["0", "4", "5"])
    ap.add_argument("--embspec", default="st:all-MiniLM-L6-v2")
    ap.add_argument("--embspecs", nargs="+",
                    default=["st:all-MiniLM-L6-v2", "ollama:embeddinggemma"])
    ap.add_argument("--protocol", choices=("split", "loo"), default="split")
    ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-sample", type=int, default=8)
    ap.add_argument("--ft-epochs", type=int, default=3)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    root = DEFAULT_DATASETS[args.dataset]
    onts = load_dataset(root)
    cat = catalog_from(onts)
    sp = split_train_test(onts, 0.3, 13)
    out_dir = _USECASE / "results_reco" / f"phases_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _USECASE / "results_reco" / args.dataset
    log.info("dataset=%s onts=%d cat=%d phases=%s", args.dataset, len(onts),
             len(cat), args.phases)

    R = {}
    if "0" in args.phases:
        log.info("== Phase 0 =="); R["phase0"] = phase0(out_dir, results_dir)
    if "1" in args.phases:
        log.info("== Phase 1 (embedders) =="); R["phase1"] = phase1_embedders(sp, cat, out_dir, args.embspecs)
    if "4" in args.phases:
        log.info("== Phase 4 (fusion) =="); R["phase4"] = phase4_fusion(sp, cat, out_dir, onts, args.protocol, args.embspec)
    if "5" in args.phases:
        log.info("== Phase 5 (fine-tune) =="); R["phase5"] = phase5_finetune(sp, cat, out_dir, epochs=args.ft_epochs)
    if "1" in args.phases or "2" in args.phases:
        log.info("== Phase 1/2 (LLM) =="); R["phase12"] = phase12_llm(sp, cat, out_dir, args.llm_model, args.embspec, args.llm_sample)

    write_report(out_dir, args.dataset, R)
    print(f"\nDone. Report: {out_dir / 'phases_report.md'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
