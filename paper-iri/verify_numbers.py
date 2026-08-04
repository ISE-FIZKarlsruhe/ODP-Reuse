#!/usr/bin/env python3
"""verify_numbers.py -- assert that every headline number in paper-iri/ and
paper-iri-label/ still matches the released data.

Run from the repository root:

    python paper-iri/verify_numbers.py

Exits non-zero on the first mismatch, naming the claim, the expected value
and the value the data actually gives.  This is the executable form of
`PROVENANCE.md`: if a pipeline re-run changes a number, this script fails and
names the paper claim that has to be updated.

Nothing here reads the manuscripts; the expected values are transcribed from
them by hand, on purpose, so that a divergence between paper and data shows
up as a failure rather than being silently absorbed.
"""
from __future__ import annotations

import collections
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURES: list[str] = []
CHECKS = 0


def check(claim: str, got, want, tol: float = 0.0) -> None:
    global CHECKS
    CHECKS += 1
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and tol else (got == want)
    if not ok:
        FAILURES.append(f"{claim}\n      paper says {want!r}, data gives {got!r}")


# The iri+label track is deliberately not published (see .gitignore). A fresh
# clone therefore has none of its files, and their absence must not read as a
# verification failure -- only as a skipped section.
TRACK_B_MARKERS = ("iri-label", "iri_label")


def optional(rel: str) -> bool:
    return any(m in rel for m in TRACK_B_MARKERS)


SKIPPED: set[str] = set()


def rows(rel: str) -> list[dict]:
    p = ROOT / rel
    if not p.exists():
        if not optional(rel):
            FAILURES.append(f"MISSING FILE: {rel}")
        else:
            SKIPPED.add(rel)
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def gate(rel: str) -> dict[str, dict[str, int]]:
    return {r["dataset"]: {k: int(v) for k, v in r.items() if k != "dataset"}
            for r in rows(rel)}


def reuse_pairs(rel: str) -> list[dict]:
    return [r for r in rows(rel) if int(r["n_reused"] or 0) > 0]


# ---------------------------------------------------------------- Track A
print("== paper-iri (exact-IRI) ==")
gA = gate("release-iri/gate_summary.csv")
if gA:
    tot = lambda k: sum(gA[d][k] for d in gA)
    check("A: discovered = 1,973", tot("discovered"), 1973)
    check("A: G1 unparseable = 269", tot("dropped_G1_parse"), 269)
    check("A: G2 pattern copy = 36", tot("dropped_G2_pattern_copy"), 36)
    check("A: G3 duplicate = 203", tot("dropped_G3_duplicate"), 203)
    check("A: G4 no own entity = 540", tot("dropped_G4_substantive"), 540)
    check("A: G5 no shared entity = 424", tot("dropped_G5_has_gt"), 424)
    check("A: kept = 501", tot("kept_reuse"), 501)
    check("A: complete reuse = 51", tot("kept_complete_reuse"), 51)
    check("A: kept share = 25.4%", round(tot("kept_reuse") / 1973 * 100, 1), 25.4)
    check("A: GitHub kept = 442", gA["github"]["kept_reuse"], 442)
    check("A: Paper&Repo kept = 51", gA["noOBO"]["kept_reuse"], 51)
    check("A: audit total removals = 1,472",
          sum(tot(f"dropped_{r}") for r in
              ("G1_parse", "G2_pattern_copy", "G3_duplicate",
               "G4_substantive", "G5_has_gt")), 1472)

# The released copy, not pipeline/out: byte-identical, but this way the
# verifier checks the artefact a user downloads rather than local scratch.
au = rows("release-iri/audit/audit_summary.csv")
if au:
    check("A: audit disputed = 0",
          sum(int(r["n_removed"]) - int(r["ok"]) for r in au), 0)
    check("A: audit re-derived = 1,472", sum(int(r["ok"]) for r in au), 1472)

pA = reuse_pairs("pipeline/out/rq_iri_gated/pairs.csv")
if pA:
    check("A: reuse pairs = 1,517", len(pA), 1517)
    check("A: distinct patterns = 488", len({r["pattern"] for r in pA}), 488)
    comp = [r for r in pA if r["complete"] == "1"]
    check("A: complete pairs = 673", len(comp), 673)
    check("A: complete distinct patterns = 311",
          len({r["pattern"] for r in comp}), 311)
    ghpr = [r for r in pA if r["dataset"] in ("github", "noOBO")]
    check("A: gated IRI pairs GH+PR = 759", len(ghpr), 759)

eA = {r["dataset"]: r for r in rows("pipeline/out/rq_iri_gated/rq1_extent.csv")}
if eA:
    check("A: RQ1 GitHub mean patterns/onto = 1.51",
          float(eA["github"]["mean_patterns_per_onto"]), 1.51, 0.005)
    check("A: RQ1 GitHub distinct patterns = 45",
          int(eA["github"]["patterns_reuse"]), 45)
    check("A: RQ1 GitHub complete share = 8.4%",
          round(float(eA["github"]["ge_100pct_share"]), 1), 8.4)
    check("A: RQ1 GitHub >=10% = 233", int(eA["github"]["ge_10pct"]), 233)
    check("A: RQ1 Paper&Repo mean patterns/onto = 1.76",
          float(eA["noOBO"]["mean_patterns_per_onto"]), 1.76, 0.005)
    check("A: RQ1 Paper&Repo >=10% share = 74.5%",
          round(float(eA["noOBO"]["ge_10pct_share"]), 1), 74.5)
    check("A: RQ1 DOS-DP mean patterns/onto = 94.75",
          float(eA["OBO"]["mean_patterns_per_onto"]), 94.75, 0.005)

fA = {r["dataset"]: r for r in rows("pipeline/out/rq_iri_gated/rq2_reuse_forms.csv")}
if fA:
    pct = lambda d, c: round(float(fA[d][c + "_pct"]), 1)
    check("A: RQ2 GitHub direct = 31.8%", pct("github", "direct"), 31.8)
    check("A: RQ2 GitHub used_as_property = 18.3%",
          pct("github", "used_as_property"), 18.3, 0.05)
    check("A: RQ2 GitHub in_restriction = 17.4%",
          pct("github", "in_restriction"), 17.4)
    check("A: RQ2 GitHub subclass = 13.9%", pct("github", "subclass_of_pattern"), 13.9)
    check("A: RQ2 Paper&Repo direct = 37.5%", pct("noOBO", "direct"), 37.5, 0.05)
    check("A: RQ2 Paper&Repo in_restriction = 25.2%",
          pct("noOBO", "in_restriction"), 25.2, 0.05)
    check("A: RQ2 DOS-DP in_restriction = 99.3%",
          pct("OBO", "in_restriction"), 99.3)
    check("A: RQ2 GitHub used_as_property is the 2nd largest form",
          max((float(fA["github"][c + "_pct"]), c) for c in
              ("subclass_of_pattern", "superclass_of_pattern",
               "subproperty_of_pattern", "superproperty_of_pattern",
               "used_as_property", "in_restriction", "equivalence"))[1],
          "used_as_property")

cA = {r["dataset"]: r for r in rows("pipeline/out/rq_iri_gated/rq3_complexity.csv")}
if cA:
    check("A: RQ3 GitHub pairs = 669", int(cA["github"]["n_pairs"]), 669)
    check("A: RQ3 GitHub modal pattern DL = ALCHI",
          cA["github"]["pattern_dl_mode"], "ALCHI")
    check("A: RQ3 GitHub onto more expressive = 70.5%",
          round(float(cA["github"]["pct_ontology_more_expressive"]), 1), 70.5)
    check("A: RQ3 Paper&Repo onto more expressive = 63.3%",
          round(float(cA["noOBO"]["pct_ontology_more_expressive"]), 1), 63.3)
    check("A: RQ3 DOS-DP modal pattern DL = EL",
          cA["OBO"]["pattern_dl_mode"], "EL")

xA = {r["dataset"]: r for r in rows("pipeline/out/rq_iri_gated/rq4_expansion.csv")}
if xA:
    check("A: RQ3 GitHub entities added = 91.6%",
          round(float(xA["github"]["pct_entities_added"]), 1), 91.6)
    check("A: RQ3 Paper&Repo entities added = 86.7%",
          round(float(xA["noOBO"]["pct_entities_added"]), 1), 86.7)

man = rows("release-iri/MANIFEST.csv")
if man:
    keptrows = [r for r in man if not r["drop_reason"]]
    share = [int(r["n_own_entities"] or 0) / int(r["n_entities_wide"])
             for r in keptrows if int(r["n_entities_wide"] or 0)]
    check("A: RQ4 median own share of kept = 92%",
          round(statistics.median(share) * 100), 92)
    band = [r for r in keptrows if 1 <= int(r["n_own_entities"] or 0) <= 9]
    check("A: RQ4 ontologies in the 1-9 own-entity band = 72", len(band), 72)
    bs = [int(r["n_own_entities"]) / int(r["n_entities_wide"]) for r in band
          if int(r["n_entities_wide"] or 0)]
    check("A: RQ4 median own share of that band = 64%",
          round(statistics.median(bs) * 100), 64)
    check("A: RQ4 of the 72, >=50% own = 55", sum(1 for x in bs if x >= 0.5), 55)
    g4 = [r for r in man if r["drop_reason"].startswith("G4")]
    check("A: RQ4 all G4 removals contribute zero own entities",
          max((int(r["n_own_entities"] or 0) for r in g4), default=-1), 0)

# RQ4: the "why not" partition of the G5 removals
g5c = rows("pipeline/out/g5/g5_causes.csv")
if g5c and gA:
    tot = collections.Counter()
    for r in g5c:
        tot[r["cause"]] += int(r["n"])
    n_g5 = sum(tot.values())
    check("A: RQ4 G5 removals characterised = 424", n_g5, 424)
    check("A: RQ4 re-modelled confirmed = 11", tot["remodelled_confirmed"], 11)
    check("A: RQ4 re-modelled candidate = 66", tot["remodelled_candidate"], 66)
    check("A: RQ4 no decidable overlap = 5", tot["no_pattern_entities"], 5)
    check("A: RQ4 disjoint residual = 342", tot["disjoint"], 342)
    check("A: RQ4 disjoint share = 80.7%",
          round(tot["disjoint"] / n_g5 * 100, 1), 80.7)
    kept_gh = gA["github"]["kept_reuse"]
    g5_gh = gA["github"]["dropped_G5_has_gt"]
    check("A: RQ4 search-as-proxy precision = 52.6%",
          round(kept_gh / (kept_gh + g5_gh) * 100, 1), 52.6)
    check("A: RQ4 substantive GitHub population = 841", kept_gh + g5_gh, 841)

# UC2: feature-target entanglement (median share of an ontology's entities
# that are reused pattern vocabulary) -- the number that answers the
# "is 0.815 MAP leakage?" objection.
ent = ROOT / "use-cases/odp-recommendation/stats_iri_github/feature_target_entanglement.txt"
if ent.exists():
    txt = ent.read_text(encoding="utf-8")
    check("A: UC2 entanglement median = 7.4%", "median : 7.4%" in txt, True)
    check("A: UC2 entanglement mean = 15.5%", "mean   : 15.5%" in txt, True)

sens = {int(r["threshold"]): r for r in rows("pipeline/out/g4_threshold_sensitivity.csv")}
if sens:
    check("A: sensitivity kept at T=1 = 501", int(sens[1]["kept"]), 501)
    check("A: sensitivity kept at T=0 = 509", int(sens[0]["kept"]), 509)
    check("A: sensitivity kept at T=20 = 360", int(sens[20]["kept"]), 360)
    check("A: distinct patterns = 45 for every T up to 10",
          {int(sens[t]["github_patterns"]) for t in (0, 1, 2, 3, 4, 5, 7, 10)}, {45})
    check("A: distinct patterns = 44 at T=15 and T=20",
          {int(sens[t]["github_patterns"]) for t in (15, 20)}, {44})
    check("A: mean patterns/onto stays in 1.51-1.62 over T=0..20",
          (min(float(sens[t]["github_mean_patterns_per_onto"])
               for t in (0, 1, 2, 3, 4, 5, 7, 10, 15, 20)),
           max(float(sens[t]["github_mean_patterns_per_onto"])
               for t in (0, 1, 2, 3, 4, 5, 7, 10, 15, 20))), (1.51, 1.62))
    check("A: complete share stays in 8.2-9.7% over T=0..20",
          (min(float(sens[t]["github_complete_pct"])
               for t in (0, 1, 2, 3, 4, 5, 7, 10, 15, 20)),
           max(float(sens[t]["github_complete_pct"])
               for t in (0, 1, 2, 3, 4, 5, 7, 10, 15, 20))), (8.2, 9.7))

uc1 = {(r["dataset"], r["method"], r["clusterer"]): r
       for r in rows("use-cases/odp-extraction/results/iri/rq5_results.csv")}
if uc1:
    gaps = [float(r["oracle_gap_f1"]) for r in uc1.values()]
    check("A: UC1 oracle gap min = 0.041", round(min(gaps), 3), 0.041)
    check("A: UC1 oracle gap max = 0.167", round(max(gaps), 3), 0.167)
    k = ("github", "owl2vec", "kmeans")
    check("A: UC1 GitHub OWL2Vec*/kmeans unsup F1 = 0.321",
          round(float(uc1[k]["unsup_f1"]), 3), 0.321)
    check("A: UC1 GitHub OWL2Vec*/kmeans oracle F1 = 0.453",
          round(float(uc1[k]["oracle_f1"]), 3), 0.453)
    check("A: UC1 GitHub best oracle is RDF2Vec/kmeans = 0.468",
          round(float(uc1[("github", "rdf2vec", "kmeans")]["oracle_f1"]), 3), 0.468)
    check("A: UC1 kmeans beats hdbscan in all 8 rows",
          all(float(uc1[(d, m, "kmeans")]["oracle_f1"])
              > float(uc1[(d, m, "hdbscan")]["oracle_f1"])
              and float(uc1[(d, m, "kmeans")]["unsup_f1"])
              > float(uc1[(d, m, "hdbscan")]["unsup_f1"])
              for d in ("github", "noOBO") for m in ("owl2vec", "rdf2vec")), True)

# UC1 diagnostics: the null baseline, the cluster counts and the split/merge
# structure. These are the numbers that say what the embeddings contribute.
CS = "use-cases/odp-extraction/results/iri/clusterstats"
nb = {(r["model"], r["clusterer"], r["regime"]): r
      for r in rows(f"{CS}/t8_null_baseline_summary.csv") if r["portion"] == "ALL"}
if nb:
    check("A: UC1 null-baseline population = 406",
          int(nb[("owl2vec", "kmeans", "oracle")]["n_ontologies"]), 406)
    check("A: UC1 oracle OWL2Vec*/kmeans lift over null = -0.003",
          round(float(nb[("owl2vec", "kmeans", "oracle")]["lift_over_null"]), 3),
          -0.003)
    check("A: UC1 unsup OWL2Vec*/kmeans lift over null = +0.066",
          round(float(nb[("owl2vec", "kmeans", "unsup")]["lift_over_null"]), 3),
          0.066)
    check("A: UC1 no oracle config exceeds lift +0.018",
          round(max(float(v["lift_over_null"]) for k, v in nb.items()
                    if k[2] == "oracle"), 3), 0.018)
    check("A: UC1 every unsup config has positive lift",
          all(float(v["lift_over_null"]) > 0 for k, v in nb.items()
              if k[2] == "unsup"), True)
    check("A: UC1 oracle ratios stay within 0.99-1.05",
          (round(min(float(v["ratio_over_null"]) for k, v in nb.items()
                     if k[2] == "oracle"), 2),
           round(max(float(v["ratio_over_null"]) for k, v in nb.items()
                     if k[2] == "oracle"), 2)), (0.99, 1.05))
    check("A: UC1 one-cluster baseline = 0.204",
          round(float(nb[("owl2vec", "kmeans", "oracle")]["mean_f1_single_cluster"]), 3),
          0.204)

RC = "use-cases/odp-recommendation"
# UC2 is reported on the QUALITY-GATED population. The earlier ungated run is
# retained on disk for comparison but is not what the paper quotes.
mA = {r["recommender"]: r for r in
      rows(f"{RC}/stats_iri_github_gated/reco_ci_map_github.csv")}
if mA:
    check("A: UC2 n held out = 89", int(mA["ontology_knn_cf"]["n"]), 89)
    check("A: UC2 CF MAP = 0.863",
          round(float(mA["ontology_knn_cf"]["mean_map"]), 3), 0.863)
    check("A: UC2 embedding CF MAP = 0.792",
          round(float(mA["embed_ontology_knn_cf"]["mean_map"]), 3), 0.792)
    check("A: UC2 popularity MAP = 0.661",
          round(float(mA["popularity"]["mean_map"]), 3), 0.661)
    check("A: UC2 dataset-catalogue MAP = 0.105",
          round(float(mA["embed_item_datasetcatalog[st_all-MiniLM-L6-v2]"]["mean_map"]), 3),
          0.105)
    check("A: UC2 doc2vec MAP = 0.058", round(float(mA["doc2vec"]["mean_map"]), 3), 0.058)
    check("A: UC2 random MAP = 0.041", round(float(mA["random"]["mean_map"]), 3), 0.041)

tA = {r["comparison"]: r for r in
      rows(f"{RC}/stats_iri_github_gated/reco_paired_tests_github.csv")}
if tA:
    doc = "CF vs Doc2Vec content matching"
    pop = "CF vs popularity prior"
    fl = "dataset-informed CF vs no-dataset floor"
    # row 1 -- the measurement of what the dataset is worth
    check("A: UC2 dataset benefit vs documentation = +0.805",
          round(float(tA[doc]["mean_delta_map"]), 3), 0.805)
    check("A: UC2 dataset benefit W/L/T = 83/3/3", tA[doc]["win_loss_tie"], "83/3/3")
    # row 2 -- personalisation over the matrix
    check("A: UC2 personalisation vs popularity = +0.201",
          round(float(tA[pop]["mean_delta_map"]), 3), 0.201)
    check("A: UC2 personalisation W/L/T = 31/2/56", tA[pop]["win_loss_tie"], "31/2/56")
    # row 3 -- the weakest comparison, reported for completeness
    check("A: UC2 vs random floor = +0.822",
          round(float(tA[fl]["mean_delta_map"]), 3), 0.822)
    check("A: UC2 all three gated comparisons significant",
          {r["significant_at_0.05"] for r in tA.values()}, {"yes"})

# The underpowered curated portion: none significant, and row 2 reverses sign.
tP = {r["comparison"]: r for r in rows(f"{RC}/stats_iri/reco_paired_tests_noOBO.csv")}
if tP:
    check("A: UC2 Paper&Repo n = 14",
          int(tP["CF vs popularity prior"]["n_paired"]), 14)
    check("A: UC2 Paper&Repo popularity comparison reverses sign",
          float(tP["CF vs popularity prior"]["mean_delta_map"]) < 0, True)
    check("A: UC2 Paper&Repo nothing significant",
          {r["significant_at_0.05"] for r in tP.values()}, {"no"})

# The structural near-duplicate probe (reviewer request (e)).
sp = ROOT / "pipeline/out/structprobe/summary.json"
if sp.exists():
    import json
    s = json.loads(sp.read_text(encoding="utf-8"))
    check("A: G7 probe ontologies scanned = 1,696", s["ontologies_scanned"], 1696)
    check("A: G7 probe pairs evaluated = 2,691", s["pairs_evaluated"], 2691)
    check("A: G7 S1 (IRI-perturbed copy) = 0 occurrences", s["S1_pairs"], 0)
    check("A: G7 S2 pairs = 28", s["S2_pairs"], 28)
    check("A: G7 S2 ontologies = 18", s["S2_ontologies"], 18)
    check("A: G7 S2 rate = 1.06%", round(s["S2_rate_of_scanned"] * 100, 2), 1.06)
    check("A: G7 S2 perturbation is namespace-only in all 28",
          s["S2_perturbation_profile"], {"namespace_only": 28})
    check("A: G7 S2 gate breakdown = 14 G5 / 2 G3 / 2 kept",
          s["S2_by_drop_reason"],
          {"(kept)": 2, "G5_has_gt": 14, "G3_duplicate": 2})

# adjudication / recall (shared by both papers)
hs = rows("pipeline/out/adjudication/human_sample.csv")
if hs and man:
    check("shared: adjudicated items = 331", len(hs), 331)
    conf = [r for r in hs if r.get("human_verdict", "").strip().lower() == "reuse"]
    check("shared: confirmed genuine pairs = 71", len(conf), 71)
    check("shared: label-mode precision = 0.215", round(len(conf) / len(hs), 3), 0.215)
    check("shared: confirmed distinct ontologies = 58",
          len({(r["dataset"], r["ontology"]) for r in conf}), 58)
    mi = {(r["dataset"], r["ontology_id"]): r for r in man}
    gp = [r for r in conf
          if mi.get((r["dataset"], r["ontology"])) and
          not mi[(r["dataset"], r["ontology"])]["drop_reason"]]
    check("shared: confirmed pairs w/ gate-passing ontology = 40", len(gp), 40)
    if pA:
        n = len([r for r in pA if r["dataset"] in ("github", "noOBO")])
        check("shared: exact-IRI recall (gated) = 95.0%",
              round(n / (n + len(gp)) * 100, 1), 95.0)
    st = collections.Counter(
        (mi[(r["dataset"], r["ontology"])]["drop_reason"] or "KEPT").split(":")[0]
        for r in {(r["dataset"], r["ontology"]): r for r in conf}.values()
        if (r["dataset"], r["ontology"]) in mi)
    check("shared: of the 58, gate-passing = 33", st["KEPT"], 33)
    check("shared: of the 58, G5-only (newly admitted) = 11", st["G5_has_gt"], 11)

# ---------------------------------------------------------------- Track B
TRACK_B = (ROOT / "release-iri-label/gate_summary.csv").exists()
if TRACK_B:
    print("== paper-iri-label (verified union) ==")
else:
    print("== paper-iri-label: not present in this checkout, skipped ==")
    print("   (the iri+label track is local-only; see .gitignore)")
gB = gate("release-iri-label/gate_summary.csv")
if gB and gA:
    totB = lambda k: sum(gB[d][k] for d in gB)
    check("B: kept = 512", totB("kept_reuse"), 512)
    check("B: complete reuse = 51", totB("kept_complete_reuse"), 51)
    check("B: GitHub kept = 450", gB["github"]["kept_reuse"], 450)
    check("B: Paper&Repo kept = 54", gB["noOBO"]["kept_reuse"], 54)
    check("B: delta kept vs iri = +11",
          totB("kept_reuse") - sum(gA[d]["kept_reuse"] for d in gA), 11)
    check("B: delta complete vs iri = 0",
          totB("kept_complete_reuse")
          - sum(gA[d]["kept_complete_reuse"] for d in gA), 0)

pB = reuse_pairs("pipeline/out/rq_iri-label_gated/pairs.csv")
if pB and pA:
    check("B: reuse pairs = 1,534", len(pB), 1534)
    check("B: delta pairs vs iri = +17", len(pB) - len(pA), 17)

manB = rows("pipeline/out_iri_label/MANIFEST.csv")
if manB:
    lab = [r for r in manB if int(r["n_label_confirmed"] or 0) > 0]
    check("B: ontologies with confirmed labels = 54", len(lab), 54)
    check("B: confirmed patterns added = 67",
          sum(int(r["n_label_confirmed"]) for r in lab), 67)
    check("B: confirmed entities added = 170",
          sum(int(r["n_label_entities"]) for r in lab), 170)
    check("B: of those, gate-passing = 44",
          sum(1 for r in lab if not r["drop_reason"]), 44)

eB = {r["dataset"]: r for r in rows("pipeline/out/rq_iri-label_gated/rq1_extent.csv")}
if eB:
    check("B: RQ1 GitHub mean patterns/onto = 1.52",
          float(eB["github"]["mean_patterns_per_onto"]), 1.52, 0.005)
    check("B: RQ1 GitHub distinct patterns = 46",
          int(eB["github"]["patterns_reuse"]), 46)
    check("B: RQ1 Paper&Repo distinct patterns = 49",
          int(eB["noOBO"]["patterns_reuse"]), 49)

cB = {r["dataset"]: r for r in rows("pipeline/out/rq_iri-label_gated/rq3_complexity.csv")}
if cB:
    check("B: RQ3 GitHub pairs = 682", int(cB["github"]["n_pairs"]), 682)
    check("B: RQ3 GitHub modal ontology DL = ALCHIQ",
          cB["github"]["ontology_dl_mode"], "ALCHIQ")

uc1B = {(r["dataset"], r["method"], r["clusterer"]): r
        for r in rows("use-cases/odp-extraction/results/iri-label/rq5_results.csv")}
if uc1B and uc1:
    k = ("github", "owl2vec", "kmeans")
    check("B: UC1 GitHub OWL2Vec*/kmeans unsup F1 = 0.329",
          round(float(uc1B[k]["unsup_f1"]), 3), 0.329)
    gh = [(d, m) for d in ("github",) for m in ("owl2vec", "rdf2vec")]
    gains = sum(1 for d, m in gh for c in ("kmeans", "hdbscan")
                if float(uc1B[(d, m, c)]["unsup_f1"]) > float(uc1[(d, m, c)]["unsup_f1"]))
    check("B: UC1 GitHub unsup improves in 3 of 4 configs", gains, 3)

# Track B's UC2 must be compared against Track A on the SAME population.
# Track A is now gated; the gated Track B re-run supersedes the ungated one.
# Until it lands, the iri vs iri+label recommendation delta is not computable
# and the paper must not quote one.
mB = {r["recommender"]: r for r in
      rows(f"{RC}/stats_iri-label_github_gated/reco_ci_map_github.csv")}
if mB and mA:
    check("B: UC2 n held out = 93", int(mB["ontology_knn_cf"]["n"]), 93)
    # 3-dp display tolerance: the file holds 0.8005; the paper rounds half-up.
    check("B: UC2 CF MAP = 0.801",
          float(mB["ontology_knn_cf"]["mean_map"]), 0.801, 0.0006)
    check("B: UC2 embedding CF MAP = 0.758",
          round(float(mB["embed_ontology_knn_cf"]["mean_map"]), 3), 0.758)
    check("B: UC2 popularity MAP = 0.609",
          round(float(mB["popularity"]["mean_map"]), 3), 0.609)
    check("B: UC2 CF MAP delta vs iri = -0.062",
          round(float(mB["ontology_knn_cf"]["mean_map"])
                - float(mA["ontology_knn_cf"]["mean_map"]), 3), -0.062)
    # the mechanism: reuse-informed recommenders lose, content-based ones gain
    check("B: UC2 every reuse-informed recommender loses accuracy",
          all(float(mB[r]["mean_map"]) < float(mA[r]["mean_map"])
              for r in ("ontology_knn_cf", "embed_ontology_knn_cf",
                        "popularity")), True)
    check("B: UC2 both content-based recommenders gain",
          all(float(mB[r]["mean_map"]) > float(mA[r]["mean_map"])
              for r in ("doc2vec",
                        "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]")), True)

tB = {r["comparison"]: r for r in
      rows(f"{RC}/stats_iri-label_github_gated/reco_paired_tests_github.csv")}
if tB and tA:
    check("B: UC2 dataset benefit vs documentation = +0.726",
          round(float(tB["CF vs Doc2Vec content matching"]["mean_delta_map"]), 3),
          0.726)
    check("B: UC2 vs random floor = +0.762",
          round(float(tB["dataset-informed CF vs no-dataset floor"]["mean_delta_map"]), 3),
          0.762)
    check("B: UC2 personalisation vs popularity = +0.192",
          round(float(tB["CF vs popularity prior"]["mean_delta_map"]), 3), 0.192)
    check("B: UC2 all three gated comparisons still significant",
          {r["significant_at_0.05"] for r in tB.values()}, {"yes"})

tB = {r["comparison"]: r for r in rows(f"{RC}/stats_iri-label_github/reco_paired_tests_github.csv")}
if tB:
    check("B: UC2 CF vs random delta = +0.721",
          round(float(tB["dataset-informed CF vs no-dataset floor"]["mean_delta_map"]), 3),
          0.721)
    check("B: UC2 all three GitHub comparisons still significant",
          {r["significant_at_0.05"] for r in tB.values()}, {"yes"})

# ---------------------------------------------------------------- verdict
print()
if FAILURES:
    print(f"[FAIL] {len(FAILURES)} of {CHECKS} checks failed:\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"[OK] all {CHECKS} headline numbers match the released data.")

# State what was skipped rather than leaving the reader to wonder why the count
# differs between a clone and a full working tree. A bare "all N passed" invites
# the assumption that N is fixed; it is not, and the difference is the Track B
# apparatus, which a clone does not contain.
if not TRACK_B:
    absent = sorted({r for r in SKIPPED})
    print(f"     {len(absent)} Track B source(s) absent, so the iri+label "
          f"assertions were skipped.")
    print(f"     A full working tree runs more; this checkout ran {CHECKS}.")
