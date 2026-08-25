"""Per-method views, summary tables and the signal documentation for the MSE
release, mirroring what ``release-iri`` provides for the benchmark.

``release-iri`` gives a reader three things beyond the raw corpus: a statistics
table per view, a summary of how the corpus was reduced, and a document stating
each filter rule with its justification. This module produces the MSE analogues:

  dataset_statistics.csv   one row per method, every yield and signal figure
  corpus_summary.md/.csv   corpus composition, the upstream currency audit, and
                           the exclusions -- the analogue of gate_summary
  views/<method>/          the groups that method proposed, flattened to CSV so
                           a reader never has to parse JSON Lines
  SIGNALS.md               what each ground-truth-free signal means, and how it
                           fails -- the analogue of FILTERS.md

The MSE release has no gate and no ground truth, so there is no attrition table
and no per-view pair file. What replaces them is the currency audit (which
ontologies were refreshed and which could not be) and the exclusion record.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

SIGNALS_MD = """# Signals

This release reports no accuracy, because the corpus has no ground truth. It
reports five signals instead. Each is a ranking device for deciding which
candidates deserve a domain expert's attention, and each has a failure mode that
a consumer needs to know. They are stated here in the way `release-iri/FILTERS.md`
states the gate rules: what the signal is, why it is computed, and what it does
*not* license.

---

## 1. Yield

**What.** Groups proposed per ontology; group size; the share of an ontology's
entities that appear in at least one group.

**Why.** Any evaluation that costs human attention per group is not evaluating
the same amount of work per method. A method proposing 51 groups per ontology
and one proposing 1.7 are not comparable on a per-group measure.

**Failure mode.** Coverage is 1.000 for any method that *partitions* — the
reference points and both K-Means configurations — and lower for methods that
*select*. The column separates two kinds of method; it does not rank them.

## 2. Cross-method agreement

**What.** Hungarian entity-level $F_1$ between two methods' groups on the same
ontology, the same metric used to score against ground truth, with one method's
output standing in for the answer key.

**Why.** Two methods built on different principles proposing the same group is
evidence the group is a structure of the ontology rather than an artefact of one
algorithm.

**Failure mode.** Agreement *within* a family measures shared implementation,
not shared insight: two configurations of the same embedder reach 0.78, and the
two stages of one pipeline 0.70, while no pair drawn from two different families
exceeds 0.23. Compute this signal across families or not at all. Trivial
reference predictions reach 0.90, which is the scale any value should be read
against.

## 3. Recurrence across ontologies

**What.** Whether a group's entity set, or its set of local names, also appears
in another ontology.

**Why.** A design pattern recurs by definition. A group that appears in several
ontologies has evidence no single-ontology measure can give.

**Failure mode.** **This is the signal most likely to mislead.** Ranking by
recurrence alone puts pairs of BFO upper-level classes at the top, because a
dozen MSE ontologies import BFO. That is shared *scaffolding*, not shared
design. Exclude groups composed entirely of foundational terms (BFO, EMMO, IAO,
RO, PMDco, CCO, QUDT) and require at least one term from the host ontology's own
namespace before using this signal to rank.

## 4. Foundational anchoring

**What.** Whether a group's entities include a term from a foundational or
mid-level vocabulary.

**Why.** A group specialising an upper-level ontology is more likely a modelling
decision than an incidental clique.

**Failure mode.** It rewards methods that key on subsumption. Axiom-tree mining
reaches 86% anchored because it mines `SubClassOf` and `EquivalentTo` axioms,
and in this corpus those axioms overwhelmingly attach a domain class to an
upper-level one. High anchoring can mean the method found the domain's
foundational commitment rather than a pattern.

## 5. Contact with a documented ODP

**What.** Whether a group overlaps any of the 120 reference patterns of the
ODP-Reuse libraries.

**Why.** A group recovering a documented pattern is a validation signal; a group
overlapping none is a candidate novel one. Both matter, differently.

**Failure mode.** As a per-group rate it rewards coarseness: a method proposing
large groups intersects a known pattern more often by chance. The ordering of
the coarse methods on this signal largely reproduces the ordering of their group
sizes.

---

## What none of them supports

No combination of these signals establishes that a candidate is a good design
pattern. They decide what an expert is asked about. The expert answers.
"""


def write_dataset_statistics(out: Path, stats: dict) -> None:
    """One row per method: yield plus every signal. The MSE analogue of
    release-iri's per-view dataset_statistics.csv."""
    anc = {a["method"]: a for a in stats.get("D_foundational_anchoring", [])}
    kn = {c["method"]: c
          for c in stats.get("E_known_odp_contact", {}).get("per_method", [])}
    rows = []
    for y in stats.get("A_yield", []):
        m = y["method"]
        a, k = anc.get(m, {}), kn.get(m, {})
        rows.append({
            "method": m,
            "n_ontologies": y["n_ontologies"],
            "groups_per_ontology_mean": y["groups_per_ontology_mean"],
            "groups_per_ontology_median": y["groups_per_ontology_median"],
            "group_size_mean": y["group_size_mean"],
            "group_size_median": y["group_size_median"],
            "groups_ge_2": y["groups_ge_min_size"],
            "entity_coverage_mean": y["entity_coverage_mean"],
            "anchored": a.get("anchored", ""),
            "anchored_pct": a.get("pct_anchored", ""),
            "touching_known_odp": k.get("touching_known_odp", ""),
            "touching_known_odp_pct": k.get("pct", ""),
        })
    if not rows:
        return
    with (out / "dataset_statistics.csv").open("w", newline="",
                                               encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_agreement_matrix(out: Path, stats: dict) -> None:
    pairs = stats.get("B_cross_method_agreement", [])
    if not pairs:
        return
    methods = sorted({p["method_a"] for p in pairs}
                     | {p["method_b"] for p in pairs})
    grid = {(p["method_a"], p["method_b"]): p["mean_best_match_f1"] for p in pairs}
    grid.update({(b, a): v for (a, b), v in list(grid.items())})
    with (out / "statistics" / "agreement_matrix.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + methods)
        for a in methods:
            w.writerow([a] + [("1.0000" if a == b
                               else f'{grid.get((a, b), ""):.4f}'
                               if (a, b) in grid else "")
                              for b in methods])


def write_recurrence_and_shortlist(out: Path, stats: dict) -> None:
    for key, name in (("C_recurrence_by_iri", "recurrence_by_iri.csv"),
                      ("C_recurrence_by_localname", "recurrence_by_localname.csv")):
        c = stats.get(key)
        if not c:
            continue
        with (out / "statistics" / name).open("w", newline="",
                                              encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["n_ontologies", "n_occurrences", "size", "methods",
                        "members"])
            for t in c.get("top", []):
                w.writerow([t["n_ontologies"], t["n_occurrences"], t["size"],
                            "|".join(t["methods"]), "|".join(t["members"])])
    sl = stats.get("F_review_shortlist", {})
    if sl.get("top"):
        with (out / "views" / "review_shortlist.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["priority", "size", "n_ontologies", "n_methods",
                        "methods", "ontologies", "local_names", "example_iris"])
            for c in sl["top"]:
                w.writerow([c["priority"], c["size"], c["n_ontologies"],
                            c["n_methods"], "|".join(c["methods"]),
                            "|".join(c["ontologies"]),
                            "|".join(c["local_names"]),
                            "|".join(c["example_iris"])])


def write_method_views(out: Path, predictions_dir: Path) -> dict:
    """views/<method>/groups.csv -- every group flattened, so the release is
    usable without parsing JSON Lines."""
    vroot = out / "views"
    vroot.mkdir(exist_ok=True)
    per_method: dict[str, list] = defaultdict(list)
    for pf in sorted(predictions_dir.glob("*.jsonl")):
        for line in pf.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for i, c in enumerate(r.get("clusters", [])):
                per_method[r["method"]].append(
                    (r["ontology"], i, len(c), "|".join(sorted(c))))
    counts = {}
    for m, rows in sorted(per_method.items()):
        d = vroot / m.replace("|", "_")
        d.mkdir(exist_ok=True)
        with (d / "groups.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["ontology", "group_id", "size", "entities"])
            w.writerows(rows)
        counts[m] = len(rows)
    with (vroot / "README.md").open("w", encoding="utf-8") as fh:
        fh.write("# Views\n\nOne directory per method, holding every group it "
                 "proposed, flattened from the JSON Lines in `predictions/` so "
                 "that the release is usable without a JSON parser.\n\n"
                 "`entities` is a `|`-separated list of entity IRIs. There is no "
                 "score column: this corpus has no ground truth.\n\n"
                 "`review_shortlist.csv` is the ranked candidate set for expert "
                 "review. Read `SIGNALS.md` before using its ordering.\n\n"
                 "| method | groups |\n|---|---:|\n")
        for m, n in sorted(counts.items()):
            fh.write(f"| `{m}` | {n:,} |\n")
    return counts


def write_corpus_summary(out: Path, rows: list[dict], skips: list[dict],
                         upstream: dict | None) -> None:
    """The analogue of gate_summary: how the corpus was assembled and reduced."""
    n = len(rows)
    parsed = sum(1 for r in rows if r["parsed"] == "ok")
    refreshed = sum(1 for r in rows if r["refreshed_from_upstream"] == "yes")
    reasons = Counter(s.get("reason", "") for s in skips)
    up = Counter()
    if upstream:
        up = Counter(r.get("status", "") for r in upstream.get("all_results", []))

    with (out / "corpus_summary.csv").open("w", newline="",
                                           encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measure", "value"])
        w.writerow(["files_in_corpus", n])
        w.writerow(["files_parsed", parsed])
        w.writerow(["files_refreshed_from_upstream", refreshed])
        for k, v in sorted(up.items()):
            w.writerow([f"upstream_{k.lower()}", v])
        for k, v in reasons.most_common():
            w.writerow([f"excluded_{k}", v])

    lines = [
        "# Corpus summary", "",
        "This release has **no quality gate and no ground truth**, so there is "
        "no attrition table of the kind `release-iri/gate_summary.md` carries. "
        "What determines the population here is whether a file parses, whether "
        "it declares any entity, and what each run's cost bound admits. All "
        "three are recorded below and per-ontology in `MANIFEST.csv` and "
        "`skipped.csv`.", "",
        "## Composition", "",
        "| measure | files |", "|---|---:|",
        f"| in the corpus | {n} |",
        f"| parse as RDF | {parsed} |",
        f"| do not parse | {n - parsed} |",
        f"| refreshed from upstream for this release | {refreshed} |",
        "",
    ]
    if up:
        lines += [
            "## Upstream currency audit", "",
            "Every parseable file was checked against its upstream before "
            "evaluation, and each proposed update was independently re-checked "
            "before it was applied. `provenance/upstream_check.json` holds the "
            "verdict for every file, including those where nothing newer was "
            "found.", "",
            "| verdict | files |", "|---|---:|",
        ]
        label = {"UP_TO_DATE": "already current",
                 "UPDATE_AVAILABLE": "newer release found",
                 "UPSTREAM_GONE": "no reachable upstream",
                 "UNKNOWN": "could not be established",
                 "NOT_AN_ONTOLOGY": "not an ontology"}
        for k, v in sorted(up.items()):
            lines.append(f"| {label.get(k, k)} | {v} |")
        lines += ["",
                  f"Of the artefacts with a newer release, {refreshed} were "
                  "applied; the rest were rejected by the re-check or by the "
                  "entity-count guard, and `provenance/corpus_updates.csv` "
                  "records each decision with its reason.", ""]
    if reasons:
        lines += ["## Exclusions", "",
                  "Every ontology a run could not use, with the reason. An "
                  "exclusion is data, not noise: a silent drop is how a "
                  "population quietly stops being the one a reader thinks it "
                  "is.", "",
                  "| reason | records |", "|---|---:|"]
        explain = {
            "no_ground_truth_predict_only":
                "no answer key, so predicted but not scored (expected here)",
            "parse_error": "the file is not parseable RDF",
            "no_entities": "parses but declares no entity (an import-only shell)",
            "too_many_entities": "above the embedding cost bound",
            "asprino_too_many_entities":
                "above the conceptual-components cost bound",
            "no_ontology_file": "no RDF file found in the directory",
        }
        for k, v in reasons.most_common():
            lines.append(f"| `{k}` — {explain.get(k, '')} | {v} |")
        lines.append("")
    (out / "corpus_summary.md").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")


def write_audit(out: Path, skips: list[dict]) -> None:
    """audit/<reason>.csv -- the analogue of release-iri/audit/."""
    a = out / "audit"
    a.mkdir(exist_ok=True)
    by_reason: dict[str, list] = defaultdict(list)
    for s in skips:
        by_reason[s.get("reason", "unknown")].append(s)
    summary = []
    for reason, rs in sorted(by_reason.items()):
        fn = f"excluded_{reason}.csv"
        keys = list(rs[0])
        with (a / fn).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rs)
        summary.append({"reason": reason, "records": len(rs), "file": fn})
    if summary:
        with (a / "audit_summary.csv").open("w", newline="",
                                            encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["reason", "records", "file"])
            w.writeheader()
            w.writerows(summary)
