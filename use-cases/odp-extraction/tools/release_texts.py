"""Text templates for the MSE release: the README and the Zenodo description.

Kept out of build_mse_release.py so that the prose can be edited without
touching the assembly logic, and so that a reviewer can read what will be
published without reading a build script.

Both are ``str.format`` templates. Literal braces in the README's JSON examples
are therefore doubled.
"""

README_TEMPLATE = """# MSE-ODP {version} — candidate ontology design patterns from materials science ontologies

Released {date}. Version {version}. `CHANGELOG.md` records what changed;
`ZENODO.md` is the deposition description.

---

## What this is

Automatically extracted **candidate ontology design patterns** from a corpus of
**{n_files} materials science and engineering (MSE) ontologies**, together with
the corpus exactly as evaluated, every method's raw output, and the statistics
used to rank candidates for expert review.

The corpus is the collection assembled for

> Ebrahim Norouzi, Jörg Waitelonis, Harald Sack.
> *The landscape of ontologies in materials science and engineering: a survey
> and evaluation.* arXiv:2408.06034, 2024. <https://arxiv.org/abs/2408.06034>

with every artefact re-checked against its upstream and **{n_refreshed} of them
refreshed to a newer release** for this version.

## What this is not

**There is no ground truth here.** No ontology in this corpus is annotated with
the design patterns it reuses. Consequently:

- **no accuracy, precision, recall or F-score appears anywhere in this release**,
  and none should be computed from it as though the groupings were labels;
- the contents of `statistics/` are *ranking signals*, not quality scores;
- `validate_release.py` **fails** if any prediction file contains a score, which
  is how that promise is enforced rather than merely stated.

For extraction methods scored against verified ground truth, use the companion
**ODP-Reuse** release, which has an answer key and a documented quality gate.

## Files

| Path | What it holds | Format |
|---|---|---|
| `corpus/` | the {n_files} ontology files, byte-identical to what was evaluated | OWL / Turtle / RDF-XML |
| `MANIFEST.csv` | one row per file: ontology IRI, title, version, triple count, imports, SHA-256, whether it parsed, whether it was refreshed and from where | CSV |
| `dataset_statistics.csv` | **one row per method**: yield, group sizes, coverage, anchoring, known-ODP contact | CSV |
| `corpus_summary.md` / `.csv` | how the corpus was assembled and reduced: composition, the upstream currency audit, and every exclusion | Markdown / CSV |
| `SIGNALS.md` | what each ground-truth-free signal means, why it is computed, **and how it fails** | Markdown |
| `views/<method>/groups.csv` | every group that method proposed, flattened — usable without a JSON parser | CSV |
| `views/review_shortlist.csv` | the ranked candidate set for expert review | CSV |
| `predictions/*.jsonl` | every predicted grouping, one file per method run | JSON Lines |
| `statistics/mse_stats.json` | yield, cross-method agreement, recurrence, foundational anchoring, known-ODP contact, review shortlist | JSON |
| `statistics/yield.csv`, `agreement.csv`, `agreement_matrix.csv`, `recurrence_by_*.csv` | the same, flattened | CSV |
| `audit/excluded_<reason>.csv` | every exclusion, one file per reason, with `audit_summary.csv` | CSV |
| `provenance/corpus_inventory.csv` | version metadata read out of every file | CSV |
| `provenance/corpus_updates.csv` | every refresh: old version to new, entity counts before and after, source URL | CSV |
| `provenance/upstream_check.json` | the **complete** currency audit, including artefacts where nothing newer was found and proposals that were rejected | JSON |
| `skipped.csv` | every ontology excluded from a run, with the reason | CSV |
| `validate_release.py` | internal-consistency check for an unpacked copy | Python 3 |

{n_parsed} of {n_files} files parse as RDF. The rest are kept, with their parse
error recorded in `MANIFEST.csv`, because an MSE practitioner looking for
ontologies today would find them in the same state.

## How to use it

### 1. Check the copy you downloaded

```bash
python validate_release.py
```

Verifies every SHA-256 in `MANIFEST.csv`, that the manifest and `corpus/` agree,
and that no prediction file carries a score. Exits non-zero on any failure.

### 2. Read the predicted groupings

Each line of a `predictions/*.jsonl` file is one `(ontology, method)` pair:

```jsonc
{{
  "corpus": "mseo",
  "ontology": "PMDCO",        // matches MANIFEST.csv, minus the extension
  "method": "asprino_cc",     // which method proposed these groupings
  "n_entities": 1530,         // entities in the host ontology
  "n_clusters": 47,
  "scored": false,            // always false: there is no ground truth
  "clusters": [["https://w3id.org/pmd/co/...", "..."], ["..."]]
}}
```

```python
import json, collections
rows = [json.loads(l) for l in
        open("predictions/structural_and_published.jsonl", encoding="utf-8")]
print(collections.Counter(r["method"] for r in rows))

# groupings of at least four entities from one method
big = [c for r in rows if r["method"] == "asprino_cc"
       for c in r["clusters"] if len(c) >= 4]
```

### 3. Read a method's groups without touching JSON

```bash
head -3 views/owl2vec_kmeans/groups.csv
# ontology,group_id,size,entities
```

`dataset_statistics.csv` gives one row per method, and `corpus_summary.md`
explains how the population was arrived at.

### 4. Find the candidates worth looking at first

`statistics/mse_stats.json` holds, under `F_review_shortlist`, the candidates
that recur across ontologies or are proposed by more than one method, ranked.
**Read the caveat below before using that ranking.**

```python
import json
stats = json.load(open("statistics/mse_stats.json", encoding="utf-8"))
for c in stats["F_review_shortlist"]["top"][:20]:
    print(c["n_ontologies"], c["n_methods"], c["local_names"][:6])
```

Other keys: `A_yield` (what each method proposes), `B_cross_method_agreement`,
`C_recurrence_by_iri` and `C_recurrence_by_localname`,
`D_foundational_anchoring`, `E_known_odp_contact`.

### 5. Trace an ontology's provenance

```python
import csv
man = {{r["file"]: r for r in csv.DictReader(open("MANIFEST.csv", encoding="utf-8"))}}
r = man["PMDCO.ttl"]
print(r["version"], r["refreshed_from_upstream"], r["previous_version"],
      r["upstream_repo"])
```

`provenance/upstream_check.json` gives the full audit for every file, including
the ones nothing was done to.

## Signals, and their failure modes

`SIGNALS.md` states each of the five signals, why it is computed and what it does
**not** license. Read it before ranking anything. The most important of them is
repeated here because it is the one most likely to mislead.

## One caveat that changes how the ranking should be read

Ranking candidates purely by how many ontologies they recur in puts **pairs of
BFO upper-level classes at the top**, because a dozen MSE ontologies import BFO.
That is shared *scaffolding*, not shared design.

Any use of the recurrence signal should therefore exclude candidates composed
entirely of foundational terms (BFO, EMMO, IAO, RO, PMDco, CCO, QUDT) and
require at least one term from the host ontology's own namespace. The released
statistics report the raw signal; applying that correction is the consumer's
responsibility, and the accompanying paper explains why.

## Methods represented

| Method | Origin |
|---|---|
| Conceptual components | Asprino, Carriero & Presutti, K-CAP 2021 — run from the authors' own code |
| Axiom-tree mining (axiom patterns; class frames) | Ławrynowicz, Potoniec, Robaczyk & Tudorache, Semantic Web Journal 2018 — reimplemented, the published source URL being dead |
| OWL2Vec* / RDF2Vec with K-Means or HDBSCAN | embedding baselines |
| Connected components; IRI namespace grouping; all-in-one; singletons | reference points that bound the task |

The reference points are **not** extraction methods. They are included because
no score is interpretable without knowing what a prediction that knows nothing
about patterns already achieves.

## Licence

The **predictions, statistics, provenance and documentation** are CC-BY-4.0.

The **ontology files in `corpus/`** remain under their own licences and are
redistributed only so that these results are reproducible against the exact
inputs used. `MANIFEST.csv` names the upstream repository of each; consult that
upstream before reusing an ontology.

## Citing

Please cite this deposition together with the survey the corpus derives from
(arXiv:2408.06034). If you use the extraction results, cite the originating
method papers listed above as well.

## Source

<https://github.com/ISE-FIZKarlsruhe/ODP-Reuse> — under
`use-cases/odp-extraction/`: the runner (`run_extraction.sh`), the currency
audit (`tools/corpus_inventory.py`, `tools/update_corpus.py`), the statistics
(`tools/mse_stats.py`), the error analysis (`tools/error_analysis.py`) and this
release builder (`tools/build_mse_release.py`).
"""


ZENODO_TEMPLATE = """MSE-ODP {version}: candidate ontology design patterns extracted from {n_files} materials science and engineering ontologies.

WHAT THIS DEPOSITION CONTAINS

Ontology Design Patterns (ODPs) are reusable modelling solutions, and several published methods extract candidate patterns from existing ontologies automatically. Those methods had not been applied to materials science and engineering, a domain where ontology development is active, foundational commitments are contested, and no catalogue of domain patterns exists. This deposition holds the result of applying them.

  corpus/       {n_files} MSE ontologies ({n_parsed} of which parse as RDF), exactly as evaluated
  predictions/  every grouping of entities each method proposed, as JSON Lines
  statistics/   ground-truth-free measures used to rank those groupings for review
  provenance/   the version audit of every ontology, including what was refreshed and what was not
  MANIFEST.csv  one row per file, with checksum, declared version and upstream repository
  validate_release.py  a consistency check for an unpacked copy

THERE IS NO GROUND TRUTH IN THIS DATASET

No ontology here is annotated with the patterns it reuses. No accuracy, precision, recall or F-score is reported, and none should be inferred. What is reported is what each method proposed, how much independent methods agree with one another, how often a grouping recurs across ontologies, whether it is anchored in a foundational vocabulary, and whether it overlaps a documented pattern. These rank candidates for expert review; they are not evidence that a candidate is good. The included validator fails the release if any prediction file carries a score.

For extraction methods scored against verified ground truth, see the companion ODP-Reuse deposition.

THE CORPUS AND ITS CURRENCY

The corpus is the collection assembled for "The landscape of ontologies in materials science and engineering: a survey and evaluation" (Norouzi, Waitelonis and Sack, arXiv:2408.06034). Every artefact was re-checked against its upstream repository before evaluation and {n_refreshed} were refreshed to a newer release; several had been held at pre-release versions two major versions behind their current one. provenance/upstream_check.json records the verdict for every file, including those where nothing newer was found and those where a proposed update was rejected as the wrong artefact, so that the corpus's currency can be re-checked rather than taken on trust.

A CAVEAT WORTH READING BEFORE USING THE RANKING

Ranking candidates by how widely they recur across ontologies puts pairs of Basic Formal Ontology upper-level classes at the top, because a dozen MSE ontologies import BFO. Recurrence in this corpus largely measures shared foundational scaffolding rather than shared design. Consumers of the recurrence signal should exclude candidates composed entirely of foundational terms and require at least one term from the host ontology's own namespace.

LICENCE

The predictions, statistics, provenance and documentation are released under CC-BY-4.0. The ontology files in corpus/ remain under their own licences and are redistributed only so that these results are reproducible against the exact inputs used; MANIFEST.csv names the upstream of each.

CITING

Please cite this deposition together with arXiv:2408.06034, and cite the originating method papers if you use the extraction results.

SOURCE CODE

https://github.com/ISE-FIZKarlsruhe/ODP-Reuse
"""
