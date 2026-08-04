# Every filter in the pipeline

Complete inventory. Two kinds exist and they must not be confused:

- **Dataset filters** decide what is *in the high-quality dataset*. They are the
  methodological contribution and are reported as findings.
- **Stage filters** are practical limits on one computation. They never remove an
  ontology from the dataset; they only mean "this step could not process it", and
  every exclusion is recorded with a reason.

**There is no size filter on the dataset.** A 1 GB ontology is a legitimate
corpus member. Size constrains only the embedding stage. Verified: 0 ontologies
are excluded for size.

---

## A. Dataset filters — the quality gate

Implemented in [`odpqual/gate.py`](odpqual/gate.py). Applied in order; each
ontology is attributed to the **first** rule it fails, so the counts below
partition the discovered total exactly.

Corpus: **1,973 discovered → 501 kept (25.4%)**.

All counts below are the **post-repair** gate — that is, the gate applied after
the 1,017 zero-byte pattern files of §B2 were refilled. Applying the same rules
to the un-repaired corpus keeps only 341, and the difference is itself a result;
see §B2 for the transition table.

| # | Filter | Test | Threshold | Dropped | Why it exists |
|---|---|---|---|---:|---|
| **G1** | Parseable | the ontology file parses as RDF under any format | — | **269** (13.6%) | an unparseable file cannot be analysed at all |
| **G2** | Not a pattern copy | (a) `owl:Ontology` IRI not in a pattern-library namespace; (b) not byte-identical to one of its patterns; (c) entity set ≠ a pattern's entity set | — | **36** (1.8%) | **the decisive rule** — a 100%-exact-IRI criterion is tautological on a file that *is* the pattern |
| **G3** | Not a duplicate | among identical SHA-256 content hashes keep the lexicographically first id | — | **203** (10.3%) | forks and mirrors inflate every count and leak across train/test splits |
| **G4** | Substantive | entities contributed beyond its patterns ≥ threshold | **≥1** (`--min-own-entities`, the released value) | **540** (27.4%) | an ontology that adds nothing to its patterns is an aggregation of them, not a reusing ontology; all 540 contribute exactly zero |
| **G5** | Has ground truth | ≥1 pattern with non-empty entity intersection | ≥1 entity | **424** (21.5%) | no overlap means no reuse signal and nothing to score against |
| **G6** | Complete reuse | ≥1 pattern reused in full, evaluated **after** G1–G5 | 100% coverage | — (view membership, not a drop) | the original criterion, now unsatisfiable by a pattern copy |

Per portion — GitHub 1,887 → 442 kept · noOBO 78 → 51 · OBO 8 → 8. Complete
reuse (G6): 37 · 6 · 8 = **51**. Source: `release-iri/gate_summary.csv`.

### Sub-reasons

All recomputed from `MANIFEST.csv` `drop_reason`, post-repair.

**G1** — exact exception types: `ParserError` 170, `SAXParseException` 32,
`BadSyntax` 31, `NoOntologyFile` 25, `TypeError` 10, `ValueError` 1. The verbatim
parser message is in `error_detail` for 244 of the 269; the remaining 25 are
`NoOntologyFile`, which has no message.

**G2** — `identical_entities` 26, `identical_file` 10; **all 36 in the GitHub
portion**, none in noOBO or OBO. Note the strict test is the
`owl:Ontology` declaration, not any `rdf:about`. Testing "any rdf:about in a
pattern namespace" over-counts, because it also flags ontologies that legitimately
*reference* pattern classes — which is the reuse being studied.

**G3** — all 203 in the GitHub portion. `duplicate_of` names the retained file.

**G4** — of the 729 dropped: **540 have zero** own entities, 84 have 1–4, 105 have
5–9. The threshold of 10 is **arbitrary** and the distribution has no gap to read
it off; §D2 prices the choice and shows that no finding depends on it. It also
shows the rule is expressed in the wrong units — a share would separate a small
ontology from a near-copy of a pattern library, and an absolute count cannot.

**G5** — all 307 are `no_pattern_overlap`. Read this rule with §B2 in mind: before
the repair it fired 459 times, and 152 of those were a fetch failure rather than a
modelling fact.

### Two normalisations applied before any filter

These change *what an entity is*, so they affect every rule and every count:

1. **Annotation stripping** — SKOS, DC, DCTERMS, DCAM, CPANN, VANN, FOAF, CC, PAV,
   PROV, schema.org, VoID triples, all literal-valued triples, and `owl:Ontology`
   header triples are removed from **both** the ontology and the pattern.
   PatternMiner stripped them on the pattern side only, which biased every
   host-vs-pattern comparison. Disable with `--keep-annotations`.
2. **WIDE entity sets** — declared classes and properties, plus IRIs occurring only
   in a class/property/individual position (subClassOf objects, restriction
   fillers, domain/range, equivalence, intersection/union members, named
   individuals); `owl:Thing`/`owl:Nothing` removed. Byte-compatible with
   PatternMiner's `_collect_entities` under `ODP_WIDE_ENTITIES=1`.

---

## B. Stage filters — practical limits, never dataset exclusions

### Embedding (`use-cases/odp-extraction/tools/run_extraction_eval.py`)

| Filter | Default | Why | Recorded as |
|---|---|---|---|
| `--max-entities` | **5000** | OWL2Vec\*/RDF2Vec build 100 walks per entity, so runtime and peak memory scale linearly. `upheno` (193,638 entities) implies ~19M walks. 5,000 admits every GitHub and Paper/Repo ontology and excludes only OBO-scale ones. | `too_many_entities` |
| `--max-mb` | **0** (no limit) | — | `file_too_large` |
| memory | — | the embedder exhausts RAM on this machine | `embed_out_of_memory` |
| parse | — | with the **verbatim exception type and message** | `parse_error` |
| ground truth | — | no pattern overlap under the requested subset | `no_ground_truth` |

All written to `rq5_skipped.csv`. **The ontology remains in the dataset.**

### Statistics (`run_rq_stats.py`)

`--population all` (default) reports **the real corpus** — every ontology that
parses, gate-passing or not, with `gate_passed` and `drop_reason` on every row.
`--population gated` restricts to those passing the gate. Reporting the real data
matters: "37% of the corpus is stubs" is a finding about ODP reuse in the wild and
it vanishes if the stubs are filtered before the statistics are computed.

### Publication (`build_release.py`)

**No filter.** The corpus is published whole under `corpus/`, gate-passing or not,
so a reader can check whether an exclusion was justified and re-gate with
different thresholds. The gate ships as metadata in `MANIFEST.csv`, and the six
views are filters *over* the published corpus rather than separate copies.

---

## B2. A harvesting defect that masqueraded as a quality filter — and its repair

**1,017 of 4,410 pattern files (23.1%) were ZERO BYTES.** 608 GitHub ontologies
had *every* one of their pattern files empty; 230 of those were counted as G5
("no ground truth") drops.

This is not a property of the ontologies — it is a defect in
`build_github_dataset.py`, which created the pattern file before the fetch and
did not delete it when the fetch returned nothing. Those ontologies may well
reuse the pattern; we simply failed to download the pattern to compare against.

Why it matters: reuse is established by intersecting the ontology's entity set
with the pattern's. A zero-byte pattern has an empty entity set, so the
intersection is necessarily empty, so G5 fires. **G5 behaved exactly as
specified. What it recorded was a fact about our network, not about the
ontology.**

### The repair

`repair_patterns.py` refills each empty file from whichever of the four pattern
libraries already on disk supplies a file with the same stem. It never touches a
non-empty file and it refuses to guess: where a stem resolves to several library
files with *differing* content, three ordered rules disambiguate and the choice
is logged with its alternatives.

**All 1,017 were refilled; 0 remain.** Rules used:

| Rule | n | |
|---|---:|---|
| `R0_single_candidate` | 639 | one distinct content among the candidates — no choice to make |
| `R3_largest_file` | 308 | several distinct versions; take the largest |
| `R1_library_precedence` | 64 | resolved by library order |
| `R2_own_directory` | 6 | a copy sits in the pattern's own directory |
| refused for want of a library match | **0** | |

R3 carries most of the ambiguous cases because of one recurring file:
`DUL.owl` appears in 26 places across the libraries in 3 distinct versions.

### Effect on the gate

Same 1,973 ontologies, same six rules; only the pattern files differ.

| Verdict | Before | After | Δ |
|---|---:|---:|---:|
| G1 unparseable | 269 | 269 | ±0 |
| G2 pattern copy | 13 | 36 | **+23** |
| G3 duplicate | 209 | 203 | −6 |
| G4 not substantive | 735 | 729 | −6 |
| G5 no ground truth | 459 | 307 | **−152** |
| **Kept** | **341** | **501** | **+160 (+46.9%)** |
| complete reuse, GitHub | 5 | 33 | +28 |

Transitions: G5→kept 141, G4→G2 15, G5→G4 9, G3→G2 6, G5→G2 2.

The second effect is the less welcome one. With an empty pattern file you cannot
show that an ontology has the *same entity set* as its pattern, so 23 verbatim
pattern copies were being classified as stubs or duplicates instead. **The defect
had been concealing exactly the cases that motivate G2.**

### A provenance defect, declared and then repaired

The repair ran in two passes and the second overwrote the first pass's log, so
`out/repair2/pattern_repair_log.csv` records only 378 of the 1,017 files. The
missing 639 were recovered from file mtimes — the base tree carries
`2026-07-29 10:xx/11:xx`, pass 1 `2026-07-30 13:xx`, pass 2 `2026-07-30 15:xx`,
and 3,393 + 639 + 378 = 4,410 — and each recovered row was re-derived through the
same library index and `disambiguate()`, then **verified byte-identical to the
library source it names**. All 639 verified; none is a guess.

Published as:

- `out/repair/pattern_repair_log_recovered.csv` — the 639 recovered rows, with mtime
- `out/repair/pattern_repair_log_complete.csv` — all **1,017**, the citable record
- `out/repair2/pattern_repair_log.csv` — pass 2 as originally written, unmodified

A re-run must write to a fresh directory, or append. Overwriting a provenance log
is the same class of error as the defect it was documenting.

### What the repair still cannot fix

- **A pattern that was never fetched at all**, leaving no file, empty or
  otherwise, is invisible to this procedure and would show up as a G5 removal.
  The 307 G5 removals are therefore an **upper bound** on genuine non-reuse.
- **Label mode cannot rescue an unfetched pattern either.** Matching by label
  needs the pattern's classes; an empty file has none.
- Any prior result computed over the affected ontologies is suspect: PatternMiner
  evaluated 184 of them and reported ground truth for some, which cannot have
  come from the empty file.

Remaining action: add an explicit `empty_pattern_file` diagnostic to the manifest
so a future recurrence is visible directly rather than inferred from G5.

---

## C. Manifest completeness

`error_detail` and `n_pattern_errors` were added after the GitHub and Paper/Repo
portions had first been analysed. Following the cache rebuild they are now
populated: **244 of the 269 G1 drops carry the verbatim parser message**, and the
remaining 25 are `NoOntologyFile`, for which there is no message to record.
`duplicate_of` is populated for all 203 G3 drops and `self_match_reason` for all
36 G2 drops.

---

## D. Tuning the gate

| Knob | Effect |
|---|---|
| `--min-own-entities N` | G4 threshold. At 10, 729 drop. Lowering it to 1 readmits **72** ontologies (not 189 — the other 117 fail G5 anyway). See §D2: the corpus is sensitive to this knob, the findings are not. |
| `--keep-annotations` | reproduces the old asymmetric behaviour |
| `--max-mb N` | reintroduces a dataset size limit (**not recommended**) |
| `--datasets` | restrict which portions are gated |

Any change to these changes the population, so record which `MANIFEST.csv` a
result came from.

---

## D2. The G4 threshold, priced

**Ten is arbitrary.** It is not derived from anything, and it cannot be read off
the data: the distribution of own-entity counts is a spike at zero (540
ontologies) followed by a flat plateau of 10–30 ontologies at every value from 1
to about 24. There is no valley at 10 or anywhere else.

What can be done is to measure the cost. Because the rules are applied in order,
moving the G4 threshold only moves ontologies between G4 and {G5, kept}, so the
curve is computable from `MANIFEST.csv` without re-running the gate:

```
kept(T) = survives G1–G3  AND  n_own_entities >= T  AND  n_gt_patterns >= 1
```

`pipeline/g4_sensitivity.py` asserts that `kept(10)` reproduces the shipped
release (501) before reporting anything.

| T | kept | vs 10 | pairs | GitHub mean patt./onto | GitHub complete | GitHub distinct patterns |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 509 | +80 | 1,530 | 1.51 | 8.2% | 45 |
| **1** | **501** | — | **1,517** | **1.51** | **8.4%** | **45** |
| 2 | 491 | +62 | 1,504 | 1.52 | 8.5% | 45 |
| 5 | 464 | +35 | 1,468 | 1.54 | 8.5% | 45 |
| 10 | 429 | −72 | 1,421 | 1.56 | 8.6% | 45 |
| 15 | 392 | −37 | 1,371 | 1.59 | 8.9% | 44 |
| 20 | 360 | −69 | 1,326 | 1.62 | 9.7% | 44 |
| 50 | 251 | −178 | 1,114 | 1.49 | 10.7% | 35 |
| 100 | 176 | −253 | 990 | 1.41 | 12.2% | 28 |

**The corpus is sensitive to the threshold; the findings are not.** Across
T = 0…20 the corpus ranges from 509 to 360 ontologies while every GitHub headline
stays inside its reporting precision, and the number of distinct patterns reused
is 45 at every one of those thresholds.

### Why the threshold is 1

It readmits **72** ontologies — not the 189 in the 1–9 band, because 117 of those
share no entity with any pattern and G5 drops them regardless. Of the 72:

- **55 contribute at least half of their own file** — small ontologies, not artefacts
- 10 contribute 25–50%
- **7 contribute under 25%**

Full list in `out/g4_band_1_to_9.csv`, sorted by own share.

### The rule is expressed in the wrong units

An absolute count conflates two different things:

| | own | total (wide) | own share | what it is |
|---|---:|---:|---:|---|
| `hobbit-project_sml-benchmark-v2_DUL.owl` | 1 | 188 | 0.5% | DUL with one class added |
| `Javierbj02_INSIGHT-UCLM_DUL.owl` | 1 | 188 | 0.5% | same |
| `dersuchendee_OE-Assist_FestSCQ12` | 8 | 9 | 89% | a real, tiny ontology |
| `marco-scarnato_Ontology-Project_list` | 8 | 10 | 80% | a real, tiny ontology |

The first two pass any absolute threshold of 1 and should not; the last two fail
at any absolute value and arguably should not. Median own share is **92%** for the 501 kept and
**64%** for the excluded band, so the relative measure separates them where the
absolute one cannot.

**A future release should express G4 as a share.** The manifest already carries
`n_own_entities` and `n_entities_wide`, so the alternative can be evaluated
without re-harvesting anything.

```bash
python pipeline/g4_sensitivity.py --manifest release-iri/MANIFEST.csv \
    --pairs pipeline/out/rq_iri_all/pairs.csv --out pipeline/out
```
