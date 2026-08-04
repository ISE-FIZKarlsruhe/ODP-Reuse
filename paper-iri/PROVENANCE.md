# Provenance — `paper-iri/`

**The provenance index is now executable.** Run it:

```bash
python paper-iri/verify_numbers.py
```

It asserts 100 headline numbers across *both* manuscripts against the released
data and exits non-zero on the first mismatch, naming the paper claim, the
expected value and the value the data gives. If a pipeline re-run changes a
number, this fails and tells you which sentence to edit. Appendix B of the
paper is the human-readable twin of the same index.

Everything below is what the script cannot check: the build state, the
decisions taken, and the items still open.

---

## 1. Build state — all Track A artefacts are at the released threshold `T = 1`

| Artefact | State |
|---|---|
| `pipeline/out/MANIFEST.csv` | T=1, 1,973 rows, 501 kept / 51 complete |
| `pipeline/out/gate_summary.{csv,md}` | T=1 |
| `pipeline/out/audit/` | T=1, **1,472 removals re-derived, 0 disputed** (G1–G5 all complete) |
| `pipeline/out/rq_iri_gated/`, `rq_iri_all/` | T=1 |
| `pipeline/out/g4_threshold_sensitivity.csv` | T=1-anchored |
| `pipeline/out/comparison/` | re-run 2026-08-04 against the T=1 manifest |
| `release-iri/` | T=1, manifest + views + statistics + audit all consistent |
| `use-cases/odp-extraction/results/iri/` | T=1, n=442 / 51 |
| `use-cases/odp-recommendation/results_reco/iri_github/` | **new**, n=235 |
| `use-cases/odp-recommendation/stats_iri_github/` | **new**, paired tests |

## 2. Numbers that changed from the ISWC/ESWC submissions, and why

| Quantity | Was | Now | Why |
|---|---|---|---|
| Corpus | 1,744 ontologies | 1,973 discovered, **501 kept** | the gate; 1,744 counted trivial and duplicate files |
| Complete reuse | 49 ontologies | **51** | 14 of the 49 were pattern copies (G2); the repair then added more genuine cases |
| Pairs | 4,064 | **1,517** | same cause as the corpus figure |
| Kept share | — | **25.4%** | new (RQ4) |
| Extraction F₁ | 0.642 (OWL2Vec\*/K-Means) | oracle **0.468**, unsup. **0.321** | 0.642 was oracle-selected on a pattern-copy-inflated subset |
| Best embedding | OWL2Vec\* | regime-dependent | RDF2Vec wins under oracle, OWL2Vec\* under unsupervised |
| Reuse forms, direct | 38–63% | **31.8%** (GH) | pattern copies + one-sided annotation stripping inflated it |
| Pattern DL | "mostly EL" | **modally ALCHI** | EL was a DOS-DP property generalised to the whole corpus |
| Recommendation | not reported | **CF 0.815 MAP, +0.785 vs floor, n=235** | new (UC2) |

## 3. Structural decisions, and why

**RQs and UCs are separated.** R1 objected that only one of the four "use
cases" demonstrated reusability; R3 objected that a use case does not analyse
anything. So: **RQ1** extent, **RQ2** forms, **RQ3** complexity, **RQ4**
corpus quality *(new)*; **UC1** extraction, **UC2** recommendation *(new)*.

**ISWC wording and framing retained.** The manuscript keeps the ISWC
submission's voice, its four downstream applications, its discovery
narrative and its pattern-library table; the numbers are replaced and RQ4 /
UC2 added.

**Everything is in the paper.** R2 w2 objected that UC2 and UC3 were
offloaded to external reports. All six sections are now self-contained.

**Consistent portion names.** R1 objection 4 (nomenclature drift): portion
names go through `\portGH{}`, `\portPR{}`, `\portDD{}` macros.

**No pooled figures anywhere.** DOS-DP is reported but never averaged with
the other two portions. This is what produced the false "mostly EL" claim.

### Reviewer requests discharged

| Request | Where |
|---|---|
| Metareview: are GitHub ontologies a sound basis? | RQ4 — measured per portion |
| Metareview: long-term adoption | §Limitations — mechanisms stated, uptake explicitly *not* claimed |
| R1: quality analysis of the acquired ontologies | RQ4 |
| R1 / R4: define the reuse threshold | RQ1 (curve) + RQ4 (G4 sensitivity) |
| R1 / R5: present the recommendation use case | **UC2, in full, n=235, paired tests** |
| R1: repeated paragraph p5–6 | methodology rewritten |
| R2 q1: what was *k*? | UC1 — worse than pinned to \|GT\|; the whole grid was oracle-selected |
| R2 q2: deduplication, forks, mirrors | G3 — 203 removals |
| R2 q3: inter-annotator agreement | §Methodology — stated as single-annotator, with the bound on its consequence |
| R2 w1: separate DOS-DP | throughout; no pooled figure anywhere |
| R2 w2: use cases offloaded | all six sections self-contained |
| R2 w3: "gold standard" language too strong | removed everywhere; RQ4 explains why it was wrong |
| R2: related work thin on reuse studies | §2.2 expanded (Eberhart, Marinelli, Blomqvist, Alghamdi, Hammar, Vrandečić) |
| R2: no simple baselines to ground F₁ | UC1 unsup. column + UC2 random/popularity floors |
| R2: CITATION.cff TODO placeholders | fixed |
| R3: how is the data made available? | §Methodology, "The release itself has four parts" |
| R3: "to the best of our knowledge" is dangerous | §2.6 states the position from the documented search |
| R3: model + hyperparameters for the classifier | §Methodology, discovery channel (ii) |
| R3: breakdown for GitHub as well as Paper/Repo | RQ4 Table 1 gives all three portions per rule |
| R3: 25 footnotes | reduced |
| R4: add Eberhart et al., Marinelli et al. | §2.2, and Eberhart used substantively in RQ3 |
| R4: is one element enough to count as reuse? | RQ1 — the curve replaces the one-entity bar |
| R5: the "why", not just the "what" | RQ2/RQ3 offer readings, labelled as interpretation; §Limitations states what a corpus study cannot reach |

## 4. Open before submission

1. **Conflict of interest — the only blocking item.** Harald Sack is a
   co-author and an SMSE guest editor; Jörg Waitelonis is on its guest
   editorial board. SWJ FAQ A8 answers "strictly speaking: no". `main.tex`
   carries a red placeholder in the declaration section. Settle with the
   editors — either request handling by a regular editor outside the special
   issue, or submit as a regular SWJ paper.
2. **Get `sagej.cls`** from SAGE (`sage_latex_template_4.zip`). `main.tex`
   falls back to `article` so it compiles without it. Do **not** use
   `TRR.cls` — wrong SAGE class, downloaded by mistake.
3. **The document has not been compiled.** No LaTeX toolchain was available
   in the session that wrote it. `python paper-iri/verify_numbers.py` and the
   structural checker both pass, so references, citations, inputs and
   environments resolve; expect the usual first-compile spacing fixes.
4. **Two figures from the ISWC submission are not yet carried over:** the
   pipeline overview (`figures/gold_dataset_pipeline.jpg`) and the extraction
   baseline diagram. Both are referenced in the ISWC source and would help
   §Methodology and UC1.
5. Optional: raise the \portPR{} recommendation sample above n=14 by running
   `recobench` on a merged portion, or drop that block and report GitHub only.

## 5. Pipeline follow-ups (not blocking)

- Add an explicit `empty_pattern_file` diagnostic to the manifest, so a
  recurrence of the zero-byte defect is visible directly rather than inferred
  from G5.
- Re-express G4 as a *share* of an ontology's own entities rather than an
  absolute count. The manifest already carries both numbers. §Limitations
  states this as a known weakness.
- `make_stats.py` crashes on a relative `--out-dir` (`path.relative_to(HERE)`
  in `write_csv`). Pass an absolute path.
- `compare_to_patternminer.py` writes all four outputs and *then* crashes
  printing Markdown to a cp1252 console (`UnicodeEncodeError` on `Δ`). Set
  `PYTHONIOENCODING=utf-8`.
- `compare_tracks.py` leaves the UC1/UC2 columns of `COMPARISON.md` empty
  when only one track's results are present in the expected layout; the
  Track B deltas in `paper-iri-label/tables/delta_uc.tex` were assembled from
  the per-track result files directly.

## What a clone does and does not contain

This file and `verify_numbers.py` are published; the LaTeX sources of the
manuscript are not, while it is under review. So a clone can re-verify every
number without containing the paper that reports them:

```bash
python paper-iri/verify_numbers.py       # asserts the headline numbers
python pipeline/check_repo_claims.py     # asserts every cited path is present
```

The companion normalised-label track is **not part of this release**:
`paper-iri-label/` and `release-iri-label/` are absent from a clone, and the two
corpora differ by only 11 ontologies, so publishing both would offer two
near-identical datasets with no basis for choosing between them.
`verify_numbers.py` detects their absence, skips the Track B assertions and
reports how many it ran: **108 on a clone, 140 in the authors' working tree**.
The 32-assertion difference is exactly the iri+label apparatus — nine source
files, none of them Track A. Verified by copying only the files git would deliver
into an empty directory and running the checker there.
