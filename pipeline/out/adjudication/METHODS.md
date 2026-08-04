# Label-match adjudication — methods appendix

Everything a reader needs to reproduce or challenge the adjudication: the exact
prompt, the models and settings, the agreement statistics and their formulas, and
an honest account of missing data. **Intended to be reproduced verbatim in the
paper's appendix.**

---

## 1. What is being adjudicated

The `label` ground-truth mode matches a pattern class to a host ontology entity
when their *normalised labels* are equal. It recovers reuse where an ontology
re-modelled a pattern under its own namespace — invisible to IRI matching — but
it also admits homonyms: a class labelled `Event` in a concert ontology is not
thereby reusing DUL's `Event`.

**Item set.** Every (ontology, pattern) pair with at least one entity matched by
label but **not** by IRI. Restricted to the GitHub and Paper/Repository portions;
DOS-DP is excluded because its patterns are generated from YAML templates and its
ontologies carry ~10⁵ entities, so label equality there measures vocabulary
overlap rather than a design decision. Ontologies whose file exceeds 40 MB are
excluded because an item must be judgeable from a prompt.

**Result: 331 items.** Skips recorded in `build_skipped.csv`.

## 2. Normalisation (no threshold anywhere)

```
norm(s) = lowercase(remove_non_alphanumeric(split_camelCase(s)))
```

`AgentRole`, `Agent Role`, `agent_role`, `agent-role` → `agentrole`. Matching is
then string **equality**.

No similarity measure is used. Jaccard, edit distance and embedding cosine each
require a cut-off that cannot be justified from first principles, and adopting a
tuned threshold would contradict the reproducibility argument this dataset rests
on. Exact-normalised equality has no free parameter.

Two precision guards, both parameter-free:

1. **Generic-label blocklist** — `thing, entity, nothing, class, concept, object,
   resource` are never matchable.
2. **Structural coherence** — an ODP is a connected fragment, so among ≥2
   label-matched host entities only the largest connected component is kept
   (edges: direct IRI→IRI triples plus one hop through a restriction/blank node).

## 3. Machine annotators

| Model | Version | Settings |
|---|---|---|
| GPT-4o | `gpt-4o` | `temperature = 0` |
| Gemini | `gemini-2.5-flash` | `temperature = 0` |
| Claude | `claude-sonnet-5` | default (this model rejects `temperature`) |
| DeepSeek | `deepseek-chat` | `temperature = 0`, OpenAI-compatible endpoint |

Each model sees the identical prompt, independently, with no knowledge of the
others' verdicts. Order of presentation is fixed by item id.

### System prompt

```
You are an expert ontology engineer adjudicating whether an ontology reuses a
specific Ontology Design Pattern (ODP). Answer strictly with JSON.
```

### User prompt (verbatim template)

```
An automated matcher proposed that this ontology reuses an ODP, because
some of the ontology's entities carry the SAME NORMALISED LABEL as classes in the
pattern. The IRIs do NOT match -- the ontology would have re-modelled the pattern
under its own namespace.

Your task: decide whether this is genuine reuse of the pattern's modelling
solution, or a coincidental name collision.

PATTERN: {pattern}
Pattern classes: {pattern_classes}

ONTOLOGY: {ontology}
Ontology subject matter, inferred from its entity names: {ontology_sample}

MATCHED LABELS (present in both): {matched}
Matched entities in the ontology: {matched_iris}

Guidance:
- Genuine reuse: the ontology models the same conceptual structure the pattern
  describes, and the matched classes play the same role. Several structurally
  connected matches are strong evidence.
- Coincidence: the label is a common word (Event, Object, Role, Place) used in an
  unrelated sense, or a single isolated match with no structural echo.
- Answer "unclear" only when the evidence genuinely does not decide it.

Respond with JSON only:
{"verdict": "reuse" | "not_reuse" | "unclear", "confidence": 0.0-1.0,
 "reason": "<one sentence>"}
```

Field values are truncated for prompt economy: up to 12 matched labels, 6 matched
IRIs, 14 pattern classes, 18 ontology entity names.

**Note on prompt design.** The guidance deliberately biases toward agreement by
naming the common failure mode (generic words in unrelated senses). This makes a
*low* agreement statistic more meaningful, since it cannot be attributed to prompt
ambiguity.

## 4. Agreement statistics

**Cohen's κ** (two raters):
`κ = (p_o − p_e) / (1 − p_e)`, `p_e = Σ_k P_a(k)·P_b(k)` over categories k.

**Fleiss' κ** (three raters, fixed n per item): standard formulation over the
category-count matrix.

**Wilson interval** for a proportion, z = 1.96, used for precision. Chosen over
the normal approximation because it behaves correctly near 0 and 1 and at small n.

Interpretation follows Landis & Koch: ≥0.81 almost perfect, ≥0.61 substantial,
≥0.41 moderate, ≥0.21 fair, >0 slight, <0 worse than chance.

## 5. Results

### Four raters

Gemini and Claude were re-run to fill the items their first pass had dropped;
`merge_verdicts.py` folds a per-rater re-run into `machine_verdicts.csv` without
overwriting a verdict already recorded. Coverage after that merge:

| Rater | n | not_reuse | reuse | unclear |
|---|---:|---:|---:|---:|
| DeepSeek | 331 | 329 (99%) | 2 (1%) | 0 (0%) |
| GPT-4o | 331 | 305 (92%) | 2 (1%) | 24 (7%) |
| Claude Sonnet 5 | 322 | 244 (76%) | 65 (20%) | 13 (4%) |
| Gemini 2.5 Flash | 331 | 179 (54%) | 150 (45%) | 2 (1%) |

**Fleiss' κ = 0.059** over the 322 items judged by all four; 155 unanimous (48%).

| Pair | n | raw agreement | Cohen's κ |
|---|---:|---:|---:|
| GPT vs DeepSeek | 331 | **92.1%** | +0.066 |
| Claude vs DeepSeek | 322 | 76.4% | +0.039 |
| GPT vs Claude | 322 | 75.8% | +0.184 |
| Gemini vs Claude | 322 | 65.5% | +0.306 |
| Gemini vs DeepSeek | 331 | 54.7% | +0.014 |
| GPT vs Gemini | 331 | 51.4% | +0.024 |

Machine majority over all 331 items: `not_reuse` 271 (81.9%), `unclear` 51
(15.4%), `reuse` 9 (2.7%).

**Filling the missing verdicts moved nothing of substance.** Every figure above
is within 0.01 of its value before the re-run, the ranking of the pairs is
unchanged, and no rater's marginal distribution shifted by more than two
percentage points. That is worth recording: the conclusion did not depend on the
items that were missing.

### Interpreting these numbers — the prevalence paradox

κ must not be read alone here. GPT and DeepSeek agree on **92.1%** of items yet
score κ = 0.066, *lower* than the pair agreeing on only 65.5%. This is the
well-known prevalence (or base-rate) paradox in Cohen's κ: when both raters
assign one category almost always, expected agreement is already ~0.90, so the
correction drives κ towards zero even when the raters genuinely concur.

The correct reading is therefore **not** that all raters disagree:

* Three raters (GPT 92%, DeepSeek 99%, Claude 76% `not_reuse`) converge
  substantively; their pairwise raw agreement is 75.8–92.1%.
* **Gemini is the outlier**, assigning `reuse` to 45% of items, and it is the
  only rater whose marginals differ sharply from the rest.
* The low Fleiss κ is driven jointly by that dissent and by the skewed category
  distribution.

Any report of these figures must give raw agreement and the marginal
distributions alongside κ; κ in isolation would misrepresent the data.

### Three-rater figures (before DeepSeek was added)

| Rater | n judged | not_reuse | reuse | unclear |
|---|---:|---:|---:|---:|
| GPT-4o | 331 | 305 (92%) | 2 (1%) | 24 (7%) |
| Gemini 2.5 Flash | 317 | 170 (54%) | 145 (46%) | 2 (1%) |
| Claude Sonnet 5 | 312 | 240 (77%) | 60 (19%) | 12 (4%) |

**Fleiss' κ = 0.114** (slight) over the 299 items judged by all three.
Unanimous on 147 (49%); contested on 152 (51%).

| Pair | n | raw agreement | Cohen's κ |
|---|---:|---:|---:|
| Gemini vs Claude | 299 | 66.2% | +0.313 (fair) |
| GPT vs Claude | 312 | 76.6% | +0.182 (slight) |
| GPT vs Gemini | 317 | 51.1% | **+0.030 (chance)** |

**Conclusion (four raters).** Automated adjudication cannot substitute for human
judgement here, but the reason is more specific than "the models disagree". Three
of four raters converge substantively on `not_reuse` (raw pairwise agreement
75.8-92.1%), while Gemini dissents at 45% `reuse`. Fleiss' κ = 0.059 understates
that concordance because the category distribution is extreme. A single dissenting
rater is enough to make the machine majority unsafe as ground truth, and no
automatic procedure can decide which rater to believe. The items were therefore
adjudicated by hand; the machine verdicts are reported for transparency and as
evidence for this conclusion, not used as ground truth.

**And the human adjudication settled it against them.** Section 7 gives the
outcome: the human agrees with the machine majority on 65.0% of the adjudicated
items at Cohen's κ = 0.304 — fair, and below any threshold at which a machine
verdict could be substituted for a human one. Per rater against the human,
κ ranges from 0.019 (DeepSeek) to 0.472 (Claude). Notably, the rater that agreed
best with the other machines (DeepSeek, 92.1% with GPT) agrees *worst* with the
human. Inter-model agreement measures consistency, not correctness, and this is
a clean demonstration of the difference.

## 6. Missing data — stated, not hidden

| Rater | missing | why |
|---|---:|---|
| GPT-4o | 0 | — |
| Gemini | 14 | HTTP 503 rate limiting, and replies not parseable as JSON |
| Claude | 19 | replies not parseable as JSON |

The missing sets are **almost disjoint** (1 item of 33 missing from both), which
indicates transient API failure rather than systematically harder items. No item
lacks a verdict from all three.

**A defect to declare.** The harness discarded an unparseable model reply without
recording it, so the per-item failure reason cannot be recovered retrospectively —
only the aggregate counts above and the logged 503s. Every other stage of this
pipeline records its exclusions with a reason; this one did not, and the counts
here were reconstructed by set difference. A corrected harness should write a
`failures.csv` alongside the verdicts.

**Self-consistency.** Gemini changed its own verdict on **18 items** across
re-runs despite `temperature = 0`. Verdicts reported here are the last observed
per item. This is further evidence against automated adjudication: a rater that
is not self-consistent cannot anchor a ground truth.

## 7. Human adjudication — COMPLETE

The sheet lists contested items first, so a partial pass still captures the
informative cases. It was worked through in three strata, and the stratification
is what makes the headline number honest:

| Stratum | n | adjudicated | reuse found |
|---|---:|---|---:|
| **judge** — models contested it, or one rater was missing | 184 | all 184 | **71** |
| **spot-check** — a sample of the items all raters called the same way | 30 | all 30 | **0** |
| **deferred** — the remaining unanimous items, accepted as judged | 117 | none | assumed 0 |
| **total** | **331** | 214 | **71** |

### The headline

> **Label-mode precision = 71 / 331 = 0.215**, 95% Wilson CI **[0.174, 0.262]**.

Roughly **four in five** of the pairs that label matching proposes and IRI
matching does not are name collisions rather than re-modelled reuse.

**Do not quote the within-subset figure as the precision.** Precision computed
over the 214 adjudicated items alone is 0.332 (Wilson [0.272, 0.397]), and it is
an overestimate: the judge stratum is deliberately enriched for hard cases, so
it concentrates whatever genuine reuse exists. The 331-denominator figure is the
one to report.

**Sensitivity of the deferred stratum.** The spot-check found 0 reuse in 30
unanimous items, which bounds that stratum's reuse rate at 0.114 (upper Wilson
limit). Even at that upper bound the 117 deferred items would contribute at most
13 further reuse pairs, giving a worst-case precision of **0.255** — still around
one in four. The conclusion does not depend on the deferred stratum.

### Validity anchor: human versus machine

| | n | raw agreement | Cohen's κ |
|---|---:|---:|---:|
| human vs **machine majority** | 214 | 65.0% | **+0.304** (fair) |
| human vs Claude | 195 | 76.9% | **+0.472** (moderate) |
| human vs Gemini | 200 | 56.0% | +0.241 (fair) |
| human vs GPT-4o | 214 | 64.5% | +0.130 (slight) |
| human vs DeepSeek | 214 | 67.3% | +0.019 (chance) |

κ against the human majority is **below the threshold at which a machine verdict
could stand in for a human one**, which retrospectively justifies adjudicating by
hand rather than accepting the machine majority.

One result here deserves emphasis, because it is the clearest thing in the whole
experiment: **the rater that agreed best with the other machines agrees worst
with the human.** DeepSeek matches GPT on 92.1% of items and the human at
κ = 0.019 — chance. Claude, which sits furthest from the machine consensus,
matches the human best at κ = 0.472. Inter-model agreement measures whether
models are *consistent*; it is not evidence that they are *correct*, and here the
two run in opposite directions.

### What the 71 confirmed pairs are

63 in the GitHub portion and 8 in Paper/Repository, spread over **58 distinct
ontologies**. The patterns involved are led by DUL (31), then
`descriptionandsituation` (7), `timeinterval` (4), `ontopic`, `agentrole` and
`collectionentity` (3 each).

Set against the gate: 29 of those 58 ontologies already pass it, 10 fail G5 (no
IRI-matched pattern — these are the ones label mode would newly admit), 10 are
G3 duplicates, 5 fail G4 and 4 are G2 pattern copies. So even taken at face
value, label mode would add on the order of **10 ontologies** and **71 pairs** to
a corpus of 429 and 1,421 — a real correction, but not a different dataset.

## 8. Reproducing

```bash
cd pipeline
python adjudicate_labels.py build  --manifest out/MANIFEST.csv --data-root <corpus> --out out/adjudication
python adjudicate_labels.py judge  --out out/adjudication
python adjudicate_labels.py sample --out out/adjudication      # all items
python adjudicate_labels.py score  --out out/adjudication
```

Item selection and sampling are seeded and deterministic. Model outputs are not
reproducible bit-for-bit across API versions; the verdicts as collected are
archived in `machine_verdicts.csv`.
