# Label-mode adjudication

Machine annotators: gpt, gemini, claude, deepseek  
Items adjudicated: 331

**Inter-model agreement (Fleiss' κ): 0.059** — slight (complete cases only: 322 of 331 items judged by all 4 raters)

> κ alone can mislead when one category dominates: with a base rate near 90%, chance agreement is already ~0.90 and the correction drives κ towards zero even when raters concur. Report raw agreement and the marginals alongside it.

> Measures whether the models are *consistent*, not whether they are *correct*. Models share training data and failure modes; this number is not evidence of validity on its own.

| Machine majority | n | share |
|---|---:|---:|
| not_reuse | 271 | 81.9% |
| unclear | 51 | 15.4% |
| reuse | 9 | 2.7% |

## Stratified precision over the FULL item set

| Stratum | n | adjudicated | reuse |
|---|---:|---|---:|
| judge (contested + incomplete) | 184 | all | 71 |
| spot-check (sample of unanimous) | 30 | all | 0 |
| deferred (unanimous, accepted) | 117 | none | assumed 0 |

**Label-mode precision = 71/331 = 0.215** (95% Wilson CI [0.174, 0.262])

Sensitivity: the spot-check found 0 reuse in 30 unanimous items, bounding that stratum's reuse rate at 0.114. Even at that upper bound the deferred stratum would add at most 13 items, giving a worst-case precision of 0.255.

> Precision computed over the adjudicated subset alone would **overstate** reuse, because the contested stratum is deliberately enriched for hard cases. The figure above is the one to report.

## Human adjudication — adjudicated subset only

Adjudicated: 214 · reuse 71 · not_reuse 143 · unclear 0

Precision *within the adjudicated subset*: 0.332 (95% Wilson CI [0.272, 0.397], n=214). This is NOT the label-mode precision — see the stratified table above.

## Validity anchor: human vs machine majority

Paired items: 214 · raw agreement 65.0%  
**Cohen's κ = 0.304** — fair

κ is below the substantial threshold. The machine verdict must **not** be extrapolated; report the human sample alone and state that automated adjudication did not reach acceptable agreement.

- human vs **gpt**: κ = 0.130
- human vs **gemini**: κ = 0.241
- human vs **claude**: κ = 0.472
- human vs **deepseek**: κ = 0.019
