# Quality-gate attrition

Counts are per dataset; each ontology is attributed to the FIRST rule
it fails, so the columns partition `discovered`.

| dataset | discovered | dropped_G1_parse | dropped_G2_pattern_copy | dropped_G3_duplicate | dropped_G4_substantive | dropped_G5_has_gt | dropped_G6_complete_reuse | kept_reuse | kept_complete_reuse |
|---|---|---|---|---|---|---|---|---|---|
| github | 3619 | 424 | 51 | 342 | 1651 | 531 | 0 | 620 | 42 |
| noOBO | 78 | 1 | 0 | 0 | 1 | 25 | 0 | 51 | 6 |
| OBO | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 8 |

Rules: G1 parseable, G2 not a pattern copy, G3 not a duplicate,
G4 substantive (>=1 own entities), G5 has ground truth,
G6 complete reuse (variant membership). See odpqual/gate.py.
