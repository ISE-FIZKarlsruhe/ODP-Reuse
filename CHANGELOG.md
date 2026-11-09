# Changelog

Every version is a git tag, a GitHub Release and a Zenodo version under the
concept DOI [10.5281/zenodo.17751740](https://doi.org/10.5281/zenodo.17751740),
which always resolves to the newest. The newest version is listed first. From
v1.3.0 on, `publish-release.yml` writes each entry from the same numbers as
`release-iri/history.csv`.

## Unreleased

Corrected two diagnostic columns of 7 rows first released in v1.2.0
(`OBO/uberon` and 6 GitHub ontologies). `n_pattern_errors` counted the
characters of a parser message instead of the pattern files that failed to
parse (240 to 1,000 instead of 1, or 6 for uberon), and `pattern_errors` held
that message one character at a time: a defect of the cache
`build_manifest.py --resume` reads, fixed before v1.3.0. The corrected values
are those of the gate re-run on the archived files
([reproduce run 42498698949](https://github.com/ISE-FIZKarlsruhe/ODP-Reuse/actions/runs/42498698949)).
No verdict, view or file depends on these columns, and none changed.

## v1.3.0 — 2026-11-08

**+178 ontologies · +202 ontology–pattern pairs** in the quality-gated views, from the harvest of 2026-11-08 ([workflow run](https://github.com/ISE-FIZKarlsruhe/ODP-Reuse/actions/runs/42466505380)). 1,732 ontologies were added to the corpus; 178 pass the quality gate and the other 1,554 are kept with the rule that excluded them.

No ontology, file or verdict of an earlier version was removed or changed.

| view | ontologies | pairs |
|---|---:|---:|
| `github-reuse` | 442 → 620 (+178) | 669 → 871 (+202) |
| `github-complete` | 37 → 42 (+5) | 100 → 108 (+8) |

Zenodo: [10.5281/zenodo.24054749](https://doi.org/10.5281/zenodo.24054749) · added ontologies: `release-iri/additions/v1.3.0.csv`

## v1.2.0 — 2026-08-05

The quality-gated release. Zenodo:
[10.5281/zenodo.21808321](https://doi.org/10.5281/zenodo.21808321).

- Every discovered ontology, **1,973**, ships unfiltered, and `MANIFEST.csv`
  records the quality gate's verdict on each (rules G1–G6); **501** pass.
- Six views over that one corpus: `*-reuse` and `*-complete` for the GitHub,
  paper/repository (noOBO) and DOS-DP (OBO) portions.
- The 1,017 zero-byte pattern files of the earlier harvest were refilled before
  gating.
- Rule G2 removes ontologies that are themselves copies of a pattern: the
  previously published complete-reuse set contained 14 such files out of 49.
- An IRI-anonymised twin of the corpus, `corpus-hashed.zip`.

## v1.0.0 — 2025-11-28

First deposit, before the quality gate: the GitHub, paper/repository and
DOS-DP datasets and the gold datasets. Zenodo:
[10.5281/zenodo.17751741](https://doi.org/10.5281/zenodo.17751741). Its counts
are not comparable with later versions.
