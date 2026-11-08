# Changelog

Every version is a git tag, a GitHub Release and a Zenodo version under the
concept DOI [10.5281/zenodo.17751740](https://doi.org/10.5281/zenodo.17751740),
which always resolves to the newest. The newest version is listed first. From
v1.3.0 on, `publish-release.yml` writes each entry from the same numbers as
`release-iri/history.csv`.

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
