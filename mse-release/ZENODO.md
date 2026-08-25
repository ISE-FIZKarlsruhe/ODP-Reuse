MSE-ODP 1.0.0: candidate ontology design patterns extracted from 116 materials science and engineering ontologies.

WHAT THIS DEPOSITION CONTAINS

Ontology Design Patterns (ODPs) are reusable modelling solutions, and several published methods extract candidate patterns from existing ontologies automatically. Those methods had not been applied to materials science and engineering, a domain where ontology development is active, foundational commitments are contested, and no catalogue of domain patterns exists. This deposition holds the result of applying them.

  corpus/       116 MSE ontologies (114 of which parse as RDF), exactly as evaluated
  predictions/  every grouping of entities each method proposed, as JSON Lines
  statistics/   ground-truth-free measures used to rank those groupings for review
  provenance/   the version audit of every ontology, including what was refreshed and what was not
  MANIFEST.csv  one row per file, with checksum, declared version and upstream repository
  validate_release.py  a consistency check for an unpacked copy

THERE IS NO GROUND TRUTH IN THIS DATASET

No ontology here is annotated with the patterns it reuses. No accuracy, precision, recall or F-score is reported, and none should be inferred. What is reported is what each method proposed, how much independent methods agree with one another, how often a grouping recurs across ontologies, whether it is anchored in a foundational vocabulary, and whether it overlaps a documented pattern. These rank candidates for expert review; they are not evidence that a candidate is good. The included validator fails the release if any prediction file carries a score.

For extraction methods scored against verified ground truth, see the companion ODP-Reuse deposition.

THE CORPUS AND ITS CURRENCY

The corpus is the collection assembled for "The landscape of ontologies in materials science and engineering: a survey and evaluation" (Norouzi, Waitelonis and Sack, arXiv:2408.06034). Every artefact was re-checked against its upstream repository before evaluation and 29 were refreshed to a newer release; several had been held at pre-release versions two major versions behind their current one. provenance/upstream_check.json records the verdict for every file, including those where nothing newer was found and those where a proposed update was rejected as the wrong artefact, so that the corpus's currency can be re-checked rather than taken on trust.

A CAVEAT WORTH READING BEFORE USING THE RANKING

Ranking candidates by how widely they recur across ontologies puts pairs of Basic Formal Ontology upper-level classes at the top, because a dozen MSE ontologies import BFO. Recurrence in this corpus largely measures shared foundational scaffolding rather than shared design. Consumers of the recurrence signal should exclude candidates composed entirely of foundational terms and require at least one term from the host ontology's own namespace.

LICENCE

The predictions, statistics, provenance and documentation are released under CC-BY-4.0. The ontology files in corpus/ remain under their own licences and are redistributed only so that these results are reproducible against the exact inputs used; MANIFEST.csv names the upstream of each.

CITING

Please cite this deposition together with arXiv:2408.06034, and cite the originating method papers if you use the extraction results.

SOURCE CODE

https://github.com/ISE-FIZKarlsruhe/ODP-Reuse
