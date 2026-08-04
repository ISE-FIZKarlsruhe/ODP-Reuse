# Research questions

All figures below are computed over the **real corpus** — every ontology that
parses, whether or not it passes the gate — with a `gate_passed` flag on every
row, so either population can be recovered.

## RQ1 · How widespread is reuse?

| Portion | Ontologies | Patterns | Complete | Mean patterns/onto |
|---|---|---|---|---|
| DOS-DP | 8 | 420 | 8 | 94.75 |
| GitHub | 442 | 45 | 37 | 1.51 |
| Paper &amp; Repository | 51 | 45 | 6 | 1.76 |

<div id="odp-thresholds" class="odp-viz" data-src="data/thresholds.json"></div>

Reuse falls away sharply as the coverage threshold rises: partial reuse is the
norm, full reuse the exception — except in the DOS-DP portion, where templating
makes complete instantiation automatic.

## RQ2 · In what structural form?

| Portion | Direct | Subclass | As property | In restriction | Equivalence |
|---|---|---|---|---|---|
| DOS-DP | 0.1% | 0.3% | 0.0% | 99.3% | 0.1% |
| GitHub | 31.8% | 13.9% | 18.3% | 17.4% | 3.0% |
| Paper &amp; Repository | 37.5% | 17.2% | 0.3% | 25.2% | 2.9% |

<div id="odp-forms" class="odp-viz" data-src="data/forms.json"></div>

## RQ3 · Where does the complexity come from?

| Portion | Pairs | Pattern DL | Ontology DL | Ontology more expressive |
|---|---|---|---|---|
| DOS-DP | 758 | EL | ALCHIQ | 100.0% |
| GitHub | 669 | ALCHI | ALCIQ | 70.5% |
| Paper &amp; Repository | 90 | ALCHI | ALC | 63.3% |

Patterns are consistently lighter than the ontologies that reuse them. The extra
logical machinery is contributed by the ontology, not inherited from the pattern.
DL labels are construct-based and approximate — they order corpora, they do not
certify them.

## RQ4 · How far do ontologies grow beyond the pattern?

| Portion | Ontologies | Classes | Added | % added |
|---|---|---|---|---|
| DOS-DP | 8 | 52953.25 | 39180.12 | 74.1% |
| GitHub | 442 | 33.9 | 31.27 | 91.6% |
| Paper &amp; Repository | 51 | 39.86 | 35.1 | 86.7% |

## Most reused patterns

<div id="odp-toppat" class="odp-viz" data-src="data/top_patterns.json"></div>

`abnormalAbsenceOfAnatomicalEntity` `abnormalAbsenceOfAnatomicalEntityInLocation` `abnormalAbsenceOfBiologicalProcess` `abnormalAbsenceOfBiologicalProcessInLocation` `abnormalAbsenceOfChemicalEntityInLocation` `abnormalAgenesisOfAnatomicalEntity` `abnormalAnatomicalEntity` `abnormalBiologicalProcess` `abnormalBiologicalProcessInLocation` `abnormalColorOfAnatomicalEntity` `DUL` `situation` `collectionentity` `partof` `timeinterval` `participation` `timeindexedsituation` `objectrole` `sequence` `description` `DUL` `situation` `actorODP` `resourceODP` 
