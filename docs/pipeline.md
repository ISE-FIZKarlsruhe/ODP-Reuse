# The pipeline

Six stages. Each writes an artefact the next reads, and each is a single command
— nothing here is done by hand.

<div class="odp-flow" markdown>

<div class="odp-step" data-n="1" markdown>
### Discover
Ontologies are found by four independent channels: GitHub code search over
pattern IRIs, Google advanced search over ontology file types, ontology portals
(BioPortal, MatPortal, LOV, AgroPortal), and manual screening of workshop
proceedings and the papers citing the foundational ODP literature.

`MultiSource/odp_multi_source_finder.py`
</div>

<div class="odp-step" data-n="2" markdown>
### Repair
The harvester wrote a pattern file even when the fetch returned nothing: **23.1%
of pattern files were zero bytes**, and every one of them was silently counted as
*no reuse*. They are refilled from the four pattern libraries, with a recorded
rule for every choice.

`pipeline/repair_patterns.py`
</div>

<div class="odp-step" data-n="3" markdown>
### Normalise
Annotation vocabularies (SKOS, DC, DCTERMS, CPANN, VANN, FOAF …) are stripped
from **both** the ontology and the pattern, so the two sides are treated
identically. Entities are collected under the WIDE definition.

`pipeline/odpqual/entities.py`
</div>

<div class="odp-step" data-n="4" markdown>
### Gate
Six rules decide what is usable. The corpus is **not** filtered destructively —
the verdict is recorded as metadata so any rule can be disputed and re-run.

`pipeline/odpqual/gate.py` → `MANIFEST.csv`
</div>

<div class="odp-step" data-n="5" markdown>
### Audit
Every removal is re-derived from the source files by an independent checker that
does not read the gate's verdict. A removal that cannot be reproduced is flagged
`DISPUTED` and fails the build.

`pipeline/audit_filters.py`
</div>

<div class="odp-step" data-n="6" markdown>
### Publish
The unfiltered corpus plus six views, each a filter over it rather than a copy.
Tables to this repository, the corpus to Zenodo under one concept DOI.

`pipeline/build_release.py`
</div>

</div>

## Ground truth, three ways

An entity counts as reused under whichever definition you choose. All three are
computed and reported **separately** — never silently merged.

| Mode | Rule | Catches | Blind to |
|---|---|---|---|
| `iri` | the IRI occurs in both pattern and ontology | verbatim reuse | re-modelled reuse |
| `label` | a pattern class and a host entity share a normalised label | patterns re-minted under a local namespace | renamed concepts |
| `iri+label` | the union | both | — |

!!! warning "There is no similarity threshold, deliberately"
    Label matching is **exact equality on a normalised string** — CamelCase split,
    non-alphanumerics dropped, lowercased, so `AgentRole`, `Agent Role`,
    `agent_role` and `agent-role` all collapse to `agentrole`. Jaccard, edit
    distance and embedding cosine would each need a cut-off that cannot be
    justified from first principles. Two precision guards apply, both
    parameter-free: a fixed generic-label blocklist, and a structural-coherence
    check that keeps only the largest connected component of matched entities.

## Parameters

Every knob, its default, and what moves if you change it.

| Parameter | Default | Effect |
|---|---|---|
| `--min-own-entities` | `10` | G4 threshold. Most failures have **zero** own entities, so lowering it recovers little. |
| `--keep-annotations` | off | Reproduces the old asymmetric behaviour where annotations were stripped from patterns but not hosts. |
| `--max-mb` | `0` (no limit) | Dataset membership is never size-limited. Size constrains only embedding. |
| `--max-entities` (embedding) | `5000` | Walk count grows linearly with entities; excluded ontologies stay in the dataset and are logged. |
| `--gt-mode` | `iri` | Which ground-truth definition to use. |
| `--population` | `all` | `all` reports the real corpus; `gated` restricts to what passes. |
| embedding seed | `42` | Fixed. Re-running reproduces the numbers exactly. |
