---
hide: [navigation]
---

<div class="odp-hero" markdown>

# Are design patterns *actually* reused in ontologies? { .odp-hero-title }

We went looking, in 1,973 ontologies harvested from code repositories,
scholarly publications and ontology portals. Most of what we found could not be
used — and **why** is the interesting part.

<div class="odp-stat-row" markdown>
<div class="odp-stat odp-stat--muted"><b>1,973</b><span>ontologies harvested</span></div>
<div class="odp-stat odp-stat--hero"><b>501</b><span>survive the quality gate</span></div>
<div class="odp-stat"><b>1,517</b><span>verified reuse pairs</span></div>
<div class="odp-stat"><b>488</b><span>distinct patterns</span></div>
<div class="odp-stat"><b>673</b><span>fully reused</span></div>
</div>

<span class="odp-version">version v1.2.0</span>

</div>

## Only 25% of what we harvested is usable

Every ontology is tested by six rules, in order. Each is attributed to the
**first** rule it fails, so the bands below partition the corpus exactly. Hover a
band to see what it means.

<div id="odp-funnel" class="odp-viz" data-src="data/funnel.json"></div>

!!! tip "This is the finding, not an apology"
    A benchmark built from web-harvested ontologies is only as good as its
    filtering. Publishing the corpus **unfiltered**, with the verdict and the
    evidence for every exclusion, is what makes the gate falsifiable — you can
    disagree with any rule and re-run it.

## What is here

<div class="odp-cards" markdown>

[:material-pipe: **The pipeline**<br><small>How the corpus is discovered, repaired, gated and released — with every parameter.</small>](pipeline.md){ .odp-card }

[:material-filter-variant: **The filters**<br><small>Six rules, what each removes and why, with the counts and the evidence.</small>](filters.md){ .odp-card }

[:material-chart-box: **Research questions**<br><small>Extent, forms, complexity and expansion of reuse across the three portions.</small>](rqs.md){ .odp-card }

[:material-rocket-launch: **Use cases**<br><small>The dataset as training signal for recommendation, and as ground truth for extraction.</small>](use-cases.md){ .odp-card }

[:material-graph: **Patterns**<br><small>Every reused pattern, rendered as an interactive graph.</small>](patterns.md){ .odp-card }

[:material-download: **Get the data**<br><small>Zenodo for the corpus, this repository for the tables.</small>](download.md){ .odp-card }

</div>
