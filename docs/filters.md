# Filters and parameters

Two kinds of filter exist here and they must not be confused.

<div class="odp-split" markdown>

<div class="odp-panel odp-panel--a" markdown>
### Dataset filters
Decide what is **in** the high-quality dataset. These are the methodological
contribution and are reported as findings, not hidden as housekeeping.
</div>

<div class="odp-panel odp-panel--b" markdown>
### Stage filters
Practical limits on **one computation**. They never remove an ontology from the
dataset — they mean "this step could not process it", and every exclusion is
recorded with its reason.
</div>

</div>

!!! success "The dataset has no size limit"
    A 1 GB ontology is a legitimate corpus member. Size constrains only the
    embedding stage. Verified: **zero** ontologies are excluded for size.

## What each rule removes

<div id="odp-rules" class="odp-viz" data-src="data/rules.json"></div>

| Portion | Rule | Removed | Share |
|---|---|---:|---:|
| GitHub | Unparseable | 268 | 14.2% |
| GitHub | Is a pattern copy | 36 | 1.9% |
| GitHub | Duplicate of another | 203 | 10.8% |
| GitHub | Stub / too small | 539 | 28.6% |
| GitHub | No pattern overlap | 399 | 21.1% |
| Paper &amp; Repository | Unparseable | 1 | 1.3% |
| Paper &amp; Repository | Is a pattern copy | 0 | 0.0% |
| Paper &amp; Repository | Duplicate of another | 0 | 0.0% |
| Paper &amp; Repository | Stub / too small | 1 | 1.3% |
| Paper &amp; Repository | No pattern overlap | 25 | 32.1% |
| DOS-DP | Unparseable | 0 | 0.0% |
| DOS-DP | Is a pattern copy | 0 | 0.0% |
| DOS-DP | Duplicate of another | 0 | 0.0% |
| DOS-DP | Stub / too small | 0 | 0.0% |
| DOS-DP | No pattern overlap | 0 | 0.0% |

## Every removal was re-verified

An independent checker re-derives each removal from the source files without
reading the gate's verdict. A removal it cannot reproduce is `DISPUTED`.

| Rule | Removed | Re-verified | Disputed |
|---|---:|---:|---:|
| Unparseable | 269 | 269 | 0 |
| Is a pattern copy | 36 | 36 | 0 |
| Duplicate of another | 203 | 203 | 0 |
| Stub / too small | 540 | 540 | 0 |
| No pattern overlap | 424 | 424 | 0 |

Per-ontology evidence ships with the release under `audit/removed_<rule>.csv`,
so any exclusion can be inspected and challenged.
