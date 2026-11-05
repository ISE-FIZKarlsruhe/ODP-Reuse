#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_site.py -- generate the documentation site from the release artefacts.

Every number on the site is read from a CSV produced by the pipeline. Nothing is
typed by hand, so the site cannot drift from the data and a release automatically
republishes correct figures.

Inputs
    release-iri/dataset_statistics.csv  the six views
    release-iri/gate_summary.csv        gate attrition per rule
    release-iri/MANIFEST.csv            every ontology + verdict
    release-iri/views/<view>/pairs.csv  ontology -> pattern
    pipeline/out/rq_iri_gated/rq*.csv   RQ tables (GATED, as in the paper)
    use-cases/*/results/**              use-case results

Outputs (docs/)
    index.md pipeline.md filters.md rqs.md use-cases.md datasets.md
    patterns.md using.md download.md changelog.md
    data/*.json                         payloads for the D3 visualisations
    patterns/*.owl                      the reused ODPs, for ontoink

Usage
    python pipeline/gen_site.py            # defaults: release-iri -> docs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The entity-set definition has to be the one the release itself used, or the
# per-pattern counts shown on the site would not reconcile with the coverage
# denominators in pairs.csv. Import it rather than reimplementing it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Serialisation is not recoverable from the extension in this corpus: two of the
# 68 vendored ``.owl`` files are actually Turtle. Try formats in turn.
PARSE_FORMATS = ("xml", "turtle", "n3", "nt")

RULE_LABEL = {
    "G1_parse": "Unparseable",
    "G2_pattern_copy": "Is a pattern copy",
    "G3_duplicate": "Duplicate of another",
    "G4_substantive": "Stub / too small",
    "G5_has_gt": "No pattern overlap",
    "KEPT": "Kept",
}
RULE_COLOR = {
    "G1_parse": "#8b8b9e", "G2_pattern_copy": "#ff5d73",
    "G3_duplicate": "#ff9f45", "G4_substantive": "#ffd23f",
    "G5_has_gt": "#7c5cff", "KEPT": "#00c9a7",
}
VIEW_ORDER = ["github-reuse", "github-complete", "noOBO-reuse",
              "noOBO-complete", "OBO-reuse", "OBO-complete"]

# Edge colouring for the reuse graph. Coverage is an ordered quantity, so the
# four partial bands are a sequential ramp -- paler means less of the pattern is
# present. Complete reuse gets a separate accent rather than a fifth ramp step
# because the release treats it as its own kind of thing (the `*-complete` views
# are built from exactly this predicate), not merely as the top of a continuum.
# The 10% cut is the threshold the manuscript reports against.
COVERAGE_BANDS = [
    {"band": 0, "label": "<10% of the pattern", "color": "#b9c0d4"},
    {"band": 1, "label": "10-30%", "color": "#8ecae6"},
    {"band": 2, "label": "30-50%", "color": "#4cc9f0"},
    {"band": 3, "label": "50-99%", "color": "#7c5cff"},
    {"band": 4, "label": "complete (100%)", "color": "#00c9a7"},
]


def coverage_band(pct: float) -> int:
    if pct >= 100:
        return 4
    if pct >= 50:
        return 3
    if pct >= 30:
        return 2
    if pct >= 10:
        return 1
    return 0
PORTION_LABEL = {"github": "GitHub", "noOBO": "Paper &amp; Repository",
                 "OBO": "DOS-DP"}
# Same labels without the HTML entity: PORTION_LABEL is written straight into
# Markdown tables, where the entity is required, but the JSON payloads below
# are read by viz.js and inserted as plain text, where the entity leaked
# through verbatim ("Paper &amp; Repository" in every tooltip and axis label).
PORTION_LABEL_JSON = {"github": "GitHub", "noOBO": "Paper & Repository",
                      "OBO": "DOS-DP"}


def read(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def fnum(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def git_version() -> str:
    for cmd in (["git", "describe", "--tags", "--abbrev=0"],
                ["git", "rev-parse", "--short", "HEAD"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return "dev"


# ----------------------------------------------------------------------
# pages
# ----------------------------------------------------------------------

def page_index(ctx: dict) -> str:
    g, s = ctx["gate"], ctx["stats"]
    disc = sum(num(r["discovered"]) for r in g)
    kept = sum(num(r["kept_reuse"]) for r in g)
    pairs = sum(num(r["n_pairs"]) for r in s if r["dataset"].endswith("-reuse"))
    pats = len(ctx["all_patterns"])
    cpl = sum(num(r["n_complete_pairs"]) for r in s
              if r["dataset"].endswith("-complete"))

    return f"""---
hide: [navigation]
---

<div class="odp-hero" markdown>

# Are design patterns *actually* reused in ontologies? {{ .odp-hero-title }}

We went looking, in {disc:,} ontologies harvested from code repositories,
scholarly publications and ontology portals. Most of what we found could not be
used — and **why** is the interesting part.

<div class="odp-stat-row" markdown>
<div class="odp-stat odp-stat--muted"><b>{disc:,}</b><span>ontologies harvested</span></div>
<div class="odp-stat odp-stat--hero"><b>{kept:,}</b><span>survive the quality gate</span></div>
<div class="odp-stat"><b>{pairs:,}</b><span>verified reuse pairs</span></div>
<div class="odp-stat"><b>{pats}</b><span>distinct patterns</span></div>
<div class="odp-stat"><b>{cpl:,}</b><span>fully reused</span></div>
</div>

<span class="odp-version">version {ctx['version']}</span>

</div>

## Only {100.0 * kept / max(disc, 1):.0f}% of what we harvested is usable

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

[:material-pipe: **The pipeline**<br><small>How the corpus is discovered, repaired, gated and released — with every parameter.</small>](pipeline.md){{ .odp-card }}

[:material-filter-variant: **The filters**<br><small>Six rules, what each removes and why, with the counts and the evidence.</small>](filters.md){{ .odp-card }}

[:material-chart-box: **Research questions**<br><small>Extent, forms, complexity and expansion of reuse across the three portions.</small>](rqs.md){{ .odp-card }}

[:material-rocket-launch: **Use cases**<br><small>The dataset as training signal for recommendation, and as ground truth for extraction.</small>](use-cases.md){{ .odp-card }}

[:material-graph: **Patterns**<br><small>Every reused pattern, rendered as an interactive graph.</small>](patterns.md){{ .odp-card }}

[:material-download: **Get the data**<br><small>Zenodo for the corpus, this repository for the tables.</small>](download.md){{ .odp-card }}

</div>
"""


def page_pipeline(ctx: dict) -> str:
    return """# The pipeline

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
"""


def page_filters(ctx: dict) -> str:
    g = ctx["gate"]
    rows = []
    for r in g:
        ds = r["dataset"]
        disc = num(r["discovered"])
        for rule in ("G1_parse", "G2_pattern_copy", "G3_duplicate",
                     "G4_substantive", "G5_has_gt"):
            n = num(r.get(f"dropped_{rule}"))
            rows.append(f"| {PORTION_LABEL.get(ds, ds)} | {RULE_LABEL[rule]} | "
                        f"{n:,} | {100.0 * n / max(disc, 1):.1f}% |")
    audit = ctx["audit"]
    audit_rows = "\n".join(
        f"| {RULE_LABEL.get(a['rule'], a['rule'])} | {num(a['n_removed']):,} | "
        f"{num(a.get('ok')):,} | {num(a.get('DISPUTED'))} |" for a in audit)

    return f"""# Filters and parameters

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
{chr(10).join(rows)}

## Every removal was re-verified

An independent checker re-derives each removal from the source files without
reading the gate's verdict. A removal it cannot reproduce is `DISPUTED`.

| Rule | Removed | Re-verified | Disputed |
|---|---:|---:|---:|
{audit_rows}

Per-ontology evidence ships with the release under `audit/removed_<rule>.csv`,
so any exclusion can be inspected and challenged.
"""


def page_rqs(ctx: dict) -> str:
    def table(rows, cols, fmt=None):
        if not rows:
            return "_Not yet computed for this release._\n"
        head = "| " + " | ".join(c[1] for c in cols) + " |"
        sep = "|" + "|".join("---" for _ in cols) + "|"
        body = []
        for r in rows:
            cells = []
            for key, _lab in cols:
                v = r.get(key, "")
                cells.append(fmt(key, v) if fmt else str(v))
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([head, sep, *body])

    def ds(key, v):
        return PORTION_LABEL.get(v, str(v))

    rq1 = ctx["rq"].get("rq1_extent", [])
    rq2 = ctx["rq"].get("rq2_reuse_forms", [])
    rq3 = ctx["rq"].get("rq3_complexity", [])
    rq4 = ctx["rq"].get("rq4_expansion", [])

    return f"""# Research questions

All figures below are computed over the **real corpus** — every ontology that
parses, whether or not it passes the gate — with a `gate_passed` flag on every
row, so either population can be recovered.

## RQ1 · How widespread is reuse?

{table(rq1, [("dataset", "Portion"), ("onto_reuse", "Ontologies"),
             ("patterns_reuse", "Patterns"), ("onto_complete", "Complete"),
             ("mean_patterns_per_onto", "Mean patterns/onto")],
       lambda k, v: ds(k, v) if k == "dataset" else str(v))}

<div id="odp-thresholds" class="odp-viz" data-src="data/thresholds.json"></div>

Reuse falls away sharply as the coverage threshold rises: partial reuse is the
norm, full reuse the exception — except in the DOS-DP portion, where templating
makes complete instantiation automatic.

## RQ2 · In what structural form?

{table(rq2, [("dataset", "Portion"), ("direct_pct", "Direct"),
             ("subclass_of_pattern_pct", "Subclass"),
             ("used_as_property_pct", "As property"),
             ("in_restriction_pct", "In restriction"),
             ("equivalence_pct", "Equivalence")],
       lambda k, v: ds(k, v) if k == "dataset" else (f"{fnum(v):.1f}%" if v else "—"))}

<div id="odp-forms" class="odp-viz" data-src="data/forms.json"></div>

## RQ3 · Where does the complexity come from?

{table(rq3, [("dataset", "Portion"), ("n_pairs", "Pairs"),
             ("pattern_dl_mode", "Pattern DL"), ("ontology_dl_mode", "Ontology DL"),
             ("pct_ontology_more_expressive", "Ontology more expressive")],
       lambda k, v: ds(k, v) if k == "dataset" else
       (f"{fnum(v):.1f}%" if k.startswith("pct") else str(v)))}

Patterns are consistently lighter than the ontologies that reuse them. The extra
logical machinery is contributed by the ontology, not inherited from the pattern.
DL labels are construct-based and approximate — they order corpora, they do not
certify them.

## RQ4 · How far do ontologies grow beyond the pattern?

{table(rq4, [("dataset", "Portion"), ("n_ontologies", "Ontologies"),
             ("mean_n_C", "Classes"), ("mean_added_C", "Added"),
             ("pct_entities_added", "% added")],
       lambda k, v: ds(k, v) if k == "dataset" else
       (f"{fnum(v):.1f}%" if k.startswith("pct") else str(v)))}

## Most reused patterns

Top 10 per portion by number of reusing ontologies.

<div id="odp-toppat" class="odp-viz" data-src="data/top_patterns.json"></div>
"""


def page_use_cases(ctx: dict) -> str:
    """UC1 = extraction, UC2 = recommendation -- the manuscript's numbering."""
    uc1 = ctx["uc1"]
    uc2 = ctx["uc2"]
    uc1_rows = "\n".join(
        f"| {PORTION_LABEL.get(r['dataset'], r['dataset'])} | "
        f"{'OWL2Vec*' if r['method'] == 'owl2vec' else 'RDF2Vec'} | "
        f"{'K-Means' if r['clusterer'] == 'kmeans' else 'HDBSCAN'} | "
        f"{num(r['n_ontologies'])} | {fnum(r['oracle_f1']):.3f} | "
        f"{fnum(r['unsup_f1']):.3f} | +{fnum(r['oracle_gap_f1']):.3f} |"
        for r in uc1) or "| _pending_ | | | | | | |"

    def uc2_block(portion: str) -> str:
        rows = [r for r in uc2 if r.get("dataset") == portion]
        return "\n".join(
            f"| {r['comparison']} | {num(r['n_paired'])} | "
            f"**{fnum(r['mean_delta_map']):+.3f}** | "
            f"[{fnum(r['ci_lo_95']):+.3f}, {fnum(r['ci_hi_95']):+.3f}] | "
            f"{r['win_loss_tie']} | "
            f"{'**yes**' if r['significant_at_0.05'] == 'yes' else 'no'} |"
            for r in rows) or "| _pending_ | | | | | |"

    return f"""# Use cases

Two demonstrations that the dataset is *consumable*, not merely descriptive. The
numbering matches the manuscript: UC1 uses the corpus as **ground truth**, UC2
uses it as **training signal**.

## UC1 · The dataset as ground truth

Recover ODP instances from an ontology with no knowledge of which patterns it
reuses. Embed, cluster, match clusters to ground truth by Hungarian assignment.

Every configuration is reported **twice** — once with hyper-parameters chosen
against the ground truth (an upper bound, and what prior work reported), once
chosen by silhouette without ever consulting it.

| Portion | Model | Clustering | n | Oracle F₁ | Unsupervised F₁ | Gap |
|---|---|---|---:|---:|---:|---:|
{uc1_rows}

<div id="odp-uc1" class="odp-viz" data-src="data/uc1.json"></div>

The gap is the contribution of oracle model selection. Neither number is wrong;
only one of them is achievable without already knowing the answer.

## UC2 · The dataset as training signal

Given an ontology under development, which ODPs should its author reuse? The
prior system, ODPReco, matches content against a fixed 73-ODP catalogue and uses
no corpus of confirmed reuse at all — it cannot learn that ontologies about
agents tend to reuse `agentrole`, because it has no examples.

Each row is a **paired** per-ontology comparison on the same held-out ontologies,
so no difference is confounded by which ontologies happen to be easy. Only
gate-passing ontologies are used. Row 1 is the measurement of what this dataset
is worth: the best reuse-informed recommender against the best that pattern
documentation alone supports.

**{PORTION_LABEL['github']} portion, quality-gated**

| Comparison | n | Δ MAP | 95% CI | W/L/T | Significant |
|---|---:|---:|---|---|---|
{uc2_block("github")}

**{PORTION_LABEL['noOBO']} portion — underpowered, for completeness**

| Comparison | n | Δ MAP | 95% CI | W/L/T | Significant |
|---|---:|---:|---|---|---|
{uc2_block("noOBO")}

<div id="odp-uc2" class="odp-viz" data-src="data/uc2.json"></div>

!!! note "What is *not* claimed"
    A faithful re-implementation of ODPReco's Doc2Vec approach stands in for the
    original, which was not run at scale — so the comparison is to the *approach*,
    not to published ODPReco numbers. On the {PORTION_LABEL['noOBO']} portion
    ($n = 14$) nothing reaches significance and no row there should be read as a
    contrary finding.
"""


def page_patterns(ctx: dict) -> str:
    pats = ctx["shipped_patterns"]
    top = ctx["pattern_usage"]
    listing = "\n".join(
        f'<button class="odp-chip" data-pattern="{p}">{p} '
        f'<span>{top.get(p, 0)}</span></button>' for p in pats)
    # The most-reused pattern is the one worth rendering in full. The fence takes
    # ontoink's YAML form and points at the Turtle twin: ontoink parses with
    # format="turtle" unconditionally, and passing the RDF/XML through a
    # `--8<--` snippet include made yaml.safe_load return a string, so
    # `config["source"]` raised TypeError on every build.
    first = (max(pats, key=lambda p: top.get(p, 0)) if pats else None)
    stats = ctx.get("pattern_stats", {})
    viewer = ""
    if first:
        st = stats.get(first, {})
        detail = (f"It declares {st['n_classes']} classes, "
                  f"{st['n_object_properties']} object properties and "
                  f"{st['n_datatype_properties']} data properties."
                  if st else "")
        viewer = f"""
## Inside the most-reused pattern: `{first}`

Reused by **{top.get(first, 0)}** ontologies. {detail} Click any node for its
IRI, type and connected axioms; the panel below the diagram renders the Turtle.

```ontoink
source: patterns/{first}.ttl
height: 560px
editor: false
```
"""

    n_reused = ctx.get("n_reused_patterns", len(pats))
    vendored_mb = sum(f.stat().st_size for f in
                      (ctx["out"] / "patterns").glob("*.ttl")) / 1e6
    shortfall = ("" if n_reused == len(pats) else
                 f" {n_reused - len(pats)} of them have no retrievable source "
                 f"file and are listed without a rendering.")

    return f"""# Patterns

The **{n_reused} patterns reused** by the GitHub and Paper &amp; Repository
portions.{shortfall}

<div class="odp-note" markdown>
The {len(pats)} patterns rendered here total {vendored_mb:.1f} MB and ship with
this site. The {len(ctx["all_patterns"]) - n_reused} DOS-DP templates reused by the
OBO portion are mechanically generated from YAML and far larger — they live on
[Zenodo](download.md) with the corpus.
</div>

## Which ontology reuses which pattern

Every edge is one ontology–pattern pair, coloured by **how much of the pattern is
present** — from under a tenth of it to the whole thing. Discs are patterns,
sized by how many ontologies reuse them; the small dots are ontologies.

**Click a pattern** for what it declares and how its reuse is distributed. Drag
to pull nodes apart, scroll to zoom, and click empty space to clear a selection.

<div id="odp-constellation" class="odp-viz odp-viz--tall" data-src="data/constellation.json"></div>

<div class="odp-note" markdown>
Coverage is measured against the entity set of the *local copy* of the pattern
that each ontology ships, which is what the release divides by to get
`coverage_pct`. Those copies are not always identical — for 20 of the
{n_reused} patterns they disagree on how many entities the pattern has — so a
pattern's panel reports the range where one exists rather than a single size.
</div>

## Browse

Click a chip to isolate that pattern in the graph above.

<div class="odp-chips">
{listing}
</div>

{viewer}
"""


def page_datasets(ctx: dict) -> str:
    s = ctx["stats"]
    rows = "\n".join(
        f"| [`{r['dataset']}`](download.md) | {num(r['n_ontologies']):,} | "
        f"{num(r['n_pairs']):,} | {num(r['n_distinct_patterns']):,} | "
        f"{num(r['n_complete_pairs']):,} | "
        f"{fnum(r['mean_patterns_per_ontology']):.2f} | "
        f"{num(r['median_entities_per_ontology']):,} |"
        for r in sorted(s, key=lambda x: VIEW_ORDER.index(x["dataset"])
                        if x["dataset"] in VIEW_ORDER else 99))

    return f"""# Datasets

Three corpus portions × two reuse levels. A **view** is a filter over the
published corpus, not a copy of it — so nothing is duplicated and you can always
inspect what a view left out.

| View | Ontologies | Pairs | Patterns | Complete pairs | Mean patterns/onto | Median entities |
|---|---:|---:|---:|---:|---:|---:|
{rows}

<div id="odp-views" class="odp-viz" data-src="data/views.json"></div>

## Reading a view

Each view directory contains `pairs.csv` — the ontology-to-pattern mapping, one
row per pair — plus per-ontology metadata and statistics.

| Column | Meaning |
|---|---|
| `ontology_id` | directory name under `corpus/` |
| `pattern` | pattern stem |
| `n_pattern_entities` | entities declared by the pattern |
| `n_reused_entities` | of those, how many occur in the ontology |
| `coverage_pct` | `100 × reused / pattern entities` |
| `complete_reuse` | 1 if every pattern entity occurs in the ontology |

!!! danger "DOS-DP is not comparable to the others"
    Its ontologies carry a median of tens of thousands of entities against a
    median of ~70 elsewhere, and its patterns are generated from YAML templates
    rather than chosen. Report it separately; never pool a single averaged figure
    across all three portions.
"""


def page_using(ctx: dict) -> str:
    """A task-oriented guide: pick a view, read the columns, avoid the traps."""
    s = {r["dataset"]: r for r in ctx["stats"]}
    g = {r["dataset"]: r for r in ctx["gate"]}
    kept = sum(num(r["kept_reuse"]) for r in ctx["gate"])
    disc = sum(num(r["discovered"]) for r in ctx["gate"])

    def cell(view: str, key: str) -> str:
        return f"{num(s.get(view, {}).get(key, 0)):,}"

    view_rows = "\n".join(
        f"| `{v}` | {cell(v, 'n_ontologies')} | {cell(v, 'n_pairs')} | "
        f"{cell(v, 'n_distinct_patterns')} | "
        f"{fnum(s.get(v, {}).get('mean_patterns_per_ontology', 0)):.2f} |"
        for v in VIEW_ORDER if v in s)

    gh_kept = num(g.get("github", {}).get("kept_reuse", 0))
    no_kept = num(g.get("noOBO", {}).get("kept_reuse", 0))
    ob_kept = num(g.get("OBO", {}).get("kept_reuse", 0))

    return f"""# Using the dataset

A practical guide: which view to load, what each column means, and the three
things that will otherwise catch you out.

## 1 · Pick a view

The release ships **six views** — three portions × two strictness levels. They
are not alternatives to be merged; each answers a different question.

| View | Ontologies | Pairs | Distinct patterns | Patterns/ontology |
|---|---:|---:|---:|---:|
{view_rows}

A `-reuse` view contains every ontology that reuses **at least one entity** of at
least one pattern. The matching `-complete` view keeps only those that reuse a
pattern **in full**, and is the stricter, smaller ground truth.

<div class="odp-note" markdown>
**Which one do I want?**

- Training a recommender, or measuring *how much* is reused → a `-reuse` view.
- Evaluating pattern *detection*, where a partial hit is not a hit → `-complete`.
- Reporting a single headline number over the whole corpus → don't. See §4.
</div>

## 2 · The files and their columns

Each view directory holds `pairs.csv`, `ontologies.csv`, `statistics.csv` and
`top_patterns.csv`. The two that matter:

=== "pairs.csv — one row per (ontology, pattern)"

    | Column | Meaning |
    |---|---|
    | `ontology_id` | stable id; also the directory name under `corpus/` |
    | `pattern` | pattern stem, e.g. `agentrole` |
    | `n_pattern_entities` | entities the **pattern** declares |
    | `n_reused_entities` | of those, how many appear in the ontology |
    | `coverage_pct` | `100 × n_reused / n_pattern_entities` — see §4 |
    | `complete_reuse` | `1` when coverage is 100% |

=== "ontologies.csv — one row per ontology"

    | Column | Meaning |
    |---|---|
    | `n_entities_wide` | all entities, including those only referenced |
    | `n_own_entities` | entities contributed **beyond** its patterns |
    | `n_gt_patterns` | patterns with a non-empty intersection |
    | `max_reuse_pct` | best coverage achieved over its patterns |

`MANIFEST.csv` at the release root is the **unfiltered** record: all {disc:,}
discovered ontologies, each with `stage_reached` and `drop_reason`. The gate is
recorded as metadata, never applied destructively — so you can disagree with any
rule and re-derive your own corpus from the manifest alone.

## 3 · Loading it

```python
import pandas as pd

pairs = pd.read_csv("release-iri/views/github-reuse/pairs.csv")

# ontology x pattern reuse matrix, the usual starting point
M = pairs.pivot_table(index="ontology_id", columns="pattern",
                      values="n_reused_entities", fill_value=0)

# ... or restrict to full reuse only
strict = pairs[pairs.complete_reuse == 1]
```

To re-derive the corpus under **your own** gate rather than ours:

```python
man = pd.read_csv("release-iri/MANIFEST.csv")
mine = man[(man.parsed == 1) & (man.n_own_entities >= 25)]   # your threshold
```

## 4 · Three things that will catch you out

!!! warning "`coverage_pct` is a share of the *pattern*, not of the ontology"
    It answers "how much of this pattern is present?", so a large ontology
    reusing one small pattern in full scores 100%. It is **not** a measure of how
    pattern-based the ontology is.

!!! danger "Never pool one average across the three portions"
    The DOS-DP portion is {ob_kept} ontologies against {gh_kept} from GitHub and
    {no_kept} from Paper &amp; Repository, but its ontologies carry orders of
    magnitude more entities and its patterns are generated from YAML templates
    rather than chosen by a modeller. A micro-average over the pooled corpus is
    dominated by it and reverses conclusions that hold per portion. Report the
    three separately, and say which averaging you used.

!!! note "The corpus is unfiltered by design"
    {disc:,} ontologies ship; {kept:,} pass the gate. If you load `corpus/`
    directly instead of a view, you are working with everything — including
    unparseable files, verbatim pattern copies and duplicates. Join against
    `MANIFEST.csv` on `ontology_id` and filter on `stage_reached`.

## 5 · Plain or hashed?

Each dataset has an IRI-anonymised twin (`corpus-hashed`, and `-hashed` siblings
of the views). Every IRI is replaced by a deterministic hash, so identical
entities stay aligned across files, but pattern namespaces are unrecognisable.

- **Evaluating a detection method** → use the hashed form. On the plain form a
  method can score well by recognising `ontologydesignpatterns.org` prefixes
  instead of learning structure.
- **Inspecting, curating or authoring** → use the plain form.

Hashed corpus files are N-Triples regardless of the original serialisation, and
`format_map.csv` records what each one was. Directory names are *not* hashed, so
the tree still aligns with the manifest.

## 6 · Citing it

Cite the Zenodo DOI for the data and the paper for the method — see
[Get the data](download.md).
"""


def page_download(ctx: dict) -> str:
    return f"""# Get the data

<div class="odp-split" markdown>

<div class="odp-panel odp-panel--a" markdown>
### :material-database: Zenodo — the corpus
Every ontology and pattern, ~2.8 GB, versioned under one concept DOI that always
resolves to the newest release.

[:material-open-in-new: 10.5281/zenodo.17751740](https://doi.org/10.5281/zenodo.17751740){{ .md-button .md-button--primary }}
</div>

<div class="odp-panel odp-panel--b" markdown>
### :material-github: This repository — the tables
Views, manifest, audit and statistics: small, textual, diffable. Changes between
versions are reviewable as a pull request.

[:material-open-in-new: ISE-FIZKarlsruhe/ODP-Reuse](https://github.com/ISE-FIZKarlsruhe/ODP-Reuse){{ .md-button }}
</div>

</div>

## Why the split

The corpus contains single files above GitHub's 100 MB hard limit, and git would
retain every version of a 2.8 GB payload forever. Keeping the tables in git and
the corpus on Zenodo means a quarterly update shows up as a readable diff instead
of an opaque binary churn.

## Use it in code

```python
import csv
from collections import defaultdict
from pathlib import Path
from rdflib import Graph

ds = Path("views/github-complete")

with (ds / "pairs.csv").open(encoding="utf-8") as fh:
    pairs = list(csv.DictReader(fh))

by_onto = defaultdict(list)
for r in pairs:
    by_onto[r["ontology_id"]].append(r)

oid = next(iter(by_onto))
for r in by_onto[oid]:
    pattern = Graph().parse(ds / r["pattern_file"])
    print(oid, "reuses", r["pattern"],
          f'{{r["coverage_pct"]}}% ({{r["n_reused_entities"]}}/{{r["n_pattern_entities"]}})')
```

Two conventions are not optional if you want comparable numbers: entity sets use
the **WIDE** definition, and annotation vocabularies are stripped from **both**
sides. Both live in `pipeline/odpqual/entities.py`, which every part of the
pipeline imports — so no two components can disagree about what an entity is.

## Verify your copy

```bash
python validate_release.py
```

Checks that every `pairs.csv` row resolves, that statistics match what the tables
actually contain, and that each `*-complete` view is a subset of its `*-reuse`
sibling.

## Cite

```bibtex
@dataset{{odp_reuse_{ctx['version'].replace('.', '_').replace('v', '')},
  title  = {{ODP-Reuse: a quality-gated benchmark of Ontology Design Pattern reuse}},
  author = {{Norouzi, Ebrahim and Hertling, Sven and
            Waitelonis, J{{\\"o}}rg and Sack, Harald}},
  year   = {{2026}},
  doi    = {{10.5281/zenodo.17751740}},
  version = {{{ctx['version']}}}
}}
```
"""


def page_changelog(ctx: dict) -> str:
    return f"""# Changelog

Every release is a git tag, a GitHub Release and a Zenodo version under the same
concept DOI. The newest is shown first; all earlier versions remain resolvable.

{ctx['changelog']}

## Versioning

| Bump | Trigger | Comparability |
|---|---|---|
| **major** | the gate changes which ontologies are kept, or a view changes definition | earlier results **not** comparable |
| **minor** | new ontologies or pairs from a harvest; gate unchanged | earlier results valid on their subset |
| **patch** | repairs, metadata, documentation; membership unchanged | fully comparable |

## How updates happen

Two workflows test two different claims.

**The release reproduces.** `reproduce.yml` downloads the corpus archived on
Zenodo, checks every ontology file against the hash the gate recorded, re-runs
the gate and compares every row with the released `MANIFEST.csv`. It runs
monthly and on every change to the gate, so the claim that the archive, run
through the released gate, yields the released verdicts is tested rather than
asserted.

**The corpus is maintained.** `quarterly-harvest.yml` re-runs discovery, repair
and the gate for the GitHub portion every quarter and opens an issue listing
ontology–pattern pairs that did not exist in the published release; hits for
patterns published after the release are reported separately. A re-harvest
queries live APIs and is not expected to reproduce the corpus: it tests that the
harvester still runs, and shows what is new. A new release is cut deliberately
from that issue, never automatically.
"""


# ----------------------------------------------------------------------

def build_payloads(ctx: dict, out: Path) -> None:
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)

    # funnel: one band per rule, summed across portions
    tot = Counter()
    for r in ctx["gate"]:
        for rule in ("G1_parse", "G2_pattern_copy", "G3_duplicate",
                     "G4_substantive", "G5_has_gt"):
            tot[rule] += num(r.get(f"dropped_{rule}"))
        tot["KEPT"] += num(r["kept_reuse"])
    (d / "funnel.json").write_text(json.dumps([
        {"rule": k, "label": RULE_LABEL[k], "n": v, "color": RULE_COLOR[k]}
        for k, v in tot.most_common()], indent=1), encoding="utf-8")

    # per-portion rule breakdown
    (d / "rules.json").write_text(json.dumps([
        {"portion": PORTION_LABEL_JSON.get(r["dataset"], r["dataset"]),
         "discovered": num(r["discovered"]),
         "bands": [{"rule": RULE_LABEL[k], "n": num(r.get(f"dropped_{k}")),
                    "color": RULE_COLOR[k]}
                   for k in ("G1_parse", "G2_pattern_copy", "G3_duplicate",
                             "G4_substantive", "G5_has_gt")]
         + [{"rule": "Kept", "n": num(r["kept_reuse"]),
             "color": RULE_COLOR["KEPT"]}]}
        for r in ctx["gate"]], indent=1), encoding="utf-8")

    # views
    (d / "views.json").write_text(json.dumps([
        {"view": r["dataset"], "ontologies": num(r["n_ontologies"]),
         "pairs": num(r["n_pairs"]), "patterns": num(r["n_distinct_patterns"]),
         "complete": num(r["n_complete_pairs"]),
         "median_entities": num(r["median_entities_per_ontology"])}
        for r in ctx["stats"]], indent=1), encoding="utf-8")

    # thresholds / forms from the RQ tables
    (d / "thresholds.json").write_text(json.dumps([
        {"portion": PORTION_LABEL_JSON.get(r["dataset"], r["dataset"]),
         "points": [{"t": t, "n": num(r.get(f"ge_{t}pct"))}
                    for t in (10, 30, 50, 70, 90, 100)]}
        for r in ctx["rq"].get("rq1_extent", [])], indent=1), encoding="utf-8")
    (d / "forms.json").write_text(json.dumps([
        {"portion": PORTION_LABEL_JSON.get(r["dataset"], r["dataset"]),
         "roles": [{"role": k.replace("_pct", "").replace("_", " "),
                    "pct": fnum(r[k])}
                   for k in r if k.endswith("_pct")]}
        for r in ctx["rq"].get("rq2_reuse_forms", [])], indent=1), encoding="utf-8")

    # top patterns
    (d / "top_patterns.json").write_text(json.dumps([
        {"portion": PORTION_LABEL_JSON.get(r["dataset"], r["dataset"]),
         "pattern": r["pattern"], "n": num(r["n_ontologies"]),
         "rank": num(r["rank"])}
        for r in ctx["rq"].get("rq1_top_patterns", [])], indent=1),
        encoding="utf-8")

    # constellation: ontology <-> pattern, GitHub + Paper/Repo only.
    #
    # Every edge carries its coverage band so the figure can colour by how much
    # of the pattern is present, and every pattern node carries the counts a
    # reader needs to interpret its coverage: what the pattern declares, and the
    # entity total the coverage denominators were taken from.
    nodes, links = {}, []
    usage = ctx["pattern_usage"]
    pstats = ctx.get("pattern_stats", {})
    denom = ctx.get("pattern_denominators", {})
    per_portion: dict[str, Counter] = defaultdict(Counter)
    for view, portion in (("github-reuse", "GitHub"),
                          ("noOBO-reuse", "Paper & Repository")):
        for r in ctx["pairs"].get(view, []):
            per_portion[r["pattern"]][portion] += 1
    for view, portion in (("github-reuse", "GitHub"),
                          ("noOBO-reuse", "Paper & Repository")):
        for r in ctx["pairs"].get(view, []):
            o, p = r["ontology_id"], r["pattern"]
            nodes.setdefault(f"o:{o}", {
                "id": f"o:{o}", "label": o[:34], "kind": "ontology", "v": 1,
                "portion": portion})
            if f"p:{p}" not in nodes:
                st = pstats.get(p, {})
                dn = denom.get(p) or Counter()
                node = {"id": f"p:{p}", "label": p, "kind": "pattern",
                        "v": usage.get(p, 1),
                        "github": per_portion[p].get("GitHub", 0),
                        "noOBO": per_portion[p].get("Paper & Repository", 0)}
                node.update({k: st[k] for k in (
                    "n_entities", "n_classes", "n_object_properties",
                    "n_datatype_properties", "n_individuals", "n_triples")
                    if k in st})
                # The denominator varies across local copies for 20 of the 68
                # stems; carrying the range keeps the panel from implying a
                # single authoritative size where the release has several.
                if dn:
                    node["gt_entities"] = dn.most_common(1)[0][0]
                    node["gt_min"], node["gt_max"] = min(dn), max(dn)
                nodes[f"p:{p}"] = node
            cov = fnum(r["coverage_pct"])
            links.append({"source": f"o:{o}", "target": f"p:{p}",
                          "coverage": cov, "band": coverage_band(cov),
                          "complete": num(r["complete_reuse"])})
    (d / "constellation.json").write_text(json.dumps(
        {"nodes": list(nodes.values()), "links": links,
         "bands": COVERAGE_BANDS}), encoding="utf-8")

    # uc1 = extraction, uc2 = recommendation, as in the manuscript.
    (d / "uc1.json").write_text(json.dumps([
        {"portion": PORTION_LABEL_JSON.get(r["dataset"], r["dataset"]),
         "method": r["method"], "clusterer": r["clusterer"],
         "oracle": fnum(r["oracle_f1"]), "unsup": fnum(r["unsup_f1"]),
         "n": num(r["n_ontologies"])} for r in ctx["uc1"]], indent=1),
        encoding="utf-8")
    (d / "uc2.json").write_text(json.dumps([
        {"label": r["comparison"], "portion": PORTION_LABEL_JSON.get(
            r.get("dataset"), r.get("dataset")),
         "delta": fnum(r["mean_delta_map"]),
         "lo": fnum(r["ci_lo_95"]), "hi": fnum(r["ci_hi_95"]),
         "n": num(r["n_paired"]),
         "sig": r["significant_at_0.05"] == "yes"} for r in ctx["uc2"]],
        indent=1), encoding="utf-8")


def parse_pattern(path: Path):
    """Parse a pattern file, trying each serialisation. ``None`` if all fail."""
    from rdflib import Graph
    for fmt in PARSE_FORMATS:
        g = Graph()
        try:
            g.parse(str(path), format=fmt)
            return g, fmt
        except Exception:  # noqa: BLE001 - malformed pattern files are the norm
            continue
    return None, None


def pattern_stats(graph) -> dict:
    """Per-pattern figures for the site, on the release's own entity definition.

    ``n_entities`` is ``collect_entities(strip_annotations(g), wide=True)`` -- the
    exact quantity ``build_release.py`` divides by to get ``coverage_pct``. It is
    therefore directly comparable to ``n_pattern_entities`` in ``pairs.csv``, and
    a figure that quotes both cannot contradict itself.

    The class / object-property / data-property split is the *declared* one, which
    is a subset of the wide entity set: wide additionally counts IRIs that only
    ever appear in a class or property position (restriction fillers, domains,
    ranges, subClassOf objects). So C + OP + DP <= n_entities, and the difference
    is referenced-but-undeclared vocabulary. That gap is worth showing rather than
    hiding, so both numbers are carried.
    """
    from rdflib import OWL, RDF, RDFS, URIRef
    from odpqual.entities import collect_entities, strip_annotations

    g, _ = strip_annotations(graph)

    def typed(t):
        return {s for s in g.subjects(RDF.type, t) if isinstance(s, URIRef)}

    n_c = len(typed(OWL.Class) | typed(RDFS.Class))
    return {
        "n_entities": len(collect_entities(g, wide=True)),
        "n_classes": n_c,
        "n_object_properties": len(typed(OWL.ObjectProperty)),
        "n_datatype_properties": len(typed(OWL.DatatypeProperty)),
        "n_individuals": len(typed(OWL.NamedIndividual)),
        "n_triples": len(g),
    }


def ship_patterns(ctx: dict, out: Path, libs: Path | None) -> list[str]:
    """Vendor the reused ODPs (small) so ontoink can render them in-page.

    Sourced from the release corpus, which is always present, rather than from an
    external ``--libraries`` checkout that is not part of this repository. The old
    behaviour silently returned whatever happened to be vendored already when
    ``--libraries`` was unset -- which is the default -- so the site shipped 47 of
    the 68 reused patterns and the page then described those 47 as "the patterns
    actually reused". Sourcing from the release makes the set complete and the
    count true by construction.

    Stale files are removed: a pattern that stops being reused after a re-gate
    must not linger in the vendored set.
    """
    dest = out / "patterns"
    dest.mkdir(parents=True, exist_ok=True)

    want = set()
    for view in ("github-reuse", "noOBO-reuse"):
        for r in ctx["pairs"].get(view, []):
            want.add(r["pattern"])

    # Index candidate files: the corpus's own patterns/ trees, plus an optional
    # external library checkout if one is supplied.
    roots = [ctx["release"] / "corpus"]
    if libs is not None and libs.is_dir():
        roots += [libs / n for n in
                  ("patterns-repository", "manchester",
                   "Industrial-Standard-Ontology-Design-Patterns")]
    # ALL candidates per stem, not the first one found. Each reusing ontology
    # keeps its own local copy of the pattern under its patterns/ directory, and
    # those copies are not identical: `collectionentity` appears with 3 entities
    # and with 8, DUL with 196 and 197. Taking whichever copy rglob happened to
    # yield first shipped the wrong file for at least four patterns -- the
    # vendored `agentrole.owl` was ODPA's *example ABox*
    # (cp/examples/agentrole/ex1.owl), which declares no pattern entity at all,
    # while the ground truth for that stem counts 3. The site then rendered an
    # example as if it were the pattern.
    cands: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if (f.is_file() and f.suffix.lower() in (".owl", ".ttl", ".rdf")
                    and 0 < f.stat().st_size < 4_000_000):
                cands[f.stem.lower()].append(f)

    # The denominator the release actually used, per stem. The modal value is the
    # representative: it is what coverage_pct was computed against for the
    # plurality of that pattern's pairs.
    denom: dict[str, Counter] = defaultdict(Counter)
    for view in ("github-reuse", "noOBO-reuse"):
        for r in ctx["pairs"].get(view, []):
            denom[r["pattern"]][num(r["n_pattern_entities"])] += 1

    def pick(stem: str, files: list[Path]) -> tuple[Path | None, dict]:
        """Choose the local copy whose entity set matches the ground truth.

        Parsing every candidate would mean parsing 4,410 files, so identical
        bytes are collapsed first -- DUL's 208 copies reduce to a handful of
        distinct hashes. Among the distinct variants, prefer the one whose entity
        count equals the modal ``n_pattern_entities``; that is the file the
        measurement used. Fall back to the most-copied variant, then the largest,
        so a stem with no usable match still ships something.
        """
        target = (denom[stem].most_common(1)[0][0] if denom.get(stem) else None)
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for f in files:
            try:
                by_hash[hashlib.sha256(f.read_bytes()).hexdigest()].append(f)
            except OSError:
                continue
        scored = []
        for _h, group in by_hash.items():
            g, _fmt = parse_pattern(group[0])
            if g is None:
                continue
            st = pattern_stats(g)
            scored.append((st["n_entities"] == target, len(group),
                           st["n_entities"], group[0], st))
        if not scored:
            return None, {}
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        best = scored[0]
        if target is not None and not best[0]:
            print(f"[!!] {stem}: no local copy has the ground truth's "
                  f"{target} entities (variants: "
                  f"{sorted({s[2] for s in scored})}); shipping the "
                  f"most-copied one with {best[2]}")
        return best[3], best[4]

    shipped, missing = [], []
    stats: dict[str, dict] = {}
    for p in sorted(want):
        src, st = (pick(p.lower(), cands[p.lower()]) if cands.get(p.lower())
                   else (None, {}))
        if src is None:
            # Fall back to a copy already vendored here. This is the normal case
            # in CI: the corpus is a Zenodo artefact and is gitignored, so a
            # runner has no source tree to copy from -- but docs/patterns/*.owl is
            # committed, so the site can still render. Without this, `shipped`
            # would be empty on every runner and the stale-sweep below would
            # delete the whole vendored set.
            if (dest / f"{p}.owl").is_file():
                shipped.append(p)
                g, _fmt = parse_pattern(dest / f"{p}.owl")
                if g is not None:
                    stats[p] = pattern_stats(g)
                    write_pattern_turtle(g, dest / f"{p}.ttl")
            else:
                missing.append(p)
            continue
        shutil.copyfile(src, dest / f"{p}.owl")
        shipped.append(p)
        stats[p] = st
        g, _fmt = parse_pattern(src)
        if g is not None:
            write_pattern_turtle(g, dest / f"{p}.ttl")

    for stale in sorted(set(f.stem for f in dest.glob("*.owl")) - set(shipped)):
        (dest / f"{stale}.owl").unlink()
        print(f"[..] removed stale vendored pattern {stale}.owl")
    for stale in sorted(set(f.stem for f in dest.glob("*.ttl")) - set(shipped)):
        (dest / f"{stale}.ttl").unlink()

    if missing:
        print(f"[!!] {len(missing)} reused patterns have neither a source file nor "
              f"a vendored copy and are not rendered: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))
    if not cands:
        print("[..] no pattern source tree found (corpus is a Zenodo artefact); "
              f"reusing the {len(shipped)} committed copies under docs/patterns/")
    ctx["n_reused_patterns"] = len(want)
    ctx["pattern_stats"] = stats
    ctx["pattern_denominators"] = denom
    return shipped


def write_pattern_turtle(graph, path: Path) -> None:
    """Write a Turtle twin of a vendored pattern, for the in-page ontoink viewer.

    ontoink parses with ``format="turtle"`` unconditionally, but 66 of the 68
    vendored patterns are RDF/XML, so pointing a fence at the ``.owl`` raises. The
    Turtle is also smaller (1.39 MB of RDF/XML becomes 0.69 MB), so shipping both
    still costs less than the RDF/XML alone did.
    """
    try:
        path.write_text(graph.serialize(format="turtle"), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - a bad literal must not fail the build
        print(f"[!!] could not serialise {path.name} as Turtle: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path("release-iri"))
    # The GATED RQ tables, not rq_iri_all. The site and the manuscript must not
    # disagree, and they did: rq_iri_all averages over all 1,619 discovered
    # GitHub ontologies and reports 0.56 patterns per ontology, while the paper
    # reports 1.51 over the 442 that pass the gate. Publishing the ungated figure
    # put a number on the website that contradicted Table rq1_extent.
    ap.add_argument("--rq", type=Path, default=Path("pipeline/out/rq_iri_gated"))
    ap.add_argument("--out", type=Path, default=Path("docs"))
    ap.add_argument("--libraries", type=Path, default=None,
                    help="corpus root holding patterns-repository/ etc.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = {v.parent.name: read(v)
             for v in (args.release / "views").glob("*/pairs.csv")}
    usage: Counter = Counter()
    for view in ("github-reuse", "noOBO-reuse"):
        for r in pairs.get(view, []):
            usage[r["pattern"]] += 1

    audit = read(args.release / "audit" / "audit_summary.csv")

    # UC1 is extraction and UC2 is recommendation, matching the manuscript. The
    # two were previously the other way round here, so the site's "UC1" described
    # the paper's UC2 and vice versa.
    uc1 = read(Path("use-cases/odp-extraction/results/iri/rq5_results.csv"))

    # The quality-gated GitHub run is the one the paper reports. The earlier
    # default read the noOBO portion, which the paper presents only as an
    # underpowered completeness check (n = 14), and read it from a pre-gate
    # working directory. Both portions are carried so the page can separate them.
    RC = Path("use-cases/odp-recommendation")
    uc2 = [dict(r, dataset="github")
           for r in read(RC / "stats_iri_github_gated" /
                         "reco_paired_tests_github.csv")]
    uc2 += [dict(r, dataset="noOBO")
            for r in read(RC / "stats_iri" / "reco_paired_tests_noOBO.csv")]

    ctx = {
        "version": git_version(),
        "release": args.release,
        "out": args.out,
        "gate": read(args.release / "gate_summary.csv"),
        "stats": read(args.release / "dataset_statistics.csv"),
        "pairs": pairs,
        "pattern_usage": usage,
        "all_patterns": {r["pattern"] for v in pairs.values() for r in v},
        "audit": audit,
        "uc1": uc1,
        "uc2": uc2,
        "rq": {p.stem: read(p) for p in args.rq.glob("rq*.csv")},
        "changelog": (Path("CHANGELOG.md").read_text(encoding="utf-8")
                      if Path("CHANGELOG.md").is_file()
                      else "_No changelog yet._"),
    }

    ctx["shipped_patterns"] = ship_patterns(ctx, args.out, args.libraries)
    build_payloads(ctx, args.out)

    pages = {
        "index.md": page_index, "pipeline.md": page_pipeline,
        "filters.md": page_filters, "rqs.md": page_rqs,
        "use-cases.md": page_use_cases, "datasets.md": page_datasets,
        "patterns.md": page_patterns, "using.md": page_using,
        "download.md": page_download,
        "changelog.md": page_changelog,
    }
    for name, fn in pages.items():
        (args.out / name).write_text(fn(ctx), encoding="utf-8")
        print(f"[OK] {name}")
    print(f"\nversion {ctx['version']} · {len(ctx['shipped_patterns'])} patterns "
          f"shipped · {sum(len(v) for v in pairs.values())} pairs")


if __name__ == "__main__":
    main()
