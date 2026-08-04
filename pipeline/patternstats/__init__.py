"""patternstats -- descriptive statistics about the four integrated ODP libraries.

This package characterises the *pattern libraries themselves* (not the ontologies
that reuse them).  Four analyses, each in its own module and each writing CSVs:

    toplevel.py   upper/top-level ontology alignment (imports vs referenced IRIs)
    overlap.py    inter-library overlap: content hash, stem, entity-set Jaccard
    domains.py    subject-domain classification + cross-tab against actual reuse
    shape.py      size/shape per library (C / OP / DP / axiom proxies / connectivity)

All four read a single cached extraction produced by ``extract.py``, so the
libraries are parsed exactly once.

Run everything:

    python -m pipeline.patternstats --data-root <MultiSource_server> \
        --out pipeline/out/patternstats

Entity-set semantics are imported from ``pipeline/odpqual/entities.py`` so that
every entity count here is byte-compatible with the numbers in MANIFEST.csv and
the RQ tables.
"""

__all__ = [
    "libraries",
    "extract",
    "vocab",
    "toplevel",
    "overlap",
    "domains",
    "shape",
    "reuse",
]

__version__ = "1.0.0"
