"""Units of analysis, and the CSV writer every analysis uses.

Four libraries, four different notions of "a pattern".  Rather than pick one and
hide the choice, every table in this package is produced for an explicit unit and
names it in a ``unit`` column or in the filename.

    file             every parsed RDF file
    content          one representative per (library, sha256)
    stem             one representative per (library, case-folded stem)
    stem_global      one representative per case-folded stem across ALL libraries
    primary          ``file`` minus ODPA's shipped dependencies
    primary_content  ``primary``, then deduplicated by (library, sha256)

``primary_content`` is the default headline unit: it is the closest honest
answer to "how many distinct patterns does this library contribute".

``stem_global`` exists for one job: joining to ``pairs.csv``.  The reuse evidence
keys a pattern by stem alone, so a table built on any other unit either
double-counts a stem that has several representatives or drops a stem whose
representative was deduplicated away.  On ``stem_global`` every pair in
``pairs.csv`` is counted at most once and exactly once if the stem exists at all.
"""

from __future__ import annotations

import csv
from pathlib import Path

UNITS: tuple[str, ...] = ("file", "content", "stem", "stem_global",
                          "primary", "primary_content")
DEFAULT_UNIT = "primary_content"

#: preference order when one representative must stand for several files
_LIB_PREF = ("ODPA", "MANCHESTER", "ISODP", "DOSDP")


def parsed_only(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("parsed")]


def _rep(group: list[dict]) -> dict:
    """Representative of a group: a primary file if any, then shortest path.

    Deterministic, so the same input always yields the same tables.
    """
    def key(r: dict) -> tuple:
        lib = r["library"]
        return (_LIB_PREF.index(lib) if lib in _LIB_PREF else len(_LIB_PREF),
                r["role"] != "primary", len(r["rel_path"]), r["rel_path"])
    return sorted(group, key=key)[0]


def select(records: list[dict], unit: str = DEFAULT_UNIT) -> list[dict]:
    if unit not in UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")
    recs = parsed_only(records)
    if unit == "file":
        pass
    elif unit == "primary":
        recs = [r for r in recs if r["role"] == "primary"]
    elif unit == "content":
        recs = _dedup(recs, lambda r: (r["library"], r["sha256"]))
    elif unit == "stem":
        recs = _dedup(recs, lambda r: (r["library"], r["stem"].casefold()))
    elif unit == "stem_global":
        recs = _dedup(recs, lambda r: r["stem"].casefold())
    elif unit == "primary_content":
        recs = _dedup([r for r in recs if r["role"] == "primary"],
                      lambda r: (r["library"], r["sha256"]))
    return sorted(recs, key=lambda r: (r["library"], r["rel_path"]))


def _dedup(recs: list[dict], keyfn) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in recs:
        buckets.setdefault(keyfn(r), []).append(r)
    return [_rep(v) for v in buckets.values()]


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    """Write ``rows`` to ``path``.  Columns default to the union, first-seen order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    columns.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})
    return path


def pct(n: int, d: int, nd: int = 1) -> float:
    return round(100.0 * n / d, nd) if d else 0.0
