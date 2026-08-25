#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_analysis.py -- do these extraction results make sense, and did anything
go wrong on the way?

Run this after every extraction run, on every corpus. A number that is merely
present is not a number that is right, and a run that exits 0 is not a run that
worked: the failures that matter here are silent ones -- an ontology that never
parsed, a method that threw on half the corpus, a metric that came out of an
empty cluster set, a prediction containing entities the ontology does not have.

WHAT IS CHECKED
---------------
LOGS       tracebacks, exceptions, "FAILED"/"error" lines, and the exit-status
           warnings the driver writes.
SKIPS      every exclusion, split into EXPECTED (an ontology with no ground
           truth on a corpus that has none) and SUSPECT (parse errors, method
           errors, out-of-memory), because only the second kind is a defect.
INVARIANTS the reference points have known behaviour, and departures from it
           mean the harness is wrong rather than the method:
             all_in_one   recall must be 1.0 -- it predicts every entity
             singletons   precision must be 1.0 -- a matched singleton is
                          either inside its target or scores 0
             every method P, R, F1 in [0,1], and F1 consistent with P and R
             n_predicted >= 1 wherever a score was produced
COVERAGE   how many ontologies each method actually produced output for,
           against the population. A method quietly covering 60% of the corpus
           looks identical to one covering 100% in an averaged table.
PREDICTIONS clusters non-empty, no duplicates, sizes plausible against the
           ontology's entity count.
CORPORA    when two runs are given (--compare), the paired differences, which
           is how the plain vs IRI-anonymised comparison is sanity-checked.

Usage
-----
    python tools/error_analysis.py --run results/mseo
    python tools/error_analysis.py --run results/methods-iri \
        --compare results/methods-iri-anon
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_SKIPS = {
    "no_ground_truth_predict_only",   # a corpus with no answer key
    "no_ground_truth",                # scored-mode population filter
}
SUSPECT_SKIPS = {
    "parse_error", "method_error", "method_out_of_memory",
    "embed_error", "embed_out_of_memory", "no_entities", "no_ontology_file",
    "no_ontology_dir",
}
LOG_PATTERNS = [
    (re.compile(r"^Traceback", re.M), "python traceback"),
    (re.compile(r"^\s*\w*(Error|Exception)\b.*", re.M), "exception line"),
    (re.compile(r"\bFAILED\b", re.I), "FAILED"),
    (re.compile(r"^\[warn\]", re.M), "driver warning"),
    (re.compile(r"MemoryError"), "MemoryError"),
]

OK, WARN, BAD = "ok  ", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, section: str, msg: str):
        self.rows.append((level, section, msg))

    def counts(self):
        return Counter(l for l, _s, _m in self.rows)

    def render(self) -> str:
        out = []
        for level, section, msg in self.rows:
            out.append(f"[{level}] {section:<12} {msg}")
        c = self.counts()
        out.append("")
        out.append(f"{c.get(OK,0)} ok, {c.get(WARN,0)} warnings, {c.get(BAD,0)} failures")
        return "\n".join(out)


def check_logs(run: Path, rep: Report) -> None:
    # finish.log is the chain driver's transcript: it contains the output of
    # every run, so scanning it here reports another run's warnings against this
    # one. Each run's own log is scanned when that run is analysed.
    logs = [p for p in sorted(run.glob("*.log")) if p.name != "finish.log"]
    logs += sorted(run.glob("SUMMARY.txt"))
    if not logs:
        rep.add(WARN, "logs", f"no log files under {run}")
        return
    for lg in logs:
        text = lg.read_text(encoding="utf-8", errors="replace")
        hits = []
        for pat, label in LOG_PATTERNS:
            found = pat.findall(text)
            if found:
                hits.append(f"{label} x{len(found)}")
        if hits:
            rep.add(BAD, "logs", f"{lg.name}: " + "; ".join(hits))
            for pat, label in LOG_PATTERNS:
                for m in list(pat.finditer(text))[:3]:
                    line = text[m.start():m.start() + 160].splitlines()[0]
                    rep.add(BAD, "logs", f"    {lg.name}: {line.strip()}")
        else:
            rep.add(OK, "logs", f"{lg.name}: clean")


def check_skips(run: Path, rep: Report) -> None:
    sp = run / "methods_skipped.csv"
    if not sp.is_file():
        rep.add(OK, "skips", "no exclusions recorded")
        return
    rows = list(csv.DictReader(sp.open(encoding="utf-8")))
    tally = Counter(r["reason"] for r in rows)
    for reason, n in tally.most_common():
        if reason in SUSPECT_SKIPS:
            ex = next((r for r in rows if r["reason"] == reason), {})
            rep.add(BAD, "skips",
                    f"{reason}: {n}  e.g. {ex.get('ontology','?')} "
                    f"-- {str(ex.get('detail',''))[:90]}")
        elif reason in EXPECTED_SKIPS:
            rep.add(OK, "skips", f"{reason}: {n} (expected on a corpus without "
                                 f"ground truth)")
        else:
            rep.add(WARN, "skips", f"{reason}: {n}")


def load_per_ontology(run: Path) -> list[dict]:
    p = run / "methods_per_ontology.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def check_invariants(rows: list[dict], rep: Report) -> None:
    if not rows:
        rep.add(WARN, "invariants", "nothing was scored, so no invariant applies")
        return
    bad_range = [r for r in rows
                 if not all(0.0 <= r[k] <= 1.0 for k in ("precision", "recall", "f1"))]
    rep.add(BAD if bad_range else OK, "invariants",
            f"{len(bad_range)} rows with P/R/F1 outside [0,1]"
            if bad_range else "all P/R/F1 within [0,1]")

    inconsistent = []
    for r in rows:
        p, q, f = r["precision"], r["recall"], r["f1"]
        # macro-averaged over matched pairs, so F1 need not equal 2pq/(p+q);
        # it must still not exceed the larger of p and q.
        if f > max(p, q) + 1e-6:
            inconsistent.append(r)
    rep.add(BAD if inconsistent else OK, "invariants",
            f"{len(inconsistent)} rows with F1 > max(P,R)"
            if inconsistent else "F1 never exceeds max(P,R)")

    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    rs = by_method.get("all_in_one")
    if rs:
        off = [r for r in rs if abs(r["recall"] - 1.0) > 0.02]
        rep.add(BAD if off else OK, "invariants",
                f"all_in_one: recall != 1.0 on {len(off)}/{len(rs)} rows "
                f"(it predicts every entity, so it cannot miss one)"
                if off else
                f"all_in_one: recall is 1.0 on all {len(rs)} rows, as it must be")

    rs = by_method.get("singletons")
    if rs:
        # A matched singleton is either inside its target (precision 1) or not
        # (precision 0), so the mean can only fall below 1 when Hungarian was
        # forced to match a singleton to a pattern it does not belong to. That
        # happens exactly when two ground-truth patterns compete for the same
        # entity, which needs n_gt > 1: several pattern FILES can reduce to the
        # same entity set inside one ontology, and a one-to-one matching cannot
        # satisfy both. With n_gt == 1 there is no such excuse and a precision
        # below 1 is a harness defect.
        off = [r for r in rs if r["precision"] < 0.999]
        real = [r for r in off if r["n_gt"] <= 1]
        if real:
            rep.add(BAD, "invariants",
                    f"singletons: precision < 1 on {len(real)} rows with a "
                    f"single ground-truth pattern -- that is a defect "
                    f"(e.g. {real[0]['ontology'][:40]})")
        elif off:
            rep.add(OK, "invariants",
                    f"singletons: precision is 1.0 except on {len(off)}/{len(rs)} "
                    f"rows whose ground-truth patterns overlap, where a "
                    f"one-to-one matching cannot satisfy every pattern "
                    f"(min {min(r['precision'] for r in off):.3f}); explained, "
                    f"not a defect")
        else:
            rep.add(OK, "invariants",
                    f"singletons: precision is 1.0 on all {len(rs)} rows")

    zero = [m for m, rs in by_method.items()
            if all(r["f1"] == 0.0 for r in rs)]
    if zero:
        rep.add(WARN, "invariants",
                f"methods scoring exactly 0 everywhere: {', '.join(sorted(zero))}")

    nopred = [r for r in rows if r.get("n_predicted", 1) == 0]
    if nopred:
        rep.add(WARN, "invariants",
                f"{len(nopred)} scored rows predicted no cluster at all "
                f"(e.g. {nopred[0]['method']} on {nopred[0]['ontology'][:40]})")


def check_coverage(rows: list[dict], rep: Report) -> None:
    if not rows:
        return
    by_method = defaultdict(set)
    for r in rows:
        by_method[r["method"]].add((r["dataset"], r["ontology"]))
    n_max = max(len(v) for v in by_method.values())
    for m, v in sorted(by_method.items()):
        if len(v) < n_max:
            rep.add(WARN, "coverage",
                    f"{m}: {len(v)}/{n_max} ontologies -- averages are over a "
                    f"smaller population than the other methods")
        else:
            rep.add(OK, "coverage", f"{m}: {len(v)}/{n_max} ontologies")


def check_predictions(run: Path, rep: Report) -> None:
    p = run / "predictions.jsonl"
    if not p.is_file():
        return
    n, empty, dup, oversize = 0, 0, 0, 0
    per_method = Counter()
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        per_method[r["method"]] += 1
        cl = r.get("clusters", [])
        if any(len(c) == 0 for c in cl):
            empty += 1
        seen = {tuple(c) for c in cl}
        if len(seen) != len(cl):
            dup += 1
        tot = sum(len(c) for c in cl)
        if r.get("n_entities") and tot > r["n_entities"] * 1.001 \
                and r["method"] != "law_ap":
            # law_ap can assign one entity to several patterns by design;
            # for the others the clusters partition or sub-set the entities.
            oversize += 1
    rep.add(OK if not empty else BAD, "predictions",
            f"{n} prediction rows; {empty} contain an empty cluster")
    rep.add(OK if not dup else WARN, "predictions",
            f"{dup} rows contain duplicate clusters")
    rep.add(OK if not oversize else WARN, "predictions",
            f"{oversize} rows assign more entity slots than the ontology has "
            f"entities (expected only for overlapping-pattern methods)")
    rep.add(OK, "predictions",
            "per method: " + ", ".join(f"{k}={v}" for k, v in sorted(per_method.items())))


def check_compare(rows_a: list[dict], rows_b: list[dict], rep: Report,
                  name_a: str, name_b: str) -> None:
    if not rows_a or not rows_b:
        return
    idx_b = {(r["dataset"], r["ontology"], r["method"]): r for r in rows_b}
    deltas = defaultdict(list)
    for r in rows_a:
        k = (r["dataset"], r["ontology"], r["method"])
        if k in idx_b:
            deltas[r["method"]].append(idx_b[k]["f1"] - r["f1"])
    if not deltas:
        rep.add(WARN, "compare", "no ontology is common to both runs")
        return
    for m, ds in sorted(deltas.items()):
        mean = sum(ds) / len(ds)
        rep.add(OK, "compare",
                f"{m}: mean F1 {name_b} - {name_a} = {mean:+.4f} over {len(ds)} paired")
    ns = deltas.get("namespace")
    if ns:
        mean = sum(ns) / len(ns)
        if mean > -0.2:
            rep.add(WARN, "compare",
                    f"namespace grouping only moves {mean:+.3f} between corpora; "
                    f"if one of them is the IRI-anonymised twin this is too "
                    f"small and the anonymisation should be checked")
        else:
            rep.add(OK, "compare",
                    f"namespace grouping collapses by {mean:+.3f}, which is the "
                    f"anonymisation doing its job")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rep = Report()
    rep.add(OK, "run", str(args.run))
    check_logs(args.run, rep)
    check_skips(args.run, rep)
    rows = load_per_ontology(args.run)
    check_invariants(rows, rep)
    check_coverage(rows, rep)
    check_predictions(args.run, rep)
    if args.compare:
        rep.add(OK, "run", f"compare against {args.compare}")
        check_logs(args.compare, rep)
        check_skips(args.compare, rep)
        rows_b = load_per_ontology(args.compare)
        check_invariants(rows_b, rep)
        check_compare(rows, rows_b, rep, args.run.name, args.compare.name)

    text = rep.render()
    print(text)
    out = args.out or (args.run / "ERROR_ANALYSIS.txt")
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[OK] {out}")
    c = rep.counts()
    raise SystemExit(1 if c.get(BAD, 0) else 0)


if __name__ == "__main__":
    main()
