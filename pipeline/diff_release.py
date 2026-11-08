#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diff_release.py -- compare a fresh harvest against the published release and
report what changed, as the monthly harvest's run summary.

Answers, for a new harvest:

  * which ontologies are NEW (discovered now, absent from the release)
  * which ontology-pattern PAIRS are new on ontologies that pass the gate --
    the headline number, because only those can enter a view
  * which ontologies have more or fewer pairs in this harvest than in the
    release (a pattern was published, withdrawn, or repaired)
  * which release ontologies this harvest did NOT RETURN. Code search keeps at
    most 1,000 results per query and ranks them differently on every run, so
    a file it does not return is usually still there (88 of 100 sampled after
    run 42239149156). The release keeps every one of them.
  * how the quality gate's verdict shifted, per rule

Sets GitHub Actions outputs `has_changes`, `new_pairs`, `new_ontologies` and
writes `REPORT.md` for the run summary. Nothing here changes the release;
merge_harvest.py adds the new ontologies.

Only the datasets present in the NEW manifest are compared. A harvest rebuilds
the GitHub portion only; the noOBO and OBO portions are not re-discovered, so
their absence from a harvest says nothing about them.

Optional inputs make the report say what the harvest actually did:

  --discovery      the finder's results CSV; hits are counted per pattern set,
                   and hits for patterns added after the release are listed
                   on their own, so "GitHub changed" stays apart from "the
                   ODPA published new patterns"
  --search-status  the finder's per-IRI status CSV; any IRI not searched makes
                   the report say INCOMPLETE
  --audit          audit_filters.py's output; the report states how many
                   removals were re-verified rather than claiming all were

Usage
-----
    python diff_release.py --old release/MANIFEST.csv \\
        --new harvest/out/MANIFEST.csv --out harvest/diff \\
        [--discovery report/results.csv] [--search-status report/results_github_status.csv] \\
        [--audit harvest/out/audit]
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path


def load(p: Path) -> dict[tuple[str, str], dict]:
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as fh:
        return {(r["dataset"], r["ontology_id"]): r for r in csv.DictReader(fh)}


def gt(r: dict) -> int:
    return int(r.get("n_gt_patterns") or 0)


def complete(r: dict) -> int:
    return int(r.get("n_complete_patterns") or 0)


def kept(r: dict) -> bool:
    return not r.get("drop_reason")


def read_rows(p: Path | None) -> list[dict]:
    if not p or not p.is_file():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def discovery_section(rows: list[dict]) -> list[str]:
    gh = [r for r in rows if (r.get("source") or "") == "github"]
    if not gh:
        return []
    by_set = Counter((r.get("pattern_set") or "unlabelled") for r in gh)
    L = ["## Discovery", "",
         "GitHub code-search hits per pattern set: "
         + ", ".join(f"{s}: {n} hits" for s, n in sorted(by_set.items())) + ".", ""]
    new = [r for r in gh if r.get("pattern_set") == "new"]
    if new:
        repos: dict[str, set[str]] = {}
        files: Counter = Counter()
        for r in new:
            repos.setdefault(r["pattern_iri"], set()).add(r.get("match_label") or "")
            files[r["pattern_iri"]] += 1
        L += ["## Hits for patterns added after the release", "",
              "These patterns were published in the ODPA repository after the "
              "release was frozen. Their hits are candidate NEW reuse; the gate "
              "counts below include them, so they are listed here to keep them "
              "apart from changes on GitHub. A pattern whose IRI is its own "
              "source ontology's IRI will match that ontology (a self-match, "
              "which the gate's G2 rule removes).", "",
              "| pattern IRI | repositories | files |", "|---|---:|---:|"]
        for iri in sorted(files):
            L.append(f"| `{iri}` | {len(repos[iri])} | {files[iri]} |")
        L.append("")
    return L


def coverage_section(status: list[dict]) -> tuple[list[str], bool]:
    if not status:
        return [], False
    missing = [s for s in status if s.get("status") == "not_searched"]
    if not missing:
        return [f"- pattern IRIs searched on GitHub: **{len(status)} of {len(status)}**", ""], False
    by_set = Counter((s.get("pattern_set") or "unlabelled") for s in missing)
    return ([f"> **INCOMPLETE harvest.** {len(missing)} of {len(status)} pattern IRIs were "
             f"not searched ({', '.join(f'{k}: {v}' for k, v in sorted(by_set.items()))}). "
             "Ontologies that reuse only those patterns are missing from this harvest, so "
             "'not returned' and 'fewer pairs' below are not evidence of change.", ""],
            True)


def audit_footer(audit_dir: Path | None) -> str:
    if not audit_dir or not audit_dir.is_dir():
        return ("Generated by `.github/workflows/harvest.yml`. Removals were not "
                "re-verified in this run.")
    verdicts: Counter = Counter()
    for f in sorted(audit_dir.glob("removed_*.csv")):
        for r in read_rows(f):
            verdicts[(r.get("verified") or "").strip() or "unchecked"] += 1
    total = sum(verdicts.values())
    checked = verdicts["ok"] + verdicts["DISPUTED"]
    scope = (f"All {total} removals were" if checked == total
             else f"{checked} of {total} removals were")
    rest = "" if checked == total else "; the rest were not sampled"
    return (f"Generated by `.github/workflows/harvest.yml`. {scope} re-verified "
            f"against the source files ({verdicts['DISPUTED']} disputed){rest}. The run fails "
            f"if any re-verified removal is disputed.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--old-pairs", type=Path)
    ap.add_argument("--new-root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--discovery", type=Path)
    ap.add_argument("--search-status", type=Path)
    ap.add_argument("--audit", type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    old_all, new = load(args.old), load(args.new)
    rebuilt = sorted({ds for ds, _ in new})
    old = {k: r for k, r in old_all.items() if k[0] in set(rebuilt)}
    not_rebuilt = sorted({ds for ds, _ in old_all} - set(rebuilt))
    coverage, incomplete = coverage_section(read_rows(args.search_status))

    added = sorted(set(new) - set(old))
    gone = sorted(set(old) - set(new))
    common = sorted(set(old) & set(new))

    gained = [(k, gt(old[k]), gt(new[k])) for k in common if gt(new[k]) > gt(old[k])]
    lost = [(k, gt(old[k]), gt(new[k])) for k in common if gt(new[k]) < gt(old[k])]
    newly_kept = [k for k in common if kept(new[k]) and not kept(old[k])]
    newly_dropped = [k for k in common if not kept(new[k]) and kept(old[k])]

    # Only pairs on ontologies that pass the gate can enter a view, so only
    # those are counted as new. Run 42239149156 counted 316, of which 58 were
    # on dropped ontologies.
    new_pairs = (sum(gt(new[k]) for k in added if kept(new[k]))
                 + sum(b - a for k, a, b in gained if kept(new[k])))
    new_complete = (sum(complete(new[k]) for k in added)
                    + sum(max(0, complete(new[k]) - complete(old[k]))
                          for k in common))

    rules_old = Counter(r["drop_reason"].split(":", 1)[0] or "KEPT"
                        for r in old.values())
    rules_new = Counter(r["drop_reason"].split(":", 1)[0] or "KEPT"
                        for r in new.values())

    has_changes = bool(added or gone or gained or lost
                       or newly_kept or newly_dropped)

    L = ["# Harvest diff", "", *coverage,
         f"- corpus ({', '.join(rebuilt) or 'nothing rebuilt'}): **{len(old)} → {len(new)}** ontologies",
         *([f"- not rebuilt by this harvest: {', '.join(not_rebuilt)} "
            f"(release rows unchanged, not compared)"] if not_rebuilt else []),
         f"- **new ontology–pattern pairs on gate-passing ontologies: {new_pairs}**",
         f"- new complete-reuse pairs: {new_complete}",
         f"- newly discovered ontologies: {len(added)} "
         f"({sum(1 for k in added if kept(new[k]))} pass the gate)",
         f"- release ontologies not returned by this harvest: {len(gone)} (kept in the release)",
         f"- more pairs than in the release: {len(gained)} · fewer: {len(lost)} "
         f"(the release keeps its pairs)",
         f"- newly passing the gate: {len(newly_kept)} · newly dropped: {len(newly_dropped)}",
         ""]
    L += discovery_section(read_rows(args.discovery))

    if added:
        L += ["## Newly discovered ontologies", "",
              "| dataset | ontology | patterns | complete | gate |",
              "|---|---|---:|---:|---|"]
        for k in added[:60]:
            r = new[k]
            L.append(f"| {k[0]} | `{k[1]}` | {gt(r)} | {complete(r)} | "
                     f"{r['drop_reason'].split(':', 1)[0] or '**kept**'} |")
        if len(added) > 60:
            L.append(f"\n_…and {len(added) - 60} more; see `new_ontologies.csv`._")
        L.append("")

    if gained:
        L += ["## More pairs than in the release", "",
              "| dataset | ontology | before | after |", "|---|---|---:|---:|"]
        for (ds, o), a, b in gained[:40]:
            L.append(f"| {ds} | `{o}` | {a} | {b} |")
        L.append("")

    if gone:
        L += ["## Not returned by this harvest", "",
              "These release ontologies were not among the files this harvest's "
              "searches returned. That is not evidence that they are gone: code "
              "search caps and re-ranks its results, and after run 42239149156, 88 "
              "of 100 sampled files were still at the same path. The release keeps "
              "all of them; see `not_returned.csv`.", "",
              *[f"- `{o}` ({ds})" for ds, o in gone[:40]],
              *([f"\n_…and {len(gone) - 40} more._"] if len(gone) > 40 else []), ""]

    L += ["## Gate verdict shift", "",
          "| rule | before | after | Δ |", "|---|---:|---:|---:|"]
    for rule in sorted(set(rules_old) | set(rules_new)):
        a, b = rules_old.get(rule, 0), rules_new.get(rule, 0)
        L.append(f"| {rule} | {a} | {b} | {b - a:+d} |")
    L += ["", "---", "", audit_footer(args.audit), ""]

    (args.out / "REPORT.md").write_text("\n".join(L), encoding="utf-8")

    def dump(name, rows, cols):
        if not rows:
            return
        with (args.out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    dump("new_ontologies.csv", [new[k] for k in added],
         ["dataset", "ontology_id", "n_pattern_files", "n_gt_patterns",
          "n_complete_patterns", "drop_reason"])
    dump("gained_ground_truth.csv",
         [{"dataset": k[0], "ontology_id": k[1], "before": a, "after": b}
          for k, a, b in gained], ["dataset", "ontology_id", "before", "after"])
    dump("not_returned.csv", [old[k] for k in gone],
         ["dataset", "ontology_id", "n_gt_patterns", "drop_reason"])

    print(f"new pairs on gate-passing ontologies: {new_pairs}   new ontologies: "
          f"{len(added)}   not returned (kept): {len(gone)}")

    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write(f"has_changes={'true' if has_changes else 'false'}\n")
            fh.write(f"new_pairs={new_pairs}\n")
            fh.write(f"new_ontologies={len(added)}\n")
            fh.write(f"incomplete={'true' if incomplete else 'false'}\n")


if __name__ == "__main__":
    main()
