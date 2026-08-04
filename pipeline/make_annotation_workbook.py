#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_annotation_workbook.py -- build an Excel workbook for adjudicating
label-only matches, with dropdown validation and the context needed to decide
each item without opening any file.

Why a workbook and not the CSV: a spreadsheet can carry a constrained dropdown,
so a verdict is never mistyped, plus conditional colouring by agreement stratum,
frozen headers, and a live progress counter. The CSV remains the interchange
format -- `export` writes verdicts back to it.

Sheets
  Adjudicate   one row per item, contested first, dropdown in the verdict column
  Guide        what each verdict means, with worked examples
  Strata       how the three models split, and what that implies
  Progress     live counts and the current precision estimate

Usage
    python make_annotation_workbook.py --out out/adjudication
    python make_annotation_workbook.py --out out/adjudication --export   # xlsx -> csv
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
except ImportError:
    raise SystemExit("pip install openpyxl")

P = ["gpt", "gemini", "claude", "deepseek"]
VERDICTS = ["reuse", "not_reuse", "unclear"]

# Stratum -> (label, fill colour). Ordered by how much human judgement they need.
STRATA = {
    "split_even":     ("even split", "FFD6D6"),
    "split_majority": ("majority split", "FFE9CC"),
    "incomplete":     ("incomplete", "E6E0FF"),
    "unanimous":      ("unanimous", "D8F5E3"),
}
TODO_FILL = {"judge": "FFD6D6", "spot-check": "FFF3B0", "deferred": "E8E8EE"}


def stratum(v: list[str]) -> str:
    """Agreement stratum over the machine raters."""
    if len(v) < len(P):
        return "incomplete"
    s = set(v)
    if len(s) == 1:
        return "unanimous"
    return "split_even" if len(s) >= 3 or _even(v) else "split_majority"


def _even(v: list[str]) -> bool:
    import collections as _c
    c = _c.Counter(v).most_common()
    return len(c) > 1 and c[0][1] == c[1][1]


def plan(rows: list[dict], n_spot: int, seed: int) -> None:
    """Assign each row a `todo` class.

    STRATIFIED ADJUDICATION. Contested and incomplete items are judged in full:
    the machine majority is unsafe there, and no automatic rule decides which
    rater to believe. The unanimous stratum is validated on a random sample --
    if every spot-check confirms the machine verdict, a Wilson bound limits how
    much could have been missed in the remainder, which is reportable. If the
    spot-check fails, the deferred rows must be judged after all.
    """
    import random
    rng = random.Random(seed)
    unan = [r for r in rows if r["stratum"] == "unanimous"]
    rng.shuffle(unan)
    spot = {id(r) for r in unan[:n_spot]}
    for r in rows:
        if r["stratum"] == "unanimous":
            r["todo"] = "spot-check" if id(r) in spot else "deferred"
        else:
            r["todo"] = "judge"


def load(out: Path):
    items = {r["id"]: r for r in
             csv.DictReader((out / "items.csv").open(encoding="utf-8"))}
    mv = out / "machine_verdicts.csv"
    mach = {r["id"]: r for r in csv.DictReader(mv.open(encoding="utf-8"))} \
        if mv.is_file() else {}
    # carry over any verdicts already entered
    hs = out / "human_sample.csv"
    prior = {}
    if hs.is_file():
        for r in csv.DictReader(hs.open(encoding="utf-8")):
            if r.get("human_verdict", "").strip():
                prior[r["id"]] = (r["human_verdict"].strip(),
                                  r.get("human_note", ""))
    return items, mach, prior


def build(out: Path, n_spot: int = 30, seed: int = 0) -> Path:
    items, mach, prior = load(out)
    rows = []
    for i, it in items.items():
        m = mach.get(i, {})
        v = [m.get(f"{p}_verdict") for p in P if m.get(f"{p}_verdict")]
        st = stratum(v)
        maj = m.get("majority", "")
        rows.append({
            "stratum": st, "id": i,
            "dataset": it["dataset"], "ontology": it["ontology"],
            "pattern": it["pattern"],
            "n_label_only": it["n_label_only"], "n_iri": it["n_iri"],
            "matched": it["matched"],
            "pattern_classes": it["pattern_classes"],
            "ontology_sample": it["ontology_sample"],
            **{p: m.get(f"{p}_verdict", "") for p in P},
            "majority": maj,
            **{f"{p}_reason": (m.get(f"{p}_reason") or "")[:180] for p in P},
            "todo": "",
            "human_verdict": prior.get(i, ("", ""))[0],
            "human_note": prior.get(i, ("", ""))[1],
        })
    plan(rows, n_spot, seed)
    sorder = {"split_even": 0, "split_majority": 1, "incomplete": 2,
              "unanimous": 3}
    torder = {"judge": 0, "spot-check": 1, "deferred": 2}
    rows.sort(key=lambda r: (torder[r["todo"]], sorder[r["stratum"]],
                             r["ontology"]))
    active = [r for r in rows if r["todo"] != "deferred"]
    deferred = [r for r in rows if r["todo"] == "deferred"]

    wb = Workbook()

    # ---------------------------------------------------------- Adjudicate
    ws = wb.active
    ws.title = "Adjudicate"
    cols = ["human_verdict", "human_note", "todo", "stratum", "ontology", "pattern",
            "matched", "pattern_classes", "ontology_sample",
            "n_label_only", "n_iri", *P, "majority",
            *[f"{p}_reason" for p in P], "id", "dataset"]
    widths = {"human_verdict": 15, "human_note": 26, "todo": 12, "stratum": 15,
              "ontology": 40, "pattern": 22, "matched": 34,
              "pattern_classes": 40, "ontology_sample": 44,
              **{f"{p}_reason": 44 for p in P}, "id": 46, "dataset": 9}
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        cell = ws.cell(1, c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1B1033")
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.column_dimensions[get_column_letter(c)].width = \
            widths.get(cols[c - 1], 13)
    ws.freeze_panes = "C2"
    ws.row_dimensions[1].height = 30

    for r in active:
        ws.append([r.get(c, "") for c in cols])

    # dropdown on the verdict column -- a verdict can never be mistyped
    dv = DataValidation(type="list", formula1=f'"{",".join(VERDICTS)}"',
                        allow_blank=True, showDropDown=False)
    dv.error = "Choose reuse, not_reuse or unclear."
    dv.errorTitle = "Invalid verdict"
    dv.prompt = ("reuse = the ontology re-models this pattern\n"
                 "not_reuse = coincidental name collision\n"
                 "unclear = evidence genuinely does not decide")
    dv.promptTitle = "Verdict"
    ws.add_data_validation(dv)
    dv.add(f"A2:A{len(active) + 1}")

    # colour each row band by stratum so the contested ones are obvious
    sc = cols.index("stratum") + 1
    tc = cols.index("todo") + 1
    for i, r in enumerate(active, start=2):
        ws.cell(i, sc).fill = PatternFill("solid", fgColor=STRATA[r["stratum"]][1])
        ws.cell(i, sc).value = STRATA[r["stratum"]][0]
        fill = PatternFill("solid", fgColor=TODO_FILL[r["todo"]])
        ws.cell(i, tc).fill = fill
        for c in (1, 2):
            ws.cell(i, c).fill = fill
        for c in range(1, len(cols) + 1):
            ws.cell(i, c).alignment = Alignment(vertical="top", wrap_text=True)

    # green when answered, so progress is visible while scrolling
    ws.conditional_formatting.add(
        f"A2:A{len(active) + 1}",
        CellIsRule(operator="equal", formula=['"reuse"'],
                   fill=PatternFill("solid", fgColor="00C9A7")))
    ws.conditional_formatting.add(
        f"A2:A{len(active) + 1}",
        CellIsRule(operator="equal", formula=['"not_reuse"'],
                   fill=PatternFill("solid", fgColor="FF9F9F")))
    ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(active) + 1}"

    # ---------------------------------------------------------- Deferred
    d = wb.create_sheet("Deferred")
    d.append(["Accepted from machine consensus — judge ONLY if the spot-check fails"])
    d.append([f"All {len(P)} raters agreed on these {len(deferred)} items, and they "
              f"are not in the spot-check sample."])
    d.append(["If any spot-check row contradicts its machine verdict, this stratum "
              "is unsafe and every row here must be adjudicated."])
    d.append([])
    dcols = ["human_verdict", "ontology", "pattern", "matched", "majority", "id"]
    d.append(dcols)
    for r in deferred:
        d.append([r.get(c, "") for c in dcols])
    for col, w in (("A", 15), ("B", 40), ("C", 22), ("D", 34), ("E", 13), ("F", 46)):
        d.column_dimensions[col].width = w
    d["A1"].font = Font(bold=True, size=12)
    for c in "ABCDEF":
        d[f"{c}5"].font = Font(bold=True, color="FFFFFF")
        d[f"{c}5"].fill = PatternFill("solid", fgColor="1B1033")
    dv2 = DataValidation(type="list", formula1=f'"{",".join(VERDICTS)}"',
                         allow_blank=True, showDropDown=False)
    d.add_data_validation(dv2)
    dv2.add(f"A6:A{len(deferred) + 5}")

    # ---------------------------------------------------------- Plan
    pl = wb.create_sheet("Plan", 1)
    n_judge = sum(1 for r in active if r["todo"] == "judge")
    n_spotc = sum(1 for r in active if r["todo"] == "spot-check")
    for row in [
        ["Stratified adjudication plan"], [],
        ["Class", "Items", "Why"],
        ["judge", n_judge,
         "Raters disagree, or fewer than all raters answered. The machine "
         "majority is unsafe and no automatic rule decides which rater to "
         "believe. Judge every one."],
        ["spot-check", n_spotc,
         "Random sample of the unanimous stratum, seeded and reproducible. "
         "Validates whether machine unanimity can be trusted."],
        ["deferred", len(deferred),
         "Unanimous and not sampled. Accepted from machine consensus unless the "
         "spot-check fails. Listed on the Deferred sheet."],
        ["TOTAL to touch", n_judge + n_spotc,
         f"{100 * (n_judge + n_spotc) / max(len(rows), 1):.0f}% of {len(rows)} items"],
        [],
        ["Decision rule after the spot-check"],
        ["All spot-checks confirm the machine verdict",
         "Accept the deferred stratum. Report the Wilson upper bound on its error "
         "rate — with 30/30 that bound is 11.4%."],
        ["Any spot-check contradicts it",
         "The unanimous stratum is unsafe. Adjudicate the Deferred sheet in full."],
        [],
        ["How to report this"],
        ["\"All contested and incomplete items were adjudicated (n=" +
         str(n_judge) + "). The unanimous stratum (n=" +
         str(n_spotc + len(deferred)) + ") was validated on a seeded random "
         "sample of " + str(n_spotc) + "; agreement was X/" + str(n_spotc) +
         ", bounding the reuse rate in the remainder at Y%, so the machine "
         "verdict was accepted for the other " + str(len(deferred)) + " items.\""],
    ]:
        pl.append(row)
    for col, w in (("A", 30), ("B", 10), ("C", 74)):
        pl.column_dimensions[col].width = w
    pl["A1"].font = Font(bold=True, size=14)
    for c in "ABC":
        pl[f"{c}3"].font = Font(bold=True, color="FFFFFF")
        pl[f"{c}3"].fill = PatternFill("solid", fgColor="1B1033")
    pl["A9"].font = Font(bold=True)
    pl["A13"].font = Font(bold=True)
    for row in pl.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    # ---------------------------------------------------------- Guide
    g = wb.create_sheet("Guide")
    for row in [
        ["How to adjudicate a label-only match"], [],
        ["The matcher found entities sharing a NORMALISED LABEL with the "
         "pattern's classes, but the IRIs do NOT match."],
        ["Decide: did this ontology re-model the pattern, or is the shared name "
         "a coincidence?"], [],
        ["Verdict", "Means", "Typical signal"],
        ["reuse", "The ontology models the same conceptual structure and the "
                  "matched classes play the same role.",
         "Several structurally connected matches; the pattern's characteristic "
         "shape is recognisable in the ontology."],
        ["not_reuse", "A coincidental name collision.",
         "A common word (Event, Object, Role, Place, Agent) used in an unrelated "
         "sense; a single isolated match with no structural echo."],
        ["unclear", "The evidence genuinely does not decide it.",
         "Use sparingly — it is excluded from the precision denominator."], [],
        ["Columns"],
        ["matched", "Labels present in BOTH the pattern and the ontology."],
        ["pattern_classes", "What the pattern declares — is the ontology doing "
                            "this?"],
        ["ontology_sample", "The ontology's own entity names — what is it ABOUT?"],
        ["n_label_only", "How many entities matched by label alone. One is weak "
                         "evidence; several connected ones are strong."],
        ["n_iri", "Entities that also matched by IRI. Greater than zero means the "
                  "ontology already reuses the pattern verbatim in part."],
        ["gpt / gemini / claude", "What each model said, with its one-line reason. "
                                  "Advisory only — the models agree only slightly "
                                  "(Fleiss κ = 0.114)."], [],
        ["Order"],
        ["Rows are sorted hardest-first: 3-way splits, then 2-1 splits, then "
         "incomplete, then unanimous."],
        ["If you stop early you will still have judged the informative cases."],
    ]:
        g.append(row)
    g.column_dimensions["A"].width = 24
    g.column_dimensions["B"].width = 60
    g.column_dimensions["C"].width = 60
    g["A1"].font = Font(bold=True, size=14)
    for c in ("A6", "B6", "C6"):
        g[c].font = Font(bold=True)
    for row in g.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    # ---------------------------------------------------------- Strata
    s = wb.create_sheet("Strata")
    cnt = collections.Counter(r["stratum"] for r in rows)
    s.append(["Stratum", "Items", "Share", "Worth your time?"])
    advice = {
        "split_even": "JUDGE — the raters split evenly; nothing decides it.",
        "split_majority": "JUDGE — a rater dissented; these carry information.",
        "incomplete": "JUDGE — not every rater answered, so no majority exists.",
        "unanimous": "SPOT-CHECK a sample; see the caveat below.",
    }
    for k in ("split_even", "split_majority", "incomplete", "unanimous"):
        s.append([STRATA[k][0], cnt.get(k, 0),
                  f"{100 * cnt.get(k, 0) / max(len(rows), 1):.1f}%", advice[k]])
    s.append([])
    s.append(["Caveat on 'unanimous'"])
    s.append(["Three of the four raters converge substantively on not_reuse "
              "(pairwise raw agreement 76.6-92.1%); Gemini dissents at 46% "
              "reuse. Fleiss kappa = 0.063 UNDERSTATES that concordance, because "
              "when one category dominates, chance agreement is already ~0.90 "
              "and the correction collapses kappa (the prevalence paradox). Read "
              "raw agreement and the marginals alongside kappa, never kappa "
              "alone."])
    s.append([])
    s.append(["Note", "146 of the 147 unanimous items are unanimous NOT_REUSE, "
                      "and exactly one is unanimous REUSE. If that holds under "
                      "review, label-mode precision is low and iri+label should "
                      "not be the headline dataset."])
    for col, w in (("A", 34), ("B", 10), ("C", 10), ("D", 62)):
        s.column_dimensions[col].width = w
    for c in "ABCD":
        s[f"{c}1"].font = Font(bold=True, color="FFFFFF")
        s[f"{c}1"].fill = PatternFill("solid", fgColor="1B1033")
    for row in s.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    # ---------------------------------------------------------- Progress
    pr = wb.create_sheet("Progress")
    n = len(active)
    pr.append(["Live progress — recalculates as you fill in the Adjudicate sheet"])
    pr.append([])
    for label, formula in [
        ("Items total", f"={n}"),
        ("Adjudicated", f'=COUNTIF(Adjudicate!A2:A{n + 1},"<>")'),
        ("Remaining", f'={n}-COUNTIF(Adjudicate!A2:A{n + 1},"<>")'),
        ("reuse", f'=COUNTIF(Adjudicate!A2:A{n + 1},"reuse")'),
        ("not_reuse", f'=COUNTIF(Adjudicate!A2:A{n + 1},"not_reuse")'),
        ("unclear", f'=COUNTIF(Adjudicate!A2:A{n + 1},"unclear")'),
        ("Decided (excl. unclear)",
         f'=COUNTIF(Adjudicate!A2:A{n + 1},"reuse")'
         f'+COUNTIF(Adjudicate!A2:A{n + 1},"not_reuse")'),
        ("Label-mode precision so far",
         f'=IFERROR(COUNTIF(Adjudicate!A2:A{n + 1},"reuse")/'
         f'(COUNTIF(Adjudicate!A2:A{n + 1},"reuse")'
         f'+COUNTIF(Adjudicate!A2:A{n + 1},"not_reuse")),"—")'),
    ]:
        pr.append([label, formula])
    pr.append([])
    pr.append(["Confidence intervals, kappas and the human-vs-machine anchor are "
               "computed by:  python adjudicate_labels.py score --out "
               "out/adjudication"])
    pr.column_dimensions["A"].width = 32
    pr.column_dimensions["B"].width = 22
    pr["A1"].font = Font(bold=True, size=12)
    pr["A9"].font = Font(bold=True)
    pr["B9"].number_format = "0.000"

    path = out / "adjudication_workbook.xlsx"
    wb.save(path)
    return path


def export(out: Path) -> Path:
    """Write verdicts from the workbook back into human_sample.csv."""
    wb = load_workbook(out / "adjudication_workbook.xlsx", data_only=True)
    ws = wb["Adjudicate"]
    hdr = [c.value for c in ws[1]]
    iv, ii, ino = (hdr.index("human_verdict"), hdr.index("id"),
                   hdr.index("human_note"))
    verdicts = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[iv]:
            verdicts[row[ii]] = (str(row[iv]).strip(),
                                 str(row[ino] or "").strip())
    src = out / "human_sample.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    for r in rows:
        if r["id"] in verdicts:
            r["human_verdict"], r["human_note"] = verdicts[r["id"]]
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {len(verdicts)} verdicts written back to {src.name}")
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--export", action="store_true",
                    help="write verdicts from the workbook back to the CSV")
    args = ap.parse_args()
    if args.export:
        export(args.out)
    else:
        p = build(args.out)
        print(f"[OK] {p}")
        print("Fill the 'human_verdict' column (dropdown). Contested rows first.")
        print("When done:  python make_annotation_workbook.py --out "
              f"{args.out} --export")


if __name__ == "__main__":
    main()
