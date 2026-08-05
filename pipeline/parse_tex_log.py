"""Attribute每 box-overflow warning in a LaTeX log to the source file it came from.

pdflatex reports "Overfull \\hbox ... at lines 45--46" without saying which file
those lines are in, and a manuscript split across a dozen \\input files gives a
dozen candidates. The file is recoverable: the log records every file it opens as
`(path` and every close as `)`, so tracking that nesting while scanning forward
tells you which file was current when the warning was emitted.

It also de-duplicates: a warning appearing once per pdflatex pass is one problem,
not three.

    python pipeline/parse_tex_log.py paper-iri-iswc/build.log
    python pipeline/parse_tex_log.py paper-iri-iswc/build.log --min-pt 2
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

OPEN = re.compile(r"\((?:\./)?([^()\s]*\.(?:tex|sty|cls|def|cfg|bst|clo|fd))")
BOX = re.compile(
    r"^(?P<kind>Over|Under)full \\(?P<dir>[hv])box "
    r"(?:\((?P<pt>[0-9.]+)pt too (?:wide|high|short)\)|\(badness (?P<bad>[0-9]+)\))"
    r"(?P<where>.*)$")
LINES = re.compile(r"(?:at lines?|detected at line) (\d+)(?:--(\d+))?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--min-pt", type=float, default=1.0,
                    help="ignore overflows smaller than this (default 1pt: "
                         "below that nothing is visible on the page)")
    ap.add_argument("--underfull", action="store_true",
                    help="also report underfull boxes (usually cosmetic)")
    args = ap.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")

    stack: list[str] = []
    found: dict[tuple, dict] = {}

    for raw in text.split("\n"):
        # Track file opens/closes. pdflatex can put several on one line.
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "(":
                m = OPEN.match(raw, i)
                if m:
                    stack.append(m.group(1))
                    i = m.end()
                    continue
                stack.append("?")
            elif ch == ")":
                if stack:
                    stack.pop()
            i += 1

        m = BOX.match(raw.strip())
        if not m:
            continue
        if m.group("kind") == "Under" and not args.underfull:
            continue
        pt = float(m.group("pt") or 0)
        if m.group("pt") and pt < args.min_pt:
            continue
        lm = LINES.search(m.group("where") or "")
        # The innermost open .tex file is the one being typeset.
        src = next((f for f in reversed(stack) if f.endswith(".tex")), "?")
        key = (src, lm.group(0) if lm else m.group("where"), m.group("pt"),
               m.group("kind"), m.group("dir"))
        rec = found.setdefault(key, {
            "file": src, "pt": pt, "kind": m.group("kind"),
            "dir": m.group("dir"),
            "lines": (int(lm.group(1)), int(lm.group(2) or lm.group(1)))
                     if lm else None,
            "n": 0})
        rec["n"] += 1

    items = sorted(found.values(), key=lambda r: -r["pt"])
    over = [r for r in items if r["kind"] == "Over"]
    print(f"== {args.log}: {len(over)} distinct overfull box(es) "
          f">= {args.min_pt}pt ==\n")
    if not over:
        print("[OK] nothing overflows the text block.")
    for r in over:
        loc = (f"lines {r['lines'][0]}"
               + (f"--{r['lines'][1]}" if r["lines"][1] != r["lines"][0] else "")
               ) if r["lines"] else "?"
        print(f"  {r['pt']:8.2f}pt  \\{r['dir']}box  {r['file']:<34} {loc}"
              f"   (seen {r['n']}x)")

    if args.underfull:
        und = [r for r in items if r["kind"] == "Under"]
        print(f"\n  {len(und)} distinct underfull box(es) (usually cosmetic)")

    # Show the offending source line so the fix is obvious.
    print()
    for r in over:
        if not r["lines"]:
            continue
        p = args.log.parent / r["file"]
        if not p.is_file():
            continue
        src_lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        a = max(0, r["lines"][0] - 1)
        b = min(len(src_lines), r["lines"][1])
        snippet = " ".join(x.strip() for x in src_lines[a:b])
        print(f"  {r['file']}:{r['lines'][0]}  ({r['pt']:.0f}pt)")
        print(f"     {snippet[:200]}")


if __name__ == "__main__":
    main()
