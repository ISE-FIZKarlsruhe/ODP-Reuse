"""Structural checks on a LaTeX manuscript, for machines without a TeX toolchain.

This does not replace a compile. It catches the failures that a compile would
catch loudly and that are easy to introduce when editing sections in isolation:

  * ``\\ref``/``\\eqref``/``\\cref`` to a label that is defined nowhere
  * duplicate ``\\label``
  * ``\\input``/``\\include`` of a file that does not exist
  * a table or figure file that nothing inputs (an orphan)
  * ``\\cite`` keys absent from the .bib
  * unbalanced braces, and ``\\begin``/``\\end`` environments that do not match
  * a literal ``%`` that is not escaped inside a number or after a word, which
    silently comments out the rest of the line

Usage
-----
    python pipeline/check_latex.py paper-iri-iswc
    python pipeline/check_latex.py paper-iri-iswc --main main.tex
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# A `%` that follows a digit or letter and is not escaped: almost always a
# forgotten \% in text like "25.4% of the corpus", which comments out the rest.
BAD_PCT = re.compile(r'(?<![\\%])(?<=[0-9A-Za-z\)])%')

LABEL = re.compile(r'\\label\{([^}]*)\}')
REF = re.compile(r'\\(?:ref|eqref|autoref|cref|Cref)\{([^}]*)\}')
INPUT = re.compile(r'\\(?:input|include)\{([^}]*)\}')
CITE = re.compile(r'\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}')
BIBSTYLE = re.compile(r'\\bibliography\{([^}]*)\}')
BEGIN = re.compile(r'\\begin\{([^}]*)\}')
END = re.compile(r'\\end\{([^}]*)\}')


def strip_comments(text: str) -> str:
    """Drop comment tails so commented-out blocks are not analysed."""
    out = []
    for line in text.split("\n"):
        i, esc = 0, False
        cut = None
        while i < len(line):
            c = line[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", type=Path)
    ap.add_argument("--main", default="main.tex")
    args = ap.parse_args()

    root: Path = args.paper
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    tex = sorted(p for p in root.rglob("*.tex"))
    if not tex:
        sys.exit(f"no .tex files under {root}")

    problems: list[str] = []
    notes: list[str] = []

    raw = {p: p.read_text(encoding="utf-8", errors="replace") for p in tex}
    live = {p: strip_comments(t) for p, t in raw.items()}

    # ---------------------------------------------------------------- labels
    labels: Counter = Counter()
    for p, t in live.items():
        for m in LABEL.findall(t):
            labels[m] += 1
    for name, n in sorted(labels.items()):
        if n > 1:
            problems.append(f"duplicate \\label{{{name}}} defined {n} times")

    refs: dict[str, set[Path]] = {}
    for p, t in live.items():
        for m in REF.findall(t):
            refs.setdefault(m, set()).add(p)
    for name, where in sorted(refs.items()):
        if name not in labels:
            w = ", ".join(sorted(str(x.relative_to(root)) for x in where))
            problems.append(f"\\ref{{{name}}} has no \\label  (cited in {w})")

    # ---------------------------------------------------------------- inputs
    inputs: set[str] = set()
    for p, t in live.items():
        for m in INPUT.findall(t):
            inputs.add(m)
            cand = [root / m, root / f"{m}.tex"]
            if not any(c.is_file() for c in cand):
                problems.append(
                    f"\\input{{{m}}} in {p.relative_to(root)} -> file not found")

    # orphan section/table files: exist but nothing inputs them
    main_path = root / args.main
    inputted = {(root / (m if m.endswith('.tex') else m + '.tex')).resolve()
                for m in inputs}
    for p in tex:
        if p.resolve() == main_path.resolve():
            continue
        if p.resolve() not in inputted:
            notes.append(f"orphan (nothing \\inputs it): {p.relative_to(root)}")

    # ---------------------------------------------------------------- graphics
    # A missing image is a hard compile error, and the paths here contain spaces
    # and repeated "figures/" segments, which makes them easy to get wrong.
    GRAPHICS = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}')
    EXT = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")
    for p, t in live.items():
        for g in GRAPHICS.findall(t):
            if not any((root / (g + e)).is_file() for e in EXT):
                problems.append(
                    f"\\includegraphics{{{g}}} in {p.relative_to(root)} "
                    f"-> image not found")

    # ---------------------------------------------------------------- cites
    bibs: set[str] = set()
    for p, t in live.items():
        for m in BIBSTYLE.findall(t):
            for name in m.split(","):
                name = name.strip()
                for cand in (root / name, root / f"{name}.bib"):
                    if cand.is_file():
                        bibs.add(str(cand))
    keys: set[str] = set()
    for b in bibs:
        btxt = Path(b).read_text(encoding="utf-8", errors="replace")
        keys |= set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', btxt))
    if keys:
        for p, t in live.items():
            for m in CITE.findall(t):
                for k in (x.strip() for x in m.split(",")):
                    if k and k not in keys:
                        problems.append(
                            f"\\cite{{{k}}} in {p.relative_to(root)} "
                            f"-> key not in the .bib")
    else:
        notes.append("no .bib resolved from \\bibliography{...}; citations unchecked")

    # ---------------------------------------------------------------- braces
    for p, t in live.items():
        depth = 0
        esc = False
        for c in t:
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth < 0:
                    problems.append(f"{p.relative_to(root)}: unbalanced '}}'")
                    break
        if depth > 0:
            problems.append(f"{p.relative_to(root)}: {depth} unclosed '{{'")

    # ---------------------------------------------------------------- envs
    for p, t in live.items():
        stack: list[str] = []
        for m in re.finditer(r'\\(begin|end)\{([^}]*)\}', t):
            kind, name = m.group(1), m.group(2)
            if kind == "begin":
                stack.append(name)
            else:
                if not stack:
                    problems.append(
                        f"{p.relative_to(root)}: \\end{{{name}}} with no \\begin")
                elif stack[-1] != name:
                    problems.append(
                        f"{p.relative_to(root)}: \\end{{{name}}} closes "
                        f"\\begin{{{stack[-1]}}}")
                    stack.pop()
                else:
                    stack.pop()
        for leftover in stack:
            problems.append(
                f"{p.relative_to(root)}: \\begin{{{leftover}}} never closed")

    # ---------------------------------------------------------------- percent
    # Deliberately scans the RAW line. Stripping comments first would delete the
    # very character being looked for -- the whole point is that this `%` was
    # meant as text and LaTeX is reading it as the start of a comment.
    for p, t in raw.items():
        for n, line in enumerate(t.split("\n"), 1):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue                      # a genuinely commented-out line
            for m in BAD_PCT.finditer(line):
                ctx = line[max(0, m.start() - 24):m.start() + 2]
                problems.append(
                    f"{p.relative_to(root)}:{n}: unescaped '%' -> rest of line "
                    f"becomes a comment: ...{ctx.strip()}")

    # ---------------------------------------------------------------- report
    print(f"== check_latex: {root}  ({len(tex)} .tex files, "
          f"{len(labels)} labels, {len(refs)} distinct refs) ==\n")
    for n in notes:
        print(f"  [note] {n}")
    if notes:
        print()
    if not problems:
        print("[OK] no structural problems found.")
        sys.exit(0)
    for pr in problems:
        print(f"  [!!] {pr}")
    print(f"\n[FAIL] {len(problems)} structural problem(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
