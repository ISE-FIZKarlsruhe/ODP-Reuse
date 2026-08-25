#!/usr/bin/env python3
"""Join hard-wrapped source lines so each paragraph is one line.

LaTeX treats a single newline as a space, so unwrapping a paragraph cannot
change the typeset output -- which is exactly how this script is verified:
compile before and after and diff the extracted PDF text. Anything that does
change the output is a bug in the rules below, not an accepted cost.

WHAT IS NEVER JOINED, and why
    blank line              paragraph break; joining merges two paragraphs
    % comment line          appending the next line would COMMENT IT OUT
    line ending in %        the trailing % suppresses the newline-space on
                            purpose (used to avoid a space before punctuation
                            or inside a tabular cell); joining changes spacing
    \\ at end of line        a tabular row or a forced break
    structural commands     \section \subsection \paragraph \label \begin \end
                            \item \input \caption-closing etc. -- keeping these
                            at line starts is what makes the source navigable
    &  in the line          a tabular cell row; the row structure is the content
    verbatim-like bodies    newlines are significant inside them

Usage
    python pipeline/unwrap_tex.py PATH...            # rewrite in place
    python pipeline/unwrap_tex.py --check PATH...    # report only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# A line that opens or closes a structure, or stands alone semantically. Joining
# either into a neighbour is legal LaTeX but destroys the source's shape, and for
# \item it also runs list entries together.
STRUCTURAL = re.compile(
    r"^\s*(?:\\(?:documentclass|usepackage|newcommand|renewcommand|def"
    r"|section|subsection|subsubsection|paragraph|subparagraph"
    r"|label|input|include|bibliography|bibliographystyle"
    r"|begin|end|item|maketitle|title|author|affiliation|abstract"
    # Title-block metadata: each is a separate field and must stay on its own
    # line. Without these, \corrauth and \email get swept into the tail of
    # \affiliation, which compiles identically but makes the block unreadable.
    r"|corrauth|email|date|keywords|runninghead|received|revised|accepted"
    r"|sagetrue|sagefalse"
    r"|caption|toprule|midrule|bottomrule|cmidrule|hline|multicolumn"
    r"|centering|raggedright|small|footnotesize|scriptsize|normalsize"
    r"|vspace|hspace|smallskip|medskip|bigskip|newpage|clearpage"
    r"|setlength|arrayrulecolor|rowcolor|if\w*|else|fi)\b"
    r"|\\\[|\\\]|\$\$)"
)

# Environments whose line structure is meaningful.
VERBATIM = re.compile(r"^\s*\\begin\{(verbatim|lstlisting|minted|Verbatim|alltt|"
                      r"tabular|tabularx|longtable|array|tabu)\*?\}")
VERBATIM_END = re.compile(r"^\s*\\end\{(verbatim|lstlisting|minted|Verbatim|alltt|"
                          r"tabular|tabularx|longtable|array|tabu)\*?\}")


def is_comment(line: str) -> bool:
    """True for a whole-line comment. A leading escaped \\% is not a comment."""
    s = line.lstrip()
    return s.startswith("%")


def ends_with_active_percent(line: str) -> bool:
    """True if the line ends in an unescaped % (deliberate newline suppression)."""
    s = line.rstrip()
    if not s.endswith("%"):
        return False
    # count trailing backslashes before the %
    body = s[:-1]
    return (len(body) - len(body.rstrip("\\"))) % 2 == 0


def unwrap(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    buf: list[str] = []
    depth = 0  # verbatim-like nesting

    def flush() -> None:
        if not buf:
            return
        if len(buf) == 1:
            # Nothing to join: keep the line byte-for-byte, indentation and all.
            out.append(buf[0])
        else:
            out.append(" ".join(x.strip() for x in buf))
        buf.clear()

    for raw in lines:
        line = raw.rstrip("\r")
        stripped = line.strip()

        if depth:
            flush()
            out.append(line)
            if VERBATIM_END.match(line):
                depth -= 1
            elif VERBATIM.match(line):
                depth += 1
            continue

        if VERBATIM.match(line):
            flush()
            out.append(line)
            depth += 1
            continue

        # Hard boundaries: emit as their own line, never merged.
        if (not stripped
                or is_comment(line)
                or STRUCTURAL.match(line)
                or "&" in stripped
                or stripped.endswith("\\\\")
                or ends_with_active_percent(line)):
            flush()
            out.append(line)
            continue

        buf.append(line)

    flush()
    # Collapse runs of >2 blank lines that the pass may expose, but keep single
    # and double blanks: a double blank is sometimes an author's section spacer.
    result = "\n".join(out)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    total = 0
    for p in args.paths:
        if not p.is_file():
            print(f"[!!] {p}: not a file")
            continue
        src = p.read_text(encoding="utf-8")
        new = unwrap(src)
        if new == src:
            print(f"[..] {p.name}: already unwrapped")
            continue
        before_max = max((len(x) for x in src.split("\n")), default=0)
        after_max = max((len(x) for x in new.split("\n")), default=0)
        joined = src.count("\n") - new.count("\n")
        print(f"[OK] {p.name}: {joined} line breaks removed, "
              f"longest line {before_max} -> {after_max}")
        total += joined
        if not args.check:
            p.write_text(new, encoding="utf-8")
    print(f"\n{total} line breaks removed in total"
          + (" (check only, nothing written)" if args.check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
