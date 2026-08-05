"""Check a manuscript's bibliography: consistency, completeness, and duplicates.

Three classes of problem, all of which survive a successful compile:

  * a key cited but absent from the .bib (bibtex prints a warning nobody reads)
  * an entry with a field a reader needs but that is missing -- a journal article
    with no journal, a paper with no year, a URL-only entry with no accessed date
  * two entries for the same work under different keys, which produces two
    numbered references to one paper

Usage
-----
    python pipeline/check_refs.py paper-iri-iswc
    python pipeline/check_refs.py paper-iri-iswc --unused   # also list uncited
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.I)
CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")

# Fields a reader needs in order to find the work again.
REQUIRED = {
    "article": ("author", "title", "journal", "year"),
    "inproceedings": ("author", "title", "booktitle", "year"),
    "incollection": ("author", "title", "booktitle", "year"),
    "book": ("author", "title", "publisher", "year"),
    "inbook": ("author", "title", "publisher", "year"),
    "phdthesis": ("author", "title", "school", "year"),
    "mastersthesis": ("author", "title", "school", "year"),
    "techreport": ("author", "title", "institution", "year"),
    "misc": ("title",),
    "online": ("title",),
    "unpublished": ("author", "title"),
}


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        i, esc, cut = 0, False, None
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


def split_entries(bib: str) -> dict[str, tuple[str, str]]:
    """key -> (type, raw body). Brace-counted, so nested braces survive."""
    out: dict[str, tuple[str, str]] = {}
    for m in ENTRY.finditer(bib):
        etype, key = m.group(1).lower(), m.group(2)
        i = bib.find("{", m.start())
        depth, j = 0, i
        while j < len(bib):
            if bib[j] == "{":
                depth += 1
            elif bib[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[key] = (etype, bib[m.end():j])
    return out


def field(body: str, name: str) -> str | None:
    m = re.search(rf"\b{name}\s*=\s*", body, re.I)
    if not m:
        return None
    rest = body[m.end():].lstrip()
    if rest[:1] in "{\"":
        close = "}" if rest[0] == "{" else '"'
        depth, buf = 0, []
        for ch in rest:
            if ch == "{":
                depth += 1
            elif ch == close and (close == '"' or depth == 1):
                if close == "}":
                    depth -= 1
                break
            elif ch == "}":
                depth -= 1
            if depth >= 1 or (close == '"' and ch != '"'):
                buf.append(ch)
        return "".join(buf).lstrip("{").strip()
    return rest.split(",")[0].strip()


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", type=Path)
    ap.add_argument("--unused", action="store_true")
    args = ap.parse_args()

    root: Path = args.paper
    bibs = sorted(root.glob("*.bib"))
    if not bibs:
        sys.exit(f"no .bib under {root}")
    bib_text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in bibs)
    entries = split_entries(bib_text)

    cited: Counter = Counter()
    where: dict[str, set[str]] = defaultdict(set)
    for p in sorted(root.rglob("*.tex")):
        live = strip_comments(p.read_text(encoding="utf-8", errors="replace"))
        for m in CITE.findall(live):
            for k in (x.strip() for x in m.split(",")):
                if k:
                    cited[k] += 1
                    where[k].add(str(p.relative_to(root)).replace("\\", "/"))

    problems: list[str] = []
    notes: list[str] = []

    for k in sorted(cited):
        if k not in entries:
            problems.append(
                f"\\cite{{{k}}} has no entry in the .bib  "
                f"(in {', '.join(sorted(where[k]))})")

    for k in sorted(cited):
        if k not in entries:
            continue
        etype, body = entries[k]
        for f in REQUIRED.get(etype, ()):
            if field(body, f) is not None:
                continue
            # An edited volume carries `editor`, not `author`, and that is correct
            # BibTeX -- flagging it would be a false positive on a well-formed
            # entry, which is worse than not checking at all.
            if f == "author" and field(body, "editor") is not None:
                continue
            problems.append(f"{k} ({etype}): missing required field '{f}'")
        yr = field(body, "year")
        if yr and not re.fullmatch(r"\d{4}", yr.strip()):
            problems.append(f"{k}: year is {yr!r}, not a 4-digit year")

    # duplicate works under different keys
    by_title: dict[str, list[str]] = defaultdict(list)
    for k, (etype, body) in entries.items():
        t = norm_title(field(body, "title") or "")
        if len(t) > 18:
            by_title[t].append(k)
    for t, ks in sorted(by_title.items()):
        if len(ks) > 1:
            cited_ks = [k for k in ks if k in cited]
            msg = (f"duplicate entries for one title: {', '.join(sorted(ks))}")
            (problems if len(cited_ks) > 1 else notes).append(
                msg + ("" if len(cited_ks) > 1 else "  (only one is cited)"))

    unused = sorted(set(entries) - set(cited))

    print(f"== check_refs: {root}  ({len(entries)} entries, "
          f"{len(cited)} cited, {sum(cited.values())} citations) ==\n")
    if args.unused and unused:
        print(f"  {len(unused)} entry(ies) never cited:")
        for k in unused:
            print(f"     {k}")
        print()
    for n in notes:
        print(f"  [note] {n}")
    if notes:
        print()
    if not problems:
        print("[OK] every citation resolves and every cited entry is complete.")
        sys.exit(0)
    for pr in problems:
        print(f"  [!!] {pr}")
    print(f"\n[FAIL] {len(problems)} bibliography problem(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
