"""Check that every repository path the manuscripts cite is actually published.

The papers claim the resource is reproducible from this repository. That claim is
only true if the files they name are present in a clone. Ignoring a provenance
file is therefore not a packaging detail -- it silently falsifies a claim in the
text, and nothing else in the toolchain notices: `verify_numbers.py` reads the
working tree, where ignored files are still present, so it passes locally and a
reader's clone fails.

This script extracts every path-looking `\\texttt{...}` / `\\url{...}` token from
the LaTeX sources and the provenance notes, then asks git whether each one would
survive a clone.

    python pipeline/check_repo_claims.py
    python pipeline/check_repo_claims.py --paper paper-iri

Exit code is 1 if any cited path is missing or ignored.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A cited token is a path if it looks like one: contains a slash, or ends in a
# known data/code extension.
EXT = (".csv", ".md", ".py", ".json", ".tex", ".owl", ".ttl", ".yml", ".cff",
       ".txt", ".bib", ".sh", ".xlsx", ".jsonl")

TOKEN_RE = re.compile(r"\\(?:texttt|path|url)\{([^{}]*)\}")

# A \url pointing into this repository is a claim about repository content just as
# much as a \texttt path is -- and a more dangerous one, because it renders as a
# link a reviewer will click. Skipping URLs let a footnote promising
# scholar/ODPs_WOP.csv (which does not exist) pass unnoticed.
SELF_URL = re.compile(
    r"https?://github\.com/ISE-FIZKarlsruhe/ODP-Reuse/(?:blob|tree|raw)/"
    r"[^/]+/(?P<path>[^\s}#?]+)")
# LaTeX escapes its underscores; undo that plus a few common wrappers.
UNESCAPE = ((r"\_", "_"), (r"\-", "-"), (r"-{}-", "--"), (r"\%", "%"),
            (r"\&", "&"), (r"\#", "#"), (r"\$", "$"), ("~", ""))


def clean(tok: str) -> str:
    for a, b in UNESCAPE:
        tok = tok.replace(a, b)
    return tok.strip().strip("`").rstrip(".,;:")


# Cited paths that legitimately live on Zenodo rather than in git: they are
# gigabytes. A citation to one is a true claim about the *release*, not a broken
# claim about the repository, so they are reported separately, not as failures.
ZENODO_ONLY = ("release-iri/corpus", "release-iri-label/corpus")

# Top-level entries that exist in the repository. A cited token only counts as a
# repository-path claim if it is anchored at one of these; otherwise it is a bare
# filename or a path relative to some directory under discussion (`out/g5/...`,
# `MANIFEST.csv`, `agentrole.owl`) and cannot be resolved from the root.
TOPLEVEL = {p.name for p in ROOT.iterdir()}


# The tables cite pipeline outputs in shorthand -- `out/patternstats/x.csv` for
# `pipeline/out/patternstats/x.csv`, and likewise for the use-case result trees.
# Resolving the shorthand matters: these are the sources behind published tables,
# and treating them as unresolvable would hide exactly the claims most at risk.
PREFIX_ALIASES = (
    ("out/", "pipeline/out/"),
    ("results/", "use-cases/odp-extraction/results/"),
)


def canonical(tok: str) -> str:
    tok = tok.lstrip("./")
    if tok.split("/", 1)[0] in TOPLEVEL:
        return tok
    for short, full in PREFIX_ALIASES:
        if tok.startswith(short) and (ROOT / (full + tok[len(short):])).exists():
            return full + tok[len(short):]
    return tok


def looks_like_path(tok: str) -> bool:
    if not tok or " " in tok or tok.startswith(("http://", "https://", "--", "-")):
        return False
    if tok.startswith("$") or "\\" in tok:
        return False
    if "/" not in tok and not tok.endswith(EXT):
        return False
    return canonical(tok).split("/", 1)[0] in TOPLEVEL


def git_ignored(rel: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "-q", rel],
                       cwd=ROOT, capture_output=True)
    return r.returncode == 0


def resolve(rel: str) -> tuple[bool, str]:
    """Does this path exist, allowing for globs and directory forms?"""
    p = ROOT / rel
    if any(ch in rel for ch in "*<>"):
        # A glob or a <placeholder>: check the nearest concrete ancestor.
        parts = Path(rel).parts
        concrete = []
        for part in parts:
            if any(ch in part for ch in "*<>"):
                break
            concrete.append(part)
        if not concrete:
            return True, "pattern"
        anc = ROOT.joinpath(*concrete)
        return anc.exists(), f"ancestor {'/'.join(concrete)}"
    return p.exists(), ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=None,
                    help="restrict to one manuscript directory")
    args = ap.parse_args()

    papers = ([ROOT / args.paper] if args.paper else
              sorted(p for p in ROOT.glob("paper-*") if p.is_dir()))

    problems: list[tuple[str, str, str]] = []
    seen: dict[str, set[str]] = {}

    for paper in papers:
        if not paper.is_dir():
            continue
        srcs = sorted(paper.rglob("*.tex")) + [
            f for f in (paper / "PROVENANCE.md", paper / "README.md")
            if f.is_file()]
        for src in srcs:
            # Only claims a reader can actually encounter count. A `.tex` always
            # counts: it is unpublished during review but becomes the PDF, so a
            # path it cites is a promise to whoever reads the paper. A `.md`
            # counts only if it is itself published -- an ignored README is an
            # internal note, and a path it mentions misleads nobody.
            if src.suffix != ".tex" and git_ignored(
                    str(src.relative_to(ROOT)).replace("\\", "/")):
                continue
            text = src.read_text(encoding="utf-8", errors="replace")
            toks = {clean(t) for t in TOKEN_RE.findall(text)}
            if src.suffix == ".md":  # backtick-quoted paths in the notes
                toks |= {clean(t) for t in re.findall(r"`([^`]+)`", text)}
            for tok in toks:
                if looks_like_path(tok):
                    seen.setdefault(canonical(tok), set()).add(
                        str(src.relative_to(ROOT)).replace("\\", "/"))
            # URLs into this repository, resolved to the paths they promise
            for m in SELF_URL.finditer(text.replace("\\_", "_")):
                seen.setdefault(m.group("path").rstrip("/"), set()).add(
                    str(src.relative_to(ROOT)).replace("\\", "/") + " [url]")

    print(f"== repository claims: {len(seen)} distinct paths cited by "
          f"{', '.join(p.name for p in papers if p.is_dir())} ==\n")

    zenodo: list[str] = []
    for rel in sorted(seen):
        exists, note = resolve(rel)
        # git treats "dir" and "dir/" differently for check-ignore; normalise.
        ignored = git_ignored(rel.rstrip("/"))
        if rel.rstrip("/").startswith(ZENODO_ONLY):
            zenodo.append(rel)
            continue
        if not exists:
            problems.append((rel, "MISSING", sorted(seen[rel])[0]))
        elif ignored:
            problems.append((rel, "IGNORED", sorted(seen[rel])[0]))

    if zenodo:
        print(f"-- {len(zenodo)} cited path(s) are Zenodo artefacts, too large for "
              f"git (expected):")
        for rel in zenodo:
            print(f"   {rel}")
        print()

    if not problems:
        print(f"[OK] all {len(seen) - len(zenodo)} repository-path claims exist "
              f"and would survive a clone.")
        sys.exit(0)

    miss = [p for p in problems if p[1] == "MISSING"]
    ign = [p for p in problems if p[1] == "IGNORED"]

    if ign:
        print(f"-- {len(ign)} cited path(s) exist locally but are GITIGNORED, so a")
        print("   reader's clone will not have them:\n")
        for rel, _, where in ign:
            print(f"   {rel}")
            print(f"       cited by {where}")
    if miss:
        print(f"\n-- {len(miss)} cited path(s) do not exist at all:\n")
        for rel, _, where in miss:
            print(f"   {rel}")
            print(f"       cited by {where}")

    print(f"\n[FAIL] {len(problems)} of {len(seen)} cited paths are unavailable "
          f"to a reader.")
    sys.exit(1)


if __name__ == "__main__":
    main()
