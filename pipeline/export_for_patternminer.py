"""Materialise a gate-passing corpus in the layout PatternMiner expects.

Why this exists
---------------
PatternMiner's ``run_evaluation.py`` walks ``ONTOLOGIES_DIR`` expecting each
*immediate* subdirectory to be one ontology, and
``evaluation.load_ground_truth`` reads that ontology's ``patterns/``
subdirectory. ``release-iri/corpus/<portion>/`` already has exactly that shape,
so a portion directory is a drop-in replacement for the old
``Ontologies_reusingODPs/notOBO``.

What is *not* a drop-in is the population. ``release-iri/corpus`` is deliberately
**unfiltered**: it holds all 1,973 discovered ontologies, including the ones the
gate rejects -- unparseable files, verbatim copies of a pattern, byte-identical
duplicates, and ontologies with no pattern overlap at all. Pointing PatternMiner
at it directly would evaluate against a population the paper does not report, and
would score pattern copies against the patterns they *are*, which is the
tautology the G2 rule exists to remove.

So this script filters by the manifest's own verdict and writes the surviving
ontologies out in the same layout. Hard links are used where possible: the OBO
portion alone is 1.37 GiB and copying it twice on one disk is pointless.

Usage
-----
    # what would be exported, and how big
    python pipeline/export_for_patternminer.py --dry-run

    # the two portions worth running (94 MiB, 493 of 501 ontologies)
    python pipeline/export_for_patternminer.py --out ../pm-corpus \\
        --portions github noOBO

    # a file list for rsync to a server, instead of a local tree
    python pipeline/export_for_patternminer.py --list > gated-files.txt
    rsync -a --files-from=gated-files.txt . user@server:/data/odp-reuse/

Level
-----
``--level reuse`` (default) keeps every gate-passing ontology. ``--level
complete`` keeps only those reusing at least one pattern in full -- the stricter
ground truth, and the right choice when a partial hit should not count.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_manifest(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select(man: list[dict], level: str, portions: list[str] | None) -> list[dict]:
    out = []
    for r in man:
        if r["drop_reason"]:
            continue                      # the gate rejected it
        if level == "complete" and int(r.get("n_complete_patterns") or 0) < 1:
            continue
        if portions and r["dataset"] not in portions:
            continue
        out.append(r)
    return out


def longpath(p: Path) -> str:
    r"""Windows-safe path string.

    GitHub-derived ontology ids run to ~140 characters and the layout repeats the
    id, so both ``os.link`` and ``open`` fail on MAX_PATH long before the tree is
    complete. The ``\\?\`` prefix lifts the limit.
    """
    s = str(Path(p).resolve())
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return s
    return "\\\\?\\" + s


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    """Hard-link if allowed, else copy. Returns what actually happened."""
    Path(longpath(dst.parent)).mkdir(parents=True, exist_ok=True)
    if Path(longpath(dst)).exists():
        return "exists"
    if mode in ("link", "auto"):
        try:
            os.link(longpath(src), longpath(dst))
            return "link"
        except OSError:
            if mode == "link":
                raise
    shutil.copyfile(longpath(src), longpath(dst))
    return "copy"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=ROOT / "release-iri" / "corpus")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "release-iri" / "MANIFEST.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="destination root; omit with --dry-run or --list")
    ap.add_argument("--level", choices=("reuse", "complete"), default="reuse")
    ap.add_argument("--portions", nargs="*", default=None,
                    help="restrict to these portions (default: all)")
    ap.add_argument("--mode", choices=("auto", "link", "copy"), default="auto",
                    help="hard-link (cheap) or copy; auto tries link then copies")
    ap.add_argument("--flat", action="store_true",
                    help="write <out>/<id>/ directly instead of "
                         "<out>/<portion>/<id>/ -- use to pool portions into one "
                         "PatternMiner run, but see the warning it prints")
    ap.add_argument("--layout", choices=("portions", "patternminer"),
                    default="portions",
                    help="'patternminer' writes the six dataset directories its "
                         "scripts already name -- <portion> for the reuse level "
                         "and <portion>_graph100 for complete reuse -- so no "
                         "dataset name in those scripts has to change")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print the file list (for rsync --files-from) and exit")
    args = ap.parse_args()

    if not args.manifest.is_file():
        sys.exit(f"manifest not found: {args.manifest}")
    man = read_manifest(args.manifest)

    # In patternminer layout both levels are emitted at once, under the dataset
    # names its scripts already use: <portion> is the reuse level and
    # <portion>_graph100 the complete-reuse subset. Those names are wired into
    # ~180 places across 45 shell scripts (log names, --datasets lists, cache
    # keys), so reproducing them is much safer than renaming them.
    if args.layout == "patternminer":
        jobs = [("reuse", ""), ("complete", "_graph100")]
    else:
        jobs = [(args.level, "")]

    plan: list[tuple[Path, Path]] = []
    per_portion = Counter()
    bytes_portion = Counter()
    missing = []
    rows: list[dict] = []

    for level, suffix in jobs:
        sel = select(man, level, args.portions)
        rows += sel
        for r in sel:
            d = args.corpus / r["dataset"] / r["ontology_id"]
            if not d.is_dir():
                missing.append(f"{r['dataset']}/{r['ontology_id']}")
                continue
            label = f"{r['dataset']}{suffix}"
            per_portion[label] += 1
            rel_base = (Path(r["ontology_id"]) if args.flat
                        else Path(label) / r["ontology_id"])
            # Enumerate through the \\?\ prefix. Without it, `f.is_file()` on a
            # path past MAX_PATH raises internally, pathlib swallows the error and
            # returns False, and the file is dropped from the plan with no warning
            # -- which silently cost two ontologies their main .owl file while the
            # run still reported success.
            base = Path(longpath(d))
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    plan.append((f, rel_base / f.relative_to(base)))
                    bytes_portion[label] += f.stat().st_size

    if args.list:
        for src, _ in plan:
            print(src.relative_to(ROOT).as_posix())
        return

    level_note = ("both levels (reuse + complete, as <portion> and "
                  "<portion>_graph100)" if args.layout == "patternminer"
                  else f"level={args.level}")
    print(f"== export_for_patternminer: {level_note}, "
          f"{len(rows)} ontology slots, {len(plan)} files ==")
    for p in sorted(per_portion):
        print(f"   {p:<8} {per_portion[p]:>4} ontologies  "
              f"{bytes_portion[p] / 2**20:>9.1f} MiB")
    print(f"   {'TOTAL':<8} {sum(per_portion.values()):>4} ontologies  "
          f"{sum(bytes_portion.values()) / 2**20:>9.1f} MiB")
    if missing:
        print(f"   [!!] {len(missing)} selected ontologies have no directory "
              f"in the corpus: {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else ""))

    if args.flat:
        print("   [!!] --flat pools portions into one directory. The three "
              "portions are not comparable: DOS-DP ontologies carry orders of "
              "magnitude more entities, so a pooled average is dominated by "
              "them. Run per portion unless you specifically want the pool.")

    if args.dry_run or args.out is None:
        print("\n   dry run -- nothing written. Pass --out to materialise.")
        return

    counts = Counter()
    for src, rel in plan:
        counts[link_or_copy(src, args.out / rel, args.mode)] += 1
    print(f"\n[OK] {args.out}: " + ", ".join(f"{v} {k}" for k, v in
                                            sorted(counts.items())))

    # A note in the tree, so nobody has to guess which population this is.
    (args.out / "README.md").write_text(f"""\
# Gate-passing corpus for PatternMiner

Exported from `release-iri` by `pipeline/export_for_patternminer.py`.

- Level: **{args.level}** ({'every gate-passing ontology' if args.level == 'reuse'
  else 'only ontologies reusing at least one pattern in full'})
- Portions: {', '.join(sorted(per_portion)) or 'none'}
- Ontologies: **{sum(per_portion.values())}**

Layout is `{'<id>/' if args.flat else '<portion>/<id>/'}` with the ontology file
and its `patterns/` subdirectory, which is what PatternMiner's
`run_evaluation.py` and `evaluation.load_ground_truth` expect. Point
`ONTOLOGIES_DIR` at {'this directory' if args.flat else 'a *portion* directory inside this one'}.

This is the **filtered** population: ontologies the quality gate rejected are
absent. The unfiltered corpus and the per-ontology verdicts are in the ODP-Reuse
release (`MANIFEST.csv`). Do not pool the portions into one average -- see
`FILTERS.md` and the dataset documentation for why.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
