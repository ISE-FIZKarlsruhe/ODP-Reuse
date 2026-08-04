"""Build the IRI-anonymised twin of a release corpus: ``corpus`` -> ``corpus-hashed``.

Why this is not ``build_release.py --hash``
-------------------------------------------
``--hash`` builds the hashed sibling of every *view*. A view is CSV only, so it
is cheap and ``hash_dataset`` handles it well. The corpus is a different problem:
6,345 RDF files and 2.7 GiB, in which seven of the eight OBO ontologies -- that
is, essentially the whole OBO portion the paper reports per-portion results for
-- are single files between 60 MiB and 400 MiB, with ``noOBO/cco/cco.owl`` at
1.1 GiB. Parsing those into an rdflib ``Graph`` needs far more memory than is
available, so ``hash_dataset``'s ``--hash-max-mb`` escape hatch would silently
drop a reported portion from the release. A hashed corpus missing its OBO
portion is not a usable twin of the plain one.

How the memory problem goes away
--------------------------------
The hash is a pure function of a single IRI string, and blank-node labels hash
the same way, so **no term's output depends on any other term or triple**. The
rewrite is therefore context-free and the whole job streams: this module parses
with a write-through store whose ``add`` serialises the hashed triple and
immediately forgets it. Memory stays flat regardless of input size, and no
intermediate graph is ever built.

Because the rewrite is term-wise identical to ``hash_dataset``'s, the two
produce the same set of triples for the same input; ``--self-test`` asserts that
against the in-memory implementation rather than asking the reader to trust it.

Output format
-------------
Every hashed file is written as N-Triples (``.nt``). Re-serialising RDF/XML
requires the whole graph in memory, which is the very thing being avoided, and
N-Triples is the format a detection experiment wants anyway -- line-oriented and
streamable. The plain corpus keeps the original serialisations, so nothing is
lost. ``format_map.csv`` records the original extension for every file.

Usage
-----
    python pipeline/hash_corpus.py --corpus release-iri/corpus
    python pipeline/hash_corpus.py --corpus release-iri/corpus --self-test
    python pipeline/hash_corpus.py --corpus release-iri/corpus --limit 20

The walk is resumable: a file whose output already exists is skipped, so an
interrupted run continues where it stopped. ``--force`` re-does everything.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_release import HASH_PREFIX, hash_iri, hash_token, longpath  # noqa: E402

RDF_EXT = (".owl", ".ttl", ".rdf", ".xml", ".nt", ".n3", ".trig", ".jsonld")

# Format preference per extension, mirroring odpqual.entities.parse_graph so a
# file that parses during the audit also parses here.
FORMAT_ORDER = {
    ".ttl": ("turtle", "xml", "n3"),
    ".n3": ("n3", "turtle", "xml"),
    ".nt": ("nt", "turtle"),
    ".trig": ("trig", "turtle"),
    ".jsonld": ("json-ld",),
}
DEFAULT_ORDER = ("xml", "turtle", "n3", "nt")


def protected_prefixes() -> tuple[str, ...]:
    from rdflib.namespace import OWL, RDF, RDFS, XSD
    return tuple(str(n) for n in (RDF, RDFS, OWL, XSD)) + (
        "http://www.w3.org/XML/1998/namespace",)


def make_sink_class():
    """Build the write-through Store class (imports rdflib lazily)."""
    from rdflib import BNode, URIRef
    from rdflib.plugins.serializers.nt import _nt_row
    from rdflib.store import Store

    PROTECTED = protected_prefixes()

    class HashingSink(Store):
        """A store that hashes each incoming triple, writes it, and forgets it.

        ``rdflib``'s parsers push triples in as they are recognised, so overriding
        ``add`` turns a parse into a streaming transform. Nothing is retained:
        there is no triple index and no IRI table, because the hash needs no
        context to be consistent.
        """

        context_aware = False
        formula_aware = False
        transaction_aware = False
        graph_aware = False

        def __init__(self, out=None, configuration=None, identifier=None):
            super().__init__(configuration)
            self._out = out
            self.n_triples = 0
            # The Turtle/N3 parser resolves `@prefix` declarations *through the
            # store*, so stubbing these out makes every prefixed name fail with
            # "Prefix ... not bound" -- silently turning valid Turtle into a
            # parse error. Bindings are per-document and few, so holding them
            # costs nothing and does not affect the streaming guarantee. They are
            # kept unhashed: N-Triples output carries full IRIs, so a binding
            # never reaches the output.
            self._ns: dict[str, str] = {}
            self._pfx: dict[str, str] = {}

        # -- the transform -------------------------------------------------
        def _term(self, t):
            if isinstance(t, URIRef):
                s = str(t)
                if s.startswith(PROTECTED):
                    return t
                return URIRef(hash_iri(s, HASH_PREFIX))
            if isinstance(t, BNode):
                return BNode(f"b{hash_token(str(t))}")
            return t  # Literals pass through, exactly as hash_dataset does.

        def add(self, triple, context=None, quoted=False):
            s, p, o = triple
            self._out.write(_nt_row((self._term(s), self._term(p),
                                     self._term(o))))
            self.n_triples += 1

        # -- Store API the parsers touch -----------------------------------
        def __len__(self, context=None):
            return self.n_triples

        def triples(self, triple_pattern, context=None):
            return iter(())

        def bind(self, prefix, namespace, *args, **kwargs):
            self._ns[prefix] = namespace
            self._pfx[str(namespace)] = prefix

        def namespace(self, prefix):
            return self._ns.get(prefix)

        def prefix(self, namespace):
            return self._pfx.get(str(namespace))

        def namespaces(self):
            return iter(self._ns.items())

        def remove(self, triple, context=None):
            return None

    return HashingSink


def hash_one(src: Path, dst: Path, sink_cls,
             max_inmemory_mb: float = 64.0) -> tuple[int, str]:
    """Stream-hash ``src`` into ``dst``. Returns ``(n_triples, error)``.

    ``max_inmemory_mb`` bounds the non-streaming fallback described at the end of
    this function: files above it are reported as failures rather than risking the
    memory blow-up the streaming design exists to avoid.
    """
    from rdflib import Graph

    order = FORMAT_ORDER.get(src.suffix.lower(), DEFAULT_ORDER)
    first_error = ""
    # Via longpath: some GitHub-derived ids are ~140 characters (percent-encoded
    # Cyrillic repository names) and the layout repeats the id, so a plain mkdir
    # dies with WinError 206 well before the file is ever opened.
    Path(longpath(dst.parent)).mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    for i, fmt in enumerate(order):
        try:
            # Read through a file object with an explicit publicID rather than a
            # path string. `longpath()`'s \\?\ prefix makes rdflib treat the
            # source as a URL and fail with URLError, while a bare path breaks on
            # the ~100-character ontology ids the release layout repeats twice.
            # A handle sidesteps both, and publicID keeps base resolution
            # identical to odpqual.parse_graph's `p.as_posix()`.
            # newline="" so the \n that _nt_row already emits is not translated
            # to \r\n on Windows: N-Triples lines end with LF.
            with open(longpath(tmp), "w", encoding="utf-8", newline="") as fh, \
                    open(longpath(src), "rb") as inp:
                sink = sink_cls(out=fh)
                Graph(store=sink).parse(source=inp, format=fmt,
                                        publicID=src.resolve().as_uri())
                n = sink.n_triples
            Path(longpath(tmp)).replace(longpath(dst))
            return n, ""
        except Exception as exc:  # noqa: BLE001
            if i == 0:
                msg = " ".join(str(exc).split())[:300]
                first_error = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
            try:
                Path(longpath(tmp)).unlink()
            except OSError:
                pass

    # Streaming failed in every format. Before giving up, try the audit's own
    # parser into an ordinary in-memory graph.
    #
    # rdflib's N3 parser refuses a non-context-aware store ("Cannot parse N3 into
    # non-context-aware store"), and a write-through sink is necessarily not one.
    # So a Turtle-suffixed file that only parses as N3 -- one using `=` for
    # owl:sameAs, say -- fails here while `odpqual.parse_graph` reads it fine.
    # That is a disagreement between the hashed corpus and the gate, which is
    # exactly what must not happen: the gate would admit the ontology and the
    # hashed twin would be missing it.
    #
    # Falling back in memory is safe precisely where it is needed: the streaming
    # path exists for the multi-hundred-megabyte OBO files, which are RDF/XML and
    # never reach this branch. The cap keeps it that way.
    if src.stat().st_size <= max_inmemory_mb * 1024 * 1024:
        from odpqual.entities import parse_graph
        g, err = parse_graph(src)
        if g is not None:
            try:
                with open(longpath(tmp), "w", encoding="utf-8", newline="") as fh:
                    sink = sink_cls(out=fh)      # the same term transform
                    for triple in g:
                        sink.add(triple)
                    n = sink.n_triples
                Path(longpath(tmp)).replace(longpath(dst))
                return n, ""
            except Exception as exc:  # noqa: BLE001
                return 0, f"in-memory fallback failed: {type(exc).__name__}"

    return 0, first_error or "parse failed in every format"


def self_test(corpus: Path, sink_cls, n: int = 6) -> bool:
    """Assert the streaming transform matches the in-memory one, triple for triple.

    ``hash_dataset`` builds a graph, maps every term, and re-serialises. This
    module maps each term as it streams past. The claim that the two agree is
    worth checking rather than asserting, because everything downstream of the
    hashed corpus depends on it.
    """
    from rdflib import BNode, Graph, URIRef
    from rdflib.compare import isomorphic
    from rdflib.namespace import OWL, RDF, RDFS, XSD

    protected = tuple(str(x) for x in (RDF, RDFS, OWL, XSD)) + (
        "http://www.w3.org/XML/1998/namespace",)

    def reference(g: Graph) -> set:
        """The in-memory transform, transcribed from build_release.hash_dataset."""
        imap: dict = {}
        bmap: dict = {}

        def m(t):
            if isinstance(t, URIRef):
                s = str(t)
                if s.startswith(protected):
                    return t
                if t not in imap:
                    imap[t] = URIRef(hash_iri(s))
                return imap[t]
            if isinstance(t, BNode):
                if t not in bmap:
                    bmap[t] = BNode(f"b{hash_token(str(t))}")
                return bmap[t]
            return t

        return {(m(s), m(p), m(o)) for s, p, o in g}

    cands = [p for p in sorted(corpus.rglob("*"))
             if p.is_file() and p.suffix.lower() in RDF_EXT
             and 2_000 < p.stat().st_size < 3_000_000]
    if not cands:
        print("[!!] self-test: no suitably sized file found")
        return False
    step = max(1, len(cands) // n)
    sample = cands[::step][:n]

    ok = True
    n_compared = 0
    for src in sample:
        g = Graph()
        try:
            g.parse(src.as_posix())  # exactly how odpqual.parse_graph reads
        except Exception as exc:  # noqa: BLE001
            print(f"  !!  {src.name:<44} reference parse failed: "
                  f"{type(exc).__name__} -- cannot compare")
            ok = False
            continue
        want = reference(g)

        out = src.parent / (src.stem + ".selftest.nt")
        got_n, err = hash_one(src, out, sink_cls)
        if err:
            print(f"[!!] self-test parse failed: {src.name}: {err}")
            ok = False
            continue
        back = Graph()
        back.parse(out.as_posix(), format="nt")
        got = set(back)
        try:
            out.unlink()
        except OSError:
            pass

        # Blank-node labels are arbitrary: rdflib mints fresh random ones on
        # every parse, so hashing them cannot agree across two runs and set
        # equality is the wrong criterion. Ground triples (no blank node) must
        # match exactly; the whole graph must be isomorphic.
        ground_w = {t for t in want if not any(isinstance(x, BNode) for x in t)}
        ground_g = {t for t in got if not any(isinstance(x, BNode) for x in t)}
        gw, gg = Graph(), Graph()
        for t in want:
            gw.add(t)
        for t in got:
            gg.add(t)

        exact = ground_w == ground_g
        iso = isomorphic(gw, gg)
        same = exact and iso
        ok &= same
        n_compared += 1
        n_bn = len(want) - len(ground_w)
        print(f"  {'OK ' if same else '!! '} {src.name:<44} "
              f"{len(want):>6} triples ({n_bn:>4} touch a bnode)  "
              f"ground-exact={'y' if exact else 'N'} isomorphic={'y' if iso else 'N'}"
              + ("" if same else
                 f"   MISMATCH: {len(ground_w - ground_g)} missing, "
                 f"{len(ground_g - ground_w)} extra"))

    if n_compared == 0:
        print("  !! nothing was actually compared -- the pass would be vacuous")
        return False
    print(f"  ({n_compared} files compared)")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="the plain corpus directory, e.g. release-iri/corpus")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <corpus>-hashed)")
    ap.add_argument("--force", action="store_true",
                    help="re-hash files whose output already exists")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (smoke test)")
    ap.add_argument("--self-test", action="store_true",
                    help="check the streaming transform against the in-memory one and exit")
    args = ap.parse_args()

    corpus: Path = args.corpus
    if not corpus.is_dir():
        sys.exit(f"not a directory: {corpus}")
    out: Path = args.out or corpus.parent / f"{corpus.name}-hashed"

    sink_cls = make_sink_class()

    if args.self_test:
        print(f"== self-test: streaming vs in-memory transform ({corpus}) ==")
        ok = self_test(corpus, sink_cls)
        print("  ->", "identical on every sampled file" if ok else "DIVERGENCE")
        sys.exit(0 if ok else 1)

    # Enumerate through the \\?\ prefix. A bare rglob calls is_file() on paths
    # past MAX_PATH; pathlib swallows the resulting OSError and returns False, so
    # 13 files -- the longest GitHub-derived ids -- were dropped from the work
    # list with no error, and a coverage check that enumerated the same way
    # confirmed itself. Both sides must use the prefix or neither is trustworthy.
    base = Path(longpath(corpus))
    cands = [p for p in sorted(base.rglob("*"))
             if p.is_file() and p.suffix.lower() in RDF_EXT]
    if args.limit:
        cands = cands[:args.limit]
    total_bytes = sum(p.stat().st_size for p in cands)
    print(f"== hash_corpus: {len(cands):,} RDF files, "
          f"{total_bytes / 2**30:.2f} GiB -> {out}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    fmt_rows: list[dict] = []
    n_done = n_skip = n_triples = 0
    t0 = time.time()

    for i, src in enumerate(cands, 1):
        rel = src.relative_to(base)   # base is the \?\-prefixed corpus
        dst = out / rel.with_suffix(".nt")
        fmt_rows.append({"file_hashed": str(dst.relative_to(out)),
                         "file_plain": str(rel),
                         "original_format": src.suffix.lower().lstrip("."),
                         "bytes_plain": src.stat().st_size})

        if not args.force and Path(longpath(dst)).exists():
            n_skip += 1
        else:
            n, err = hash_one(src, dst, sink_cls)
            if err:
                failures.append({"file": str(rel), "reason": "parse_error",
                                 "detail": err})
            else:
                n_done += 1
                n_triples += n

        if i % 200 == 0 or i == len(cands):
            el = time.time() - t0
            print(f"[..] {i:>5}/{len(cands)}  hashed={n_done} resumed={n_skip} "
                  f"failed={len(failures)}  {n_triples:,} triples  "
                  f"{el / 60:.1f} min", flush=True)

    with (out / "format_map.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file_hashed", "file_plain", "original_format",
                                "bytes_plain"])
        w.writeheader()
        w.writerows(fmt_rows)

    if failures:
        with (out / "hash_failures.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["file", "reason", "detail"])
            w.writeheader()
            w.writerows(failures)

    (out / "README.md").write_text(f"""\
# {out.name}

IRI-anonymised twin of [`{corpus.name}`](../{corpus.name}). **Same content, same
reuse relations** -- only identifiers differ.

Every IRI is replaced by `{HASH_PREFIX}` + URL-safe base64 of its SHA-1, the same
function `build_release.py` uses for the hashed views, so all parts of the
release interoperate: an entity reused by two ontologies keeps one identifier
here too. RDF, RDFS, OWL and XSD IRIs are left intact so the files stay valid
OWL. Blank-node labels are hashed deterministically. Literals are untouched.

## Why an anonymised twin exists

A detection method evaluated on the plain corpus can cheat: it can recognise
`ontologydesignpatterns.org` namespace prefixes instead of learning structure.
Here that shortcut is gone, so a score reflects structural and semantic signal.
Use this form to **evaluate** detection methods and the plain form to **inspect**
or **curate**.

## Two things to know before using it

**Serialisation.** Every file is N-Triples, whatever the original was. The
transform streams -- it never holds a whole graph -- because the OBO portion
includes single files up to 1.1 GiB that do not fit in memory as an rdflib graph.
Re-serialising RDF/XML would require exactly the graph being avoided.
`format_map.csv` gives the original extension and byte size for every file.

**Paths are not hashed.** Directory and file names still carry ontology ids, so
this tree aligns with `../{corpus.name}`, `../MANIFEST.csv` and the view tables.
The anonymisation covers identifiers *inside* the files. If an experiment must
not see ontology identity either, read through `format_map.csv` and rename.

Files: {n_done + n_skip} of {len(cands)} rewritten, {n_triples:,} triples\
{f", {len(failures)} could not be parsed (see `hash_failures.csv`)" if failures else ""}.
""", encoding="utf-8")

    el = (time.time() - t0) / 60
    print(f"[OK] {out.name}: {n_done} hashed, {n_skip} resumed, "
          f"{len(failures)} failures, {n_triples:,} triples, {el:.1f} min")


if __name__ == "__main__":
    main()
