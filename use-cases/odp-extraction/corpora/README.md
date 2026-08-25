# Corpora

Working copies of ontology collections the extraction methods are run over.
**Everything in this directory except this file and `.gitignore` is ignored by
git**: each collection has an upstream home and its own licence, and nothing is
redistributed here. The copies exist so that a run is reproducible from one
command without depending on paths outside the repository.

| directory | what it is | how to re-create |
|---|---|---|
| `mseo/` | 108 materials-science and engineering ontologies (EMMO, PMDco, BFO/IOF, CCO, BWMD, MSEO and relatives). **No ground truth**: there is no `patterns/` directory, so nothing here can be scored. | `cp -r <mseo.github.io>/Ontology_files corpora/mseo` from <https://github.com/ebrahimnorouzi/mseo.github.io> |

The ODP-Reuse corpus itself is *not* copied here — it is read from its own
release directory (`release-iri/corpus`, `release-iri/corpus-hashed`) or from
the harvest tree, via `--manifest` and `--data-root`.

## Running

One command covers both cases, because ground truth is optional:

```bash
# a corpus with no answer key: methods run, predictions are written, nothing
# is scored
./run_extraction.sh --corpus corpora/mseo --out results/mseo

# the ODP-Reuse corpus: the same methods, scored against the gate's ground truth
./run_extraction.sh --manifest ../../pipeline/out/MANIFEST.csv \
    --data-root <MultiSource> --out results/methods-iri \
    --datasets "github noOBO"
```

An ontology is scored when its directory carries a `patterns/` subdirectory
whose entities overlap it, and is predicted-only otherwise. A run reports
which ontologies fell into which case; it never reports a score of zero for an
ontology that simply had nothing to be scored against.
