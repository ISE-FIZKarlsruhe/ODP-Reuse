"""write_release_readme regenerates the release README from the release's tables.

The merge job runs it in CI, where the hashed corpus (a Zenodo-only file tree)
is absent. Without it the README's paragraph about `corpus-hashed/` would claim
0 files, so --keep-hashed-note keeps the paragraph already there.
"""
import shutil
from pathlib import Path

import pytest

import write_release_readme as wrr

RELEASE = Path(__file__).resolve().parents[2] / "release-iri"
TABLES = ["MANIFEST.csv", "README.md", "dataset_statistics.csv", "gate_summary.csv",
          "validate_release.py", "zenodo.json"]


def norm(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.skipif(not (RELEASE / "MANIFEST.csv").is_file(), reason="release not checked out")
def test_regenerating_without_the_hashed_corpus_keeps_the_readme(tmp_path):
    for f in TABLES:
        shutil.copy(RELEASE / f, tmp_path / f)

    wrr.main(["--release", str(tmp_path), "--keep-hashed-note"])

    for f in ("README.md", "zenodo.json", "validate_release.py"):
        assert norm(tmp_path / f) == norm(RELEASE / f), f
