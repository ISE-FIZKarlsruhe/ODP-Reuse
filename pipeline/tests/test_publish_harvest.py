"""publish_harvest: turn a merged harvest into a new, fully recorded version.

It runs unattended, so it carries the checks a reviewer would have made: the
update only adds (every released row, pair and audit record is still there,
unchanged, and every new row is one the merge said it added), every added file
matches the hash the gate recorded, and the version, changelog and history are
written from the same numbers.
"""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import build_manifest
import publish_harvest as ph

COLS = build_manifest.CSV_COLUMNS
PAIR_COLS = ["ontology_id", "ontology_dir", "pattern", "pattern_file", "n_pattern_entities",
             "n_reused_entities", "coverage_pct", "complete_reuse", "n_label_entities"]


def write(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def mrow(oid, **kw):
    r = {c: "" for c in COLS}
    r.update({"ontology_id": oid, "dataset": "github", "content_sha256": "s-" + oid,
              "n_gt_patterns": "1"}, **kw)
    return r


def pair(oid, pattern="p"):
    return {"ontology_id": oid, "ontology_dir": f"../../corpus/github/{oid}", "pattern": pattern,
            "pattern_file": f"../../corpus/github/{oid}/patterns/{pattern}.owl",
            "n_pattern_entities": "2", "n_reused_entities": "1", "coverage_pct": "50.0",
            "complete_reuse": "0", "n_label_entities": "0"}


def release(root: Path, rows, pairs, audit=()):
    write(root / "MANIFEST.csv", rows, COLS)
    write(root / "views" / "github-reuse" / "pairs.csv", pairs, PAIR_COLS)
    write(root / "audit" / "removed_G5_has_gt.csv", list(audit),
          ["dataset", "ontology_id", "verified"])
    return root


# ------------------------------------------------------------ additive check

@pytest.fixture
def old(tmp_path):
    return release(tmp_path / "old", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x")],
                   [pair("a")], [{"dataset": "github", "ontology_id": "b", "verified": "ok"}])


def test_an_update_that_only_adds_passes(tmp_path, old):
    new = release(tmp_path / "new", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x"), mrow("c")],
                  [pair("a"), pair("c")], [{"dataset": "github", "ontology_id": "b", "verified": "ok"}])

    assert ph.check_additive(old, new, {("github", "c")}) == []


def test_recomputed_view_statistics_may_change(tmp_path, old):
    # statistics.csv and top_patterns.csv are summaries recomputed from the
    # tables; adding ontologies changes them by design.
    for d, n in ((old, "1"), (tmp_path / "new", "2")):
        write(d / "views" / "github-reuse" / "statistics.csv", [{"n_ontologies": n}], ["n_ontologies"])
        write(d / "views" / "github-reuse" / "top_patterns.csv",
              [{"rank": "1", "pattern": "p", "n_ontologies": n}], ["rank", "pattern", "n_ontologies"])
    new = release(tmp_path / "new", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x"), mrow("c")],
                  [pair("a"), pair("c")], [{"dataset": "github", "ontology_id": "b", "verified": "ok"}])

    assert ph.check_additive(old, new, {("github", "c")}) == []


def test_a_changed_release_row_is_refused(tmp_path, old):
    new = release(tmp_path / "new", [mrow("a", n_gt_patterns="2"), mrow("b", drop_reason="G5_has_gt:x")],
                  [pair("a")], [{"dataset": "github", "ontology_id": "b", "verified": "ok"}])

    problems = ph.check_additive(old, new, set())

    assert any("MANIFEST.csv" in p and "github/a" in p for p in problems)


def test_a_removed_pair_or_audit_record_is_refused(tmp_path, old):
    new = release(tmp_path / "new", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x")], [], [])

    problems = ph.check_additive(old, new, set())

    assert any("pairs.csv" in p for p in problems)
    assert any("removed_G5_has_gt.csv" in p for p in problems)


def test_ids_differing_only_in_letter_case_are_refused(tmp_path, old):
    # Harvest run 42415403905 was merged before merge_harvest.py checked this
    # and added three such ids; an archive with both directories cannot be
    # unpacked on Windows or macOS.
    new = release(tmp_path / "new", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x"), mrow("A")],
                  [pair("a")], [{"dataset": "github", "ontology_id": "b", "verified": "ok"}])

    problems = ph.check_additive(old, new, {("github", "A")})

    assert any("letter case" in p and "github/A" in p for p in problems)


def test_rows_the_merge_did_not_report_are_refused(tmp_path, old):
    new = release(tmp_path / "new", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x"), mrow("c")],
                  [pair("a"), pair("c")], [{"dataset": "github", "ontology_id": "b", "verified": "ok"}])

    problems = ph.check_additive(old, new, set())

    assert any("github/c" in p for p in problems)


# ----------------------------------------------------------------- versions

def test_a_harvest_is_a_minor_version():
    assert ph.next_minor("1.2.0") == "1.3.0"
    assert ph.next_minor("1.9.4") == "1.10.0"


def test_citation_gets_the_version_and_date():
    text = "cff-version: 1.2.0\nversion: 1.2.0\ndate-released: '2026-08-05'\nurl: x\n"

    out = ph.set_citation(text, "1.3.0", "2026-12-01")

    assert "version: 1.3.0\n" in out and "date-released: '2026-12-01'\n" in out
    assert out.startswith("cff-version: 1.2.0\n")


VIEW_STATS = [{"dataset": v, "n_ontologies": n, "n_pairs": p} for v, n, p in (
    ("github-reuse", "522", "771"), ("github-complete", "40", "106"),
    ("noOBO-reuse", "51", "90"), ("noOBO-complete", "6", "16"),
    ("OBO-reuse", "8", "758"), ("OBO-complete", "8", "758"))]


def history_release(root: Path):
    write(root / "MANIFEST.csv", [mrow("a"), mrow("b", drop_reason="G5_has_gt:x"), mrow("c")], COLS)
    write(root / "dataset_statistics.csv", VIEW_STATS, ["dataset", "n_ontologies", "n_pairs"])
    return root


def test_a_history_row_records_the_version_and_what_it_added(tmp_path):
    rel = history_release(tmp_path / "rel")
    prev = {"version": "1.2.0", "github-reuse_ontologies": "442", "github-reuse_pairs": "669"}

    row = ph.history_row(rel, version="1.3.0", date="2026-12-01", tag="v1.3.0",
                         record="99", doi="10.5281/zenodo.99", previous=prev,
                         added=[mrow("c"), mrow("d", drop_reason="G4_substantive:x")],
                         harvest_run="https://run/1")

    assert (row["discovered"], row["kept"]) == (3, 2)
    assert (row["github-reuse_ontologies"], row["github-reuse_pairs"]) == (522, 771)
    assert (row["added_ontologies"], row["added_kept"], row["added_pairs"]) == (2, 1, 102)
    assert row["doi"] == "10.5281/zenodo.99"


def test_the_changelog_entry_states_what_was_added_and_that_nothing_was_removed():
    row = {"version": "1.3.0", "date": "2026-12-01", "doi": "10.5281/zenodo.99",
           "harvest_run": "https://run/1", "added_ontologies": 828, "added_kept": 80,
           "added_pairs": 102, "github-reuse_ontologies": 522, "github-reuse_pairs": 771}
    prev = {"github-reuse_ontologies": "442", "github-reuse_pairs": "669"}

    entry = ph.changelog_entry(row, prev)

    assert entry.startswith("## v1.3.0 — 2026-12-01\n")
    assert "+80 ontologies" in entry and "+102 ontology–pattern pairs" in entry
    assert "| `github-reuse` | 442 → 522 (+80) | 669 → 771 (+102) |" in entry
    assert "No ontology, file or verdict of an earlier version was removed or changed" in entry
    assert "[10.5281/zenodo.99](https://doi.org/10.5281/zenodo.99)" in entry
    assert "additions/v1.3.0.csv" in entry


def test_a_new_entry_goes_above_the_older_ones(tmp_path):
    p = tmp_path / "CHANGELOG.md"
    p.write_text("# Changelog\n\nIntro.\n\n## v1.2.0 — 2026-08-05\n\nOld.\n", encoding="utf-8")

    ph.prepend_changelog(p, "## v1.3.0 — 2026-12-01\n\nNew.\n")

    text = p.read_text(encoding="utf-8")
    assert text.index("## v1.3.0") < text.index("## v1.2.0")
    assert text.startswith("# Changelog\n\nIntro.\n\n## v1.3.0")


def test_recording_a_version_writes_every_record_from_the_same_numbers(tmp_path):
    rel = history_release(tmp_path / "rel")
    prev = {c: "" for c in ph.HISTORY_COLUMNS}
    prev.update({"version": "1.2.0", "date": "2026-08-05", "github-reuse_ontologies": "442",
                 "github-reuse_pairs": "669"})
    write(rel / "history.csv", [prev], ph.HISTORY_COLUMNS)
    added = write(tmp_path / "added.csv", [mrow("c")], COLS) or tmp_path / "added.csv"
    (tmp_path / "draft.json").write_text(json.dumps(
        {"draft_id": 99, "doi": "10.5281/zenodo.99", "concept": 17751740}), encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("version: 1.2.0\ndate-released: '2026-08-05'\n",
                                           encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\nIntro.\n\n## v1.2.0 — x\n", encoding="utf-8")
    args = ["--release", str(rel), "--version", "1.3.0", "--date", "2026-12-01",
            "--added", str(added), "--harvest-run", "https://run/1"]

    assert ph.main(["notes", *args, "--out", str(tmp_path / "notes.txt")]) == 0
    assert ph.main(["record", *args, "--citation", str(tmp_path / "CITATION.cff"),
                    "--changelog", str(tmp_path / "CHANGELOG.md"),
                    "--draft", str(tmp_path / "draft.json"),
                    "--notes-out", str(tmp_path / "release-notes.md")]) == 0

    last = ph.read_csv(rel / "history.csv")[-1]
    assert (last["version"], last["doi"], last["github-reuse_pairs"]) == ("1.3.0", "10.5281/zenodo.99", "771")
    assert json.loads((rel / "archive.json").read_text(encoding="utf-8"))["release_tag"] == "v1.3.0"
    assert ph.read_csv(rel / "additions" / "v1.3.0.csv")[0]["ontology_id"] == "c"
    assert "version: 1.3.0" in (tmp_path / "CITATION.cff").read_text(encoding="utf-8")
    assert "## v1.3.0 — 2026-12-01" in (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert (tmp_path / "release-notes.md").read_text(encoding="utf-8").startswith("## v1.3.0")
    assert "adds 1 ontology to the corpus" in (tmp_path / "notes.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------ corpus

def onto_dir(root: Path, oid: str, body: bytes) -> Path:
    d = root / "github" / oid
    (d / "patterns").mkdir(parents=True)
    (d / f"{oid}.owl").write_bytes(body)
    (d / "patterns" / "p.owl").write_bytes(b"pattern")
    return d


def test_added_files_must_match_the_recorded_hash_and_be_new(tmp_path):
    adds, corpus = tmp_path / "adds", tmp_path / "corpus"
    onto_dir(adds, "c", b"C")
    onto_dir(adds, "d", b"D")
    onto_dir(corpus, "d", b"old D")
    rows = [mrow("c", content_sha256=hashlib.sha256(b"C").hexdigest()),
            mrow("d", content_sha256=hashlib.sha256(b"D").hexdigest()),
            mrow("e", content_sha256="whatever")]

    problems = ph.add_files(adds, corpus, rows)

    assert any("github/d" in p and "already" in p for p in problems)
    assert any("github/e" in p and "missing" in p for p in problems)
    assert not (corpus / "github" / "c").exists()      # nothing copied when anything is wrong


def test_added_files_are_copied_into_the_corpus(tmp_path):
    adds, corpus = tmp_path / "adds", tmp_path / "corpus"
    onto_dir(adds, "c", b"C")
    (corpus / "github").mkdir(parents=True)

    assert ph.add_files(adds, corpus, [mrow("c", content_sha256=hashlib.sha256(b"C").hexdigest())]) == []

    assert (corpus / "github" / "c" / "patterns" / "p.owl").read_bytes() == b"pattern"


def test_hashed_additions_keep_the_existing_format_map_rows(tmp_path):
    hashed, new = tmp_path / "corpus-hashed", tmp_path / "new"
    hashed.mkdir()
    (hashed / "format_map.csv").write_bytes(
        b"file_hashed,file_plain,original_format,bytes_plain\r\n"
        b"github\\a\\a.nt,github\\a\\a.owl,owl,10\r\n")
    (new / "github" / "c").mkdir(parents=True)
    (new / "github" / "c" / "c.nt").write_text("<x> <y> <z> .\n", encoding="utf-8")
    (new / "format_map.csv").write_text(
        "file_hashed,file_plain,original_format,bytes_plain\n"
        "github/c/c.nt,github/c/c.ttl,ttl,5\n", encoding="utf-8")
    (new / "hash_failures.csv").write_text("file,reason,detail\ngithub/c/bad.owl,parse_error,x\n",
                                           encoding="utf-8")

    ph.merge_hashed(new, hashed)

    fmap = (hashed / "format_map.csv").read_bytes().decode("utf-8")
    assert fmap.startswith("file_hashed,file_plain,original_format,bytes_plain\r\n"
                           "github\\a\\a.nt,github\\a\\a.owl,owl,10\r\n")
    assert "github\\c\\c.nt,github\\c\\c.ttl,ttl,5\r\n" in fmap
    assert (hashed / "github" / "c" / "c.nt").is_file()
    assert "github\\c\\bad.owl,parse_error,x" in (hashed / "hash_failures.csv").read_text(encoding="utf-8")


def test_zip_tree_writes_the_layout_the_archive_uses(tmp_path):
    root = tmp_path / "corpus"
    onto_dir(root, "c", b"C")

    ph.zip_tree(root, "corpus", tmp_path / "corpus.zip")

    names = zipfile.ZipFile(tmp_path / "corpus.zip").namelist()
    assert names == ["corpus/", "corpus/github/", "corpus/github/c/", "corpus/github/c/c.owl",
                     "corpus/github/c/patterns/", "corpus/github/c/patterns/p.owl"]


# --------------------------------------------------------- the real release

REL = Path(__file__).resolve().parents[2] / "release-iri"


@pytest.mark.skipif(not (REL / "history.csv").is_file(), reason="release not checked out")
def test_the_recorded_history_matches_the_current_release():
    last = ph.read_csv(REL / "history.csv")[-1]
    archive = json.loads((REL / "archive.json").read_text(encoding="utf-8"))
    rows = ph.read_csv(REL / "MANIFEST.csv")

    assert last["tag"] == archive["release_tag"]
    assert last["zenodo_record"] == archive["zenodo_record"]
    assert int(last["discovered"]) == len(rows)
    assert int(last["kept"]) == sum(1 for r in rows if not r["drop_reason"])
    for s in ph.read_csv(REL / "dataset_statistics.csv"):
        assert int(last[f"{s['dataset']}_ontologies"]) == int(s["n_ontologies"])
        assert int(last[f"{s['dataset']}_pairs"]) == int(s["n_pairs"])
