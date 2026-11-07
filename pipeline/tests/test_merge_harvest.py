"""Tests for merge_harvest: a harvest may add to the release, never remove.

A re-harvest is a partial sample of GitHub. Run 42239149156 did not return 525
release ontologies, of which 88 in a sample of 100 were still at the same path,
so replacing the release with a harvest would have dropped 117 gate-passing
ontologies for no reason. The merge keeps every release row exactly as it is,
adds only ontologies the release does not have, and gates those against the
release (a new copy of a released file is a duplicate of the released one).
"""
import csv
import hashlib
import shutil
from pathlib import Path

import pytest

import build_manifest
import merge_harvest as mh

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "release-iri"

MCOLS = build_manifest.CSV_COLUMNS


def mrow(oid, ds="github", **kw):
    base = {c: "" for c in MCOLS}
    base.update({"ontology_id": oid, "dataset": ds, "path": f"/x/{oid}",
                 "content_sha256": "sha-" + oid, "n_pattern_files": "1", "parsed": "ok",
                 "n_triples": "10", "n_entities_narrow": "5", "n_entities_wide": "5",
                 "n_own_entities": "3", "n_gt_patterns": "1", "n_complete_patterns": "0",
                 "max_reuse_pct": "50.0", "n_label_confirmed": "0", "n_label_entities": "0",
                 "stage_reached": "has_gt", "drop_reason": "", "n_pattern_errors": "0"})
    base.update(kw)
    return base


def ids(rows):
    return [r["ontology_id"] for r in rows]


# ------------------------------------------------------------- the manifest

def test_every_release_row_is_kept_unchanged_and_in_place():
    release = [mrow("a"), mrow("b", drop_reason="G5_has_gt:no_pattern_overlap",
                               stage_reached="substantive", n_gt_patterns="0")]
    harvest = [mrow("a", n_gt_patterns="7", n_triples="99"), mrow("c")]

    m = mh.merge_manifest(release, harvest)

    assert m.rows[:2] == release
    assert ids(m.rows) == ["a", "b", "c"]


def test_release_ontologies_the_harvest_did_not_return_are_kept():
    release = [mrow("a"), mrow("b"), mrow("go", ds="OBO")]
    harvest = [mrow("a")]

    m = mh.merge_manifest(release, harvest)

    assert ids(m.rows) == ["a", "b", "go"]
    assert m.not_returned == [("github", "b")]   # OBO is not harvested, so not listed


def test_new_rows_go_after_the_last_release_row_of_their_portion():
    release = [mrow("g1"), mrow("n1", ds="noOBO")]
    harvest = [mrow("g3"), mrow("g2")]

    m = mh.merge_manifest(release, harvest)

    assert ids(m.rows) == ["g1", "g2", "g3", "n1"]
    assert ids(m.added) == ["g2", "g3"]


def test_a_new_copy_of_a_released_file_is_a_duplicate_of_the_released_row():
    # "0copy" sorts before "a", so re-gating the union would make the RELEASED
    # row the duplicate. The released row wins: it is already published.
    release = [mrow("a", content_sha256="X"),
               mrow("b", content_sha256="X", duplicate_of="a",
                    drop_reason="G3_duplicate:of(a)", stage_reached="parsed")]
    harvest = [mrow("0copy", content_sha256="X")]

    m = mh.merge_manifest(release, harvest)

    new = m.added[0]
    assert new["drop_reason"] == "G3_duplicate:of(a)"
    assert new["duplicate_of"] == "a"
    assert new["stage_reached"] == "parsed"
    assert m.rows[:2] == release


def test_new_rows_are_regated_against_the_release_not_the_harvest():
    # The harvest dropped "x" as a copy of "c", but the release holds an older
    # version of "c" (different hash), so in the release "x" is no duplicate.
    release = [mrow("c", content_sha256="OLD")]
    harvest = [mrow("c", content_sha256="NEW"),
               mrow("x", content_sha256="NEW", duplicate_of="c",
                    drop_reason="G3_duplicate:of(c)", stage_reached="parsed"),
               mrow("y", content_sha256="NEW2", n_own_entities="0", duplicate_of="c",
                    drop_reason="G3_duplicate:of(c)", stage_reached="parsed")]

    m = mh.merge_manifest(release, harvest)

    by = {r["ontology_id"]: r for r in m.added}
    assert by["x"]["drop_reason"] == "" and by["x"]["duplicate_of"] == ""
    assert by["x"]["stage_reached"] == "has_gt"
    assert by["y"]["drop_reason"].startswith("G4_substantive")
    assert m.changed_upstream == [("github", "c")]


def test_two_new_copies_of_one_file_keep_the_first_id():
    harvest = [mrow("q", content_sha256="S"), mrow("p", content_sha256="S")]

    m = mh.merge_manifest([], harvest)

    by = {r["ontology_id"]: r for r in m.added}
    assert by["p"]["drop_reason"] == ""
    assert by["q"]["drop_reason"] == "G3_duplicate:of(p)"


def test_parse_and_pattern_copy_verdicts_of_new_rows_stand():
    harvest = [mrow("bad", parsed="BadSyntax", drop_reason="G1_parse:BadSyntax",
                    stage_reached="discovered", content_sha256="S1"),
               mrow("copy", drop_reason="G2_pattern_copy:identical_file",
                    stage_reached="parsed", content_sha256="S2")]

    m = mh.merge_manifest([], harvest)

    assert [r["drop_reason"] for r in m.added] == ["G1_parse:BadSyntax",
                                                    "G2_pattern_copy:identical_file"]
    assert [r["stage_reached"] for r in m.added] == ["discovered", "parsed"]


def test_more_pairs_on_a_released_file_are_reported_but_not_merged():
    # Adding them means re-gating the archived file with the new patterns, which
    # needs the archived corpus; until then they are candidates.
    release = [mrow("a", content_sha256="X", n_gt_patterns="1")]
    harvest = [mrow("a", content_sha256="X", n_gt_patterns="3")]

    m = mh.merge_manifest(release, harvest)

    assert m.rows == release
    assert m.gained == [("github", "a", 1, 3)]


# ---------------------------------------------------- views, from the tables

def view_dirs():
    return sorted(d for d in (RELEASE / "views").iterdir() if d.is_dir())


@pytest.mark.skipif(not (RELEASE / "views").is_dir(), reason="release not checked out")
@pytest.mark.parametrize("view", view_dirs() if (RELEASE / "views").is_dir() else [],
                         ids=lambda d: d.name)
def test_rewriting_a_view_from_its_own_tables_changes_nothing(tmp_path, view):
    # The merge recomputes statistics from pairs.csv and ontologies.csv rather
    # than from the corpus. On an unchanged view that must give back every file.
    dest = tmp_path / view.name
    shutil.copytree(view, dest)

    mh.rewrite_view(dest)

    for f in sorted(view.iterdir()):
        assert norm(dest / f.name) == norm(f), f.name


def norm(p: Path) -> str:
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


@pytest.mark.skipif(not (RELEASE / "MANIFEST.csv").is_file(), reason="release not checked out")
def test_gate_summary_recomputed_from_the_manifest_matches_the_release(tmp_path):
    rows = mh.read_csv(RELEASE / "MANIFEST.csv")

    mh.write_gate_summary(rows, tmp_path)

    for f in ("gate_summary.csv", "gate_summary.md"):
        assert norm(tmp_path / f) == norm(RELEASE / f), f


@pytest.mark.skipif(not (RELEASE / "audit").is_dir(), reason="release not checked out")
def test_audit_summary_recomputed_from_the_removal_files_matches_the_release(tmp_path):
    shutil.copytree(RELEASE / "audit", tmp_path / "audit")

    mh.append_audit(tmp_path / "audit", {})

    for f in sorted((RELEASE / "audit").iterdir()):
        assert norm(tmp_path / "audit" / f.name) == norm(f), f.name


# ------------------------------------------------------- end to end, on RDF

OWL = "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
PATTERN = OWL + ("<http://odp.org/p> a owl:Ontology .\n"
                 "<http://odp.org/p#A> a owl:Class .\n<http://odp.org/p#B> a owl:Class .\n")


def onto(*classes):
    return OWL + "<http://ex.org/o> a owl:Ontology .\n" + "".join(
        f"<{c}> a owl:Class .\n" for c in classes)


def put(root: Path, oid: str, text: str, patterns: dict | None = None) -> Path:
    d = root / "github_dataset" / oid
    (d / "patterns").mkdir(parents=True)
    (d / f"{oid}.ttl").write_text(text, encoding="utf-8")
    for name, body in (patterns or {"p": PATTERN}).items():
        (d / "patterns" / f"{name}.ttl").write_text(body, encoding="utf-8")
    return d


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_rows(path: Path, rows, cols=MCOLS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


VIEW_PAIR_COLS = ["ontology_id", "ontology_dir", "pattern", "pattern_file", "n_pattern_entities",
                  "n_reused_entities", "coverage_pct", "complete_reuse", "n_label_entities"]
VIEW_ONTO_COLS = ["ontology_id", "dataset", "n_pattern_files", "n_triples", "n_entities_narrow",
                  "n_entities_wide", "n_own_entities", "n_gt_patterns", "n_complete_patterns",
                  "max_reuse_pct"]


@pytest.fixture
def world(tmp_path):
    """A one-ontology release, and a harvest that finds it again plus three new ones."""
    release, harvest = tmp_path / "release", tmp_path / "harvest"
    old = put(tmp_path / "archive", "old", onto("http://odp.org/p#A", "http://ex.org/o#Mine"))
    rel_rows = [mrow("old", content_sha256=sha(old / "old.ttl"), n_entities_wide="2",
                     n_own_entities="1", max_reuse_pct="50.0")]
    write_rows(release / "MANIFEST.csv", rel_rows)
    old_pair = {"ontology_id": "old", "ontology_dir": "../../corpus/github/old", "pattern": "p",
                "pattern_file": "../../corpus/github/old/patterns/p.ttl",
                "n_pattern_entities": 2, "n_reused_entities": 1, "coverage_pct": 50.0,
                "complete_reuse": 0, "n_label_entities": 0}
    onto_row = {k: rel_rows[0][k] for k in VIEW_ONTO_COLS}
    for level in ("reuse", "complete"):
        v = release / "views" / f"github-{level}"
        write_rows(v / "pairs.csv", [old_pair] if level == "reuse" else [], VIEW_PAIR_COLS)
        write_rows(v / "ontologies.csv", [onto_row] if level == "reuse" else [], VIEW_ONTO_COLS)
        mh.rewrite_view(v)
    stats = [{"dataset": f"github-{lv}", **mh.read_csv(release / "views" / f"github-{lv}" /
                                                       "statistics.csv")[0]}
             for lv in ("reuse", "complete")]
    write_rows(release / "dataset_statistics.csv", stats, list(stats[0]))
    mh.write_gate_summary(rel_rows, release)
    (release / "audit").mkdir()
    write_rows(release / "audit" / "audit_summary.csv",
               [], ["n_removed", "rule", "ok"])

    # harvest: "old" again, a complete reuser, a non-reuser, a copy of "old"
    put(harvest, "old", onto("http://odp.org/p#A", "http://ex.org/o#Mine"))
    full = put(harvest, "full", onto("http://odp.org/p#A", "http://odp.org/p#B", "http://ex.org/o#F"))
    none = put(harvest, "none", onto("http://ex.org/o#Z"))
    copy = put(harvest, "acopy", onto("http://odp.org/p#A", "http://ex.org/o#Mine"))
    hrows = [
        mrow("old", content_sha256=sha(old / "old.ttl")),
        mrow("full", content_sha256=sha(full / "full.ttl"), n_entities_wide="3",
             n_own_entities="1", n_gt_patterns="1", n_complete_patterns="1",
             max_reuse_pct="100.0", stage_reached="complete_reuse"),
        mrow("none", content_sha256=sha(none / "none.ttl"), n_entities_wide="1",
             n_own_entities="1", n_gt_patterns="0", max_reuse_pct="0.0",
             drop_reason="G5_has_gt:no_pattern_overlap", stage_reached="substantive"),
        mrow("acopy", content_sha256=sha(copy / "acopy.ttl"), n_entities_wide="2",
             n_own_entities="1"),
    ]
    write_rows(harvest / "out" / "MANIFEST.csv", hrows)
    return tmp_path, release, harvest


def run_merge(tmp_path, release, harvest):
    rc = mh.main(["--release", str(release), "--harvest-manifest", str(harvest / "out" / "MANIFEST.csv"),
                  "--harvest-root", str(harvest), "--additions", str(tmp_path / "additions"),
                  "--report", str(tmp_path / "merge")])
    return rc


def test_merge_adds_new_ontologies_to_the_manifest_and_views(world):
    tmp_path, release, harvest = world

    assert run_merge(tmp_path, release, harvest) == 0

    man = mh.read_csv(release / "MANIFEST.csv")
    assert ids(man) == ["old", "acopy", "full", "none"]
    by = {r["ontology_id"]: r for r in man}
    assert by["acopy"]["drop_reason"] == "G3_duplicate:of(old)"
    reuse = mh.read_csv(release / "views" / "github-reuse" / "pairs.csv")
    assert ids(reuse) == ["full", "old"]
    assert reuse[0]["complete_reuse"] == "1"
    assert reuse[0]["ontology_dir"] == "../../corpus/github/full"
    assert ids(mh.read_csv(release / "views" / "github-complete" / "ontologies.csv")) == ["full"]
    stats = {r["dataset"]: r for r in mh.read_csv(release / "dataset_statistics.csv")}
    assert stats["github-reuse"]["n_ontologies"] == "2"
    assert stats["github-reuse"]["n_pairs"] == "2"
    assert stats["github-complete"]["n_complete_pairs"] == "1"
    gs = mh.read_csv(release / "gate_summary.csv")[0]
    assert (gs["discovered"], gs["kept_reuse"], gs["dropped_G3_duplicate"]) == ("4", "2", "1")


def test_merge_writes_lf_line_endings(world):
    # The index stores LF. CRLF from the csv module's default would make every
    # line of a rewritten table show as changed in the pull request.
    tmp_path, release, harvest = world
    run_merge(tmp_path, release, harvest)

    for f in ("MANIFEST.csv", "views/github-reuse/pairs.csv", "gate_summary.md"):
        assert b"\r\n" not in (release / f).read_bytes(), f


def test_new_removals_are_reverified_and_added_to_the_audit(world):
    tmp_path, release, harvest = world
    run_merge(tmp_path, release, harvest)

    g3 = mh.read_csv(release / "audit" / "removed_G3_duplicate.csv")
    g5 = mh.read_csv(release / "audit" / "removed_G5_has_gt.csv")
    assert [(r["ontology_id"], r["verified"]) for r in g3] == [("acopy", "ok")]
    assert "recorded" in g3[0]["evidence"] and g3[0]["kept_instead"] == "old"
    assert [(r["ontology_id"], r["verified"]) for r in g5] == [("none", "ok")]
    summary = {r["rule"]: r for r in mh.read_csv(release / "audit" / "audit_summary.csv")}
    assert summary["G5_has_gt"]["n_removed"] == "1" and summary["G5_has_gt"]["ok"] == "1"


def test_corpus_additions_hold_every_new_ontology_kept_or_dropped(world):
    tmp_path, release, harvest = world
    run_merge(tmp_path, release, harvest)

    add = tmp_path / "additions" / "corpus" / "github"
    assert sorted(p.name for p in add.iterdir()) == ["acopy", "full", "none"]
    assert (add / "full" / "full.ttl").is_file()
    assert (add / "full" / "patterns" / "p.ttl").is_file()


def test_report_says_nothing_was_removed(world):
    tmp_path, release, harvest = world
    run_merge(tmp_path, release, harvest)

    report = (tmp_path / "merge" / "REPORT.md").read_text(encoding="utf-8")
    assert "No release row was removed or changed" in report
    assert "| `github-reuse` | 1 | 2 |" in report
    assert "3 ontologies added, 1 passes the gate" in report


def test_a_disputed_new_removal_stops_the_merge(world):
    tmp_path, release, harvest = world
    # Claim "none" has no overlap although it now reuses the pattern.
    (harvest / "github_dataset" / "none" / "none.ttl").write_text(
        onto("http://odp.org/p#A", "http://ex.org/o#Z"), encoding="utf-8")
    before = (release / "MANIFEST.csv").read_bytes()

    assert run_merge(tmp_path, release, harvest) == 1

    assert (release / "MANIFEST.csv").read_bytes() == before
    assert "DISPUTED" in (tmp_path / "merge" / "REPORT.md").read_text(encoding="utf-8")


def test_a_harvest_with_nothing_new_changes_no_file(world):
    tmp_path, release, harvest = world
    write_rows(harvest / "out" / "MANIFEST.csv", [mrow("old")])
    before = {p: p.read_bytes() for p in release.rglob("*") if p.is_file()}

    assert run_merge(tmp_path, release, harvest) == 0

    assert {p: p.read_bytes() for p in release.rglob("*") if p.is_file()} == before
