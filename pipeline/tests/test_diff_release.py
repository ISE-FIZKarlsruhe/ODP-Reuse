"""Tests for diff_release, the harvest's report.

A harvest rebuilds only the GitHub portion, so comparing it with the full
release manifest used to list every noOBO and OBO ontology as "disappeared
upstream". The report must also say which pattern IRIs were actually searched,
which hits come from patterns added after the release, and how many removals
were really re-verified.
"""
import csv

import diff_release

COLS = ["ontology_id", "dataset", "n_pattern_files", "n_gt_patterns",
        "n_complete_patterns", "drop_reason"]


def manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({"n_pattern_files": 1, "n_complete_patterns": 0, "drop_reason": "", **r})
    return path


def write_csv(path, cols, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def run(tmp_path, *extra):
    out = tmp_path / "diff"
    diff_release.main(["--old", str(tmp_path / "old.csv"), "--new", str(tmp_path / "new.csv"),
                       "--out", str(out), *extra])
    return (out / "REPORT.md").read_text(encoding="utf-8"), out


def test_datasets_the_harvest_did_not_rebuild_are_not_disappeared(tmp_path):
    manifest(tmp_path / "old.csv", [
        {"ontology_id": "a_b_x", "dataset": "github", "n_gt_patterns": 2},
        {"ontology_id": "go", "dataset": "OBO", "n_gt_patterns": 5},
        {"ontology_id": "arco", "dataset": "noOBO", "n_gt_patterns": 3},
    ])
    manifest(tmp_path / "new.csv", [
        {"ontology_id": "a_b_x", "dataset": "github", "n_gt_patterns": 2},
    ])

    report, out = run(tmp_path)

    assert "release ontologies not returned by this harvest: 0" in report
    assert not (out / "not_returned.csv").exists()
    assert "not rebuilt by this harvest: OBO, noOBO" in report
    assert "corpus (github): **1 → 1**" in report


def test_release_ontologies_the_harvest_missed_are_not_returned_and_kept(tmp_path):
    # Run 42239149156 called 525 release ontologies "disappeared upstream"; in a
    # sample of 100, 88 files were still at the same path. A search that does not
    # return a file is not evidence that it is gone, and the release keeps it.
    manifest(tmp_path / "old.csv", [
        {"ontology_id": "a", "dataset": "github", "n_gt_patterns": 1},
        {"ontology_id": "b", "dataset": "github", "n_gt_patterns": 1},
    ])
    manifest(tmp_path / "new.csv", [{"ontology_id": "a", "dataset": "github", "n_gt_patterns": 1}])

    report, out = run(tmp_path)

    assert "disappeared" not in report.lower()
    assert "release ontologies not returned by this harvest: 1 (kept in the release)" in report
    assert "## Not returned by this harvest" in report
    assert (out / "not_returned.csv").exists()
    assert not (out / "disappeared.csv").exists()


def test_new_pairs_count_only_ontologies_that_pass_the_gate(tmp_path, monkeypatch):
    # The issue for run 42239149156 announced 316 new pairs; 58 of them were on
    # ontologies the gate drops, which can never enter a view.
    manifest(tmp_path / "old.csv", [
        {"ontology_id": "common_kept", "dataset": "github", "n_gt_patterns": 1},
        {"ontology_id": "common_dropped", "dataset": "github", "n_gt_patterns": 1,
         "drop_reason": "G4_substantive:own_entities=0<1"},
    ])
    manifest(tmp_path / "new.csv", [
        {"ontology_id": "common_kept", "dataset": "github", "n_gt_patterns": 3},
        {"ontology_id": "common_dropped", "dataset": "github", "n_gt_patterns": 4,
         "drop_reason": "G4_substantive:own_entities=0<1"},
        {"ontology_id": "new_kept", "dataset": "github", "n_gt_patterns": 2},
        {"ontology_id": "new_dropped", "dataset": "github", "n_gt_patterns": 5,
         "drop_reason": "G3_duplicate:of(new_kept)"},
    ])
    gh_out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))

    report, _ = run(tmp_path)

    # 2 on the new kept ontology + 2 gained by the kept common one
    assert "new ontology–pattern pairs on gate-passing ontologies: 4**" in report
    assert "new_pairs=4" in gh_out.read_text(encoding="utf-8").splitlines()


def test_hits_for_patterns_added_after_the_release_are_listed_separately(tmp_path):
    manifest(tmp_path / "old.csv", [{"ontology_id": "a", "dataset": "github", "n_gt_patterns": 1}])
    manifest(tmp_path / "new.csv", [{"ontology_id": "a", "dataset": "github", "n_gt_patterns": 1}])
    disc = write_csv(tmp_path / "results.csv",
                     ["source", "pattern_label", "pattern_iri", "match_label", "url", "details",
                      "pattern_set"], [
        {"source": "github", "pattern_iri": "http://odp.org/old", "match_label": "r/1",
         "url": "u1", "pattern_set": "release"},
        {"source": "github", "pattern_iri": "http://odp.org/chs#", "match_label": "e-huseyin/semRTI",
         "url": "u2", "pattern_set": "new"},
        {"source": "github", "pattern_iri": "http://odp.org/chs#", "match_label": "e-huseyin/semRTI",
         "url": "u3", "pattern_set": "new"},
    ])

    report, _ = run(tmp_path, "--discovery", str(disc))

    assert "## Hits for patterns added after the release" in report
    assert "| `http://odp.org/chs#` | 1 | 2 |" in report
    assert "release: 1 hits" in report and "new: 2 hits" in report


def test_unsearched_pattern_iris_make_the_report_say_incomplete(tmp_path):
    manifest(tmp_path / "old.csv", [])
    manifest(tmp_path / "new.csv", [])
    status = write_csv(tmp_path / "status.csv",
                       ["source", "pattern_iri", "pattern_set", "status", "queries", "hits", "note"], [
        {"source": "github", "pattern_iri": "http://x/a", "pattern_set": "release", "status": "done"},
        {"source": "github", "pattern_iri": "http://x/b", "pattern_set": "release",
         "status": "not_searched"},
    ])

    report, _ = run(tmp_path, "--search-status", str(status))

    assert "INCOMPLETE" in report
    assert "1 of 2 pattern IRIs were not searched" in report


def test_footer_states_how_many_removals_were_reverified(tmp_path):
    manifest(tmp_path / "old.csv", [])
    manifest(tmp_path / "new.csv", [])
    audit = tmp_path / "audit"
    audit.mkdir()
    write_csv(audit / "removed_G4_substantive.csv", ["ontology_id", "verified"],
              [{"ontology_id": "a", "verified": "ok"}, {"ontology_id": "b", "verified": "unchecked"}])
    write_csv(audit / "removed_G2_pattern_copy.csv", ["ontology_id", "verified"],
              [{"ontology_id": "c", "verified": "ok"}])

    report, _ = run(tmp_path, "--audit", str(audit))

    assert "Every removal" not in report
    assert "2 of 3 removals were re-verified" in report
    assert "0 disputed" in report


def test_footer_does_not_mention_unsampled_removals_when_all_were_checked(tmp_path):
    manifest(tmp_path / "old.csv", [])
    manifest(tmp_path / "new.csv", [])
    audit = tmp_path / "audit"
    audit.mkdir()
    write_csv(audit / "removed_G3_duplicate.csv", ["ontology_id", "verified"],
              [{"ontology_id": "a", "verified": "ok"}])

    report, _ = run(tmp_path, "--audit", str(audit))

    assert "All 1 removals were re-verified" in report
    assert "not sampled" not in report
