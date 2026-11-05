"""Tests for verify_reproduction: does the archived corpus reproduce MANIFEST.csv?

Two checks. ``hashes``: every ontology file in the archive is byte-identical to
the one the gate hashed. ``compare``: re-running the gate gives the released
verdicts. Columns that legitimately differ between machines (the absolute
``path``, and ``error_detail``, which embeds it) are not compared; rows the
re-run skipped for size are reported, not failed -- their hash check stands in.
"""
import csv
import hashlib

import verify_reproduction as vr

COLS = ["ontology_id", "dataset", "path", "content_sha256", "n_pattern_files", "parsed",
        "error_detail", "n_triples", "n_gt_patterns", "stage_reached", "drop_reason"]


def row(oid, ds="github", **kw):
    base = {"ontology_id": oid, "dataset": ds, "path": f"C:/release/{oid}",
            "content_sha256": "h-" + oid, "n_pattern_files": "1", "parsed": "ok",
            "error_detail": "", "n_triples": "10", "n_gt_patterns": "1",
            "stage_reached": "has_gt", "drop_reason": ""}
    return {**base, **kw}


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_runs_on_different_machines_match(tmp_path):
    rel = write(tmp_path / "rel.csv", [row("a"), row("b", parsed="BadSyntax",
                                                    error_detail="BadSyntax: C:/release/b/b.ttl line 3",
                                                    drop_reason="G1_parse:BadSyntax",
                                                    stage_reached="discovered")])
    new = write(tmp_path / "new.csv", [row("a", path="/home/runner/a"),
                                       row("b", path="/home/runner/b", parsed="BadSyntax",
                                           error_detail="BadSyntax: /home/runner/b/b.ttl line 3",
                                           drop_reason="G1_parse:BadSyntax",
                                           stage_reached="discovered")])

    res = vr.compare_manifests(rel, new)

    assert res.ok
    assert res.n_identical == 2 and res.mismatches == [] and res.size_skipped == []


def test_a_changed_verdict_fails_and_names_the_columns(tmp_path):
    rel = write(tmp_path / "rel.csv", [row("a")])
    new = write(tmp_path / "new.csv", [row("a", n_gt_patterns="0", stage_reached="substantive",
                                           drop_reason="G5_has_gt:no_pattern_overlap")])

    res = vr.compare_manifests(rel, new)

    assert not res.ok
    assert res.mismatches == [(("github", "a"), ["n_gt_patterns", "stage_reached", "drop_reason"])]


def test_rows_skipped_for_size_are_listed_not_failed(tmp_path):
    rel = write(tmp_path / "rel.csv", [row("go", ds="OBO")])
    new = write(tmp_path / "new.csv", [row("go", ds="OBO", parsed="SizeLimitExceeded",
                                           drop_reason="G1_parse:SizeLimitExceeded(>100.0MB)",
                                           stage_reached="discovered", n_triples="")])

    res = vr.compare_manifests(rel, new)

    assert res.ok
    assert res.size_skipped == [("OBO", "go")]


def test_only_datasets_that_were_regated_are_compared(tmp_path):
    rel = write(tmp_path / "rel.csv", [row("a"), row("cco", ds="noOBO")])
    new = write(tmp_path / "new.csv", [row("a")])

    res = vr.compare_manifests(rel, new)

    assert res.ok and res.datasets == ["github"]


def test_a_missing_ontology_fails(tmp_path):
    rel = write(tmp_path / "rel.csv", [row("a"), row("b")])
    new = write(tmp_path / "new.csv", [row("a")])

    res = vr.compare_manifests(rel, new)

    assert not res.ok and res.missing == [("github", "b")]


def make_onto(corpus, ds_dir, oid, text):
    d = corpus / ds_dir / oid
    (d / "patterns").mkdir(parents=True)
    (d / f"{oid}.owl").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def test_hash_check_detects_an_altered_archive_file(tmp_path):
    corpus = tmp_path / "corpus"
    ha = make_onto(corpus, "github", "a", "<rdf/>")
    hb = make_onto(corpus, "noOBO", "b", "<rdf>b</rdf>")
    rel = write(tmp_path / "rel.csv", [row("a", content_sha256=ha),
                                       row("b", ds="noOBO", content_sha256=hb)])
    assert vr.verify_hashes(rel, corpus).ok

    (corpus / "noOBO" / "b" / "b.owl").write_text("<rdf>tampered</rdf>", encoding="utf-8")
    res = vr.verify_hashes(rel, corpus)

    assert not res.ok
    assert res.mismatched == [("noOBO", "b")]
    assert res.n_verified == 1


def test_hash_check_reports_ontologies_missing_from_the_archive(tmp_path):
    corpus = tmp_path / "corpus"
    ha = make_onto(corpus, "github", "a", "<rdf/>")
    rel = write(tmp_path / "rel.csv", [row("a", content_sha256=ha), row("gone")])

    res = vr.verify_hashes(rel, corpus)

    assert not res.ok and res.missing == [("github", "gone")]
