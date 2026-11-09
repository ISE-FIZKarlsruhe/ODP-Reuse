"""Tests for verify_reproduction: does the archived corpus reproduce MANIFEST.csv?

Two checks. ``hashes``: every ontology file in the archive is byte-identical to
the one the gate hashed. ``compare``: re-running the gate gives the released
verdicts. Columns that legitimately differ between machines (the absolute
``path``, and ``error_detail``, which embeds it) are not compared; rows the
re-run skipped for size are reported, not failed -- their hash check stands in.
"""
import csv
import hashlib

import build_manifest
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


def test_pattern_error_text_is_machine_specific_but_its_count_is_not(tmp_path):
    # The parser's message names the pattern file by its absolute path.
    msg = "datamodel.owl: ParserError: file:///{}/github_dataset/a/patterns/datamodel.owl"
    fields = ["n_pattern_errors", "pattern_errors"]
    cols = COLS + fields
    rel_row = {**row("a"), "n_pattern_errors": "1", "pattern_errors": msg.format("C:/release")}
    same = {**row("a"), "n_pattern_errors": "1", "pattern_errors": msg.format("home/runner")}
    more = {**same, "n_pattern_errors": "2"}

    def put(name, r):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerow(r)
        return tmp_path / name

    rel = put("rel.csv", rel_row)
    assert vr.compare_manifests(rel, put("same.csv", same)).ok
    res = vr.compare_manifests(rel, put("more.csv", more))
    assert res.mismatches == [(("github", "a"), ["n_pattern_errors"])]


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


# ---- a release extended by additive versions -----------------------------
#
# publish-release.yml gates a new version's ontologies against the published
# ones: a new copy of a published file is a duplicate of the published row,
# even when its id sorts first. Gating the whole archive at once keeps the
# smallest id instead, so the comparison must replay the versions in order.

def gated(oid, sha, ds="github", **kw):
    base = {c: "" for c in build_manifest.CSV_COLUMNS}
    base.update({"ontology_id": oid, "dataset": ds, "path": f"/x/{oid}",
                 "content_sha256": sha, "n_pattern_files": "1", "parsed": "ok",
                 "n_triples": "10", "n_own_entities": "5", "n_gt_patterns": "1",
                 "n_complete_patterns": "0", "stage_reached": "has_gt",
                 "n_pattern_errors": "0"})
    base.update(kw)
    return base


def dup(oid, sha, of, ds="github"):
    return gated(oid, sha, ds, duplicate_of=of, drop_reason=f"G3_duplicate:of({of})",
                 stage_reached="parsed")


def write_gated(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=build_manifest.CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def additions(tmp_path, **versions):
    d = tmp_path / "additions"
    d.mkdir()
    for version, rows in versions.items():
        write_gated(d / f"{version}.csv", rows)
    return d


def test_a_published_copy_stays_the_kept_one(tmp_path):
    # v1.2.0 published "old"; v1.3.0 found an identical "Anew", which sorts first.
    rel = write_gated(tmp_path / "rel.csv", [gated("old", "S"), dup("Anew", "S", of="old")])
    add = additions(tmp_path, **{"v1.3.0": [dup("Anew", "S", of="old")]})
    new = write_gated(tmp_path / "new.csv", [dup("old", "S", of="Anew"), gated("Anew", "S")])

    assert not vr.compare_manifests(rel, new).ok      # all at once: Anew would be kept
    res = vr.compare_manifests(rel, new, additions=add)

    assert res.ok, res.mismatches
    assert res.n_identical == 2
    assert res.replayed == [("base", 1), ("1.3.0", 1)]
    assert "in version order" in vr.report_compare(res)


def test_versions_replay_in_numeric_order(tmp_path):
    # v1.9.0 published "z"; v1.10.0 then found a copy "a". Sorted as text,
    # v1.10.0 would come first and "a" would be the kept copy.
    rel = write_gated(tmp_path / "rel.csv", [gated("m", "M"), gated("z", "T"), dup("a", "T", of="z")])
    add = additions(tmp_path, **{"v1.9.0": [gated("z", "T")], "v1.10.0": [dup("a", "T", of="z")]})
    new = write_gated(tmp_path / "new.csv", [gated("m", "M"), dup("z", "T", of="a"), gated("a", "T")])

    res = vr.compare_manifests(rel, new, additions=add)

    assert res.ok, res.mismatches
    assert [v for v, _n in res.replayed] == ["base", "1.9.0", "1.10.0"]


def test_the_replay_still_catches_a_wrong_verdict(tmp_path):
    # The release kept the new copy as well: the replay must flag it.
    rel = write_gated(tmp_path / "rel.csv", [gated("old", "S"), gated("Anew", "S")])
    add = additions(tmp_path, **{"v1.3.0": [gated("Anew", "S")]})
    new = write_gated(tmp_path / "new.csv", [dup("old", "S", of="Anew"), gated("Anew", "S")])

    res = vr.compare_manifests(rel, new, additions=add)

    assert not res.ok
    assert [k for k, _cols in res.mismatches] == [("github", "Anew")]


def test_a_release_without_additions_compares_as_before(tmp_path):
    rel = write_gated(tmp_path / "rel.csv", [gated("a", "S"), dup("b", "S", of="a")])
    new = write_gated(tmp_path / "new.csv", [gated("a", "S"), dup("b", "S", of="a")])

    res = vr.compare_manifests(rel, new, additions=tmp_path / "no-additions")

    assert res.ok and res.n_identical == 2 and res.replayed == [("base", 2)]


def test_a_file_skipped_for_size_still_keeps_its_duplicates(tmp_path):
    # The re-run skipped "go" for size; its released row stands in for it
    # (its hash was verified), so "go-copy" is still its duplicate.
    rel = write_gated(tmp_path / "rel.csv", [gated("go", "G", ds="OBO"), dup("go-copy", "G", of="go", ds="OBO")])
    new = write_gated(tmp_path / "new.csv", [
        gated("go", "G", ds="OBO", parsed="SizeLimitExceeded", stage_reached="discovered",
              drop_reason="G1_parse:SizeLimitExceeded(>100.0MB)"),
        gated("go-copy", "G", ds="OBO")])

    res = vr.compare_manifests(rel, new, additions=tmp_path / "no-additions")

    assert res.ok, res.mismatches
    assert res.size_skipped == [("OBO", "go")] and res.n_identical == 1


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
