"""Tests for build_github_dataset's work per discovery row.

The release's discovery file has 4,828 GitHub rows but 2,881 distinct files and
67 distinct pattern IRIs: an ontology that reuses three patterns appears three
times. Downloading it once and indexing the pattern repository once keeps the
quarterly harvest inside GitHub's 6-hour job limit without changing its output.
"""
import csv

import build_github_dataset as bgd

ONTO = "https://github.com/a/b/blob/x/o.owl"


def discovery(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "pattern_label", "pattern_iri", "match_label",
                                           "url", "details", "pattern_set"])
        w.writeheader()
        w.writerows({"source": "github", "match_label": "a/b", "details": "o.owl",
                     "pattern_set": "release", **r} for r in rows)
    return path


def patterns_repo(root):
    (root / "AgentRole").mkdir(parents=True)
    (root / "AgentRole" / "agentrole.owl").write_text('xml:base="http://odp.org/agentrole.owl"',
                                                      encoding="utf-8")
    (root / "TimeInterval").mkdir()
    (root / "TimeInterval" / "timeinterval.owl").write_text('xml:base="http://odp.org/timeinterval.owl"',
                                                            encoding="utf-8")
    return root


def run(tmp_path, monkeypatch, rows):
    calls = []

    def fake_download(url, out_path):
        calls.append(url)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<rdf/>", encoding="utf-8")
        return True, "downloaded (6 bytes)", str(out_path)

    monkeypatch.setattr(bgd, "download_file", fake_download)
    monkeypatch.setattr(bgd.time, "sleep", lambda s: None)
    out = tmp_path / "github_dataset"
    monkeypatch.setattr(bgd.sys, "argv", [
        "bgd", "--csv", str(discovery(tmp_path / "results.csv", rows)),
        "--patterns-repo", str(patterns_repo(tmp_path / "repo")), "--out-dir", str(out)])
    bgd.main()
    log = list(csv.DictReader((out / "download_log.csv").open(encoding="utf-8")))
    return calls, out, log


def test_an_ontology_found_for_several_patterns_is_downloaded_once(tmp_path, monkeypatch):
    calls, out, log = run(tmp_path, monkeypatch, [
        {"pattern_label": "agentrole", "pattern_iri": "http://odp.org/agentrole.owl", "url": ONTO},
        {"pattern_label": "timeinterval", "pattern_iri": "http://odp.org/timeinterval.owl", "url": ONTO},
    ])

    assert calls == [ONTO]
    pats = sorted(p.name for p in (out / "a_b_o" / "patterns").iterdir())
    assert pats == ["agentrole.owl", "timeinterval.owl"]
    assert [r["status"] for r in log] == ["ok", "ok"]
    assert [r["pattern_files_copied"] for r in log] == ["1", "1"]


def test_the_pattern_repository_is_read_once_per_run(tmp_path, monkeypatch):
    reads = []
    real = bgd.read_pattern_texts
    monkeypatch.setattr(bgd, "read_pattern_texts", lambda repo: reads.append(repo) or real(repo))

    run(tmp_path, monkeypatch, [
        {"pattern_label": "agentrole", "pattern_iri": "http://odp.org/agentrole.owl", "url": ONTO},
        {"pattern_label": "timeinterval", "pattern_iri": "http://odp.org/timeinterval.owl",
         "url": "https://github.com/c/d/blob/x/p.owl"},
    ])

    assert len(reads) == 1


def test_a_failed_download_is_retried_by_the_next_row(tmp_path, monkeypatch):
    attempts = []

    def flaky(url, out_path):
        attempts.append(url)
        if len(attempts) == 1:
            return False, "503 Service Unavailable", None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("<rdf/>", encoding="utf-8")
        return True, "downloaded", str(out_path)

    monkeypatch.setattr(bgd, "download_file", flaky)
    monkeypatch.setattr(bgd.time, "sleep", lambda s: None)
    out = tmp_path / "ds"
    monkeypatch.setattr(bgd.sys, "argv", [
        "bgd", "--csv", str(discovery(tmp_path / "r.csv", [
            {"pattern_label": "agentrole", "pattern_iri": "http://odp.org/agentrole.owl", "url": ONTO},
            {"pattern_label": "timeinterval", "pattern_iri": "http://odp.org/timeinterval.owl", "url": ONTO},
        ])), "--patterns-repo", str(patterns_repo(tmp_path / "repo")), "--out-dir", str(out)])

    bgd.main()

    assert attempts == [ONTO, ONTO]
