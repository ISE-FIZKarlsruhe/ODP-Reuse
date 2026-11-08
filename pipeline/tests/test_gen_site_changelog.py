"""The changelog page: every version, what it added, and how the dataset grew.

The page is generated from release-iri/history.csv (one row per version,
written by publish_harvest.py) and CHANGELOG.md, so the figures cannot drift
from what was published.
"""
import json

import gen_site

V1 = {"version": "1.0.0", "date": "2025-11-28", "tag": "", "zenodo_record": "17751741",
      "doi": "10.5281/zenodo.17751741", "discovered": "", "kept": "",
      "summary": "First deposit, before the quality gate"}
V12 = {"version": "1.2.0", "date": "2026-08-05", "tag": "v1.2.0", "zenodo_record": "21808321",
       "doi": "10.5281/zenodo.21808321", "discovered": "1973", "kept": "501",
       "github-reuse_ontologies": "442", "github-reuse_pairs": "669",
       "noOBO-reuse_ontologies": "51", "noOBO-reuse_pairs": "90",
       "OBO-reuse_ontologies": "8", "OBO-reuse_pairs": "758",
       "added_ontologies": "", "added_kept": "", "added_pairs": "", "summary": "Quality gate"}
V13 = dict(V12, version="1.3.0", date="2026-12-01", tag="v1.3.0", doi="10.5281/zenodo.99",
           discovered="2801", kept="581", **{"github-reuse_ontologies": "522",
                                             "github-reuse_pairs": "771"},
           added_ontologies="828", added_kept="80", added_pairs="102",
           harvest_run="https://run/1", summary="+80 ontologies")

CHANGELOG = """# Changelog

Intro paragraph.

## v1.3.0 — 2026-12-01

Added things.

## v1.2.0 — 2026-08-05

Gated.
"""


def test_the_history_payload_has_one_point_per_version():
    data = gen_site.history_payload([V1, V12, V13])

    assert [d["version"] for d in data] == ["1.0.0", "1.2.0", "1.3.0"]
    assert data[0]["comparable"] is False and data[0]["discovered"] is None
    assert (data[1]["discovered"], data[1]["kept"], data[1]["pairs"]) == (1973, 501, 1517)
    assert data[1]["added"] is None
    assert data[2]["added"] == {"ontologies": 828, "kept": 80, "excluded": 748, "pairs": 102}
    json.dumps(data)


def test_the_changelog_page_shows_the_history_and_the_release_notes():
    page = gen_site.page_changelog({"history": [V1, V12, V13], "changelog": CHANGELOG})

    assert 'id="odp-history"' in page and 'data-src="data/history.json"' in page
    assert 'id="odp-additions"' in page
    assert ("| **v1.3.0** · 2026-12-01 | 2,801 | 581 | 1,619 | +80 · +102 pairs | "
            "[zenodo.99](https://doi.org/10.5281/zenodo.99) |") in page
    assert "### v1.3.0 — 2026-12-01" in page and "Intro paragraph." not in page
    assert page.index("v1.3.0") < page.index("v1.2.0 —")
    assert "```mermaid" in page


def test_without_a_history_the_page_still_renders():
    page = gen_site.page_changelog({"history": [], "changelog": "_No changelog yet._"})

    assert "# Changelog" in page
