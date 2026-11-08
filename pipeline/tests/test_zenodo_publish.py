"""zenodo_publish: a new Zenodo version under the existing concept DOI.

The publish workflow runs it unattended, in steps: open a draft (which reserves
the version's DOI, so the DOI can be written into the files and the git tag
before anything is public), upload, then publish -- or discard the draft if
anything before the publish fails. These tests run it against a fake Zenodo.
"""
import pytest

import zenodo_publish as zp

BASE = "https://zenodo.example"
INHERITED = {
    "title": "old title", "description": "old", "version": "1.2.0",
    "doi": "10.5281/zenodo.21808321", "upload_type": "dataset",
    "communities": [{"identifier": "nfdi-matwerk"}],
    "related_identifiers": [{"identifier": "10.1/paper", "relation": "isSupplementTo"}],
    "creators": [{"name": "Norouzi, Ebrahim"}],
}


class Resp:
    def __init__(self, body=None, status=200):
        self.body, self.status_code = body or {}, status

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeZenodo:
    """Stands in for the requests module; records every call."""

    def __init__(self, reserve=True):
        self.calls, self.reserve, self.metadata = [], reserve, None

    def _log(self, method, url, **kw):
        self.calls.append((method, url.replace(BASE, ""), kw))

    def get(self, url, params=None, timeout=None):
        self._log("GET", url)
        if url.endswith("/api/records/17751740"):        # concept -> newest version
            return Resp({"id": 21808321, "links": {
                "latest": f"{BASE}/api/records/21808321/versions/latest"}})
        if url.endswith("/api/deposit/depositions/99"):
            return Resp({"id": 99, "files": [{"id": "f1"}, {"id": "f2"}],
                         "links": {"bucket": f"{BASE}/api/files/b1",
                                   "html": f"{BASE}/deposit/99"},
                         "metadata": dict(INHERITED)})
        return Resp(status=404)

    def post(self, url, params=None, timeout=None):
        self._log("POST", url)
        if url.endswith("/depositions/21808321/actions/newversion"):
            return Resp({"links": {"latest_draft": f"{BASE}/api/deposit/depositions/99"}})
        if url.endswith("/depositions/99/actions/publish"):
            return Resp({"doi": "10.5281/zenodo.99", "conceptdoi": "10.5281/zenodo.17751740"})
        return Resp(status=404)

    def put(self, url, params=None, json=None, data=None, timeout=None):
        self._log("PUT", url, json=json, name=url.rsplit("/", 1)[-1])
        if url.endswith("/depositions/99"):
            self.metadata = json["metadata"]
            md = dict(json["metadata"])
            if self.reserve:
                md["prereserve_doi"] = {"doi": "10.5281/zenodo.99", "recid": 99}
            return Resp({"id": 99, "metadata": md})
        return Resp()

    def delete(self, url, params=None, timeout=None):
        self._log("DELETE", url)
        return Resp(status=204)


@pytest.fixture
def fake(monkeypatch):
    z = FakeZenodo()
    monkeypatch.setattr(zp, "requests", z)
    return z


def test_the_newest_version_is_found_through_the_concept_record(fake):
    # Zenodo answers /records/<concept> with the newest version; its links.latest
    # ends in "/versions/latest", which the old code tried to read as an id.
    assert zp.latest_version_id(BASE, "t", 17751740) == 21808321


def test_a_new_version_keeps_the_inherited_metadata(fake):
    draft = zp.open_draft(BASE, "t", 17751740,
                          overrides={"title": "ODP-Reuse", "description": "new"},
                          version="v1.3.0", notes="Adds 80 ontologies.", today="2026-12-01")

    md = fake.metadata
    assert md["communities"] == INHERITED["communities"]
    assert md["related_identifiers"] == INHERITED["related_identifiers"]
    assert "doi" not in md
    assert (md["title"], md["description"]) == ("ODP-Reuse", "new")
    assert (md["version"], md["publication_date"], md["notes"]) == (
        "1.3.0", "2026-12-01", "Adds 80 ontologies.")
    assert draft == {"draft_id": 99, "doi": "10.5281/zenodo.99",
                     "url": f"{BASE}/deposit/99", "concept": 17751740}


def test_files_inherited_from_the_previous_version_are_cleared(fake):
    zp.open_draft(BASE, "t", 17751740, overrides={}, version="v1.3.0", today="2026-12-01")

    deleted = [u for m, u, _ in fake.calls if m == "DELETE"]
    assert deleted == ["/api/deposit/depositions/99/files/f1",
                       "/api/deposit/depositions/99/files/f2"]


def test_without_a_reserved_doi_the_record_id_names_it(monkeypatch):
    z = FakeZenodo(reserve=False)
    monkeypatch.setattr(zp, "requests", z)

    draft = zp.open_draft(BASE, "t", 17751740, overrides={}, version="1.3.0", today="2026-12-01")

    assert draft["doi"] == "10.5281/zenodo.99"


def test_upload_puts_each_file_into_the_draft_bucket(fake, tmp_path):
    files = []
    for name in ("corpus.zip", "corpus-hashed.zip"):
        (tmp_path / name).write_bytes(b"x")
        files.append(tmp_path / name)

    zp.upload(BASE, "t", 99, files)

    puts = [kw["name"] for m, u, kw in fake.calls if m == "PUT" and "/files/b1/" in u]
    assert puts == ["corpus.zip", "corpus-hashed.zip"]


def test_publish_returns_the_version_doi_and_discard_deletes_the_draft(fake):
    assert zp.publish(BASE, "t", 99) == "10.5281/zenodo.99"

    zp.discard(BASE, "t", 99)

    assert ("DELETE", "/api/deposit/depositions/99", {}) in fake.calls
