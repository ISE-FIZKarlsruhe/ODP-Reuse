"""Tests for the GitHub half of odp_multi_source_finder.

The quarterly harvest of 2026-10-01 spent 5.5 of its 6 hours asleep on GitHub
rate-limit back-offs and covered 21 of 86 pattern IRIs. These tests pin the
three causes: a query GitHub always rejects, a 422 retried as if transient, and
pacing far above the 10 requests/minute code-search limit.
"""
import csv
import json

import pytest
import requests

import odp_multi_source_finder as finder

IRI = "http://www.ontologydesignpatterns.org/cp/owl/timeinterval.owl"


def response(status, body=None, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body or {}).encode()
    r.headers.update(headers or {})
    r.url = "https://api.github.com/search/code"
    return r


class OneResponseSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, 0

    def get(self, url, params=None):
        self.calls += 1
        return self.resp


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_query_parse_error_is_raised_without_retrying(monkeypatch):
    slept = []
    monkeypatch.setattr(finder.time, "sleep", slept.append)
    session = OneResponseSession(response(
        422, {"message": "ERROR_TYPE_QUERY_PARSING_FATAL unable to parse query!"}))

    with pytest.raises(requests.HTTPError):
        finder.gh_request(session, "https://api.github.com/search/code")

    assert session.calls == 1
    assert slept == []


class ScriptedSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params=None):
        self.calls += 1
        return self.responses.pop(0)


def test_a_throttled_request_logs_which_rate_limit_fired(monkeypatch, capsys):
    # The 2026-11-05 CI run was throttled from its first request while a local
    # run with a personal token never was; the headers say which limit it is.
    monkeypatch.setattr(finder.time, "sleep", lambda s: None)
    session = ScriptedSession([
        response(403, {"message": "You have exceeded a secondary rate limit."},
                 {"Retry-After": "734", "x-ratelimit-resource": "code_search",
                  "x-ratelimit-limit": "10", "x-ratelimit-remaining": "8",
                  "x-ratelimit-used": "2"}),
        response(200, {"items": []}),
    ])

    finder.gh_request(session, "https://api.github.com/search/code")

    err = capsys.readouterr().err
    assert "Retry-After=734s" in err
    assert "resource=code_search" in err and "limit=10" in err and "remaining=8" in err
    assert "secondary rate limit" in err


class ScriptedGet:
    """Stands in for requests.get: returns (or raises) the scripted items in order."""

    def __init__(self, items):
        self.items, self.calls = list(items), 0

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


def test_portal_search_retries_a_bad_gateway(monkeypatch):
    # Run 42223848653: BioPortal's first query returned 502 and the uncaught
    # error ended discovery after GitHub search had finished all 147 IRIs.
    slept = []
    monkeypatch.setattr(finder.time, "sleep", slept.append)
    get = ScriptedGet([response(502), response(200, {"collection": [{"@id": "x"}]})])
    monkeypatch.setattr(finder.requests, "get", get)

    pages = list(finder.bioportal_search("key", IRI, pages=1))

    assert pages == [{"collection": [{"@id": "x"}]}]
    assert get.calls == 2 and slept


def test_portal_search_gives_up_on_one_iri_without_raising(monkeypatch, capsys):
    monkeypatch.setattr(finder.time, "sleep", lambda s: None)
    get = ScriptedGet([response(502)])
    monkeypatch.setattr(finder.requests, "get", get)

    pages = list(finder.matportal_search("key", IRI, pages=2))

    assert pages == []
    assert get.calls == finder.PORTAL_ATTEMPTS
    assert "502" in capsys.readouterr().err


def test_portal_connection_error_does_not_raise(monkeypatch):
    monkeypatch.setattr(finder.time, "sleep", lambda s: None)
    monkeypatch.setattr(finder.requests, "get",
                        ScriptedGet([requests.ConnectionError("reset by peer")]))

    assert list(finder.bioportal_search("key", IRI)) == []


def test_code_search_queries_never_group_qualifiers():
    # GitHub's REST code search rejects "( extension:a OR extension:b )" with a
    # 422 every time, so a grouped query only wastes a request per IRI.
    queries = finder.build_code_search_queries_for_iri(IRI)

    assert queries, "at least one query per IRI"
    for q in queries:
        assert " OR " not in q and "(" not in q, q
        assert f'"{IRI}"' in q


def test_throttle_keeps_search_calls_under_the_rate():
    clock = FakeClock()
    throttle = finder.SearchThrottle(per_minute=9, clock=clock, sleep=clock.sleep)

    starts = []
    for _ in range(4):
        throttle.wait()
        starts.append(clock.now)
        clock.now += 0.5          # the request itself takes some time

    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 60 / 9 - 1e-9 for g in gaps), gaps
    assert starts[0] == 0.0, "the first call is not delayed"


def test_patterns_file_carries_the_pattern_set(tmp_path):
    p = tmp_path / "pattern_sets.csv"
    p.write_text("pattern_iri,pattern_set\n"
                 f"{IRI},release\n"
                 "https://w3id.org/hacid/onto/data/,new\n", encoding="utf-8")

    patterns, sets = finder.load_patterns_file(p)

    assert patterns == [("timeinterval", IRI), ("data", "https://w3id.org/hacid/onto/data/")]
    assert sets == {IRI: "release", "https://w3id.org/hacid/onto/data/": "new"}


def test_plain_pattern_list_without_header_is_accepted(tmp_path):
    p = tmp_path / "patterns.txt"
    p.write_text(f"pattern_iri\n{IRI}\n\n", encoding="utf-8")

    patterns, sets = finder.load_patterns_file(p)

    assert patterns == [("timeinterval", IRI)]
    assert sets == {}


def item(repo, path):
    return {"repository": {"full_name": repo}, "path": path,
            "html_url": f"https://github.com/{repo}/blob/x/{path}"}


def test_search_records_hits_and_skips_rejected_queries():
    def search(q):
        if "extension:owl" in q:
            return [item("a/b", "o.owl"), item("c/d", "p.owl")]
        if "extension:ttl" in q:
            raise requests.HTTPError(response=response(422))
        return []

    rows, status = finder.search_github_patterns([("timeinterval", IRI)], search)

    assert [r["match_label"] for r in rows] == ["a/b", "c/d"]
    assert status == [{"source": "github", "pattern_iri": IRI, "status": "done",
                       "queries": len(finder.build_code_search_queries_for_iri(IRI)),
                       "hits": 2, "note": "1 query rejected (422)"}]


def test_exhausted_time_budget_marks_the_rest_not_searched():
    clock = FakeClock()
    searched = []

    def search(q):
        searched.append(q)
        clock.now += 100          # each query costs 100 s
        return []

    patterns = [("a", "http://x.org/a"), ("b", "http://x.org/b"), ("c", "http://x.org/c")]
    rows, status = finder.search_github_patterns(patterns, search, deadline=150, clock=clock)

    assert [s["status"] for s in status] == ["done", "not_searched", "not_searched"]
    assert all("x.org/a" in q for q in searched)


def test_main_exits_3_and_writes_status_when_budget_is_exhausted(tmp_path, monkeypatch):
    pats = tmp_path / "pattern_sets.csv"
    pats.write_text(f"pattern_iri,pattern_set\n{IRI},release\nhttp://x.org/new,new\n",
                    encoding="utf-8")
    out = tmp_path / "results.csv"
    ticks = iter(range(10**6))
    monkeypatch.setattr(finder.time, "monotonic", lambda: float(next(ticks)))  # 1 s per read

    def no_network(*a, **k):
        raise AssertionError("an exhausted budget must not issue requests")

    monkeypatch.setattr(finder.requests.Session, "get", no_network)
    monkeypatch.setenv("GITHUB_TOKEN", "not-used")
    monkeypatch.setattr(finder.sys, "argv", [
        "finder", "--sources", "github", "--patterns-file", str(pats),
        "--github-max-minutes", "0.001", "--out", str(out)])

    with pytest.raises(SystemExit) as exc:
        finder.main()

    assert exc.value.code == 3
    assert out.read_text(encoding="utf-8").splitlines() == [",".join(finder.OUT_COLUMNS)]
    status = list(csv.DictReader((tmp_path / "results_github_status.csv").open(encoding="utf-8")))
    assert [(s["pattern_iri"], s["pattern_set"], s["status"]) for s in status] == [
        (IRI, "release", "not_searched"), ("http://x.org/new", "new", "not_searched")]


def test_writer_appends_unique_rows_as_they_arrive(tmp_path):
    out = tmp_path / "results.csv"
    writer = finder.IncrementalCsv(out, pattern_sets={IRI: "release"})
    row = {"source": "github", "pattern_label": "timeinterval", "pattern_iri": IRI,
           "match_label": "a/b", "url": "u1", "details": "o.owl"}

    rows = [row]
    writer.sync(rows)
    rows += [dict(row), dict(row, url="u2")]      # a duplicate, then a new row
    writer.sync(rows)
    writer.close()

    got = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["url"] for r in got] == ["u1", "u2"]
    assert {r["pattern_set"] for r in got} == {"release"}
    assert writer.n_written == 2
