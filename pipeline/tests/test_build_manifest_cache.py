"""A row restored from build_manifest's --resume cache must equal the row it
was cached from.

The cache stores ``pattern_errors`` joined into one string. Restoring that
string where a list is expected made ``n_pattern_errors`` the number of
CHARACTERS and spelled the message out one character at a time -- which is how
seven rows of the v1.2.0 MANIFEST.csv came to report 240-1000 pattern errors
instead of one.
"""
import json

from build_manifest import record_from_cache
from odpqual.gate import OntologyRecord


def roundtrip(rec):
    return record_from_cache(json.loads(json.dumps(rec.csv_row()))).csv_row()


def make(errors):
    rec = OntologyRecord(ontology_id="o", dataset="github", path="/x/o")
    rec.pattern_errors = list(errors)
    return rec


def test_one_pattern_error_survives_the_cache():
    rec = make(["mp-prop.owl: ParserError: line 21: Repeat node-elements"])

    row = roundtrip(rec)

    assert row["n_pattern_errors"] == 1
    assert row["pattern_errors"] == rec.csv_row()["pattern_errors"]


def test_several_pattern_errors_survive_the_cache():
    rec = make(["a.owl: BadSyntax", "b.ttl: ParserError"])

    assert roundtrip(rec) == rec.csv_row()


def test_no_pattern_errors_survive_the_cache():
    rec = make([])

    row = roundtrip(rec)

    assert row["n_pattern_errors"] == 0 and row["pattern_errors"] == ""


def test_a_truncated_error_list_keeps_its_count():
    # csv_row cuts the joined string at 1000 characters; the count must not shrink.
    rec = make([f"p{i}.owl: " + "x" * 300 for i in range(5)])

    row = roundtrip(rec)

    assert row["n_pattern_errors"] == 5
    assert row["pattern_errors"] == rec.csv_row()["pattern_errors"]
