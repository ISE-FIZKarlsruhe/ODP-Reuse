"""Tests for pattern_iris: which IRIs the harvester searches for.

The release searched 144 IRIs extracted by run_multiple_source.sh; the CI
workflow's auto-discovery found only 86, so a "reproducibility" diff would have
reported 58 patterns' pairs as vanished. Both now use this module.
"""
import csv

import pattern_iris as pi

RDF_XML = '''<rdf:RDF xmlns="http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl#"
     xml:base="http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl">
  <owl:Ontology rdf:about="http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl"/>
</rdf:RDF>'''

TURTLE = '''@prefix : <https://w3id.org/hacid/onto/data/> .
@base <https://w3id.org/hacid/onto/data/> .
<https://w3id.org/hacid/onto/data/> a owl:Ontology ;
    owl:versionIRI <https://w3id.org/hacid/onto/data/1.0> .
'''


def test_extracts_rdfxml_ontology_iri_and_namespace():
    found = pi.extract_iris(RDF_XML)

    assert ("http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl", "xml:base") in found
    assert ("http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl",
            "xmlns (default namespace)") in found
    assert {i for i, _ in found} == {"http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl"}


def test_extracts_turtle_base_subject_and_version_iri():
    iris = {i for i, _ in pi.extract_iris(TURTLE)}

    assert iris == {"https://w3id.org/hacid/onto/data/", "https://w3id.org/hacid/onto/data/1.0"}


def test_host_filter_keeps_only_matching_iris():
    assert pi.extract_iris(TURTLE, host_filter="ontologydesignpatterns") == set()


def test_generic_template_and_homepage_iris_are_flagged():
    assert pi.is_generic("http://www.ontologydesignpatterns.org/cp/owl/template")
    assert pi.is_generic("http://www.ontologydesignpatterns.org/cp/owl/template/x")
    assert pi.is_generic("https://ontologydesignpatterns.org/")
    assert not pi.is_generic("http://www.ontologydesignpatterns.org/cp/owl/timeinterval.owl")


def test_pattern_sets_keep_every_release_iri_and_label_the_rest_new():
    release = ["http://a.org/p1", "http://a.org/p2"]
    current = ["http://a.org/p1", "http://b.org/p3", "http://b.org/template"]

    rows = pi.build_pattern_sets(release, current)

    # p2 is no longer in the pattern repository but stays in the search set, so
    # a harvest searches like-for-like with the release.
    assert rows == [
        {"pattern_iri": "http://a.org/p1", "pattern_set": "release"},
        {"pattern_iri": "http://a.org/p2", "pattern_set": "release"},
        {"pattern_iri": "http://b.org/p3", "pattern_set": "new"},
    ]


def test_w3c_language_namespaces_are_not_searched_as_patterns():
    # Extracted from pattern files' default namespaces, these matched any OWL,
    # XHTML or SVG file on GitHub: 2,235 of 9,577 hits in run 42181530957.
    release = ["http://a.org/p1", "http://www.w3.org/2002/07/owl",
               "http://www.w3.org/1999/xhtml", "http://www.w3.org/2000/svg"]

    rows = pi.build_pattern_sets(release, ["http://www.w3.org/2002/07/owl#", "http://b.org/p2"])

    assert rows == [{"pattern_iri": "http://a.org/p1", "pattern_set": "release"},
                    {"pattern_iri": "http://b.org/p2", "pattern_set": "new"}]
    assert pi.is_language_namespace("http://www.w3.org/2002/07/owl#")
    assert not pi.is_language_namespace("http://www.w3.org/ns/sosa/")


def test_sets_command_writes_csv_from_a_cloned_repository(tmp_path):
    repo = tmp_path / "patterns-repository"
    (repo / "AgentRole").mkdir(parents=True)
    (repo / "AgentRole" / "agentrole.owl").write_text(RDF_XML, encoding="utf-8")
    (repo / "VariableDimensionalSpace").mkdir()
    (repo / "VariableDimensionalSpace" / "vds.ttl").write_text(TURTLE, encoding="utf-8")
    release = tmp_path / "retrieved_patterns.csv"
    release.write_text("pattern_iri\nhttp://www.ontologydesignpatterns.org/cp/owl/agentrole.owl\n",
                       encoding="utf-8")
    out = tmp_path / "pattern_sets.csv"

    pi.main(["sets", "--release", str(release), "--repo", str(repo), "--out", str(out)])

    got = [(r["pattern_iri"], r["pattern_set"]) for r in csv.DictReader(out.open(encoding="utf-8"))]
    assert got == [
        ("http://www.ontologydesignpatterns.org/cp/owl/agentrole.owl", "release"),
        ("https://w3id.org/hacid/onto/data/", "new"),
        ("https://w3id.org/hacid/onto/data/1.0", "new"),
    ]
