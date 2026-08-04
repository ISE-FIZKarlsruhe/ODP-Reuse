"""Vocabulary catalogues.

Two independent catalogues live here, kept in one file so that the whole
classification surface of the package is auditable in one place.

1. ``NAMESPACES`` -- external vocabularies recognised by IRI prefix, each with a
   ``tier``:

   ``foundational``  a top-level / foundational ontology in the usual sense
                     (DOLCE-family, BFO, SUMO, UFO, GFO, GUM, YAMATO ...)
   ``cross_domain``  a widely reused W3C/community vocabulary that is not a
                     top-level ontology but does supply upper-level structure
                     (PROV-O, SKOS, SSN/SOSA, OWL-Time, schema.org, FOAF,
                     Dublin Core, CIDOC-CRM, GeoSPARQL, QUDT, ORG, Event ...)
   ``odp_infra``     ODP-library plumbing, not modelling content
                     (cpannotationschema, the ODP submission namespaces)
   ``domain``        a subject-matter vocabulary (UBERON, GO, CHEBI, PATO,
                     ISO/DIN/VDI industrial namespaces ...)
   ``rdf_builtin``   RDF/RDFS/OWL/XSD and friends; never counted as alignment

   Matching is longest-prefix-wins, so ``.../DUL.owl#`` beats ``ontologydesignpatterns.org/``.

2. ``DOMAIN_LEXICON`` -- the token lists behind the subject-domain classifier in
   ``domains.py``.  Tokens are matched against camel/snake-split local names and
   ``rdfs:label`` values of a pattern's *own* entities.  This is a heuristic; see
   the docstring of ``domains.py`` for how its uncertainty is quantified.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------
# 1. external vocabulary catalogue
# --------------------------------------------------------------------------

#: (iri prefix, short name, tier)
NAMESPACES: tuple[tuple[str, str, str], ...] = (
    # ---- RDF built-ins (never alignment) --------------------------------
    ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf", "rdf_builtin"),
    ("http://www.w3.org/2000/01/rdf-schema#", "rdfs", "rdf_builtin"),
    ("http://www.w3.org/2002/07/owl#", "owl", "rdf_builtin"),
    ("http://www.w3.org/2001/XMLSchema#", "xsd", "rdf_builtin"),
    ("http://www.w3.org/2003/11/swrl", "swrl", "rdf_builtin"),
    ("http://www.w3.org/ns/shacl#", "shacl", "rdf_builtin"),
    ("http://swrl.stanford.edu/ontologies/", "swrlb", "rdf_builtin"),

    # ---- foundational / top-level ---------------------------------------
    # DOLCE family.  DUL (DOLCE+DnS Ultralite) is the ODP portal's own upper
    # layer and is by far the most frequent; keep its variants distinct so the
    # report can show which serialisation a pattern actually points at.
    ("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl", "DUL", "foundational"),
    ("http://www.ontologydesignpatterns.org/ont/dul/", "DUL-family", "foundational"),
    ("http://www.loa-cnr.it/ontologies/DUL.owl", "DUL", "foundational"),
    ("http://www.loa-cnr.it/ontologies/DOLCE-Lite", "DOLCE-Lite", "foundational"),
    ("http://www.loa-cnr.it/ontologies/ExtendedDnS", "DOLCE-ExtendedDnS", "foundational"),
    ("http://www.loa-cnr.it/ontologies/", "DOLCE-other", "foundational"),
    ("http://www.loa.istc.cnr.it/ontologies/", "DOLCE-other", "foundational"),
    ("http://www.ontologydesignpatterns.org/ont/dolce/", "DOLCE-other", "foundational"),
    ("http://purl.org/dolce/", "DOLCE-other", "foundational"),
    ("http://www.w3.org/2006/time-entry", "DOLCE-other", "foundational"),
    # BFO and its OBO upper layer
    ("http://purl.obolibrary.org/obo/BFO_", "BFO", "foundational"),
    ("http://purl.obolibrary.org/obo/bfo", "BFO", "foundational"),
    ("http://www.ifomis.org/bfo", "BFO", "foundational"),
    ("http://purl.obolibrary.org/obo/COB_", "COB", "foundational"),
    # SUMO / MILO
    ("http://www.ontologyportal.org/SUMO.owl", "SUMO", "foundational"),
    ("http://www.owl-ontologies.com/sumo.owl", "SUMO", "foundational"),
    ("http://www.adampease.org/OP/SUMO.owl", "SUMO", "foundational"),
    # UFO / gUFO
    ("http://purl.org/nemo/gufo#", "gUFO", "foundational"),
    ("http://ontology.com.br/ufo", "UFO", "foundational"),
    # GFO
    ("http://www.onto-med.de/ontologies/gfo", "GFO", "foundational"),
    # others occasionally used as an upper layer
    ("http://www.ihmc.us/users/phayes/", "IKL", "foundational"),
    ("http://sw.opencyc.org/", "OpenCyc", "foundational"),
    ("http://sw.cyc.com/", "OpenCyc", "foundational"),
    ("http://www.yamato-ontology.org/", "YAMATO", "foundational"),
    ("http://www.ontologydesignpatterns.org/ont/cos/", "COS", "foundational"),
    ("http://www.ontologydesignpatterns.org/ont/kco/", "KCO", "foundational"),

    # ---- cross-domain upper vocabularies --------------------------------
    ("http://www.w3.org/ns/prov#", "PROV-O", "cross_domain"),
    ("http://www.w3.org/ns/prov-o#", "PROV-O", "cross_domain"),
    ("http://www.w3.org/2004/02/skos/core#", "SKOS", "cross_domain"),
    ("http://www.w3.org/2008/05/skos", "SKOS", "cross_domain"),
    ("http://www.w3.org/ns/ssn/", "SSN", "cross_domain"),
    ("http://purl.oclc.org/NET/ssnx/ssn", "SSN", "cross_domain"),
    ("http://www.w3.org/ns/sosa/", "SOSA", "cross_domain"),
    ("http://www.w3.org/2006/time#", "OWL-Time", "cross_domain"),
    ("http://www.w3.org/TR/owl-time", "OWL-Time", "cross_domain"),
    ("http://schema.org/", "schema.org", "cross_domain"),
    ("https://schema.org/", "schema.org", "cross_domain"),
    ("http://xmlns.com/foaf/0.1/", "FOAF", "cross_domain"),
    ("http://purl.org/dc/elements/1.1/", "DC", "cross_domain"),
    ("http://purl.org/dc/terms/", "DCTERMS", "cross_domain"),
    ("http://purl.org/dc/dcam/", "DCAM", "cross_domain"),
    ("http://www.cidoc-crm.org/", "CIDOC-CRM", "cross_domain"),
    ("http://erlangen-crm.org/", "CIDOC-CRM", "cross_domain"),
    ("http://www.opengis.net/ont/geosparql#", "GeoSPARQL", "cross_domain"),
    ("http://www.opengis.net/ont/sf#", "GeoSPARQL", "cross_domain"),
    ("http://www.w3.org/2003/01/geo/wgs84_pos#", "WGS84-geo", "cross_domain"),
    ("http://qudt.org/", "QUDT", "cross_domain"),
    ("http://www.wurvoc.org/vocabularies/om-1.8/", "OM-units", "cross_domain"),
    ("http://www.ontology-of-units-of-measure.org/", "OM-units", "cross_domain"),
    ("http://www.w3.org/ns/org#", "ORG", "cross_domain"),
    ("http://purl.org/NET/c4dm/event.owl#", "Event-ontology", "cross_domain"),
    ("http://purl.org/ontology/mo/", "Music-Ontology", "cross_domain"),
    ("http://purl.org/vocab/vann/", "VANN", "cross_domain"),
    ("http://purl.org/pav/", "PAV", "cross_domain"),
    ("http://creativecommons.org/ns#", "CC", "cross_domain"),
    ("http://rdfs.org/ns/void#", "VoID", "cross_domain"),
    ("http://www.w3.org/2006/vcard/ns#", "vCard", "cross_domain"),
    ("http://purl.org/goodrelations/v1#", "GoodRelations", "cross_domain"),
    ("http://purl.org/spar/", "SPAR", "cross_domain"),
    ("http://purl.org/ontology/bibo/", "BIBO", "cross_domain"),
    ("http://semanticscience.org/resource/", "SIO", "cross_domain"),
    ("http://purl.obolibrary.org/obo/IAO_", "IAO", "cross_domain"),
    ("http://purl.obolibrary.org/obo/RO_", "RO", "cross_domain"),
    ("http://purl.obolibrary.org/obo/ro/", "RO", "cross_domain"),
    ("http://purl.obolibrary.org/obo/PATO_", "PATO", "cross_domain"),
    ("http://purl.obolibrary.org/obo/OBI_", "OBI", "cross_domain"),

    # ---- ODP library plumbing -------------------------------------------
    ("http://www.ontologydesignpatterns.org/schemas/cpannotationschema.owl", "CPAnnotationSchema", "odp_infra"),
    ("http://ontologydesignpatterns.org/schemas/cpannotationschema.owl", "CPAnnotationSchema", "odp_infra"),
    ("http://www.ontologydesignpatterns.org/cp/owl/", "ODPA-cp", "odp_infra"),
    ("http://ontologydesignpatterns.org/cp/owl/", "ODPA-cp", "odp_infra"),
    ("http://www.ontologydesignpatterns.org/", "ODPA-other", "odp_infra"),
    ("http://ontologydesignpatterns.org/", "ODPA-other", "odp_infra"),
    ("https://www.ontologydesignpatterns.org/", "ODPA-other", "odp_infra"),
    ("https://ontologydesignpatterns.org/", "ODPA-other", "odp_infra"),
    ("http://www.gong.manchester.ac.uk/odp/", "Manchester-ODP", "odp_infra"),
    ("http://odps.sourceforge.net/", "Manchester-ODP", "odp_infra"),
    ("http://www.co-ode.org/", "CO-ODE", "odp_infra"),
    # DOS-DP files carry no ontology IRI of their own; every one declares
    # <urn:unnamed:ontology#ont1>.  Recorded so the report can say so.
    ("urn:unnamed:ontology", "urn:unnamed (DOS-DP)", "odp_infra"),
    ("http://www.geneontology.org/formats/oboInOwl", "oboInOwl", "odp_infra"),
    ("http://purl.obolibrary.org/obo/IAO_0000115", "IAO", "cross_domain"),

    # ---- artefacts of the tooling, not vocabularies ---------------------
    # These are recorded because they are data-quality findings in their own
    # right: an IRI under file:/// means the file declared no ontology IRI and
    # every entity in it was resolved against the local filesystem, so its
    # entity set is machine-specific and cannot be matched against anything.
    ("file:///", "local-file-IRI", "artefact"),
    ("file:/", "local-file-IRI", "artefact"),
    ("mailto:", "mailto", "artefact"),
    ("http://org.semanticweb.owlapi/error#", "OWLAPI-error", "artefact"),
    ("xsd:", "unresolved-prefix", "artefact"),
    ("owl:", "unresolved-prefix", "artefact"),
    ("rdfs:", "unresolved-prefix", "artefact"),

    # ---- domain vocabularies --------------------------------------------
    ("http://purl.obolibrary.org/obo/UBERON_", "UBERON", "domain"),
    ("http://purl.obolibrary.org/obo/uberon", "UBERON", "domain"),
    ("http://purl.obolibrary.org/obo/GO_", "GO", "domain"),
    ("http://purl.obolibrary.org/obo/go", "GO", "domain"),
    ("http://purl.obolibrary.org/obo/CHEBI_", "CHEBI", "domain"),
    ("http://purl.obolibrary.org/obo/chebi", "CHEBI", "domain"),
    ("http://purl.obolibrary.org/obo/CL_", "CL", "domain"),
    ("http://purl.obolibrary.org/obo/PR_", "PR", "domain"),
    ("http://purl.obolibrary.org/obo/NCBITaxon", "NCBITaxon", "domain"),
    ("http://purl.obolibrary.org/obo/MOP_", "MOP", "domain"),
    ("http://purl.obolibrary.org/obo/NBO_", "NBO", "domain"),
    ("http://purl.obolibrary.org/obo/HP_", "HP", "domain"),
    ("http://purl.obolibrary.org/obo/MP_", "MP", "domain"),
    ("http://purl.obolibrary.org/obo/DOID_", "DOID", "domain"),
    ("http://purl.obolibrary.org/obo/MONDO_", "MONDO", "domain"),
    ("http://purl.obolibrary.org/obo/UPHENO_", "UPHENO", "domain"),
    ("http://purl.obolibrary.org/obo/PO_", "PO", "domain"),
    ("http://purl.obolibrary.org/obo/ENVO_", "ENVO", "domain"),
    ("http://purl.obolibrary.org/obo/", "OBO-other", "domain"),
    # IS-ODP publishes under hsu-ifa.de; hsu-hh.de is the newer institute domain.
    ("http://www.hsu-ifa.de/kb/opcua/", "OPC-UA", "domain"),
    ("http://www.hsu-ifa.de/ontologies/", "IS-ODP-local", "domain"),
    ("http://www.hsu-ifa.de/", "IS-ODP-local", "domain"),
    ("http://www.hsu-hh.de/aut/ontologies/", "IS-ODP-local", "domain"),
    ("http://www.hsu-hh.de/aut/", "IS-ODP-local", "domain"),
    ("http://opcfoundation.org/", "OPC-UA", "domain"),
    ("http://www.cellcycleontology.org/", "CCO-cellcycle", "domain"),
    ("http://iais.fraunhofer.de/", "Fraunhofer", "domain"),
    ("http://aims.fao.org/aos/", "AGROVOC", "domain"),
    ("http://www.fao.org/", "FAO", "domain"),
    ("http://purl.bioontology.org/ontology/", "BioPortal", "domain"),
    ("http://bioportal.bioontology.org/", "BioPortal", "domain"),
    ("http://sweetontology.net/", "SWEET", "domain"),
    ("http://sweet.jpl.nasa.gov/", "SWEET", "domain"),
    ("http://data.ordnancesurvey.co.uk/", "OS-geo", "domain"),
    ("http://dbpedia.org/", "DBpedia", "domain"),
    ("http://www.w3.org/ns/dcat#", "DCAT", "domain"),
    ("http://purl.org/vocommons/voaf#", "VOAF", "domain"),
)

#: precomputed, longest prefix first, so lookup is a simple scan
_NS_SORTED: tuple[tuple[str, str, str], ...] = tuple(
    sorted(NAMESPACES, key=lambda t: -len(t[0]))
)

#: tiers that count as "aligned to something upper-level" in the headline table
ALIGNMENT_TIERS: frozenset[str] = frozenset({"foundational", "cross_domain"})

#: tiers reported but excluded from the alignment headline
NON_ALIGNMENT_TIERS: frozenset[str] = frozenset(
    {"rdf_builtin", "odp_infra", "domain", "artefact"})


def lookup(iri: str) -> tuple[str, str] | None:
    """Return ``(short_name, tier)`` for ``iri``, or ``None`` if unrecognised."""
    for prefix, name, tier in _NS_SORTED:
        if iri.startswith(prefix):
            return name, tier
    return None


_NS_SPLIT = re.compile(r"^(.*[#/:])[^#/]*$")


def namespace_of(iri: str) -> str:
    """The namespace part of ``iri`` (everything up to the last ``#`` or ``/``)."""
    if "#" in iri:
        return iri.rsplit("#", 1)[0] + "#"
    m = _NS_SPLIT.match(iri)
    return m.group(1) if m else iri


def signature_of(iri: str) -> str:
    """A prefix key that survives caching, so the catalogue stays revisable.

    ``extract.py`` stores counts keyed by this string, never by resolved
    vocabulary name, so ``NAMESPACES`` can be corrected without reparsing 147 MB
    of RDF.  The key is the namespace plus, for flat OBO-style local names, the
    local-name prefix up to and including the first underscore -- enough to tell
    ``obo/UBERON_0000061`` from ``obo/BFO_0000050``, which share a namespace but
    not an ontology.

    >>> signature_of("http://purl.obolibrary.org/obo/UBERON_0000061")
    'http://purl.obolibrary.org/obo/UBERON_'
    >>> signature_of("http://www.w3.org/ns/prov#wasDerivedFrom")
    'http://www.w3.org/ns/prov#'
    """
    ns = namespace_of(iri)
    local = iri[len(ns):]
    if "_" in local:
        head = local.split("_", 1)[0]
        if head and len(head) <= 24:
            return ns + head + "_"
    return ns


def map_signature(sig: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """Resolve a stored signature histogram.

    Returns ``(vocab_counts, unrecognised_counts)`` where the first maps
    catalogue short names to distinct-IRI counts and the second keeps the
    signature keys nothing matched.
    """
    known: dict[str, int] = {}
    unknown: dict[str, int] = {}
    for key, n in sig.items():
        hit = lookup(key)
        if hit is None:
            unknown[key] = unknown.get(key, 0) + n
        else:
            known[hit[0]] = known.get(hit[0], 0) + n
    return known, unknown


# --------------------------------------------------------------------------
# 2. subject-domain lexicon
# --------------------------------------------------------------------------

#: domain -> tokens.  Single words only; matching is on split, lower-cased
#: tokens, so "TimeInterval" -> {"time", "interval"} hits ``time_space``.
DOMAIN_LEXICON: dict[str, tuple[str, ...]] = {
    "time_space": (
        "time", "temporal", "interval", "instant", "duration", "date", "datetime",
        "period", "moment", "calendar", "year", "month", "day", "hour", "minute",
        "second", "timezone", "clock", "schedule", "before", "after", "during",
        "space", "spatial", "place", "location", "region", "area", "zone",
        "geometry", "geographic", "geo", "coordinate", "latitude", "longitude",
        "boundary", "adjacency", "path", "trajectory", "route", "distance",
        "proximity", "containment", "topology", "position", "site", "locus",
    ),
    "agents_roles": (
        "agent", "person", "people", "human", "user", "actor", "role", "author",
        "organisation", "organization", "institution", "company", "team",
        "group", "member", "membership", "employee", "employer", "customer",
        "stakeholder", "responsibility", "responsible", "authority",
        "permission", "obligation", "right", "social", "community", "party",
        "identity", "profile", "affiliation", "competence", "skill", "expert",
        "delegation", "delegate", "represent", "act", "acting",
    ),
    "parts_collections": (
        "part", "whole", "component", "componency", "constituent", "constituency",
        "collection", "set", "bag", "list", "sequence", "series", "member",
        "aggregate", "aggregation", "composition", "composite", "assembly",
        "subassembly", "element", "item", "container", "group", "cluster",
        "hierarchy", "tree", "nesting", "nested", "subcollection",
        "partonomy", "haspart", "partof", "hasmember", "memberof",
    ),
    "events_situations": (
        "event", "situation", "state", "process", "action", "activity", "task",
        "plan", "planning", "execution", "occurrence", "happening", "episode",
        "participation", "participant", "coparticipation", "workflow",
        "controlflow", "step", "phase", "stage", "transition", "change",
        "cause", "causal", "effect", "trigger", "outcome", "goal", "objective",
        "precondition", "postcondition", "scenario", "case", "incident",
        "procedure", "operation", "behaviour", "behavior", "affordance",
    ),
    "information_objects": (
        "information", "informationobject", "document", "text", "record",
        "description", "representation", "realization", "realisation",
        "expression", "manifestation", "language", "symbol", "sign", "concept",
        "term", "definition", "annotation", "metadata", "label", "name",
        "identifier", "code", "message", "communication", "publication",
        "article", "paper", "book", "citation", "reference", "topic", "subject",
        "classification", "taxonomy", "vocabulary", "thesaurus", "catalogue",
        "catalog", "schema", "template", "model", "dataset", "data", "file",
        "query", "argument", "claim", "evidence", "note", "score", "music",
        "musical", "image", "picture", "media", "web", "link", "url",
    ),
    "biomedical": (
        "anatomical", "anatomy", "cell", "cellular", "tissue", "organ", "organism",
        "gene", "genetic", "protein", "molecular", "molecule", "chemical",
        "biological", "biology", "phenotype", "phenotypic", "disease", "disorder",
        "abnormal", "abnormality", "pathological", "pathology", "clinical",
        "patient", "diagnosis", "symptom", "syndrome", "morphology",
        "physiology", "physiological", "metabolic", "metabolism", "enzyme",
        "receptor", "membrane", "nucleus", "chromosome", "species", "taxon",
        "phylogenetic", "neuron", "neural", "brain", "muscle", "bone", "blood",
        "immune", "infection", "carcinoma", "tumor", "tumour", "neoplasm",
        "apoptosis", "mitosis", "secretion", "biosynthesis", "regulation",
        "innervates", "gland", "epithelium", "vertebrate", "embryonic",
        "development", "degeneration", "agenesis", "pigmentation", "motility",
        "hormone", "drug", "dose", "therapy", "treatment", "vaccine",
    ),
    "industrial": (
        "manufacturing", "manufacture", "production", "produce", "factory",
        "plant", "machine", "machinery", "equipment", "device", "tool",
        "workpiece", "product", "material", "resource", "batch", "recipe",
        "procedure", "shopfloor", "line", "cell", "station", "robot",
        "mechatronic", "automation", "controller", "actuator", "sensor",
        "signal", "energy", "power", "throughput", "kpi", "oee", "downtime",
        "maintenance", "quality", "tolerance", "assembly", "joining",
        "forming", "cutting", "coating", "casting", "welding", "machining",
        "handling", "transport", "storage", "warehouse", "logistics",
        "supply", "order", "job", "shift", "operator", "asset", "engineering",
        "requirement", "specification", "standard", "norm", "din", "iso",
        "vdi", "opc", "technical", "function", "structure", "system",
        "subsystem", "module", "interface", "port", "connector", "wiring",
    ),
    "measurement_units": (
        "measurement", "measure", "measured", "observation", "observed",
        "quantity", "quality", "value", "amount", "magnitude", "unit", "units",
        "scale", "dimension", "metric", "parameter", "variable", "datum",
        "reading", "sample", "sampling", "calibration", "accuracy", "precision",
        "error", "uncertainty", "threshold", "range", "min", "max", "average",
        "count", "rate", "ratio", "percentage", "price", "cost", "money",
        "currency", "weight", "length", "temperature", "pressure", "volume",
        "density", "speed", "velocity", "frequency", "region", "valuepartition",
        "feature", "property", "attribute", "characteristic",
    ),
    "provenance": (
        "provenance", "source", "derivation", "derived", "generated",
        "generation", "attribution", "attributed", "revision", "version",
        "versioning", "history", "trace", "traceability", "audit", "log",
        "workflow", "pipeline", "run", "invocation", "influence", "influenced",
        "wasderivedfrom", "wasgeneratedby", "wasattributedto", "responsibility",
        "provenanceof", "creator", "created", "modified", "timestamp",
        "certainty", "trust", "reliability", "credibility", "citation",
    ),
}

#: tokens that appear in so many patterns that they carry no domain signal.
#: Removed before scoring; listed here so the exclusion is auditable.
STOPTOKENS: frozenset[str] = frozenset({
    "has", "is", "of", "the", "a", "an", "and", "or", "in", "on", "at", "to",
    "for", "by", "with", "as", "from", "into", "onto", "not", "no", "any",
    "some", "all", "thing", "entity", "object", "class", "property",
    "relation", "type", "kind", "generic", "abstract", "concrete", "other",
    "misc", "general", "core", "base", "top", "upper", "lower", "main",
    "new", "old", "test", "example", "ex", "sample", "pattern", "odp",
    "ontology", "owl", "rdf", "rdfs", "xsd", "id", "uri", "iri", "ns",
    "1", "2", "3", "n", "x", "y", "z",
})

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NONWORD = re.compile(r"[^A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split a local name or label into lower-cased content tokens."""
    if not text:
        return []
    s = _CAMEL.sub(" ", str(text))
    s = _NONWORD.sub(" ", s)
    toks = [t.lower() for t in s.split() if t]
    return [t for t in toks if t not in STOPTOKENS and len(t) > 1]


def local_name(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]


#: inverse index token -> domains (a token may vote for more than one domain,
#: e.g. "cell" is both biomedical and industrial; the scorer splits the vote)
TOKEN_DOMAINS: dict[str, tuple[str, ...]] = {}
for _dom, _toks in DOMAIN_LEXICON.items():
    for _t in _toks:
        TOKEN_DOMAINS.setdefault(_t, ())
        TOKEN_DOMAINS[_t] = TOKEN_DOMAINS[_t] + (_dom,)

DOMAIN_ORDER: tuple[str, ...] = tuple(DOMAIN_LEXICON) + ("unclassified",)

#: external vocabularies that, when imported, are direct evidence for a domain.
#: Used as an *independent* check on the lexicon classifier, never as its input.
VOCAB_DOMAIN_HINT: dict[str, str] = {
    "UBERON": "biomedical", "GO": "biomedical", "CHEBI": "biomedical",
    "CL": "biomedical", "PR": "biomedical", "NCBITaxon": "biomedical",
    "HP": "biomedical", "MP": "biomedical", "DOID": "biomedical",
    "MONDO": "biomedical", "UPHENO": "biomedical", "PATO": "measurement_units",
    "NBO": "biomedical", "MOP": "biomedical", "OBI": "biomedical",
    "PO": "biomedical", "ENVO": "biomedical", "OBO-other": "biomedical",
    "BioPortal": "biomedical",
    "IS-ODP-local": "industrial", "OPC-UA": "industrial",
    "CCO-cellcycle": "biomedical",
    "OWL-Time": "time_space", "GeoSPARQL": "time_space",
    "WGS84-geo": "time_space", "OS-geo": "time_space",
    "FOAF": "agents_roles", "ORG": "agents_roles", "vCard": "agents_roles",
    "PROV-O": "provenance", "PAV": "provenance",
    "SSN": "measurement_units", "SOSA": "measurement_units",
    "QUDT": "measurement_units", "OM-units": "measurement_units",
    # Deliberately absent: DC, DCTERMS, VANN, PAV, CC, VoID.  A pattern that
    # carries dcterms:creator says nothing about its subject domain, and letting
    # metadata vocabularies vote would inflate the agreement rate this table
    # exists to measure honestly.
    "SKOS": "information_objects", "BIBO": "information_objects",
    "SPAR": "information_objects", "CIDOC-CRM": "information_objects",
    "Music-Ontology": "information_objects",
    "Event-ontology": "events_situations",
    "GoodRelations": "measurement_units",
}
