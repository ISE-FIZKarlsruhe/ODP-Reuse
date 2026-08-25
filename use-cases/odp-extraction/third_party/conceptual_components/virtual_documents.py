# Vendored from stlab-istc-cnr/conceptual-components
# @ e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5
# path: conceptual-components-extraction/project_root/
#       ontology_entities_to_annotatedtext_from_txt.py   (STEP 2)
#
# Step 2 of the upstream pipeline: turn each community's entity IRIs into a
# virtual document, by concatenating English labels where an entity has one and
# the camel-case/underscore-split local name where it does not, optionally
# followed by rdfs:comment / skos:definition.
#
# PATCHES (see PROVENANCE.md):
#  P2.1  `Graph.preferredLabel` was removed in rdflib 6. `_preferred_label`
#        reimplements the rdflib 4.2.2 behaviour the upstream code relied on:
#        return the (predicate, object) pairs for skos:prefLabel then
#        rdfs:label, filtered by language.
#  P2.2  The upstream entry point walks a flat folder of ontology files and a
#        parallel folder of `*_original_uris.txt`. The per-entity and
#        per-community functions are unchanged; only the folder walk is replaced
#        by a function taking (graph, list-of-communities), because this corpus
#        stores one ontology per directory and the runner already holds the
#        parsed graph. No text-construction rule is altered.
#  P2.3  `langdetect` is imported lazily and, if absent, language detection
#        returns False -- the same value the upstream `except: pass` produces.
#
# The namespace exclusion list is upstream's, taken from project_root/readme.md
# step 2: XMLSchema, owl, rdf-schema, rdf-syntax-ns.

from rdflib import BNode, OWL, RDF, RDFS, URIRef

DEFAULT_EXCLUDED_NAMESPACES = (
    "http://www.w3.org/2001/XMLSchema",
    "http://www.w3.org/2002/07/owl",
    "http://www.w3.org/2000/01/rdf-schema",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns",
)

SKOS_PREF_LABEL = URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
SKOS_DEFINITION = URIRef("http://www.w3.org/2004/02/skos/core#definition")

_PROPERTY_TYPES = {
    RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty,
    OWL.AnnotationProperty, OWL.FunctionalProperty,
    OWL.InverseFunctionalProperty, OWL.TransitiveProperty,
    OWL.SymmetricProperty, OWL.AsymmetricProperty,
    OWL.IrreflexiveProperty, OWL.ReflexiveProperty,
}


def detect_language(text, language_to_detect):
    """Upstream `detect_language`; P2.3 makes langdetect optional."""
    try:
        from langdetect import detect
        return detect(str(text)) == language_to_detect
    except Exception:  # noqa: BLE001 - upstream swallows every failure
        return False


def get_onto_local_name(uri):
    if "#" in uri:
        return str(uri).split("#")[1]
    return str(uri).split("/")[-1]


def splitAtUpperCaseUnderscore(s):
    """Upstream, verbatim."""
    for i in range(len(s) - 1)[::-1]:
        if s[i].isupper() and s[i + 1].islower():
            s = s[:i] + ' ' + s[i:]
        if s[i].isupper() and s[i - 1].islower():
            s = s[:i] + ' ' + s[i:]
    s_underscore = " ".join(s.split("_"))
    return " ".join(s_underscore.split())


def _preferred_label(graph, subject, lang):
    """P2.1: rdflib 4.2.2's Graph.preferredLabel, reduced to what step 2 uses.

    Returns a list of (predicate, object) pairs: skos:prefLabel first, falling
    back to rdfs:label, keeping only literals whose language tag matches.
    """
    for pred in (SKOS_PREF_LABEL, RDFS.label):
        hits = [(pred, o) for o in graph.objects(subject, pred)
                if getattr(o, "language", None) == lang]
        if hits:
            return hits
    return []


def get_language_labels(graph, entity, language):
    """Upstream, with P2.1 substituted for Graph.preferredLabel."""
    labels = []
    label_list = _preferred_label(graph, URIRef(entity), lang=language)
    if len(label_list) != 0:
        for i in range(len(label_list)):
            labels.append(str(label_list[i][1]))
    elif (URIRef(entity), RDFS.label, None) in graph:
        for _a, _b, c in graph.triples((URIRef(entity), RDFS.label, None)):
            if detect_language(c, language):
                labels.append(str(c))
    return labels


def get_language_comments(graph, entity, language):
    """Upstream, verbatim apart from str() coercion."""
    comments = []
    try:
        for _a, _b, c in graph.triples((URIRef(entity), RDFS.comment, None)):
            if getattr(c, "language", None) == language and c is not None:
                comments.append(str(c))
            elif detect_language(c, language):
                comments.append(str(c))
        if len(comments) == 0:
            for _a, _b, c in graph.triples((URIRef(entity), SKOS_DEFINITION, None)):
                if getattr(c, "language", None) == language and c is not None:
                    comments.append(str(c))
                elif detect_language(c, language):
                    comments.append(str(c))
    except Exception:  # noqa: BLE001
        return comments
    return comments


def _annotate(graph, entity, entity_text):
    """Upstream's [C]/[P]/[OTHER] marker loop."""
    annotated = ""
    for _s, _p, o in graph.triples((URIRef(entity), RDF.type, None)):
        if o in _PROPERTY_TYPES:
            annotated = "[" + entity_text + "[P]]"
        elif o == RDFS.Class or o == OWL.Class:
            return "[" + entity_text + "[C]]"
        else:
            annotated = "[" + entity_text + "[OTHER]]"
    return annotated


def create_text_from_ontology_entity_noblanknodes(graph, entity, language,
                                                  also_comments):
    """Upstream's two per-entity functions, merged on the `also_comments` flag
    that upstream's caller uses to choose between them. The label branch, the
    local-name fallback and the comment branch are unchanged."""
    entity_text = ""
    annotated_entity_text = ""
    if isinstance(URIRef(entity), BNode):
        return None
    entity_uri = str(entity)
    entity_local_name = get_onto_local_name(entity_uri)
    split_entity_local_name = splitAtUpperCaseUnderscore(entity_local_name)

    label_list = get_language_labels(graph, entity, language)
    if len(label_list) != 0:
        for label in label_list:
            if label not in entity_text:
                entity_text += label + " "
                annotated_entity_text = _annotate(graph, entity, entity_text)
    else:
        if entity_local_name not in entity_text:
            entity_text += split_entity_local_name + " "
            annotated_entity_text = _annotate(graph, entity, entity_text)

    if also_comments:
        comment_list = get_language_comments(graph, entity, language)
        if len(comment_list) != 0:
            for comment in comment_list:
                if comment not in entity_text:
                    entity_text += comment + " "
                    annotated_entity_text = _annotate(graph, entity, entity_text)

    return {"entity localname": split_entity_local_name,
            "entity uri": entity_uri,
            "entity text": entity_text,
            "annotated text": annotated_entity_text}


def community_documents(graph, communities, also_comments=False,
                        remove_duplicates=True,
                        excluded_namespaces=DEFAULT_EXCLUDED_NAMESPACES):
    """P2.2: the per-community loop of upstream's `communities_text_from_folder`,
    over communities held in memory instead of read from
    `*_original_uris.txt` files.

    Returns one (text, annotated_text) pair per community, in order. The
    duplicate-word removal is upstream's: unique words in first-occurrence
    order, applied to the concatenated text.
    """
    out = []
    for community in communities:
        text_concatenated = ""
        annotated_concatenated = ""
        for url in sorted(community):
            if any(x in str(url) for x in excluded_namespaces):
                continue
            d = create_text_from_ontology_entity_noblanknodes(
                graph, url, "en", also_comments)
            if d is None:
                continue
            text_concatenated += d["entity text"] + " "
            annotated_concatenated += (d["annotated text"] or "") + " "
        text_rejoined = "".join(text_concatenated.splitlines())
        annotated_rejoined = "".join(annotated_concatenated.splitlines())
        if remove_duplicates:
            words = text_rejoined.split()
            text_rejoined = " ".join(sorted(set(words), key=words.index))
        out.append((text_rejoined.strip(), annotated_rejoined.strip()))
    return out
