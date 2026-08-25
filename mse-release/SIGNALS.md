# Signals

This release reports no accuracy, because the corpus has no ground truth. It
reports five signals instead. Each is a ranking device for deciding which
candidates deserve a domain expert's attention, and each has a failure mode that
a consumer needs to know. They are stated here in the way `release-iri/FILTERS.md`
states the gate rules: what the signal is, why it is computed, and what it does
*not* license.

---

## 1. Yield

**What.** Groups proposed per ontology; group size; the share of an ontology's
entities that appear in at least one group.

**Why.** Any evaluation that costs human attention per group is not evaluating
the same amount of work per method. A method proposing 51 groups per ontology
and one proposing 1.7 are not comparable on a per-group measure.

**Failure mode.** Coverage is 1.000 for any method that *partitions* — the
reference points and both K-Means configurations — and lower for methods that
*select*. The column separates two kinds of method; it does not rank them.

## 2. Cross-method agreement

**What.** Hungarian entity-level $F_1$ between two methods' groups on the same
ontology, the same metric used to score against ground truth, with one method's
output standing in for the answer key.

**Why.** Two methods built on different principles proposing the same group is
evidence the group is a structure of the ontology rather than an artefact of one
algorithm.

**Failure mode.** Agreement *within* a family measures shared implementation,
not shared insight: two configurations of the same embedder reach 0.78, and the
two stages of one pipeline 0.70, while no pair drawn from two different families
exceeds 0.23. Compute this signal across families or not at all. Trivial
reference predictions reach 0.90, which is the scale any value should be read
against.

## 3. Recurrence across ontologies

**What.** Whether a group's entity set, or its set of local names, also appears
in another ontology.

**Why.** A design pattern recurs by definition. A group that appears in several
ontologies has evidence no single-ontology measure can give.

**Failure mode.** **This is the signal most likely to mislead.** Ranking by
recurrence alone puts pairs of BFO upper-level classes at the top, because a
dozen MSE ontologies import BFO. That is shared *scaffolding*, not shared
design. Exclude groups composed entirely of foundational terms (BFO, EMMO, IAO,
RO, PMDco, CCO, QUDT) and require at least one term from the host ontology's own
namespace before using this signal to rank.

## 4. Foundational anchoring

**What.** Whether a group's entities include a term from a foundational or
mid-level vocabulary.

**Why.** A group specialising an upper-level ontology is more likely a modelling
decision than an incidental clique.

**Failure mode.** It rewards methods that key on subsumption. Axiom-tree mining
reaches 86% anchored because it mines `SubClassOf` and `EquivalentTo` axioms,
and in this corpus those axioms overwhelmingly attach a domain class to an
upper-level one. High anchoring can mean the method found the domain's
foundational commitment rather than a pattern.

## 5. Contact with a documented ODP

**What.** Whether a group overlaps any of the 120 reference patterns of the
ODP-Reuse libraries.

**Why.** A group recovering a documented pattern is a validation signal; a group
overlapping none is a candidate novel one. Both matter, differently.

**Failure mode.** As a per-group rate it rewards coarseness: a method proposing
large groups intersects a known pattern more often by chance. The ordering of
the coarse methods on this signal largely reproduces the ordering of their group
sizes.

---

## What none of them supports

No combination of these signals establishes that a candidate is a good design
pattern. They decide what an expert is asked about. The expert answers.
