"""Lawrynowicz et al. (SWJ 2018) emerging-design-pattern discovery by tree mining.

    Agnieszka Lawrynowicz, Jedrzej Potoniec, Michal Robaczyk, Tania Tudorache.
    "Discovery of emerging design patterns in ontologies using tree mining."
    Semantic Web 9(4):517-544, 2018. doi:10.3233/SW-170280

This is a **reimplementation**. The authors' FF-SLEUTH is published at
``https://bitbucket.org/leolod/jsleuth``, which returns HTTP 404; no other
runnable release of the method was found. Each stage below cites the section of
the paper it implements, and every departure is stated in the module that makes
it. The mined-pattern archive the authors published for their own corpus
(``http://semantic.cs.put.poznan.pl/bioportal-patterns/``) is still online but
holds patterns mined from BioPortal, not a runnable miner, so it cannot be
applied to a different corpus.

    axiom_trees.py   §4.2  ontology -> forest of labelled axiom trees
    treemine.py      §4.3-4.5  frequent unordered induced subtrees, maximal,
                               filtered to those containing an axiom operator
    cfp.py           §4.6  propositionalisation + maximal frequent itemsets
                           -> class frame patterns
    method.py              turning a mined pattern back into a set of entities
                           of one ontology, which is what this benchmark scores

WHAT THE METHOD EMITS, AND WHAT IS SCORED
-----------------------------------------
The paper's deliverable is an abstract **syntactic template** -- a decoded tree
rendered in Manchester syntax with typed variables in structurally missing
positions -- not a set of entities inside one ontology. ODP-Reuse scores a
per-ontology grouping of entities, so the templates have to be read back onto
the ontology they were mined from. That reading is explicit in ``method.py`` and
is reported as such: the number below is what the method's output implies about
one ontology, not a number the paper itself reports.
"""

from .method import extract, fit_corpus  # noqa: F401
