"""
recobench.llm
=============

Open-source generative-LLM recommender backends (Thread 2), via local Ollama
(default model 'qwen2.5:7b-instruct' -- already pulled on this machine, CPU).

  * LLMRerankRecommender    -- a base retriever proposes a shortlist; Qwen reranks
                               it given the ontology's vocabulary and short pattern
                               descriptions. The recommended hybrid: cheap recall
                               from embeddings, LLM precision on a small candidate
                               set. ~1 LLM call per test ontology.
  * LLMGenerativeRecommender-- pure RAG: give Qwen the ontology summary and the
                               full ODP menu, ask it to output a ranked shortlist.

Both are CPU-bound on this host (~seconds/ontology), so run them on the test
split or a sample -- not leave-one-out over the whole dataset. They degrade
gracefully: any Ollama/parse failure falls back to the base/menu order, so a flaky
model never crashes the benchmark (the failure is logged).
"""
from __future__ import annotations

import logging
import re

from .core import Ontology, Recommender
from .content import build_pattern_docs

log = logging.getLogger("recobench.llm")

DEFAULT_MODEL = "qwen2.5:7b-instruct"


def _menu(stems: list[str], pattern_docs: dict[str, str], max_terms: int = 12
          ) -> str:
    """One 'stem: term, term, ...' line per candidate, to ground the LLM."""
    lines = []
    for s in stems:
        terms = (pattern_docs.get(s, "") or s).split()
        hint = ", ".join(terms[:max_terms])
        lines.append(f"- {s}: {hint}")
    return "\n".join(lines)


def _ask_ollama(model: str, system: str, user: str, num_predict: int = 256,
                temperature: float = 0.0) -> str:
    import ollama
    r = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return r["message"]["content"]


def _parse_ranked(text: str, allowed: list[str]) -> list[str]:
    """Pull the allowed stems out of the model output, in first-seen order."""
    low = text.lower()
    allow = {a.lower(): a for a in allowed}
    # find occurrences with their position; prefer explicit list tokens
    hits: list[tuple[int, str]] = []
    for a_low, a in allow.items():
        m = re.search(r"(?<![a-z0-9])" + re.escape(a_low) + r"(?![a-z0-9])", low)
        if m:
            hits.append((m.start(), a))
    hits.sort(key=lambda x: x[0])
    seen, out = set(), []
    for _pos, a in hits:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


_SYSTEM = (
    "You are an expert ontology engineer. You recommend Ontology Design Patterns "
    "(ODPs) that a given ontology most likely reuses. You only choose from the "
    "provided candidate list and you answer with a ranked, comma-separated list of "
    "candidate identifiers, most relevant first, and nothing else."
)


class LLMRerankRecommender(Recommender):
    """Shortlist from a base retriever, then Qwen reranks it."""
    uses_dataset = True

    def __init__(self, base: Recommender, model: str = DEFAULT_MODEL,
                 shortlist: int = 12, label: str | None = None):
        self.base = base
        self.model = model
        self.shortlist = shortlist
        self.name = label or f"llm_rerank[{model}]"

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.base.fit(train, catalog)
        self.pattern_docs = build_pattern_docs(train)

    def recommend(self, onto: Ontology, top_n: int = 10):
        cand = self.base.recommend(onto, top_n=self.shortlist)
        if len(cand) <= 1:
            return cand[:top_n]
        user = (
            f"Ontology vocabulary (terms):\n{(onto.text or onto.name)[:1500]}\n\n"
            f"Candidate ODPs (identifier: hint terms):\n"
            f"{_menu(cand, self.pattern_docs)}\n\n"
            f"Return the {top_n} most relevant candidate identifiers, ranked, "
            f"comma-separated, choosing only from the list above."
        )
        try:
            txt = _ask_ollama(self.model, _SYSTEM, user)
            ranked = _parse_ranked(txt, cand)
        except Exception as e:  # ollama down / timeout
            log.warning("LLM rerank failed for %s: %s; using base order",
                        onto.name, e)
            ranked = []
        # fill any LLM omissions with the base order so recall is never worse
        for s in cand:
            if s not in ranked:
                ranked.append(s)
        return ranked[:top_n]


class LLMGenerativeRecommender(Recommender):
    """Pure RAG: Qwen picks from a bounded menu of catalog ODPs."""
    uses_dataset = True

    def __init__(self, model: str = DEFAULT_MODEL, menu_size: int = 24,
                 label: str | None = None):
        self.model = model
        self.menu_size = menu_size
        self.name = label or f"llm_generative[{model}]"

    def fit(self, train, catalog):
        from .core import pattern_popularity
        self.catalog = list(catalog)
        self.pattern_docs = build_pattern_docs(train)
        pop = pattern_popularity(train)
        # bound the menu to the most-common patterns so the prompt stays small
        self.menu_stems = sorted(catalog, key=lambda s: pop.get(s, 0),
                                 reverse=True)[: self.menu_size]
        self._pop_order = self.menu_stems

    def recommend(self, onto: Ontology, top_n: int = 10):
        user = (
            f"Ontology vocabulary (terms):\n{(onto.text or onto.name)[:1500]}\n\n"
            f"Catalog of ODPs (identifier: hint terms):\n"
            f"{_menu(self.menu_stems, self.pattern_docs)}\n\n"
            f"Which ODPs does this ontology most likely reuse? Return the {top_n} "
            f"most relevant identifiers, ranked, comma-separated, only from the "
            f"catalog above."
        )
        try:
            txt = _ask_ollama(self.model, _SYSTEM, user)
            ranked = _parse_ranked(txt, self.menu_stems)
        except Exception as e:
            log.warning("LLM generative failed for %s: %s; using popularity",
                        onto.name, e)
            ranked = []
        for s in self._pop_order:
            if s not in ranked:
                ranked.append(s)
        return ranked[:top_n]


# ======================================================================
# Phase 1/2: hybrid LLM backends
# ======================================================================

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / len(a | b) if i else 0.0


def _neighbours(onto, train, n: int):
    """Top-n most term-similar (Jaccard) training ontologies, as
    (train_ontology, similarity) sorted descending."""
    sims = [(_jaccard(onto.tokens, t.tokens), t) for t in train]
    sims = [(s, t) for s, t in sims if s > 0]
    sims.sort(key=lambda x: x[0], reverse=True)
    return [(t, s) for s, t in sims[:n]]


class LLMFewShotRerank(Recommender):
    """Phase 2: in-context collaborative filtering.

    A base retriever proposes a shortlist; the prompt is augmented with a few
    solved examples --- the reuse records of the most term-similar TRAINING
    ontologies --- so the LLM can imitate the reuse behaviour of similar
    ontologies rather than guessing purely from content. This injects the
    collaborative signal (which similar ontologies reused what) that the plain
    rerank lacks.
    """
    uses_dataset = True

    def __init__(self, base: Recommender, model: str = DEFAULT_MODEL,
                 shortlist: int = 12, n_examples: int = 3,
                 cf_evidence: bool = True, label: str | None = None):
        self.base = base
        self.model = model
        self.shortlist = shortlist
        self.n_examples = n_examples
        self.cf_evidence = cf_evidence
        self.name = label or f"llm_fewshot_rerank[{model}]"

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.base.fit(train, catalog)
        self.pattern_docs = build_pattern_docs(train)
        self.train = train

    def _examples_block(self, onto) -> str:
        lines = []
        for tr, sim in _neighbours(onto, self.train, self.n_examples):
            terms = " ".join((tr.text or tr.name).split()[:40])
            reused = ", ".join(sorted(tr.gt_stems))
            lines.append(f"Example ontology (terms: {terms})\n  reused: {reused}")
        return "\n".join(lines)

    def _cf_hint(self, onto) -> str:
        counts = {}
        for tr, sim in _neighbours(onto, self.train, 15):
            for s in tr.gt_stems:
                counts[s] = counts.get(s, 0.0) + sim
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        return ", ".join(f"{s}" for s, _ in top)

    def recommend(self, onto, top_n: int = 10):
        cand = self.base.recommend(onto, top_n=self.shortlist)
        if len(cand) <= 1:
            return cand[:top_n]
        parts = [f"Similar solved cases:\n{self._examples_block(onto)}\n"]
        if self.cf_evidence:
            parts.append(f"Patterns favoured by similar ontologies: "
                         f"{self._cf_hint(onto)}\n")
        parts.append(
            f"Now the target ontology (terms):\n{(onto.text or onto.name)[:1500]}\n\n"
            f"Candidate ODPs (identifier: hint terms):\n"
            f"{_menu(cand, self.pattern_docs)}\n\n"
            f"Using the solved cases as guidance, return the {top_n} most relevant "
            f"candidate identifiers for the target ontology, ranked, "
            f"comma-separated, choosing only from the candidate list.")
        user = "\n".join(parts)
        try:
            ranked = _parse_ranked(_ask_ollama(self.model, _SYSTEM, user), cand)
        except Exception as e:
            log.warning("few-shot rerank failed for %s: %s", onto.name, e)
            ranked = []
        for s in cand:
            if s not in ranked:
                ranked.append(s)
        return ranked[:top_n]


_SCORE_SYSTEM = (
    "You are an expert ontology engineer scoring how likely a given ontology "
    "reuses each candidate Ontology Design Pattern. For every candidate you output "
    "one line 'identifier: score' where score is an integer 0 (irrelevant) to 3 "
    "(certainly reused), and nothing else.")


class LLMPointwiseScorer(Recommender):
    """Phase 1: LLM-as-scorer instead of LLM-as-ranker.

    A base retriever proposes a shortlist; the LLM assigns each candidate an
    independent 0--3 relevance score in a single call. Ranking by that score is
    more stable than list reordering and yields a calibrated per-candidate signal
    that can be fused (Phase 4).
    """
    uses_dataset = True

    def __init__(self, base: Recommender, model: str = DEFAULT_MODEL,
                 shortlist: int = 12, label: str | None = None):
        self.base = base
        self.model = model
        self.shortlist = shortlist
        self.name = label or f"llm_pointwise[{model}]"

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.base.fit(train, catalog)
        self.pattern_docs = build_pattern_docs(train)

    def recommend(self, onto, top_n: int = 10):
        cand = self.base.recommend(onto, top_n=self.shortlist)
        if len(cand) <= 1:
            return cand[:top_n]
        user = (
            f"Ontology (terms):\n{(onto.text or onto.name)[:1500]}\n\n"
            f"Candidate ODPs (identifier: hint terms):\n"
            f"{_menu(cand, self.pattern_docs)}\n\n"
            f"Score every candidate 0--3 as 'identifier: score', one per line.")
        scores: dict[str, float] = {}
        try:
            txt = _ask_ollama(self.model, _SCORE_SYSTEM, user, num_predict=400)
            allow = {c.lower(): c for c in cand}
            for ln in txt.splitlines():
                m = re.match(r"\s*[-*]?\s*([A-Za-z0-9_\-]+)\s*[:=]\s*(\d)", ln)
                if m and m.group(1).lower() in allow:
                    scores[allow[m.group(1).lower()]] = float(m.group(2))
        except Exception as e:
            log.warning("pointwise scoring failed for %s: %s", onto.name, e)
        # rank by score; base order breaks ties and fills unscored candidates
        ranked = sorted(cand, key=lambda c: (-scores.get(c, -1), cand.index(c)))
        return ranked[:top_n]


class SelfConsistentRecommender(Recommender):
    """Phase 2: self-consistency. Runs a stochastic backend `runs` times and
    aggregates the rankings by Borda count (rank-position voting)."""
    uses_dataset = True

    def __init__(self, make_base, runs: int = 3, temperature: float = 0.5,
                 label: str = "llm_selfconsistent"):
        self.make_base = make_base
        self.runs = runs
        self.temperature = temperature
        self.name = label

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.bases = []
        for _ in range(self.runs):
            b = self.make_base()
            b.temperature = self.temperature  # backends read this if present
            b.fit(train, catalog)
            self.bases.append(b)

    def recommend(self, onto, top_n: int = 10):
        borda: dict[str, float] = {}
        for b in self.bases:
            ranked = b.recommend(onto, top_n=top_n)
            for pos, stem in enumerate(ranked):
                borda[stem] = borda.get(stem, 0.0) + (top_n - pos)
        return sorted(borda, key=lambda s: borda[s], reverse=True)[:top_n]
