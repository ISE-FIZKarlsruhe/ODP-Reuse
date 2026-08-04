"""Analysis 3 -- subject domain of each pattern, and whether reuse concentrates.

METHOD (and it is a heuristic; the uncertainty is measured, see below)
---------------------------------------------------------------------
Each pattern gets three *independent* domain votes, then one combined label.

1. ``domain_stem`` -- tokens of the file stem, camel/snake split, stopped
   (``vocab.STOPTOKENS``), scored against ``vocab.DOMAIN_LEXICON``.  The stem is
   the name the library gives the pattern, so it is the single most
   authoritative lexical signal, but it is one to four tokens long.

2. ``domain_entities`` -- tokens of the local names and ``rdfs:label`` values of
   the pattern's *own-namespace* declared entities.  Where a pattern declares no
   own-namespace entity at all (every DOS-DP file: they are all
   ``urn:unnamed:ontology#ont1`` and every entity in them is an imported OBO
   IRI) the evidence falls back to all declared entities and the record says so
   in ``naming_evidence_source``.  Both are capped at 2000 terms per file
   (``extract.MAX_NAMING_TERMS``) so a physically merged import cannot outvote
   the pattern's own vocabulary without limit.

3. ``domain_vocab`` -- the external vocabularies the pattern imports or
   references, mapped through ``vocab.VOCAB_DOMAIN_HINT``.  Structural, not
   lexical: importing UBERON is evidence of a biomedical pattern regardless of
   what anything is called.  Available for only a minority of patterns.

Scoring.  A token votes for every domain whose lexicon contains it, with weight
``1/len(domains(token))`` -- so ``cell``, in both the biomedical and the
industrial lexicon, contributes 0.5 to each rather than 1 to both.  Stem tokens
carry ``STEM_WEIGHT`` (3.0), entity tokens 1.0.  The label is the arg-max; ties
and thin evidence go to ``unclassified``:

    total score < MIN_SCORE (1.0)            -> unclassified
    top1 - top2 < MIN_MARGIN_FRAC of top1    -> ambiguous, label kept, flagged

UNCERTAINTY, QUANTIFIED
    ``dm_uncertainty.csv`` reports, per library: share unclassified, share
    ambiguous under the margin rule, the mean normalised margin, and -- the only
    genuinely external check available -- the agreement rate between the lexical
    label and ``domain_vocab`` on the subset where a vocabulary hint exists.
    That agreement rate is the number to quote as the classifier's accuracy
    proxy.  There is no hand-labelled gold set for pattern domains in this
    repository, so no true accuracy can be reported.

REUSE CROSS-TABULATION
    ``pairs.csv`` names a pattern by stem only, so the join is on case-folded
    stem (see ``reuse.py``).  ``dm_reuse_crosstab.csv`` gives, per domain:
    patterns in the libraries, patterns with >=1 reuse pair, pairs, ontologies,
    and the reuse rate.  ``dm_reuse_unmatched.csv`` lists reused stems that no
    library file supplies, which bounds how much of the reuse evidence the
    cross-tabulation can actually cover.

Outputs
-------
dm_per_pattern.csv     one row per pattern with all three votes and the scores
dm_by_library.csv      library x domain counts
dm_uncertainty.csv     per-library uncertainty measures
dm_reuse_crosstab.csv  domain x reuse, over the joinable subset
dm_reuse_by_library.csv domain x library x reuse
dm_reuse_unmatched.csv reused stems with no library file
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import toplevel, units, vocab
from .libraries import LIBRARIES, LIB_ORDER
from .reuse import ambiguous_stems, load_pairs, reuse_by_stem

STEM_WEIGHT = 3.0
ENTITY_WEIGHT = 1.0
MIN_SCORE = 1.0
MIN_MARGIN_FRAC = 0.15
UNCLASSIFIED = "unclassified"


def _score(tokens: list[str], weight: float, into: dict[str, float]) -> None:
    for t in tokens:
        doms = vocab.TOKEN_DOMAINS.get(t)
        if not doms:
            continue
        w = weight / len(doms)
        for d in doms:
            into[d] = into.get(d, 0.0) + w


def _argmax(scores: dict[str, float]) -> tuple[str, float, float]:
    if not scores:
        return UNCLASSIFIED, 0.0, 0.0
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    return top, s1, s2


def classify(rec: dict) -> dict:
    stem_tokens = vocab.tokenize(rec["stem"])
    ent_tokens: list[str] = []
    for ln in rec.get("naming_localnames", []):
        ent_tokens += vocab.tokenize(ln)
    for lb in rec.get("naming_labels", []):
        ent_tokens += vocab.tokenize(lb)

    s_stem: dict[str, float] = {}
    _score(stem_tokens, 1.0, s_stem)
    s_ent: dict[str, float] = {}
    _score(ent_tokens, 1.0, s_ent)

    combined: dict[str, float] = {}
    _score(stem_tokens, STEM_WEIGHT, combined)
    _score(ent_tokens, ENTITY_WEIGHT, combined)

    # vocabulary hint (structural, independent of the lexicon)
    ref_vocab, _unknown, imp_vocab = toplevel.resolve(rec)
    vote: Counter = Counter()
    for name, n in ref_vocab.items():
        d = vocab.VOCAB_DOMAIN_HINT.get(name)
        if d:
            vote[d] += n
    for name in imp_vocab:
        d = vocab.VOCAB_DOMAIN_HINT.get(name)
        if d:
            vote[d] += 5           # an import is stronger than a mention
    dom_vocab = vote.most_common(1)[0][0] if vote else ""

    d_stem, _, _ = _argmax(s_stem)
    d_ent, _, _ = _argmax(s_ent)
    label, s1, s2 = _argmax(combined)
    total = sum(combined.values())
    if total < MIN_SCORE:
        label = UNCLASSIFIED
    margin = (s1 - s2) / s1 if s1 else 0.0
    ambiguous = int(label != UNCLASSIFIED and margin < MIN_MARGIN_FRAC)

    return {
        "domain": label,
        "domain_score_top": round(s1, 3),
        "domain_score_second": round(s2, 3),
        "domain_score_total": round(total, 3),
        "domain_margin": round(margin, 3),
        "domain_ambiguous": ambiguous,
        "domain_stem": d_stem if s_stem else UNCLASSIFIED,
        "domain_entities": d_ent if s_ent else UNCLASSIFIED,
        "domain_vocab": dom_vocab,
        "n_stem_tokens": len(stem_tokens),
        "n_entity_tokens": len(ent_tokens),
        "runner_up": sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))[1][0]
                     if len(combined) > 1 else "",
        "all_scores": "|".join(f"{k}:{v:.2f}" for k, v in
                               sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


# --------------------------------------------------------------------------

def _classify_all(records: list[dict], unit: str) -> list[dict]:
    out: list[dict] = []
    for r in units.select(records, unit):
        row = {
            "unit": unit,
            "library": r["library"],
            "key": r["key"],
            "stem": r["stem"],
            "group": r["group"],
            "role": r["role"],
            "naming_evidence_source": r.get("naming_evidence_source", ""),
            "n_own_declared": r.get("n_own_declared", 0),
            "n_naming_iris": r.get("n_naming_iris", 0),
        }
        row.update(classify(r))
        out.append(row)
    return out


def run(records: list[dict], out_dir: Path, pairs_csv: Path | None = None,
        unit: str = units.DEFAULT_UNIT, reuse_unit: str = "stem_global") -> dict:
    per = _classify_all(records, unit)
    units.write_csv(out_dir / "dm_per_pattern.csv", per)

    # The reuse cross-tabulation runs on a *different* unit on purpose.
    # ``primary_content`` excludes ODPA's shipped dependencies -- and the two
    # most-reused patterns in the whole benchmark, DUL (218 pairs) and
    # cpannotationschema (51), are exactly such files.  ``content`` keeps them but
    # collapses the 110 byte-identical DOS-DP stubs onto one representative, so
    # 110 reused stems lose theirs.  ``stem_global`` is the only unit on which
    # each stem in pairs.csv is represented exactly once.
    per_reuse = per if reuse_unit == unit else _classify_all(records, reuse_unit)

    # ---- library x domain --------------------------------------------
    lib_tot = Counter(p["library"] for p in per)
    cross = Counter((p["library"], p["domain"]) for p in per)
    rows = []
    for lib in LIB_ORDER:
        if not lib_tot[lib]:
            continue
        for dom in vocab.DOMAIN_ORDER:
            n = cross[(lib, dom)]
            rows.append({
                "library": lib, "library_name": LIBRARIES[lib][1],
                "domain": dom, "n_patterns": n,
                "n_patterns_in_library": lib_tot[lib],
                "pct_of_library": units.pct(n, lib_tot[lib]),
            })
    units.write_csv(out_dir / "dm_by_library.csv", rows)

    # ---- uncertainty --------------------------------------------------
    unc = []
    for lib in list(LIB_ORDER) + ["ALL"]:
        sub = per if lib == "ALL" else [p for p in per if p["library"] == lib]
        if not sub:
            continue
        n = len(sub)
        n_unc = sum(1 for p in sub if p["domain"] == UNCLASSIFIED)
        n_amb = sum(p["domain_ambiguous"] for p in sub)
        with_hint = [p for p in sub if p["domain_vocab"]]
        agree = sum(1 for p in with_hint if p["domain"] == p["domain_vocab"])
        stem_ok = [p for p in sub if p["domain_stem"] != UNCLASSIFIED]
        stem_agree = sum(1 for p in stem_ok if p["domain"] == p["domain_stem"])
        ent_ok = [p for p in sub if p["domain_entities"] != UNCLASSIFIED]
        ent_agree = sum(1 for p in ent_ok if p["domain"] == p["domain_entities"])
        unc.append({
            "library": lib,
            "library_name": LIBRARIES[lib][1] if lib in LIBRARIES else "all libraries",
            "n_patterns": n,
            "n_unclassified": n_unc,
            "pct_unclassified": units.pct(n_unc, n),
            "n_ambiguous_margin_lt_%.2f" % MIN_MARGIN_FRAC: n_amb,
            "pct_ambiguous": units.pct(n_amb, n),
            "mean_margin": round(sum(p["domain_margin"] for p in sub) / n, 3),
            "median_margin": round(sorted(p["domain_margin"] for p in sub)[n // 2], 3),
            "n_with_vocab_hint": len(with_hint),
            "n_vocab_hint_agrees": agree,
            "pct_vocab_hint_agrees": units.pct(agree, len(with_hint)),
            "n_stem_signal": len(stem_ok),
            "pct_stem_agrees_with_combined": units.pct(stem_agree, len(stem_ok)),
            "n_entity_signal": len(ent_ok),
            "pct_entity_agrees_with_combined": units.pct(ent_agree, len(ent_ok)),
        })
    units.write_csv(out_dir / "dm_uncertainty.csv", unc)

    out = {"per_pattern": per, "by_library": rows, "uncertainty": unc}

    # ---- reuse cross-tabulation ---------------------------------------
    if pairs_csv and Path(pairs_csv).is_file():
        pairs = load_pairs(Path(pairs_csv))
        ru = reuse_by_stem(pairs)
        amb = ambiguous_stems(units.parsed_only(records))

        # stem -> the domain labels the libraries give it
        by_stem: dict[str, list[dict]] = defaultdict(list)
        for p in per_reuse:
            by_stem[p["stem"].casefold()].append(p)

        dom_stats: dict[str, dict] = {}
        dom_lib_stats: dict[tuple[str, str], dict] = {}
        counted_stems: set[str] = set()
        for p in per_reuse:
            k = p["stem"].casefold()
            ev = ru.get(k)
            d = dom_stats.setdefault(p["domain"], dict(
                n_patterns=0, n_reused=0, n_pairs=0, n_complete=0,
                ontologies=set()))
            d["n_patterns"] += 1
            dl = dom_lib_stats.setdefault((p["domain"], p["library"]), dict(
                n_patterns=0, n_reused=0, n_pairs=0, ontologies=set()))
            dl["n_patterns"] += 1
            if ev:
                counted_stems.add(k)
                d["n_reused"] += 1
                d["n_pairs"] += ev["n_pairs"]
                # union, not sum: two patterns of the same domain reused by the
                # same ontology are one reusing ontology, not two
                d["ontologies"] |= ev["ontologies"]
                d["n_complete"] += ev["n_complete"]
                dl["n_reused"] += 1
                dl["n_pairs"] += ev["n_pairs"]
                dl["ontologies"] |= ev["ontologies"]

        ct = []
        for dom in vocab.DOMAIN_ORDER:
            d = dom_stats.get(dom)
            if not d:
                continue
            ct.append({
                "unit": reuse_unit,
                "domain": dom,
                "n_patterns": d["n_patterns"],
                "n_patterns_reused": d["n_reused"],
                "pct_patterns_reused": units.pct(d["n_reused"], d["n_patterns"]),
                "n_reuse_pairs": d["n_pairs"],
                "n_distinct_reusing_ontologies": len(d["ontologies"]),
                "n_complete_pairs": d["n_complete"],
                "pairs_per_pattern": round(d["n_pairs"] / d["n_patterns"], 2),
            })
        tot_p = sum(r["n_patterns"] for r in ct)
        tot_pr = sum(r["n_patterns_reused"] for r in ct)
        tot_pairs = sum(r["n_reuse_pairs"] for r in ct)
        for r in ct:
            r["pct_of_all_patterns"] = units.pct(r["n_patterns"], tot_p)
            r["pct_of_all_reused_patterns"] = units.pct(r["n_patterns_reused"], tot_pr)
            r["pct_of_all_pairs"] = units.pct(r["n_reuse_pairs"], tot_pairs)
        units.write_csv(out_dir / "dm_reuse_crosstab.csv", ct)

        cl = []
        for (dom, lib), d in sorted(dom_lib_stats.items(),
                                    key=lambda kv: (vocab.DOMAIN_ORDER.index(kv[0][0]),
                                                    LIB_ORDER.index(kv[0][1]))):
            cl.append({
                "unit": reuse_unit,
                "domain": dom, "library": lib,
                "n_patterns": d["n_patterns"],
                "n_patterns_reused": d["n_reused"],
                "pct_patterns_reused": units.pct(d["n_reused"], d["n_patterns"]),
                "n_reuse_pairs": d["n_pairs"],
                "n_distinct_reusing_ontologies": len(d["ontologies"]),
            })
        units.write_csv(out_dir / "dm_reuse_by_library.csv", cl)

        # reused stems the libraries do not supply
        have = set(by_stem)
        unmatched = []
        for k, ev in sorted(ru.items(), key=lambda kv: -kv[1]["n_pairs"]):
            if k in have:
                continue
            unmatched.append({
                "pattern": ev["pattern"],
                "n_pairs": ev["n_pairs"],
                "n_ontologies": ev["n_ontologies"],
                "datasets": "|".join(f"{a}:{b}" for a, b in sorted(ev["by_dataset"].items())),
            })
        units.write_csv(out_dir / "dm_reuse_unmatched.csv", unmatched)

        out["reuse_crosstab"] = ct
        out["reuse_by_library"] = cl
        out["unmatched"] = unmatched
        # Coverage accounting: every pair in pairs.csv is either represented in
        # the cross-tabulation, or dropped because no library file has that stem,
        # or dropped because the representative for that stem is not in the unit.
        n_pairs_ct = sum(r["n_reuse_pairs"] for r in ct)
        n_pairs_unmatched = sum(u["n_pairs"] for u in unmatched)
        out["reuse_join"] = {
            "unit": reuse_unit,
            "n_reused_stems": len(ru),
            "n_reused_stems_matched": len(ru) - len(unmatched),
            "n_reused_stems_unmatched": len(unmatched),
            "n_pairs_total": len(pairs),
            "n_pairs_in_crosstab": n_pairs_ct,
            "pct_pairs_in_crosstab": units.pct(n_pairs_ct, len(pairs)),
            "n_pairs_dropped_no_library_file": n_pairs_unmatched,
            "n_pairs_dropped_by_unit": len(pairs) - n_pairs_ct - n_pairs_unmatched,
            "n_ambiguous_stems": len(amb),
            "ambiguous_stems": amb,
        }
        units.write_csv(out_dir / "dm_reuse_coverage.csv",
                        [{k: (v if not isinstance(v, dict) else
                              "|".join(f"{a}={'+'.join(b)}" for a, b in sorted(v.items())))
                          for k, v in out["reuse_join"].items()}])
    return out
