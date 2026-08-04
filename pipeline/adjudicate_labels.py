#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjudicate_labels.py -- decide which label-only matches are genuine ODP reuse,
using several LLMs as annotators anchored to a human sample.

THE PROBLEM
-----------
The `label` ground-truth mode matches a pattern class to a host entity when their
normalised labels are equal. That recovers reuse where an ontology re-modelled a
pattern under its own namespace -- invisible to IRI matching -- but it also
admits homonyms: a class labelled `Event` in a concert ontology is not thereby
reusing DUL's `Event`. No automatic check settles this; it needs judgement.

WHY NOT LLMs ALONE
------------------
Agreement between GPT, Gemini and Claude measures whether the models are
CONSISTENT, not whether they are CORRECT. They share training data and failure
modes and can agree while being wrong. An inter-LLM kappa is therefore not
evidence of validity, and offering one in place of human validation invites the
obvious objection.

THE DESIGN
----------
LLMs supply scale; a human supplies the anchor.

  1. Each model independently adjudicates EVERY label-only match.
  2. The majority vote is the machine verdict; ties are `unclear`.
  3. A human adjudicates a sample STRATIFIED ON MODEL DISAGREEMENT -- unanimous
     cases are cheap to spot-check, split cases are where judgement matters.
  4. Three statistics are reported, and they answer different questions:
       Fleiss' kappa across models      -> machine consistency
       Cohen's kappa, human vs majority -> VALIDITY ANCHOR
       precision on the human sample    -> the number the paper cites,
                                           extrapolated only if the anchor holds

If human-vs-majority agreement is weak, the honest conclusion is that the models
cannot do this task, and the human sample stands alone. That is a result too.

Providers are optional and independent: run with whichever keys are present.
Nothing here requires a key -- `--dry-run` prints the prompts and exits, and
`--human-only` emits the sample sheet for manual adjudication with no API calls.

Usage
-----
    export OPENAI_API_KEY=... GOOGLE_API_KEY=... ANTHROPIC_API_KEY=...

    # 1. build the item set from the corpus
    python adjudicate_labels.py build --manifest out/MANIFEST.csv \\
        --data-root <corpus> --out out/adjudication

    # 2. machine adjudication (skips providers whose key is absent)
    python adjudicate_labels.py judge --out out/adjudication

    # 3. the human sheet, stratified on disagreement
    python adjudicate_labels.py sample --out out/adjudication --n 50

    # 4. after filling in human_verdict, compute the statistics
    python adjudicate_labels.py score --out out/adjudication
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:                                   # load API keys without ever printing them
    import _load_creds
    _load_creds.apply(verbose=False)
except Exception:                      # noqa: BLE001 - keys may come from the env
    pass

VERDICTS = ("reuse", "not_reuse", "unclear")

SYSTEM = (
    "You are an expert ontology engineer adjudicating whether an ontology reuses "
    "a specific Ontology Design Pattern (ODP). Answer strictly with JSON."
)

PROMPT = """An automated matcher proposed that this ontology reuses an ODP, because
some of the ontology's entities carry the SAME NORMALISED LABEL as classes in the
pattern. The IRIs do NOT match -- the ontology would have re-modelled the pattern
under its own namespace.

Your task: decide whether this is genuine reuse of the pattern's modelling
solution, or a coincidental name collision.

PATTERN: {pattern}
Pattern classes: {pattern_classes}

ONTOLOGY: {ontology}
Ontology subject matter, inferred from its entity names: {ontology_sample}

MATCHED LABELS (present in both): {matched}
Matched entities in the ontology: {matched_iris}

Guidance:
- Genuine reuse: the ontology models the same conceptual structure the pattern
  describes, and the matched classes play the same role. Several structurally
  connected matches are strong evidence.
- Coincidence: the label is a common word (Event, Object, Role, Place) used in an
  unrelated sense, or a single isolated match with no structural echo.
- Answer "unclear" only when the evidence genuinely does not decide it.

Respond with JSON only:
{{"verdict": "reuse" | "not_reuse" | "unclear", "confidence": 0.0-1.0,
  "reason": "<one sentence>"}}"""


# ----------------------------------------------------------------------
# providers -- each returns (verdict, confidence, reason) or None if unavailable
# ----------------------------------------------------------------------

def _parse(text: str) -> tuple[str, float, str] | None:
    try:
        s = text[text.index("{"):text.rindex("}") + 1]
        d = json.loads(s)
        v = str(d.get("verdict", "")).strip().lower()
        if v not in VERDICTS:
            return None
        return v, float(d.get("confidence", 0.0)), str(d.get("reason", ""))[:300]
    except Exception:  # noqa: BLE001
        return None


def ask_openai(prompt: str, model: str = "gpt-4o") -> tuple | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        r = OpenAI(api_key=key).chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}])
        return _parse(r.choices[0].message.content or "")
    except Exception as e:  # noqa: BLE001
        print(f"    [openai] {type(e).__name__}: {e}", file=sys.stderr)
        return None


def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> tuple | None:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        # google.generativeai is deprecated upstream; prefer google.genai and
        # fall back so this keeps working on either SDK.
        try:
            from google import genai as _g
            from google.genai import types as _t
            global _GENAI_CLIENT
            if _GENAI_CLIENT is None:
                _GENAI_CLIENT = _g.Client(api_key=key)
            c = _GENAI_CLIENT
            r = c.models.generate_content(
                model=model, contents=prompt,
                config=_t.GenerateContentConfig(system_instruction=SYSTEM,
                                                temperature=0))
            return _parse(r.text or "")
        except ImportError:
            import google.generativeai as genai
            genai.configure(api_key=key)
            r = genai.GenerativeModel(
                model, system_instruction=SYSTEM).generate_content(
                    prompt, generation_config={"temperature": 0})
            return _parse(r.text or "")
    except Exception as e:  # noqa: BLE001
        print(f"    [gemini] {type(e).__name__}: {e}", file=sys.stderr)
        return None


def ask_claude(prompt: str, model: str = "claude-sonnet-5") -> tuple | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        # `temperature` is rejected by the newer models, so it is simply not
        # sent; these models are already low-variance for a classification like
        # this one.
        r = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=400, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}])
        return _parse("".join(b.text for b in r.content if b.type == "text"))
    except Exception as e:  # noqa: BLE001
        print(f"    [claude] {type(e).__name__}: {e}", file=sys.stderr)
        return None


_GENAI_CLIENT = None

def ask_deepseek(prompt: str, model: str = "deepseek-chat") -> tuple | None:
    """DeepSeek exposes an OpenAI-compatible endpoint, so the same client works."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        r = OpenAI(api_key=key,
                   base_url="https://api.deepseek.com").chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}])
        return _parse(r.choices[0].message.content or "")
    except Exception as e:  # noqa: BLE001
        print(f"    [deepseek] {type(e).__name__}: {e}", file=sys.stderr)
        return None


PROVIDERS = {"gpt": ask_openai, "gemini": ask_gemini, "claude": ask_claude,
             "deepseek": ask_deepseek}


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------

def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Two raters, categorical."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Several raters per item; ratings[i] = verdicts for item i."""
    items = [r for r in ratings if len(r) >= 2]
    if not items:
        return float("nan")
    n = len(items[0])
    if any(len(r) != n for r in items):          # ragged -> pairwise mean instead
        return float("nan")
    cats = sorted({v for r in items for v in r})
    N = len(items)
    P = []
    for r in items:
        c = Counter(r)
        P.append((sum(v * v for v in c.values()) - n) / (n * (n - 1)))
    pbar = sum(P) / N
    pj = [sum(Counter(r)[c] for r in items) / (N * n) for c in cats]
    pe = sum(p * p for p in pj)
    return 1.0 if pe == 1 else (pbar - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate and Wilson interval for a proportion."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def interpret(k: float) -> str:
    if k != k:
        return "not computable"
    for t, s in ((.81, "almost perfect"), (.61, "substantial"), (.41, "moderate"),
                 (.21, "fair"), (0, "slight")):
        if k >= t:
            return s
    return "poor (worse than chance)"


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------

def cmd_build(args) -> None:
    """Collect every label-only match as an adjudicable item."""
    import logging
    logging.getLogger("rdflib").setLevel(logging.ERROR)
    from build_manifest import DATASETS
    from odpqual.entities import collect_entities, parse_graph, strip_annotations
    from odpqual.gate import main_ontology_file, pattern_files
    from odpqual.gtmodes import gt_by_iri, gt_by_label, local_name

    rows = [r for r in csv.DictReader(args.manifest.open(encoding="utf-8"))
            if not r["drop_reason"].startswith("G1_parse")
            and r["dataset"] in args.datasets]
    items = []
    skipped: list[dict] = []
    for i, r in enumerate(rows, 1):
        d = args.data_root / DATASETS[r["dataset"]] / r["ontology_id"]
        main = main_ontology_file(d)
        if main is None:
            continue
        # Size is checked BEFORE parsing, not after. An entity-count cap applied
        # post-parse is useless: parsing the 1 GB cco.owl is itself the hours-long
        # step, so the ontology must be excluded on a cheap signal first.
        if args.max_mb and main.stat().st_size > args.max_mb * 1024 * 1024:
            skipped.append({"ontology": r["ontology_id"], "dataset": r["dataset"],
                            "reason": "file_too_large",
                            "detail": f"{main.stat().st_size / 1e6:.0f}MB"
                                      f">{args.max_mb}MB"})
            continue
        g, _ = parse_graph(main)
        if g is None:
            continue
        g, _ = strip_annotations(g)
        ents = set(collect_entities(g, wide=True))
        if not ents:
            continue
        # Adjudication is a per-item human/LLM judgement, so an ontology with
        # 10^5 entities cannot be judged from a prompt anyway -- and parsing it
        # dominates the scan (cco alone is 1 GB / 264,906 entities). Skipped
        # ontologies are recorded, never silently dropped.
        if args.max_entities and len(ents) > args.max_entities:
            skipped.append({"ontology": r["ontology_id"], "dataset": r["dataset"],
                            "reason": "too_many_entities",
                            "detail": f"{len(ents)}>{args.max_entities}"})
            continue
        for pf in pattern_files(d):
            pg, _ = parse_graph(pf)
            if pg is None:
                continue
            pgs, _ = strip_annotations(pg)
            pents = set(collect_entities(pgs, wide=True))
            if not pents:
                continue
            by_iri = gt_by_iri(ents, pents)
            by_lab = gt_by_label(g, ents, pg)
            only = by_lab - by_iri
            if not only:
                continue                     # nothing label-specific to judge
            items.append({
                "id": f"{r['dataset']}/{r['ontology_id']}/{pf.stem}",
                "dataset": r["dataset"], "ontology": r["ontology_id"],
                "pattern": pf.stem,
                "n_iri": len(by_iri), "n_label_only": len(only),
                "matched": ";".join(sorted(local_name(e) for e in only)[:12]),
                "matched_iris": ";".join(sorted(only)[:6]),
                "pattern_classes": ";".join(
                    sorted({local_name(str(c)) for c in pents})[:14]),
                "ontology_sample": ";".join(
                    sorted({local_name(e) for e in ents})[:18]),
            })
        if i % 100 == 0:
            print(f"  scanned {i}/{len(rows)}, {len(items)} items", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / "items.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(items[0]) if items else ["id"])
        w.writeheader()
        w.writerows(items)
    print(f"\n[OK] {p} -- {len(items)} label-only matches to adjudicate")

    # The skip log. This used to be collected and then thrown away: --help
    # promised `build_skipped.csv` and nothing wrote it, so an exclusion this
    # pipeline claims to record was in fact silent. Written unconditionally --
    # a zero-row file with a header is the honest way to say "nothing skipped",
    # and is distinguishable from a run that never wrote the file at all.
    s = args.out / "build_skipped.csv"
    fields = ["ontology", "dataset", "reason", "detail"]
    with s.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(skipped)
    by_reason = Counter(x["reason"] for x in skipped)
    print(f"[OK] {s} -- {len(skipped)} skipped"
          + (f" ({dict(by_reason)})" if skipped else ""))


def cmd_judge(args) -> None:
    items = list(csv.DictReader((args.out / "items.csv").open(encoding="utf-8")))
    # probe by key presence only -- never by issuing a throwaway API call
    usable = {}
    for k in args.providers:
        fn = PROVIDERS[k]
        env = {"gpt": "OPENAI_API_KEY", "gemini": "GOOGLE_API_KEY",
               "claude": "ANTHROPIC_API_KEY",
               "deepseek": "DEEPSEEK_API_KEY"}[k]
        if os.environ.get(env) or (k == "gemini" and os.environ.get("GEMINI_API_KEY")):
            usable[k] = fn
        else:
            print(f"[skip] {k}: {env} not set")
    if not usable:
        sys.exit("no provider available -- set at least one API key, or use "
                 "`sample --human-only`")
    print(f"adjudicating {len(items)} items with {', '.join(usable)}")

    tag = "" if set(args.providers) == set(PROVIDERS) else         "_" + "_".join(sorted(args.providers))
    out = args.out / f"machine_verdicts{tag}.csv"
    done = set()
    if args.resume and out.is_file():
        done = {r["id"] for r in csv.DictReader(out.open(encoding="utf-8"))}
        print(f"resuming: {len(done)} already judged")

    cols = ["id"] + [f"{k}_{s}" for k in usable for s in
                     ("verdict", "confidence", "reason")] + ["majority", "n_agree"]
    mode = "a" if (args.resume and out.is_file()) else "w"
    with out.open(mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if mode == "w":
            w.writeheader()
        for i, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            prompt = PROMPT.format(**it)
            if args.dry_run:
                print(prompt[:1400]); return
            row = {"id": it["id"]}
            votes = []
            for k, fn in usable.items():
                r = fn(prompt)
                if r:
                    row[f"{k}_verdict"], row[f"{k}_confidence"], row[f"{k}_reason"] = r
                    votes.append(r[0])
            if votes:
                c = Counter(votes).most_common()
                row["majority"] = c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) \
                    else "unclear"
                row["n_agree"] = c[0][1]
            w.writerow(row); fh.flush()
            if i % 5 == 0 or i == len(items):
                print(f"  {i}/{len(items)}", flush=True)
    print(f"[OK] {out}")


def cmd_sample(args) -> None:
    """Human sheet, stratified on model disagreement."""
    items = {r["id"]: r for r in
             csv.DictReader((args.out / "items.csv").open(encoding="utf-8"))}
    mv = args.out / "machine_verdicts.csv"
    machine = {r["id"]: r for r in csv.DictReader(mv.open(encoding="utf-8"))} \
        if mv.is_file() else {}

    split, unanimous = [], []
    for i, it in items.items():
        m = machine.get(i)
        if m and m.get("n_agree"):
            vs = [m.get(f"{k}_verdict") for k in PROVIDERS if m.get(f"{k}_verdict")]
            (unanimous if len(set(vs)) == 1 else split).append(i)
        else:
            split.append(i)                  # unjudged counts as informative

    rng = random.Random(args.seed)
    rng.shuffle(split); rng.shuffle(unanimous)
    if args.n <= 0:
        # Full coverage. At this population size (~300) adjudicating everything
        # costs little more than a sample and removes the need to extrapolate:
        # precision becomes a count, not an estimate, and the label mode is
        # human-verified end to end rather than "verified on a sample of 50".
        chosen = split + unanimous
        n_split = len(split)
    else:
        # Disagreements carry the information; unanimous cases are the spot-check.
        n_split = min(len(split), int(args.n * 0.6))
        chosen = split[:n_split] + unanimous[:args.n - n_split]
    # Contested items first, so partial adjudication still yields the informative
    # verdicts if the sheet is not finished in one sitting.
    rng.shuffle(chosen)
    contested = set(split)
    chosen.sort(key=lambda i: i not in contested)

    p = args.out / "human_sample.csv"
    cols = ["id", "dataset", "ontology", "pattern", "n_iri", "n_label_only",
            "matched", "pattern_classes", "ontology_sample",
            "human_verdict", "human_note"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i in chosen:
            r = dict(items[i]); r["human_verdict"] = ""; r["human_note"] = ""
            w.writerow(r)
    print(f"[OK] {p} -- {len(chosen)} items "
          f"({n_split} where models disagree, {len(chosen) - n_split} unanimous)")
    if args.n <= 0:
        print("Full coverage: no extrapolation needed, precision is a count.")
    print("Contested items are listed FIRST, so a partial pass still captures "
          "the informative ones.")
    print("Fill in human_verdict with: reuse | not_reuse | unclear")


def cmd_score(args) -> None:
    mv = args.out / "machine_verdicts.csv"
    hs = args.out / "human_sample.csv"
    machine = {r["id"]: r for r in csv.DictReader(mv.open(encoding="utf-8"))} \
        if mv.is_file() else {}
    human = {r["id"]: r for r in csv.DictReader(hs.open(encoding="utf-8"))
             if r.get("human_verdict", "").strip()} if hs.is_file() else {}

    lines = ["# Label-mode adjudication", ""]

    if machine:
        provs = [k for k in PROVIDERS
                 if any(r.get(f"{k}_verdict") for r in machine.values())]
        lines += [f"Machine annotators: {', '.join(provs) or 'none'}  ",
                  f"Items adjudicated: {len(machine)}", ""]
        ratings = [[r[f"{k}_verdict"] for k in provs if r.get(f"{k}_verdict")]
                   for r in machine.values()]
        # Fleiss' kappa requires the same number of raters per item; restrict to
        # items every rater judged and say so, rather than returning nan.
        complete = [x for x in ratings if len(x) == len(provs)]
        fk = fleiss_kappa(complete)
        lines += [f"**Inter-model agreement (Fleiss' κ): {fk:.3f}** "
                  f"— {interpret(fk)} "
                  f"(complete cases only: {len(complete)} of {len(ratings)} "
                  f"items judged by all {len(provs)} raters)", "",
                  "> κ alone can mislead when one category dominates: with a "
                  "base rate near 90%, chance agreement is already ~0.90 and the "
                  "correction drives κ towards zero even when raters concur. "
                  "Report raw agreement and the marginals alongside it.", "",
                  "> Measures whether the models are *consistent*, not whether "
                  "they are *correct*. Models share training data and failure "
                  "modes; this number is not evidence of validity on its own.", ""]
        dist = Counter(r.get("majority", "") for r in machine.values())
        lines += ["| Machine majority | n | share |", "|---|---:|---:|"]
        for v, n in dist.most_common():
            lines.append(f"| {v or '(none)'} | {n} | "
                         f"{100 * n / len(machine):.1f}% |")
        lines.append("")

    # The adjudication is STRATIFIED: contested and incomplete items are judged
    # in full, the unanimous stratum is validated on a sample and the remainder
    # accepted if that sample holds. Precision over the adjudicated subset alone
    # would therefore overstate the reuse rate, because the contested stratum is
    # deliberately enriched for hard (and more often positive) cases.
    plan_path = args.out / "adjudication_workbook.xlsx"
    strata = {}
    if plan_path.is_file():
        try:
            from openpyxl import load_workbook
            wb = load_workbook(plan_path, data_only=True)
            ws = wb["Adjudicate"]
            hdr = [c.value for c in ws[1]]
            iv, it_, ii = (hdr.index("human_verdict"), hdr.index("todo"),
                           hdr.index("id"))
            for row in ws.iter_rows(min_row=2, values_only=True):
                strata[row[ii]] = (row[it_], (row[iv] or "").strip().lower())
            if "Deferred" in wb.sheetnames:
                d = wb["Deferred"]
                dh = [c.value for c in d[5]]
                dv_, di = dh.index("human_verdict"), dh.index("id")
                for row in d.iter_rows(min_row=6, values_only=True):
                    if row[di]:
                        strata[row[di]] = ("deferred",
                                           (row[dv_] or "").strip().lower())
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read the stratification plan: {exc}")

    if strata:
        n_judge = sum(1 for t_, _v in strata.values() if t_ == "judge")
        spot = [(v) for t_, v in strata.values() if t_ == "spot-check"]
        defer = [(v) for t_, v in strata.values() if t_ == "deferred"]
        spot_reuse = sum(1 for v in spot if v == "reuse")
        judge_reuse = sum(1 for t_, v in strata.values()
                          if t_ == "judge" and v == "reuse")
        total = len(strata)
        _p, _lo, hi_spot = wilson(spot_reuse, len(spot)) if spot else (0, 0, 0)
        # Deferred items are accepted as not_reuse when the spot-check finds no
        # reuse; the Wilson upper bound quantifies what that could have missed.
        est_reuse = judge_reuse + spot_reuse
        lines += ["## Stratified precision over the FULL item set", "",
                  f"| Stratum | n | adjudicated | reuse |",
                  "|---|---:|---|---:|",
                  f"| judge (contested + incomplete) | {n_judge} | all | "
                  f"{judge_reuse} |",
                  f"| spot-check (sample of unanimous) | {len(spot)} | all | "
                  f"{spot_reuse} |",
                  f"| deferred (unanimous, accepted) | {len(defer)} | none | "
                  f"assumed 0 |", ""]
        pp, plo, phi = wilson(est_reuse, total)
        lines += [f"**Label-mode precision = {est_reuse}/{total} = {pp:.3f}** "
                  f"(95% Wilson CI [{plo:.3f}, {phi:.3f}])", ""]
        if spot:
            worst = est_reuse + hi_spot * len(defer)
            lines += [f"Sensitivity: the spot-check found {spot_reuse} reuse in "
                      f"{len(spot)} unanimous items, bounding that stratum's "
                      f"reuse rate at {hi_spot:.3f}. Even at that upper bound the "
                      f"deferred stratum would add at most "
                      f"{hi_spot * len(defer):.0f} items, giving a worst-case "
                      f"precision of {worst / total:.3f}.", ""]
        lines += ["> Precision computed over the adjudicated subset alone would "
                  "**overstate** reuse, because the contested stratum is "
                  "deliberately enriched for hard cases. The figure above is the "
                  "one to report.", ""]

    if human:
        ids = sorted(human)
        hv = [human[i]["human_verdict"].strip().lower() for i in ids]
        k_ok = sum(v == "reuse" for v in hv)
        n_dec = sum(v in ("reuse", "not_reuse") for v in hv)
        k_dec = sum(v == "reuse" for v in hv if v in ("reuse", "not_reuse"))
        p, lo, hi = wilson(k_dec, n_dec)
        lines += ["## Human adjudication — adjudicated subset only", "",
                  f"Adjudicated: {len(hv)} · reuse {k_ok} · "
                  f"not_reuse {sum(v == 'not_reuse' for v in hv)} · "
                  f"unclear {sum(v == 'unclear' for v in hv)}", "",
                  f"Precision *within the adjudicated subset*: {p:.3f} "
                  f"(95% Wilson CI [{lo:.3f}, {hi:.3f}], n={n_dec}). This is "
                  f"NOT the label-mode precision — see the stratified table "
                  f"above.", ""]

        paired = [(i, human[i]["human_verdict"].strip().lower())
                  for i in ids if i in machine and machine[i].get("majority")]
        if paired:
            h = [v for _i, v in paired]
            m = [machine[i]["majority"] for i, _v in paired]
            ck = cohen_kappa(h, m)
            agree = sum(x == y for x, y in zip(h, m)) / len(h)
            lines += ["## Validity anchor: human vs machine majority", "",
                      f"Paired items: {len(paired)} · raw agreement "
                      f"{100 * agree:.1f}%  ",
                      f"**Cohen's κ = {ck:.3f}** — {interpret(ck)}", ""]
            if ck >= .61:
                lines.append("κ is substantial or better, so the machine verdict "
                             "may be reported over the **full** item set, "
                             "explicitly conditioned on this anchor.")
            else:
                lines.append("κ is below the substantial threshold. The machine "
                             "verdict must **not** be extrapolated; report the "
                             "human sample alone and state that automated "
                             "adjudication did not reach acceptable agreement.")
            lines.append("")
            for k in [x for x in PROVIDERS
                      if any(r.get(f"{x}_verdict") for r in machine.values())]:
                pr = [(human[i]["human_verdict"].strip().lower(),
                       machine[i].get(f"{k}_verdict"))
                      for i, _v in paired if machine[i].get(f"{k}_verdict")]
                if pr:
                    lines.append(f"- human vs **{k}**: κ = "
                                 f"{cohen_kappa([a for a, _ in pr], [b for _, b in pr]):.3f}")
            lines.append("")
    else:
        lines += ["## Human adjudication", "",
                  "_Not yet done._ Fill `human_verdict` in `human_sample.csv`, "
                  "then re-run `score`. Without it there is no validity anchor "
                  "and the label mode is not citable.", ""]

    p = args.out / "ADJUDICATION.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[OK] {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build"); b.set_defaults(fn=cmd_build)
    b.add_argument("--manifest", required=True, type=Path)
    b.add_argument("--data-root", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    # DOS-DP is excluded by default: its patterns are generated from YAML
    # templates and its ontologies carry ~10^5 entities, so label equality there
    # measures vocabulary overlap, not a deliberate reuse decision.
    b.add_argument("--datasets", nargs="+", default=["github", "noOBO"],
                   choices=["github", "noOBO", "OBO"])
    b.add_argument("--max-mb", type=float, default=40.0,
                   help="skip ontology FILES larger than this before parsing "
                        "(0 = no limit). Checked first because parsing is the "
                        "expensive step; skips are recorded.")
    b.add_argument("--max-entities", type=int, default=20000,
                   help="skip ontologies larger than this (0 = no limit). An "
                        "item is judged from a prompt, so a 10^5-entity "
                        "ontology cannot be adjudicated meaningfully; skips are "
                        "recorded in build_skipped.csv.")

    j = sub.add_parser("judge"); j.set_defaults(fn=cmd_judge)
    j.add_argument("--out", required=True, type=Path)
    j.add_argument("--providers", nargs="+", default=list(PROVIDERS),
                   choices=list(PROVIDERS))
    j.add_argument("--resume", action="store_true")
    j.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("sample"); s.set_defaults(fn=cmd_sample)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--n", type=int, default=0,
                   help="items to adjudicate; 0 (default) = ALL of them, which "
                        "at this population size removes the need to "
                        "extrapolate from a sample")
    s.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("score"); c.set_defaults(fn=cmd_score)
    c.add_argument("--out", required=True, type=Path)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
