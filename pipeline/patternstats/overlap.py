"""Analysis 2 -- is the same pattern in more than one library?

Four questions, four independent measurements.

(a) EXACT DUPLICATES BY CONTENT
    Group every parsed file by ``sha256``.  A group spanning two libraries is a
    byte-identical duplicate across libraries; a group inside one library is an
    internal copy (ODPA ships ``DUL.owl`` next to every pattern that imports it).

(b) SAME STEM, DIFFERENT CONTENT
    Group by case-folded stem.  A stem with more than one sha256 is a *variant*.
    This is the case that matters for the benchmark, because ``pairs.csv``
    identifies a pattern by stem: if a stem has three contents, "reuse of that
    stem" is three different claims.

(c) ENTITY-SET JACCARD, CROSS-LIBRARY
    ``J(A,B) = |E_A n E_B| / |E_A u E_B|`` over the wide entity IRI sets from
    ``odpqual.entities.collect_entities``.  Computed exactly, but only for pairs
    that share at least one entity IRI -- found through an inverted index, so
    the |A|x|B| pairs that share nothing are never enumerated.  Pairs with
    ``J = 0`` are counted, not listed.

    Note what Jaccard on IRIs can and cannot see: it detects *literal reuse of
    the same IRIs*, not structural similarity.  Two libraries that model the same
    thing in their own namespaces score 0.  So (c) is a lower bound on
    conceptual overlap and an exact measure of IRI-level overlap.

(d) CROSS-LIBRARY IRI REFERENCES
    Does a pattern in library X mention an IRI that a pattern in library Y
    *declares in its own namespace*?  Uses ``own_declared`` (own-namespace
    declarations only), so a shared import of DUL does not register as X
    referencing Y.

Outputs
-------
ov_hash_groups.csv        every sha256 group with >1 file, with its libraries
ov_stem_groups.csv        every stem present >1 time, n distinct contents
ov_cross_library.csv      cross-library stem collisions, and whether content matches
ov_jaccard_pairs.csv      cross-library pairs with J > 0, descending
ov_jaccard_summary.csv    distribution of J per library pair
ov_cross_refs.csv         library X -> library Y IRI references, with examples
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from . import units
from .extract import entity_set
from .libraries import LIBRARIES, LIB_ORDER

#: Jaccard histogram edges (left-closed, right-open; the last is [0.9, 1.0])
J_BINS: tuple[float, ...] = (0.0, 1e-12, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

#: at most this many cross-library J>0 pairs written out, highest J first
MAX_JACCARD_ROWS = 3000


def _bin_label(j: float) -> str:
    if j <= 0:
        return "0"
    for lo, hi in zip(J_BINS[1:-1], J_BINS[2:]):
        if lo <= j < hi:
            return f"[{lo:g},{hi:g})"
    return "[0.9,1.0]"


# --------------------------------------------------------------------------

def run(records: list[dict], cache_dir: Path, out_dir: Path,
        unit: str = units.DEFAULT_UNIT,
        jaccard_unit: str = "content") -> dict:
    parsed = units.parsed_only(records)

    # ---------------- (a) exact duplicates by content hash ------------
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in parsed:
        by_hash[r["sha256"]].append(r)

    hash_rows = []
    for h, group in sorted(by_hash.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(group) < 2:
            continue
        libs = Counter(g["library"] for g in group)
        hash_rows.append({
            "sha256": h,
            "n_files": len(group),
            "n_libraries": len(libs),
            "cross_library": int(len(libs) > 1),
            "libraries": "|".join(f"{k}:{v}" for k, v in sorted(libs.items())),
            "stems": "|".join(sorted({g["stem"] for g in group})),
            "n_bytes": group[0]["n_bytes"],
            "example_paths": "|".join(sorted(g["key"] for g in group)[:6]),
        })
    units.write_csv(out_dir / "ov_hash_groups.csv", hash_rows)

    # ---------------- (b) same stem, different content ----------------
    by_stem: dict[str, list[dict]] = defaultdict(list)
    for r in parsed:
        by_stem[r["stem"].casefold()].append(r)

    stem_rows = []
    for s, group in sorted(by_stem.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        hashes = {g["sha256"] for g in group}
        libs = Counter(g["library"] for g in group)
        if len(group) < 2:
            continue
        stem_rows.append({
            "stem": group[0]["stem"],
            "n_files": len(group),
            "n_distinct_contents": len(hashes),
            "n_libraries": len(libs),
            "cross_library": int(len(libs) > 1),
            "libraries": "|".join(f"{k}:{v}" for k, v in sorted(libs.items())),
            "same_stem_different_content": int(len(hashes) > 1),
            "byte_sizes": "|".join(str(x) for x in sorted({g["n_bytes"] for g in group})),
            "example_paths": "|".join(sorted(g["key"] for g in group)[:6]),
        })
    units.write_csv(out_dir / "ov_stem_groups.csv", stem_rows)

    # ---------------- cross-library stem collisions -------------------
    cross_rows = []
    for s, group in sorted(by_stem.items()):
        libs = {g["library"] for g in group}
        if len(libs) < 2:
            continue
        hashes = {g["sha256"] for g in group}
        per_lib_hash = {lib: sorted({g["sha256"] for g in group if g["library"] == lib})
                        for lib in sorted(libs)}
        shared = set.intersection(*(set(v) for v in per_lib_hash.values()))
        cross_rows.append({
            "stem": group[0]["stem"],
            "libraries": "|".join(sorted(libs)),
            "n_files": len(group),
            "n_distinct_contents": len(hashes),
            "identical_across_libraries": int(bool(shared)),
            "paths": "|".join(sorted(g["key"] for g in group)),
        })
    units.write_csv(out_dir / "ov_cross_library.csv", cross_rows)

    # ---------------- (c) entity-set Jaccard --------------------------
    jrecs = units.select(records, jaccard_unit)
    jrecs = [r for r in jrecs if r.get("n_entities_wide", 0) > 0]

    # Intern IRIs to ints, then build an inverted index term -> pattern indices.
    # Cross-library candidate pairs are generated only from terms whose holders
    # span more than one library.  Without that guard the pass is quadratic in
    # the 477 DOS-DP patterns for every one of the ~10^5 OBO IRIs they share.
    iri_id: dict[str, int] = {}
    sets: list[frozenset[int]] = []
    meta: list[dict] = []
    inverted: dict[int, list[int]] = defaultdict(list)
    for r in jrecs:
        ents = entity_set(cache_dir, r)
        if not ents:
            continue
        ids = set()
        for e in ents:
            i = iri_id.get(e)
            if i is None:
                i = len(iri_id)
                iri_id[e] = i
            ids.add(i)
        idx = len(sets)
        sets.append(frozenset(ids))
        meta.append(r)
        for i in ids:
            inverted[i].append(idx)

    lib_of = [m["library"] for m in meta]
    inter: dict[tuple[int, int], int] = Counter()
    n_terms_cross_library = 0
    for holders in inverted.values():
        if len(holders) < 2:
            continue
        libs = {lib_of[h] for h in holders}
        if len(libs) < 2:
            continue
        n_terms_cross_library += 1
        for a in range(len(holders)):
            x = holders[a]
            lx = lib_of[x]
            for b in range(a + 1, len(holders)):
                y = holders[b]
                if lib_of[y] != lx:
                    inter[(x, y)] += 1

    id_iri = {v: k for k, v in iri_id.items()}

    jac_rows = []
    hist: dict[tuple[str, str], Counter] = defaultdict(Counter)
    sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (x, y), n_shared in inter.items():
        a, b = sets[x], sets[y]
        union = len(a) + len(b) - n_shared
        j = n_shared / union if union else 0.0
        shared_iris = sorted(id_iri[i] for i in (a & b))[:5]
        la, lb = sorted((meta[x]["library"], meta[y]["library"]))
        hist[(la, lb)][_bin_label(j)] += 1
        sums[(la, lb)].append(j)
        jac_rows.append({
            "library_a": meta[x]["library"], "key_a": meta[x]["key"],
            "stem_a": meta[x]["stem"], "n_entities_a": len(a),
            "library_b": meta[y]["library"], "key_b": meta[y]["key"],
            "stem_b": meta[y]["stem"], "n_entities_b": len(b),
            "n_shared_entities": n_shared,
            "n_union_entities": union,
            "jaccard": round(j, 6),
            "containment_a_in_b": round(n_shared / len(a), 6) if a else 0.0,
            "containment_b_in_a": round(n_shared / len(b), 6) if b else 0.0,
            "example_shared_iris": "|".join(shared_iris),
        })
    # full sort key, so repeated runs produce byte-identical files
    jac_rows.sort(key=lambda r: (-r["jaccard"], -r["n_shared_entities"],
                                 r["key_a"], r["key_b"]))
    units.write_csv(out_dir / "ov_jaccard_pairs.csv", jac_rows[:MAX_JACCARD_ROWS])

    # summary per library pair, including the J=0 pairs never enumerated
    n_by_lib = Counter(r["library"] for r in meta)
    jsum = []
    for la in LIB_ORDER:
        for lb in LIB_ORDER:
            if la >= lb:
                continue
            total_pairs = n_by_lib[la] * n_by_lib[lb]
            if not total_pairs:
                continue
            key = tuple(sorted((la, lb)))
            vals = sorted(sums.get(key, []), reverse=True)
            nz = len(vals)
            row = {
                "library_a": la, "library_b": lb,
                "n_patterns_a": n_by_lib[la], "n_patterns_b": n_by_lib[lb],
                "n_pairs_total": total_pairs,
                "n_pairs_jaccard_gt0": nz,
                "pct_pairs_jaccard_gt0": units.pct(nz, total_pairs, 3),
                "n_pairs_jaccard_0": total_pairs - nz,
                "max_jaccard": round(vals[0], 6) if vals else 0.0,
                "mean_jaccard_over_all_pairs": round(sum(vals) / total_pairs, 8) if total_pairs else 0.0,
                "mean_jaccard_over_nonzero": round(sum(vals) / nz, 6) if nz else 0.0,
                "median_jaccard_over_nonzero": round(vals[nz // 2], 6) if nz else 0.0,
            }
            for lo, hi in zip(J_BINS[1:-1], J_BINS[2:]):
                row[f"bin_[{lo:g},{hi:g})"] = hist[key].get(f"[{lo:g},{hi:g})", 0)
            row["bin_[0.9,1.0]"] = hist[key].get("[0.9,1.0]", 0)
            jsum.append(row)
    units.write_csv(out_dir / "ov_jaccard_summary.csv", jsum)

    # ---------------- (d) cross-library IRI references ----------------
    # owner index: own-namespace declared IRI -> library.  A shared *import*
    # (both libraries pointing at DUL) is not a cross-library reference, which is
    # why only own-namespace declarations define ownership.
    owner_lib: dict[int, str] = {}
    for r in parsed:
        for iri in r.get("own_declared", []):
            i = iri_id.get(iri)
            if i is not None:
                owner_lib.setdefault(i, r["library"])
    owner_ids = set(owner_lib)

    # Ownership is per *file*, so a reference is only counted when the owning
    # file is a different file; the within-library rows are reported alongside
    # the cross-library ones because in IS-ODP the sibling-pattern references are
    # the interesting structure and would otherwise be invisible.
    owner_key: dict[int, str] = {}
    for r in parsed:
        for iri in r.get("own_declared", []):
            i = iri_id.get(iri)
            if i is not None:
                owner_key.setdefault(i, r["key"])

    ref_counts: dict[tuple[str, str], set[int]] = defaultdict(set)
    ref_patterns: dict[tuple[str, str], set[str]] = defaultdict(set)
    for idx, r in enumerate(meta):
        me = r["library"]
        for i in sets[idx] & owner_ids:
            if owner_key.get(i) == r["key"]:
                continue                      # its own declaration
            dst = owner_lib[i]
            k = (me, dst)
            ref_counts[k].add(i)
            ref_patterns[k].add(r["key"])
    # examples chosen after the fact, sorted, so reruns are byte-identical
    ref_examples = {k: sorted(id_iri[i] for i in v)[:8]
                    for k, v in ref_counts.items()}

    ref_rows = []
    for (src, dst) in sorted(ref_counts, key=lambda t: (t[0] != t[1], t[0], t[1])):
        ref_rows.append({
            "referencing_library": src,
            "referencing_library_name": LIBRARIES[src][1],
            "referenced_library": dst,
            "referenced_library_name": LIBRARIES[dst][1],
            "same_library": int(src == dst),
            "n_referencing_patterns": len(ref_patterns[(src, dst)]),
            "n_distinct_referenced_iris": len(ref_counts[(src, dst)]),
            "example_iris": "|".join(ref_examples[(src, dst)]),
        })
    units.write_csv(out_dir / "ov_cross_refs.csv", ref_rows)

    return {
        "hash_groups": hash_rows,
        "stem_groups": stem_rows,
        "cross_library_stems": cross_rows,
        "jaccard_pairs": jac_rows,
        "jaccard_summary": jsum,
        "cross_refs": ref_rows,
        "jaccard_unit": jaccard_unit,
        "n_jaccard_patterns": len(meta),
        "n_distinct_entity_iris": len(iri_id),
        "n_terms_spanning_libraries": n_terms_cross_library,
        "unit": unit,
    }
