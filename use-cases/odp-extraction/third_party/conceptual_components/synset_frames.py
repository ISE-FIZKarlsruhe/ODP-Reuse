# Vendored from stlab-istc-cnr/conceptual-components
# @ e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5
# paths: conceptual-components-extraction/project_root/
#        disambiguation_from_csv.py                (STEP 4)
#        filter_have_be_enrich_synsetframe.py      (STEP 5)
#        clustering_synsetframe_communities.py     (STEP 6a)
#        clustering_elbowmethod_synsetframe_communities.py (STEP 6b)
#
# Step 3 is the server these talk to: the Framester jar (Google Drive, 591 MB)
# in front of a locally built UKB word-sense disambiguator. It is NOT vendored;
# it is downloaded and built, and the runner requires a live instance.
#
# PATCHES (see PROVENANCE.md):
#  P4.1  Requests are issued with a proper URL encoder and a retry, and may be
#        issued concurrently. Upstream concatenates the raw text into the query
#        string, which breaks on any text containing '&' or '#' -- unavoidable
#        on a corpus of IRI local names. The endpoints, their parameters
#        (senseInventory=fn17, serialization=json) and the namespace shortening
#        dictionary are upstream's.
#  P5.1  Upstream mutates `sentence['multiWords']` while iterating it to drop
#        'be'/'have', which skips the element after each removal. The
#        list-comprehension here removes every 'be'/'have' multiword, which is
#        what the code says it does. This is the one place where the port is
#        deliberately not bug-compatible; the alternative is reproducing a
#        skip whose effect depends on element order.
#  P6.1  `TfidfVectorizer.get_feature_names` was removed in scikit-learn 1.2;
#        it is unused by the clustering itself and is dropped. The vectoriser
#        and KMeans parameters are upstream's, exactly:
#        max_df=0.90, max_features=200000, min_df=1, use_idf=True,
#        tokenizer=split-on-comma, ngram_range=(1,1); KMeans(random_state=42).

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# upstream's namespace shortening dictionary (disambiguation_from_csv.py)
_SHORTEN = {
    "https://w3id.org/framester/wn/wn30/instances/": "wn30:",
    "https://w3id.org/framester/framenet/abox/frame/": "frame:",
}


def replace_all(text, dic=_SHORTEN):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


# ----------------------------------------------------------------------
# STEP 4 -- disambiguation against the Framester/UKB server
# ----------------------------------------------------------------------

def _get(base: str, endpoint: str, params: dict, timeout: int = 300,
         retries: int = 2) -> str:
    url = f"{base}/{endpoint}?" + urllib.parse.urlencode(params)
    last = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{endpoint} failed: {type(last).__name__}: {last}")


def disambiguate_one(text: str, base: str = "http://localhost:8080") -> dict:
    """Upstream's per-row body of `from_communities_text_to_synsetframe`."""
    if not text.strip():
        return {"synset frame": text, "annotated synset frame": text}
    resp = _get(base, "cd", {"text": text, "senseInventory": "fn17"})
    if resp.strip() == "[]" or not resp.strip():
        # upstream: fall back to the original text in both columns
        return {"synset frame": text, "annotated synset frame": text}
    start = resp.find("[") + 1
    end = resp.find("]")
    processed = resp[start:end].replace('"', "")
    annotated = _get(base, "wfd", {"text": text, "serialization": "json"})
    return {"synset frame": replace_all(processed),
            "annotated synset frame": replace_all(annotated)}


def disambiguate_many(texts: list[str], base: str = "http://localhost:8080",
                      workers: int = 4, progress=None) -> list[dict]:
    out: list[dict | None] = [None] * len(texts)

    def job(i):
        try:
            out[i] = disambiguate_one(texts[i], base)
        except Exception as exc:  # noqa: BLE001
            out[i] = {"synset frame": texts[i], "annotated synset frame": texts[i],
                      "error": f"{type(exc).__name__}: {exc}"}
        return i

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(job, range(len(texts))):
            done += 1
            if progress and done % 100 == 0:
                progress(f"    disambiguated {done}/{len(texts)}")
    return [o for o in out if o is not None]


# ----------------------------------------------------------------------
# STEP 5 -- drop be/have, keep the max-scoring synset, attach frames and
#           their top-level ancestors
# ----------------------------------------------------------------------

def load_top_level_ancestors(csv_path) -> dict:
    """`data/topLevelAnchestors-prefix.csv`: frame -> top-level ancestor frame."""
    import csv as _csv
    out = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for row in _csv.DictReader(fh):
            f = (row.get("f") or "").strip()
            anc = (row.get("topLevelAnchestor") or "").strip()
            if f:
                out[f] = anc
    return out


def filter_and_enrich(annotated_synset_frame: str, fallback_text: str,
                      top_level: dict) -> str:
    """Upstream's `remove_have_be_synsetframe`, per row.

    Returns the `filtered synset frame` column: a comma-joined list of, for each
    multiword that is not 'be' or 'have', its highest-scoring WN30_FRAMESTER
    synset followed by each of its FN17 frames.
    """
    try:
        d = json.loads(annotated_synset_frame)
    except Exception:  # noqa: BLE001 - upstream's bare except
        return fallback_text

    filtered: list[str] = []
    try:
        for sentence in d["sentences"]:
            # P5.1: remove every be/have multiword (see header).
            mws = [m for m in sentence["multiWords"]
                   if m.get("lemma") not in ("be", "have")]
            for multiword in mws:
                scored = multiword.get("scoredAnnotations", {})
                if "WN30_FRAMESTER" not in scored:
                    continue
                cands = [s for s in scored["WN30_FRAMESTER"]
                         if s.get("annotation") != "Oregon"]
                if not cands:
                    continue
                max_score = max(s["score"] for s in cands)
                max_synset = None
                for s in scored["WN30_FRAMESTER"]:
                    if s["score"] == max_score:
                        filtered.append(s["annotation"])
                        max_synset = s["annotation"]
                if "FN17" in multiword.get("annotations", {}):
                    for frame in multiword["annotations"]["FN17"]:
                        filtered.append(frame)
                        anc = top_level.get(frame)
                        if anc:
                            filtered.append(anc)
                del max_synset
    except Exception:  # noqa: BLE001
        return fallback_text

    if not filtered:
        return fallback_text
    return ",".join(replace_all(x) for x in filtered)


# ----------------------------------------------------------------------
# STEP 6 -- cluster the community documents
# ----------------------------------------------------------------------

def my_tokenizer(text):
    """Upstream, verbatim."""
    return text.split(",")


def cluster_documents(filtered_synset_frames: list[str], n_clusters: int,
                      random_state: int = 42):
    """Upstream step 6a: TF-IDF over the filtered synset/frame column, K-Means."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_df=0.90, max_features=200000, min_df=1,
                          use_idf=True, tokenizer=my_tokenizer,
                          ngram_range=(1, 1), token_pattern=None)
    X = vec.fit_transform(filtered_synset_frames)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X)
    return km.labels_.tolist(), X


def elbow_n_clusters(filtered_synset_frames: list[str], k_min: int = 2,
                     k_max: int = 200, step: int = 1,
                     random_state: int = 42) -> int:
    """Upstream step 6b: choose k by the elbow of the K-Means inertia curve.

    Upstream plots the curve and the operator reads the elbow off it. The knee
    is selected here by maximum distance to the chord joining the first and last
    points of the inertia curve, which is the standard closed form of that
    reading, so that the choice is reproducible rather than manual.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_df=0.90, max_features=200000, min_df=1,
                          use_idf=True, tokenizer=my_tokenizer,
                          ngram_range=(1, 1), token_pattern=None)
    X = vec.fit_transform(filtered_synset_frames)
    k_max = min(k_max, X.shape[0] - 1)
    ks = list(range(k_min, max(k_min + 1, k_max + 1), step))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=3)
        km.fit(X)
        inertias.append(km.inertia_)
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    d = np.abs(y - (1 - x))          # distance to the chord (1,0)-(0,1)
    return ks[int(np.argmax(d))]
