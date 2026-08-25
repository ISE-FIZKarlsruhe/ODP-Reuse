#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
framester_client.py -- step 4 of the conceptual-components pipeline, standalone.

Reads the community documents written by ``run_asprino_full.py --stage
documents`` and writes the synset/frame annotations that ``--stage score``
consumes. It exists as a separate program because the step-3 server hard-codes
its bind address to ``http://localhost:8080/``: it is reachable only from the
machine (or WSL distribution) it runs on, so on Windows the disambiguation has
to happen beside the server while the rest of the pipeline runs on the corpus.

    # inside WSL, next to the running jar
    python3 tools/framester_client.py \
        --in  /mnt/c/.../results/asprino-full/documents.json \
        --out /mnt/c/.../results/asprino-full/synsets.json \
        --workers 6

Only the standard library is used, so no packages need installing beside the
server.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHORTEN = {
    "https://w3id.org/framester/wn/wn30/instances/": "wn30:",
    "https://w3id.org/framester/framenet/abox/frame/": "frame:",
}


def replace_all(text: str) -> str:
    for i, j in SHORTEN.items():
        text = text.replace(i, j)
    return text


def _get(base: str, endpoint: str, params: dict, timeout: int, retries: int) -> str:
    url = f"{base}/{endpoint}?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    last = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{endpoint}: {type(last).__name__}: {last}")


def disambiguate_one(text: str, base: str, timeout: int, retries: int) -> dict:
    if not text.strip():
        return {"synset frame": text, "annotated synset frame": text}
    resp = _get(base, "cd", {"text": text, "senseInventory": "fn17"}, timeout, retries)
    if resp.strip() in ("", "[]"):
        return {"synset frame": text, "annotated synset frame": text}
    start, end = resp.find("[") + 1, resp.find("]")
    processed = resp[start:end].replace('"', "")
    annotated = _get(base, "wfd", {"text": text, "serialization": "json"},
                     timeout, retries)
    return {"synset frame": replace_all(processed),
            "annotated synset frame": replace_all(annotated)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true",
                    help="keep rows already present in --out and only fill gaps")
    args = ap.parse_args()

    records = json.loads(args.inp.read_text(encoding="utf-8"))
    texts = [r["text"] for r in records]
    print(f"{len(texts)} documents -> {args.server}", flush=True)

    out: list[dict | None] = [None] * len(texts)
    if args.resume and args.out.is_file():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        for i, p in enumerate(prev[:len(out)]):
            if p and "error" not in p:
                out[i] = p
        print(f"[resume] {sum(1 for o in out if o)} already done", flush=True)

    todo = [i for i, o in enumerate(out) if o is None]
    t0 = time.time()
    done = 0

    def job(i: int):
        nonlocal done
        try:
            out[i] = disambiguate_one(texts[i], args.server, args.timeout,
                                      args.retries)
        except Exception as exc:  # noqa: BLE001
            out[i] = {"synset frame": texts[i],
                      "annotated synset frame": texts[i],
                      "error": f"{type(exc).__name__}: {exc}"}
        done += 1
        if done % 100 == 0:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"  {done}/{len(todo)}  {el:.0f}s  "
                  f"{rate:.1f}/s  eta {(len(todo)-done)/max(rate,1e-9):.0f}s",
                  flush=True)
        return i

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(job, todo))

    n_err = sum(1 for o in out if o and "error" in o)
    args.out.write_text(json.dumps([o for o in out]), encoding="utf-8")
    print(f"[OK] {args.out}  ({n_err} errors, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
