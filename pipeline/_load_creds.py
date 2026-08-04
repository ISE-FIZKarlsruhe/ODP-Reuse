#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_load_creds.py -- read API keys from a local credentials file into the
environment, without ever printing their values.

The file format is label-on-one-line, value-on-the-next:

    OPENAI_API_KEY
    sk-...
    GEMINI_API_KEY
    AQ...

Only the provider keys this project needs are exported. Values are never echoed;
confirmation is masked. This file is git-ignored and must never be committed --
see .gitignore.

Usage
    python -c "import _load_creds; _load_creds.apply()"       # in-process
    python _load_creds.py --check                             # masked report
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# label fragment -> environment variable the tooling reads
WANTED = {
    "OPENAI": "OPENAI_API_KEY",
    "ANTHROPIC": "ANTHROPIC_API_KEY",
    "GEMINI": "GEMINI_API_KEY",
    "GOOGLE_API": "GOOGLE_API_KEY",
    "GITHUB": "GITHUB_TOKEN",
    "DEEPSEEK": "DEEPSEEK_API_KEY",
}
# value shapes, so a label is never mistaken for a value
VALUE_HINT = re.compile(r"^(sk-|AQ\.|ghp_|gh[pousr]_|hf_|pypi-|AIza)")

CANDIDATES = [
    Path.home() / "Desktop" / "cred.txt",
    Path.home() / "OneDrive" / "Desktop" / "cred.txt",
]
CANDIDATES += list(Path.home().glob("OneDrive*/Desktop/cred.txt"))


def find() -> Path | None:
    return next((p for p in CANDIDATES if p.is_file()), None)


def parse(path: Path) -> dict[str, str]:
    lines = [ln.strip() for ln in
             path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln]
    found: dict[str, str] = {}
    for i, ln in enumerate(lines):
        up = ln.upper()
        for frag, env in WANTED.items():
            if up.startswith(frag) and len(ln) < 42:      # this line is a LABEL
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                if nxt and (VALUE_HINT.match(nxt) or len(nxt) > 20):
                    found.setdefault(env, nxt)
    # Gemini and Google are the same credential for the generative API
    if "GEMINI_API_KEY" in found:
        found.setdefault("GOOGLE_API_KEY", found["GEMINI_API_KEY"])
    return found


def apply(verbose: bool = True) -> dict[str, str]:
    p = find()
    if p is None:
        if verbose:
            print("[creds] no credentials file found", file=sys.stderr)
        return {}
    got = parse(p)
    for env, val in got.items():
        os.environ[env] = val
    if verbose:
        for env, val in sorted(got.items()):
            print(f"[creds] {env:20s} loaded  ({val[:4]}…{val[-3:]}, "
                  f"{len(val)} chars)")
    return got


if __name__ == "__main__":
    got = apply()
    print(f"\n{len(got)} credential(s) loaded from {find()}")
    print("Values are never printed in full and this file is git-ignored.")
