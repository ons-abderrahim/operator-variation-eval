#!/usr/bin/env python3
"""Reproduce the participant-structure analysis. Requires no dataset access.

Assembly101 video and features are gated but the list of recording names is
public, and participant identity is encoded in those names. So the question
this project rests on, whether there are enough recordings per person to hold
people out, can be checked by anyone straight away.

    python scripts/check_participants.py
"""
from __future__ import annotations

import urllib.request
from collections import Counter
from pathlib import Path


from participants import build_index

URL = (
    "https://raw.githubusercontent.com/assembly-101/"
    "assembly101-download-scripts/main/recording_names.txt"
)
CACHE = Path(__file__).resolve().parent / "results" / "recording_names.txt"


def load_names() -> list[str]:
    if not CACHE.exists():
        print(f"fetching {URL}")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    return [l.strip() for l in CACHE.read_text().splitlines() if l.strip()]


def main() -> int:
    names = load_names()
    idx = build_index(names)

    print(f"\ninput recordings    {len(names)}")
    print(idx.summary())

    if idx.aliased:
        pairs = sorted({v for v in idx.aliased.values()})
        print("\nBenign aliases (second id is not an independent participant;")
        print("grouping is unchanged, only the label differs):")
        for a, b in pairs:
            n = sum(1 for v in idx.aliased.values() if v == (a, b))
            print(f"  {a} <- {b}   {n} recordings")

    if idx.ambiguous:
        print("\nAmbiguous (both ids are real participants with their own")
        print("recordings; assignment is unresolvable, so these are excluded):")
        for name, (a, b) in sorted(idx.ambiguous.items()):
            print(f"  {a} vs {b}   {name}")

    counts = Counter(idx.assignment.values())
    thin = {p: c for p, c in counts.items() if c < 4}
    if thin:
        print(f"\nParticipants with fewer than 4 recordings: {thin}")

    print(
        f"\nLeave-one-participant-out is viable: {len(counts)} people, "
        f"at least {min(counts.values())} recordings each."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
