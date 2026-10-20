"""Schema discovery: shows which key paths exist in your export, WITHOUT showing any personal values.
The output of `explore` is safe to paste into a chat. Use it to fix field names in extract.py."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from .models import Candidate


def _type_name(v: Any) -> str:
    return "null" if v is None else type(v).__name__


def _walk(node: Any, path: str, out: Set[Tuple[str, str]], depth: int = 0) -> None:
    if depth > 9:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            out.add((p, _type_name(v)))
            _walk(v, p, out, depth + 1)
    elif isinstance(node, list):
        for item in node[:5]:                     # first few items are enough to learn the shape
            p = path + "[]"
            out.add((p, _type_name(item)))
            _walk(item, p, out, depth + 1)


def path_inventory(records: List[dict]) -> List[Tuple[str, str, int]]:
    """(path, type, number of records containing it), most common first."""
    counter: Counter = Counter()
    for rec in records:
        seen: Set[Tuple[str, str]] = set()
        _walk(rec.get("raw", rec), "", seen)
        counter.update(seen)
    return sorted(((p, t, n) for (p, t), n in counter.items()), key=lambda x: (-x[2], x[0]))


def coverage(cands: List[Candidate]) -> Dict[str, int]:
    return {
        "candidates": len(cands),
        "name": sum(1 for c in cands if c.name),
        "headline": sum(1 for c in cands if c.headline),
        "location": sum(1 for c in cands if c.location),
        "connections": sum(1 for c in cands if c.connections is not None),
        "has positions": sum(1 for c in cands if c.positions),
        "positions with start date": sum(1 for c in cands if any(p.start for p in c.positions)),
        "positions with company name": sum(1 for c in cands if any(p.company for p in c.positions)),
        "has current position (no end date)": sum(1 for c in cands if any(p.start and not p.end for p in c.positions)),
        "skills": sum(1 for c in cands if c.skills),
        "education": sum(1 for c in cands if c.education),
        "profile link": sum(1 for c in cands if c.profile_url),
    }
