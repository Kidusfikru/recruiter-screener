"""Ranked CSV export. The name and profile link live here (your sheet) and nowhere in the AI prompt."""
from __future__ import annotations

import csv
from typing import List, Optional, Tuple

from .llm import ScoreResult
from .models import Candidate
from .rules import Verdict

COLUMNS = ["rank", "bucket", "score", "name", "profile_url", "link_is_guess", "current_role", "location",
           "rules", "model", "evaluation_error", "mandatory", "reason", "red_flags", "keyword_hits", "capped_by",
           "requirement_detail"]

Row = Tuple[Candidate, Verdict, Optional[ScoreResult]]


def _current_role(c: Candidate) -> str:
    cur = [p for p in c.positions if p.end is None and (p.title or p.company)]
    p = cur[0] if cur else (c.positions[0] if c.positions else None)
    return f"{p.title} @ {p.company}".strip(" @") if p else c.headline


def write_csv(rows: List[Row], path: str, include_failed: bool = False) -> int:
    scored = sorted([r for r in rows if r[2] is not None],
                    key=lambda r: (-r[2].score, -len(r[2].keyword_hits), r[0].name.lower()))
    failed = [r for r in rows if r[1].outcome == "fail"] if include_failed else []
    n = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as f:     # utf-8-sig: Excel/Sheets open it cleanly
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, (c, v, s) in enumerate(scored, 1):
            mand = "; ".join(f"{r.status}" for r in s.requirements if r.mandatory) or "-"
            detail = " | ".join(f"[{r.status}] {r.requirement}" + (f' "{r.evidence}"' if r.evidence else "")
                                for r in s.requirements)
            w.writerow([i, s.bucket, s.score, c.name, c.profile_url, "yes" if c.profile_url_is_guess else "",
                        _current_role(c), c.location,
                        "REVIEW: " + "; ".join(v.reasons) if v.outcome == "review" else "ok",
                        s.model, s.evaluation_error, mand, s.reason, "; ".join(s.red_flags),
                        ", ".join(s.keyword_hits), s.capped_by, detail])
            n += 1
        for c, v, _ in failed:
            w.writerow(["", "-", "", c.name, c.profile_url, "yes" if c.profile_url_is_guess else "",
                        _current_role(c), c.location, "FAILED: " + "; ".join(v.reasons), "", "", "", "", "", "", "", ""])
            n += 1
    return n
