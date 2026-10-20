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


def _spreadsheet_safe(value: str) -> str:
    if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


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
            detail = " | ".join(
                f"[{r.status}] {r.requirement}"
                + (f' "{r.evidence}"' if r.evidence else "")
                + (f" ({r.note})" if r.note else "")
                for r in s.requirements
            )
            w.writerow([i, s.bucket, s.score, _spreadsheet_safe(c.name),
                        _spreadsheet_safe(c.profile_url), "yes" if c.profile_url_is_guess else "",
                        _spreadsheet_safe(_current_role(c)), _spreadsheet_safe(c.location),
                        _spreadsheet_safe("REVIEW: " + "; ".join(v.reasons)
                                          if v.outcome == "review" else "ok"),
                        _spreadsheet_safe(s.model), _spreadsheet_safe(s.evaluation_error),
                        _spreadsheet_safe(mand), _spreadsheet_safe(s.reason),
                        _spreadsheet_safe("; ".join(s.red_flags)),
                        _spreadsheet_safe(", ".join(s.keyword_hits)),
                        _spreadsheet_safe(s.capped_by), _spreadsheet_safe(detail)])
            n += 1
        for c, v, _ in failed:
            w.writerow(["", "-", "", _spreadsheet_safe(c.name),
                        _spreadsheet_safe(c.profile_url), "yes" if c.profile_url_is_guess else "",
                        _spreadsheet_safe(_current_role(c)), _spreadsheet_safe(c.location),
                        _spreadsheet_safe("FAILED: " + "; ".join(v.reasons)),
                        "", "", "", "", "", "", "", ""])
            n += 1
    return n
