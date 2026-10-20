"""Tiny SQLite store (standard library only)."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

from .llm import ScoreResult
from .models import Candidate


def _merge_candidates(old: Candidate, new: Candidate) -> Candidate:
    d_old = old.model_dump()
    d_new = new.model_dump()
    for k, v in d_new.items():
        if k in {"skills", "positions", "education", "projects"}:
            continue
        if k == "connections_capped":
            if new.connections is not None:
                d_old[k] = v
        elif k == "profile_url_is_guess":
            if new.profile_url:
                d_old[k] = v
        elif v is not None and v != "":
            d_old[k] = v
    d_old["skills"] = list(dict.fromkeys(old.skills + new.skills))
    known_pos = {(p.title.lower(), p.company_key, p.start) for p in old.positions}
    combined_pos = list(old.positions)
    for p in new.positions:
        sig = (p.title.lower(), p.company_key, p.start)
        if sig not in known_pos:
            known_pos.add(sig)
            combined_pos.append(p)
    d_old["positions"] = [p.model_dump() if hasattr(p, "model_dump") else p for p in combined_pos]
    for key, old_items, new_items, signature in (
        ("education", old.education, new.education,
         lambda item: (item.school.lower(), item.degree.lower(), item.field.lower())),
        ("projects", old.projects, new.projects,
         lambda item: (item.title.lower(), item.description.lower())),
    ):
        known = {signature(item) for item in old_items}
        combined = list(old_items)
        for item in new_items:
            if signature(item) not in known:
                known.add(signature(item))
                combined.append(item)
        d_old[key] = [item.model_dump() for item in combined]
    return Candidate.model_validate(d_old)


class Store:
    def __init__(self, path: str = "data/screener.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS candidates (id TEXT PRIMARY KEY, json TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (cand_id TEXT, key TEXT, job_slug TEXT, "
                        "json TEXT NOT NULL, PRIMARY KEY (cand_id, key))")

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def upsert_candidates(self, cands: List[Candidate]) -> int:
        with self.db:
            for c in cands:
                row = self.db.execute("SELECT json FROM candidates WHERE id=?", (c.id,)).fetchone()
                if row:
                    merged = _merge_candidates(Candidate.model_validate_json(row[0]), c)
                    self.db.execute("UPDATE candidates SET json=? WHERE id=?", (merged.model_dump_json(), c.id))
                else:
                    self.db.execute("INSERT INTO candidates VALUES (?, ?)", (c.id, c.model_dump_json()))
        return len(cands)

    def all_candidates(self) -> List[Candidate]:
        rows = self.db.execute("SELECT json FROM candidates").fetchall()
        return [Candidate.model_validate_json(r[0]) for r in rows]

    def candidates_by_ids(self, cand_ids: List[str]) -> List[Candidate]:
        unique_ids = list(dict.fromkeys(cand_ids))
        found = {}
        for start in range(0, len(unique_ids), 900):
            batch = unique_ids[start:start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = self.db.execute(
                f"SELECT id, json FROM candidates WHERE id IN ({placeholders})", batch
            ).fetchall()
            found.update((cand_id, Candidate.model_validate_json(data)) for cand_id, data in rows)
        return [found[cand_id] for cand_id in unique_ids if cand_id in found]

    def get_score(self, cand_id: str, key: str) -> Optional[ScoreResult]:
        row = self.db.execute("SELECT json FROM scores WHERE cand_id=? AND key=?", (cand_id, key)).fetchone()
        return ScoreResult.model_validate_json(row[0]) if row else None

    def put_score(self, cand_id: str, key: str, job_slug: str, result: ScoreResult) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                            (cand_id, key, job_slug, result.model_dump_json()))

    def delete_candidate(self, cand_id: str) -> bool:
        with self.db:
            cursor = self.db.execute("DELETE FROM candidates WHERE id=?", (cand_id,))
            self.db.execute("DELETE FROM scores WHERE cand_id=?", (cand_id,))
        return cursor.rowcount > 0

    def backup(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup_db = sqlite3.connect(destination)
        try:
            self.db.backup(backup_db)
        finally:
            backup_db.close()

    def purge(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM candidates")
            self.db.execute("DELETE FROM scores")
        self.db.execute("VACUUM")