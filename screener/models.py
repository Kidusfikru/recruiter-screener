"""Plain data shapes used everywhere else (pydantic validates and serialises them)."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class Position(BaseModel):
    title: str = ""
    company: str = ""
    company_urn: str = ""
    start: Optional[date] = None
    end: Optional[date] = None
    description: str = ""

    @property
    def company_key(self) -> str:
        return (self.company or self.company_urn).strip().lower()


class Education(BaseModel):
    school: str = ""
    degree: str = ""
    field: str = ""


class Project(BaseModel):
    title: str = ""
    description: str = ""


class Candidate(BaseModel):
    id: str
    name: str = ""
    headline: str = ""
    location: str = ""
    summary: str = ""
    connections: Optional[int] = None
    connections_capped: bool = False
    profile_url: str = ""
    profile_url_is_guess: bool = False
    skills: List[str] = Field(default_factory=list)
    positions: List[Position] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    semantic_similarity: Optional[float] = None