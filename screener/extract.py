"""Defensive extractor: raw Recruiter JSON -> Candidate."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, List, Optional, Tuple

from .models import Candidate, Education, Position, Project

CURRENT_POS_KEYS = ["currentPositions"]
PAST_POS_KEYS = ["pastPositions", "positions", "experience", "workExperience", "workExperiences",
                 "experiences", "profilePositions", "positionHistory"]
SKILL_KEYS = ["skills", "topSkills", "profileSkills", "associatedProfileSkillNames"]
EDU_KEYS = ["educations", "education", "educationHistory", "schools"]
CONNECTION_KEYS = ["numConnections", "connectionsCount", "numOfConnections", "connectionCount",
                   "connections", "numberOfConnections"]
TITLE_KEYS = ["title", "positionTitle", "jobTitle", "localizedTitle"]
COMPANY_KEYS = ["companyName", "company", "organization", "employerName"]
COMPANY_URN_KEYS = ["companyUrn", "companyId", "organizationUrn"]
DESC_KEYS = ["description", "summary"]
START_KEYS = ["startDateOn", "startedOn", "startDate", "start", "startsAt"]
END_KEYS = ["endDateOn", "endedOn", "endDate", "end", "endsAt"]
RANGE_KEYS = ["dateRange", "timePeriod", "timeRange", "tenure"]

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _text(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, dict):
        for k in ("name", "displayName", "localizedName", "text", "value", "defaultLocalizedName"):
            if k in v:
                t = _text(v[k])
                if t:
                    return t
    return ""


def bfs(obj: Any, keys: Iterable[str], test: Callable[[Any], bool], max_depth: int = 6) -> Any:
    for key in keys:
        queue: List[Tuple[Any, int]] = [(obj, 0)]
        while queue:
            node, depth = queue.pop(0)
            if isinstance(node, dict):
                if key in node and test(node[key]):
                    return node[key]
                if depth < max_depth:
                    queue.extend((v, depth + 1) for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list) and depth < max_depth:
                queue.extend((v, depth + 1) for v in node if isinstance(v, (dict, list)))
    return None


def _nonempty_text(v: Any) -> bool:
    return bool(_text(v))


def _list_of_dicts(v: Any) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(i, (dict, str)) for i in v)


def find_all_lists(obj: Any, keys: Iterable[str], max_depth: int = 6) -> List[Tuple[str, List[Any]]]:
    wanted, out = set(keys), []
    queue: List[Tuple[Any, int]] = [(obj, 0)]
    while queue:
        node, depth = queue.pop(0)
        if isinstance(node, dict):
            for k, v in node.items():
                if k in wanted and _list_of_dicts(v):
                    out.append((k, v))
                elif depth < max_depth and isinstance(v, (dict, list)):
                    queue.append((v, depth + 1))
        elif isinstance(node, list) and depth < max_depth:
            queue.extend((v, depth + 1) for v in node if isinstance(v, (dict, list)))
    return out


def parse_date(v: Any, *, default_month: int = 6) -> Optional[date]:
    try:
        if v is None or v == "":
            return None
        if isinstance(v, dict):
            for inner in ("date", "value"):
                if inner in v and isinstance(v[inner], (dict, str, int)):
                    return parse_date(v[inner], default_month=default_month)
            year = v.get("year")
            if not year:
                return None
            month = v.get("month") or default_month
            return date(int(year), min(max(int(month), 1), 12), min(max(int(v.get("day") or 1), 1), 28))
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            if 1900 <= v <= 2100:
                return date(int(v), default_month, 1)
            secs = v / 1000 if v > 1e11 else v
            return datetime.fromtimestamp(secs, tz=timezone.utc).date()
        s = str(v).strip().lower()
        if s in ("present", "current", "now", "today"):
            return None
        m = re.match(r"^(\d{4})-(\d{1,2})", s)
        if m:
            return date(int(m.group(1)), min(max(int(m.group(2)), 1), 12), 1)
        m = re.match(r"^([a-z]{3})[a-z]*\.?,?\s+(\d{4})$", s)
        if m and m.group(1) in _MONTHS:
            return date(int(m.group(2)), _MONTHS[m.group(1)], 1)
        m = re.match(r"^(\d{4})$", s)
        if m:
            return date(int(m.group(1)), default_month, 1)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def _first_date(d: dict, keys: List[str]) -> Optional[date]:
    for k in keys:
        if k in d:
            dt = parse_date(d[k])
            if dt:
                return dt
    return None


def parse_position(d: Any, *, force_current: bool = False, default_company: str = "") -> Optional[Position]:
    if not isinstance(d, dict):
        return None
    title = _text(bfs(d, TITLE_KEYS, _nonempty_text, 1) or "")
    company = _text(bfs(d, COMPANY_KEYS, _nonempty_text, 1) or "") or default_company
    company_urn = _text(bfs(d, COMPANY_URN_KEYS, _nonempty_text, 1) or "")
    desc = _text(bfs(d, DESC_KEYS, lambda v: isinstance(v, str) and bool(v.strip()), 1) or "")

    # Look for nested resume details
    res_pos = bfs(d, ["resumePosition"], lambda v: isinstance(v, dict))
    if res_pos and isinstance(res_pos, dict):
        extra = _text(res_pos.get("details"))
        if extra and extra not in desc:
            desc = f"{desc}\n{extra}".strip()

    start = _first_date(d, START_KEYS)
    end = _first_date(d, END_KEYS)
    for rk in RANGE_KEYS:
        rng = d.get(rk)
        if isinstance(rng, dict):
            start = start or _first_date(rng, START_KEYS)
            end = end or _first_date(rng, END_KEYS)
    if d.get("startYear") and not start:
        start = parse_date({"year": d["startYear"], "month": d.get("startMonth")})
    if d.get("endYear") and not end:
        end = parse_date({"year": d["endYear"], "month": d.get("endMonth")})
    if force_current or d.get("current") is True or d.get("isCurrent") is True:
        end = None
    if not (title or company or company_urn):
        return None
    return Position(title=title, company=company, company_urn=company_urn,
                    start=start, end=end, description=desc)


def parse_education(d: Any) -> Optional[Education]:
    if isinstance(d, str):
        return Education(school=d.strip()) if d.strip() else None
    if not isinstance(d, dict):
        return None
    school = _text(bfs(d, ["schoolName", "school", "institution", "name"], _nonempty_text, 1) or "")
    degree = _text(bfs(d, ["degreeName", "degree"], _nonempty_text, 1) or "")
    fld = _text(bfs(d, ["fieldOfStudy", "field", "major"], _nonempty_text, 1) or "")
    return Education(school=school, degree=degree, field=fld) if (school or degree or fld) else None


def parse_connections(v: Any) -> Tuple[Optional[int], bool]:
    if isinstance(v, bool):
        return None, False
    if isinstance(v, (int, float)):
        return int(v), False
    if isinstance(v, str):
        m = re.search(r"(\d[\d,\.]*)\s*(\+)?", v)
        if m:
            n = int(re.sub(r"[^\d]", "", m.group(1)) or 0)
            return n, bool(m.group(2))
    return None, False


def _is_connection_value(v: Any) -> bool:
    return parse_connections(v)[0] is not None


def _stable_id(rec: dict, raw: dict) -> str:
    for k in ("referenceUrn", "id", "urn"):
        val = rec.get(k) or raw.get(k)
        if val:
            m = re.search(r"urn:li:ts_profile:[A-Za-z0-9_-]+", str(val))
            if m:
                return m.group(0)
            m_aema = re.search(r"AEMA[A-Za-z0-9_-]+", str(val))
            if m_aema:
                return f"urn:li:ts_profile:{m_aema.group(0)}"
            return str(val)

    urn = bfs(raw, ["referenceUrn", "memberUrn", "profileUrn", "entityUrn"], _nonempty_text)
    if urn:
        s = _text(urn)
        m = re.search(r"urn:li:ts_profile:[A-Za-z0-9_-]+", s)
        return m.group(0) if m else s
    return "h-" + hashlib.sha1(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:12]


def extract_candidate(record: dict) -> Candidate:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else record
    cid = _stable_id(record, raw)

    first = _text(bfs(raw, ["unobfuscatedFirstName", "firstName"], _nonempty_text) or "")
    last = _text(bfs(raw, ["unobfuscatedLastName", "lastName"], _nonempty_text) or "")
    name = f"{first} {last}".strip() or _text(bfs(raw, ["fullName", "name"], _nonempty_text) or "") \
        or str(record.get("name") or "")

    headline = _text(bfs(raw, ["headline"], _nonempty_text) or "") or str(record.get("headline") or "")
    location = _text(bfs(raw, ["location", "geoRegion", "locationName", "geoLocation"], _nonempty_text) or "") \
        or str(record.get("location") or "")
    summary = _text(bfs(raw, ["summary", "about"], lambda v: isinstance(v, str) and bool(v.strip())) or "")

    n_conn, capped = parse_connections(bfs(raw, CONNECTION_KEYS, _is_connection_value))

    # --- Extract Positions ---
    positions: List[Position] = []
    seen = set()

    # 1. Grouped work experience (detail view)
    grouped = raw.get("groupedWorkExperience")
    if isinstance(grouped, list):
        for grp in grouped:
            comp = _text(grp.get("companyName") or grp.get("companyUrnResolutionResult") or "")
            pos_list = grp.get("positions") or []
            is_cur = any(p.get("endDateOn") is None for p in pos_list if isinstance(p, dict))
            for item in pos_list:
                p = parse_position(item, force_current=(is_cur and item.get("endDateOn") is None), default_company=comp)
                if p:
                    sig = (p.title.lower(), p.company_key, p.start)
                    if sig not in seen:
                        seen.add(sig)
                        positions.append(p)

    # 2. Regular current & past positions
    for forced, keys in ((True, CURRENT_POS_KEYS), (False, PAST_POS_KEYS)):
        for _, lst in find_all_lists(raw, keys):
            for item in lst:
                p = parse_position(item, force_current=forced)
                if not p:
                    continue
                sig = (p.title.lower(), p.company_key, p.start)
                if sig not in seen:
                    seen.add(sig)
                    positions.append(p)

    if not positions:
        t, c = str(record.get("currentTitle") or ""), str(record.get("currentCompany") or "")
        if t or c:
            positions.append(Position(title=t, company=c))

    # --- Extract Skills ---
    skills: List[str] = []
    for _, lst in find_all_lists(raw, SKILL_KEYS):
        for s in lst:
            t = _text(s)
            if t and t.lower() not in {x.lower() for x in skills}:
                skills.append(t)

    # --- Extract Education ---
    education: List[Education] = []
    for _, lst in find_all_lists(raw, EDU_KEYS):
        education += [e for e in (parse_education(i) for i in lst) if e]

    # --- Extract Accomplishments / Projects ---
    projects: List[Project] = []
    acc_projects = raw.get("accomplishments", {}).get("projects") if isinstance(raw.get("accomplishments"), dict) else None
    if isinstance(acc_projects, list):
        for prj in acc_projects:
            if isinstance(prj, dict):
                p_title = _text(prj.get("title") or "")
                p_desc = _text(prj.get("description") or "")
                if p_title or p_desc:
                    projects.append(Project(title=p_title, description=p_desc))

    url = str(record.get("profileUrl") or "") or _text(
        bfs(raw, ["publicProfileUrl", "profileUrl", "navigationUrl", "linkedInUrl"], _nonempty_text) or "")
    is_guess = False
    if not url:
        tail = cid.split(":")[-1]
        if tail and not cid.startswith("h-"):
            url, is_guess = f"https://www.linkedin.com/talent/profile/{tail}", True

    return Candidate(
        id=cid, name=name, headline=headline, location=location, summary=summary,
        connections=n_conn, connections_capped=capped, profile_url=url,
        profile_url_is_guess=is_guess, skills=skills, positions=positions,
        education=education, projects=projects
    )


def load_records(data: Any) -> List[dict]:
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for k in ("candidates", "elements", "hits"):
            if isinstance(data.get(k), list):
                return [d for d in data[k] if isinstance(d, dict)]
        return [data]
    return []