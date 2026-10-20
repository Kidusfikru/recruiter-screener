"""Configurable hard-rule engine with a what-if funnel.

Each rule returns pass / fail / missing. `missing` is resolved by the rule's `on_missing` setting:
pass -> ok, fail -> rejected, flag -> kept but marked "review". The funnel shows how many people each
rule removes so you can loosen or tighten rules by looking at numbers instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from .models import Candidate, Position

DAYS_PER_MONTH = 30.4375
MERGE_GAP = timedelta(days=62)        # a gap this small between two roles at one company is a promotion


@dataclass
class Check:
    rule: str
    status: str                        # pass | fail | missing
    detail: str = ""


@dataclass
class Verdict:
    candidate_id: str
    checks: List[Check] = field(default_factory=list)
    outcome: str = "pass"              # pass | review | fail

    @property
    def reasons(self) -> List[str]:
        return [f"{c.rule}: {c.detail}" for c in self.checks if c.status != "pass" and c.detail]


@dataclass
class RuleSet:
    as_of: date
    rules: Dict[str, Dict[str, Any]]
    profile: str = ""


# ---- loading + overrides -----------------------------------------------------------------------
def load_rules(path: str = "config/rules.yaml", overrides: Optional[List[str]] = None,
               profile: Optional[str] = None, *, require_profile: bool = False) -> RuleSet:
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid rules YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError("rules YAML must contain a mapping")

    profiles = cfg.get("profiles")
    if profiles is not None:
        if not isinstance(profiles, dict):
            raise ValueError("profiles must be a mapping")
        if require_profile and not profile:
            raise ValueError("the job must select a hard-rule profile with rules_profile")
        profile = profile or cfg.get("default_profile")
        if not profile:
            raise ValueError("a rules profile is required; set default_profile or select one with --profile")
        if profile not in profiles:
            raise ValueError(f"unknown rules profile {profile!r}; known: {', '.join(profiles)}")
        selected = profiles[profile] or {}
        common = cfg.get("defaults", {}) or {}
        if not isinstance(selected, dict) or not isinstance(common, dict):
            raise ValueError("selected profile and defaults must be mappings")
        common_rules = common.get("rules", {}) or {}
        selected_rules = selected.get("rules", {}) or {}
        if not isinstance(common_rules, dict) or not isinstance(selected_rules, dict):
            raise ValueError("defaults.rules and profile.rules must be mappings")
        for name, values in common_rules.items():
            if not isinstance(values, dict):
                raise ValueError(f"configuration for rule {name!r} must be a mapping")
        rules = {name: dict(values) for name, values in common_rules.items()}
        for name, values in selected_rules.items():
            if not isinstance(values, dict):
                raise ValueError(f"configuration for rule {name!r} must be a mapping")
            rules[name] = {**rules.get(name, {}), **values}
        as_of = selected.get("as_of", common.get("as_of", cfg.get("as_of")))
    else:
        # Continue accepting the original single-rule-set format.
        profile = ""
        rules = cfg.get("rules", {}) or {}
        if not isinstance(rules, dict):
            raise ValueError("rules must be a mapping")
        as_of = cfg.get("as_of")

    _validate_rules(rules)
    for item in overrides or []:                               # "min_connections.threshold=20"
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--set expects rule.key=value, got: {item!r}")
        lhs, rhs = item.split("=", 1)
        rule, key = lhs.split(".", 1)
        if rule not in rules:
            raise ValueError(f"unknown rule {rule!r}; known: {', '.join(rules)}")
        rules[rule][key.strip()] = yaml.safe_load(rhs)
    _validate_rules(rules)
    if isinstance(as_of, str):
        try:
            as_of = date.fromisoformat(as_of)
        except ValueError as e:
            raise ValueError(f"as_of must be an ISO date, got {as_of!r}") from e
    elif as_of is not None and not isinstance(as_of, date):
        raise ValueError("as_of must be an ISO date or null")
    return RuleSet(as_of=as_of or date.today(), rules=rules, profile=profile or "")


# ---- helpers -----------------------------------------------------------------------------------
def _months(a: date, b: date) -> float:
    """Calendar months, counted inclusively like LinkedIn does (Jan 2020 - Dec 2020 = 12 months).
    Recruiter dates only have month precision, so day-level maths would wrongly shave a month off."""
    return (b.year - a.year) * 12 + (b.month - a.month) + 1


def _fmt_months(m: float) -> str:
    y, mo = divmod(int(round(m)), 12)
    return f"{y}y {mo}m" if y else f"{mo}m"


@dataclass
class Stint:
    company: str
    start: date
    end: date
    current: bool


def build_stints(cand: Candidate, as_of: date, *, merge: bool, ignore_keywords: List[str]) -> List[Stint]:
    kws = [k.lower() for k in ignore_keywords]
    items: List[Tuple[Position, date, date, bool]] = []
    for p in cand.positions:
        if not p.start or any(k in p.title.lower() for k in kws):
            continue
        end = p.end or as_of
        if end < p.start:                                      # bad data, ignore rather than guess
            continue
        items.append((p, p.start, end, p.end is None))
    items.sort(key=lambda t: t[1])

    stints: List[Stint] = []
    for p, s, e, cur in items:
        key = p.company_key
        if merge and key:
            for st in reversed(stints):
                if st.company == key and s <= st.end + MERGE_GAP:
                    st.end, st.current = max(st.end, e), st.current or cur
                    break
            else:
                stints.append(Stint(key, s, e, cur))
        else:
            stints.append(Stint(key or f"?{len(stints)}", s, e, cur))
    return stints


# ---- the rules ---------------------------------------------------------------------------------
def rule_min_connections(c: Candidate, cfg: dict, as_of: date) -> Check:
    name, need = "min_connections", cfg.get("threshold", 30)
    if c.connections is None:
        return Check(name, "missing", "connection count not in data")
    if c.connections_capped and c.connections >= need:
        return Check(name, "pass", f"{c.connections}+")
    ok = c.connections >= need
    return Check(name, "pass" if ok else "fail", f"{c.connections} connections (need {need})")


def rule_job_hopper(c: Candidate, cfg: dict, as_of: date) -> Check:
    name = "job_hopper"
    stints = build_stints(c, as_of, merge=cfg.get("merge_same_company", True),
                          ignore_keywords=cfg.get("ignore_title_keywords", []))
    if not stints:
        return Check(name, "missing", "no dated positions")
    limit_m = cfg.get("short_stint_months", 12)
    ignore_cur = cfg.get("ignore_current", True)
    short = [s for s in stints if _months(s.start, s.end) < limit_m and not (s.current and ignore_cur)]
    ok = len(short) <= cfg.get("max_short_stints", 1)
    return Check(name, "pass" if ok else "fail",
                 f"{len(short)} stint(s) under {limit_m}m (allowed {cfg.get('max_short_stints', 1)})")


def rule_min_current_tenure(c: Candidate, cfg: dict, as_of: date) -> Check:
    name, need = "min_current_tenure", cfg.get("months", 10)
    stints = build_stints(c, as_of, merge=cfg.get("merge_same_company", True), ignore_keywords=[])
    cur = [s for s in stints if s.current]
    if not cur:
        return Check(name, "missing", "no current dated position")
    best = max(_months(s.start, s.end) for s in cur)         # longest current role wins (side gigs don't count)
    return Check(name, "pass" if best >= need else "fail", f"{_fmt_months(best)} in current role (need {need}m)")


def rule_min_total_experience(c: Candidate, cfg: dict, as_of: date) -> Check:
    name, need = "min_total_experience", cfg.get("years", 4)
    spans = sorted(
        (p.start.year * 12 + p.start.month - 1,
         (p.end or as_of).year * 12 + (p.end or as_of).month - 1)
        for p in c.positions
        if p.start and (p.end or as_of) >= p.start
    )
    if not spans:
        return Check(name, "missing", "no dated positions")
    total_months, (cs, ce) = 0, spans[0]
    for s, e in spans[1:]:
        if s <= ce + 1:
            ce = max(ce, e)
        else:
            total_months += ce - cs + 1
            cs, ce = s, e
    total_months += ce - cs + 1
    years = total_months / 12
    return Check(name, "pass" if years >= need else "fail", f"{years:.1f} years total (need {need})")


def rule_timezone_preference(c: Candidate, cfg: dict, as_of: date) -> Check:
    name = "timezone_preference"
    if not c.location:
        return Check(name, "missing", "location not listed on profile")

    loc_lower = c.location.lower()
    preferred = [r.lower() for r in cfg.get("preferred_regions", [])]

    # Check if candidate location matches any preferred UTC+2..UTC+4 country/city
    if any(p in loc_lower for p in preferred):
        return Check(name, "pass", f"Location '{c.location}' matches preferred UTC+2..UTC+4")

    # Outside preferred zones: flag for working hours review per client brief
    return Check(name, "missing", f"Location '{c.location}' is outside UTC+2..UTC+4 (check working hours)")


RULES: Dict[str, Callable[[Candidate, dict, date], Check]] = {
    "min_connections": rule_min_connections,
    "job_hopper": rule_job_hopper,
    "min_current_tenure": rule_min_current_tenure,
    "min_total_experience": rule_min_total_experience,
    "timezone_preference": rule_timezone_preference,
}

RULE_KEYS = {
    "min_connections": {"enabled", "threshold", "on_missing"},
    "job_hopper": {
        "enabled", "short_stint_months", "max_short_stints", "ignore_current",
        "merge_same_company", "ignore_title_keywords", "on_missing",
    },
    "min_current_tenure": {"enabled", "months", "merge_same_company", "on_missing"},
    "min_total_experience": {"enabled", "years", "on_missing"},
    "timezone_preference": {"enabled", "preferred_regions", "on_missing"},
}


def _validate_rules(rules: Dict[str, Dict[str, Any]]) -> None:
    for name, cfg in rules.items():
        if name not in RULES:
            raise ValueError(f"unknown hard rule {name!r}; known: {', '.join(RULES)}")
        if not isinstance(cfg, dict):
            raise ValueError(f"configuration for rule {name!r} must be a mapping")
        unknown = set(cfg) - RULE_KEYS[name]
        if unknown:
            raise ValueError(f"unknown setting(s) for rule {name!r}: {', '.join(sorted(unknown))}")
        if "enabled" in cfg and not isinstance(cfg["enabled"], bool):
            raise ValueError(f"{name}.enabled must be true or false")
        if "on_missing" in cfg and (
            not isinstance(cfg["on_missing"], str)
            or cfg["on_missing"] not in {"pass", "fail", "flag"}
        ):
            raise ValueError(f"{name}.on_missing must be pass, fail, or flag")
        for key in ("ignore_current", "merge_same_company"):
            if key in cfg and not isinstance(cfg[key], bool):
                raise ValueError(f"{name}.{key} must be true or false")
        for key in ("threshold", "short_stint_months", "max_short_stints", "months"):
            if key in cfg and (
                isinstance(cfg[key], bool)
                or not isinstance(cfg[key], int)
                or cfg[key] < 0
            ):
                raise ValueError(f"{name}.{key} must be a non-negative integer")
        for key in ("years",):
            if key in cfg and (
                isinstance(cfg[key], bool)
                or not isinstance(cfg[key], (int, float))
                or cfg[key] < 0
            ):
                raise ValueError(f"{name}.{key} must be a non-negative number")
        if "preferred_regions" in cfg and (
            not isinstance(cfg["preferred_regions"], list)
            or any(not isinstance(region, str) or not region.strip()
                   for region in cfg["preferred_regions"])
        ):
            raise ValueError("timezone_preference.preferred_regions must be a list of non-empty strings")
        if "ignore_title_keywords" in cfg and (
            not isinstance(cfg["ignore_title_keywords"], list)
            or any(not isinstance(keyword, str) or not keyword.strip()
                   for keyword in cfg["ignore_title_keywords"])
        ):
            raise ValueError("job_hopper.ignore_title_keywords must be a list of non-empty strings")


# ---- evaluation + funnel -----------------------------------------------------------------------
def _effective(check: Check, cfg: dict) -> str:
    """pass | fail | review after applying on_missing."""
    if check.status != "missing":
        return check.status
    return {"pass": "pass", "fail": "fail"}.get(cfg.get("on_missing", "flag"), "review")


def evaluate(c: Candidate, rs: RuleSet) -> Verdict:
    v = Verdict(candidate_id=c.id)
    outcomes = []
    for name, fn in RULES.items():
        cfg = rs.rules.get(name)
        if not cfg or not cfg.get("enabled", True):
            continue
        check = fn(c, cfg, rs.as_of)
        v.checks.append(check)
        outcomes.append(_effective(check, cfg))
    v.outcome = "fail" if "fail" in outcomes else "review" if "review" in outcomes else "pass"
    return v


def funnel(cands: List[Candidate], rs: RuleSet) -> List[dict]:
    """Per rule: how many people it rejects on its own, and how many remain after applying rules in order."""
    rows, remaining = [], list(cands)
    for name, fn in RULES.items():
        cfg = rs.rules.get(name)
        if not cfg or not cfg.get("enabled", True):
            continue
        alone_fail = review = 0
        survivors = []
        for c in cands:
            eff = _effective(fn(c, cfg, rs.as_of), cfg)
            alone_fail += eff == "fail"
            review += eff == "review"
        for c in remaining:
            if _effective(fn(c, cfg, rs.as_of), cfg) != "fail":
                survivors.append(c)
        remaining = survivors
        rows.append({"rule": name, "rejects_alone": alone_fail, "flagged": review, "remaining": len(remaining)})
    return rows
