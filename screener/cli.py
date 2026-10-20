"""Command line: ingest -> filter -> Google Sheets dedup -> score -> export."""
from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import quote, urlsplit, urlunsplit

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from .explore import coverage
from .extract import extract_candidate, load_records
from .llm import (
    DEFAULT_EMBED_MODEL,
    DEEP_MODEL,
    FAST_MODEL,
    DEFAULT_HOST,
    DEFAULT_MODEL,
    Job,
    OllamaClient,
    cache_key,
    candidate_text,
    cosine_similarity,
    score_candidate,
)
from .models import Candidate
from .report import write_csv
from .rules import RuleSet, evaluate, funnel, load_rules
from .store import Store

app = typer.Typer(add_completion=False, help="Screen Recruiter Lite exports into a ranked shortlist.")
console = Console()

DB_OPT = typer.Option("data/screener.db", "--db", help="SQLite file")
RULES_OPT = typer.Option("config/rules.yaml", "--rules", help="Rules YAML")
SET_OPT = typer.Option(None, "--set", help="Override a rule e.g. min_current_tenure.months=6")
MODEL_OPT = typer.Option(None, "--model", help="Use this model (first stage with --two-stage)")
EMBED_OPT = typer.Option(DEFAULT_EMBED_MODEL, "--embed-model", help="Ollama embedding model")
SHEET_OPT = typer.Option(None, "--sheet-url", help="Google Sheet URL to check for already-contacted candidates")

DOWNLOADS = Path.home() / "Downloads"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        console.print(f"[red]Cannot read {path}: {e}[/red]")
        raise typer.Exit(1)


def _rules(path: str, overrides: Optional[List[str]], profile: Optional[str] = None,
           require_profile: bool = False) -> RuleSet:
    try:
        return load_rules(path, overrides, profile, require_profile=require_profile)
    except (ValueError, OSError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


def find_export(arg: Optional[Path], folder: Path = DOWNLOADS) -> Path:
    if arg is not None and arg.is_file():
        return arg
    if arg is not None and (folder / arg.name).is_file():
        return folder / arg.name
    if arg is not None and not arg.is_dir():
        console.print(f"[red]Requested candidate export does not exist: {arg}[/red]")
        raise typer.Exit(1)
    search = arg if (arg is not None and arg.is_dir()) else folder

    detail_files = sorted(search.glob("candidates-detail-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if detail_files:
        return detail_files[0]

    general_files = sorted(search.glob("candidates-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if general_files:
        return general_files[0]

    console.print(f"[red]No candidate export found in {folder}.[/red]")
    raise typer.Exit(1)


def _fetch_sheet_urls(sheet_url: str, sheet_tab: Optional[str] = None) -> Set[str]:
    """Download Google Sheet CSV from a specific tab and collect existing LinkedIn profile URLs."""
    try:
        parsed = urlsplit(sheet_url)
        if parsed.scheme != "https" or parsed.netloc.lower() != "docs.google.com":
            raise ValueError("expected an https://docs.google.com spreadsheet URL")
        path = parsed.path.split("/edit", 1)[0].rstrip("/")
        query = "tqx=out:csv"
        if sheet_tab:
            query += f"&sheet={quote(sheet_tab, safe='')}"
        csv_url = urlunsplit((parsed.scheme, parsed.netloc, path + "/gviz/tq", query, ""))
        req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8-sig")
        if content.lstrip().lower().startswith(("<!doctype html", "<html")):
            raise ValueError("Google returned an HTML page instead of CSV; check sheet sharing and tab settings")
        rows = list(csv.reader(content.splitlines()))
        if not rows:
            raise ValueError("the spreadsheet returned an empty CSV")
    except (urllib.error.URLError, UnicodeDecodeError, csv.Error, ValueError) as e:
        raise RuntimeError(f"Could not read Google Sheet: {e}") from e

    urls = {
        normalized
        for row in rows
        for cell in row
        if (normalized := _normalize_linkedin_url(cell))
    }
    if not urls:
        raise RuntimeError(
            "Could not find LinkedIn profile URLs in the selected sheet tab; "
            "check the tab or explicitly run with --no-sheet-dedup."
        )
    console.print(f"[green]Loaded {len(urls)} existing candidates from tab '{sheet_tab or 'Default'}'.[/green]")
    return urls


def _normalize_linkedin_url(value: str) -> str:
    match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/([^/?#\s]+)",
        value.strip(),
        flags=re.IGNORECASE,
    )
    return f"linkedin.com/in/{match.group(1).rstrip('/').lower()}" if match else ""


def _load_candidates(file: Path) -> List[Candidate]:
    records = load_records(_read_json(file))
    if not records:
        console.print(f"[red]No candidate records found in {file}; database was not changed.[/red]")
        raise typer.Exit(1)
    candidates = [extract_candidate(record) for record in records]
    usable = [
        candidate for candidate in candidates
        if any((
            candidate.name, candidate.headline, candidate.location, candidate.summary,
            candidate.skills, candidate.positions, candidate.education, candidate.projects,
        ))
    ]
    if not usable:
        console.print(f"[red]No usable candidate profile data found in {file}; database was not changed.[/red]")
        raise typer.Exit(1)
    if len(usable) != len(candidates):
        console.print(f"[yellow]Skipped {len(candidates) - len(usable)} record(s) with no usable profile fields.[/yellow]")
    profile_coverage = coverage(usable)
    console.print(
        f"Extraction coverage: {profile_coverage['candidates']} profiles; "
        f"{profile_coverage['name']} names; {profile_coverage['has positions']} work histories; "
        f"{profile_coverage['profile link']} profile links."
    )
    return usable


def _save_candidates(file: Path, db: str, cands: List[Candidate]) -> List[Candidate]:
    with Store(db) as store:
        count = store.upsert_candidates(cands)
        result = store.candidates_by_ids([c.id for c in cands])
    console.print(f"Ingested [bold]{count}[/bold] candidates from [cyan]{file.name}[/cyan] into {db}")
    return result


def _ingest_file(file: Path, db: str) -> List[Candidate]:
    return _save_candidates(file, db, _load_candidates(file))


@app.command()
def ingest(file: Optional[Path] = typer.Argument(None), db: str = DB_OPT,
           fresh: bool = typer.Option(False, "--fresh", help="Clear the existing DB before ingesting this export")):
    export = find_export(file)
    candidates = _load_candidates(export)
    if fresh:
        db_path = Path(db)
        if db_path.is_file():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup = db_path.with_name(f"{db_path.stem}.backup-{stamp}{db_path.suffix}")
            with Store(db) as store:
                store.backup(str(backup))
                store.purge()
            console.print(f"Existing database backed up to [bold]{backup}[/bold].")
        else:
            with Store(db) as store:
                store.purge()
    _save_candidates(export, db, candidates)


@app.command("filter")
def filter_cmd(rules: str = RULES_OPT, set_: Optional[List[str]] = SET_OPT, db: str = DB_OPT,
               profile: Optional[str] = typer.Option(None, "--profile", help="Named hard-rule profile"),
               show_fails: bool = typer.Option(False, "--show-fails")):
    rs = _rules(rules, set_, profile)
    with Store(db) as store:
        cands = store.all_candidates()
    verdicts = [evaluate(c, rs) for c in cands]
    t = Table(title=f"Funnel ({len(cands)} candidates)")
    for h in ("rule", "rejects alone", "flagged for review", "remaining"):
        t.add_column(h, justify="right" if h != "rule" else "left")
    for r in funnel(cands, rs):
        t.add_row(r["rule"], str(r["rejects_alone"]), str(r["flagged"]), str(r["remaining"]))
    console.print(t)
    out = {o: sum(v.outcome == o for v in verdicts) for o in ("pass", "review", "fail")}
    console.print(f"Pass: [green]{out['pass']}[/green] | Review: [yellow]{out['review']}[/yellow] | Fail: [red]{out['fail']}[/red]")
    if show_fails:
        for c, v in zip(cands, verdicts):
            if v.outcome == "fail":
                console.print(f"- {c.name or c.id}: {'; '.join(v.reasons)}")


def _score_all(rows, job: Job, model: str, embed_model: str, host: str, db: str,
               limit: Optional[int], fast: bool = False):
    survivors = [(c, v) for c, v in rows if v.outcome != "fail"]
    if not survivors:
        return {}

    with Store(db) as store:
        num_ctx = 4096 if model == DEFAULT_MODEL else 2048 if fast else 3072
        client = OllamaClient(model=model, embed_model=embed_model, host=host,
                              num_ctx=num_ctx)
        semantic = limit is not None
        cached = {}
        keys = {}
        sim_scores = {}
        for c, _ in survivors:
            key = cache_key(
                job, candidate_text(c), model, semantic=semantic,
                embed_model=embed_model, num_ctx=num_ctx,
            )
            keys[c.id] = key
            result = store.get_score(c.id, key)
            if result is not None:
                cached[c.id] = result
                sim_scores[c.id] = result.semantic_similarity

        console.print(f"Batch candidates eligible for scoring: {len(survivors)} "
                      f"({len(cached)} cached results found).")
        to_embed = []
        if semantic:
            to_embed = [
                (c, v) for c, v in survivors
                if c.id not in cached or cached[c.id].semantic_similarity is None
            ]
        if to_embed:
            console.print("[cyan]Calculating semantic similarity for candidates without a cached score...[/cyan]")
            with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                          TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
                task = prog.add_task("Embedding candidates", total=len(to_embed) + 1)
                job_embedding = client.embed(job.search_query_text())
                prog.advance(task)
                for c, _ in to_embed:
                    c_emb = client.embed(candidate_text(c, max_desc=400))
                    if len(c_emb) != len(job_embedding):
                        raise RuntimeError(
                            f"Embedding dimension mismatch for candidate {c.id}; "
                            "check the configured embedding model."
                        )
                    sim_scores[c.id] = round(cosine_similarity(job_embedding, c_emb), 3)
                    prog.advance(task)

        ranked = sorted(survivors, key=lambda item: sim_scores.get(item[0].id) or 0.0, reverse=True)
        if limit is not None:
            ranked = ranked[:limit]

        scored = {c.id: cached[c.id] for c, _ in ranked if c.id in cached}
        with Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      TimeElapsedColumn(), TimeRemainingColumn(), console=console) as prog:
            task = prog.add_task(f"Scoring with {model}", total=len(ranked))
            for index, (c, _) in enumerate(ranked, 1):
                prog.update(task, description=f"{model}: candidate {index}/{len(ranked)}")
                if c.id in cached:
                    prog.advance(task)
                    continue
                res = score_candidate(c, job, client, model, sim_score=sim_scores.get(c.id))
                if not res.evaluation_error:
                    store.put_score(c.id, keys[c.id], job.slug, res)
                else:
                    console.print(f"[yellow]Could not validate {model} result for candidate "
                                  f"{index}; it will be marked REVIEW and retried on the next run.[/yellow]")
                scored[c.id] = res
                prog.advance(task)
        return scored


def _score_all_or_exit(*args, **kwargs):
    try:
        return _score_all(*args, **kwargs)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def doctor(job: Path = typer.Option(
               Path("jobs/defi-intents-1010.yaml"), "--job", help="Job YAML to validate"),
           rules: str = RULES_OPT, host: str = typer.Option(DEFAULT_HOST, "--host")):
    """Check the job, rule profile, local Ollama API and configured models."""
    try:
        job_obj = Job.load(str(job))
        ruleset = load_rules(rules, profile=job_obj.rules_profile or None, require_profile=True)
    except (OSError, ValueError) as e:
        console.print(f"[red]Configuration check failed: {e}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Job OK:[/green] {job_obj.slug} (profile {ruleset.profile})")

    try:
        host_name = urlsplit(host).hostname
    except ValueError as e:
        console.print(f"[red]Invalid Ollama host: {e}[/red]")
        raise typer.Exit(1)
    if host_name not in {"localhost", "127.0.0.1", "::1"}:
        console.print(f"[yellow]Ollama host is not local; candidate text may leave this computer: {host}[/yellow]")

    request = urllib.request.Request(host.rstrip("/") + "/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        console.print(f"[red]Ollama API is unavailable at {host}: {e}[/red]")
        raise typer.Exit(1)
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        console.print("[red]Ollama returned an invalid model list.[/red]")
        raise typer.Exit(1)
    installed = {
        item.get("name", "")
        for item in models
        if isinstance(item, dict)
    } if isinstance(payload, dict) else set()
    if not installed:
        console.print("[red]Ollama returned no installed models.[/red]")
        raise typer.Exit(1)

    required = (DEFAULT_MODEL, DEEP_MODEL, FAST_MODEL, DEFAULT_EMBED_MODEL)
    missing = []
    for name in required:
        available = name in installed or (":" not in name and f"{name}:latest" in installed)
        if available:
            console.print(f"[green]Model available:[/green] {name}")
        else:
            console.print(f"[yellow]Model missing:[/yellow] {name} (install with ollama pull {name})")
            missing.append(name)
    if missing:
        raise typer.Exit(1)
    console.print("[bold green]All configured local checks passed.[/bold green]")


def _needs_deep_review(result, job: Job) -> bool:
    if result is None or result.evaluation_error:
        return True
    threshold = job.deep_review_threshold
    if threshold is None:
        threshold = max(0, job.bucket_b - 10)
    if result.score >= threshold:
        return True
    if len(result.keyword_hits) >= job.deep_review_min_keyword_hits:
        return True
    uncertain_mandatory = any(
        requirement.mandatory and requirement.status in {"partial", "unknown", "not_met"}
        for requirement in result.requirements
    )
    return uncertain_mandatory and bool(result.keyword_hits)


@app.command()
def go(job: str = typer.Argument(..., help="Job YAML"),
       file: Optional[Path] = typer.Argument(None),
       rules: str = RULES_OPT, set_: Optional[List[str]] = SET_OPT, db: str = DB_OPT,
       profile: Optional[str] = typer.Option(None, "--profile", help="Override the job's hard-rule profile"),
       model: Optional[str] = MODEL_OPT, embed_model: str = EMBED_OPT,
       host: str = typer.Option(DEFAULT_HOST, "--host"),
       allow_remote_host: bool = typer.Option(
           False, "--allow-remote-host", help="Acknowledge that a remote Ollama server receives candidate text"
       ),
       sheet_url: Optional[str] = SHEET_OPT,
       no_sheet_dedup: bool = typer.Option(
           False, "--no-sheet-dedup", help="Explicitly run without the configured Google Sheets deduplication"
       ),
       limit: Optional[int] = typer.Option(None, help="Score top N candidates"),
       fast: bool = typer.Option(False, "--fast", help="Use the fastest model for a single-stage run"),
       deep: bool = typer.Option(False, "--deep", help="Use the 7B model for every eligible candidate"),
       two_stage: bool = typer.Option(False, "--two-stage", help="Use Qwen 3B, then send selected candidates to Qwen 7B")):
    """ONE command: Ingest -> Filter -> Dedup vs Google Sheet -> Cosine Ranking -> LLM Scoring -> CSV."""
    if sum((fast, deep, two_stage)) > 1:
        console.print("[red]Choose only one of --fast, --deep, or --two-stage.[/red]")
        raise typer.Exit(2)
    try:
        job_obj = Job.load(job)
    except (OSError, ValueError) as e:
        console.print(f"[red]Cannot load job configuration: {e}[/red]")
        raise typer.Exit(1)
    rs = _rules(rules, set_, profile or job_obj.rules_profile or None, require_profile=True)
    if limit is not None and limit < 1:
        console.print("[red]--limit must be a positive integer.[/red]")
        raise typer.Exit(2)
    # Automatically use sheet settings from YAML if not passed via CLI
    effective_sheet_url = sheet_url or getattr(job_obj, "sheet_url", None)
    effective_sheet_tab = getattr(job_obj, "sheet_tab", None)

    try:
        host_name = urlsplit(host).hostname
    except ValueError as e:
        console.print(f"[red]Invalid Ollama host: {e}[/red]")
        raise typer.Exit(2)
    if host_name not in {"localhost", "127.0.0.1", "::1"}:
        if not allow_remote_host:
            console.print("[red]Non-local Ollama hosts can receive candidate profile data. Use a local host "
                          "or explicitly acknowledge this with --allow-remote-host.[/red]")
            raise typer.Exit(2)
        console.print(f"[yellow]Candidate profile text will be sent to remote Ollama host: {host}[/yellow]")

    if effective_sheet_url and no_sheet_dedup:
        console.print("[yellow]Spreadsheet deduplication explicitly disabled for this run.[/yellow]")
        sheet_urls = set()
    elif effective_sheet_url:
        try:
            sheet_urls = _fetch_sheet_urls(effective_sheet_url, effective_sheet_tab)
        except RuntimeError as e:
            console.print(f"[red]{e} Use --no-sheet-dedup only if you intend to proceed without it.[/red]")
            raise typer.Exit(1)
    else:
        sheet_urls = set()
    export = find_export(file)
    cands = _ingest_file(export, db)
    rows = []
    skipped_existing = 0
    missing_profile_urls = 0
    for c in cands:
        clean_url = _normalize_linkedin_url(c.profile_url)
        missing_profile_urls += bool(sheet_urls and not clean_url)
        if clean_url and clean_url in sheet_urls:
            skipped_existing += 1
            continue
        rows.append((c, evaluate(c, rs)))

    if skipped_existing > 0:
        console.print(f"[yellow]Skipped {skipped_existing} candidates already in '{effective_sheet_tab}'.[/yellow]")
    if missing_profile_urls:
        console.print(f"[yellow]{missing_profile_urls} candidate(s) had no LinkedIn profile URL and could not be sheet-deduplicated.[/yellow]")

    automatic_two_stage = two_stage
    selected_model = model or (FAST_MODEL if fast else DEEP_MODEL if deep else DEFAULT_MODEL)
    if automatic_two_stage:
        console.print(f"[cyan]First pass: {selected_model}; eligible candidates will be sent to {DEEP_MODEL} for deep review.[/cyan]")
        scored = _score_all_or_exit(rows, job_obj, selected_model, embed_model, host, db, limit)
        review_rows = [
            (c, v) for c, v in rows
            if c.id in scored and _needs_deep_review(scored[c.id], job_obj)
        ]
        threshold = job_obj.deep_review_threshold
        if threshold is None:
            threshold = max(0, job_obj.bucket_b - 10)
        console.print(f"Deep review: {len(review_rows)} candidate(s); score threshold {threshold}, "
                      f"keyword/evaluation uncertainty may also qualify.")
        if review_rows:
            deep_scores = _score_all_or_exit(review_rows, job_obj, DEEP_MODEL, embed_model, host, db, None)
            scored.update(deep_scores)
    else:
        if fast:
            console.print(f"[cyan]Fast mode: using {selected_model} with reduced context.[/cyan]")
        elif deep:
            console.print(f"[cyan]Deep mode: using {selected_model} for the full batch.[/cyan]")
        else:
            console.print(f"[cyan]Single-stage mode: using {selected_model}.[/cyan]")
        scored = _score_all_or_exit(
            rows, job_obj, selected_model, embed_model, host, db, limit, fast=fast
        )
    full = [(c, v, scored.get(c.id)) for c, v in rows]

    out_csv = f"exports/{job_obj.slug}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    write_csv(full, out_csv, include_failed=True)

    buckets = {b: sum(1 for *_, s in full if s and s.bucket == b) for b in ("A", "B", "C", "REVIEW")}
    console.print(f"\nWrote results to [bold green]{out_csv}[/bold green] "
                  f"(A={buckets['A']} | B={buckets['B']} | C={buckets['C']} | REVIEW={buckets['REVIEW']})")

    t = Table(title="Top Evaluated Candidates")
    for h in ("#", "Bucket", "Score", "Model", "Sim", "Name", "Key Reason"):
        t.add_column(h)
    top = sorted([r for r in full if r[2]], key=lambda r: -r[2].score)[:15]
    for i, (c, _, s) in enumerate(top, 1):
        sim = f"{s.semantic_similarity:.2f}" if s.semantic_similarity else "-"
        t.add_row(str(i), s.bucket, str(s.score), s.model, sim, c.name or c.id, s.reason[:80])
    console.print(t)


@app.command()
def purge(yes: bool = typer.Option(False, "--yes"), db: str = DB_OPT):
    if yes or typer.confirm("Delete all candidates and cached scores?"):
        with Store(db) as store:
            store.purge()
        console.print("Purged database.")


@app.command("candidates")
def candidates_cmd(db: str = DB_OPT, limit: int = typer.Option(50, "--limit", help="Maximum rows to display")):
    """List local candidate IDs and names for review or individual deletion."""
    if limit < 1:
        console.print("[red]--limit must be a positive integer.[/red]")
        raise typer.Exit(2)
    with Store(db) as store:
        candidates = store.all_candidates()
    candidates.sort(key=lambda candidate: (candidate.name.casefold(), candidate.id))
    table = Table(title=f"Local candidates ({len(candidates)} total)")
    table.add_column("ID")
    table.add_column("Name")
    for candidate in candidates[:limit]:
        table.add_row(candidate.id, candidate.name or "-")
    console.print(table)
    if len(candidates) > limit:
        console.print(f"Showing {limit} of {len(candidates)}; increase --limit to see more.")


@app.command()
def forget(candidate_id: str = typer.Argument(..., help="Exact candidate ID to remove"),
           yes: bool = typer.Option(False, "--yes"), db: str = DB_OPT):
    if not yes and not typer.confirm(f"Delete candidate {candidate_id!r} and all of its cached scores?"):
        raise typer.Exit()
    with Store(db) as store:
        deleted = store.delete_candidate(candidate_id)
    if deleted:
        console.print(f"Deleted candidate {candidate_id} and cached scores.")
    else:
        console.print(f"[yellow]Candidate {candidate_id!r} was not found.[/yellow]")
