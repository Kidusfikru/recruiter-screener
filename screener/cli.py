"""Command line: ingest -> filter -> Google Sheets dedup -> score -> export."""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path
from typing import List, Optional, Set

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

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
import urllib.parse

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
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
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
    urls = set()
    try:
        csv_url = sheet_url
        if "/edit" in sheet_url:
            csv_url = sheet_url.split("/edit")[0] + "/gviz/tq?tqx=out:csv"

        # Target the specific tab if provided
        if sheet_tab:
            csv_url += f"&sheet={urllib.parse.quote(sheet_tab)}"

        req = urllib.request.Request(csv_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            lines = resp.read().decode("utf-8-sig").splitlines()
            reader = csv.reader(lines)
            for row in reader:
                for cell in row:
                    cell_clean = cell.strip().lower()
                    if "linkedin.com/in/" in cell_clean:
                        urls.add(cell_clean.rstrip("/"))
        console.print(f"[green]Loaded {len(urls)} existing candidates from tab '{sheet_tab or 'Default'}'.[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not fetch Google Sheet ({e}). Proceeding without deduplication.[/yellow]")
    return urls

def _ingest_file(file: Path, db: str) -> List[Candidate]:
    records = load_records(_read_json(file))
    cands = [extract_candidate(r) for r in records]
    store = Store(db)
    n = store.upsert_candidates(cands)
    console.print(f"Ingested [bold]{n}[/bold] candidates from [cyan]{file.name}[/cyan] into {db}")
    return store.candidates_by_ids([c.id for c in cands])


@app.command()
def ingest(file: Optional[Path] = typer.Argument(None), db: str = DB_OPT,
           fresh: bool = typer.Option(False, "--fresh", help="Clear the existing DB before ingesting this export")):
    if fresh:
        Store(db).purge()
    _ingest_file(find_export(file), db)


@app.command("filter")
def filter_cmd(rules: str = RULES_OPT, set_: Optional[List[str]] = SET_OPT, db: str = DB_OPT,
               profile: Optional[str] = typer.Option(None, "--profile", help="Named hard-rule profile"),
               show_fails: bool = typer.Option(False, "--show-fails")):
    rs, cands = _rules(rules, set_, profile), Store(db).all_candidates()
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
    store = Store(db)
    client = OllamaClient(model=model, embed_model=embed_model, host=host,
                          num_ctx=4096 if model == DEFAULT_MODEL else 2048 if fast else 3072)

    survivors = [(c, v) for c, v in rows if v.outcome != "fail"]
    if not survivors:
        return {}

    cached = {}
    keys = {}
    sim_scores = {}
    for c, _ in survivors:
        key = cache_key(job, candidate_text(c), model)
        keys[c.id] = key
        result = store.get_score(c.id, key)
        if result is not None:
            cached[c.id] = result
            sim_scores[c.id] = result.semantic_similarity

    console.print(f"Batch candidates eligible for scoring: {len(survivors)} "
                  f"({len(cached)} cached results found).")
    to_embed = []
    if limit is not None:
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
                if job_embedding:
                    c_emb = client.embed(candidate_text(c, max_desc=400))
                    sim_scores[c.id] = round(cosine_similarity(job_embedding, c_emb), 3)
                else:
                    sim_scores[c.id] = None
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
       sheet_url: Optional[str] = SHEET_OPT,
       limit: Optional[int] = typer.Option(None, help="Score top N candidates"),
       fast: bool = typer.Option(False, "--fast", help="Use the fastest model for a single-stage run"),
       deep: bool = typer.Option(False, "--deep", help="Use the 7B model for every eligible candidate"),
       two_stage: bool = typer.Option(False, "--two-stage", help="Use Qwen 3B, then send selected candidates to Qwen 7B")):
    """ONE command: Ingest -> Filter -> Dedup vs Google Sheet -> Cosine Ranking -> LLM Scoring -> CSV."""
    if sum((fast, deep, two_stage)) > 1:
        console.print("[red]Choose only one of --fast, --deep, or --two-stage.[/red]")
        raise typer.Exit(2)
    job_obj = Job.load(job)
    rs = _rules(rules, set_, profile or job_obj.rules_profile or None, require_profile=True)
    cands = _ingest_file(find_export(file), db)

    # Automatically use sheet settings from YAML if not passed via CLI
    effective_sheet_url = sheet_url or getattr(job_obj, "sheet_url", None)
    effective_sheet_tab = getattr(job_obj, "sheet_tab", None)

    sheet_urls = _fetch_sheet_urls(effective_sheet_url, effective_sheet_tab) if effective_sheet_url else set()
    rows = []
    skipped_existing = 0
    for c in cands:
        clean_url = (c.profile_url or "").strip().lower().rstrip("/")
        if sheet_urls and clean_url in sheet_urls:
            skipped_existing += 1
            continue
        rows.append((c, evaluate(c, rs)))

    if skipped_existing > 0:
        console.print(f"[yellow]Skipped {skipped_existing} candidates already in '{effective_sheet_tab}'.[/yellow]")

    automatic_two_stage = two_stage
    selected_model = model or (FAST_MODEL if fast else DEEP_MODEL if deep else DEFAULT_MODEL)
    if automatic_two_stage:
        console.print(f"[cyan]First pass: {selected_model}; eligible candidates will be sent to {DEEP_MODEL} for deep review.[/cyan]")
        scored = _score_all(rows, job_obj, selected_model, embed_model, host, db, limit)
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
            deep_scores = _score_all(review_rows, job_obj, DEEP_MODEL, embed_model, host, db, None)
            scored.update(deep_scores)
    else:
        if fast:
            console.print(f"[cyan]Fast mode: using {selected_model} with reduced context.[/cyan]")
        elif deep:
            console.print(f"[cyan]Deep mode: using {selected_model} for the full batch.[/cyan]")
        else:
            console.print(f"[cyan]Single-stage mode: using {selected_model}.[/cyan]")
        scored = _score_all(rows, job_obj, selected_model, embed_model, host, db, limit, fast=fast)
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
        Store(db).purge()
        console.print("Purged database.")
