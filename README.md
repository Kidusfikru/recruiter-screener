# Recruiter Screener

A local candidate-screening tool for importing Recruiter JSON exports, applying configurable hard rules, deduplicating against a Google Sheet, scoring against a job-specific rubric with Ollama, and exporting a ranked CSV.

## Quick start (Windows PowerShell)

Run commands from the project directory:

```powershell
# Create an isolated environment and install the versions tested with this copy.
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Check that the CLI starts.
python -m screener --help
```

`requirements.txt` pins the working local setup, including transitive packages.
To update a package, change the pins deliberately, run the test suite, and verify
one small scoring run before relying on the updated environment.

Ollama is a separate application and must be installed and running locally. Pull the default first-pass model, deep-review model, fast-only model, and embedding model if they are not already present:

```powershell
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull llama3.2
ollama pull nomic-embed-text
```

Screen the newest export in your Downloads folder using the default Qwen 3B model:

```powershell
python -m screener go jobs\defi-intents-1010.yaml
```

Use a specific export file instead:

```powershell
python -m screener go jobs\defi-intents-1010.yaml "C:\Users\you\Downloads\candidates-detail-export.json"
```

Optionally run Qwen 3B first and send selected candidates to Qwen 7B:

```powershell
python -m screener go jobs\defi-intents-1010.yaml --two-stage
```

Results are written to `exports\<job-slug>.csv`.

### Personal-use data safety

Candidate profiles and screening results can contain personal information. The
local database and generated CSV/JSON files are excluded by `.gitignore`; keep
them on a trusted device and do not share them unintentionally. Back up
`data\screener.db` periodically while the screener is closed, and store the
backup somewhere protected. Ollama runs locally by default; do not change its
host to a remote service unless you have reviewed that service's privacy terms.

### Quick checks

Run the offline tests after changing code or dependencies:

```powershell
python -m unittest discover -v
python -m compileall -q screener tests
```

These tests use synthetic profiles and do not contact Ollama or Google Sheets.
Before a real screening batch, confirm Ollama is running and inspect the CSV,
especially rows marked `REVIEW` or `evaluation_error`.

## New position setup

The system separates **job scoring criteria** from **hard screening rules**:

1. Create a job file in `jobs\`, usually by copying `jobs\defi-intents-1010.yaml`.
2. Give it a unique `slug` and update its title, description, requirements, and evidence terms.
3. Add a hard-rule profile under `profiles` in `config\rules.yaml`. Give each profile a clear name and set its company/position metadata.
4. Set the job's `rules_profile` to that profile name.
5. Set the correct `sheet_url` and exact `sheet_tab` in the job file.
6. Run `go` with the new job YAML and the new candidate export.

For example, a second job file might contain:

```yaml
slug: backend-engineer-example
title: Backend Engineer, Example Co
rules_profile: example-co-backend
sheet_url: "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
sheet_tab: "Example Co — Backend"
deep_review_threshold: 35
deep_review_min_keyword_hits: 2

description: |
  Describe the role, responsibilities, work arrangement, location/time-zone needs,
  and other relevant context.

must_have:
  - Production backend engineering experience (MANDATORY)
  - Experience with the required technology

nice_to_have:
  - Experience in the company's industry

evidence_terms:
  - example technology
  - domain term

bucket_a: 70
bucket_b: 45
```

Add the matching profile to `config\rules.yaml`:

```yaml
profiles:
  "example-co-backend":
    company: "Example Co"
    position: "backend-engineer-example"
    rules:
      min_connections:
        enabled: true
        threshold: 20
        on_missing: flag
      min_total_experience:
        enabled: true
        years: 2
        on_missing: flag
```

The existing `"1010"` profile is the saved 1010.trading hard-rule configuration. Avoid changing it when setting criteria for another role; create a new profile instead. Shared values can go under `defaults.rules`, with profile-specific values overriding them.

### What can be configured without code changes?

- **Job description and role requirements:** `description`, `must_have`, and `nice_to_have` in a job YAML.
- **Mandatory scoring caps:** Put the word `MANDATORY` in a `must_have` item. This is an LLM scoring cap, not a hard-rule rejection: unsupported mandatory evidence caps the score. It does not automatically remove the candidate before scoring.
- **Evidence hints:** `evidence_terms` are included in the scoring prompt and reported as keyword hits.
- **Score buckets:** `bucket_a` and `bucket_b`.
- **Existing hard-rule settings:** Enable/disable supported rules and change their thresholds in each profile.
- **Spreadsheet deduplication:** `sheet_url` and `sheet_tab` in each job YAML.
- **Per-run hard-rule overrides:** Use `--set rule.key=value`.

Supported hard-rule evaluator names are currently `min_connections`, `job_hopper`, `min_current_tenure`, `min_total_experience`, and `timezone_preference`. A new threshold/profile using one of these is configuration-only. A genuinely new kind of hard rule needs a Python evaluator and registration in `screener\rules.py`; adding a new YAML name alone does not implement new behavior.

## Hard-rule behavior

Each enabled rule returns pass, fail, or missing. A failed rule excludes a candidate from LLM scoring. Missing data is handled by `on_missing`:

- `flag` (default): keep the candidate and mark the result for review.
- `pass`: treat missing data as passed.
- `fail`: reject the candidate when the data is absent.

Rules currently available:

- `min_connections`: minimum connection count; a capped count such as `500+` passes when the threshold is met.
- `job_hopper`: checks short job stints. It can merge same-company roles, ignore current roles, and ignore configured title keywords.
- `min_current_tenure`: checks the longest current-company stint. Same-company roles separated by up to about 62 days can be merged; month counts are inclusive.
- `min_total_experience`: totals dated work intervals, counting overlapping dates only once.
- `timezone_preference`: checks whether the candidate's location contains one of the configured `preferred_regions`; a non-match is missing and follows `on_missing`.

For 1010, the current profile sets 30 minimum connections, disables `job_hopper`, requires 6 months in a current role, and 3 years of total experience. The 1010 job's UTC+2 to UTC+4 preference is currently a **nice-to-have LLM criterion**, not an enabled hard timezone rule.

Inspect the hard-rule funnel:

```powershell
python -m screener filter --profile 1010
python -m screener filter --profile 1010 --show-fails
```

Override a rule for one run:

```powershell
python -m screener go jobs\defi-intents-1010.yaml --set min_current_tenure.months=3
```

## Candidate batches, database, and cached scores

The default database is `data\screener.db`. It stores extracted candidate profiles and cached scores across runs.

**`go` screens only candidates in the export selected for that run.** It does not load every old candidate from the database. The database is used to retain candidate records and look up reusable scores for candidates in the current batch.

When no export path is supplied, the program searches the user's Downloads folder, preferring the newest `candidates-detail-*.json` file, then the newest `candidates-*.json`. For repeatable runs, pass the intended export file explicitly.

Candidate identity is based on an extracted Recruiter profile identifier when available, otherwise a hash of the raw profile. Re-importing updates/merges the stored profile. A cached score is reusable when the candidate text, job criteria, and selected model produce the same cache key. Changing a profile, job criteria, or model normally requires a new score.

Use a separate database when you intentionally want isolated history:

```powershell
python -m screener go jobs\defi-intents-1010.yaml --db data\1010-test.db
```

`--fresh` is available for `ingest` only and clears **both candidates and cached scores** in the selected database before ingesting. Do not use it if you want to retain previous score-cache entries:

```powershell
python -m screener ingest "C:\path\to\candidates.json" --fresh
```

`purge` also deletes all candidates and scores for the selected database:

```powershell
python -m screener purge --db data\screener.db
```

It asks for confirmation unless `--yes` is supplied. `filter` reads all candidates in the selected database; unlike `go`, it is not limited to a particular export.

## Google Sheets deduplication

Each job may specify:

```yaml
sheet_url: "https://docs.google.com/spreadsheets/d/..."
sheet_tab: "Exact tab name"
```

The `go` command fetches CSV data from the spreadsheet tab and scans every cell for `linkedin.com/in/` URLs. It normalizes them to lowercase and removes trailing slashes, then skips matching candidate URLs. The `--sheet-url` option can override the job URL; the tab name continues to come from the job YAML.

Important limitations:

- The sheet needs to be accessible to the screener's unauthenticated CSV download. This code does not sign in to Google.
- The screener only **reads** the sheet for deduplication. It does not add or update rows in Google Sheets.
- Only URLs containing `linkedin.com/in/` are collected. Guessed Talent profile links and other URL formats may not match.
- If fetching or parsing the sheet fails, the program warns and proceeds **without deduplication**. Check the warning before relying on a run to exclude already-contacted people.
- If `sheet_tab` is omitted, the default sheet tab is used.

## Default, two-stage, deep, and fast scoring

The default is a single `qwen2.5:3b-instruct` scoring pass. Use `--two-stage` to run Qwen 3B first and then `qwen2.5:7b-instruct` on selected candidates. Use `--deep` to run Qwen 7B on every candidate that passes hard rules, or `--fast` to run Llama 3.2 alone. Model defaults can be changed with `RL_MODEL`, `RL_DEEP_MODEL`, and `RL_FAST_MODEL`.

```powershell
# Default: Qwen 3B only (fast single-pass shortlist).
python -m screener go jobs\defi-intents-1010.yaml

# Optional: Qwen 3B first, then Qwen 7B on selected candidates.
python -m screener go jobs\defi-intents-1010.yaml --two-stage

# Run Qwen 7B on every candidate that passes hard rules.
python -m screener go jobs\defi-intents-1010.yaml --deep

# Run the faster Llama model alone; no Qwen 7B follow-up.
python -m screener go jobs\defi-intents-1010.yaml --fast

# Run one explicit model only.
python -m screener go jobs\defi-intents-1010.yaml --model qwen2.5:3b-instruct
```

Only one of `--fast`, `--deep`, and `--two-stage` may be specified. `--model` selects one model unless `--two-stage` is set, in which case it replaces the first-pass model.

The 1010 job sets `deep_review_threshold: 35` (10 below its B-bucket cutoff of 45) and `deep_review_min_keyword_hits: 2`. When `--two-stage` is used, Qwen 3B results go to Qwen 7B if they meet the threshold, contain enough exact evidence-term hits, have uncertain mandatory evidence plus at least one evidence hit, or have invalid/incomplete output. These settings are per job and should be tuned when a role has different score buckets or evidence terms. A Qwen 3B score below the deep threshold is **not** a hard rejection: every candidate that passes hard rules remains in the CSV.

Full-batch scoring skips semantic embeddings; use `--limit` to embed and rank the eligible batch before scoring the top N. In that limited workflow, only the top N scored candidates and hard-rule failures appear in the CSV.

### Small local model comparison

On six profiles with cached Qwen 7B evaluations, schema-constrained Qwen 3B took about 5–7 seconds per response with a 4,096-token context and matched the 7B bucket on only one profile. It assigned higher buckets to some profiles rated C by 7B. Under the conservative 1010 escalation rules, all six profiles were selected for deep review. This small local comparison is not a general benchmark, but it shows why the two-stage mode is opt-in: with these settings it did not reduce the number of 7B calls, and Qwen 3B alone should be treated as a quick shortlist, not an equivalent replacement for Qwen 7B. Long profiles can still take longer.

The CSV's `model` column identifies which model produced each final score. Malformed or incomplete outputs are marked `REVIEW` in the bucket and described in `evaluation_error`; they are not cached as valid evaluations. Review borderline candidates and rows with evaluation errors.

Ollama must be running locally (default host `http://localhost:11434`). The embedding model defaults to `nomic-embed-text`; override it with `--embed-model` or `RL_EMBED_MODEL`. Use `ollama list` to view installed models.

Use **Ctrl+C** in the terminal to interrupt a run. Completed first-pass and deep scores remain cached. Rerun the same export and job settings to reuse matching scores. Cache entries are model- and prompt-version-specific; changing a model or scoring prompt causes that model to score again.

## Commands

```powershell
# Import a JSON export into the DB without scoring.
python -m screener ingest "C:\path\to\candidates.json"

# Import into a cleared DB (also deletes cached scores).
python -m screener ingest "C:\path\to\candidates.json" --fresh

# Show the hard-rule funnel for all DB candidates.
python -m screener filter --profile 1010

# Run full screening for only the provided export.
python -m screener go jobs\defi-intents-1010.yaml "C:\path\to\candidates.json"

# Run 7B on all hard-rule-eligible candidates.
python -m screener go jobs\defi-intents-1010.yaml "C:\path\to\candidates.json" --deep

# Run a Qwen 3B first pass and selectively review with 7B.
python -m screener go jobs\defi-intents-1010.yaml "C:\path\to\candidates.json" --two-stage

# Use the fastest model only for this batch.
python -m screener go jobs\defi-intents-1010.yaml "C:\path\to\candidates.json" --fast

# Score at most the top 20 candidates after embedding-based ranking.
python -m screener go jobs\defi-intents-1010.yaml "C:\path\to\candidates.json" --limit 20

# Delete all stored candidates and scores after confirmation.
python -m screener purge
```

Run `python -m screener <command> --help` for the current command options.

## Output CSV

The output is `exports\<slug>.csv`, encoded with a UTF-8 BOM for spreadsheet compatibility. It includes rank, bucket, score, candidate name and profile URL, current role, location, hard-rule review status, model name, evaluation errors, mandatory requirement status, reason, red flags, keyword hits, score cap, and requirement-level details. Hard-rule failures are included at the end when running `go`; other eligible candidates are listed when they receive a score.

Candidate names and profile URLs are used in the local CSV report, not in the candidate profile text sent to the local LLM prompt.

## Project layout

```text
config\rules.yaml             Shared defaults and named hard-rule profiles
jobs\<job>.yaml               Role description, rubric, sheet settings, profile selection
screener\cli.py               CLI workflow and batch selection
screener\rules.py             Hard-rule evaluators and profile loading
screener\llm.py               Job schema, Ollama integration, scoring and cache key
screener\extract.py           Recruiter JSON extraction
screener\store.py             SQLite candidate and score cache
screener\report.py            CSV output
data\screener.db              Local persistent profile and score store
exports\                      Screening reports
```

`screener\explore.py` can summarize candidate-export field paths and field coverage without printing personal values, which is useful when an export's JSON structure changes.
