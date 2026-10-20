import csv
import json
import tempfile
import unittest
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError
import typer

from screener.cli import _fetch_sheet_urls, _normalize_linkedin_url, app, _score_all, find_export
from screener.extract import extract_candidate, parse_connections, parse_date
from screener.llm import Job, ScoreResult, cache_key, quote_is_real, score_candidate
from screener.models import Candidate, Position
from screener.report import write_csv
from screener.rules import RuleSet, evaluate, load_rules, rule_min_total_experience
from screener.store import Store
from typer.testing import CliRunner


class ExtractorTests(unittest.TestCase):
    def test_parse_date_supports_recruiter_shapes(self):
        self.assertEqual(parse_date({"year": 2020, "month": 3}), date(2020, 3, 1))
        self.assertEqual(parse_date("Mar 2020"), date(2020, 3, 1))
        self.assertIsNone(parse_date("Present"))
        self.assertIsNone(parse_date("not a date"))

    def test_parse_connections_preserves_capped_values(self):
        self.assertEqual(parse_connections("500+"), (500, True))
        self.assertEqual(parse_connections("1,234"), (1234, False))
        self.assertEqual(parse_connections("unknown"), (None, False))

    def test_extract_candidate_maps_basic_profile(self):
        candidate = extract_candidate({
            "id": "test-profile",
            "raw": {
                "firstName": "Test",
                "lastName": "Candidate",
                "headline": "Protocol engineer",
                "numConnections": "500+",
                "currentPositions": [{
                    "title": "Engineer",
                    "companyName": "Example",
                    "startDateOn": {"year": 2022, "month": 1},
                }],
            },
        })
        self.assertEqual(candidate.id, "test-profile")
        self.assertEqual(candidate.name, "Test Candidate")
        self.assertEqual(candidate.connections, 500)
        self.assertTrue(candidate.connections_capped)
        self.assertEqual(candidate.positions[0].company, "Example")
        self.assertEqual(candidate.positions[0].start, date(2022, 1, 1))

    def test_malformed_grouped_experience_entries_are_skipped(self):
        candidate = extract_candidate({
            "id": "malformed-groups",
            "raw": {"groupedWorkExperience": [
                None,
                {"positions": [None, "bad", {
                    "title": "Engineer",
                    "startDateOn": {"year": 2022, "month": 1},
                }]},
            ]},
        })
        self.assertEqual(len(candidate.positions), 1)
        self.assertEqual(candidate.positions[0].title, "Engineer")


class RuleTests(unittest.TestCase):
    def test_overlapping_positions_are_counted_once(self):
        candidate = Candidate(
            id="experience-test",
            positions=[
                Position(title="Engineer", company="A", start=date(2019, 1, 1), end=date(2020, 12, 1)),
                Position(title="Consultant", company="B", start=date(2020, 1, 1), end=date(2021, 12, 1)),
            ],
        )
        check = rule_min_total_experience(
            candidate, {"years": 3}, date(2022, 1, 1)
        )
        self.assertEqual(check.status, "pass")
        self.assertIn("3.0 years total", check.detail)

    def test_january_through_december_counts_as_one_year(self):
        candidate = Candidate(
            id="calendar-year",
            positions=[Position(
                title="Engineer", company="Example",
                start=date(2020, 1, 1), end=date(2020, 12, 1),
            )],
        )
        check = rule_min_total_experience(candidate, {"years": 1}, date(2021, 1, 1))
        self.assertEqual(check.status, "pass")
        self.assertIn("1.0 years total", check.detail)

    def test_missing_experience_can_be_flagged_for_review(self):
        candidate = Candidate(id="missing-experience")
        rules = RuleSet(
            as_of=date(2024, 1, 1),
            rules={"min_total_experience": {"enabled": True, "years": 3, "on_missing": "flag"}},
        )
        self.assertEqual(evaluate(candidate, rules).outcome, "review")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.temp_dir.name) / "test.db"))

    def tearDown(self):
        self.store.db.close()
        self.temp_dir.cleanup()

    def test_upsert_keeps_same_title_and_company_with_different_start_dates(self):
        first = Candidate(
            id="same-person",
            positions=[Position(title="Engineer", company="Example", start=date(2019, 1, 1))],
        )
        second = Candidate(
            id="same-person",
            positions=[Position(title="Engineer", company="Example", start=date(2022, 1, 1))],
        )
        self.store.upsert_candidates([first])
        self.store.upsert_candidates([second])
        saved = self.store.candidates_by_ids(["same-person"])
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            {position.start for position in saved[0].positions},
            {date(2019, 1, 1), date(2022, 1, 1)},
        )

    def test_score_cache_round_trip(self):
        score = ScoreResult(score=82, bucket="A", model="test-model")
        self.store.put_score("candidate", "cache-key", "test-job", score)
        cached = self.store.get_score("candidate", "cache-key")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.score, 82)
        self.assertEqual(cached.model, "test-model")

    def test_merge_updates_zero_and_profile_url_provenance(self):
        first = Candidate(
            id="same-person", connections=500, connections_capped=True,
            profile_url="https://www.linkedin.com/talent/profile/abc",
            profile_url_is_guess=True,
        )
        update = Candidate(
            id="same-person", connections=0, connections_capped=False,
            profile_url="https://www.linkedin.com/in/example",
            profile_url_is_guess=False,
        )
        self.store.upsert_candidates([first])
        self.store.upsert_candidates([update])
        saved = self.store.candidates_by_ids(["same-person"])[0]
        self.assertEqual(saved.connections, 0)
        self.assertFalse(saved.connections_capped)
        self.assertEqual(saved.profile_url, update.profile_url)
        self.assertFalse(saved.profile_url_is_guess)

    def test_forget_removes_candidate_and_cached_scores(self):
        self.store.upsert_candidates([Candidate(id="forget-me")])
        self.store.put_score("forget-me", "key", "job", ScoreResult(score=50))
        self.assertTrue(self.store.delete_candidate("forget-me"))
        self.assertEqual(self.store.candidates_by_ids(["forget-me"]), [])
        self.assertIsNone(self.store.get_score("forget-me", "key"))


class ImportSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.db = root / "candidates.db"
        with Store(str(self.db)) as store:
            store.upsert_candidates([Candidate(id="keep-me")])
        self.runner = CliRunner()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fresh_import_does_not_clear_db_for_invalid_json(self):
        invalid = Path(self.temp_dir.name) / "invalid.json"
        invalid.write_text("[", encoding="utf-8")
        result = self.runner.invoke(
            app, ["ingest", str(invalid), "--fresh", "--db", str(self.db)]
        )
        self.assertNotEqual(result.exit_code, 0)
        with Store(str(self.db)) as store:
            self.assertEqual([c.id for c in store.all_candidates()], ["keep-me"])
        self.assertEqual(list(Path(self.temp_dir.name).glob("*.backup-*.db")), [])

    def test_fresh_import_does_not_clear_db_for_empty_export(self):
        empty = Path(self.temp_dir.name) / "empty.json"
        empty.write_text('{"candidates": []}', encoding="utf-8")
        result = self.runner.invoke(
            app, ["ingest", str(empty), "--fresh", "--db", str(self.db)]
        )
        self.assertNotEqual(result.exit_code, 0)
        with Store(str(self.db)) as store:
            self.assertEqual([c.id for c in store.all_candidates()], ["keep-me"])

    def test_fresh_import_does_not_clear_db_for_unusable_profile_data(self):
        empty_profile = Path(self.temp_dir.name) / "empty-profile.json"
        empty_profile.write_text('{"candidates": [{"id": "empty"}]}', encoding="utf-8")
        result = self.runner.invoke(
            app, ["ingest", str(empty_profile), "--fresh", "--db", str(self.db)]
        )
        self.assertNotEqual(result.exit_code, 0)
        with Store(str(self.db)) as store:
            self.assertEqual([c.id for c in store.all_candidates()], ["keep-me"])

    def test_fresh_import_creates_backup_before_replacement(self):
        export = Path(self.temp_dir.name) / "valid.json"
        export.write_text(
            json.dumps({"candidates": [{"id": "new-person", "raw": {"headline": "Engineer"}}]}),
            encoding="utf-8",
        )
        result = self.runner.invoke(
            app, ["ingest", str(export), "--fresh", "--db", str(self.db)]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        backups = list(Path(self.temp_dir.name).glob("*.backup-*.db"))
        self.assertEqual(len(backups), 1)
        with Store(str(backups[0])) as backup:
            self.assertEqual([c.id for c in backup.all_candidates()], ["keep-me"])
        with Store(str(self.db)) as store:
            self.assertEqual([c.id for c in store.all_candidates()], ["new-person"])

    def test_sheet_url_normalization_ignores_query_and_host_case(self):
        self.assertEqual(
            _normalize_linkedin_url("https://WWW.LinkedIn.com/in/Example/?trk=profile"),
            "linkedin.com/in/example",
        )

    def test_missing_explicit_export_does_not_fall_back_to_newest_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fallback = Path(temp_dir) / "candidates-detail-newest.json"
            fallback.write_text("[]", encoding="utf-8")
            with self.assertRaises(typer.Exit):
                find_export(Path(temp_dir) / "typo.json", folder=Path(temp_dir))

    def test_sheet_fetch_parses_and_normalizes_urls(self):
        response = BytesIO(b"Name,Profile\nExample,https://www.linkedin.com/in/Example/?trk=foo\n")
        with patch("screener.cli.urllib.request.urlopen", return_value=response) as open_url:
            urls = _fetch_sheet_urls(
                "https://docs.google.com/spreadsheets/d/example/edit?gid=123",
                "Example tab",
            )
        self.assertEqual(urls, {"linkedin.com/in/example"})
        self.assertIn("sheet=Example%20tab", open_url.call_args.args[0].full_url)

    def test_sheet_fetch_rejects_html_and_unusable_tabs(self):
        html = BytesIO(b"<!doctype html><html>Sign in</html>")
        with patch("screener.cli.urllib.request.urlopen", return_value=html):
            with self.assertRaisesRegex(RuntimeError, "HTML page"):
                _fetch_sheet_urls("https://docs.google.com/spreadsheets/d/example/edit")
        no_links = BytesIO(b"Name,Email\nExample,example@example.test\n")
        with patch("screener.cli.urllib.request.urlopen", return_value=no_links):
            with self.assertRaisesRegex(RuntimeError, "LinkedIn profile URLs"):
                _fetch_sheet_urls("https://docs.google.com/spreadsheets/d/example/edit")


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def chat_json(self, _system, _user):
        return json.dumps(self.response)

    def embed(self, _text):
        return []


class FailingEmbeddingClient:
    def __init__(self, **_kwargs):
        pass

    def embed(self, _text):
        raise RuntimeError("embedding service unavailable")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.job = Job(
            slug="test-job",
            title="Protocol Engineer",
            must_have=["Protocol engineering experience (MANDATORY)", "Production experience"],
            nice_to_have=["Python experience"],
        )
        self.candidate = Candidate(
            id="scoring-test",
            headline="Protocol engineer",
            summary="Built protocol engineering systems for production using Python.",
        )

    def test_valid_model_response_produces_scored_result(self):
        client = FakeLLMClient({
            "requirements": [
                {"n": 1, "status": "met", "evidence_quote": "Protocol engineer"},
                {"n": 2, "status": "met", "evidence_quote": "systems for production"},
                {"n": 3, "status": "met", "evidence_quote": "using Python"},
            ],
            "reason": "Relevant experience.",
        })
        result = score_candidate(self.candidate, self.job, client, model="fake")
        self.assertEqual(result.bucket, "A")
        self.assertEqual(result.score, 100)
        self.assertFalse(result.evaluation_error)

    def test_incomplete_model_response_is_marked_for_review(self):
        client = FakeLLMClient({
            "requirements": [{"n": 1, "status": "unknown", "evidence_quote": ""}],
        })
        result = score_candidate(self.candidate, self.job, client, model="fake")
        self.assertEqual(result.bucket, "REVIEW")
        self.assertIn("missing requirement judgments", result.evaluation_error)

    def test_mandatory_met_without_verifiable_evidence_is_unknown(self):
        client = FakeLLMClient({
            "requirements": [
                {"n": 1, "status": "met", "evidence_quote": ""},
                {"n": 2, "status": "met", "evidence_quote": "systems for production"},
                {"n": 3, "status": "met", "evidence_quote": "using Python"},
            ],
        })
        result = score_candidate(self.candidate, self.job, client, model="fake")
        self.assertEqual(result.requirements[0].status, "unknown")
        self.assertEqual(result.score, 40)
        self.assertEqual(result.bucket, "REVIEW")
        self.assertIn("unsupported evidence quote", result.evaluation_error)

    def test_fabricated_quote_does_not_support_met(self):
        client = FakeLLMClient({
            "requirements": [
                {"n": 1, "status": "met", "evidence_quote": "Protocol engineering production experience"},
                {"n": 2, "status": "met", "evidence_quote": "systems for production"},
                {"n": 3, "status": "met", "evidence_quote": "using Python"},
            ],
        })
        result = score_candidate(self.candidate, self.job, client, model="fake")
        self.assertEqual(result.requirements[0].status, "unknown")
        self.assertEqual(result.bucket, "REVIEW")

    def test_evidence_quote_must_be_a_substantive_phrase(self):
        self.assertFalse(quote_is_real("the", "the candidate built systems"))
        self.assertTrue(quote_is_real("built systems", "The candidate built systems."))

    def test_semantic_and_rubric_cache_keys_are_separate(self):
        text = "candidate profile"
        plain = cache_key(self.job, text, "fake")
        semantic = cache_key(self.job, text, "fake", semantic=True, embed_model="embed-a")
        other_embedder = cache_key(self.job, text, "fake", semantic=True, embed_model="embed-b")
        other_context = cache_key(self.job, text, "fake", num_ctx=4096)
        self.assertNotEqual(plain, semantic)
        self.assertNotEqual(semantic, other_embedder)
        self.assertNotEqual(plain, other_context)

    def test_embedding_failure_is_explicit(self):
        candidate = Candidate(id="embedding-test")
        verdict = evaluate(candidate, RuleSet(as_of=date(2024, 1, 1), rules={}))
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("screener.cli.OllamaClient", FailingEmbeddingClient):
                with self.assertRaisesRegex(RuntimeError, "embedding service unavailable"):
                    _score_all(
                        [(candidate, verdict)], self.job, "fake", "embed", "http://localhost:11434",
                        str(Path(temp_dir) / "scores.db"), limit=1,
                    )

    def test_job_rejects_unsafe_slug_and_invalid_bucket_order(self):
        with self.assertRaises(ValidationError):
            Job(slug="../output", title="Test", must_have=["Requirement"])
        with self.assertRaises(ValidationError):
            Job(slug="test", title="Test", must_have=["Requirement"], bucket_a=40, bucket_b=70)

    def test_rules_reject_unknown_rule_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.yaml"
            rules_path.write_text(
                "rules:\n  min_connections:\n    enabled: true\n    threshhold: 30\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown setting"):
                load_rules(str(rules_path))

    def test_report_writes_candidate_and_score(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.csv"
            candidate = Candidate(id="report-test", name="Example Candidate")
            verdict = evaluate(candidate, RuleSet(as_of=date(2024, 1, 1), rules={}))
            score = ScoreResult(score=70, bucket="A", model="fake")
            self.assertEqual(write_csv([(candidate, verdict, score)], str(path)), 1)
            with path.open(encoding="utf-8-sig", newline="") as report:
                rows = list(csv.DictReader(report))
            self.assertEqual(rows[0]["name"], "Example Candidate")
            self.assertEqual(rows[0]["score"], "70")

    def test_report_neutralizes_spreadsheet_formula_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.csv"
            candidate = Candidate(id="formula", name=" =1+1")
            verdict = evaluate(candidate, RuleSet(as_of=date(2024, 1, 1), rules={}))
            score = ScoreResult(score=70, bucket="A", reason="=HYPERLINK(\"x\")")
            write_csv([(candidate, verdict, score)], str(path))
            with path.open(encoding="utf-8-sig", newline="") as report:
                row = next(csv.DictReader(report))
            self.assertEqual(row["name"], "' =1+1")
            self.assertEqual(row["reason"], "'=HYPERLINK(\"x\")")


if __name__ == "__main__":
    unittest.main()
