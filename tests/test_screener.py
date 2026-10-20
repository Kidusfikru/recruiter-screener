import csv
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from screener.extract import extract_candidate, parse_connections, parse_date
from screener.llm import Job, ScoreResult, score_candidate
from screener.models import Candidate, Position
from screener.report import write_csv
from screener.rules import RuleSet, evaluate, rule_min_total_experience
from screener.store import Store


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
            candidate, {"years": 2.9}, date(2022, 1, 1)
        )
        self.assertEqual(check.status, "pass")
        self.assertIn("2.9 years total", check.detail)

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


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    def chat_json(self, system, user):
        return json.dumps(self.response)

    def embed(self, text):
        return []


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
                {"n": 2, "status": "met", "evidence_quote": "production"},
                {"n": 3, "status": "met", "evidence_quote": "Python"},
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


if __name__ == "__main__":
    unittest.main()
