"""Offline replay of the three saved explicit-settings answers.

The fixture beside this file holds the three real glm-5.3-flash answers saved by
the 2026-10-22 opt-in live run, the corrected checker's verdict on each, and that
run's provenance. Replaying them exercises the checker on saved text, not a
model: no provider, sandbox or vendor call happens here, and no preview is
claimed. The saved reports themselves live under the gitignored ``output/`` tree;
this fixture is the durable copy.
"""

import hashlib
import json
import unittest
from pathlib import Path

try:
    from distribute_agent.live_eval import BRIEFS, _check_answer
    from distribute_agent.package import Package
    from distribute_agent.plan import parse_plan
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "this suite needs the installed distribute_agent package: see "
        "src/distribute_agent/README.md (missing: " f"{exc})"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SOURCE = PROJECT_ROOT / "distribute"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "glm-5.3-flash-settings-replay.json"

# _check_answer owns these. "preview executed and matched" is host evidence that
# _preview_evidence produced from sandbox invocations during the live run; saved
# answers cannot reproduce it, so it stays recorded history.
REPLAYED_CHECK_NAMES = (
    "plan valid",
    "seed preserved",
    "width preserved",
    "height preserved",
    "prompt non-empty",
    "prompt keeps the brief",
    "prompt keeps exclusions",
)
PREVIEW_CHECK_NAME = "preview executed and matched"


class HistoricalSettingsReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.package = Package.load(PACKAGE_SOURCE)
        brief_id = cls.fixture["brief"]["id"]
        cls.brief = next(brief for brief in BRIEFS if brief.id == brief_id)

    def test_the_fixture_carries_three_answers_and_their_provenance(self):
        fixture = self.fixture
        self.assertEqual(fixture["kind"], "historical-offline-replay")
        self.assertEqual(len(fixture["runs"]), 3)
        self.assertEqual(fixture["brief"]["expected_settings"],
                         {"seed": 42, "width": 832, "height": 1216})
        provenance = fixture["provenance"]
        self.assertEqual(provenance["additional_model_requests"], 0)
        self.assertEqual(provenance["additional_vendor_jobs"], 0)
        self.assertEqual(provenance["original_status"], "failed")
        self.assertEqual(provenance["evaluation_mode"], "offline-rescore")
        self.assertEqual(provenance["model"]["model_id"], "glm-5.3-flash")
        self.assertEqual(provenance["model"]["max_tokens"], 4096)
        self.assertEqual(provenance["model"]["temperature"], 0.2)
        self.assertFalse(provenance["model"]["automatic_model_retries"])
        self.assertEqual(provenance["bounds"]["max_steps"], 6)
        self.assertEqual(provenance["bounds"]["deadline_seconds"], 180.0)
        self.assertIn("previews were not rerun", provenance["preview_evidence"])
        for key in ("source_report_sha256", "rescored_report_sha256", "checker_sha256"):
            self.assertEqual(len(provenance[key]), 64, key)
            int(provenance[key], 16)
        self.assertEqual(provenance["environment"]["skill_name"], "runpod-oneshot")
        self.assertEqual(provenance["environment"]["smolagents"], "1.26.0")

    def test_the_saved_source_report_matches_the_recorded_hash_when_present(self):
        provenance = self.fixture["provenance"]
        report = PROJECT_ROOT / provenance["source_report_path"]
        if not report.is_file():
            # output/ is gitignored, so a clean checkout has only the fixture. Do
            # not invent a pass: assert the recorded digest is a real sha256.
            self.assertRegex(provenance["source_report_sha256"], r"^[0-9a-f]{64}$")
            return
        self.assertEqual(
            hashlib.sha256(report.read_bytes()).hexdigest(),
            provenance["source_report_sha256"],
        )

    def test_each_saved_answer_replays_through_the_current_checker(self):
        for run in self.fixture["runs"]:
            with self.subTest(repetition=run["repetition"]):
                self.assertEqual(run["original_status"], "fail")
                self.assertEqual(run["rescored_status"], "pass")
                checks = _check_answer(self.brief, run["answer"], self.package)
                self.assertEqual(tuple(check["name"] for check in checks),
                                 REPLAYED_CHECK_NAMES)
                self.assertTrue(all(check["ok"] for check in checks), checks)
                recorded = {
                    check["name"]: check["ok"] for check in run["rescored_checks"]
                }
                self.assertEqual(
                    recorded[PREVIEW_CHECK_NAME], True,
                    "the saved run records matching preview evidence",
                )
                for check in checks:
                    self.assertEqual(recorded[check["name"]], check["ok"], check["name"])

    def test_the_saved_settings_are_the_requested_seed_and_size(self):
        for run in self.fixture["runs"]:
            with self.subTest(repetition=run["repetition"]):
                plan = parse_plan(self.package.manifest, run["answer"])
                self.assertEqual((plan.seed, plan.width, plan.height),
                                 (42, 832, 1216))
                self.assertIn(plan.template, self.package.template_names())

    def test_both_exclusion_branches_are_covered_by_the_saved_answers(self):
        # One answer states the exclusion in the positive prompt, two in the
        # negative prompt; the checker must accept both spellings.
        details = [
            next(check["detail"] for check in run["rescored_checks"]
                 if check["name"] == "prompt keeps exclusions")
            for run in self.fixture["runs"]
        ]
        self.assertIn("positive exclusion=True", details[0])
        self.assertIn("negative exclusion=False", details[0])
        for detail in details[1:]:
            self.assertIn("positive exclusion=False", detail)
            self.assertIn("negative exclusion=True", detail)
        for run in self.fixture["runs"]:
            checks = _check_answer(self.brief, run["answer"], self.package)
            exclusion = next(
                check for check in checks if check["name"] == "prompt keeps exclusions"
            )
            self.assertTrue(exclusion["ok"], exclusion["detail"])


if __name__ == "__main__":
    unittest.main()
