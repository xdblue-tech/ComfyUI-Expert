"""Tests for scripts/skill_eval.py — the agent-agnostic skill eval runner.

No test invokes a real agent or touches the network: agent runs use the stub at
tests/fixtures/fake_agent.py, and every path that writes files is pointed at a
temporary directory.
"""

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import skill_eval as se  # noqa: E402  # pyright: ignore[reportMissingImports]

STUB_AGENT = REPO_ROOT / "tests" / "fixtures" / "fake_agent.py"
STUB_AGENT_CMD = f'"{sys.executable}" "{STUB_AGENT}"'

# skill name -> (case count, assertion count) for the four data files.
REAL_SKILLS = {
    "comfyui-character-gen": (7, 28),
    "comfyui-prompt-interview": (7, 28),
    "comfyui-video-production": (7, 29),
    "comfyui-workflow-builder": (7, 31),
}

STUB_TEST_CASES = """\
- id: TC-001
  name: stub-case-one
  prompt: "Which model should I target?"
  assertions:
    - type: contains
      target: "Which model"
      critical: true
    - type: not_contains
      target: "banana"
      critical: false
    - type: contains
      target: "definitely-absent"
      critical: true
    - type: structure_check
      target: "Looks structured"
      critical: false
    - type: judge_me
      target: "not a real type"
      critical: false
  expected_behavior: >
    Stub case used to exercise the scored path.
  edge_case: false

- id: TC-002
  name: stub-case-two
  prompt: >-
    Which model should I target for
    a square 1024 render?
  assertions:
    - type: question_before_code
      critical: false
    - type: json_valid
      critical: true
    - type: json_fields
      target: "method,cfg"
      critical: true
  expected_behavior: "Stub case proving folded prompts reach the agent."
  edge_case: false
"""

STUB_CASE_ONE_SCORE = "2/5 (3 anchored, 1 soft, 1 skipped) [CRITICAL FAIL]"
STUB_CASE_TWO_SCORE = "3/3 (3 anchored, 0 soft, 0 skipped)"
# Only used as a value in argv assertions; the stub never reads it in those tests.
PROMPT_FILE = Path("eval-prompt.txt")


def write_stub_skill(root: Path) -> Path:
    """Create a throwaway skill directory with a stub test-cases.yaml."""

    skill_dir = root / "stub-skill"
    (skill_dir / "eval").mkdir(parents=True)
    (skill_dir / "eval" / "test-cases.yaml").write_text(
        STUB_TEST_CASES, encoding="utf-8"
    )
    return skill_dir


def run_main(argv, env=None):
    """Run skill_eval.main() with captured streams and a controlled environment."""

    stdout = io.StringIO()
    stderr = io.StringIO()
    with (
        mock.patch.dict(os.environ, env or {}, clear=False),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        code = se.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class YamlSubsetParserTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-cases.yaml"
            path.write_text(text, encoding="utf-8")
            return se.parse_yaml_subset(path)

    def parse_error(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test-cases.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(se.YamlSubsetError) as caught:
                se.parse_yaml_subset(path)
            message = str(caught.exception)
            self.assertIn("test-cases.yaml:", message)
            return message

    def test_supported_scalar_styles(self):
        cases = self.parse(
            "\n".join(
                [
                    '- id: TC-001',
                    '  name: plain-name',
                    '  prompt: "double quoted: with colon"',
                    '  assertions:',
                    '    - type: contains',
                    "      target: 'single ''quoted'' text'",
                    '  expected_behavior: >',
                    '    folded line one',
                    '    folded line two',
                    '  edge_case: true',
                ]
            )
        )
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case["id"], "TC-001")
        self.assertEqual(case["name"], "plain-name")
        self.assertEqual(case["prompt"], "double quoted: with colon")
        self.assertEqual(case["assertions"][0]["target"], "single 'quoted' text")
        self.assertEqual(case["edge_case"], True)

    def test_double_quoted_escapes_are_decoded(self):
        cases = self.parse(
            "\n".join(
                [
                    "- id: TC-001",
                    '  prompt: "tab\\there\\nnewline"',
                    "  assertions:",
                    "    - type: regex",
                    '      target: "(0\\\\.5|0\\\\.[5-7])"',
                ]
            )
        )
        self.assertEqual(cases[0]["prompt"], "tab\there\nnewline")
        self.assertEqual(cases[0]["assertions"][0]["target"], r"(0\.5|0\.[5-7])")

    def test_single_quoted_scalar_keeps_backslashes(self):
        cases = self.parse(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: 'keeps \\d literal'",
                    "  assertions:",
                    "    - type: contains",
                    "      target: 'a'",
                ]
            )
        )
        self.assertEqual(cases[0]["prompt"], r"keeps \d literal")

    def test_block_literal_keeps_newlines_and_folded_folds_them(self):
        cases = self.parse(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: |",
                    "    line one",
                    "    line two",
                    "  assertions:",
                    "    - type: contains",
                    "      target: 'x'",
                    "  expected_behavior: >-",
                    "    folded one",
                    "    folded two",
                ]
            )
        )
        self.assertEqual(cases[0]["prompt"], "line one\nline two\n")
        self.assertEqual(cases[0]["expected_behavior"], "folded one folded two")

    def test_block_scalar_chomping_and_indent_indicator(self):
        cases = self.parse(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: |2-",
                    "    kept",
                    "    trailing",
                    "  assertions:",
                    "    - type: contains",
                    "      target: 'x'",
                    "  expected_behavior: |+",
                    "    plus",
                    "",
                    "",
                ]
            )
        )
        self.assertEqual(cases[0]["prompt"], "kept\ntrailing")
        self.assertEqual(cases[0]["expected_behavior"], "plus\n\n")

    def test_plain_scalar_comment_is_stripped(self):
        cases = self.parse(
            "\n".join(
                [
                    "- id: TC-001  # a comment",
                    "  prompt: hello  # trailing comment",
                    "  assertions:",
                    "    - type: contains",
                    "      target: world",
                ]
            )
        )
        self.assertEqual(cases[0]["prompt"], "hello")

    def test_nested_assertion_sequence_keeps_optional_keys(self):
        cases = self.parse(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: hi",
                    "  assertions:",
                    "    - type: regex",
                    "      target: 'a'",
                    '      description: "Named check"',
                    "      critical: true",
                    "      weight: 2",
                ]
            )
        )
        self.assertEqual(
            cases[0]["assertions"][0],
            {
                "type": "regex",
                "target": "a",
                "description": "Named check",
                "critical": True,
                "weight": 2,
            },
        )

    def test_empty_file_has_no_cases(self):
        self.assertEqual(self.parse("# only a comment\n"), [])

    def test_unknown_case_key_fails_with_location(self):
        message = self.parse_error(
            "\n".join(["- id: TC-001", "  prompt: hi", "  typo_key: nope"])
        )
        self.assertIn("typo_key", message)

    def test_unknown_assertion_key_fails_with_location(self):
        message = self.parse_error(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: hi",
                    "  assertions:",
                    "    - type: contains",
                    "      targets: oops",
                ]
            )
        )
        self.assertIn("unknown assertion key 'targets'", message)

    def test_missing_assertion_type_fails_with_location(self):
        message = self.parse_error(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: hi",
                    "  assertions:",
                    "    - target: no type here",
                ]
            )
        )
        self.assertIn("missing required key 'type'", message)

    def test_flow_collection_fails_loudly(self):
        message = self.parse_error(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: [a, b]",
                    "  assertions:",
                    "    - type: contains",
                ]
            )
        )
        self.assertIn("flow collections are not supported", message)

    def test_anchor_fails_loudly(self):
        message = self.parse_error(
            "\n".join(
                [
                    "- id: TC-001",
                    "  prompt: &anchor hi",
                    "  assertions:",
                    "    - type: contains",
                ]
            )
        )
        self.assertIn("anchors", message)

    def test_unterminated_quote_fails_loudly(self):
        message = self.parse_error(
            "\n".join(
                [
                    "- id: TC-001",
                    '  prompt: "never closed',
                    "  assertions:",
                    "    - type: contains",
                ]
            )
        )
        self.assertIn("unterminated double-quoted scalar", message)

    def test_tabs_in_indentation_fail_loudly(self):
        message = self.parse_error("- id: TC-001\n\tprompt: hi\n")
        self.assertIn("tabs are not supported", message)

    def test_missing_required_case_key_fails_loudly(self):
        message = self.parse_error("- id: TC-001\n  name: no-prompt\n")
        self.assertIn("missing required key 'prompt'", message)


class AssertionResultTests(unittest.TestCase):
    """Every assertion type needs one PASS case and one FAIL case."""

    def evaluate(self, assertion_type, response, target=""):
        return se.evaluate_assertion(response, assertion_type, target)

    def evaluate_with_json(self, assertion_type, json_block, target):
        return se.evaluate_assertion(assertion_type=assertion_type, target=target, response="", json_block=json_block)

    def test_contains(self):
        self.assertEqual(
            self.evaluate("contains", "Use an InstantID workflow", "instantid"), "PASS"
        )
        self.assertEqual(self.evaluate("contains", "nothing here", "InstantID"), "FAIL")

    def test_not_contains(self):
        self.assertEqual(self.evaluate("not_contains", "clean output", "InstantID"), "PASS")
        self.assertEqual(
            self.evaluate("not_contains", "InstantID as primary method", "InstantID"), "FAIL"
        )

    def test_not_contains_uses_quoted_literals_from_descriptive_targets(self):
        target = "Do not mention 'regenerate from scratch' or 'start over'"
        self.assertEqual(self.evaluate("not_contains", "we edit in place", target), "PASS")
        self.assertEqual(
            self.evaluate("not_contains", "you should start over", target), "FAIL"
        )

    def test_regex(self):
        self.assertEqual(
            self.evaluate("regex", "Set CFG to 4.0", r"(cfg|CFG).*[0-9]"), "PASS"
        )
        self.assertEqual(self.evaluate("regex", "no numbers", r"[0-9]+"), "FAIL")
        self.assertEqual(self.evaluate("regex", "anything", r"("), "FAIL")

    def test_question_before_code(self):
        self.assertEqual(
            self.evaluate("question_before_code", "Which model?\n```\ncfg 4\n```"), "PASS"
        )
        self.assertEqual(
            self.evaluate("question_before_code", "```\ncfg 4\n```\nWhich model?"), "FAIL"
        )
        self.assertEqual(self.evaluate("question_before_code", "Which model?"), "PASS")
        self.assertEqual(self.evaluate("question_before_code", "```\ncfg 4\n```"), "FAIL")

    def test_json_valid(self):
        self.assertEqual(self.evaluate("json_valid", '{"a": 1}'), "PASS")
        self.assertEqual(
            self.evaluate("json_valid", 'Here:\n```json\n{"a": 1}\n```\nDone'), "PASS"
        )
        self.assertEqual(self.evaluate("json_valid", "no json at all"), "FAIL")

    def test_json_fields_and_schema_alias(self):
        response = '{"method": "InstantID", "cfg": 4}'
        self.assertEqual(self.evaluate("json_fields", response, "method,cfg"), "PASS")
        self.assertEqual(self.evaluate("json_schema", response, "method"), "PASS")
        self.assertEqual(self.evaluate("json_fields", response, "method,steps"), "FAIL")
        self.assertEqual(self.evaluate("json_fields", "no json", "method"), "FAIL")

    def test_json_fields_requires_real_keys(self):
        self.assertEqual(self.evaluate("json_fields", '{"other": "method"}', "method"), "FAIL")

    def test_token_limit(self):
        self.assertEqual(self.evaluate("token_limit", "one two three", "5"), "PASS")
        self.assertEqual(self.evaluate("token_limit", " ".join(["word"] * 40), "5"), "FAIL")

    def test_range_check_numeric_bounds(self):
        assertion = "bias_score >= -1 AND bias_score <= 1"
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"bias_score": 0.2}', assertion), "PASS"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"bias_score": 9}', assertion), "FAIL"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"other": 1}', assertion), "SKIP"
        )

    def test_range_check_absolute_value(self):
        assertion = "abs(pose_delta) <= 0.2"
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"pose_delta": -0.1}', assertion), "PASS"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"pose_delta": -0.9}', assertion), "FAIL"
        )

    def test_range_check_minimum(self):
        assertion = "quality_score >= 7"
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"quality_score": 8}', assertion), "PASS"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"quality_score": 3}', assertion), "FAIL"
        )

    def test_range_check_length_bounds(self):
        assertion = "len(steps) >= 2 AND len(steps) <= 5"
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": "hello"}', assertion), "PASS"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": "way too long"}', assertion), "FAIL"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"other": "hello"}', assertion), "FAIL"
        )

    def test_range_check_length_maximum_and_minimum(self):
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": "abc"}', "len(steps) <= 5"), "PASS"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": "abcdefgh"}', "len(steps) <= 5"),
            "FAIL",
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": [1, 2, 3]}', "len(steps) >= 3"),
            "PASS",
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": [1]}', "len(steps) >= 3"), "FAIL"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"steps": "abcdef"}', "len(steps) >= 3"),
            "PASS",
        )

    def test_range_check_word_count(self):
        assertion = "word_count(notes) >= 2 AND word_count(notes) <= 4"
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"notes": "one two three"}', assertion),
            "PASS",
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"notes": "one"}', assertion), "FAIL"
        )

    def test_range_check_requires_json(self):
        self.assertEqual(
            self.evaluate_with_json("range_check", "not json", "len(steps) <= 5"), "FAIL"
        )
        self.assertEqual(
            self.evaluate_with_json("range_check", "", "quality_score >= 7"), "FAIL"
        )

    def test_range_check_unknown_expression_is_skipped(self):
        self.assertEqual(
            self.evaluate_with_json("range_check", '{"a": 1}', "a is roughly fine"), "SKIP"
        )

    def test_structure_and_sequence_checks_are_soft(self):
        self.assertEqual(self.evaluate("structure_check", "anything", "target"), "SOFT")
        self.assertEqual(self.evaluate("sequence_check", "anything", "target"), "SOFT")

    def test_unknown_assertion_type_is_skipped(self):
        self.assertEqual(self.evaluate("made_up_type", "anything", "target"), "SKIP")


class ScoringTests(unittest.TestCase):
    def test_summary_matches_shell_format(self):
        assertions = [
            {"type": "contains", "target": "alpha", "description": "Alpha present"},
            {"type": "contains", "target": "beta", "description": "Beta present"},
            {"type": "structure_check", "target": "soft", "critical": True},
            {"type": "unknown_kind", "target": "skip"},
        ]
        scored = se.score_case_response("alpha only", assertions)
        self.assertEqual(scored.summary, "1/4 (2 anchored, 1 soft, 1 skipped)")
        self.assertEqual(
            scored.details,
            [
                "  PASS — Alpha present",
                "  FAIL — Beta present",
                "  SOFT [CRITICAL] — structure_check(soft)",
                "  SKIP — unknown_kind(skip)",
            ],
        )

    def test_critical_failure_is_flagged(self):
        assertions = [
            {"type": "contains", "target": "alpha", "critical": True},
            {"type": "contains", "target": "alpha"},
        ]
        self.assertEqual(
            se.score_case_response("nothing", assertions).summary,
            "0/2 (2 anchored, 0 soft, 0 skipped) [CRITICAL FAIL]",
        )
        score_text = se._score_text(se.score_case_response("nothing", assertions))
        self.assertTrue(score_text.endswith("\n\n"))

    def test_soft_critical_assertion_never_fails_the_case(self):
        assertions = [{"type": "sequence_check", "target": "order", "critical": True}]
        self.assertEqual(
            se.score_case_response("anything", assertions).summary,
            "0/1 (0 anchored, 1 soft, 0 skipped)",
        )

    def test_eval_error_response_scores_as_error(self):
        assertions = [{"type": "contains", "target": "alpha"}]
        scored = se.score_case_response("EVAL_ERROR: agent command failed", assertions)
        self.assertEqual(scored.summary, "ERROR")
        self.assertEqual(scored.details, [])


class AgentCommandTests(unittest.TestCase):
    def run_stub(self, argv):
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_prompt_placeholder_is_substituted(self):
        argv = se.build_agent_argv(
            f'{STUB_AGENT_CMD} --prompt {{prompt}}', "hello world", PROMPT_FILE
        )
        self.assertIn("hello world", argv)
        self.assertTrue(self.run_stub(argv).startswith("PROMPT-RECEIVED: hello world"))

    def test_prompt_file_placeholder_is_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_file = Path(tmp) / "TC-001.txt"
            prompt_file.write_text("from the file", encoding="utf-8")
            argv = se.build_agent_argv(
                f'{STUB_AGENT_CMD} --prompt-file {{prompt_file}}',
                "from the file",
                prompt_file,
            )
            self.assertIn(str(prompt_file), argv)
            self.assertTrue(
                self.run_stub(argv).startswith("PROMPT-RECEIVED: from the file")
            )

    def test_prompt_is_appended_without_a_placeholder(self):
        argv = se.build_agent_argv(
            STUB_AGENT_CMD, 'quoted "text" with spaces', PROMPT_FILE
        )
        self.assertEqual(argv[-1], 'quoted "text" with spaces')
        self.assertIn(
            'PROMPT-RECEIVED: quoted "text" with spaces', self.run_stub(argv)
        )

    def test_quoted_arguments_are_split_on_posix(self):
        argv = se.split_agent_command(
            'my-agent --model "big model" --flag', platform="posix"
        )
        self.assertEqual(
            argv, ["my-agent", "--model", "big model", "--flag"]
        )

    def test_windows_splitting_keeps_backslashes(self):
        argv = se.split_agent_command(
            r'"C:\Program Files\agent.exe" -p', platform="nt"
        )
        self.assertEqual(argv, [r"C:\Program Files\agent.exe", "-p"])

    def test_empty_command_is_a_harness_error(self):
        with self.assertRaises(se.HarnessError):
            se.split_agent_command("   ")

    def test_unbalanced_quotes_are_a_harness_error(self):
        with self.assertRaises(se.HarnessError):
            se.split_agent_command('agent "unclosed', platform="posix")

    def test_missing_executable_is_a_harness_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not-an-agent"
            with self.assertRaises(se.HarnessError) as caught:
                se.validate_agent_command([str(missing)])
            self.assertIn("not executable", str(caught.exception))


class DryRunTests(unittest.TestCase):
    def test_real_skills_report_exact_counts(self):
        for skill_name, (case_count, assertion_count) in REAL_SKILLS.items():
            with self.subTest(skill=skill_name):
                yaml_path = (
                    REPO_ROOT / "skills" / skill_name / "eval" / "test-cases.yaml"
                )
                cases = se.parse_yaml_subset(yaml_path)
                self.assertEqual(len(cases), case_count)
                self.assertEqual(
                    sum(len(case["assertions"]) for case in cases), assertion_count
                )

                code, stdout, _stderr = run_main(
                    ["--skill", f"skills/{skill_name}", "--dry-run"]
                )
                self.assertEqual(code, 0)
                self.assertIn(
                    f"Found {case_count} test cases and {assertion_count} assertions",
                    stdout,
                )

    def test_dry_run_never_invokes_the_agent(self):
        code, stdout, _stderr = run_main(
            [
                "--skill",
                "comfyui-character-gen",
                "--dry-run",
                "--agent-cmd",
                "/definitely/missing/agent",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("Dry run: no agent invoked and no result files written.", stdout)

    def test_dry_run_writes_no_results(self):
        results_dir = REPO_ROOT / "skills" / "comfyui-character-gen" / "eval" / "results"
        before = sorted(entry.name for entry in results_dir.iterdir())
        run_main(["--skill", "comfyui-character-gen", "--dry-run"])
        self.assertEqual(
            sorted(entry.name for entry in results_dir.iterdir()), before
        )

    def test_dry_run_honours_the_case_filter(self):
        code, stdout, _stderr = run_main(
            ["--skill", "comfyui-character-gen", "--dry-run", "--case", "TC-002"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Found 1 test cases and 5 assertions", stdout)
        self.assertNotIn("TC-001", stdout)

    def test_unknown_case_is_a_harness_error(self):
        code, _stdout, stderr = run_main(
            ["--skill", "comfyui-character-gen", "--dry-run", "--case", "TC-999"]
        )
        self.assertEqual(code, 2)
        self.assertIn("TC-999", stderr)


class ScoredRunTests(unittest.TestCase):
    def test_full_run_writes_scorecard_and_scores_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            results_dir = root / "results"
            code, stdout, stderr = run_main(
                ["--skill", str(skill_dir), "--results-dir", str(results_dir)],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            self.assertEqual(code, 0, stderr)

            for relative in (
                "prompts/TC-001.txt",
                "with-skill/TC-001.md",
                "baseline/TC-001.md",
                "with-skill/TC-001.score.txt",
                "baseline/TC-001.score.txt",
                "with-skill/TC-002.score.txt",
                "scorecard.md",
            ):
                self.assertTrue((results_dir / relative).is_file(), relative)

            with_skill = (results_dir / "with-skill" / "TC-001.md").read_text(
                encoding="utf-8"
            )
            baseline = (results_dir / "baseline" / "TC-001.md").read_text(
                encoding="utf-8"
            )
            self.assertTrue(
                with_skill.startswith(
                    "PROMPT-RECEIVED: Use the stub-skill skill to answer this:"
                ),
                with_skill.splitlines()[0],
            )
            self.assertTrue(
                baseline.startswith("PROMPT-RECEIVED: Which model should I target?"),
                baseline.splitlines()[0],
            )

            scorecard = (results_dir / "scorecard.md").read_text(encoding="utf-8")
            self.assertIn("| Test Case | With Skill | Baseline |", scorecard)
            self.assertIn(
                f"| TC-001 | {STUB_CASE_ONE_SCORE} | {STUB_CASE_ONE_SCORE} |", scorecard
            )
            self.assertIn(
                f"| TC-002 | {STUB_CASE_TWO_SCORE} | {STUB_CASE_TWO_SCORE} |", scorecard
            )
            self.assertIn("## Assertion Details", scorecard)
            self.assertIn("**with-skill:**", scorecard)
            self.assertIn("  SOFT — structure_check(Looks structured)", scorecard)
            self.assertIn("  SKIP — judge_me(not a real type)", scorecard)

            self.assertIn("=== Scorecard ===", stdout)
            self.assertEqual(
                (results_dir / "with-skill" / "TC-001.score.txt").read_text(
                    encoding="utf-8"
                ),
                STUB_CASE_ONE_SCORE + "\n  PASS [CRITICAL] — contains(Which model)\n"
                "  PASS — not_contains(banana)\n"
                "  FAIL [CRITICAL] — contains(definitely-absent)\n"
                "  SOFT — structure_check(Looks structured)\n"
                "  SKIP — judge_me(not a real type)\n\n",
            )

    def test_exit_code_is_zero_even_when_assertions_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            code, _stdout, _stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(root / "results"),
                ],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            self.assertEqual(code, 0)
            scorecard = (root / "results" / "scorecard.md").read_text(encoding="utf-8")
            self.assertIn("FAIL", scorecard)

    def test_skill_only_and_baseline_only_select_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            code, _stdout, _stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(root / "skill-only"),
                    "--skill-only",
                    "--case",
                    "TC-001",
                ],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            self.assertEqual(code, 0)
            self.assertTrue((root / "skill-only" / "with-skill" / "TC-001.md").is_file())
            scorecard = (root / "skill-only" / "scorecard.md").read_text(encoding="utf-8")
            self.assertIn(f"| TC-001 | {STUB_CASE_ONE_SCORE} | — |", scorecard)

            code, _stdout, _stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(root / "baseline-only"),
                    "--baseline-only",
                    "--case",
                    "TC-001",
                ],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            self.assertEqual(code, 0)
            self.assertTrue((root / "baseline-only" / "baseline" / "TC-001.md").is_file())
            scorecard = (root / "baseline-only" / "scorecard.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(f"| TC-001 | — | {STUB_CASE_ONE_SCORE} |", scorecard)

    def test_score_only_rescores_an_existing_results_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            results_dir = root / "results"
            run_main(
                ["--skill", str(skill_dir), "--results-dir", str(results_dir)],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            scorecard_path = results_dir / "scorecard.md"
            scorecard_path.unlink()
            (results_dir / "with-skill" / "TC-001.score.txt").unlink()

            code, _stdout, stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(results_dir),
                    "--score-only",
                ],
                env={"EVAL_AGENT_CMD": "/definitely/missing/agent"},
            )
            self.assertEqual(code, 0, stderr)
            self.assertTrue(scorecard_path.is_file())
            self.assertTrue((results_dir / "with-skill" / "TC-001.score.txt").is_file())
            self.assertIn(
                STUB_CASE_ONE_SCORE, scorecard_path.read_text(encoding="utf-8")
            )

    def test_score_only_uses_the_newest_results_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            results_root = skill_dir / "eval" / "results"
            (results_root / "20260101-000000" / "with-skill").mkdir(parents=True)
            (results_root / "20260101-000000" / "with-skill" / "TC-001.md").write_text(
                "Which model are you targeting?", encoding="utf-8"
            )
            code, _stdout, stderr = run_main(
                ["--skill", str(skill_dir), "--score-only"],
                env={"EVAL_AGENT_CMD": "/definitely/missing/agent"},
            )
            self.assertEqual(code, 0, stderr)
            scorecard = (
                results_root / "20260101-000000" / "scorecard.md"
            ).read_text(encoding="utf-8")
            self.assertIn("| TC-001 |", scorecard)

    def test_score_only_without_results_is_a_harness_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = write_stub_skill(Path(tmp))
            code, _stdout, stderr = run_main(
                ["--skill", str(skill_dir), "--score-only"],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            self.assertEqual(code, 2)
            self.assertIn("no existing results found", stderr)

    def test_agent_flag_wins_over_the_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            missing = root / "missing-agent"
            code, _stdout, stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(root / "results"),
                    "--agent-cmd",
                    str(missing),
                ],
                env={"EVAL_AGENT_CMD": STUB_AGENT_CMD},
            )
            self.assertEqual(code, 2)
            self.assertIn("not executable", stderr)

    def test_missing_agent_command_is_a_harness_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            code, _stdout, stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(root / "results"),
                    "--agent-cmd",
                    str(root / "no-such-agent"),
                ],
                env={"EVAL_AGENT_CMD": ""},
            )
            self.assertEqual(code, 2)
            self.assertIn("no-such-agent", stderr)

    def test_invalid_yaml_is_a_harness_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "broken-skill"
            (skill_dir / "eval").mkdir(parents=True)
            (skill_dir / "eval" / "test-cases.yaml").write_text(
                "- id: TC-001\n  prompt: hi\n  unexpected_key: true\n", encoding="utf-8"
            )
            code, _stdout, stderr = run_main(
                ["--skill", str(skill_dir), "--dry-run"]
            )
            self.assertEqual(code, 2)
            self.assertIn("unexpected_key", stderr)
            self.assertIn("test-cases.yaml:3", stderr)

    def test_empty_test_case_file_is_a_harness_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "empty-skill"
            (skill_dir / "eval").mkdir(parents=True)
            (skill_dir / "eval" / "test-cases.yaml").write_text("# nothing\n", "utf-8")
            code, _stdout, stderr = run_main(["--skill", str(skill_dir), "--dry-run"])
            self.assertEqual(code, 2)
            self.assertIn("no test cases found", stderr)

    def test_unknown_skill_is_a_harness_error(self):
        code, _stdout, stderr = run_main(["--skill", "no-such-skill", "--dry-run"])
        self.assertEqual(code, 2)
        self.assertIn("no-such-skill", stderr)

    def test_eval_error_output_is_scored_without_exit_code_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = write_stub_skill(root)
            results_dir = root / "results"
            (results_dir / "with-skill").mkdir(parents=True)
            (results_dir / "with-skill" / "TC-001.md").write_text(
                "EVAL_ERROR: agent command failed for TC-001 (with-skill)\n",
                encoding="utf-8",
            )
            code, _stdout, _stderr = run_main(
                [
                    "--skill",
                    str(skill_dir),
                    "--results-dir",
                    str(results_dir),
                    "--score-only",
                    "--case",
                    "TC-001",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(
                (results_dir / "with-skill" / "TC-001.score.txt").read_text(
                    encoding="utf-8"
                ),
                "ERROR\n",
            )
            scorecard = (results_dir / "scorecard.md").read_text(encoding="utf-8")
            self.assertIn("| TC-001 | ERROR | — |", scorecard)

    def test_missing_skill_argument_exits_two(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            se.main(["--dry-run"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
