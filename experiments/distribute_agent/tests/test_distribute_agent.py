"""Tests for the experiments/distribute_agent prototype.

This suite lives with the prototype because it needs smolagents and a Python 3.12
interpreter, which the repository itself does not: run it from this directory
(``python -m unittest discover -s tests``) with the prototype's environment, not
through the repository's default ``tests/`` discovery. Missing dependencies fail
this module loudly instead of skipping, so a security test can never pass by
silently not running.

Coverage tags map to the approved integration test plan: S1-S4 skill/package/
isolation, P1-P4 prompt and request preservation, E1-E4 end-to-end dry run,
V1-V4 vendor job handling (tests/test_rp_oneshot.py), A1-A4 isolation, security
and opt-in live harnesses.
"""

import hashlib
import io
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_DIR = PROJECT_ROOT / "experiments" / "distribute_agent"
PACKAGE_SOURCE = PROJECT_ROOT / "distribute"

sys.path.insert(0, str(PROTOTYPE_DIR))

try:
    from distribute_agent import agent as agent_module
    from distribute_agent.agent import AgentRunError, build_agent, redact, run_agent
    from distribute_agent.bridge import BridgeError, submit_plan
    from distribute_agent.executor import SandboxedPythonExecutor
    from distribute_agent.live_eval import (
        BRIEFS,
        DEFAULT_REPETITIONS,
        _check_answer,
        _preview_evidence,
        run_live_eval,
    )
    from distribute_agent.package import Package, PackageError
    from distribute_agent.plan import (
        DIMENSION_STEP,
        MAX_DIMENSION,
        MAX_SEED,
        PLAN_FIELDS,
        PlanError,
        parse_plan,
        plan_output_instructions,
        plan_schema,
    )
    from distribute_agent.sandbox import (
        SandboxPolicy,
        SandboxUnavailable,
        run_sandboxed,
    )
    from distribute_agent.scripted_model import ScriptedModel, code_reply
    from distribute_agent.skill import SkillError, load_skill
    from distribute_agent.vendor_smoke import (
        MAX_ARTIFACT_BYTES,
        VendorSmokeError,
        run_vendor_smoke,
        sniff_image,
    )
    from PIL import Image
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "the distribute_agent prototype needs its pinned environment: "
        "python3.12 -m venv /tmp/da-venv && "
        "/tmp/da-venv/bin/python -m pip install 'smolagents==1.26.0' "
        f"(missing: {exc})"
    ) from exc


def copy_package(destination):
    """Copy the shipped package so tests never write into the repo's copy."""
    shutil.copytree(
        PACKAGE_SOURCE,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "output", "projects"),
    )
    return Path(destination)


def make_skill_copy(package_root, marker):
    """Copy the shipped skill and append one harmless instruction (S1)."""
    skill_dir = Path(package_root) / "skills" / "runpod-oneshot"
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8")
        + f"\n\n## Local marker\n\nAlways include the token {marker} in the final answer.\n",
        encoding="utf-8",
    )
    return skill_dir


def png_chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def build_png(width=1, height=1, color=(220, 40, 40)):
    """A complete, valid PNG: signature, IHDR, IDAT and IEND with real CRCs."""
    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


PNG_BYTES = build_png()

PNG_IHDR_1X1_RGB = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)


def build_empty_idat_png():
    """A PNG whose single IDAT is a valid zlib stream of *zero* image bytes."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", PNG_IHDR_1X1_RGB)
        + png_chunk(b"IDAT", zlib.compress(b""))
        + png_chunk(b"IEND", b"")
    )


def build_invalid_idat_png():
    """A PNG whose IDAT payload is not a zlib stream; chunk CRCs are correct."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", PNG_IHDR_1X1_RGB)
        + png_chunk(b"IDAT", b"\x00\x01\x02\x03")
        + png_chunk(b"IEND", b"")
    )


def build_frame_header_only_jpeg():
    """SOI, a 1x1 frame header and EOI: no scan, so no encoded pixels."""
    return bytes.fromhex("ffd8ffc0001108000100010301110002110003110000ffd9")


def build_jpeg(width=2, height=3):
    """A complete baseline JPEG produced by Pillow (already a smolagents dep)."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 120, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def message_text(message):
    """Flatten a ChatMessage's content, which SmolAgents may send as blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


def tree_digest(root):
    digest = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def write_stub_runner(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def dry_run_script(prompt, seed=42, width=832, height=1216):
    return code_reply(
        "plan = run_oneshot_preview(\n"
        '    template="unholy-sdxl",\n'
        f"    prompt={prompt!r},\n"
        f"    seed={seed},\n"
        f"    width={width},\n"
        f"    height={height},\n"
        ")\n"
        "final_answer(plan)"
    )


E1_PROMPT = (
    "masterpiece, best quality, a lighthouse keeper winding a brass clock "
    "on a stormy cliff, oilskin coat, cold blue light, no text"
)


class CodeAgentDryRunTests(unittest.TestCase):
    """E1: the first tracer bullet — real CodeAgent, real runner, no network."""

    def test_actual_codeagent_executes_the_real_runner_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            skill = load_skill(package.root / "skills" / "runpod-oneshot")
            model = ScriptedModel(
                [
                    code_reply(
                        "plan = run_oneshot_preview(\n"
                        '    template="unholy-sdxl",\n'
                        f"    prompt={E1_PROMPT!r},\n"
                        "    seed=42,\n"
                        "    width=832,\n"
                        "    height=1216,\n"
                        ")\n"
                        "final_answer(plan)"
                    )
                ]
            )
            agent = build_agent(
                model=model,
                skill=skill,
                package=package,
                workspace_dir=Path(tmp) / "run",
                max_steps=4,
                deadline_seconds=120,
            )
            result = run_agent(
                agent,
                "Create a plan for the lighthouse keeper image with seed 42 at 832x1216.",
                deadline_seconds=120,
            )

            plan = result.answer
            self.assertEqual(plan["exit_code"], 0, plan["stderr"])
            self.assertIn("--dry-run", plan["argv"])
            self.assertNotIn("--wait", plan["argv"])
            envelope = plan["envelope"]
            workflow = envelope["input"]["workflow"]

            # The template is identified by its selected node content: the
            # RunPod envelope does not carry a template id.
            template = json.loads(
                (package.root / "workflows" / "unholy-sdxl.json").read_text(encoding="utf-8")
            )
            self.assertEqual(workflow["2"]["class_type"], template["2"]["class_type"])
            self.assertEqual(workflow["1"]["inputs"]["ckpt_name"],
                             template["1"]["inputs"]["ckpt_name"])
            self.assertEqual(workflow["2"]["inputs"]["text"], E1_PROMPT)
            self.assertEqual(workflow["5"]["inputs"]["seed"], 42)
            self.assertEqual(workflow["4"]["inputs"]["width"], 832)
            self.assertEqual(workflow["4"]["inputs"]["height"], 1216)


class SkillLoadingTests(unittest.TestCase):
    """S1/S3/P2: the skill is parsed, validated and consumed at startup."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.package = Package.load(copy_package(Path(self._tmp.name) / "package"))

    def test_the_shipped_skill_is_parsed_and_validated(self):
        skill = load_skill(self.package.skill_path())
        self.assertEqual(skill.name, "runpod-oneshot")
        self.assertTrue(skill.description.startswith("Create and run a customized"))
        self.assertIn("## The 3-step loop", skill.body)
        self.assertLessEqual(len(skill.description), 1024)

    def test_missing_and_malformed_skill_files_fail_actionably(self):
        with self.assertRaises(SkillError):
            load_skill(self.package.root / "skills" / "absent")
        cases = {
            "no-frontmatter": "# Just a body\n",
            "missing-description": "---\nname: runpod-oneshot\n---\nbody\n",
            "bad-name": "---\nname: Runpod_Oneshot\ndescription: d\n---\nbody\n",
            "empty-body": "---\nname: runpod-oneshot\ndescription: d\n---\n\n",
            "block-scalar": "---\nname: runpod-oneshot\ndescription: |\n  multi\n---\nbody\n",
        }
        for label, content in cases.items():
            with self.subTest(case=label):
                skill_dir = self.package.root / "skills" / "runpod-oneshot"
                original = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
                (skill_dir / "SKILL.md").write_text(
                    content.replace("runpod-oneshot", "runpod-oneshot"), encoding="utf-8"
                )
                try:
                    with self.assertRaises(SkillError) as caught:
                        load_skill(skill_dir)
                    self.assertTrue(str(caught.exception))
                finally:
                    (skill_dir / "SKILL.md").write_text(original, encoding="utf-8")

    def test_name_must_match_the_skill_directory(self):
        source = self.package.root / "skills" / "runpod-oneshot" / "SKILL.md"
        renamed = self.package.root / "skills" / "other-name"
        renamed.mkdir()
        (renamed / "SKILL.md").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaises(SkillError) as caught:
            load_skill(renamed)
        self.assertIn("must match the skill directory", str(caught.exception))

    def test_marker_instruction_changes_observable_agent_behavior(self):
        # S1 wiring test, not an LLM-compliance test: the scripted model is the
        # deterministic stand-in, and it reacts to the real system prompt.
        make_skill_copy(self.package.root, "ZXQ-77")
        skill = load_skill(self.package.skill_path())

        def reply(messages):
            token = "ZXQ-77" if "ZXQ-77" in message_text(messages[0]) else "marker-missing"
            return code_reply(f'final_answer({{"marker": {token!r}}})')

        model = ScriptedModel([reply])
        agent = build_agent(
            model=model,
            skill=skill,
            package=self.package,
            workspace_dir=Path(self._tmp.name) / "run-marker",
            deadline_seconds=60,
        )
        result = run_agent(agent, "Answer with the marker token.", deadline_seconds=60)
        system_prompt = message_text(model.calls[0][0])
        self.assertIn("ZXQ-77", system_prompt)
        self.assertIn("run_oneshot_preview", system_prompt)
        self.assertIn("unholy-sdxl", system_prompt)
        self.assertEqual(result.answer, {"marker": "ZXQ-77"})

        stock = Package.load(copy_package(Path(self._tmp.name) / "stock"))
        model = ScriptedModel([reply])
        agent = build_agent(
            model=model,
            skill=load_skill(stock.skill_path()),
            package=stock,
            workspace_dir=Path(self._tmp.name) / "run-stock",
            deadline_seconds=60,
        )
        result = run_agent(agent, "Answer with the marker token.", deadline_seconds=60)
        self.assertNotIn("ZXQ-77", message_text(model.calls[0][0]))
        self.assertEqual(result.answer, {"marker": "marker-missing"})

    def test_skill_recipes_and_manifest_metadata_reach_the_instructions(self):
        skill = load_skill(self.package.skill_path())
        context = self.package.template("unholy-sdxl")
        self.assertEqual(context.family, "sdxl")
        self.assertIn("1024", context.size_guidance)
        self.assertFalse(context.accepts_image)
        instructions = self.package.instructions(skill)
        self.assertIn("masterpiece, best quality", instructions)  # skill recipe, not rewritten
        self.assertIn("unholyDesireMixSinister_v90.safetensors", instructions)

        # A synthetic template variation lives in the temp copy only.
        manifest_path = self.package.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["templates"]["unholy-sdxl"]["model_family"] = "flux"
        manifest["templates"]["unholy-sdxl"]["model_info"]["trigger_word"] = "mystyle"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        varied = Package.load(self.package.root).template("unholy-sdxl")
        self.assertEqual(varied.family, "flux")
        self.assertEqual(varied.trigger_word, "mystyle")
        self.assertIn("mystyle", varied.as_instructions())


class PackageResolutionTests(unittest.TestCase):
    """S2/S3: resolution from any cwd and path, actionable incomplete packages."""

    def test_package_resolves_from_another_cwd_and_a_path_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_root = copy_package(root / "package with spaces")
            elsewhere = root / "somewhere else"
            elsewhere.mkdir()
            original_cwd = os.getcwd()
            os.chdir(elsewhere)
            try:
                package = Package.load(package_root)
                self.assertEqual(package.template_names(), ("unholy-sdxl",))
                skill = load_skill(package.skill_path())
                model = ScriptedModel([dry_run_script("a prompt from another cwd")])
                agent = build_agent(
                    model=model,
                    skill=skill,
                    package=package,
                    workspace_dir=root / "run with spaces",
                    deadline_seconds=120,
                )
                result = run_agent(agent, "Plan a dry run.", deadline_seconds=120)
            finally:
                os.chdir(original_cwd)
            workflow = result.answer["envelope"]["input"]["workflow"]
            self.assertEqual(workflow["2"]["inputs"]["text"], "a prompt from another cwd")

    def test_incomplete_or_malformed_packages_fail_actionably(self):
        cases = [
            ("manifest.json", None, "missing manifest"),
            ("manifest.json", "{broken", "not valid JSON"),
            ("endpoints.json", None, "missing endpoint snapshot"),
            ("endpoints.json", "{}", "no endpoint recorded"),
            ("workflows/unholy-sdxl.json", None, "missing workflow"),
            ("bin/rp_oneshot.py", None, "missing runner"),
        ]
        for relative, content, needle in cases:
            with self.subTest(file=relative, content=content):
                with tempfile.TemporaryDirectory() as tmp:
                    package_root = copy_package(Path(tmp) / "package")
                    target = package_root / relative
                    if content is None:
                        target.unlink()
                    else:
                        target.write_text(content, encoding="utf-8")
                    with self.assertRaises(PackageError) as caught:
                        Package.load(package_root)
                    self.assertIn(needle, str(caught.exception))

    def test_repo_package_manifest_is_never_modified_by_loading(self):
        manifest = PROJECT_ROOT / "distribute" / "manifest.json"
        before = manifest.read_bytes()
        Package.load(PROJECT_ROOT / "distribute")
        self.assertEqual(manifest.read_bytes(), before)


class PlanValidationTests(unittest.TestCase):
    """P4: unsupported requests are rejected before the host bridge runs."""

    def setUp(self):
        self.manifest = json.loads(
            (PROJECT_ROOT / "distribute" / "manifest.json").read_text(encoding="utf-8")
        )

    def test_a_valid_plan_maps_to_runner_arguments(self):
        plan = parse_plan(
            self.manifest,
            {
                "template": "unholy-sdxl",
                "prompt": "a red apple",
                "seed": 42,
                "width": 832,
                "height": 1216,
            },
        )
        argv = plan.to_argv("/tmp/out")
        self.assertEqual(argv[0], "--template")
        self.assertIn("unholy-sdxl", argv)
        self.assertEqual(argv[argv.index("--seed") + 1], "42")
        self.assertEqual(argv[argv.index("--width") + 1], "832")

    def test_unknown_template_settings_and_bad_ranges_are_rejected(self):
        base = {"template": "unholy-sdxl", "prompt": "x"}
        cases = {
            "unknown-template": {**base, "template": "not-shipped"},
            "boolean-seed": {**base, "seed": True},
            "seed-too-large": {**base, "seed": 2**64},
            "negative-seed": {**base, "seed": -1},
            "width-not-multiple": {**base, "width": 7},
            "width-zero": {**base, "width": 0},
            "height-too-large": {**base, "height": 20000},
            "unsupported-image": {**base, "image": "input.png"},
            "unknown-setting": {**base, "cfg": 8},
            "nul-in-prompt": {**base, "prompt": "bad\x00prompt"},
        }
        for label, data in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(PlanError) as caught:
                    parse_plan(self.manifest, data)
                self.assertTrue(str(caught.exception))

    def test_the_output_schema_matches_the_parser_it_describes(self):
        schema = plan_schema(self.manifest)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], ["template"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(list(schema["properties"]), list(PLAN_FIELDS))
        properties = schema["properties"]
        self.assertEqual(properties["template"]["type"], "string")
        self.assertEqual(
            properties["template"]["enum"], sorted(self.manifest["templates"])
        )
        # The enum is what rules out an empty template; no minLength is declared.
        self.assertNotIn("", properties["template"]["enum"])
        self.assertEqual(properties["seed"]["minimum"], 0)
        self.assertEqual(properties["seed"]["maximum"], MAX_SEED)
        for name in ("width", "height"):
            self.assertEqual(properties[name]["minimum"], 1)
            self.assertEqual(properties[name]["maximum"], MAX_DIMENSION)
            self.assertEqual(properties[name]["multipleOf"], DIMENSION_STEP)
        for name in ("prompt", "negative"):
            self.assertEqual(properties[name]["type"], ["string", "null"])
        for name in ("seed", "width", "height"):
            self.assertEqual(properties[name]["type"], ["integer", "null"])
        other = plan_schema({"templates": {"b-template": {}, "a-template": {}}})
        self.assertEqual(
            other["properties"]["template"]["enum"], ["a-template", "b-template"]
        )
        with self.assertRaises(PlanError):
            plan_schema({"templates": {}})

    def test_parse_plan_accepts_and_rejects_the_fixture_variations(self):
        template = sorted(self.manifest["templates"])[0]
        valid = {
            "template": template,
            "prompt": "a red apple",
            "negative": "",
            "seed": MAX_SEED,
            "width": MAX_DIMENSION,
            "height": DIMENSION_STEP,
        }
        parse_plan(self.manifest, valid)
        parse_plan(self.manifest, {"template": template})
        parse_plan(
            self.manifest,
            {
                "template": template,
                "prompt": None,
                "negative": None,
                "seed": None,
                "width": None,
                "height": None,
            },
        )
        invalid = {
            "nested-plan-wrapper": {"plan": valid},
            "extra-negative-prompt-alias": {**valid, "negative_prompt": "x"},
            "extra-metadata": {**valid, "exit_code": 0},
            "unknown-template": {**valid, "template": "not-shipped"},
            "seed-above-maximum": {**valid, "seed": MAX_SEED + 1},
            "seed-negative": {**valid, "seed": -1},
            "seed-boolean": {**valid, "seed": True},
            "width-not-multiple": {**valid, "width": DIMENSION_STEP + 1},
            "width-above-maximum": {**valid, "width": MAX_DIMENSION * 2},
            "height-string": {**valid, "height": str(DIMENSION_STEP)},
            "prompt-nul": {**valid, "prompt": "x\x00y"},
            "prompt-number": {**valid, "prompt": 7},
            "missing-template": {"prompt": "a red apple"},
        }
        for label, data in invalid.items():
            with self.subTest(case=label):
                with self.assertRaises(PlanError):
                    parse_plan(self.manifest, data)

    def test_the_output_instructions_carry_the_schema_and_the_prose_rules(self):
        text = plan_output_instructions(self.manifest)
        self.assertIn(json.dumps(plan_schema(self.manifest), indent=2), text)
        for phrase in (
            "final_answer(plan)",
            "negative_prompt",
            "NUL",
            f"0..{MAX_SEED}",
            f"1..{MAX_DIMENSION}",
            f"multiples of {DIMENSION_STEP}",
            "evidence, not a plan",
        ):
            self.assertIn(phrase, text)

    def test_invalid_plans_never_reach_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "runner-was-called"
            stub = write_stub_runner(
                Path(tmp) / "stub_runner.py", f"open({str(marker)!r}, 'w').write('yes')\n"
            )
            package = Package.load(copy_package(Path(tmp) / "package"))
            with self.assertRaises(PlanError):
                submit_plan(
                    package,
                    {"template": "unholy-sdxl", "image": "input.png"},
                    output_dir=Path(tmp) / "out",
                    env={"RUNPOD_API_KEY": "rpa_test_key"},
                    runner_prefix=[sys.executable, str(stub)],
                )
            self.assertFalse(marker.exists())


class SandboxBoundaryTests(unittest.TestCase):
    """A2/S4: the OS boundary, not prompt text, protects the host."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.package = Package.load(copy_package(self.root / "package"))
        self.workspace = self.root / "workspace"
        self.policy = SandboxPolicy(package_root=self.package.root, workspace=self.workspace)

    def run_source(self, source, timeout=30):
        return run_sandboxed(self.policy, source, timeout=timeout)

    def test_host_files_and_credentials_are_not_visible(self):
        source = r"""
import os
import pathlib
print("key:", os.environ.get("RUNPOD_API_KEY"))
for path in ("/home/xdblue/Repos/mediagen/AGENTS.md", "/etc/passwd", "/root/.ssh/id_rsa"):
    try:
        pathlib.Path(path).read_text()
        print("READ", path)
    except OSError as exc:
        print("DENIED", path, type(exc).__name__)
"""
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "sk-host-secret"}, clear=False):
            result = self.run_source(source)
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertIn("key: None", result.stdout)
        self.assertNotIn("READ", result.stdout)
        self.assertEqual(result.stdout.count("DENIED"), 3)

    def test_package_is_read_only_and_outputs_stay_in_the_workspace(self):
        before = tree_digest(self.package.root)
        source = r"""
import pathlib
try:
    with open("/package/manifest.json", "a", encoding="utf-8") as handle:
        handle.write("mutation")
    print("WROTE package")
except OSError as exc:
    print("DENIED package", type(exc).__name__)
out = pathlib.Path("/workspace/out")
out.mkdir(parents=True, exist_ok=True)
(out / "artifact.txt").write_text("hello", encoding="utf-8")
print("workspace-ok")
"""
        result = self.run_source(source)
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertIn("DENIED package", result.stdout)
        self.assertNotIn("WROTE package", result.stdout)
        self.assertEqual((self.workspace / "out" / "artifact.txt").read_text(encoding="utf-8"), "hello")
        self.assertFalse(Path("/tmp/sandbox-escape-marker").exists(),
                         "/tmp must be a private tmpfs, not the host /tmp")
        self.assertEqual(tree_digest(self.package.root), before)

    def test_network_shell_and_unrelated_commands_are_unavailable(self):
        source = r"""
import socket
import subprocess
for command in ("sh", "curl", "runpodctl"):
    argv = [command, "-c", "true"] if command == "sh" else [command, "--version"]
    try:
        subprocess.run(argv, capture_output=True, timeout=5)
        print("RAN", command)
    except OSError as exc:
        print("DENIED command", command, type(exc).__name__)
try:
    socket.create_connection(("1.1.1.1", 80), timeout=2)
    print("NETWORK open")
except OSError as exc:
    print("DENIED network", type(exc).__name__)
"""
        result = self.run_source(source)
        self.assertEqual(result.exit_code, 0, result.stderr)
        self.assertNotIn("RAN", result.stdout)
        self.assertNotIn("NETWORK open", result.stdout)
        self.assertEqual(result.stdout.count("DENIED command"), 3)
        self.assertIn("DENIED network", result.stdout)

    def test_missing_or_broken_bwrap_fails_closed(self):
        with self.assertRaises(SandboxUnavailable) as caught:
            SandboxPolicy(
                package_root=self.package.root,
                workspace=self.root / "ws-missing",
                bwrap=Path("/nonexistent/bwrap"),
            )
        self.assertIn("bubblewrap", str(caught.exception))

        fake = write_stub_runner(
            self.root / "fake_bwrap.py",
            "import sys\nsys.stderr.write('bwrap: setting up uid map: Permission denied\\n')\n"
            "raise SystemExit(1)\n",
        )
        policy = SandboxPolicy(
            package_root=self.package.root,
            workspace=self.root / "ws-broken",
            bwrap=fake,
        )
        with self.assertRaises(SandboxUnavailable) as caught:
            run_sandboxed(policy, 'print("never runs")')
        message = str(caught.exception)
        self.assertIn("probe", message)
        self.assertIn("Permission denied", message)

    def test_a_planted_program_symlink_cannot_redirect_the_next_host_write(self):
        # P0 regression: generated code may unlink program.py and leave a symlink
        # to a host path. The next host-side program write must replace the link,
        # never write through it.
        host_target = self.root / "host-file.txt"
        host_target.write_text("host original", encoding="utf-8")
        first = (
            "import os\n"
            "os.unlink('/workspace/program.py')\n"
            f"os.symlink({str(host_target)!r}, '/workspace/program.py')\n"
            "print('planted')\n"
        )
        planted = self.run_source(first)
        self.assertEqual(planted.exit_code, 0, planted.stderr)
        self.assertIn("planted", planted.stdout)
        self.assertTrue((self.workspace / "program.py").is_symlink())

        second = self.run_source("print('second action ran')")
        self.assertEqual(second.exit_code, 0, second.stderr)
        self.assertIn("second action ran", second.stdout)
        self.assertEqual(host_target.read_text(encoding="utf-8"), "host original")
        self.assertFalse((self.workspace / "program.py").is_symlink())

    def test_a_planted_result_symlink_cannot_be_read_from_the_host(self):
        # P0 regression: result.json is host-read. A symlink there must not turn
        # into a host file read or a final answer taken from outside the sandbox.
        host_secret = self.root / "host-secret.json"
        host_secret.write_text(json.dumps({"value": "host secret"}), encoding="utf-8")
        executor = SandboxedPythonExecutor(self.policy, deadline_seconds=30)
        executor.send_tools(["run_oneshot_preview"])
        source = (
            "import os\n"
            f"os.symlink({str(host_secret)!r}, '/workspace/result.json')\n"
            "print('planted result link')\n"
        )
        output = executor(source)
        self.assertFalse(output.is_final_answer)
        self.assertNotIn("host secret", str(output.output))
        self.assertIn("regular file", str(output.output))
        self.assertEqual(
            json.loads(host_secret.read_text(encoding="utf-8")), {"value": "host secret"}
        )
        self.assertFalse((self.workspace / "result.json").exists())
        self.assertFalse((self.workspace / "result.json").is_symlink())

    def test_a_directory_at_the_result_path_cannot_crash_the_executor(self):
        executor = SandboxedPythonExecutor(self.policy, deadline_seconds=30)
        executor.send_tools(["run_oneshot_preview"])
        planted = executor("import os\nos.mkdir('/workspace/result.json')")
        self.assertFalse(planted.is_final_answer)
        following = executor("print('next action ran')")
        self.assertFalse(following.is_final_answer)
        self.assertIn("next action ran", str(following.output))

    def test_runaway_output_is_stopped_instead_of_buffered(self):
        policy = SandboxPolicy(
            package_root=self.package.root,
            workspace=self.root / "ws-output",
            max_output_chars=200,
            max_capture_bytes=64 * 1024,
        )
        source = "import sys\nwhile True:\n    sys.stdout.write('x' * 4096)\n"
        result = run_sandboxed(policy, source, timeout=30)
        self.assertTrue(result.output_limit_exceeded)
        self.assertLessEqual(len(result.stdout), 400)
        self.assertLess(result.duration_seconds, 20)

    def test_a_fifo_result_file_is_rejected_without_blocking_the_host(self):
        # P1 regression: generated code can call os.mkfifo('/workspace/result.json')
        # and exit 0. Opening a FIFO for reading blocks until a writer appears, so a
        # plain O_RDONLY open would hang the host past every sandbox deadline. The
        # read runs in a child process with its own timeout: on the unfixed code this
        # test fails on the child's timeout instead of hanging the whole suite.
        self.workspace.mkdir(parents=True, exist_ok=True)
        fifo = self.workspace / "result.json"
        os.mkfifo(fifo)
        script = (
            "import sys\n"
            "from distribute_agent.sandbox import SandboxResultRejected, read_result_bytes\n"
            "try:\n"
            "    read_result_bytes(sys.argv[1])\n"
            "except SandboxResultRejected as exc:\n"
            "    print('rejected', exc)\n"
            "else:\n"
            "    print('read without rejection')\n"
            "    raise SystemExit(3)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(fifo)],
            cwd=str(PROTOTYPE_DIR),
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("rejected", completed.stdout)

    def test_oversized_result_files_are_rejected_before_parsing(self):
        policy = SandboxPolicy(package_root=self.package.root, workspace=self.root / "ws-result")
        executor = SandboxedPythonExecutor(
            policy, deadline_seconds=30, max_result_bytes=1024
        )
        executor.send_tools(["run_oneshot_preview"])
        source = (
            "import json\n"
            "with open('/workspace/result.json', 'w', encoding='utf-8') as handle:\n"
            "    handle.write(json.dumps({'value': 'y' * 4096}))\n"
        )
        output = executor(source)
        self.assertFalse(output.is_final_answer)
        self.assertIn("cap", str(output.output))


class AgentBoundaryTests(unittest.TestCase):
    """E2/E3/E4 and A1: what generated code can and cannot do during a run."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.package = Package.load(copy_package(self.root / "package"))
        self.skill = load_skill(self.package.skill_path())

    def make_agent(self, replies, *, workspace="run", max_steps=4, deadline=60, per_call=10,
                   plan_manifest=None):
        model = ScriptedModel(replies)
        agent = build_agent(
            model=model,
            skill=self.skill,
            package=self.package,
            workspace_dir=self.root / workspace,
            max_steps=max_steps,
            deadline_seconds=deadline,
            per_call_timeout=per_call,
            plan_manifest=plan_manifest,
        )
        return agent, model

    def test_plan_runs_inject_the_flat_schema_and_accept_a_flat_object(self):
        flat = {
            "template": "unholy-sdxl",
            "prompt": "quiet",
            "seed": 42,
            "width": 832,
            "height": 1216,
        }
        script = code_reply(
            'preview = run_oneshot_preview(template="unholy-sdxl", prompt="quiet", '
            'seed=42, width=832, height=1216)\n'
            f"final_answer({flat!r})"
        )
        agent, model = self.make_agent([script], plan_manifest=self.package.manifest)
        result = run_agent(agent, "Plan the quiet image.", deadline_seconds=60)

        self.assertEqual(result.answer, flat)
        self.assertEqual(len(model.calls), 1)
        system_prompt = message_text(model.calls[0][0])
        schema = json.dumps(plan_schema(self.package.manifest), indent=2)
        self.assertIn(schema, system_prompt)
        self.assertIn("negative_prompt", system_prompt)
        self.assertIn("no `plan` wrapper", system_prompt)

    def test_native_checks_reject_impostor_answers_and_the_loop_repairs_them(self):
        # Each first answer is a realistic model mistake; the repair reply runs only
        # after the model has seen SmolAgents' native check failure in memory.
        cases = {
            "nested-plan": (
                'final_answer({"plan": {"template": "unholy-sdxl", "prompt": "quiet"}})',
                "unsupported plan field(s): plan",
            ),
            "preview-metadata": (
                'preview = run_oneshot_preview(template="unholy-sdxl", prompt="quiet", '
                'seed=42, width=832, height=1216)\nfinal_answer(preview)',
                "unsupported plan field(s):",
            ),
            "negative-prompt-alias": (
                'final_answer({"template": "unholy-sdxl", "negative_prompt": "x"})',
                "unsupported plan field(s): negative_prompt",
            ),
            "range": (
                'final_answer({"template": "unholy-sdxl", "width": 7})',
                "plan field 'width' must be a multiple of 8",
            ),
            "type": (
                'final_answer({"template": "unholy-sdxl", "seed": "42"})',
                "plan field 'seed' must be an integer or omitted",
            ),
        }
        for label, (impostor, reason) in cases.items():
            with self.subTest(case=label):
                seen = {}

                def repair(messages):
                    feedback = "\n".join(
                        message_text(message) for message in messages
                    )
                    seen["feedback"] = feedback
                    if reason not in feedback:
                        return code_reply(
                            'final_answer({"template": "unholy-sdxl", "width": 7})'
                        )
                    return code_reply(
                        'final_answer({"template": "unholy-sdxl", "prompt": "quiet", '
                        '"seed": 42, "width": 832, "height": 1216})'
                    )

                agent, model = self.make_agent(
                    [code_reply(impostor), repair],
                    workspace=f"repair-{label}",
                    max_steps=3,
                    plan_manifest=self.package.manifest,
                )
                result = run_agent(agent, "Plan the quiet image.", deadline_seconds=60)

                self.assertEqual(result.answer["template"], "unholy-sdxl")
                self.assertEqual(len(model.calls), 2)
                self.assertIn("Check parse_plan_check failed with error:", seen["feedback"])
                self.assertIn(reason, seen["feedback"])
                self.assertTrue(
                    any(getattr(step, "error", None) for step in agent.memory.steps)
                )

    def test_a_run_without_the_plan_contract_can_answer_in_prose(self):
        refusal = (
            "unknown template 'not-a-shipped-template'; the package ships unholy-sdxl"
        )
        agent, model = self.make_agent([code_reply(f"final_answer({refusal!r})")])
        result = run_agent(agent, "Plan with an unknown template.", deadline_seconds=60)

        self.assertEqual(result.answer, refusal)
        self.assertEqual(agent.final_answer_checks, [])
        self.assertNotIn("Final answer: one flat plan", message_text(model.calls[0][0]))

    def test_an_invalid_max_step_fallback_is_rejected_by_the_host(self):
        # SmolAgents' _handle_max_steps_reached bypasses final_answer_checks, so the
        # host boundary is the last line of defence for a strict plan run.
        secret = "sk-live-secret-abcdef"
        fallback = f"not a plan: no template {secret}"
        agent, model = self.make_agent(
            [code_reply('print("working")'), fallback],
            workspace="fallback-invalid",
            max_steps=1,
            plan_manifest=self.package.manifest,
        )
        with self.assertRaises(AgentRunError) as caught:
            run_agent(agent, "Plan the image.", deadline_seconds=60)

        diagnostics = caught.exception.diagnostics
        self.assertTrue(agent.final_answer_checks)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("invalid plan", str(caught.exception))
        self.assertIn("state max_steps_error", str(caught.exception))
        self.assertIn("plan must be a JSON object, got str", str(caught.exception))
        self.assertEqual(diagnostics["state"], "max_steps_error")
        self.assertNotIn(secret, json.dumps(diagnostics))
        self.assertIn("[redacted]", diagnostics["final_answer"])

    def test_the_host_does_not_unwrap_a_json_string_plan(self):
        flat = {
            "template": "unholy-sdxl",
            "prompt": "quiet",
            "seed": 42,
            "width": 832,
            "height": 1216,
        }
        agent, _model = self.make_agent(
            [code_reply('print("working")'), json.dumps(flat)],
            workspace="fallback-json-string",
            max_steps=1,
            plan_manifest=self.package.manifest,
        )
        with self.assertRaises(AgentRunError) as caught:
            run_agent(agent, "Plan the image.", deadline_seconds=60)

        self.assertIn("plan must be a JSON object, got str", str(caught.exception))

    def test_dry_run_writes_only_the_program_and_no_key_is_visible(self):
        script = code_reply(
            'import os\n'
            'key = os.environ.get("RUNPOD_API_KEY")\n'
            'plan = run_oneshot_preview(template="unholy-sdxl", prompt="quiet", seed=42, width=832, height=1216)\n'
            'final_answer({"key": key, "exit_code": plan["exit_code"], '
            '"prompt": plan["envelope"]["input"]["workflow"]["2"]["inputs"]["text"]})'
        )
        agent, _model = self.make_agent([script])
        before = tree_digest(self.package.root)
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "sk-host-secret"}, clear=False):
            result = run_agent(agent, "Plan a quiet dry run.", deadline_seconds=60)
        self.assertEqual(result.answer["key"], None)
        self.assertEqual(result.answer["exit_code"], 0)
        self.assertEqual(result.answer["prompt"], "quiet")
        workspace = self.root / "run" / "workspace"
        self.assertEqual([path.name for path in workspace.iterdir()], ["program.py"])
        self.assertFalse((workspace / "out").exists())
        self.assertEqual(tree_digest(self.package.root), before)

    def test_shell_metacharacters_stay_data(self):
        prompt = 'a $(touch /workspace/pwned) `id` "quoted" & ; \\ newline\nline two — ✓'
        agent, _model = self.make_agent([dry_run_script(prompt)])
        result = run_agent(agent, "Plan with an unusual prompt.", deadline_seconds=60)
        plan = result.answer
        self.assertEqual(plan["exit_code"], 0, plan["stderr"])
        self.assertEqual(
            plan["envelope"]["input"]["workflow"]["2"]["inputs"]["text"], prompt
        )
        self.assertEqual(plan["argv"][0], "--template")
        self.assertIn(prompt, plan["argv"])
        self.assertFalse(list(self.root.rglob("pwned")))

    def test_wall_clock_budget_stops_runaway_generated_code(self):
        agent, _model = self.make_agent(
            [code_reply("import time\ntime.sleep(10)\nfinal_answer('never')")],
            deadline=1.0,
            per_call=30,
        )
        started = __import__("time").monotonic()
        with self.assertRaises(AgentRunError) as caught:
            run_agent(agent, "Sleep forever.", deadline_seconds=1.0)
        elapsed = __import__("time").monotonic() - started
        self.assertLess(elapsed, 6.0)
        self.assertTrue(caught.exception.diagnostics["deadline_exceeded"])
        self.assertIn("budget", str(caught.exception))

    def test_max_steps_bounds_the_agent_loop(self):
        replies = [code_reply('print("working")') for _ in range(2)]
        replies.append("Stopped by the step budget.")
        agent, model = self.make_agent(replies, max_steps=2)
        result = run_agent(agent, "Keep working forever.", deadline_seconds=60)
        self.assertEqual(result.state, "max_steps_error")
        self.assertEqual(result.diagnostics["executor_calls"], 2)
        self.assertEqual(result.answer, "Stopped by the step budget.")
        self.assertEqual(len(model.calls), 3)  # two actions + the final-answer prompt

    def test_each_code_action_runs_in_a_fresh_process_and_the_prompt_says_so(self):
        # P1 regression: SmolAgents promises persistent variables unless the
        # prompt is overridden. The executor starts a fresh interpreter per
        # action, so the contract the model is told must match the executor.
        first = code_reply(
            'preview = run_oneshot_preview(template="unholy-sdxl", prompt="quiet", '
            'seed=42, width=832, height=1216)\n'
            'print(preview["exit_code"])'
        )
        replies = [
            first,
            code_reply("final_answer(preview)"),
            code_reply("final_answer({'recovered': True})"),
        ]
        agent, model = self.make_agent(replies)
        result = run_agent(agent, "Plan then answer.", deadline_seconds=60)

        self.assertEqual(result.answer, {"recovered": True})
        system_prompt = message_text(model.calls[0][0])
        self.assertNotIn("The state persists between code executions", system_prompt)
        self.assertIn("fresh Python process", system_prompt)
        observations = [
            step.observations for step in agent.memory.steps
            if getattr(step, "observations", None)
        ]
        self.assertTrue(
            any("NameError" in observation and "preview" in observation
                for observation in observations),
            observations,
        )

    def test_model_request_budget_stops_the_run_before_an_extra_provider_call(self):
        replies = [code_reply('print("working")'), code_reply('print("working")'), "done"]
        model = ScriptedModel(replies)
        agent = build_agent(
            model=model,
            skill=self.skill,
            package=self.package,
            workspace_dir=self.root / "run-budget",
            max_steps=4,
            deadline_seconds=60,
            max_model_requests=1,
        )
        with self.assertRaises(AgentRunError) as caught:
            run_agent(agent, "Keep working forever.", deadline_seconds=60)
        self.assertIn("request budget", str(caught.exception))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(caught.exception.diagnostics["model_requests"], 1)

    def test_a_stalled_model_call_cannot_outlive_the_run_budget(self):
        def stall(_messages):
            __import__("time").sleep(10)
            return code_reply("final_answer('too late')")

        agent, model = self.make_agent([stall], deadline=1.0, per_call=30)
        started = __import__("time").monotonic()
        with self.assertRaises(AgentRunError) as caught:
            run_agent(agent, "Take forever.", deadline_seconds=1.0)
        elapsed = __import__("time").monotonic() - started
        self.assertLess(elapsed, 5.0)
        self.assertIn("wall-clock budget", str(caught.exception))
        self.assertEqual(len(model.calls), 1)

    def test_failure_diagnostics_are_redacted_and_useful(self):
        secret = "sk-live-secret-abcdef"
        script = code_reply(f'secret = "{secret}"\nraise RuntimeError("boom {secret}")')
        agent, _model = self.make_agent([script])
        with self.assertRaises(AgentRunError) as caught:
            run_agent(agent, "Fail on purpose.", deadline_seconds=60)
        diagnostics = caught.exception.diagnostics
        rendered = json.dumps(diagnostics)
        self.assertNotIn(secret, rendered)
        self.assertIn("[redacted]", diagnostics["last_code"])
        self.assertGreaterEqual(diagnostics["executor_calls"], 1)
        self.assertIn("bwrap", diagnostics["sandbox"])
        self.assertIn("duration_seconds", diagnostics)

    def test_redact_removes_credential_shapes(self):
        text = "RUNPOD_API_KEY=rpa_live_123 Bearer abc.def sk-live-123456 hf_abcdefghij"
        cleaned = redact(text)
        for secret in ("rpa_live_123", "abc.def", "sk-live-123456", "hf_abcdefghij"):
            self.assertNotIn(secret, cleaned)

    def test_concurrent_runs_keep_workspaces_and_answers_isolated(self):
        prompts = ["first concurrent brief", "second concurrent brief"]
        before = tree_digest(self.package.root)

        def run(prompt, index):
            model = ScriptedModel([dry_run_script(prompt)])
            agent = build_agent(
                model=model,
                skill=self.skill,
                package=self.package,
                workspace_dir=self.root / f"concurrent-{index}",
                deadline_seconds=120,
            )
            return run_agent(agent, "Plan the dry run.", deadline_seconds=120)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda pair: run(pair[1], pair[0]), enumerate(prompts)))
        for prompt, result in zip(prompts, results):
            workflow = result.answer["envelope"]["input"]["workflow"]
            self.assertEqual(workflow["2"]["inputs"]["text"], prompt)
        workspace_dirs = {result.diagnostics["sandbox"]["workspace"] for result in results}
        self.assertEqual(len(workspace_dirs), 2)
        self.assertEqual(tree_digest(self.package.root), before)


class LiveHarnessTests(unittest.TestCase):
    """A3: semantic evaluation is opt-in and explicitly not run here."""

    def test_live_evaluation_refuses_without_explicit_authorization(self):
        calls = []

        def factory():
            calls.append(1)
            raise AssertionError("no model may be constructed without the flag")

        report = run_live_eval(enable_live_model=False, model_factory=factory)
        self.assertEqual(report["status"], "not-run")
        self.assertIn("--enable-live-model", report["reason"])
        self.assertEqual(calls, [])
        self.assertEqual(report["repetitions"], DEFAULT_REPETITIONS)
        self.assertEqual(report["bounds"]["max_requests"], None)

    def test_the_brief_matrix_covers_the_required_five_cases(self):
        briefs = {brief.id: brief for brief in BRIEFS}
        self.assertEqual(
            set(briefs),
            {"simple", "explicit-settings", "unicode-quotes", "unknown-template",
             "unsupported-image"},
        )
        self.assertEqual(DEFAULT_REPETITIONS, 3)
        explicit = briefs["explicit-settings"]
        self.assertEqual(explicit.expected_settings, {"seed": 42, "width": 832, "height": 1216})
        self.assertTrue(briefs["unknown-template"].expect_rejection)
        self.assertTrue(briefs["unsupported-image"].expect_rejection)
        self.assertFalse(explicit.expect_rejection)

    def test_a_plan_that_ignores_the_brief_fails_the_semantic_checks(self):
        # The old checks passed any valid plan. A valid plan for an unrelated
        # image must fail, and the exact Unicode prompt must match verbatim.
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            briefs = {brief.id: brief for brief in BRIEFS}
            base = {"template": "unholy-sdxl", "seed": 42, "width": 832, "height": 1216}

            ignored = _check_answer(briefs["simple"], {**base, "prompt": "banana"}, package)
            self.assertFalse(all(check["ok"] for check in ignored), ignored)

            thin = _check_answer(
                briefs["simple"],
                {**base, "prompt": "lighthouse clock cliff"},
                package,
            )
            self.assertFalse(all(check["ok"] for check in thin), thin)

            no_action = _check_answer(
                briefs["simple"],
                {**base, "prompt": "a lighthouse keeper beside a brass clock on a cliff"},
                package,
            )
            self.assertFalse(all(check["ok"] for check in no_action), no_action)

            faithful = _check_answer(
                briefs["simple"],
                {**base, "prompt": "a lighthouse keeper winding a brass clock on a cliff"},
                package,
            )
            self.assertTrue(all(check["ok"] for check in faithful), faithful)

            no_text_dropped = _check_answer(
                briefs["explicit-settings"],
                {**base, "prompt": "a rain-soaked cyberpunk alley at night, neon "
                 "reflections"},
                package,
            )
            self.assertFalse(all(check["ok"] for check in no_text_dropped), no_text_dropped)

            wrong_unicode = _check_answer(
                briefs["unicode-quotes"], {**base, "prompt": "a cafe terrace"}, package
            )
            self.assertFalse(all(check["ok"] for check in wrong_unicode), wrong_unicode)

            exact = _check_answer(
                briefs["unicode-quotes"],
                {**base, "prompt": "「雨の街」 — a café terrace, “warm light”, 35mm film"},
                package,
            )
            self.assertTrue(all(check["ok"] for check in exact), exact)

    def test_no_text_accepts_positive_exclusions_or_negative_prompt_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            brief = next(brief for brief in BRIEFS if brief.id == "explicit-settings")
            base = {
                "template": "unholy-sdxl", "seed": 42, "width": 832, "height": 1216,
                "prompt": "a rain-soaked cyberpunk alley at night, neon reflections",
            }
            cases = [
                {"prompt": base["prompt"] + ", " + phrase}
                for phrase in (
                    "holographic billboards without lettering", "NO TEXT",
                    "without text", "text-free", "no lettering", "no writing",
                    "without writing", "lettering-free",
                )
            ]
            cases.extend({"negative": negative} for negative in (
                "text, watermark, signature, letters, words, captions, logo",
                "text, letters, words, watermark, signature, logo, writing",
                "watermark, TEXT , signature", "letters", "words", "lettering", "writing",
            ))
            for changes in cases:
                with self.subTest(changes=changes):
                    checks = _check_answer(brief, {**base, **changes}, package)
                    self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_no_text_rejects_missing_exclusions_and_similar_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            brief = next(brief for brief in BRIEFS if brief.id == "explicit-settings")
            base = {
                "template": "unholy-sdxl", "seed": 42, "width": 832, "height": 1216,
                "prompt": "a rain-soaked cyberpunk alley at night, neon reflections",
            }
            cases = (
                {},
                {"negative": ""},
                {"negative": None},
                {"negative": "watermark, logo"},
                {"negative": "texture, context"},
                {"negative": "no text"},
                {"prompt": base["prompt"] + ", no texture"},
                {"prompt": base["prompt"] + ", context-free lighting"},
                {"prompt": base["prompt"] + ", text on signs"},
            )
            for changes in cases:
                with self.subTest(changes=changes):
                    checks = _check_answer(brief, {**base, **changes}, package)
                    self.assertFalse(all(check["ok"] for check in checks), checks)

    def test_exclusions_do_not_relax_the_scene_or_settings_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            brief = next(brief for brief in BRIEFS if brief.id == "explicit-settings")
            base = {
                "template": "unholy-sdxl", "seed": 42, "width": 832, "height": 1216,
                "prompt": "a rain-soaked cyberpunk alley at night, neon reflections",
                "negative": "text",
            }
            cases = (
                {"prompt": "a sunny beach", "negative": "cyberpunk, alley, text"},
                {"seed": 43},
                {"width": 1024},
                {"height": 1024},
            )
            for changes in cases:
                with self.subTest(changes=changes):
                    checks = _check_answer(brief, {**base, **changes}, package)
                    self.assertFalse(all(check["ok"] for check in checks), checks)

    def test_rejection_briefs_need_a_relevant_actionable_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            briefs = {brief.id: brief for brief in BRIEFS}

            for answer in ("", "I could not do that.", None):
                with self.subTest(answer=answer):
                    checks = _check_answer(briefs["unknown-template"], answer, package)
                    self.assertFalse(all(check["ok"] for check in checks), checks)

            checks = _check_answer(
                briefs["unknown-template"],
                "unknown template 'not-a-shipped-template'; the package ships unholy-sdxl",
                package,
            )
            self.assertTrue(all(check["ok"] for check in checks), checks)

            submittable = {
                "template": "unholy-sdxl", "prompt": "made up", "seed": 1,
                "width": 832, "height": 1216,
            }
            checks = _check_answer(briefs["unsupported-image"], submittable, package)
            self.assertFalse(all(check["ok"] for check in checks), checks)

            checks = _check_answer(
                briefs["unsupported-image"],
                "unholy-sdxl is text-only and does not accept an image input",
                package,
            )
            self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_preview_execution_requires_a_successful_preview_of_the_returned_plan(self):
        # Finding 2 regression: a source substring plus any executor call is not
        # preview evidence. Each case below runs the real prelude and the real
        # runner inside the sandbox; only a successful preview whose request matches
        # the returned plan passes. The record is sandbox-authored, so this is
        # invocation/result evidence, not tamperproof attestation.
        prompt = "a lighthouse keeper winding a brass clock on a stormy cliff, no text"
        plan = {
            "template": "unholy-sdxl",
            "prompt": prompt,
            "seed": 42,
            "width": 832,
            "height": 1216,
        }
        plan_source = repr(plan)
        preview = (
            "run_oneshot_preview(template='unholy-sdxl', prompt=" + repr(prompt)
            + ", seed=42, width=832, height=1216)"
        )
        cases = {
            "comment-only": (
                "# run_oneshot_preview(template='unholy-sdxl')\n"
                f"final_answer({plan_source})",
                False,
            ),
            "unexecuted-branch": (
                "if False:\n"
                f"    {preview}\n"
                f"final_answer({plan_source})",
                False,
            ),
            "failed-preview": (
                "run_oneshot_preview(template='not-a-shipped-template', prompt="
                + repr(prompt) + ")\n"
                + f"final_answer({plan_source})",
                False,
            ),
            "mismatched-plan": (
                "run_oneshot_preview(template='unholy-sdxl', prompt='a different "
                "scene', seed=42, width=832, height=1216)\n"
                f"final_answer({plan_source})",
                False,
            ),
            "matching-preview": (
                f"{preview}\nfinal_answer({plan_source})",
                True,
            ),
        }
        briefs = {brief.id: brief for brief in BRIEFS}
        for label, (code, expected) in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = Package.load(copy_package(root / "package"))
                policy = SandboxPolicy(
                    package_root=package.root, workspace=root / "workspace"
                )
                executor = SandboxedPythonExecutor(
                    policy, deadline_seconds=120, per_call_timeout=120
                )
                executor.send_tools(["run_oneshot_preview"])
                executor(code)
                check = _preview_evidence(
                    briefs["simple"], plan, package, executor.preview_runs
                )
            self.assertEqual(check["ok"], expected, (label, check))

    def test_rejection_briefs_need_no_preview(self):
        # Finding 2: a correct refusal made straight from the supplied manifest must
        # pass even though no preview ran; those briefs never ask for one.
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            briefs = {brief.id: brief for brief in BRIEFS}
            answers = {
                "unknown-template": (
                    "unknown template 'not-a-shipped-template'; the package ships "
                    "unholy-sdxl"
                ),
                "unsupported-image": (
                    "unholy-sdxl is text-only and does not accept an image input"
                ),
            }
            for brief_id, answer in answers.items():
                with self.subTest(brief=brief_id):
                    checks = _check_answer(briefs[brief_id], answer, package)
                    checks.append(
                        _preview_evidence(briefs[brief_id], answer, package, [])
                    )
                    self.assertTrue(all(check["ok"] for check in checks), checks)

    def test_the_matrix_runs_offline_and_stops_at_the_request_cap(self):
        # No provider is called: the scripted model stands in for the LLM, so this
        # proves the harness wiring (checks, preview evidence, shared request
        # budget) without claiming a live semantic result.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            factory_calls = []

            prompt = "a lighthouse keeper winding a clock on a cliff"

            def factory():
                factory_calls.append(1)
                return ScriptedModel([code_reply(
                    "preview = run_oneshot_preview(\n"
                    '    template="unholy-sdxl",\n'
                    f"    prompt={prompt!r},\n"
                    "    seed=42,\n"
                    "    width=832,\n"
                    "    height=1216,\n"
                    ")\n"
                    f"final_answer({{'template': 'unholy-sdxl', 'prompt': {prompt!r}, "
                    "'seed': 42, 'width': 832, 'height': 1216})"
                )])

            report = run_live_eval(
                enable_live_model=True,
                model_factory=factory,
                package_root=package.root,
                workspace_root=root / "workspaces",
                repetitions=1,
                deadline_seconds=120,
                max_requests=1,
            )
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual(report["runs"][0]["status"], "pass", report["runs"][0])
        self.assertEqual(report["runs"][0]["model_requests"], 1)
        self.assertIn(
            "preview executed and matched",
            [check["name"] for check in report["runs"][0]["checks"]],
        )
        self.assertEqual(report["status"], "budget-exhausted")
        self.assertEqual(report["requests_used"], 1)

    def test_generation_briefs_are_strict_and_refusals_stay_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            briefs = {brief.id: brief for brief in BRIEFS}
            prompts = {
                "simple": "a lighthouse keeper winding a brass clock on a stormy cliff",
                "explicit-settings": (
                    "a rain-soaked cyberpunk alley at night, neon reflections, no text"
                ),
                "unicode-quotes": "「雨の街」 — a café terrace, “warm light”, 35mm film",
            }
            refusals = {
                "unknown-template": (
                    "unknown template 'not-a-shipped-template'; the package ships "
                    "unholy-sdxl"
                ),
                "unsupported-image": (
                    "unholy-sdxl is text-only and does not accept an image input"
                ),
            }

            def which_brief(task):
                if "「雨の街」" in task:
                    return "unicode-quotes"
                if "cyberpunk" in task:
                    return "explicit-settings"
                if "not-a-shipped-template" in task:
                    return "unknown-template"
                if "input.png" in task:
                    return "unsupported-image"
                return "simple"

            def reply(messages):
                task = next(
                    message_text(message) for message in messages
                    if getattr(message, "role", None) == "user"
                )
                name = which_brief(task)
                if name in refusals:
                    return code_reply(f"final_answer({refusals[name]!r})")
                plan = {
                    "template": "unholy-sdxl",
                    "prompt": prompts[name],
                    "seed": 42,
                    "width": 832,
                    "height": 1216,
                }
                return code_reply(
                    f"preview = run_oneshot_preview(**{plan!r})\n"
                    f"final_answer({plan!r})"
                )

            contracts = {}
            real_build = agent_module.build_agent

            def recording_build(**kwargs):
                contracts[Path(kwargs["workspace_dir"]).name] = kwargs.get("plan_manifest")
                return real_build(**kwargs)

            with mock.patch.object(agent_module, "build_agent", recording_build):
                report = run_live_eval(
                    enable_live_model=True,
                    model_factory=lambda: ScriptedModel([reply]),
                    package_root=package.root,
                    workspace_root=root / "workspaces",
                    repetitions=1,
                    max_steps=4,
                    deadline_seconds=120,
                    max_requests=10,
                )

            self.assertEqual(report["status"], "completed", report["runs"])
            for entry in report["runs"]:
                self.assertEqual(entry["status"], "pass", entry)
            self.assertEqual(set(contracts), {f"{brief.id}-1" for brief in BRIEFS})
            for workspace, manifest in contracts.items():
                name = workspace.rsplit("-", 1)[0]
                if briefs[name].expect_rejection:
                    self.assertIsNone(manifest)
                else:
                    self.assertEqual(manifest, package.manifest)

    def test_the_live_backend_is_preflighted_before_the_matrix(self):
        from smolagents import LiteLLMModel

        from distribute_agent.__main__ import _model_factory

        arguments = mock.Mock(
            model_backend="litellm",
            model_id="openai/gpt-4o-mini",
            api_key_env="HF_TOKEN",
        )
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with mock.patch.object(
                LiteLLMModel,
                "create_client",
                side_effect=ModuleNotFoundError("Please install 'litellm' extra"),
            ):
                with self.assertRaises(SystemExit) as caught:
                    _model_factory(arguments)
        self.assertIn("litellm", str(caught.exception))
        self.assertIn("No provider call was made", str(caught.exception))


class VendorSmokeTests(unittest.TestCase):
    """A4: a real vendor smoke test exists but is opt-in; its plumbing is tested offline."""

    def test_vendor_smoke_refuses_without_the_enable_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Package.load(copy_package(Path(tmp) / "package"))
            report = run_vendor_smoke(
                enable_live_vendor=False,
                package_root=package.root,
                output_dir=Path(tmp) / "out",
                env={},
            )
        self.assertEqual(report["status"], "not-run")
        self.assertIn("--enable-live-vendor", report["reason"])

    def test_vendor_smoke_requires_the_key_without_calling_the_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            marker = root / "runner-called"
            stub = write_stub_runner(
                root / "stub_runner.py", f"open({str(marker)!r}, 'w').write('yes')\n"
            )
            with self.assertRaises(VendorSmokeError) as caught:
                run_vendor_smoke(
                    enable_live_vendor=True,
                    package_root=package.root,
                    output_dir=root / "out",
                    env={},
                    runner_prefix=[sys.executable, str(stub)],
                )
            self.assertIn("RUNPOD_API_KEY is not set", str(caught.exception))
            self.assertFalse(marker.exists())

    def test_vendor_smoke_checks_one_job_and_the_artifact_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            stub_body = (
                "import sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "out = Path(args[args.index('--output') + 1]) / 'job-smoke'\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'image.png').write_bytes(" + repr(PNG_BYTES) + ")\n"
                "(out / 'response.json').write_text('{}', encoding='utf-8')\n"
                "print(out)\n"
            )
            stub = write_stub_runner(root / "stub_runner.py", stub_body)
            report = run_vendor_smoke(
                enable_live_vendor=True,
                package_root=package.root,
                output_dir=root / "out",
                env={"RUNPOD_API_KEY": "rpa_test_secret_value"},
                runner_prefix=[sys.executable, str(stub)],
                plan_data={
                    "template": "unholy-sdxl",
                    "prompt": "smoke",
                    "seed": 42,
                    "width": 832,
                    "height": 1216,
                },
            )
            self.assertEqual(report["status"], "passed", report)
            self.assertEqual(report["job_id"], "job-smoke")
            self.assertEqual(report["artifact"]["format"], "png")
            self.assertEqual((report["artifact"]["width"], report["artifact"]["height"]), (1, 1))

    def run_smoke_with_artifact(self, root, image_bytes):
        package = Package.load(copy_package(root / "package"))
        stub_body = (
            "import sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "out = Path(args[args.index('--output') + 1]) / 'job-smoke'\n"
            "out.mkdir(parents=True, exist_ok=True)\n"
            "(out / 'image.png').write_bytes(" + repr(image_bytes) + ")\n"
            "print(out)\n"
        )
        stub = write_stub_runner(root / "stub_runner.py", stub_body)
        return run_vendor_smoke(
            enable_live_vendor=True,
            package_root=package.root,
            output_dir=root / "out",
            env={"RUNPOD_API_KEY": "rpa_test_secret_value"},
            runner_prefix=[sys.executable, str(stub)],
            plan_data={
                "template": "unholy-sdxl",
                "prompt": "smoke",
                "seed": 42,
                "width": 832,
                "height": 1216,
            },
        )

    def test_truncated_and_corrupt_artifacts_fail_the_image_check(self):
        corrupt = bytearray(PNG_BYTES)
        corrupt[len(PNG_BYTES) - 20] ^= 0xFF  # flip a byte inside IDAT's payload
        cases = {
            "header-only": PNG_BYTES[:24],
            "truncated": PNG_BYTES[:-12],
            "corrupt-idat": bytes(corrupt),
            "not-an-image": b"<html>error</html>",
        }
        for label, payload in cases.items():
            with self.subTest(case=label):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    report = self.run_smoke_with_artifact(root, payload)
                self.assertEqual(report["status"], "failed", report)
                self.assertTrue(report["failures"])
            self.assertNotIn("rpa_test_secret_value", json.dumps(report))



    def test_artifacts_must_decode_to_pixels_not_just_match_structure(self):
        # Finding 3 regression: correct chunk CRCs over an empty IDAT, and a JPEG
        # that stops after its frame header, are structurally valid and contain no
        # image. The smoke check now decodes locally through Pillow (bounded pixels,
        # no display, no model) and must reject both while accepting real images.
        rejected = {
            "empty-idat": build_empty_idat_png(),
            "invalid-idat": build_invalid_idat_png(),
            "frame-header-only-jpeg": build_frame_header_only_jpeg(),
        }
        for label, payload in rejected.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                report = self.run_smoke_with_artifact(Path(tmp), payload)
            self.assertEqual(report["status"], "failed", report)
            self.assertTrue(
                any("decode" in failure for failure in report["failures"]),
                report["failures"],
            )

        accepted = {
            "png": (PNG_BYTES, "png", 1, 1),
            "jpeg": (build_jpeg(2, 3), "jpeg", 2, 3),
        }
        for label, (payload, image_format, width, height) in accepted.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                report = self.run_smoke_with_artifact(Path(tmp), payload)
            self.assertEqual(report["status"], "passed", report)
            artifact = report["artifact"]
            self.assertEqual(
                (artifact["format"], artifact["width"], artifact["height"]),
                (image_format, width, height),
            )

    def test_decoding_is_byte_and_pixel_bounded(self):
        # The decoder runs on untrusted vendor bytes: the 64 MiB input cap stays and
        # an oversized canvas is refused from its header, before any pixel buffer is
        # allocated.
        bomb = (
            b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8192, 8192, 8, 2, 0, 0, 0))
            + png_chunk(b"IDAT", zlib.compress(b"\x00" * 16))
            + png_chunk(b"IEND", b"")
        )
        with self.assertRaises(VendorSmokeError) as caught:
            sniff_image(bomb)
        self.assertIn("pixels", str(caught.exception))

        oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_ARTIFACT_BYTES
        with self.assertRaises(VendorSmokeError) as caught:
            sniff_image(oversized)
        self.assertIn("cap", str(caught.exception))


class OptInCliTests(unittest.TestCase):
    """The enable flags are CLI-level too, not just Python arguments."""

    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "distribute_agent", *arguments],
            cwd=PROTOTYPE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_live_eval_cli_returns_not_run_without_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = self.run_cli(
                "live-eval",
                "--package", str(PROJECT_ROOT / "distribute"),
                "--workspace", str(Path(tmp) / "workspace"),
            )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("not-run", completed.stdout)
        self.assertIn("--enable-live-model", completed.stdout)

    def test_vendor_smoke_cli_returns_not_run_without_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = self.run_cli(
                "vendor-smoke",
                "--package", str(PROJECT_ROOT / "distribute"),
                "--output", str(Path(tmp) / "output"),
            )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("not-run", completed.stdout)
        self.assertIn("--enable-live-vendor", completed.stdout)



class BridgeTests(unittest.TestCase):
    """A1/A2/V2: one host-side invocation, a minimal child environment, no leak."""

    def test_bridge_runs_the_runner_once_with_no_host_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            record = root / "child-env.json"
            calls = root / "calls.txt"
            stub = write_stub_runner(
                root / "stub_runner.py",
                "import json, os, sys\n"
                "from pathlib import Path\n"
                f"Path({str(calls)!r}).open('a').write('x')\n"
                f"Path({str(record)!r}).write_text(json.dumps(sorted(os.environ)))\n"
                "args = sys.argv[1:]\n"
                "out = Path(args[args.index('--output') + 1]) / 'job-bridge'\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'response.json').write_text('{}')\n"
                "print(out)\n",
            )
            result = submit_plan(
                package,
                {"template": "unholy-sdxl", "prompt": "bridge", "seed": 42,
                 "width": 832, "height": 1216},
                output_dir=root / "out",
                env={"RUNPOD_API_KEY": "rpa_bridge_secret", "AWS_SECRET_ACCESS_KEY": "aws"},
                runner_prefix=[sys.executable, str(stub)],
            )
            self.assertEqual(result.job_id, "job-bridge")
            self.assertEqual(calls.read_text(encoding="utf-8"), "x")
            child_env = set(json.loads(record.read_text(encoding="utf-8")))
            # CPython adds LC_CTYPE when it coerces the locale; nothing else is inherited.
            self.assertEqual(child_env - {"LC_CTYPE"},
                             {"PATH", "RUNPOD_API_KEY", "HOME", "NO_COLOR"})
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", child_env)

    def test_ambiguous_submission_failure_is_not_retried_and_hides_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            calls = root / "calls.txt"
            stub = write_stub_runner(
                root / "stub_runner.py",
                "import os, sys\n"
                "open(" + repr(str(calls)) + ", 'a').write('x')\n"
                "sys.stderr.write('[FAIL] network error submitting: timed out — the job MAY '\n"
                "                 'have been accepted anyway — submission outcome is UNKNOWN; '\n"
                "                 'key=' + os.environ['RUNPOD_API_KEY'] + '\\n')\n"
                "raise SystemExit(1)\n",
            )
            with self.assertRaises(BridgeError) as caught:
                submit_plan(
                    package,
                    {"template": "unholy-sdxl", "prompt": "bridge"},
                    output_dir=root / "out",
                    env={"RUNPOD_API_KEY": "rpa_bridge_secret"},
                    runner_prefix=[sys.executable, str(stub)],
                )
            self.assertTrue(caught.exception.may_have_submitted)
            self.assertNotIn("rpa_bridge_secret", str(caught.exception))
            self.assertEqual(calls.read_text(encoding="utf-8"), "x")

    def test_polling_failure_after_a_job_id_is_reported_as_submitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            stub = write_stub_runner(
                root / "stub_runner.py",
                "import sys\n"
                "sys.stderr.write('[info] job job-77 accepted\\n')\n"
                "sys.stderr.write('[FAIL] network error on GET https://api/status/job-77: reset\\n')\n"
                "raise SystemExit(1)\n",
            )
            with self.assertRaises(BridgeError) as caught:
                submit_plan(
                    package,
                    {"template": "unholy-sdxl", "prompt": "bridge"},
                    output_dir=root / "out",
                    env={"RUNPOD_API_KEY": "rpa_bridge_secret"},
                    runner_prefix=[sys.executable, str(stub)],
                )
            self.assertTrue(caught.exception.may_have_submitted)
            self.assertEqual(caught.exception.job_id, "job-77")
            self.assertIn("job-77", str(caught.exception))

    def test_an_unusable_post_response_is_reported_as_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            stub = write_stub_runner(
                root / "stub_runner.py",
                "import sys\n"
                "sys.stderr.write('[FAIL] malformed JSON from POST https://api/run: bad; "
                "submission outcome is UNKNOWN\\n')\n"
                "raise SystemExit(1)\n",
            )
            with self.assertRaises(BridgeError) as caught:
                submit_plan(
                    package,
                    {"template": "unholy-sdxl", "prompt": "bridge"},
                    output_dir=root / "out",
                    env={"RUNPOD_API_KEY": "rpa_bridge_secret"},
                    runner_prefix=[sys.executable, str(stub)],
                )
            self.assertTrue(caught.exception.may_have_submitted)
            self.assertIsNone(caught.exception.job_id)

    def test_a_rejected_submission_is_not_reported_as_submitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = Package.load(copy_package(root / "package"))
            stub = write_stub_runner(
                root / "stub_runner.py",
                "import sys\n"
                "sys.stderr.write('[FAIL] authentication failed (HTTP 401) on POST https://api/run\\n')\n"
                "raise SystemExit(1)\n",
            )
            with self.assertRaises(BridgeError) as caught:
                submit_plan(
                    package,
                    {"template": "unholy-sdxl", "prompt": "bridge"},
                    output_dir=root / "out",
                    env={"RUNPOD_API_KEY": "rpa_bridge_secret"},
                    runner_prefix=[sys.executable, str(stub)],
                )
            self.assertFalse(caught.exception.may_have_submitted)
            self.assertIsNone(caught.exception.job_id)



if __name__ == "__main__":
    unittest.main()
