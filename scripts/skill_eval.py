#!/usr/bin/env python3
"""Run skill evaluation cases with a configurable command-line agent.

The test-case files use a deliberately small YAML subset so this runner stays
Python-stdlib-only. Supported YAML is a top-level sequence of case mappings
with these keys: ``id``, ``name``, ``prompt``, ``assertions``,
``expected_behavior``, and ``edge_case``. ``assertions`` is a nested sequence
of mappings with only ``type``, ``target``, ``description``, ``critical``, and
``weight`` keys. Scalars may be plain, single-quoted, double-quoted (including
standard YAML escapes), or literal/folded block scalars (``|``/``>``) with an
optional indentation indicator and chomping modifier. Unsupported YAML raises
``file:line`` errors instead of being guessed at.

Supported assertion types: ``contains``, ``not_contains``, ``regex``,
``question_before_code``, ``json_valid``, ``json_fields``/``json_schema``,
``token_limit``, ``range_check``, ``structure_check``, ``sequence_check``, and
``SKIP`` for unknown types. ``structure_check`` and ``sequence_check`` are
recorded as ``SOFT`` because they need a human or model judge; they are
reported in the scorecard but never scored as pass or fail.

The evaluator is agent-agnostic. Set ``EVAL_AGENT_CMD`` or pass
``--agent-cmd`` to select the executable command. ``{prompt}`` and
``{prompt_file}`` placeholders are expanded in individual command arguments;
without either placeholder, the prompt is appended as the final argument.

Exit codes: ``0`` when the run completes, even if assertions fail (failures
live in the scorecard); ``2`` for harness errors such as unreadable or invalid
YAML, no test cases found, or a missing or non-executable agent command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT_COMMAND = "claude -p"
TOP_LEVEL_KEYS = {
    "id",
    "name",
    "prompt",
    "assertions",
    "expected_behavior",
    "edge_case",
}
ASSERTION_KEYS = {"type", "target", "description", "critical", "weight"}
_UNSET = object()
_MISSING = object()


class HarnessError(Exception):
    """An error that should be shown to a runner user without a traceback."""


class YamlSubsetError(HarnessError):
    """A YAML-subset parsing error with a source location."""


@dataclass
class ScoredCase:
    """The score text and assertion detail generated for one case/mode pair."""

    summary: str
    details: list[str]


class YamlSubsetParser:
    """Parser for the documented, intentionally limited eval YAML format."""

    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        self.lines = text.splitlines()
        self.index = 0

    def fail(self, line_number: int, message: str) -> NoReturn:
        raise YamlSubsetError(f"{self.path}:{line_number}: {message}")

    def _indent(self, line: str, line_number: int) -> int:
        prefix_length = len(line) - len(line.lstrip(" \t"))
        prefix = line[:prefix_length]
        if "\t" in prefix:
            self.fail(line_number, "tabs are not supported for indentation")
        return len(prefix)

    @staticmethod
    def _is_ignorable(line: str) -> bool:
        stripped = line.strip()
        return not stripped or stripped.startswith("#")

    def _skip_ignorable(self) -> None:
        while self.index < len(self.lines):
            line = self.lines[self.index]
            self._indent(line, self.index + 1)
            if not self._is_ignorable(line):
                return
            self.index += 1

    def parse(self) -> list[dict[str, Any]]:
        cases: list[dict[str, Any]] = []
        while True:
            self._skip_ignorable()
            if self.index >= len(self.lines):
                break
            line_number = self.index + 1
            line = self.lines[self.index]
            indent = self._indent(line, line_number)
            if indent != 0 or not line.startswith("-"):
                self.fail(line_number, "expected a top-level sequence mapping")
            cases.append(self._parse_case())
        return cases

    def _parse_case(self) -> dict[str, Any]:
        case, start_line = self._parse_mapping_sequence_item(
            item_indent=0,
            child_indent=2,
            item_label="case",
            key_error="expected a case mapping key indented by two spaces",
            stop_at_next_item=True,
            add_value=self._add_case_value,
        )
        for required_key in ("id", "prompt", "assertions"):
            if required_key not in case:
                self.fail(start_line, f"case is missing required key '{required_key}'")
        if not isinstance(case["assertions"], list):
            self.fail(start_line, "'assertions' must be a sequence")
        return case

    def _parse_mapping_sequence_item(
        self,
        item_indent: int,
        child_indent: int,
        item_label: str,
        key_error: str,
        stop_at_next_item: bool,
        add_value: Callable[[dict[str, Any], str, str, int, int], None],
    ) -> tuple[dict[str, Any], int]:
        line_number = self.index + 1
        line = self.lines[self.index]
        if not line[item_indent:].startswith("-"):
            self.fail(line_number, f"expected a {item_label} sequence item")

        item = line[item_indent + 1 :]
        if item and not item.startswith(" "):
            self.fail(line_number, f"expected '- ' before a {item_label} mapping")

        mapping: dict[str, Any] = {}
        start_line = line_number
        self.index += 1
        inline = item.strip()
        if inline:
            key, value = self._split_mapping(inline, line_number)
            add_value(mapping, key, value, item_indent, line_number)

        while True:
            self._skip_ignorable()
            if self.index >= len(self.lines):
                break

            current_number = self.index + 1
            current = self.lines[self.index]
            indent = self._indent(current, current_number)
            if indent <= item_indent:
                if stop_at_next_item and indent == item_indent and current.startswith("-"):
                    break
                if stop_at_next_item and indent == item_indent:
                    self.fail(current_number, "expected the next case sequence item")
                break
            if indent != child_indent:
                self.fail(current_number, key_error)

            key, value = self._split_mapping(current[child_indent:], current_number)
            self.index += 1
            add_value(mapping, key, value, child_indent, current_number)

        return mapping, start_line

    def _add_case_value(
        self,
        case: dict[str, Any],
        key: str,
        value: str,
        key_indent: int,
        line_number: int,
    ) -> None:
        if key not in TOP_LEVEL_KEYS:
            self.fail(line_number, f"unknown case key '{key}'")
        if key in case:
            self.fail(line_number, f"duplicate case key '{key}'")
        if key == "assertions":
            if value:
                self.fail(
                    line_number,
                    "'assertions' must be a nested sequence, not an inline value",
                )
            case[key] = self._parse_assertions(key_indent, line_number)
            return
        case[key] = self._parse_value(value, key_indent, line_number)

    def _parse_assertions(
        self, parent_indent: int, parent_line: int
    ) -> list[dict[str, Any]]:
        assertions: list[dict[str, Any]] = []
        item_indent = parent_indent + 2
        while True:
            self._skip_ignorable()
            if self.index >= len(self.lines):
                break

            line_number = self.index + 1
            line = self.lines[self.index]
            indent = self._indent(line, line_number)
            if indent <= parent_indent:
                break
            if indent != item_indent or not line[item_indent:].startswith("-"):
                self.fail(line_number, "expected an assertion sequence item")
            assertions.append(self._parse_assertion(item_indent))

        if not assertions:
            self.fail(parent_line, "'assertions' must contain at least one assertion")
        return assertions

    def _parse_assertion(self, item_indent: int) -> dict[str, Any]:
        assertion, start_line = self._parse_mapping_sequence_item(
            item_indent=item_indent,
            child_indent=item_indent + 2,
            item_label="assertion",
            key_error="expected an assertion mapping key",
            stop_at_next_item=False,
            add_value=self._add_assertion_value,
        )
        if "type" not in assertion:
            self.fail(start_line, "assertion is missing required key 'type'")
        return assertion

    def _add_assertion_value(
        self,
        assertion: dict[str, Any],
        key: str,
        value: str,
        key_indent: int,
        line_number: int,
    ) -> None:
        if key not in ASSERTION_KEYS:
            self.fail(line_number, f"unknown assertion key '{key}'")
        if key in assertion:
            self.fail(line_number, f"duplicate assertion key '{key}'")
        assertion[key] = self._parse_value(value, key_indent, line_number)

    def _split_mapping(self, text: str, line_number: int) -> tuple[str, str]:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:(.*)$", text)
        if not match:
            self.fail(line_number, "expected a mapping key followed by ':'")
        key, remainder = match.groups()
        if remainder and not remainder[0].isspace():
            self.fail(line_number, "expected whitespace after ':'")
        return key, remainder.strip()

    def _parse_value(self, value: str, key_indent: int, line_number: int) -> Any:
        if not value:
            self.fail(line_number, "expected a scalar value or block scalar")
        if value.startswith(("|", ">")):
            return self._parse_block_scalar(value, key_indent, line_number)
        return self._parse_scalar(value, line_number)

    def _parse_block_scalar(
        self, header: str, key_indent: int, line_number: int
    ) -> str:
        clean_header = self._strip_plain_comment(header).strip()
        style = clean_header[:1]
        indicators = clean_header[1:]
        if style not in "|>" or not re.fullmatch(r"[1-9+\-]{0,2}", indicators):
            self.fail(line_number, f"unsupported block scalar header '{header}'")
        chomp = ""
        indent_indicator: int | None = None
        for indicator in indicators:
            if indicator in "+-":
                if chomp:
                    self.fail(line_number, "duplicate block chomping indicator")
                chomp = indicator
            else:
                if indent_indicator is not None:
                    self.fail(line_number, "duplicate block indentation indicator")
                indent_indicator = int(indicator)

        content_start = self.index
        first_content_index: int | None = None
        while self.index < len(self.lines):
            candidate = self.lines[self.index]
            candidate_number = self.index + 1
            candidate_indent = self._indent(candidate, candidate_number)
            if not candidate.strip():
                self.index += 1
                continue
            if candidate_indent <= key_indent:
                self.index = content_start
                return ""
            first_content_index = self.index
            break

        if first_content_index is None:
            self.index = content_start
            return ""

        if indent_indicator is None:
            content_indent = self._indent(
                self.lines[first_content_index], first_content_index + 1
            )
        else:
            content_indent = key_indent + indent_indicator
            first_indent = self._indent(
                self.lines[first_content_index], first_content_index + 1
            )
            if first_indent < content_indent:
                self.fail(
                    first_content_index + 1,
                    "block content is less indented than its indicator",
                )

        self.index = content_start
        content_lines: list[str] = []
        while self.index < len(self.lines):
            candidate = self.lines[self.index]
            candidate_number = self.index + 1
            candidate_indent = self._indent(candidate, candidate_number)
            if candidate.strip() and candidate_indent <= key_indent:
                break
            if candidate.strip() and candidate_indent < content_indent:
                self.fail(
                    candidate_number, "block content has inconsistent indentation"
                )

            if candidate.strip():
                content_lines.append(candidate[content_indent:])
            else:
                content_lines.append("")
            self.index += 1

        if style == "|":
            body = "\n".join(content_lines)
        else:
            body = self._fold_block_lines(content_lines)
        raw = body + ("\n" if content_lines else "")
        if chomp == "-":
            return raw.rstrip("\n")
        if chomp == "+":
            return raw
        stripped = raw.rstrip("\n")
        return stripped + ("\n" if raw else "")

    @staticmethod
    def _fold_block_lines(lines: Sequence[str]) -> str:
        if not lines:
            return ""
        folded = lines[0]
        for index in range(1, len(lines)):
            previous = lines[index - 1]
            current = lines[index]
            if (
                not previous
                or not current
                or previous.startswith(" ")
                or current.startswith(" ")
            ):
                separator = "\n"
            else:
                separator = " "
            folded += separator + current
        return folded

    def _parse_scalar(self, value: str, line_number: int) -> Any:
        stripped = value.strip()
        if not stripped:
            self.fail(line_number, "expected a scalar value")
        if stripped.startswith('"'):
            return self._parse_double_quoted(stripped, line_number)
        if stripped.startswith("'"):
            return self._parse_single_quoted(stripped, line_number)

        plain = self._strip_plain_comment(stripped).strip()
        if not plain:
            self.fail(line_number, "expected a scalar value before the comment")
        if plain.startswith(("[", "{")):
            self.fail(line_number, "flow collections are not supported")
        if plain.startswith(("&", "*", "!")) or re.search(
            r"(?:^|\s)[&*][A-Za-z0-9_-]+", plain
        ):
            self.fail(line_number, "anchors, aliases, and tags are not supported")

        lower = plain.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"null", "~"}:
            return None
        if re.fullmatch(r"[-+]?\d+", plain):
            try:
                return int(plain)
            except ValueError:
                pass
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", plain):
            try:
                return float(plain)
            except ValueError:
                pass
        return plain

    @staticmethod
    def _strip_plain_comment(value: str) -> str:
        for index, character in enumerate(value):
            if character == "#" and (index == 0 or value[index - 1].isspace()):
                return value[:index].rstrip()
        return value

    def _parse_double_quoted(self, value: str, line_number: int) -> str:
        output: list[str] = []
        index = 1
        simple_escapes = {
            "0": "\0",
            "a": "\a",
            "b": "\b",
            "t": "\t",
            "n": "\n",
            "v": "\v",
            "f": "\f",
            "r": "\r",
            "e": "\x1b",
            " ": " ",
            '"': '"',
            "/": "/",
            "\\": "\\",
            "_": "\u00a0",
            "N": "\u0085",
            "L": "\u2028",
            "P": "\u2029",
        }
        while index < len(value):
            character = value[index]
            if character == '"':
                remainder = value[index + 1 :].strip()
                if remainder and not remainder.startswith("#"):
                    self.fail(
                        line_number, "unexpected text after a double-quoted scalar"
                    )
                return "".join(output)
            if character != "\\":
                output.append(character)
                index += 1
                continue
            if index + 1 >= len(value):
                self.fail(
                    line_number, "unfinished escape sequence in double-quoted scalar"
                )

            escape = value[index + 1]
            if escape in simple_escapes:
                output.append(simple_escapes[escape])
                index += 2
                continue
            hex_lengths = {"x": 2, "u": 4, "U": 8}
            if escape in hex_lengths:
                length = hex_lengths[escape]
                hex_start = index + 2
                hex_end = hex_start + length
                digits = value[hex_start:hex_end]
                if len(digits) != length or not re.fullmatch(r"[0-9A-Fa-f]+", digits):
                    self.fail(
                        line_number,
                        "invalid hexadecimal escape in double-quoted scalar",
                    )
                try:
                    output.append(chr(int(digits, 16)))
                except ValueError:
                    self.fail(
                        line_number, "invalid Unicode escape in double-quoted scalar"
                    )
                index = hex_end
                continue
            self.fail(
                line_number, f"unsupported escape '\\{escape}' in double-quoted scalar"
            )
        self.fail(line_number, "unterminated double-quoted scalar")
        return ""  # Unreachable; keeps static type checkers satisfied.

    def _parse_single_quoted(self, value: str, line_number: int) -> str:
        output: list[str] = []
        index = 1
        while index < len(value):
            character = value[index]
            if character == "'":
                if index + 1 < len(value) and value[index + 1] == "'":
                    output.append("'")
                    index += 2
                    continue
                remainder = value[index + 1 :].strip()
                if remainder and not remainder.startswith("#"):
                    self.fail(
                        line_number, "unexpected text after a single-quoted scalar"
                    )
                return "".join(output)
            output.append(character)
            index += 1
        self.fail(line_number, "unterminated single-quoted scalar")
        return ""  # Unreachable; keeps static type checkers satisfied.


def parse_yaml_subset(path: Path) -> list[dict[str, Any]]:
    """Load eval cases from one YAML-subset file, preserving source errors."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(
            f"unable to read test cases YAML '{path}': {error}"
        ) from error
    return YamlSubsetParser(path, text).parse()


def resolve_skill(skill_argument: str) -> Path:
    """Resolve a skill name or directory path to its skill directory."""

    requested = Path(skill_argument).expanduser()
    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend((Path.cwd() / requested, REPOSITORY_ROOT / requested))
        if len(requested.parts) == 1:
            candidates.append(REPOSITORY_ROOT / "skills" / requested)

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "eval" / "test-cases.yaml").is_file():
            return candidate.resolve()
        if candidate.name == "eval" and (candidate / "test-cases.yaml").is_file():
            return candidate.parent.resolve()

    raise HarnessError(
        f"skill '{skill_argument}' was not found or has no eval/test-cases.yaml file"
    )


def text_word_count(text: str) -> int:
    """Approximate the shell runner's wc -w behaviour for normal text."""

    return len(text.split())


def extract_json_block(response: str) -> str | None:
    """Extract the first fenced JSON object or balanced bare JSON object."""

    fenced_open = re.compile(r"(?im)^[ \t]*```json[ \t]*(?:\n|$)")
    fenced_close = re.compile(r"(?m)^[ \t]*```[ \t]*(?:\n|$)")
    opening = fenced_open.search(response)
    if opening:
        closing = fenced_close.search(response, opening.end())
        if closing:
            return response[opening.end() : closing.start()]
        return response[opening.end() :]

    for start, character in enumerate(response):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(response)):
            current = response[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    return response[start : end + 1]
    return None


def _load_json(json_block: str | None) -> tuple[Any | None, bool]:
    if not json_block:
        return None, False
    try:
        return json.loads(json_block), True
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, False


def _contains_case_insensitive(response: str, target: str) -> bool:
    return target.casefold() in response.casefold()


def _target_text(target: Any) -> str:
    if target is None:
        return ""
    if isinstance(target, bool):
        return "true" if target else "false"
    return str(target)


def _casefolded_quoted_literals(target: str) -> list[str]:
    return re.findall(r"'([^']+)'", target)


def _is_number(value: Any) -> float | None:
    if isinstance(value, list):
        return float(len(value))
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _field_value(data: Any | None, field: str) -> Any:
    if isinstance(data, dict) and field in data:
        return data[field]
    return _MISSING


def _value_length(value: Any) -> int | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, (str, list, tuple, dict)):
        return len(value)
    return len(str(value))


def _value_text(value: Any) -> str:
    if value is _MISSING:
        return ""
    if isinstance(value, list):
        return str(len(value))
    return str(value)


_FIELD_PATTERN = r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)"
_RANGE_PATTERN = re.compile(
    rf"^\s*{_FIELD_PATTERN}\s*>=\s*(?P<minimum>-?(?:\d+(?:\.\d*)?|\.\d+))\s+AND\s+"
    r"(?P=field)\s*<=\s*(?P<maximum>-?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_ABS_PATTERN = re.compile(
    rf"^\s*abs\({_FIELD_PATTERN}\)\s*<=\s*(?P<limit>-?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_MINIMUM_PATTERN = re.compile(
    rf"^\s*{_FIELD_PATTERN}\s*>=\s*(?P<minimum>-?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_LENGTH_RANGE_PATTERN = re.compile(
    rf"^\s*len\({_FIELD_PATTERN}\)\s*>=\s*(?P<minimum>\d+)\s+AND\s+len\((?P=field)\)\s*<=\s*"
    r"(?P<maximum>\d+)\s*$"
)
_LENGTH_MAXIMUM_PATTERN = re.compile(
    rf"^\s*len\({_FIELD_PATTERN}\)\s*<=\s*(?P<maximum>\d+)\s*$"
)
_LENGTH_MINIMUM_PATTERN = re.compile(
    rf"^\s*len\({_FIELD_PATTERN}\)\s*>=\s*(?P<minimum>\d+)\s*$"
)
_WORD_COUNT_PATTERN = re.compile(
    rf"^\s*word_count\({_FIELD_PATTERN}\)\s*>=\s*(?P<minimum>\d+)\s+AND\s+"
    r"word_count\((?P=field)\)\s*<=\s*(?P<maximum>\d+)\s*$"
)


def _evaluate_range_check(target: str, data: Any | None, has_json: bool) -> str:
    if not has_json:
        return "FAIL"

    match = _RANGE_PATTERN.fullmatch(target)
    if match:
        value = _field_value(data, match.group("field"))
        numeric = _is_number(value)
        if numeric is None:
            return "SKIP" if value is _MISSING or value == "" else "FAIL"
        return (
            "PASS"
            if float(match.group("minimum")) <= numeric <= float(match.group("maximum"))
            else "FAIL"
        )

    match = _ABS_PATTERN.fullmatch(target)
    if match:
        value = _field_value(data, match.group("field"))
        numeric = _is_number(value)
        if numeric is None:
            return "SKIP" if value is _MISSING or value == "" else "FAIL"
        return "PASS" if abs(numeric) <= float(match.group("limit")) else "FAIL"

    match = _MINIMUM_PATTERN.fullmatch(target)
    if match:
        value = _field_value(data, match.group("field"))
        numeric = _is_number(value)
        if numeric is None:
            return "SKIP" if value is _MISSING or value == "" else "FAIL"
        return "PASS" if numeric >= float(match.group("minimum")) else "FAIL"

    match = _LENGTH_RANGE_PATTERN.fullmatch(target)
    if match:
        length = _value_length(_field_value(data, match.group("field")))
        if length is None:
            return "FAIL"
        return (
            "PASS"
            if int(match.group("minimum")) <= length <= int(match.group("maximum"))
            else "FAIL"
        )

    match = _LENGTH_MAXIMUM_PATTERN.fullmatch(target)
    if match:
        length = _value_length(_field_value(data, match.group("field")))
        if length is None:
            return "FAIL"
        return "PASS" if length <= int(match.group("maximum")) else "FAIL"

    match = _LENGTH_MINIMUM_PATTERN.fullmatch(target)
    if match:
        length = _value_length(_field_value(data, match.group("field")))
        if length is None:
            return "FAIL"
        return "PASS" if length >= int(match.group("minimum")) else "FAIL"

    match = _WORD_COUNT_PATTERN.fullmatch(target)
    if match:
        words = text_word_count(_value_text(_field_value(data, match.group("field"))))
        return (
            "PASS"
            if int(match.group("minimum")) <= words <= int(match.group("maximum"))
            else "FAIL"
        )

    return "SKIP"


def evaluate_assertion(
    response: str,
    assertion_type: Any,
    target: Any = "",
    json_block: Any = _UNSET,
) -> str:
    """Return PASS, FAIL, SOFT, or SKIP for one eval assertion."""

    assertion_name = _target_text(assertion_type)
    target_text = _target_text(target)
    if json_block is _UNSET:
        json_block = extract_json_block(response)
    data, has_json = _load_json(json_block)

    if assertion_name == "contains":
        return "PASS" if _contains_case_insensitive(response, target_text) else "FAIL"
    if assertion_name == "not_contains":
        quoted_literals = _casefolded_quoted_literals(target_text)
        needles = quoted_literals or [target_text]
        return (
            "FAIL"
            if any(_contains_case_insensitive(response, needle) for needle in needles)
            else "PASS"
        )
    if assertion_name == "regex":
        try:
            matched = re.search(
                target_text, response, flags=re.IGNORECASE | re.MULTILINE
            )
        except re.error:
            return "FAIL"
        return "PASS" if matched else "FAIL"
    if assertion_name == "question_before_code":
        question_line = next(
            (
                index
                for index, line in enumerate(response.splitlines(), start=1)
                if "?" in line
            ),
            None,
        )
        code_line = next(
            (
                index
                for index, line in enumerate(response.splitlines(), start=1)
                if "```" in line
            ),
            None,
        )
        if code_line is None or (
            question_line is not None and question_line < code_line
        ):
            return "PASS"
        return "FAIL"
    if assertion_name == "json_valid":
        return "PASS" if has_json else "FAIL"
    if assertion_name in {"json_fields", "json_schema"}:
        if not has_json or not isinstance(data, dict):
            return "FAIL"
        required_fields = [
            field.strip() for field in target_text.split(",") if field.strip()
        ]
        return "PASS" if all(field in data for field in required_fields) else "FAIL"
    if assertion_name == "token_limit":
        try:
            limit = int(target_text or "99999")
        except ValueError:
            return "FAIL"
        estimated_tokens = text_word_count(response) * 13 // 10
        return "PASS" if estimated_tokens <= limit else "FAIL"
    if assertion_name == "range_check":
        return _evaluate_range_check(target_text, data, has_json)
    if assertion_name in {"structure_check", "sequence_check"}:
        return "SOFT"
    return "SKIP"


def _is_critical(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def score_case_response(
    response: str, assertions: Iterable[dict[str, Any]]
) -> ScoredCase:
    """Score an output response and retain the shell scorecard's detail format."""

    if response.startswith("EVAL_ERROR"):
        return ScoredCase("ERROR", [])

    json_block = extract_json_block(response)
    passed = 0
    failed = 0
    soft = 0
    skipped = 0
    total = 0
    critical_failed = False
    details: list[str] = []

    for assertion in assertions:
        total += 1
        assertion_type = _target_text(assertion.get("type", ""))
        target = _target_text(assertion.get("target", ""))
        result = evaluate_assertion(response, assertion_type, target, json_block)
        if result == "PASS":
            passed += 1
        elif result == "FAIL":
            failed += 1
            if _is_critical(assertion.get("critical", False)):
                critical_failed = True
        elif result == "SOFT":
            soft += 1
        else:
            skipped += 1

        label = _target_text(assertion.get("description", ""))
        if not label:
            label = f"{assertion_type}({target})"
        marker = " [CRITICAL]" if _is_critical(assertion.get("critical", False)) else ""
        details.append(f"  {result}{marker} — {label}")

    anchored = passed + failed
    summary = f"{passed}/{total} ({anchored} anchored, {soft} soft, {skipped} skipped)"
    if critical_failed:
        summary += " [CRITICAL FAIL]"
    return ScoredCase(summary, details)


def split_agent_command(command: str, platform: str | None = None) -> list[str]:
    """Split a configured command without applying POSIX escaping on Windows."""

    if not command or not command.strip():
        raise HarnessError(
            "agent command is missing; set EVAL_AGENT_CMD or pass --agent-cmd"
        )
    is_windows = (platform == "nt") if platform is not None else os.name == "nt"
    try:
        arguments = shlex.split(command, posix=not is_windows)
    except ValueError as error:
        raise HarnessError(f"invalid agent command: {error}") from error
    if is_windows:
        # posix=False preserves Windows backslashes. shlex keeps outer double
        # quotes, so remove only the grouping quotes that Windows would consume.
        normalized: list[str] = []
        for argument in arguments:
            if (
                len(argument) >= 2
                and argument.startswith('"')
                and argument.endswith('"')
            ):
                normalized.append(argument[1:-1])
            else:
                normalized.append(argument)
        arguments = normalized
    if not arguments:
        raise HarnessError(
            "agent command is missing; set EVAL_AGENT_CMD or pass --agent-cmd"
        )
    return arguments


def build_agent_argv(
    command: str,
    prompt: str,
    prompt_file: Path,
    platform: str | None = None,
) -> list[str]:
    """Build one subprocess argv, expanding prompt placeholders safely."""

    arguments = split_agent_command(command, platform=platform)
    has_placeholder = any(
        "{prompt}" in argument or "{prompt_file}" in argument for argument in arguments
    )
    expanded = [
        argument.replace("{prompt_file}", str(prompt_file)).replace("{prompt}", prompt)
        for argument in arguments
    ]
    if not has_placeholder:
        expanded.append(prompt)
    return expanded


def validate_agent_command(argv: Sequence[str]) -> None:
    """Fail before evaluation if the configured executable cannot be launched."""

    if not argv:
        raise HarnessError(
            "agent command is missing; set EVAL_AGENT_CMD or pass --agent-cmd"
        )
    executable = argv[0]
    has_path_separator = os.path.sep in executable or (
        os.path.altsep and os.path.altsep in executable
    )
    if has_path_separator:
        candidate = Path(executable).expanduser()
        if not candidate.is_file() or (
            os.name != "nt" and not os.access(str(candidate), os.X_OK)
        ):
            raise HarnessError(f"agent command is not executable: {executable}")
        return
    if shutil.which(executable) is None:
        raise HarnessError(f"agent command is not executable: {executable}")


def _prompt_for_mode(skill_name: str, prompt: str, mode: str) -> str:
    if mode == "with-skill":
        return f"Use the {skill_name} skill to answer this: {prompt}"
    return prompt


def _write_text(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"unable to write '{path}': {error}") from error


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise HarnessError(f"unable to read '{path}': {error}") from error


def _safe_case_id(case: dict[str, Any]) -> str:
    case_id = _target_text(case.get("id", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id):
        raise HarnessError(f"invalid case id for a result filename: {case_id!r}")
    return case_id


def _create_results_layout(results_dir: Path) -> None:
    try:
        (results_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (results_dir / "with-skill").mkdir(parents=True, exist_ok=True)
        (results_dir / "baseline").mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise HarnessError(
            f"unable to create results directory '{results_dir}': {error}"
        ) from error


def _latest_results_directory(results_root: Path) -> Path:
    if not results_root.is_dir():
        raise HarnessError(
            "no existing results found; pass --results-dir to --score-only or "
            "run an evaluation first"
        )
    try:
        candidates = sorted(
            candidate
            for candidate in results_root.iterdir()
            if candidate.is_dir()
            and (
                (candidate / "with-skill").is_dir() or (candidate / "baseline").is_dir()
            )
        )
    except OSError as error:
        raise HarnessError(
            f"unable to inspect results directory '{results_root}': {error}"
        ) from error
    if not candidates:
        raise HarnessError(
            "no existing results found; pass --results-dir to --score-only or run an evaluation first"
        )
    return candidates[-1]


def _run_agent(command: str, prompt: str, prompt_file: Path, failure_note: str) -> str:
    argv = build_agent_argv(command, prompt, prompt_file)
    try:
        completed = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        raise HarnessError(f"agent command could not be executed: {error}") from error
    if completed.returncode != 0:
        return f"EVAL_ERROR: agent command failed for {failure_note}"
    return completed.stdout


def _run_case(
    skill_name: str,
    case: dict[str, Any],
    mode: str,
    command: str,
    results_dir: Path,
) -> None:
    case_id = _safe_case_id(case)
    raw_prompt = _target_text(case.get("prompt", ""))
    prompt = _prompt_for_mode(skill_name, raw_prompt, mode)
    prompt_file = results_dir / "prompts" / (case_id + ".txt")
    output_file = results_dir / mode / (case_id + ".md")
    _write_text(prompt_file, prompt)
    print(f"  [{mode}] Running {case_id}...")
    response = _run_agent(command, prompt, prompt_file, f"{case_id} ({mode})")
    _write_text(output_file, response)
    print(f"  [{mode}] {case_id} complete ({text_word_count(response)} words)")


def _score_text(scored: ScoredCase) -> str:
    """Reproduce the shell runner's score file, including its final blank line."""

    if not scored.details:
        return scored.summary + "\n"
    return scored.summary + "\n" + "\n".join(scored.details) + "\n\n"


def _score_mode(
    case: dict[str, Any], mode: str, results_dir: Path
) -> ScoredCase | None:
    case_id = _safe_case_id(case)
    output_file = results_dir / mode / (case_id + ".md")
    if not output_file.is_file():
        return None
    scored = score_case_response(_read_text(output_file), case["assertions"])
    score_file = results_dir / mode / (case_id + ".score.txt")
    _write_text(score_file, _score_text(scored))
    return scored


def generate_scorecard(
    skill_name: str,
    cases: Sequence[dict[str, Any]],
    run_skill: bool,
    run_baseline: bool,
    results_dir: Path,
    timestamp: str,
) -> Path:
    """Generate the shell-compatible scorecard table and per-assertion section."""

    scorecard = results_dir / "scorecard.md"
    lines = [
        f"# Eval Scorecard — {skill_name}",
        f"**Timestamp:** {timestamp}",
        "",
        "| Test Case | With Skill | Baseline |",
        "|-----------|-----------|----------|",
    ]
    scored_by_case: dict[str, dict[str, ScoredCase]] = {}

    for case in cases:
        case_id = _safe_case_id(case)
        by_mode: dict[str, ScoredCase] = {}
        skill_score = "—"
        baseline_score = "—"
        if run_skill:
            scored = _score_mode(case, "with-skill", results_dir)
            if scored is not None:
                by_mode["with-skill"] = scored
                skill_score = scored.summary
        if run_baseline:
            scored = _score_mode(case, "baseline", results_dir)
            if scored is not None:
                by_mode["baseline"] = scored
                baseline_score = scored.summary
        scored_by_case[case_id] = by_mode
        lines.append(f"| {case_id} | {skill_score} | {baseline_score} |")

    lines.extend(("", "## Assertion Details"))
    for case in cases:
        case_id = _safe_case_id(case)
        lines.extend(("", f"### {case_id}"))
        for mode in ("with-skill", "baseline"):
            scored = scored_by_case[case_id].get(mode)
            if scored is None:
                continue
            lines.append(f"**{mode}:**")
            lines.append("```")
            lines.extend(_score_text(scored).splitlines())
            lines.append("```")

    _write_text(scorecard, "\n".join(lines) + "\n")
    return scorecard


def _selected_cases(
    cases: Sequence[dict[str, Any]], case_id: str | None
) -> list[dict[str, Any]]:
    if case_id is None:
        return list(cases)
    selected = [case for case in cases if _target_text(case.get("id", "")) == case_id]
    if not selected:
        raise HarnessError(f"no test case found with id '{case_id}'")
    return selected


def _print_dry_run(skill_name: str, cases: Sequence[dict[str, Any]]) -> None:
    assertion_count = sum(len(case["assertions"]) for case in cases)
    print("=== Skill Eval Runner ===")
    print(f"Skill: {skill_name}")
    print(f"Found {len(cases)} test cases and {assertion_count} assertions.")
    for case in cases:
        print(f"  {_safe_case_id(case)}: {len(case['assertions'])} assertions")
    print("Dry run: no agent invoked and no result files written.")


def run_evaluation(arguments: argparse.Namespace) -> int:
    skill_dir = resolve_skill(arguments.skill)
    yaml_path = skill_dir / "eval" / "test-cases.yaml"
    cases = parse_yaml_subset(yaml_path)
    if not cases:
        raise HarnessError(f"no test cases found in '{yaml_path}'")
    cases = _selected_cases(cases, arguments.case)

    skill_name = skill_dir.name
    if arguments.dry_run:
        _print_dry_run(skill_name, cases)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_results_root = skill_dir / "eval" / "results"
    if arguments.score_only:
        if arguments.results_dir:
            results_dir = Path(arguments.results_dir).expanduser().resolve()
            if not results_dir.is_dir():
                raise HarnessError(f"results directory does not exist: {results_dir}")
        else:
            results_dir = _latest_results_directory(default_results_root)
    else:
        results_dir = (
            Path(arguments.results_dir).expanduser().resolve()
            if arguments.results_dir
            else default_results_root / timestamp
        )
        _create_results_layout(results_dir)

    run_skill = not arguments.baseline_only
    run_baseline = not arguments.skill_only

    print("=== Skill Eval Runner ===")
    print(f"Skill:     {skill_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Results:   {results_dir}")
    print(
        "Found {} test cases: {}".format(
            len(cases), " ".join(_safe_case_id(case) for case in cases)
        )
    )
    print("")

    if not arguments.score_only:
        command = (
            arguments.agent_cmd
            or os.environ.get("EVAL_AGENT_CMD")
            or DEFAULT_AGENT_COMMAND
        )
        probe_prompt_file = results_dir / "prompts" / (_safe_case_id(cases[0]) + ".txt")
        validate_agent_command(build_agent_argv(command, "", probe_prompt_file))

        for mode, enabled, heading in (
            ("with-skill", run_skill, "── With Skill ──"),
            ("baseline", run_baseline, "── Baseline (no skill) ──"),
        ):
            if not enabled:
                continue
            print(heading)
            for case in cases:
                _run_case(skill_name, case, mode, command, results_dir)
            print("")

    scorecard = generate_scorecard(
        skill_name, cases, run_skill, run_baseline, results_dir, timestamp
    )
    print("=== Scorecard ===")
    print(_read_text(scorecard), end="")
    print(f"=== Eval complete — {results_dir}/ ===")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run agent-agnostic skill evaluations."
    )
    parser.add_argument(
        "--skill",
        required=True,
        help="Skill name or path containing eval/test-cases.yaml",
    )
    parser.add_argument("--case", help="Run or score one case ID")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--skill-only", action="store_true", help="Skip baseline evaluation"
    )
    modes.add_argument(
        "--baseline-only", action="store_true", help="Skip with-skill evaluation"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score existing result files without invoking an agent",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List cases and assertions without invoking an agent",
    )
    parser.add_argument(
        "--results-dir",
        help="Output directory, or existing input directory for --score-only",
    )
    parser.add_argument(
        "--agent-cmd",
        help="Agent command; takes precedence over EVAL_AGENT_CMD and supports {prompt}/{prompt_file}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        return run_evaluation(arguments)
    except HarnessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
