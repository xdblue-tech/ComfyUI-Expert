#!/usr/bin/env python3
"""Stub agent used by tests/test_skill_eval.py.

It never contacts a model or the network. The prompt arrives either through
``--prompt-file PATH`` or as a positional argument, exactly like a real agent
command, and the fixed response is printed on stdout so scored-run tests get a
deterministic output to grade.
"""

import argparse
import sys
from pathlib import Path

FIXED_RESPONSE = """Which model are you targeting?

Here is your prompt and settings:

```json
{"method": "InstantID", "cfg": 4, "notes": "one two three"}
```
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stub eval agent")
    parser.add_argument("prompt", nargs="?", help="Prompt text appended by the runner")
    parser.add_argument(
        "--prompt", dest="inline_prompt", help="Prompt text passed as a named argument"
    )
    parser.add_argument(
        "--prompt-file", type=Path, help="File containing the prompt text"
    )
    arguments = parser.parse_args(argv)

    if arguments.prompt_file is not None:
        prompt = arguments.prompt_file.read_text(encoding="utf-8")
    elif arguments.inline_prompt is not None:
        prompt = arguments.inline_prompt
    elif arguments.prompt is not None:
        prompt = arguments.prompt
    else:
        print("fake_agent: no prompt received", file=sys.stderr)
        return 2

    print(f"PROMPT-RECEIVED: {prompt}")
    print(FIXED_RESPONSE, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
