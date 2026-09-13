#!/usr/bin/env sh
# Shim: delegates to the shared runner so the eval harness stays agent-agnostic.
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$here/../../../scripts/skill_eval.py" --skill comfyui-prompt-interview "$@"
