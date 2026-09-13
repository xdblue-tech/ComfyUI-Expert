# comfyui-video-production — Eval Configuration

## Classification

- **Type**: Capability Uplift
- **Category**: End-to-end video pipeline orchestration with validation gates and error recovery

## What "Good" Looks Like

1. Correct pipeline selection for the task (img2vid, txt2vid, vid2vid, multi-shot) with appropriate model choices
2. Step ordering follows the validation-gated pattern: generate → validate → animate → validate → concat
3. Error recovery paths are defined: seed randomization → parameter adjustment → model fallback
4. Model fallback chains are correct (e.g., AnimateDiff → SVD → Wan if primary fails)
5. Resource management accounts for VRAM constraints across pipeline stages (unload between heavy steps)

## Known Limitations

- Cannot predict actual generation quality or detect visual artifacts programmatically
- Video model landscape changes rapidly; fallback chains may need updating
- Concatenation quality depends on consistency between shots, which is hard to guarantee

## Benchmark Strategy

- **Without skill**: A general-purpose agent with no skill loaded suggests a linear pipeline without validation gates, no error recovery, and ignores VRAM management between stages
- **With skill**: Produces validation-gated pipelines with retry strategies, model fallbacks, and VRAM-aware step ordering
- **Key differentiator**: Validation gates between pipeline stages and structured error recovery — the difference between a pipeline that fails silently and one that catches and recovers from errors

## Security — Eval Sandboxing

The runner only reads `eval/test-cases.yaml` and writes under `eval/results/<timestamp>/`, which is
gitignored because eval output can expose secrets from real configuration files. Tool access belongs
to the agent command: pass the read-only restriction flags your agent supports.

## Running Evals

The harness is agent-agnostic. This skill's `eval/run-eval.sh` is a four-line shim that execs the
shared runner, `scripts/skill_eval.py` (Python 3.9+ standard library only), with `--skill comfyui-video-production`.

```bash
bash eval/run-eval.sh --dry-run            # list cases and assertions; invokes no agent
bash eval/run-eval.sh                      # full run: with-skill and baseline
bash eval/run-eval.sh --skill-only         # with-skill only
bash eval/run-eval.sh --baseline-only      # baseline only
bash eval/run-eval.sh --case TC-001        # single case
bash eval/run-eval.sh --score-only         # re-score the newest results/ directory

python3 scripts/skill_eval.py --skill comfyui-video-production --dry-run   # same runner, from the repo root
```

Select the agent with `EVAL_AGENT_CMD`, or with `--agent-cmd` when the flag should win. The default
is `claude -p`; any command that accepts a prompt works:

```bash
EVAL_AGENT_CMD="codex exec" bash eval/run-eval.sh --skill-only
EVAL_AGENT_CMD="pi -p" bash eval/run-eval.sh --case TC-001
bash eval/run-eval.sh --agent-cmd "my-agent --prompt {prompt}"
```

`{prompt}` and `{prompt_file}` are replaced inside individual arguments; when neither placeholder
appears, the prompt is appended as the final argument. Results land in
`eval/results/<YYYYmmdd-HHMMSS>/` as `prompts/<case>.txt`, `with-skill/<case>.md`,
`baseline/<case>.md`, `<case>.score.txt`, and `scorecard.md`.

Exit codes: `0` once a run completes, even when assertions fail — failures are reported in
`scorecard.md`. `2` means a harness error: unreadable or invalid YAML, no test cases found, or an
agent command that is missing or not executable.

`structure_check` and `sequence_check` assertions are recorded as `SOFT` in the scorecard but are not
scored; they need a human or model judge.

## Retirement Signal

When a general-purpose agent with no skill loaded consistently produces video pipelines with inter-stage validation, structured retry strategies (seed → params → model fallback), and VRAM-aware step ordering without needing the skill's pipeline templates.
