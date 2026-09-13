# comfyui-character-gen — Eval Configuration

## Classification

- **Type**: Capability Uplift
- **Category**: Identity-preserving character generation pipeline design with method selection

## What "Good" Looks Like

1. Correct identity preservation method selected for the use case (InfiniteYou for 3D-to-photo, FLUX Kontext for iterative editing, PuLID for style transfer, InstantID for direct likeness)
2. CFG scale appropriate for the selected method (InstantID needs 4-5, not the typical 7-8)
3. Resolution matches the base model requirements (FLUX at 1024x1024, SDXL at 1024x1024, SD1.5 at 512x512)
4. Pipeline ordering is correct — identity injection at the right stage, not conflicting with other conditioning
5. Multi-reference handling follows correct strategy when multiple identity images are provided

## Known Limitations

- Cannot evaluate actual likeness preservation quality — only structural/parameter correctness
- New identity methods emerge frequently; skill may not cover the latest releases
- Quality of input reference images significantly affects results but is outside skill scope

## Benchmark Strategy

- **Without skill**: A general-purpose agent with no skill loaded reaches for IP-Adapter or generic img2img for every case, keeps default CFG values, and misses method-specific parameter requirements
- **With skill**: Selects the optimal identity method per use case, sets method-specific parameters correctly, orders pipeline stages properly
- **Key differentiator**: Identity method selection logic — knowing when InfiniteYou beats InstantID, and why CFG must be lowered for certain methods

## Security — Eval Sandboxing

The runner only reads `eval/test-cases.yaml` and writes under `eval/results/<timestamp>/`, which is
gitignored because eval output can expose secrets from real configuration files. Tool access belongs
to the agent command: pass the read-only restriction flags your agent supports.

## Running Evals

The harness is agent-agnostic. This skill's `eval/run-eval.sh` is a four-line shim that execs the
shared runner, `scripts/skill_eval.py` (Python 3.9+ standard library only), with `--skill comfyui-character-gen`.

```bash
bash eval/run-eval.sh --dry-run            # list cases and assertions; invokes no agent
bash eval/run-eval.sh                      # full run: with-skill and baseline
bash eval/run-eval.sh --skill-only         # with-skill only
bash eval/run-eval.sh --baseline-only      # baseline only
bash eval/run-eval.sh --case TC-001        # single case
bash eval/run-eval.sh --score-only         # re-score the newest results/ directory

python3 scripts/skill_eval.py --skill comfyui-character-gen --dry-run   # same runner, from the repo root
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

When a general-purpose agent with no skill loaded can reliably distinguish between identity preservation methods (InfiniteYou, FLUX Kontext, PuLID, InstantID) and correctly recommends method-specific parameters without the skill's decision tree.
