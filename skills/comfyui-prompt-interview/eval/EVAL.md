# comfyui-prompt-interview — Eval Configuration

## Classification

- **Type**: Encoded Preference
- **Category**: Guided conversational interview for creative vision extraction before prompt synthesis

## What "Good" Looks Like

1. ASKS QUESTIONS FIRST — does not generate prompts until the user's vision is understood (sequence is interview THEN generate)
2. Interview length is appropriate: 4-7 exchanges max, not dragging on or cutting short
3. Prompts are formatted for the target model (SDXL tag-style vs. FLUX natural language vs. SD1.5 weighted tokens)
4. Negative prompt uses the correct template for the model (SDXL negatives differ from SD1.5)
5. Output includes recommended settings table (sampler, steps, CFG, resolution) and pipeline recommendation

## Known Limitations

- Cannot assess subjective prompt quality — only structural correctness and interview flow
- User engagement quality varies; skill can't force good answers from terse users
- Model-specific prompt optimization evolves as models update

## Benchmark Strategy

- **Without skill**: A general-purpose agent with no skill loaded dumps a prompt immediately when asked to "help me create an image," skipping discovery of the user's actual vision
- **With skill**: Conducts a structured interview to understand subject, mood, style, technical preferences BEFORE generating model-appropriate prompts
- **Key differentiator**: The interview-first sequence — encoded preference that creative vision must be understood before prompt generation begins

## Security — Eval Sandboxing

The runner only reads `eval/test-cases.yaml` and writes under `eval/results/<timestamp>/`, which is
gitignored because eval output can expose secrets from real configuration files. Tool access belongs
to the agent command: pass the read-only restriction flags your agent supports.

## Running Evals

The harness is agent-agnostic. This skill's `eval/run-eval.sh` is a four-line shim that execs the
shared runner, `scripts/skill_eval.py` (Python 3.9+ standard library only), with `--skill comfyui-prompt-interview`.

```bash
bash eval/run-eval.sh --dry-run            # list cases and assertions; invokes no agent
bash eval/run-eval.sh                      # full run: with-skill and baseline
bash eval/run-eval.sh --skill-only         # with-skill only
bash eval/run-eval.sh --baseline-only      # baseline only
bash eval/run-eval.sh --case TC-001        # single case
bash eval/run-eval.sh --score-only         # re-score the newest results/ directory

python3 scripts/skill_eval.py --skill comfyui-prompt-interview --dry-run   # same runner, from the repo root
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

N/A — This is an encoded preference for conversational workflow. Even if a general-purpose agent with no skill loaded improves at prompt writing, the interview-first behavior is a deliberate UX choice that won't be absorbed into default model behavior.
