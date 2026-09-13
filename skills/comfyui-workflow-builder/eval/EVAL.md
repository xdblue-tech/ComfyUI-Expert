# comfyui-workflow-builder — Eval Configuration

## Classification

- **Type**: Capability Uplift
- **Category**: Structured workflow generation from natural language with hardware-aware validation

## What "Good" Looks Like

1. Output is valid ComfyUI workflow JSON with correct node IDs, class_types, and connections
2. All class_types reference nodes that actually exist in ComfyUI (no hallucinated node names)
3. Model filenames match real checkpoint/LoRA files (e.g., `juggernautXL_v9.safetensors`, not invented names)
4. Connections are correct — no dangling inputs, output indices match node output slots (e.g., CheckpointLoader outputs [MODEL, CLIP, VAE] at indices [0, 1, 2])
5. Resolution matches the selected model's training resolution (1024x1024 for SDXL, 512x512 for SD1.5, etc.) and VRAM estimate is reasonable for the hardware

## Known Limitations

- Cannot verify at generation time whether the user actually has a specific model installed
- Custom node availability varies per installation — skill uses common nodes but can't guarantee all are present
- VRAM estimates are approximations based on typical configurations

## Benchmark Strategy

- **Without skill**: A general-purpose agent with no skill loaded produces plausible-looking JSON but frequently hallucinates node class_types, uses wrong output indices, and ignores resolution/model compatibility
- **With skill**: Generates validated workflows with correct node names, proper output slot indices, model-appropriate resolutions, and VRAM estimates
- **Key differentiator**: Output index correctness and node class_type accuracy — the difference between a workflow that loads vs. one that errors immediately

## Security — Eval Sandboxing

The runner only reads `eval/test-cases.yaml` and writes under `eval/results/<timestamp>/`, which is
gitignored because eval output can expose secrets from real configuration files. Tool access belongs
to the agent command: pass the read-only restriction flags your agent supports.

## Running Evals

The harness is agent-agnostic. This skill's `eval/run-eval.sh` is a four-line shim that execs the
shared runner, `scripts/skill_eval.py` (Python 3.9+ standard library only), with `--skill comfyui-workflow-builder`.

```bash
bash eval/run-eval.sh --dry-run            # list cases and assertions; invokes no agent
bash eval/run-eval.sh                      # full run: with-skill and baseline
bash eval/run-eval.sh --skill-only         # with-skill only
bash eval/run-eval.sh --baseline-only      # baseline only
bash eval/run-eval.sh --case TC-001        # single case
bash eval/run-eval.sh --score-only         # re-score the newest results/ directory

python3 scripts/skill_eval.py --skill comfyui-workflow-builder --dry-run   # same runner, from the repo root
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

When a general-purpose agent with no skill loaded consistently produces ComfyUI workflows with correct output indices, valid class_types from the actual node registry, and model-appropriate resolutions without needing the skill's node/model reference data.
