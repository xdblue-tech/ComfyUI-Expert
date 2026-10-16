---
name: runpod-oneshot
description: Create and run a customized oneshot RunPod workflow from a shipped template — pick the template from manifest.json, craft a model-appropriate prompt (SDXL quality tags, FLUX natural language, Wan video, Krea2), set seed (random or fixed) and image size via rp_oneshot.py flags, submit to the RunPod serverless endpoint, and receive the generated image or video.
---

# RunPod Oneshot Workflows

Produce one image or video from a shipped template on a hosted RunPod endpoint. You
customize with command flags only — NEVER hand-edit the workflow JSON.

## The 3-step loop

1. **Pick a template.** Read `manifest.json` at the package root. Each entry lists the
   model family, the baked checkpoint/LoRA, and which nodes the flags target. Use
   `--dry-run` to preview what will be sent.
2. **Customize.** Pass prompt/seed/size as flags to `bin/rp_oneshot.py` (all optional —
   the template defaults are complete):
   - `--output DIR` — **REQUIRED on every run**: the directory this job's outputs go to
   - `--prompt "..."` — replaces the positive prompt
   - `--negative "..."` — replaces the negative prompt
   - `--seed 12345` or `--seed random` (default: random)
   - `--width N --height N` — output resolution
   - `--image ref.png` — input image for img2img templates (repeatable)
   - `--set NODE.FIELD=VALUE` — raw escape hatch for anything else (e.g. `--set 5.cfg=8`)
3. **Run.** Add `--wait` and read the printed absolute output directory.

## Command examples

Preview without spending anything (no API key needed):
```bash
python3 bin/rp_oneshot.py --template unholy-sdxl \
  --prompt "masterpiece, best quality, a red apple on a wooden table, photorealistic" \
  --width 1024 --height 1024 --output ./output --dry-run
```

Reproducible run (same seed = same image):
```bash
RUNPOD_API_KEY=... python3 bin/rp_oneshot.py --template unholy-sdxl \
  --prompt "masterpiece, best quality, a red apple on a wooden table, photorealistic" \
  --seed 42 --width 1024 --height 1024 --output ./output --wait
```

img2img with an input image (only when the manifest entry says the template takes
an image input — unholy-sdxl currently does not):
```bash
RUNPOD_API_KEY=... python3 bin/rp_oneshot.py --template <img2img-template> \
  --prompt "..." --image input.png --seed random --output ./output --wait
```

Outputs land in `<--output DIR>/<job-id>/` (`--project NAME` nests them under
`<--output>/<project>/<job-id>/`), each job dir also holds `response.json`. The tool
prints the absolute output directory as its last line.

## Writing the prompt — match the model family

Read `model_family` from the template's manifest entry:

| Family | Style | CFG | Length | Quality tags |
|---|---|---|---|---|
| SDXL | tag phrases, weighted syntax `(term:1.3)` | 7–9 | 50–150 words | FIRST: `masterpiece, best quality,` |
| FLUX | natural language sentences | 3.5–4 | 50–100 words | none — they hurt |
| Wan (video) | natural language, describe motion + camera | per template | 50–100 words | none |
| Krea2 | natural language, descriptive | per template | 50–100 words | minimal |

SDXL example (quality tags front, then trigger/subject, details, setting, style):
```
masterpiece, best quality, sage_character, photorealistic portrait of a woman,
detailed skin texture with freckles, emerald green eyes, auburn copper hair,
natural lighting from window, indoor setting, shallow depth of field,
RAW photo quality, 8k uhd, film grain
```

SDXL anti-pattern (do not send quality-tag spam to FLUX/Wan, and do not send prose-only
prompts to SDXL): check the family before writing.

If the manifest entry lists a `trigger_word`, include it in every prompt — it summons
the baked LoRA's style. The baked negative prompt is already tuned; only pass
`--negative` when the user asks for a change.

## Seed and size semantics

- `--seed random` (default): a fresh seed every run — different image each time.
- `--seed N`: reproducible — the same seed, prompt, and size produce the same output.
- Size: SDXL is native around 1024×1024; keep total pixels near one million
  (e.g. 1024×1024, 832×1216, 1216×832). Other families: follow the template's
  `size.guidance` in the manifest.

## Operational facts

- **Cold start:** the first job after an idle period takes 2–5 minutes (the endpoint
  pulls a ~22 GB image before sampling). Subsequent jobs take seconds.
- **Stuck in IN_QUEUE > 10 minutes:** the endpoint has likely idle-dropped its workers.
  Tell the user to contact the package maintainer — retrying will not help.
- **Request caps:** the whole request body is capped (10 MB `/run`, 20 MB `/runsync`),
  workflow + base64 input images share it. Keep input images under ~5 MB, or host them
  at a URL and reference that instead.
- **API key:** `RUNPOD_API_KEY` must be exported. The tool refuses to submit without it
  (exit 2). `--dry-run` works without a key.
- Exit codes: 0 = success, 1 = job failed or bad override (the job's error is printed
  verbatim), 2 = missing API key.