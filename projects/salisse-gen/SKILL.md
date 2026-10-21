---
name: salisse-gen
description: Generate one image on the unholydesiremix RunPod endpoint from the user's description — compile the prompt (positive and negative) with the bundled nsfw-prompt compiler, always prefixed with the fixed quality header, then submit via the skill's scripts/rp_oneshot.py using its shipped unholy-sdxl workflow template with prompt/negative/seed/size flags. Use when the user says salisse_gen, or asks to generate an image through the unholydesiremix / unholy-sdxl endpoint.
---

# salisse-gen — unholydesiremix one-shot generation

Two jobs, in order: (1) turn the user's description into a prompt written for the
unholydesiremix checkpoint, (2) submit it with `rp_oneshot.py` against the shipped
`unholy-sdxl` template. Standalone package — ship this directory as a unit; it carries its own copies of
the script, workflow, manifest, and endpoint mapping. It uses ONLY those
skill-local artifacts — never hand-edit the workflow JSON; never use another
template, script, or local ComfyUI.

All paths below are relative to the skill root (this SKILL.md's directory),
wherever the skill is installed:

- script: `scripts/rp_oneshot.py`
- workflow template: `workflows/unholy-sdxl.json`
- template metadata: `manifest.json` (`templates["unholy-sdxl"]`)
- endpoint id: `endpoints.json`
- prompt compiler: `references/nsfw-prompt/` — bundled copy of the
  nsfw-prompt skill (compiler instructions in its `SKILL.md`, deep material in
  its `references/`). Always used for prompt compilation; standalone, ships
  with this skill.

Run commands from the skill root. The script resolves its package root as its
own parent's parent — the skill root — so it reads the skill-local manifest,
endpoints, and workflows wherever the directory is placed.

## Step 1 — Compile the prompt with the nsfw-prompt compiler

ALWAYS compile the prompt with `references/nsfw-prompt/SKILL.md` — both the
positive and the negative. Follow its workflow (IR, lexicon, renderers,
negatives, 自审) for every request. Dialect: the unholydesiremix checkpoint is
Illustrious-family, so route to **Illustrious tags** in its model-routing table
and take the negative from its `references/05-negatives.md` (Pony / Illustrious /
SDXL anime standard block).

Every positive starts with this fixed quality header, verbatim, before the
compiled subject/scene content:

```text
masterpiece,best quality,amazing quality,very aesthetic,absurdres,ultra detailed face,ultra detailed eyes, detailed, depth of field, high detail, best quality, very aesthetic, 8k, volumetric lighting, cinematic, dynamic shadows, depth of field, <lora:StS-Illustrious-Detail-Slider-v1.0.safetensors:0.3>, <lora:Quality_Modifier_Illust.safetensors:0.5>,
```

Do not drop, reorder, or reweight header entries. The compiled content follows
it as comma-separated tag phrases with explicit limb counts; keep the whole
positive within SDXL's token budget (trim scene depth before subject detail).
Weighted syntax works: `(term:1.3)`. Do not paste `score_9` Pony headers — the
fixed header above replaces them.

## Step 2 — Submit via rp_oneshot.py

Preview first (no API key, no spend):
```bash
python3 scripts/rp_oneshot.py --template unholy-sdxl \
  --prompt "<header + compiled positive>" \
  --negative "<compiled negative>" \
  --width 832 --height 1216 \
  --output output --project salisse --dry-run
```

Real run (needs `RUNPOD_API_KEY` exported):
```bash
RUNPOD_API_KEY=... python3 scripts/rp_oneshot.py --template unholy-sdxl \
  --prompt "<header + compiled positive>" \
  --negative "<compiled negative>" \
  --seed random --width 832 --height 1216 \
  --output output --project salisse --wait
```

Flag rules:

- `--negative` is passed on EVERY run with the negative compiled in Step 1.
  It replaces the template's baked negative text entirely (the script swaps
  node 3's text); the compiler's Illustrious negative from
  `references/nsfw-prompt/references/05-negatives.md` is the source of truth.
- `--output DIR` is required every run. Default `output` under the skill root,
  `--project salisse`; change only if the user says so.
- `--seed random` (default) for variety; `--seed N` when the user wants to
  reproduce or iterate on a render.
- Size: keep ~1M pixels. `832×1216` portrait (template default), `1216×832`
  landscape, `1024×1024` square. Let the aspect follow the described scene.
- The workflow fixes steps 30, cfg 7, `dpmpp_2m` + `karras`. Only change via the
  raw escape hatch (`--set 5.cfg=8`, `--set 5.steps=25`, `--set 5.sampler_name=euler_ancestral`)
  and only when the user asks for it.
- No image input: unholy-sdxl is txt2img only. Never pass `--image`.

Outputs land in `<output>/<project>/<job-id>/`; the script prints the absolute
output directory as its last line. Tell the user that path.

## Operational facts

- Cold start: the first job after idle takes 2–5 minutes (endpoint pulls the
  image). Subsequent jobs take seconds.
- IN_QUEUE longer than 10 minutes: workers have idle-dropped; tell the user to
  contact the package maintainer. Retrying does not help.
- Exit codes: 0 success, 1 job failed or bad override (job error printed
  verbatim), 2 missing `RUNPOD_API_KEY`. `--dry-run` works without a key.
- Spends RunPod money per job: generation is pre-approved. Do NOT ask for
  confirmation before launching a real run — compile, `--wait`, and report the
  output path. Mention spend only when a single request would launch an
  unusual number of jobs (batch/multi-seed runs beyond a handful).