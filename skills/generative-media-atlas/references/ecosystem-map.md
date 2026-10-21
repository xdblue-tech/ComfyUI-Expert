# The ecosystem map — canonical skills published outside this suite

This suite deliberately does not restate what its vendors already publish well. This file lists
what those vendors publish, explains what each source does *not* cover, and shows how to judge a
third-party skill before you trust it. It also carries the other side of the map: the open models
this suite does **not** cover yet, with where to route and when to re-check, and the hosted-only
launches that are out of scope on purpose.

**These inventories were read directly from the repository trees on 2026-09-09.** Vendor skill
repositories add and rename skills without notice, so re-check before relying on a name. The model
landscape in §6–§7 comes from a sweep of the same day against Comfy-Org's `workflow_templates`
index, the Hugging Face API and Civitai. Use `docs.comfy.org` or that template index to verify a
ComfyUI-support claim; `blog.comfy.org` moved behind a Substack subscription wall and returns
nothing to a checker.

## Contents

1. [RunPod — the infrastructure layer](#1-runpod--the-infrastructure-layer)
2. [Comfy-Org — the execution layer](#2-comfy-org--the-execution-layer)
3. [Hugging Face — the weights and memory layer](#3-hugging-face--the-weights-and-memory-layer)
4. [Black Forest Labs — and what nobody else publishes](#4-black-forest-labs--and-what-nobody-else-publishes)
5. [Judging a third-party skill](#5-judging-a-third-party-skill)
6. [Open models with no skill yet](#6-open-models-with-no-skill-yet)
7. [Hosted-only, and checked-but-unreleased](#7-hosted-only-and-checked-but-unreleased)

---

## 1. RunPod — the infrastructure layer

**`runpod/runpod-plugins-official`** (the `runpod/skills` repo redirects here).

```bash
npx skills add runpod/runpod-plugins-official
```

The repo holds **eight skills** — its README says "one router plus seven lanes" — against the six
still widely cited. `runpod-migrate` arrived before the 2026-08-23 pass and `runpod-templates`
after it `[official — repo tree and README, read 2026-09-09]`:

| Skill | What it owns |
|---|---|
| `runpod` | The router — start here when it is not obvious which of the others applies. Also hosts the golden paths |
| `runpod-usage` | Concepts: pods vs serverless, containers, storage tiers, **GPU selection** |
| `runpodctl` | The CLI — infrastructure, Hub, file transfer, SSH, doctor |
| `runpod-mcp` | The same management surface as structured MCP tool calls |
| `flash` | Writing and deploying your own code on RunPod serverless |
| `companion-clis` | Prerequisite CLIs: `hf`, `docker`, `gh`, `aws` |
| `runpod-migrate` | Migrating from the GraphQL API or REST v1 to REST v2 |
| `runpod-templates` | RunPod's official pod templates — what each image ships and how to pick one |

**One correction that outranks the inventory.** `runpodctl` 2.12.0 (2026-08-27) removed
`--stop-after` and `--terminate-after` from `pod create`, and the removal PR (#330) states the
values were forwarded to the API but **never enforced** — the pod kept running and billing. Any
skill, including older versions of this one, that calls those flags a cost guard is describing a
guard that did not exist. The guard that works is a clock outside the pod: a scheduler or shell
`trap` that calls `runpodctl pod remove` unconditionally, plus a detached watchdog on the pod.
[`comfyui-on-runpod`](../../comfyui-on-runpod/) teaches it, and RunPod's own scheduling guide
(2026-09-08) recommends the same explicit-terminate shape `[official — runpodctl v2.12.0 release
notes; PR #330]`.

**Golden paths** live under `plugins/runpod/skills/runpod/golden-paths/`. There are 24 numbered
recipes (unchanged on 2026-09-09). These are the ones this suite routes to:

| Path | Why it matters here |
|---|---|
| `02-comfyui-pod` | ComfyUI on a pod, two variants — from scratch, and the prebuilt `runpod/comfyui` image (the default) |
| `07-network-volume-handoff` | Moving a volume between contexts |
| `20-model-caching-endpoint` | Keeping weights warm for a serverless endpoint |
| `21-storage-tiers` | Which storage class for which job |
| `25-bake-vs-mount` | Bake weights into the image, or mount from a volume |

The other recipes cover autoscaling, webhooks, streaming, load balancing, multi-region and
monitoring.

**RunPod's skills are not ComfyUI-aware.** They will get you a pod with ComfyUI running, but they
stop there. They will not tell you where a text encoder must sit so `CLIPLoader` finds it. They
will not tell you that the volume mounts at a different root under serverless, or how to make a
workflow JSON open with every node resolved. That gap is exactly the scope of
[`comfyui-on-runpod`](../../comfyui-on-runpod/), and it is why this suite routes to that skill
instead of restating it.

**One deliberate omission is worth knowing about:** no skill in this suite quotes GPU prices.
Prices go stale within weeks and depend on the model. `runpod-usage` owns the general question,
and `runpodctl gpu list` owns the current number.

---

## 2. Comfy-Org — the execution layer

**`Comfy-Org/comfy-skills`**, installed as a Claude Code plugin:

```bash
/plugin marketplace add Comfy-Org/comfy-skills
/plugin install comfy-cloud@comfy-skills
```

The plugin holds **twelve skills** (unchanged on 2026-09-09), and they run on the **Comfy Cloud
MCP** server: `comfy-generate-image`, `comfy-generate-video`, `comfy-generate-audio`,
`comfy-generate-3d`, `comfy-upscale-image`, `comfy-remove-background`, `comfy-search-models`,
`comfy-search-nodes`, `comfy-search-templates`, `comfy-help`, `technique-combine-people`, and one
joke.

**What eleven of them are: command wrappers.** Each skill is a procedure. It searches for a
template, builds API-format workflow JSON, submits the job, polls, and retrieves the result. They
are genuinely good at that, and two details in them are worth borrowing no matter where you run.
First, **validate that a workflow has both an input node and a save node before submitting**.
Partner API nodes commonly produce a tensor with no save node, so the job succeeds and produces
nothing. Second, **tell OSS and paid partner routes apart by node category** (`partner/…` versus
`model/…`), not by name.

**What the twelfth is: a technique.** `technique-combine-people` is a face-consistency compositing
recipe — several people from separate references into one frame — built on the hosted
`GeminiNanoBanana2V2` and `KlingOmniProImageNode` partner nodes. It has been in the repo since its
June 2026 seed, so this is a correction to the atlas's earlier "all wrappers" framing, not new
drift. It touches this suite's character-consistency territory, but on closed partner models, so
it does not change the open-weight routing in [`model-rankings.md`](model-rankings.md) §3. Watch
this repo for more of its kind: a Comfy-published recipe on an *open* model would overlap a model
skill here and should be routed to rather than duplicated.

**What they are not: craft for open models.** They carry no per-model settings, no prompt
dialects, no LoRA hyperparameters, no licence analysis, and no comparison between models.
`comfy-search-models` finds a checkpoint, but it does not tell you which one to want. They also
**do nothing without the Comfy Cloud MCP server connected**, and they target Comfy Cloud rather
than your own ComfyUI.

**They pair with this suite cleanly.** Comfy's skills know how to execute; this suite knows what
to execute and why. Use both when you are on Comfy Cloud. Use this suite plus
[`comfyui-on-runpod`](../../comfyui-on-runpod/) when you are running your own ComfyUI.

---

## 3. Hugging Face — the weights and memory layer

**`huggingface/skills`**, installed through the Hugging Face CLI:

```bash
hf skills add hf-cli
hf skills add hf-mem
hf skills add huggingface-lora-space-builder
```

The repo holds around 25 skills. Three matter for generative media:

- **`hf-cli`** — the Hub CLI for models, datasets, spaces and repos. This is how weights get
  fetched, and it is the tool RunPod's `companion-clis` sets up.
- **`hf-mem`** — estimates the memory needed to load safetensors or GGUF weights. It is useful for
  sanity-checking a quant against a card before renting one, in a field where several vendors
  publish no VRAM figure at all.
- **`huggingface-lora-space-builder`** — builds and publishes a Gradio ZeroGPU Space that demos a
  trained LoRA, and its own frontmatter scopes it to [Qwen-Image](../../qwen-image/), Qwen-Image-Edit, LTX-Video, Wan,
  FLUX and SDXL `[official — huggingface/skills tree, read 2026-09-09]`. That is a real downstream
  step after [`character-lora-training`](../../character-lora-training/)'s evaluation, for anyone
  who wants a shareable demo rather than a Civitai upload. It is a publishing tool, not a trainer,
  and it says nothing about whether the LoRA is any good.

The rest are shaped for LLM, Spaces, SageMaker and evaluation work. Note that
**`huggingface-vision-trainer` is not a diffusion trainer** — it fine-tunes detection,
classification and segmentation models. There is still no Hugging Face skill for training a
diffusion LoRA.

---

## 4. Black Forest Labs — and what nobody else publishes

**`black-forest-labs/skills`** is the first official skills repo from a model vendor in this
suite's field. It was created 2026-01-24 and was still being pushed to on 2026-09-05, so it is
maintained, not abandoned `[official — repo metadata and skills/ listing, read 2026-09-09]`. It
ships **ten skills**:

| Skill | What it is |
|---|---|
| `bfl-api` | The hosted BFL API: auth, endpoints, polling |
| `flux-image-best-practices` | Prompting guidance for FLUX images through that API |
| `flux-3-generate` | FLUX 3: image, and video up to 20 s with native audio, t2v / i2v / v2v, plus a cached `draft_enhance` mode |
| `flux-3-keyframes-continuation` | i2v from pinned keyframes; v2v continuation from a clip's final frames |
| `flux-3-audio-dialogue`, `flux-3-cinematic-inserts`, `flux-3-product-ads`, `flux-3-archival-formats`, `flux-3-prompt-doctor`, and one more `flux-3-*` | Task recipes on the same hosted surface |

It uses the standard `skills/<name>/SKILL.md` layout the `skills` CLI scans, so
`npx skills add black-forest-labs/skills` should list them; that command was not run in this pass.

**What it changes here, and what it does not.** FLUX 3 (announced ~2026-07-25; FLUX 3 Video GA
2026-08-04) is **hosted only**. There are no weights, and BFL's own launch plan puts an open-weight
"FLUX 3 Dev" last in the rollout with no date, licence or parameter count `[pending release]`. So
the repo does not overlap [`flux-2`](../../flux-2/)'s open-weight craft (quantisation, LoRA
training, ComfyUI wiring, the [klein] licence split), and the suite keeps all of that. Where it
*may* overlap is `flux-2`'s `references/api-and-hosted.md`: `bfl-api` and
`flux-image-best-practices` cover the same hosted surface, and their content was not read
side-by-side in this pass. The rule from the previous version of this section applies: **route to
the vendor's skill for the hosted API and delete the overlap here** once someone has read both.
Both open items — the install shorthand and the overlap — resolve on one read of that repo
`[flagged — shorthand unrun, overlap unread; re-verify]`. FLUX 3 Video itself sits with Kling, Veo and Sora as an out-of-scope
hosted model in [`model-rankings.md`](model-rankings.md) §9.

**Everyone else still publishes nothing.** As of **2026-09-09**, no agent skills for their models
could be found from Stability AI, Alibaba/Tongyi, Lightricks, MiniMax, Z.ai or Civitai
`[flagged — negative result from search; re-verify]`. What exists instead is model cards, inference
repos and community write-ups of very uneven quality. That is the gap this suite fills, and it is
the reason the suite's two-bar provenance discipline exists.

**Two consequences follow.** First, for any model in this suite, the model skill *is* the closest
thing to a canonical agent-readable source. The official artefacts it cites sit in the layer
beneath it. Second, when one of these vendors does publish skills — as BFL now has — the right
move is to route to them and delete the overlap, rather than maintain a second copy. That is the
pattern [`comfyui-on-runpod`](../../comfyui-on-runpod/) already follows against RunPod.

---

## 5. Judging a third-party skill

The ecosystem is large — more than a thousand skills across community indexes — and quality varies
enormously. Ask these five questions, in order of how much each one tells you:

1. **Does it name sources, per claim?** A skill full of numbers with no attribution is somebody's
   single workflow generalised into confident prose. It may still be right, but you cannot tell,
   and you cannot re-verify it when it breaks.
2. **Is it dated?** Anything in this field without a date stamp cannot be dated later, and
   therefore cannot be maintained. A skill that says when its facts were checked is making a
   falsifiable claim.
3. **Does it separate hard facts from craft?** Filenames and licence terms fail loudly and must be
   exact. Denoise bands and rank choices are ranges that depend on your data. A skill that treats
   both with the same confidence is wrong about one of them.
4. **Is it a command wrapper, or is it knowledge?** Wrappers age with an API and are cheap to
   replace. Knowledge ages with an ecosystem and is expensive. Both are legitimate, but know which
   one you are installing, because they fail differently.
5. **Does it say who it is not for?** A skill that never routes you elsewhere has not thought
   about its own boundary. That usually means it overlaps something you already have.

**Then check what the skill does to your context.** A skill is loaded, not called: its
`description` decides when the agent reaches for it, and an over-broad description makes it fire
on unrelated work. If a newly installed skill starts appearing in answers it has nothing to do
with, the description is at fault, not the model.

---

## 6. Open models with no skill yet

**Why this section exists.** The atlas routes readers to models. When it routes them to one with
no skill behind it, the honest thing is to say so in the same breath, and to say where to go
instead. A dangling `../bernini-r/` link would tell the reader to run an install command that
cannot work. Every model here is named in plain bold, never linked, with its status as of
**2026-09-09**. The evidence is a same-day sweep of Comfy-Org's `workflow_templates/index.json`
(every template carries a date, an `openSource` flag and a usage counter), the ComfyUI commit log,
the Hugging Face API for licence and downloads, and Civitai by full cursor pagination. **Reddit and
Banodoco were not measured** in that sweep, so community sentiment below is unmeasured, not low.

### 6.1 Uncovered and worth a skill — in priority order

The Qwen-Image family headed this table until 2026-09-09; it is now [`qwen-image`](../../qwen-image/).

| Model | Why it matters | Route to, until a skill exists | Verify first |
|---|---|---|---|
| **Bernini / Bernini-R** — ByteDance, paper 2026-05-22, renderer weights 2026-06-01, 1.3B 06-09, full pipeline 06-11, training code 07-13, **Apache-2.0** | The video-*editing* job nobody in the suite owns: an MLLM planner feeding a DiT renderer fine-tuned from Wan (1.3B from Wan2.1, 14B from Wan2.2-T2V-A14B), six modes — t2i, i2i, t2v, v2v, rv2v, r2v — for relighting, restyling, subject insertion and local editing on stills and footage. Native ComfyUI 2026-06-14; `Comfy-Org/Bernini-R` at ~109k pulls/30d; and a real derivative layer already (two GGUF repacks, a 4-step LightX2V LoRA set, an fp8 Wan2.2 build, a motion-enhancer LoRA). 14B is H100-class; 1.3B is the small one. No territory clause, so it clears gates 1 and 3 where H3 does not | `github.com/bytedance/Bernini`; [`scail-2`](../../scail-2/) for the boundary — SCAIL tracks a person, Bernini re-renders a scene | Nothing on licence; VRAM for 14B before promising it on a rented card |
| **SenseNova U1 / U1.5-8B-MoT** — SenseTime, U1 2026-04-22, U1.5 2026-08-19 | Native 4K on U1.5; t2i, single- and multi-image edit, **region-controlled edit via masks, boxes and visual markers**; ~17 GB peak on a 24 GB card. **ComfyUI core since 2026-09-01** (`Support SenseNova U1.5 (CORE-411)`, #15922). Official `SenseNova-U1.5-8B-MoT-LoRAs` (2026-08-20) and community GGUFs with real pull. The ecosystem fact that outranks its download numbers: **the NoobAI team published a LoRA trainer for it** (`Laxhar/sensenova-u1-lora-trainer`, 2026-05-09) and has shipped no checkpoint since 2024-12 — the anime pool is migrating | `github.com/OpenSenseNova/SenseNova-U1`; `docs.comfy.org` for the core node | **Two things** `[flagged — re-verify]`: the Apache-2.0 claim comes from the GitHub repo, not the gated (401) HF card; and "8B" in the name sits against ~18B total for the MoT. No VRAM prose until both are read |
| **HunyuanVideo-1.5** — Tencent, 8.3B | Native ComfyUI 2025-11-24; musubi-tuner LoRA support; ~14 GB VRAM, which makes it the other open answer to "video on a 16 GB card" beside Wan 5B. **The Tencent Hunyuan Community Licence excludes the EU, UK and South Korea** — a second territory gate beside H3's | `huggingface.co/tencent/HunyuanVideo-1.5` | Whether your territory is excluded, before anything else |

### 6.2 Held — strong on paper, absent in practice

Both have core ComfyUI support, permissive licences and trainers, and both land on the axis
[`z-image`](../../z-image/) owns, where the atlas's bar is adoption rather than a leaderboard.
**Re-check on or after 2026-10-07** against the Civitai LoRA count and the Comfy template run count;
**if either has doubled, commission the skill.**

- **HiDream-O1-Image** (2026-05-08, MIT). The I1 successor — the line went I1 → O1, so there is
  no "HiDream 2". Pixel-level unified transformer, no VAE, no separate text encoder; t2i,
  instruction edit and subject-driven personalisation to 2048². Vendor benchmarks put it over
  FLUX.2 [dev] and Qwen-Image (GenEval 0.90 / DPG 89.83) and Dev-2604 reportedly placed #8 in the
  Artificial Analysis T2I Arena; both unconfirmed. Core ComfyUI 2026-05-12; musubi-tuner LoRA and
  full finetune 2026-06-08; core added its DiffSynth LoRA format 2026-09-03. **47 Civitai LoRAs,
  10 character-tagged**, ~190 template runs. Parameter count contested — card says 8B, the HF
  `totalSize` implies ~17.6B at bf16 `[contested]`. Commission regardless if the arena placing is
  independently confirmed.
- **Mage-Flow** (Microsoft Research, 2026-07-22, MIT). Six 4B checkpoints (Base/RL/Turbo for t2i
  and for edit), native 512–2048, ~18–20 GB peak, Turbo at 0.59 s per 1024² on an A100. Core
  ComfyUI in three days (#15026); ai-toolkit and SimpleTuner support. **Zero Civitai LoRAs** after
  seven weeks against ~181k repack downloads — the clearest case in the sweep of distribution
  without practice. `microsoft/Mage-Flow*` returns HTTP 401, so the licence text is secondhand
  `[flagged — re-verify]`.

### 6.3 Open and reachable, no job the suite lacks

One line each, so the next sweep does not re-discover them.

- **FireRed-Image-Edit-1.1** (Xiaohongshu, Apache-2.0, 2026-03-09) — 10+ element fusion, strong
  identity retention, ~30 GB. **2,740 template runs**, the busiest open edit template after Qwen
  Edit 2509, and zero Civitai LoRAs. If [`qwen-image`](../../qwen-image/) wants a rival, this is the first candidate.
- **GLM-Image** (Z.ai — SCAIL-2's lab — MIT, 2026-01-08) — 16B, edit, style transfer,
  identity-preserving, multi-subject; ~23 GB with offload; 1,105 HF likes. ComfyUI custom node only.
- **Open, ComfyUI-reachable, and without a LoRA ecosystem.** None of these has a job the suite
  lacks:
  - **ERNIE-Image** (Baidu, Apache-2.0, 2026-04-07).
  - **LongCat-Image** (Meituan, Apache-2.0, 2025-12-04) — 6B, bilingual, strong typography.
  - **NVIDIA PixelDiT** (2026-05-25, licence `other`).
  - **Boogu Image 0.1** (Apache-2.0, 2026-06-16, 10B, no named lab).
  - **Lens** (Microsoft, MIT, 2026-05-20, 3.8B).
  - **LLaDA-Image** (Ant, 2026-08-28) — days old, watch only.
  - **JoyAI-Image-Edit** (JD) — template shipped with zero runs.
  - **Bria FIBO 1.5**.
  - **LTX 2.3 Image** (2026-08-28) — an image model inside a video family. It is
    [`ltx-2-5`](../../ltx-2-5/)'s to note.
- **Three licence traps.** `nvidia/PiD` is NSCLv1 non-commercial and ships with a Comfy template.
  **Pony V7** bars inference services, >$1M companies and professional video. **HunyuanImage 2.1 /
  3.0** exclude the EU, UK and South Korea; 2.1 has zero Civitai LoRAs, and 3.0 needs three 80 GB
  cards and has no trainer anywhere. All three are on the elimination ladder in SKILL.md and in
  [`model-rankings.md`](model-rankings.md) §8.
- **Already covered, do not re-file.** **Wan-Animate-2** (Apache-2.0, 2026-08-07) is in
  [`wan-2-2`](../../wan-2-2/). **SeedVR2** is the suite's upscaler, in
  [`image-production-workflows`](../../image-production-workflows/). Utility-only, no skill
  needed: Wan-Dancer, WanSong, Wan-Streamer, VOID video inpainting, Depth Anything 3, SAM3.

---

## 7. Hosted-only, and checked-but-unreleased

**The closed frontier, so the silence is not ambiguous.** Every one of these is hosted with no
open component, and therefore out of scope for this suite by design. Dates are the vendor's
announcement, not Comfy's partner node, which can lag by weeks.

*Image, hosted only:*

- **Seedream 5.0 Pro** — 2026-07-08. The busiest hosted image template in Comfy.
- **Qwen-Image 3.0** — 2026-07-20. No weights, no licence, no technical report. The open family is
  [`qwen-image`](../../qwen-image/), whose `api-and-hosted.md` covers the 3.0 Pro API nodes.
- **Grok Imagine Image 2.0** — 2026-08-07.
- **Meta Muse Image / Muse Video** — 2026-07-07. Meta's first, and agentic: it invokes search and
  code mid-generation.
- **Nano Banana 2** — GA 2026-05-28, Lite 2026-06-30.
- **GPT Image 2** — 2026-04-21; **2.5** — 2026-09-08.
- **Midjourney V8.2** — 2026-07-24; a unified **V8 Edit Model** — 2026-08-27.
- **Reve 2.0 / 2.1**, **Recraft V4.1**, **Ideogram P-Image** (2026-07-30 — there is no "Ideogram 5").

*Video, hosted only:*

- **FLUX 3** — 2026-07-23; **FLUX 3 Video** GA 2026-08-04 (§4).
- **Seedance 2.5** — 2026-07-31.
- **Wan 3.0** — public beta 2026-08-06, API only.
- **Gemini Omni Flash** and **Omni 1.1 Flash** — 2026-08-27. Google's active line, which
  supersedes Veo. Comfy archived Veo 3 on 2026-08-28.
- **Kling 3.0 Turbo + Omni editing** — 2026-06-17.
- **Luma Ray3.2**, **Vidu S1**, **Runway Characters** (2026-07-19), **Mirage Avatar X**,
  **Pixverse V6** (2026-08-25), **Hunyuan3D 3.0** (Comfy Cloud API only).
- **MiniMax "Hailuo 03"** does not exist. What shipped are hosted tiers: **H3 Max** (2026-09-02)
  and **H3 Max Turbo** (2026-09-04).

**Checked on 2026-09-09 and not released.** Listed so the next sweep does not redo them.

- **Z-Image-Edit** and **Z-Image-Omni-Base** — still "to be released" in the Z-Image model zoo.
  `Tongyi-MAI/Z-Image#163` (2026-04-04) is unanswered.
- **FLUX 3 open weights** — none. "FLUX 3 Dev" is a roadmap name.
- **Wan 2.5 / 2.6 / 2.7 / 3.0 weights** — none. The "Apache-2.0 1.3B + 14B" story for 3.0 appears
  in no Alibaba announcement.
- **LTX-3**, **HunyuanVideo 2 / 3**, **HunyuanImage 3.1+** — not found.
- **Chroma 2** — `Chroma2-Kaleidoscope` exists as an idle work-in-progress with zero recent
  downloads. It never had a release.
- **HiDream 2** — does not exist. O1 is the successor line (§6.2).
- **Illustrious 3.0 / 3.5** — platform-gated, no open weights. Shakker shuts down 2026-09-30.
- **NoobAI-XL successor** — none. The team moved to SenseNova (§6.1).
- **Stability SD4** — no primary source. Treat aggregators publishing specific dates as fabricated.
- **Misnamed hosted models:** Runway Gen-5 (Gen-4.5 is flagship), Luma Ray 4 (Ray3.2), Veo 4
  (Gemini Omni), Sora 3 (the API tops out at `sora-2-pro`), Krea 3 (preview only, no card).
