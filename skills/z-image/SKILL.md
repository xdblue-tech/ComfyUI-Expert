---
name: z-image
description: >
  Authoritative guide for the Z-Image model family (Z-Image and Z-Image-Turbo, Alibaba Tongyi Lab) in ComfyUI
  or the diffusers API. Use this whenever the user touches Z-Image in any way, even obliquely: choosing
  between Z-Image and Turbo, or weighing Z-Image against other models. Installing or setting it up in ComfyUI
  (file layout, loaders, quantisation, ControlNet). Writing or fixing prompts, given that the Qwen-3 LLM
  encoder needs sentences not tags: realism, killing plastic/waxy stock-photo skin, bilingual text rendering,
  and high/low-angle gaze control. Pose control and structural conditioning with the Fun Union ControlNet
  (Pose, Depth, Canny, HED, Scribble), the ModelPatchLoader + QwenImageDiffsynthControlnet nodes, Turbo-only,
  V2.1 model files. Face identity, where there is no PuLID for Z-Image and a character LoRA applied via
  FaceDetailer is the standard approach. Building single- or multi-stage workflows (hires refine, tiled
  upscale, face detailer, img2img/inpaint) with practical sampler/CFG/denoise/resolution/step settings.
  Using LoRAs: loading any downloaded style/realism/character LoRA with the right node, weight-by-type
  tuning, stacking with rgthree, the diffusers-format QKV silent-failure gotcha, and Base↔Turbo
  cross-compatibility. Generating a dataset and training a LoRA with the Ostris AI-Toolkit. Creating a
  consistent original character (anchor image → edit-model dataset factory → character LoRA → FaceDetailer
  deployment) and its multi-outfit and multi-character limits. Training a style LoRA with diverse-subject
  datasets, prose captions, and XY-grid evaluation. Using Z-Image-Turbo as the realism refiner in a
  mixed-model pipeline, e.g. refining SDXL renders. Or checking what the family's Apache-2.0 licence permits
  for commercial work compared with the gated models in this suite. Use this for any question about Z-Image
  in any context. Choosing between models, comparing them, or working out which skills and install commands
  a job needs is [`generative-media-atlas`](../generative-media-atlas/)'s job. Start there when the model is
  not already settled.
---

# Z-Image Family

Z-Image is Alibaba Tongyi Lab's open-weights image generation family. It is a **6B-parameter Scalable Single-Stream DiT (S3-DiT)**: text tokens, visual semantic tokens, and image VAE tokens all travel through one unified sequence. The text encoder is **Qwen 3 4B**, an LLM-grade encoder that parses your prompt as natural language. It is bilingual (English + Chinese). Apache-2.0 covers the whole family.

## Variant selector

| Variant | Distilled | Steps | CFG | Negatives | VRAM ³ | Licence | Use when… |
|---|---|---|---|---|---|---|---|
| **Z-Image** | No | 25–50 | 3.0–5.0 | Yes | ~16 GB comfortable ³ | Apache-2.0 | final renders, LoRA training base, fine-grained negative control |
| **Z-Image-Turbo** | Yes (Decoupled-DMD) | 8 effective ¹ | 1.0 ² | No ² | ~16 GB comfortable; 6 GB workable ³ | Apache-2.0 | rapid iteration, dataset generation, drafting |
| **Z-Image-Edit** | — | — | — | — | — | Apache-2.0 | instruction-based image editing — *announced, not yet released* `[pending release]` |
| **Z-Image-Omni-Base** | — | — | — | — | — | Apache-2.0 | generation + editing in one model — *announced, not yet released* `[pending release]` |

> ¹ **ComfyUI KSampler: 8 steps** (sampler `res_multistep`, scheduler `simple`, per the official template). In **diffusers**, set `num_inference_steps=9`. The official model card explains that "this actually results in 8 DiT forwards".
> ² **CFG 1.0 in a ComfyUI KSampler *is* guidance-off.** It equals `guidance_scale=0` in diffusers, and it is the value in the official template. Never type CFG 0.0 into a KSampler: that outputs the unconditional result and ignores the prompt. Negatives do nothing at this setting, so zero them with `ConditioningZeroOut`. Raising CFG to 1.2–1.5 re-introduces weak negative subtraction, but it roughly doubles the cost and causes over-saturation. It is a community workaround, not a supported feature `[contested]`. Use it only for stubborn artefacts.
> ³ **These are comfort bands, not floors.** Tongyi-MAI publishes no minimum for either variant. At 16 GB, the bf16 DiT and the Qwen-3 encoder both stay resident. Below that, ComfyUI shuttles weights to system RAM between passes, which costs time but not the ability to render. Official int8 DiT builds (~6.2 GB) ship for **base as well as Turbo**, and Turbo is reported running end-to-end on an **RTX 2060** `[community — Royal_Carpenter_1338]`. So read a small card as "pull a quantised build and expect it to run slower", not as "unsupported". See **Setup & ecosystem**.

**Default workflow:** Draft in Turbo (8 steps, CFG 1.0) to find the composition and seed. Re-render the keepers in Z-Image (40 steps, CFG 4.0) for the final asset `[community — re-verify]`. Use the same prompt and seed, but expect Z-Image to reinterpret them slightly. That is normal. For layered production pipelines, see **Production pipelines & mixing models** below.

> **A `../link/` on this page that doesn't resolve is a skill you have not installed, not a broken
> page.** [`generative-media-atlas`](../generative-media-atlas/) is the map of this suite: which
> model fits a job, which skills that job needs, and the commands to install them. It works on its
> own, so it is the one to add first — `npx skills add ryannel/skills --skill generative-media-atlas`

---

## The one rule that changes everything

Qwen 3 4B is an LLM-grade encoder. It parses syntax and clause structure, so write a **sentence**, not a tag list. This rule comes from the encoder class, not from folklore. The same rule governs FLUX.2's Mistral/Qwen3 encoders. The opposite rule, weighted tags and verbatim trigger tokens, governs CLIP models like SDXL. Prompting dialect, LoRA trigger handling, and caption style all follow the encoder here.

One refinement, which does not undo the rule: the encoder is a proxy for the **caption corpus**, and on two suite models the two come apart. [`anima`](../anima/) has an LLM encoder but takes weighted Danbooru tags. [`ideogram-4`](../ideogram-4/) also has one, but takes JSON. The encoder class sets the ceiling on what a dialect *can* express. The corpus decides what the model is actually fluent in. The two agree here, and they agree on most models, which is why the encoder remains the right first guess. But **check what a model was captioned on before inferring its dialect from its encoder name.**

| Don't | Do |
|---|---|
| `1girl, solo, masterpiece, 8k, best quality` | *A young woman standing alone in a sunlit kitchen, candid documentary photograph.* |

Contradictions do not average out; they create uncanny artefacts. Pick one medium, one mood, one lens.

**Prompt anatomy** — six parts in this order (details in `references/prompting-guide.md` §1):

1. **Subject**: who or what, with concrete details. Include at least one non-idealised trait if the subject is a person
2. **Scene**: where and when. Z-Image has strong geographic priors, so name locations specifically
3. **Composition**: shot type, framing, lens (focal length + aperture: "85 mm f/1.4")
4. **Lighting**: source + direction + quality + colour temperature. Always name all four
5. **Style / medium**: exactly one
6. **Constraints**: negatives in Z-Image, positive phrasing inside the prompt in Turbo

**Length sweet spot:** 80–250 words. Attention drifts past about 75 tokens, so put the subject, the key text-to-render, and the primary lens in the first 75 `[community]`. The soft cap is 512 tokens (about 384 words). Set `max_sequence_length=1024` locally if you genuinely need more.

---

## Setup & ecosystem

Z-Image runs in **ComfyUI core** with no custom nodes since **v0.3.75** (Nov 2025). The DiT is **not a checkpoint**, so load it with three separate nodes: **Load Diffusion Model** + **CLIPLoader** (the Qwen-3 encoder) + **Load VAE**.

**File layout**: download the ComfyUI-repackaged files from the `Comfy-Org/z_image` and `Comfy-Org/z_image_turbo` Hugging Face repos:

| File | ComfyUI folder | Loader node |
|---|---|---|
| `z_image_bf16.safetensors` / `z_image_turbo_bf16.safetensors` (DiT) | `models/diffusion_models/` | Load Diffusion Model |
| `qwen_3_4b.safetensors` (text encoder, **shared**) | `models/text_encoders/` | CLIPLoader |
| `ae.safetensors` (VAE, **shared** — the Flux.1 VAE) | `models/vae/` | Load VAE |

The text encoder and VAE are common to every variant, so download them once. Official ComfyUI templates ship for all three entry points: **`image_z_image`** (base), **`image_z_image_turbo`** (Turbo), and **`image_z_image_turbo_fun_union_controlnet`** (Turbo plus an official Fun union ControlNet for structural conditioning). See also comfyanonymous's `ComfyUI_examples/z_image` page.

**Stock node settings** (from the official template): set the **CLIPLoader type to `lumina2`**. Use **`EmptySD3LatentImage`** for the latent, not the legacy `EmptyLatentImage`. Put a **`ModelSamplingAuraFlow` node (shift 3)** on the model path before the sampler.

**Low-VRAM / quantisation:**
- **Official** (in the Comfy-Org repos): int8 DiT `z_image_int8_convrot.safetensors` / `z_image_turbo_int8_convrot.safetensors` (~6.2 GB each). This build exists for **base as well as Turbo**, and it ships with its own ComfyUI templates, `image_z_image_int8` and `image_z_image_turbo_int8`. There is also an NVFP4 DiT `z_image_turbo_nvfp4.safetensors` (~4.5 GB, **Turbo only**), plus quantised encoders `qwen_3_4b_fp8_mixed` (~5.6 GB) and `qwen_3_4b_fp4_mixed` (~3.5 GB).
- **Community** (requants, not official): fp8 DiT (e.g. Kijai's `fp8_scaled_e4m3fn`) and GGUF Q2–Q8 (e.g. unsloth). GGUF requires the `ComfyUI-GGUF` custom node (city96). There is no official fp8 DiT.
- **Nunchaku SVDQuant** (third-party, actively maintained): 4-bit **Turbo** DiT builds in `nunchaku-ai/nunchaku-z-image-turbo`, INT4 for pre-Blackwell cards and NVFP4 for the 50-series, at three ranks (32/128/256). They load through the `ComfyUI-nunchaku` custom node's own loader and LoRA nodes, not `Load Diffusion Model`. Z-Image support has been stable since v1.1.0 (2025-12-27), and v1.2.0 (2026-01-12) claims a further 20–30% speed-up. This is a fourth low-VRAM path next to the official int8/NVFP4 files and the community fp8/GGUF requants, and the one to try when the official NVFP4 file does not match your card's generation `[community — nunchaku-ai; strong]`. Turbo builds only, as of 2026-09-09. Wiring note in `references/setup-and-workflows.md §6`.
- **The 16 GB number is a comfort band, not a floor**, and nothing official states a floor. At 16 GB, the bf16 DiT stays resident alongside the encoder with no offload. Below that, ComfyUI pages weights in and out per pass, so you lose throughput, not capability. Quantised builds sit comfortably on 6–8 GB `[community — re-verify]`, and Turbo is reported working on an **RTX 2060** `[community — Royal_Carpenter_1338]`. Treat a small card as a reason to pull an int8 or NVFP4 DiT plus a quantised encoder, not as a reason to skip the model. Base is the harsher variant to run this way, for a reason unrelated to size: it wants 25–50 steps where Turbo wants 8, so you pay the offload penalty several times over per image.

**diffusers API:** `from diffusers import ZImagePipeline`. This shipped in **stable diffusers ≥ 0.36** (`pip install -U diffusers`); the model card's `pip install git+…` line is launch-day legacy from before 0.36. Latent-space `ZImageImg2ImgPipeline` and `ZImageInpaintPipeline` also exist. These run on the released base/Turbo weights, and they are **not** the still-unreleased, instruction-based Z-Image-Edit.

**ControlNet (pose, depth, canny, and more).** The official Fun Union ControlNet for Z-Image **Turbo** is from Alibaba PAI. A single Union model file handles all conditioning types. File: `Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors` (6.71 GB) or the lite variant (2.02 GB) → `models/model_patches/`. The ComfyUI nodes are both built-in core: `ModelPatchLoader` loads the patch and `QwenImageDiffsynthControlnet` applies it. Pose uses `DWPreprocessor` (DWPose). **Turbo only** — base Z-Image ControlNet has no ComfyUI support yet. The full file table, all V2.1 variants, node wiring, and per-type preprocessors are in **`references/setup-and-workflows.md §9`**.

**Face identity & characters.** No PuLID or IP-Adapter face model exists for Z-Image. The character LoRA *is* the path, loaded at the FaceDetailer stage (references/setup-and-workflows.md §6). The full character pipeline (anchor image, the [Qwen-Image-Edit](../qwen-image/) dataset factory, rotation/expression coverage, multi-outfit and multi-character craft, failure modes) lives in **`references/characters.md`**. A method comparison is in `references/setup-and-workflows.md §10`.

---

## Per-variant settings

### Z-Image (undistilled)

Preserves the full training signal, so it has the best texture diversity and is the best base for LoRA training.

- **Steps:** 25–50. The official ComfyUI base template uses **25**; the diffusers card recommends 28–50 (example: 50). 40 is a safe all-rounder
- **CFG / guidance:** 3.0–5.0. 4.0 is typical, and it is the value in the official base template and the diffusers example
- **Sampler:** the official ComfyUI base template uses **`res_multistep` / `simple`** with `ModelSamplingAuraFlow` shift 3, the same chain as Turbo. Community finetune pipelines often substitute `euler` / `simple` (see `references/setup-and-workflows.md`)
- **Resolution:** 1024×1024 natively (officially 512–2048 px); 1:1, 4:3, 3:4 all work well
- **Negative prompts:** Use them. Baseline: `text, watermark, extra fingers, deformed hands, plastic skin, waxy skin, airbrushed, blurry, oversaturated`. Keep the negative under ~180 characters to avoid side effects `[community — re-verify]`
- **Seed diversity:** High; randomise freely

### Z-Image-Turbo (distilled)

Decoupled-DMD distillation removes the CFG dependency. This is the fastest path from prompt to draft.

- **Steps:** ComfyUI KSampler **8 steps**; diffusers `num_inference_steps=9` (8 DiT forwards)
- **Sampler (official ComfyUI):** `res_multistep` / `simple`
- **CFG / guidance:** diffusers `guidance_scale=0`; **ComfyUI KSampler CFG 1.0** (cfg 1 = guidance-off — never 0.0 in a KSampler)
- **Resolution:** 1024×1024
- **Negative prompts:** Do not use them. Phrase all constraints positively inside the prompt
- **Seed diversity:** Lower than Z-Image; Turbo converges toward the strongest mode
- **LoRAs:** start around 0.7–0.8 and sweep 0.5–1.2. There is **no hard 0.8 cap**: style LoRAs often want 0.3–0.5, and character LoRAs 0.7–1.0. These are per-LoRA tunings read off named authors' cards, not a model-wide ceiling `[community — per-LoRA, not a hard rule]`, so read the card of the LoRA you actually loaded. A LoRA that "does almost nothing" on a current ComfyUI is a strength or variant problem, not a loading one. The old diffusers-format QKV loading bug was fixed in core in March 2026 and only bites a pinned or frozen install (footnote in `references/setup-and-workflows.md §6`). **Check which variant a LoRA you downloaded was trained on before wiring it in.** The published ecosystem is lopsided: ~2,190 LoRAs tagged `ZImageTurbo` against ~670 on base `[community — Civitai census, Aug 2026]`. So the odds are it is a Turbo LoRA even when your graph is a base one. The shared S3-DiT means a mismatched LoRA loads without error either way, which is exactly what makes the mismatch easy to miss. Full loading, stacking, and cross-compat guidance: `references/setup-and-workflows.md §6`. Training: `references/lora-training.md`

---

## Production pipelines & mixing models

For production results, don't render once. **Layer passes: ZIB builds structure, ZIT refines and upscales.** Generate at a *low* base resolution, then climb. ZIB and ZIT work together here; it is not an either/or choice. Sometimes you mix in *other* model families entirely, and there Z-Image-Turbo's usual job is the realism finish on someone else's render. For that, see **Where Z-Image sits in the suite** below and `references/setup-and-workflows.md §11`. That section carries the decode-to-pixels handoff rule that makes cross-family chains work at all.

**The pipeline** (every optional stage is bypassable, so you can preview cheaply and pay for the heavy passes only once the base is right):

1. **1st-gen (ZIB)**: composition + pose at low base res (e.g. 640×960). Judge *layout only*, and reroll the seed freely.
2. **Latent upscale**: `LatentUpscaleBy`, `bislerp`, ×1.7.
3. **2nd-gen (ZIT)**: hires refine for fingers, face, text. Judge *fine details* here.
4. **Tiled SD upscale (ZIT)**: `UltimateSDUpscale` ×2, low denoise (~0.23).
5. **FaceDetailer (ZIT)**: re-render the face. **Swap the character LoRA in here**, not in the base gen.
6. *Optional:* manual inpaint, skin contrast, SeedVR2 final upscale.

Final resolution ≈ **base × 1.7 × 2**. Keep the **same seed across passes** for consistency.

The ×1.7 and ×2 multipliers and the ~0.23 tiled-upscale denoise come from a widely shared community layered pipeline `[community — layered ZIB+ZIT graphs; convergent]`. Those graphs often run **custom Z-Image finetunes**, but the *architecture* (generate small, upscale, refine, detail) transfers to stock weights unchanged. Treat the exact numbers as well-tuned starting points to nudge, not as stock requirements. Sourcing and the per-stage table: `references/setup-and-workflows.md §3`.

**The ladder also runs backwards.** Drafting in ZIT and finishing with a low-denoise **base** pass is a reported alternative that buys texture with compute `[community — Royal_Carpenter_1338]`. In practice this means loading the ComfyUI Z-Image upscaling template with the base DiT where it normally loads Turbo. The undistilled weights carry the texture diversity that Turbo's 8-step trajectory flattens, and they bring back real CFG and real negatives. Use it as a finishing pass on keepers, not as a default, because the expensive model is now doing the expensive high-resolution passes. That is the inverse of why the ladder above is ordered as it is. Settings, the cost, and why it suits a small card: `references/setup-and-workflows.md §3`.

**Character LoRA tip:** generate the base image from a *detailed prompt* with **no** character LoRA. Then swap the LoRA in at the FaceDetailer stage, and match its prompt to the image. Otherwise you get the LoRA's generic default face. A LoRA on ZIB shapes structure, and on ZIT it shapes detail. Load it on **both** for maximum likeness `[community — MyAIForce; strong]`.

Full per-stage settings, the resolution table, and optional improvement layers: `references/setup-and-workflows.md`.

---

## Realism — killing the plastic default

Z-Image defaults to airbrushed stock-photo gloss. The fix is **photographic specificity**, not adjectives `[community]`.

Stack all three:
1. A real **camera body + lens**: "Sony A7R IV, 85 mm f/1.4 GM"
2. A real **film stock or sensor emulation**: "Kodak Portra 400, fine grain"
3. One **non-idealised human feature**: "visible skin pores, a small mole below the left jaw"

"Realistic", "8k", and "masterpiece" do almost nothing. See `references/prompting-guide.md` §2 for vocabulary tables and texture anchors.

---

## Gaze control for high- and low-angle shots

The characteristic failure is that the model tilts the subject's chin to keep eye contact with the lens, which defeats the angle. Fix it by giving the gaze a **concrete anchor away from the camera** `[community]`.

- **High angle:** "head bowed, gaze fixed on the book in her lap". Never pair "looking at viewer" with "from above"
- **Low angle:** "chin lifted, gaze locked on the horizon far above the camera"
- **Z-Image:** add the negative `looking at camera, looking at viewer, eye contact, face tilted up`
- **Turbo:** phrase it positively: "completely unaware of the camera, candid documentary photograph"

A high-angle **face close-up** is the hardest case, because there is no surface below for the gaze to anchor to. Widen to a medium shot or accept a retry. See `references/prompting-guide.md` §5 for working prompt templates.

---

## Failure modes & QC

Characteristic artefacts of the family and the fastest fix for each:

| Artefact | Cause | Fix |
|---|---|---|
| Plastic / waxy / airbrushed skin, stock-photo gloss | Family default aesthetic | Stack camera body + film stock + one non-idealised feature (see the realism technique above) |
| Chin tilts up/down to keep eye contact on a high/low angle | Model maintains eye contact with the lens | Anchor the gaze on a concrete object away from the camera; Z-Image: add `looking at camera` negatives |
| Over-saturated, over-cooked edges (Turbo) | CFG raised above the guidance-free baseline, or a LoRA loaded at 1.0 | Keep Turbo at CFG 1.0 (ComfyUI); load the LoRA at ~0.8 |
| Outputs look samey across seeds (Turbo) | Distillation converges toward the strongest mode | Switch to Z-Image for diversity, or vary the prompt — not just the seed |
| Garbled / wrong rendered text | Too many words, or curly/smart quotes | ≤10 words per text block, wrapped in straight double quotes |
| Deformed hands, extra fingers | Common DiT weakness | Z-Image: add `extra fingers, deformed hands` negatives; Turbo: re-roll the seed or describe the hands positively |
| Uncanny / contradictory look | Mixed mediums, moods, or lenses in one prompt | One medium, one mood, one lens — contradictions don't average out |
| Negatives seem ignored (Turbo) | Guidance-free (CFG 1.0 in ComfyUI) doesn't apply negatives | Phrase the constraints positively inside the prompt (the CFG > 1 workaround is in the Turbo-negatives note under *How to read the claims in this skill*) |
| Text, a tattoo or a logo you asked for once is stamped onto shoulder, arm or back after the tiled upscale | `UltimateSDUpscale` samples each tile against the *whole* prompt, but each tile only sees its own patch, so a smooth-skin tile has nothing else to satisfy "a tattoo reading X" with | Give the upscale pass a **simplified prompt** with the localised text/mark removed, or use the per-tile conditioning switch most graphs expose — `references/setup-and-workflows.md §7` `[community — convergent]` |

---

## Pre-flight checklist

Before hitting Queue Prompt:

1. Sentence, not a tag list?
2. Camera body + lens named?
3. Light source + direction + quality + colour temperature named?
4. At least one non-idealised feature if the subject is a person?
5. Exactly one style / medium?
6. **Turbo:** all constraints phrased positively inside the prompt?
7. **Z-Image:** negative prompt under ~180 characters, free of contradictions?
8. High / low angle shot: gaze anchored on a concrete object away from the lens?
9. LoRA dataset: only the rotation / shot-size clause varies between set images?
10. Rendered text wrapped in straight double quotes, under ~10 words per block?

---

## Where Z-Image sits in the suite

Choose the model for the job. Defaults like realism direction and prompting dialect are model-specific, not universal:

| Job | Z-Image | Reach for instead |
|---|---|---|
| Consistent characters | Strong via character LoRA + FaceDetailer (`references/characters.md`); no adapter shortcut | [`qwen-image`](../qwen-image/): Edit-2511 is the suite's no-training identity engine and the dataset factory `references/characters.md` builds on, with no adapter needed; [`flux-2`](../flux-2/) for PuLID-class adapters and its own multi-reference path |
| Style / character LoRA ecosystem | Fully supported and cheap to train (AI-Toolkit; `references/lora-training.md`) | [`sdxl`](../sdxl/) for the deepest trained-LoRA ecosystem and mature recipes; [`character-lora-training`](../character-lora-training/) for the craft that transfers across all of them |
| In-image typography | Workable for short bilingual text | [`ideogram-4`](../ideogram-4/): the open-weights typography leader |
| Structural control (pose/depth/canny) | Fun Union ControlNet, **Turbo only** | [`sdxl`](../sdxl/) for the most mature, complete control stack |
| Photoreal faces and skin | **The headline strength**, once you pay the anti-gloss tax above. It is also the suite's standard face-pass finisher. This is settled rather than claimed: photoreal faces are the axis the rest of the suite routes *here* for, and [`sdxl`](../sdxl/), [`krea-2`](../krea-2/), [`ideogram-4`](../ideogram-4/) and [`anima`](../anima/) all send you here on it | Nothing in the suite is better on faces and skin themselves. [`sdxl`](../sdxl/) when the frame needs pose/depth control or an identity adapter more than it needs skin — compose and control there, finish here. [`krea-2`](../krea-2/) if you want a *non*-photoreal look without fighting a realism prior |
| Aesthetic range / stylistic exploration | One strong realism-leaning default | [`krea-2`](../krea-2/): deliberately no house look (style refs, moodboards, official style LoRAs). Z-Image is *its* standard face/detail finisher, so the pairing runs both ways |
| Anime and illustration | Weak: the realism prior fights stylisation, and the anime LoRA ecosystem is thin here | [`anima`](../anima/) is the closest neighbour: also small (2B) and also trainable on a consumer card, but its **weights** are non-commercial where Z-Image's Apache-2.0 is not. Anima's outputs are commercially free, so the difference bites only when you ship or host the model itself. [`sdxl`](../sdxl/) via the Illustrious/Pony dialects for the mature anime checkpoint ecosystem |
| Commercial use under the licence | **Apache-2.0, weights and outputs alike**: no revenue cap, no territory clause, no gate. The least legally encumbered model in the suite | Nothing is freer, so the reach-for runs in reverse. If a job forces you onto a gated model, read that skill's licence section first: [`ideogram-4`](../ideogram-4/) is non-commercial on its open weights, and [`anima`](../anima/) is non-commercial on the weights but not on what they produce |
| Mixed-model pipelines | **The realism refiner**: ZIT finishes other models' renders (`references/setup-and-workflows.md §11`), it is the standard face-pass and repair-inpaint partner for [`krea-2`](../krea-2/) scenes (~0.2 denoise `[community]`), and it is the skin finisher for a [`qwen-image`](../qwen-image/) composition. The anchor travels the other way: Qwen-Image-Edit multiplies it into the LoRA set | [`image-production-workflows`](../image-production-workflows/) for the cross-model craft itself |
| Making it move | Still images only | [`wan-2-2`](../wan-2-2/): image-to-video. Wan's I2V path is much stronger than its text-to-video, so a still locked here (character LoRA + FaceDetailer included) is what actually controls the shot. The character work in `references/characters.md` is the upstream half of consistent characters in video |
| **Choosing between all of these in the first place** | — this table is one model's view of the suite | [`generative-media-atlas`](../generative-media-atlas/): the whole suite ranked by job (realism, identity, LoRA trainability, control, licence, video), the elimination ladder that settles most choices, and end-to-end routes across several skills |

---

## Licence & limitations

**Licence:** Apache-2.0 for all currently released variants (Z-Image and Z-Image-Turbo). Z-Image-Edit and Z-Image-Omni-Base are still unreleased as of 2026-09-09; both are marked "to be released" on the official GitHub (issues #163 and #169). Verify their licences before use when they land.

**Release timeline:** Z-Image-Turbo shipped 26 Nov 2025, and the undistilled Z-Image base followed on 27 Jan 2026. This is a fast-moving family, so re-verify volatile specifics (quant filenames, VRAM numbers, ComfyUI template details, LoRA tooling) before relying on them.

**VRAM for Z-Image (undistilled):** Tongyi-MAI has not published an inference VRAM figure. At 6B parameters in bfloat16, plus the shared encoder and VAE, **16 GB is the comfort band**: the point at which the weights stay resident and no offload tax is paid. 24 GB+ adds headroom for larger batches or resolutions. That number is inferred from the parameter count, not measured or published `[flagged — re-verify]`, and it is **not a floor**. An official int8 DiT (~6.2 GB) with its own ComfyUI template ships for base as well as Turbo, and Turbo is reported running on an RTX 2060 `[community — Royal_Carpenter_1338]`. Below the band you lose speed, not the model.

---

## How to read the claims in this skill — two bars, by claim type

This skill holds two kinds of claim to two different standards, because they fail in two different ways.

**Hard facts — must be exact or it breaks.** This bar covers the architecture (6B S3-DiT, Qwen-3 4B encoder), the Apache-2.0 licence, exact filenames, node names (`ModelSamplingAuraFlow`, `EmptySD3LatentImage`, `LoraLoader` / `LoraLoaderModelOnly`, `QwenImageDiffsynthControlnet`), the CLIPLoader `lumina2` type, what a setting *numerically* does (CFG 1.0 = guidance-off = `guidance_scale=0`), the diffusers pipeline classes, and the QKV / ComfyUI PR #12717 LoRA-loading fix. **The source of truth is official**: the model card, the ComfyUI template, ComfyUI PRs and issues, and diffusers. That is where these claims are verified. A wrong filename 404s the download, and a wrong node name won't wire. These are also the **volatile** claims: filenames, VRAM figures, quant builds, tooling, and template details move week to week in this young family. **Re-verify them before relying on them, regardless of who said it.**

**Craft — what actually makes a good image.** This bar covers the sampler/CFG/denoise ladders, the multi-stage ZIB→ZIT pipeline numbers and the ×1.7/×2 upscale ladder, LoRA weights by type and stacking, the realism camera/film stack, the gaze anchors, and the 8-point rotation phrasings. **The authoritative source here is the community**: named workflow authors (Mickmumpitz, WeirdWonderfulAI, MyAIForce, Khanykov01, neurocanvas, Cordina, rgthree) and reproducible Civitai/Reddit/Banodoco results that have run thousands of generations. The model card is *not* the source here; it ships one example image and moves on. Community craft is deep, battle-tested knowledge, not a lesser tier, and this skill states it with confidence. Where a craft claim is given as a range or flagged "tune this," that is because **your weights, finetune, and resolution differ from the author's**, not because community sourcing is suspect. Where strong named sources genuinely disagree, as on exact LoRA weights and the magnitude of Base→Turbo transfer, the skill shows the disagreement rather than papering over it.

Contested and unresolved points, all greppable as markers:
- **Turbo negatives:** Tongyi-MAI states negatives are inert at the guidance-free setting (CFG 1.0 KSampler / `guidance_scale=0`). ComfyUI users report that CFG 1.2–1.5 re-introduces weak negative subtraction. The official guidance is the fact; CFG > 1 on Turbo is a community workaround, not a supported feature `[contested]`.
- **LoRA cross-compat & weights:** that a LoRA loads on either variant is a fact (shared S3-DiT). Whether it transfers *cleanly*, and the exact per-type weights, are contested craft `[contested]` — see `references/setup-and-workflows.md §6`.
- **Base VRAM:** the 16 GB figure above is inferred from the parameter count, not measured; Tongyi-MAI publishes none `[flagged — re-verify]`. Read it as a comfort band, not a requirement. Every *measured* data point is a community one, and they run lower, down to quantised Turbo on an RTX 2060 `[community — Royal_Carpenter_1338]`.
- **Tooling that used to be absent and is now merely early.** Per-layer (block-weighted) LoRA loaders for the S3-DiT and regional-prompting tools both appeared between June and September 2026, each from one or two authors and none validated beyond its own README `[flagged — early tooling; re-verify]`. Per-face FaceDetailer passes and dataset-time prevention stay the reliable paths (`references/characters.md §5–6`, `references/setup-and-workflows.md §6`).
- **Further open flags live in the references**, all greppable there: no validated map of which Z-Image blocks carry style versus identity (`references/setup-and-workflows.md §6`); the style-LoRA rank ceiling is agreed in direction but not in value (`references/lora-training.md §2.1`); and a single report of training a character LoRA on layers 14–25 only is unconfirmed (`references/lora-training.md §3.2`).

**Facts dated 2026-09-09.** What moves fastest in this family is the *plumbing*, not the craft: quant filenames, ComfyUI template contents, the AI-Toolkit Z-Image config, and the Fun ControlNet file names have all changed inside a release cycle before, and the regional-prompting and per-layer LoRA tools are weeks old. Re-verify those against their repos before relying on them, regardless of who said it. The pipeline shapes and prompting craft in the references age far more slowly.

---

## Reference files

| File | When to read it |
|---|---|
| `references/prompting-guide.md` | 6-part prompt anatomy in full detail; realism vocabulary; camera vocabulary (8-point rotation, shot sizes, high/low angle); lighting vocabulary; bilingual text rendering; common mistakes; drop-in templates |
| `references/setup-and-workflows.md` | Multi-stage ComfyUI pipelines: the minimal build, the layered ZIB+ZIT pipeline with per-stage settings, resolution table, universal node settings (ModelSamplingAuraFlow / lumina2 / EmptySD3LatentImage), optional layers (skin contrast, SeedVR2, tiled upscale); **§6: Using LoRAs** (node wiring + `LoraLoaderModelOnly`, the Nunchaku path, the new per-layer/block-weighted loaders, weight-by-type, rgthree stacking, ZIB↔ZIT cross-compat and the Turbo training adapter, "fights distillation," ecosystem, the character-via-detailer high-likeness method, and the QKV version-floor footnote); **§9: Fun Union ControlNet** (V2.1 files, ModelPatchLoader + QwenImageDiffsynthControlnet nodes, all conditioning types and preprocessors, Turbo-only caveat); **§10: face identity methods** (LoRA, inpaint, IP-Adapter status); **§11: Z-Image in mixed-model pipelines** (the ZIT-as-refiner role, decode-to-pixels handoff rule, refine denoise bands) |
| `references/lora-training.md` | **Making** a LoRA on Z-Image specifically (loading/using is setup-and-workflows.md §6; the model-agnostic training craft is [`character-lora-training`](../character-lora-training/)): the train-on-Base/generate-on-Turbo decision and why, what changes about dataset building because the encoder is Qwen-3, **style-LoRA craft** (diverse-subject rule, prose captions, color-cast lock-in, the out-of-set acceptance test, rank-by-type), the Turbo training-adapter requirement and the de-turbo alternative, Ostris AI-Toolkit hyperparameters with the two knobs that make them readable, the RunPod fast path, the bucket/noise-sigma findings, adult-work coverage, and Z-Image-specific evaluation and debugging |
| `references/characters.md` | Creating a **consistent original character** end-to-end: the two paths (edit-model engine vs LoRA pipeline) and how they chain, the anchor image, the **Qwen-Image-Edit dataset factory** (2511 file table + ComfyUI wiring; generate ~60 → curate ~30; the multiple-angles coverage LoRA; where explicit material must be generated natively instead), rotation/elevation/expression coverage, the detailer-deploy step, **multi-outfit LoRAs** (~6-outfit ceiling), **multi-character scenes** (per-face detailer passes as the reliable path; two experimental regional-prompting tools, zit-regions and ZIT-Ideogram), character failure modes (angle collapse, same-face overfit, expression lock-in, style bleed) |
