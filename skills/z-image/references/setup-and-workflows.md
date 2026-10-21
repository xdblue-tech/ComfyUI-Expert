# Z-Image Setup & Multi-Stage Workflows

This file shows you how to build layered rendering pipelines for production-quality results. The core pattern is to generate small, refine, and then upscale in passes, using **ZIB for structure and ZIT for detail**. There is also a reverse ladder for when you want it the other way round. The file also covers everything that hangs off the graph: node settings, **using and stacking LoRAs**, ControlNet, identity methods, and handoffs to other model families. Two things live elsewhere. The one-time install and file layout is in `SKILL.md § Setup & ecosystem`, which you only read once. **Making** a LoRA rather than loading one is in `references/lora-training.md`.

The settings here come from two sources. They are labelled because you cannot swap one for the other:
- **Official** — the stock ComfyUI Z-Image-Turbo template (`Comfy-Org/workflow_templates`).
- **Community** — the layered pipeline pattern popularised by widely-shared community ComfyUI workflows (e.g. the "Moody" ZIB+ZIT graphs on Civitai). These often run **custom community finetunes** of Z-Image. Treat their step/CFG numbers as well-tuned starting points, not stock requirements. The *architecture* transfers to stock Z-Image, but the exact numbers may need a nudge.

That split is why you should read this file in order rather than grep it for a number. A number's tier changes what you do when it doesn't reproduce.

## Contents

1. The minimal build (start here)
2. The layered pipeline (the full build)
3. Per-stage settings (community layered pipeline) — and the reverse ZIT→ZIB ladder
4. Resolution strategy — generate small, upscale in layers
5. Universal node settings
6. Using LoRAs — wiring (incl. the Nunchaku path), the QKV version floor, weights, per-layer loaders, stacking, ZIB↔ZIT cross-compat and the Turbo training adapter, the detailer "face-swap" method
7. Optional improvement layers — skin contrast, upscale models, SeedVR2, the tiled-upscale prompt caveat, colour drift
8. Build order summary
9. Fun Union ControlNet (pose, depth, canny, and more)
10. Face identity — available methods
11. Z-Image in mixed-model pipelines

---

## 1. The minimal build (start here)

A single-stage text-to-image graph — the official template:

```
UNETLoader (z_image_turbo_bf16) → ModelSamplingAuraFlow (shift 3) → KSampler
CLIPLoader (qwen_3_4b, type "lumina2") → CLIPTextEncode (positive) ─┘
EmptySD3LatentImage (1024×1024) ──────────────────────────────────┘
KSampler → VAEDecode → SaveImage   (VAELoader = ae.safetensors)
```

Official Turbo KSampler: **8 steps, CFG 1.0, sampler `res_multistep`, scheduler `simple`, denoise 1.0.** That is the whole baseline. Everything below is layered on top of it.

> **Latent node:** use **`EmptySD3LatentImage`** (the official template's choice), not the legacy `EmptyLatentImage`.

---

## 2. The layered pipeline (the full build)

Each stage takes the previous stage's output and improves one thing. Skip any optional stage you don't need.

| # | Stage | Model | Purpose |
|---|---|---|---|
| 1 | **1st-gen** (txt2img) | ZIB | Composition, pose, overall structure |
| 2 | **Latent upscale** | — | Raise resolution before detailing |
| 3 | **2nd-gen** (hires refine) | ZIT | Fine details: fingers, face, text |
| 4 | **Tiled SD upscale** | ZIT | Resolution + micro-detail, tile by tile |
| 5 | **Face/auto detailer** | ZIT | Re-render the face; swap in a character LoRA |
| 6 | **Manual inpaint** *(opt)* | ZIT | Fix a region the auto-detector misses |
| 7 | **Skin contrast** *(opt)* | — | Add skin micro-texture for realism |
| 8 | **SeedVR2** *(opt)* | SeedVR2 | Heavy final upscale (high VRAM) |

**Two-pass discipline.** This is the reason the layered build beats one big render:
- **1st gen — judge composition only.** Reroll the seed until the layout and pose are right. Minor defects are fine, because they get fixed downstream.
- **2nd gen — judge the fine details.** Look at fingers and toes, facial expression, rendered text, and small artefacts.

Preview after stage 1, and again after stage 3, *before* you commit to the expensive upscale stages.

---

## 3. Per-stage settings (community layered pipeline)

Verified from a representative community ZIB+ZIT graph. CFG values are **ComfyUI KSampler** values (see §5 on the CFG=1 convention).

| Stage | Sampler / scheduler | Steps | CFG | Denoise | Negatives | Notes |
|---|---|---|---|---|---|---|
| 1st-gen (ZIB) | `euler` / `simple` | ~17 | 3.0 | 1.0 | **real prompt** | Base dims (e.g. 640×960) |
| 2nd-gen (ZIT) | `dpmpp_2m_sde` / `sgm_uniform` | ~11 | 1.0 | partial (hires) | **zeroed** | Latent from stage 2 |
| Tiled SD upscale (ZIT) | `dpmpp_2m_sde` / `sgm_uniform` | 3 | 1.0 | **0.23** | zeroed | ~1024 tiles, ×2 factor |
| Face detailer (ZIT) | `dpmpp_2m_sde` / `sgm_uniform` | 4 | 1.0 | **0.42** | zeroed | yolov8m bbox + SAM |
| Manual inpaint (ZIT) | `dpmpp_2m_sde` / `sgm_uniform` | 4 | 1.0 | **0.39** | zeroed | mask → SEGS, dilate ~8 px |

**Detailer/refine denoise rule:** raise denoise up to ~**0.54** for more prompt- and LoRA-adherence; lower it to preserve the original. Keep the **tiled upscale denoise low (~0.2–0.25)**, because high denoise there invents new content per tile.

> Some community graphs handle the 1st-gen → 2nd-gen handoff with `return_with_leftover_noise` + `add_noise` across mismatched step schedules. That is one author's hires-fix trick, not a canonical requirement. A clean "decode → latent-upscale → re-noise → refine" chain works fine.

### The reverse ladder: ZIT draft → ZIB finish (Z-Image-SPECIFIC)

The ladder above runs base-first because the expensive high-resolution passes are exactly where you want the *cheap* model. Running it the other way round is also a live pattern: draft in **ZIT**, then push the keepers back through **Z-Image base** as the refinement pass. One low-VRAM report does this with the **ComfyUI Z-Image upscaling node template**. In practice that means pointing the template's Load Diffusion Model node at the base DiT instead of the Turbo one. The report describes the result as more detail and more realism `[community — Royal_Carpenter_1338]`.

The mechanism is the distillation trade run in reverse. Decoupled-DMD buys Turbo its 8 steps by collapsing toward the strongest mode. That is the same property the variant selector credits base with *not* having ("best texture diversity"). The undistilled weights still carry the full training signal, and they sample with real CFG and real negatives. So a **low-denoise** base pass over a Turbo render re-renders the skin, fabric and hair micro-texture that Turbo's short trajectory flattened. It leaves the composition Turbo already settled alone. This is the same bargain §11 strikes when ZIT finishes *another family's* render, only pointed inward.

Here is what it costs, and when that cost is worth paying:

- **Compute, in the place the base-first ladder deliberately avoids.** The base pass runs 25–50 steps against ZIT's 8, and it runs them at or near output resolution. Treat it as a finishing pass on selected frames. That is the same logic that makes stage 5 above a face crop rather than a full re-render.
- **Negatives come back on.** This is a real-CFG pass, so wire an actual negative prompt into it rather than carrying through the `ConditioningZeroOut` the Turbo stages use (§5). Forgetting is not an error, just a wasted pass.
- **It is kinder to a small card than it looks.** It is a single img2img pass, not a tiled sweep, so peak VRAM is one latent plus the model. That is how the report above runs it on hardware that could not host the full layered graph. Pair it with the int8 base DiT if the bf16 one will not fit.

Starting settings, derived from the official base template plus the refine-denoise rule above and **not** published by the report `[flagged — re-verify]`: `res_multistep` / `simple`, **25–30 steps**, **CFG 3.0–4.0**, **denoise 0.25–0.40**, `ModelSamplingAuraFlow` shift 3, a real negative prompt, same seed as the draft. Below ~0.2 the pass mostly adds noise texture. Above ~0.5 base begins re-composing and you lose the draft you were trying to keep.

---

## 4. Resolution strategy — generate small, upscale in layers

Z-Image works best generating at a **low base resolution** (~0.6 MP), then climbing in passes. Final resolution = **base × 1.7 (2nd-gen latent upscale) × 2 (SD upscale)**.

| Aspect | Use | Base (W×H) | Final (×1.7 ×2) |
|---|---|---|---|
| 2:3 | Portrait | 640 × 960 | 2176 × 3264 |
| 3:4 | Portrait (stable) | 672 × 896 | 2285 × 3046 |
| 9:16 | Tall / mobile | 576 × 1024 | 1958 × 3482 |
| 1:1 | Square | 768 × 768 | 2611 × 2611 |
| 4:3 | Landscape | 896 × 672 | 3046 × 2285 |
| 3:2 | Landscape | 960 × 640 | 3264 × 2176 |
| 16:9 | Widescreen | 1024 × 576 | 3482 × 1958 |

Latent upscale node: `LatentUpscaleBy`, method `bislerp`, factor ~1.7. SD upscale: `UltimateSDUpscale`, factor 2, with a dedicated upscale model (§7).

---

## 5. Universal node settings

- **ModelSamplingAuraFlow on every model path, `shift = 3`.** This is stock; the official template includes it. Apply it to the ZIB path, the ZIT path, and every detailer/upscaler model input.
- **CLIPLoader type `lumina2`** for `qwen_3_4b.safetensors`. Stock.
- **CFG = 1.0 is "guidance-off" in ComfyUI.** At cfg 1, the KSampler formula collapses to just the conditional. That is why Turbo runs at **cfg 1.0 in ComfyUI**, which is the same thing as `guidance_scale = 0` in diffusers. Do **not** type cfg 0.0 into a KSampler: that outputs the unconditional and ignores your prompt. Use cfg 1.0 for every ZIT pass, and cfg 3–5 only for ZIB passes that need true classifier-free guidance.
- **Negatives:** ZIB passes take a **real negative prompt**. ZIT passes wire the negative through `ConditioningZeroOut`, because turbo ignores negatives, so you zero them rather than feed text.
- **Seed:** keep the **same seed across 1st-gen, 2nd-gen, and SD upscale** for consistent results. Reset it to a fixed value (e.g. 0) once you've found a good composition. Only vary the later-stage seed if a stage needs a localised re-roll.

---

## 6. Using LoRAs (any LoRA — style, concept, character, not just ones you train)

This is the generic "I downloaded a LoRA, how do I run it" path. Training your own is in `references/lora-training.md`. The high-likeness character method is the subsection at the end.

> **Sourcing:** the node names and the QKV/PR facts below are hard facts, verified against ComfyUI PRs and issues. The weight numbers and stacking guidance are **community craft** from named Civitai/HF authors who've run thousands of generations. They are given as ranges, and the fast-moving ones are flagged. Read a LoRA's own model card for the weight its author tested.

### Node wiring

Z-Image is loaded as a diffusion model, not a checkpoint. A Z-Image LoRA patches the **DiT** (attention + MLP). Load it with a LoRA loader on the model path, right after the model loader and before `ModelSamplingAuraFlow`:

```
Load Diffusion Model → LoRA loader → ModelSamplingAuraFlow (shift 3) → KSampler
CLIPLoader (qwen_3_4b, "lumina2") ─────────────────────────────────┘
```

**Which loader:** the ComfyUI core PR #12717 repro adds the LoRA with the **full `LoraLoader`** (model + clip). **`LoraLoaderModelOnly`** also works and is the clean choice. Z-Image LoRAs target the DiT, and most (Ostris-trained, diffusers-format) carry **no text-encoder weights**, so the Qwen-3/CLIP side is usually a no-op either way. The DiT target is verified `[official — ComfyUI PR #12717]`. Which of the two loader nodes you pick is a usage detail, not a break-or-not fact. Don't agonize over it. If a LoRA *does* ship Qwen-3 keys, use the full `LoraLoader` so they apply.

**If the DiT is a Nunchaku SVDQuant build** (`SKILL.md § Setup & ecosystem`), the model path changes. `ComfyUI-nunchaku` loads the 4-bit weights with its own loader node and applies LoRAs through its own LoRA nodes, because the stock loaders cannot patch that format. Wire your LoRA into the Nunchaku LoRA node instead of `LoraLoader`; everything downstream (`ModelSamplingAuraFlow`, the sampler settings) is unchanged `[community — nunchaku-ai; strong]`.

### The QKV version floor (Z-Image-SPECIFIC, now a footnote)

Most published Z-Image LoRAs ship in **diffusers format** with separate `to_q` / `to_k` / `to_v` attention keys, while Z-Image stores attention as one **fused QKV** matrix. ComfyUI builds older than **2026-03-17** silently dropped those attention deltas: `lora key not loaded` warnings, and a LoRA that "barely does anything" because only its MLP/FFN weights applied. Core PR #12717 ("fix: Z-Image LoRA and model loading for HuggingFace format weights", issue #10973) fixed it, and every stable release since carries the fix `[official — ComfyUI PR #12717, merged 2026-03-17]`. On a current install this is not a suspect. It only matters on a pinned or frozen ComfyUI, such as a RunPod template image that has not been rebuilt, and there the fix is to update, not to raise the weight. (`Comfyui-ZiT-Lora-loader` began as a workaround for this bug and is now the per-layer loader described below.)

### Weight — there is no magic 0.8 cap

Start **~0.7–0.8** and sweep **0.5–1.2**. Named authors land at very different places, depending on LoRA type:

| LoRA type | Typical weight | Why |
|---|---|---|
| **Style / film** | often **0.3–0.5** | full strength over-smooths skin and crushes the base look |
| **Realism enhancer** | **~0.6–0.7** | potent; start low and raise |
| **Character / concept** | **~0.7–1.0** | needs more strength to carry identity |

These are **per-LoRA tunings, not a model-wide ceiling.** The old "1.0 overcooks Turbo" framing was overstated. There is no documented hard cap. A saturated result means *that* LoRA is hot, not that 1.0 is illegal. The exact numbers are fast-moving and sources mildly disagree, so treat the table as starting points and prefer the weight printed on the LoRA's own model card `[community — single report; re-verify]`.

### Per-layer (block-weighted) loading — new and experimental (Z-Image-SPECIFIC)

SDXL users reach for block-weighted loading when a LoRA drags a style or colour cast along with the thing it was trained for: lower the blocks carrying the cast and keep the rest. Until mid-2026 Z-Image had no equivalent, so the only lever was the dataset (`characters.md §6`). Two community loaders now expose that knob on the 30-block S3-DiT:

- **Comfyui-ZiT-Lora-loader** (capitan01R, per-layer graph added 2026-09-01). An interactive per-layer strength graph, 30 columns with attention and feed-forward split, plus an "Auto Strength" node that sets each layer's strength from its Frobenius norm. It also fuses separate Q/K/V keys into Z-Image's fused QKV on load, which is the conversion job it started life doing `[community — capitan01R; single report, early]`.
- **ComfyUI-Realtime-Lora** (shootthesound, 2026-08-01). A "Z-Image Analyzer + Selective Loader V2" that analyses a LoRA per block and applies it selectively. The same pack does this for Flux, SDXL and Wan 2.2, so the idea is spreading across families rather than staying a one-off `[community — shootthesound; early]`.

What neither gives you is a **map**. Nobody has published a validated account of which Z-Image blocks carry style and which carry identity, so plan to sweep. The Auto Strength heuristic and one training-side report of restricting a character LoRA to layers 14–25 (`lora-training.md §3.2`) are the only starting points, and both are single reports `[flagged — no validated DiT block map; re-verify]`. Keep dataset-time prevention as the default. Use the loaders to rescue a LoRA you cannot retrain, and re-verify both before building a pipeline on them.

### Stacking multiple LoRAs (general ComfyUI craft — stable across models)

Use the rgthree **Power Lora Loader** node. It takes multiple LoRAs in one node, with per-LoRA strength, on/off toggles, and "no real limit" (`FlexibleOptionalInputType`). Toggle to **separate model/clip strengths** via the advanced view if needed. On Turbo, **keep combined strength near or under ~1.0** to avoid burning/overexposure. That is a conservative heuristic; with normalization you can go higher. A Z-Image-specific LoRA-merger node (`ComfyUI-ZImage-LoRA-Merger`) exists precisely because chained strengths accumulate on distilled models. Ordering has minor effects `[community — rgthree README]`.

### ZIB ↔ ZIT cross-compatibility: loads fine, doesn't transfer cleanly (Z-Image-SPECIFIC)

Base and Turbo share the **identical S3-DiT**, so any LoRA **loads on either without a format error.** But "loads" does not mean "transfers." A **Base-trained LoRA run on Turbo** shows **softer identity, dropped face consistency, shifted color/background**, and may need a strength bump.

**Community best practice (incl. the official Tongyi-MAI HF discussion #18): train on Z-Image Base, generate on Z-Image-Turbo.** Base is the better base for cross-prompt control. Train *on* Turbo only if you specifically want fast-delivery behavior. If you do, train through Ostris's **`zimage_turbo_training_adapter`** (v2). It de-distills Turbo *during training only*, so the LoRA learns against a normal diffusion target, and it is dropped at inference, so the result still runs the 8-step path. That is the named alternative to train-on-Base for people who need Turbo speed from the first sample. Its own write-up still calls Base training the straightforward choice `[official — Ostris adapter repo; lilting.ch 2026-05-04]`. Mechanics in `references/lora-training.md §3.1`. The exact magnitude of Base→Turbo degradation is **genuinely contested** `[contested]` across sources. Reports range from "~100% similarity" to "little impact" `[community — Tongyi-MAI #18, RunComfy, lilting.ch]`. So **test it, don't assume.**

### "Fights distillation" — why a Turbo-trained LoRA can look blurry (Z-Image-SPECIFIC)

A LoRA trained **directly on Turbo** can disturb Turbo's few-step "landing trajectory." It alters not just the subject and style but how the model converges in 8 steps, producing **blurry output at 8 steps that only cleans up at ~30.** This is the other reason the train-on-Base path is preferred. It is also why inference-time **DistillPatch** correction LoRAs exist (DiffSynth-Studio `Z-Image-Turbo-DistillPatch`; a Civitai equivalent). If a Turbo LoRA renders soft at 8 steps, suspect this before blaming your prompt `[community — lilting.ch, DiffSynth-Studio]`.

### Trigger words

Z-Image's encoder is the Qwen-3 LLM, not a CLIP tag-matcher. So a trigger generally works best **folded into natural-language description** rather than dropped as a bare tag. This matches what FLUX.2 trainers report for *its* LLM encoder (Mistral/Qwen3): bare trigger words "confuse the model," and descriptive captions activate a LoRA better. So it is a consistent LLM-encoder pattern, not a Z-Image quirk. Still, follow the LoRA author's stated trigger and include it if they define one. The precise token-weighting behavior isn't formally documented for Z-Image, so treat heavy trigger-engineering as lower-yield than on tag-based SDXL `[community — cross-model pattern; re-verify]`.

### Ecosystem (early–mid 2026)

An active Civitai/HF ecosystem tags its uploads **`ZImageTurbo`**: **style** (e.g. "Technically Color Z," trigger `t3chnic4lly`), **realism enhancers** (e.g. "Realistic Snapshot"), and **character/concept** LoRAs. Most are trained with the **Ostris AI-Toolkit**, which ships a dedicated Z-Image-Turbo training adapter. That is why so many ship in diffusers format, which ties straight back to the QKV gotcha above. Community "apply any LoRA" ComfyUI workflows exist (Civitai 2194203, Next Diffusion, RunComfy) `[community — Civitai, Ostris HF]`.

**The published ecosystem is lopsidedly Turbo.** A Civitai census on 2026-08-23 counted **2,191+ LoRAs tagged `ZImageTurbo` against 671 tagged `ZImageBase`** `[community — Civitai census, Aug 2026]` — better than 3 to 1. That is the doctrine above surfacing in the tags, not contradicting it: authors train on base and *deploy* on Turbo. What they tag is the variant they expect you to generate with, so a tag records intended runtime rather than training variant. Two things follow. First, the cross-compat question above is not an edge case for anyone loading a downloaded LoRA; it is the default case. So read the author's card for what it was trained on before reaching for a strength number. Second, a card that says only `ZImageTurbo` leaves open whether it was in fact trained there. If it renders soft at 8 steps, the distillation-trajectory failure above is the first suspect, ahead of your prompt.

---

### High-likeness character LoRAs (the detailer "face-swap" method)

Loading a character LoRA into the *base* generation often gives mediocre results. Here is the community-proven pattern instead:

1. **Generate the base WITHOUT the character LoRA.** Describe the character in as much detail as you can in the positive prompt.
2. **Swap the character LoRA in at the FaceDetailer stage** (stage 5). It re-renders just the face with the LoRA applied.
3. **Match the detailer's prompt to the rest of the image.** If you leave a generic prompt, you get the LoRA's built-in default face/scene instead of your subject.

**Where to place a LoRA — ZIB vs ZIT:**
- A LoRA on the **ZIB** pass affects **image structure** more strongly.
- A LoRA on the **ZIT** pass leans toward **detail/finish**.
- For **maximum likeness**, load the same character LoRA on **both** ZIB and ZIT passes.
- LoRAs load on either variant but transfer best from a Base-trained source (see cross-compat above), so **a Base-trained character LoRA is the safest choice** for either slot. Use the `Power Lora Loader` (rgthree) to stack/toggle. (See `references/lora-training.md` for training and `## 6` above for load-weight guidance.)

---

## 7. Optional improvement layers

**Skin contrast (realism).** After detailing, blend the image with a skin-detail upscale model pass:
- Model: a `1x` skin-contrast ESRGAN model (e.g. `1xSkinContrast-High-SuperUltraCompact`), run via `ImageUpscaleWithModel`.
- Blend back over the original with `ImageBlend`, mode `normal`, intensity ~**0.4** (1.0 = full strength, 0 = off).
- Best for Western / photorealistic styles; skip for stylised or everyday work.

**Upscale models** (for the tiled SD upscale, drop in `models/upscale_models/`):
- `4xNomosWebPhoto_RealPLKSR` — balanced, good default.
- `4xNomos8k_atd_jpg` — high quality, slower.
- `4xUltraSharp` — sharper, more aggressive.

**SeedVR2 final upscale** (optional, heavy). This is a separate diffusion upscaler (`SeedVR2` nodes, e.g. the 7B or 3B GGUF) targeting ~4096 px. The first run downloads large models, so it is not recommended on low-VRAM GPUs. Use 3B if the 7B OOMs.

**Tiled-upscale prompt caveat.** When `UltimateSDUpscale` splits the image into tiles, a tile only "sees" its local patch. So a prompt that says "a tattoo reading 'X' below the collarbone" can make the model stamp that text onto unrelated smooth-skin tiles (shoulder, arm, back). If you render localised text or marks, give the upscale pass a **simpler prompt** (or the conditioning switch many graphs expose) so per-tile hallucinations don't reproduce it.

**Color drift across passes.** Every VAE decode/encode round-trip and every re-sample pass shifts color slightly, and the shifts compound through a long pipeline. Fix it once at the end, not per stage. Use a **ColorMatch node (KJNodes**; `mkl` or `hm-mvgd-hm` method) to compare the final image against the stage-1/stage-3 composition reference. It pulls the grade back without touching detail `[community]`. The full colour-management treatment is in [`image-production-workflows`](../../image-production-workflows/).

---

## 8. Build order summary

1. Stand up the **minimal build** (§1) and confirm models load.
2. Add **ModelSamplingAuraFlow shift 3** and the **ZIB→latent-upscale→ZIT** two-pass (§2–4).
3. Add the **FaceDetailer** stage; wire the character LoRA here (§6).
4. Add **UltimateSDUpscale** with a 4x model (§4, §7).
5. Layer in **skin contrast / SeedVR2 / manual inpaint** only as needed (§7).

Skip freely: the optional stages are toggleable, so you can preview cheaply and only pay for the heavy passes once the base is right.

---

## 9. Fun Union ControlNet (pose, depth, canny, and more)

Source: `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` on Hugging Face (official), plus the `Comfy-Org/workflow_templates/image_z_image_turbo_fun_union_controlnet.json` template (verified from raw JSON).

**Critical: Z-Image Turbo only.** The Fun Union ControlNet is distilled for Z-Image-Turbo. The undistilled Z-Image base has its own patch files (`alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1`), but ComfyUI support for the base variant is an open feature request with no current resolution (issue #12243). Use the Turbo + ControlNet combination for all structure-guided work.

### Model files

All files install to `models/model_patches/` (not `models/controlnet/`).

> **Verify the exact filename against the `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` HF repo before downloading.** These are third-party files on a fast-moving repo. The names below are a snapshot, and a wrong one just 404s.

**Recommended file (V2.1, 8-step optimised):**

| File | Size | Conditions | Notes |
|---|---|---|---|
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors` | 6.71 GB | Canny, Depth, Pose, MLSD, HED, Scribble | Tuned for 8-step Turbo — use this |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors` | 6.71 GB | Canny, Depth, Pose, MLSD, HED, Scribble | Timestep 2601 variant |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors` | 6.71 GB | Adds Gray | Timestep 2602 variant |
| `Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors` | 2.02 GB | Same types, weaker control | 3 blocks (vs 6 full); use on low VRAM |

For tiled upscaling with ControlNet: `Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors` (6.71 GB) — same repo, same install path.

### ComfyUI nodes (both built-in core, no custom install needed)

| Node | Role |
|---|---|
| `ModelPatchLoader` | Loads `.safetensors` from `models/model_patches/` |
| `QwenImageDiffsynthControlnet` | Applies the patch to the model; key inputs: `model`, `model_patch`, `vae`, `image` (preprocessed), `strength` (default 1.0), optional `mask` |

These are standard ComfyUI core nodes added in PR #11062 (ComfyUI v0.3.51+). The `QwenImageDiffsynthControlnet` name is misleading: it handles both [Qwen-Image](../../qwen-image/) and Z-Image patches.

**Recommended `control_context_scale`:** 0.65–0.80 `[community]` (the parameter on `QwenImageDiffsynthControlnet`). This is a community band, not a documented default; the official template ships `strength=1.0` and leaves this alone `[official — Comfy-Org template JSON]`.

### Conditioning types and preprocessors

The conditioning type is determined by which preprocessor feeds the image — there is no explicit "mode" switch on the apply node. One model handles all types (Union architecture).

| Type | Preprocessor node | Notes |
|---|---|---|
| **Pose** | `DWPreprocessor` (DWPose) | Generates skeleton with body + face + hand keypoints |
| Depth | `DepthAnythingV2Preprocessor` | Depth Anything V2 for real-photo depth |
| Canny | `CannyEdgePreprocessor` | Default thresholds 0.1–0.32 (from template) |
| HED | `HEDPreprocessor` | Soft edges; better than Canny for complex outlines |
| Scribble | `ScribblePreprocessor` or hand-drawn | Rough structural constraint |
| MLSD | `MLSDPreprocessor` | Line segments; good for architectural/interior |

All preprocessors are available via `comfyui_controlnet_aux` (install via ComfyUI Manager).

### Sampler settings with ControlNet

Keep all standard Z-Image-Turbo settings. The ControlNet is additive, and it does not change the base sampler:
- **Steps:** 8, **CFG:** 1.0, **Sampler:** `res_multistep`, **Scheduler:** `simple`
- **ModelSamplingAuraFlow shift 3** — required on the ControlNet-patched model path as well
- Negative conditioning: wire through `ConditioningZeroOut` as usual

### Official template

The Comfy-Org template (`image_z_image_turbo_fun_union_controlnet.json`) defaults to Canny edge detection. To use it with pose: swap the `Canny` node for `DWPreprocessor`, then connect the skeleton image output to `QwenImageDiffsynthControlnet`'s `image` input. The rest of the graph is unchanged.

---

## 10. Face identity — available methods

No PuLID or IP-Adapter face implementation exists for Z-Image as of June 2026. The following methods are in order of reliability:

| Method | When to use | Notes |
|---|---|---|
| **Subject LoRA** (via FaceDetailer) | Recurring character, product persona | Most reliable; train on 20–50 images; load at stage 5 (§6 above) |
| **Inpaint + SAM mask** | Swap or refine the face region of an existing image | Use `SAM3Grounding` → `InpaintModelConditioning` + `DifferentialDiffusion` |
| **Z-Image-Edit** | Instruction-based editing of an existing image | Announced; verify availability at time of use |
| **Community IP-Adapter (Boyonodes)** | Experimental generic image conditioning | No published weights; requires manual SD3 weight conversion; not recommended for production |

**The practical identity path for Z-Image is: train a character LoRA and inject it at the FaceDetailer stage (§6).** This gives consistent identity and composites cleanly with the multi-stage pipeline. The full character pipeline — anchor image, edit-model dataset factory, coverage rules, multi-character craft — is in **`references/characters.md`**.

---

## 11. Z-Image in mixed-model pipelines

Z-Image-Turbo has earned a specific role in cross-model production: **the realism refiner for other models' renders.** ZIT is fast (8 steps), guidance-free, and has a strong photographic prior. Because of that, named community workflows compose in another model and finish in ZIT. An example is Cordina's "ZIT Refiner workflows – SDXL v1" `[community — Cordina, Civitai Jan 2026]`: SDXL base gen → ZIT img2img pass "to add realism" → detailers → upscale.

If you build a handoff yourself, here is the mechanics:
- **Always decode to pixels between model families.** Z-Image's latent space is not SDXL's or FLUX's: `VAE Decode` (model A) → image → `VAE Encode` (Z-Image's `ae.safetensors`) → ZIT KSampler. Feeding a foreign latent straight in produces garbage.
- **Refine denoise 0.2–0.5:** ~0.25–0.35 preserves the source composition and identity while re-rendering texture. Above ~0.5 ZIT starts re-composing; below ~0.2 it only adds noise-pattern detail.
- **Match resolution** to Z-Image's comfortable range before encoding (downscale a 4 MP source or tile it).
- The reverse direction also exists: ZIB/ZIT base → an SDXL photoreal finetune img2img for its texture character, or → SDXL for its mature ControlNet/IP-Adapter ecosystem when a control type Z-Image lacks is required.

The model-agnostic craft — the full production ladder, cross-family handoff rules, color management, workflows-as-code (ComfyScript, comfy-cli, the `/prompt` API) — lives in the [`image-production-workflows`](../../image-production-workflows/) skill in this suite.
