---
name: qwen-image
description: >
  Authoritative guide for the Qwen-Image family (Alibaba Qwen team) across both open generations and the hosted
  2.0/3.0 surface: the Apache-2.0 20B generation — Qwen-Image and Qwen-Image-2512 for text-to-image,
  Qwen-Image-Edit / Edit-2509 / Edit-2511 for instruction and multi-image editing, Qwen-Image-Layered for RGBA
  layer decomposition, the Lightning speed LoRAs — and Qwen-Image-2.1 (open weights 2026-09-20), the 7B
  single-stream model that does T2I and editing in one checkpoint, generates and edits transparent RGBA images
  natively, takes up to 10 reference images tagged <image1>…<image10>, accepts circles, painted regions or a
  separate mask for local edits, runs 40 steps with CFG off, and ships under the non-commercial Qwen Research
  License. Use this whenever the user touches Qwen-Image in any way, even obliquely: choosing a variant
  (T2I vs Edit generation, which Edit for multi-image or character work, 2.1 vs the 20B family, why 2.0 and
  3.0 are API-only, why 2.1 fails the commercial-licence gate the 20B models pass),
  installing it in ComfyUI (Comfy-Org filenames, CLIPLoader type "qwen_image", ModelSamplingAuraFlow shift
  3.1 vs 3.0 vs 1.0, the switch-node templates, fp8_e4m3fn vs fp8mixed vs int8_convrot vs GGUF vs Nunchaku,
  VRAM from 6 GB up; for 2.1 the TextEncodeQwenImage21 and QwenImage21Cache nodes, the qwen3vl_8b encoder
  files, the pixel-budget "resolution" field and the three 2.1 templates), setting CFG and steps correctly
  (true CFG 2.5–4.0 at 20–50 steps on the 20B base, exactly 1.0 at 4–8 steps with Lightning, 40 steps at CFG
  1.0 on 2.1, and why negatives die at CFG 1), writing or fixing prompts (the Qwen2.5-VL encoder wants
  sentences; the "Picture 1: … Picture 2: …" multi-image format on the 20B Edits and <image1> tags on 2.1;
  the RGBA prompt wrapper; the "Relight to warm backlit" keyword grammar; naming what must not change; quoted
  text rendering in English and Chinese; the official PE-T2I / PE-I2I rewriters), fixing the
  "Qwen Edit is blurry" complaint (TextEncodeQwenImageEditPlus force-downscales to 1 MP with AREA
  resampling — bypass it with VAE-encoded ReferenceLatent), the VAE halftone grid, plastic skin and the
  realism LoRA stack, using ControlNet (native 2509, InstantX Union, DiffSynth model patches), calling the
  DashScope / fal / Replicate endpoints or the ComfyUI API nodes for Qwen-Image 3.0 Pro, running it in
  diffusers (QwenImagePipeline, QwenImageEditPlusPipeline, true_cfg_scale), training a LoRA (ai-toolkit,
  musubi-tuner, DiffSynth; train on the undistilled base and stack Lightning at inference; paired-data Edit
  LoRAs; DiffSynth as the only 2.1 trainer, with RGBA datasets), building a consistent character with Edit-2511
  as the identity tool where no PuLID or InstantID exists, or checking the licence (Apache-2.0 on the 20B
  family, Qwen Research License on 2.1). Use this for any question about Qwen-Image in any context.
  Choosing between models or working out which skills a job needs is
  [`generative-media-atlas`](../generative-media-atlas/)'s job — start there when the model is not settled.
---

# Qwen-Image family

Qwen-Image is the Qwen team's (Alibaba) open-weights image family, and as of **2026-09-20 it is two generations that share a name and almost nothing else.**

**The 20B generation** (2025-08 → 2025-12). Every variant is the same **20B-parameter dual-stream MMDiT** (60 layers, 16-channel latent), conditioned by **Qwen2.5-VL 7B**. That vision-language model is the text encoder, and in the Edit variants it reads the input image as well. The VAE is a **Wan-2.1-family** autoencoder: hyperparameter-identical to Wan 2.1's, but its own class and weights, shipped as `qwen_image_vae.safetensors`. The family is bilingual (English and Chinese), specifically for rendering text. **Code and weights are Apache 2.0.** Six open variants shipped between **2025-08-04** (Qwen-Image) and **2025-12-31** (Qwen-Image-2512). **Qwen-Image 2.0 (2026-02) and 3.0 (2026-07/08) are hosted-only**: no weights, an API node in ComfyUI, nothing to download.

**Qwen-Image-2.1** (open weights **2026-09-20**). A **7B single-stream DiT** (32 layers, block-causal attention), conditioned by **Qwen3-VL-8B**, with a **new 64-channel RGBA VAE at 16× compression**. One checkpoint does text-to-image *and* instruction editing; it generates and edits transparent images natively, takes up to **10 reference images**, and accepts circles, painted regions or a separate mask to localise an edit. It absorbs what Qwen-Image-Layered did. It is the first open 7B in the line (2.0's 7B never shipped as weights). **Its licence is the Qwen Research License: non-commercial, research or evaluation only**, with a separate commercial licence on request. Nothing from the 20B shelf — LoRAs, Lightning, ControlNets, GGUF loaders, Nunchaku — carries over.

**The defining trait of both generations:** the text encoder reads pictures. The same VL model that parses your prompt parses your reference images. That is why Qwen-Image-Edit is the suite's identity-without-training tool, and why in-image text survives an edit. The cost, on the 20B family, is a realism default the community calls plastic; whether 2.1 pays the same tax is unmeasured.

> **A `../link/` on this page that doesn't resolve is a skill you have not installed, not a broken
> page.** [`generative-media-atlas`](../generative-media-atlas/) is the map of this suite: which
> model fits a job, which skills that job needs, and the commands to install them. It works on its
> own, so it is the one to add first — `npx skills add ryannel/skills --skill generative-media-atlas`

---

## Variant selector

The first axis is **licence**: 2.1 is research-only, the 20B family is Apache 2.0. Then, on the 20B side, **generate or edit** and **which Edit generation**. Every 20B variant shares the encoder and the VAE, so the download is one transformer file per variant; 2.1 needs its own encoder and VAE.

| Variant | Released | What it is | Template base steps / CFG | Inputs | Use when… |
|---|---|---|---|---|---|
| **Qwen-Image** | 2025-08-04 | T2I foundation; the family root | 20 / 4.0 (card: 50 / 4.0) | text | the LoRA training base; stylised work; most community craft was written here |
| **Qwen-Image-2512** | 2025-12-31 | T2I refresh: human realism, natural detail, text | 50 / 4.0 | text | photoreal T2I; ships a real negative prompt; own Lightning LoRA and own LoRAs |
| **Qwen-Image-Edit** (2508) | 2025-08-18 | single-image instruction edit; VL + VAE dual conditioning | 20 / 2.5 | 1 image | legacy — 2509 supersedes it |
| **Qwen-Image-Edit-2509** | 2025-09-22 | multi-image (1–3), native ControlNet, person/product/text consistency | 20 / 4.0 (card: 40) | 1–3 images | **the most-pulled Edit**; widest LoRA shelf; the dataset factory |
| **Qwen-Image-Edit-2511** | 2025-12-23 | drift mitigation, character consistency, multi-person fusion, popular LoRAs folded in | 40 / 4.0 | 1–3 images | **the identity tool**; new shots of a character; needs `-2511` LoRA retrains |
| **Qwen-Image-Layered** | 2025-12-19 | decomposes an image into N RGBA layers; own VAE file | 20 / 2.5, shift 1.0 | image or text | layer separation for compositing; no settings-level craft exists yet `[flagged — gap; re-verify]` |
| **Qwen-Image-Lightning** LoRAs | 2025-08 → 2026-01 | 4/8-step distillation LoRAs, one per base variant (lightx2v, not Qwen) | 4 or 8 / **1.0** | — | iteration and dataset generation; stacked on any variant above at inference |
| **Qwen-Image-2.1** | **2026-09-20** | 7B single-stream, unified T2I + edit, native RGBA, native 2K, mask-guided local edits; own encoder (Qwen3-VL-8B) and VAE | **40 / 1.0** (template 25 / 1.0), CFG off | text, 0–10 images (+ a mask image) | research and evaluation; transparency; many-reference composition; mask-guided edits. **Non-commercial licence**; no LoRA shelf, no Lightning, no quant ladder yet |
| **Qwen-Image 2.0 / 3.0 Pro** | 2026-02 / 2026-08 | hosted only — native 2K, prompt rewriting on by default | managed | text, 1–2 images | when the newest model is worth an API key; nothing to install |

**Default workflow (commercial or LoRA-dependent work):** compose in **Qwen-Image or 2512** on the base regime (20–50 steps, CFG 4.0), or with Lightning for drafts. Then hand the still to **Edit-2511** for every subsequent shot of the same subject. Re-anchor on the original each time rather than editing the edit. Edit-2509 is the safer target if you depend on the community LoRA shelf, because 2511 broke LoRA compatibility when it absorbed popular LoRAs into the base.

**When 2.1 is the right tool:** the job is research or evaluation, and it needs one of the things only 2.1 does — transparent output without a separate Layered model, more than three references, a mask-guided local edit, or a 7B footprint. Its craft layer is **empty** as of 2026-09-20 (weights are hours old); everything below about 2.1 is official documentation plus the ComfyUI implementation, not practice.

---

## The one rule that changes everything

**Every Qwen-Image variant is a true-CFG model. The 20B family runs in exactly two regimes, and you never mix them; 2.1 adds a third that looks like Lightning but isn't.** The transformer config of all six 20B variants carries `guidance_embeds: false`. Nothing was distilled into the weights, so guidance is computed at sampling time from a real unconditional branch. That one fact decides the numbers you type:

| Regime | Steps | CFG | Negative prompt | Sampler / shift |
|---|---|---|---|---|
| **Base** (no Lightning) | 20 → 40 → 50 by release in the templates; 40–50 on the cards | **2.5–4.0** (`true_cfg_scale` in diffusers) | live, real | `euler` / `simple`, shift 3.0–3.1 |
| **Lightning** (distillation LoRA at strength 1) | **4 or 8** (2 with the third-party Wuli LoRA) | **exactly 1.0** | inert — the sampler never computes the unconditional branch | unchanged |
| **2.1** (undistilled, guidance-free by design) | **40** (card and diffusers default; Comfy template 25, "40–50 official") | **1.0 by default**; raise it only with a negative, which doubles per-step cost | inert at 1.0; live above it | `euler` / `simple`; shift is dynamic and baked into the model (0.69 at 1 MP), no shift node |

The mechanism is arithmetic, not folklore. At CFG 1.0 a KSampler returns the conditional prediction alone, so a negative prompt has nothing to subtract from. Negatives die in the Lightning regime because you turned guidance off, not because Qwen2.5-VL ignores them. 2.1 ships with guidance off and 40 undistilled steps: it is not distilled, it is simply trained to sample well without an unconditional branch, so the "4 steps at CFG 1" habit from Lightning does not transfer and the "CFG 4 with a negative" habit from the 20B base costs you double the compute for an unmeasured gain. [`z-image`](../z-image/) uses the same encoder, and its Turbo behaves the same way for the same reason. Reason from the guidance state, never from the encoder.

Three things follow:

- **The cross is the commonest misconfiguration.** Lightning at CFG 4 burns; base at 4 steps and CFG 1 is mush. The stock templates guard this with `ComfySwitchNode`s driven by one `PrimitiveBoolean` ("Enable Lightning LoRA"): flip it, and steps, CFG and the LoRA switch together. **Do not edit the KSampler's own widgets.** They are stale display values; the primitives behind the switches are what runs.
- **In diffusers, `true_cfg_scale` is the CFG knob.** The 2509 and 2511 cards pass `true_cfg_scale=4.0` and `guidance_scale=1.0` side by side; with no guidance embedding, the second is inert. Every card passes `negative_prompt=" "` (a single space) except 2512, which ships a real Chinese negative (`references/prompting-guide.md §4`).
- **The stock base template pairs a downcast fp8 with a bf16-trained Lightning LoRA, and the Lightning maintainers mark that pairing as broken.** `qwen_image_fp8_e4m3fn.safetensors` was made by direct downcast with no calibrated scaling; a bf16-trained Lightning LoRA on it produces a **grid pattern**. The fixes are the scaled fp8 base from the Lightning repo, the Lightning LoRA distilled on the fp8 base, or bf16 `[official — ModelTC/Qwen-Image-Lightning README, issue #32]`. Whether ComfyUI has since patched fp8 handling is unstated anywhere `[flagged — re-verify]`. The same interaction applies to your own LoRA on that file.

**The other rule, for Edit only:** ComfyUI's `TextEncodeQwenImageEditPlus` force-downscales every reference to 1 MP with AREA resampling before the model sees it. That is where "Qwen Edit is blurry" comes from. It is a workflow bug, not a model property. The bypass is in *The Edit reference path* below.

---

## Setup & ecosystem

Qwen-Image runs in **ComfyUI core** with no custom nodes (2511 needs ComfyUI ≥ 0.6.0; **2.1 needs a build newer than v0.36.0**, merged 2026-09-19 `[flagged — check for the tagged release]`). The DiT is not a checkpoint: three loaders, plus an optional LoRA loader.

### File layout

Repackaged by Comfy-Org (`Comfy-Org/Qwen-Image_ComfyUI`, `…/Qwen-Image-Edit_ComfyUI`, `…/Qwen-Image-Layered_ComfyUI`); names verbatim from the HF trees, 2026-09-09. **`CLIPLoader`, type `qwen_image`** ← `qwen_2.5_vl_7b_fp8_scaled.safetensors` (9.38 GB, `text_encoders/`, shared by every variant). **`VAELoader`** ← `qwen_image_vae.safetensors` (0.25 GB, shared; Layered needs its own `qwen_image_layered_vae`). **`UNETLoader`** ← one transformer per variant in `diffusion_models/`: `qwen_image_fp8_e4m3fn`, `qwen_image_2512_fp8_e4m3fn`, `qwen_image_edit_2509_fp8_e4m3fn` (20.43 GB each), `qwen_image_edit_2511_fp8mixed` (20.53 GB — **no plain e4m3fn build exists for 2511**), or `qwen_image_layered_bf16` (40.86 GB). **`LoraLoaderModelOnly`** at strength 1 for the Lightning LoRA of the *exact* variant. The table with folders and sizes: `references/setup-and-workflows.md §1`.

The Lightning LoRAs live in **three `lightx2v` repos** (base and 2509; Edit-2511; 2512), and a version mismatch either no-ops or produces garbage. Exact filenames per template: `references/setup-and-workflows.md §1`.

**2.1 is a separate loader set.** `Comfy-Org/Qwen-Image-2.1`, names verbatim 2026-09-20: **`UNETLoader`** ← `qwen_image_2.1_int8_convrot` (7.26 GB, the template default) or `qwen_image_2.1_bf16` (14.23 GB); **`CLIPLoader`, type `qwen_image`** ← `qwen3vl_8b_int8_convrot` (9.35 GB, template default), `qwen3vl_8b_bf16` (17.53 GB) or `qwen3vl_8b_w4a8` (6.31 GB); **`VAELoader`** ← `qwen_image_2.1_vae_bf16` (0.68 GB). None of the 20B files load into it and none of its files load into a 20B graph. Two new nodes replace the Edit text encoders: **`TextEncodeQwenImage21`** (prompt, negative, optional VAE, a `resolution` pixel budget, and autogrow `image_1`…`image_16` slots; outputs positive, negative, and an empty latent sized to `image_1`), and **`QwenImage21Cache`** (device auto/gpu/cpu/off, dtype default/int8/int4) for the prefix KV cache. There is **no `ModelSamplingAuraFlow`** in a 2.1 graph; shift is in the model config. Templates: `image_qwen_image_2_1_t2i`, `image_qwen_image_2_1_image_edit`, `image_qwen_image_2_1_background_removal`, all `euler` / `simple` / 25 steps / CFG 1 / denoise 1. Node semantics and the sizing rules: `references/setup-and-workflows.md §11`.

### Stock node settings — the anchor numbers

Every Qwen template samples with **`euler` / `simple` / denoise 1.0**, and steps and CFG are `PrimitiveInt` / `PrimitiveFloat` nodes routed through switches, so the KSampler widgets are stale. Three numbers move between templates, and readers carry the wrong one over. **Shift** (`ModelSamplingAuraFlow`) is **3.1** for T2I, 2511 and 2512, **3.0** for 2508 and 2509, **1.0** for Layered. **Base steps climb** 20 (T2I, 2508, 2509) → **40** (2511) → **50** (2512). **Base CFG** is 4.0, except 2.5 on 2508 and Layered. Lightning is 8 / 1.0 on the base template and 4 / 1.0 everywhere else, and the 2509 template defaults to its Lightning branch. Every Edit template carries **`CFGNorm`** on the model path; 2511 sets `FluxKontextMultiReferenceLatentMethod` to `index_timestep_zero`. The per-template table — latent and scaling nodes, LoRA filenames, dates — is `references/setup-and-workflows.md §1`.

### Quantisation & VRAM

**The fp8 file is 20.4 GB and you do not need 24 GB of VRAM to run it.** ComfyUI streams blocks from system RAM. What you need is roughly **26 GB of VRAM plus RAM combined**; less VRAM costs time, not capability `[community — nsfwVariant; strong]`. Official builds: `bf16` (40.86 GB) and the ~20 GB quants — `fp8_e4m3fn`, `fp8mixed` (the 2511 and Layered default), `int8_convrot` (2511 only), `nvfp4` (base only). **GGUF changed maintainer at 2511**: city96 then QuantStack up to 2509, **`unsloth/Qwen-Image-Edit-2511-GGUF`** from there. **Nunchaku SVDQuant stopped at 2509** (org renamed to `nunchaku-ai`, last update 2025-11-16); there is no official 4-bit build for 2511, 2512 or Layered. Whether Q8 GGUF matches fp8 is contested (two-bar section). **2.1 has no quant ladder yet**: bf16 and `int8_convrot` only, no fp8 build, community GGUFs appeared within hours of release but `ComfyUI-GGUF` has not been updated since 2026-01-12 so there is no loader for them `[flagged — re-verify]`. The only VRAM data points are a 4.5 s / 34 GB-peak 1024² run on a GB300 under vLLM-Omni and ~46 GB bf16 / ~31 GB q8 on an M5 Max under mflux; SGLang says the full resident pipeline exceeds a 4090 or 5090 and streams the encoder; DiffSynth claims a 7 GB floor with disk offload `[official-adjacent; no consumer-GPU Comfy report yet]`.

The floor is **6 GB VRAM + 32 GB RAM** (Edit-2509 Q8 GGUF, 4-step Lightning, 90–120 s an image via offload); **8 GB + 16 GB RAM** runs the Nunchaku int4 fused-Lightning 2509; **24 GB** runs fp8 native at 20–30 steps with double-ref on, and a 5090 does 2–3 MP edits in 90–130 s `[community — gebba, nsfwVariant, Puzzled-Valuable-985]`.

Full ladder, GGUF repos, the Nunchaku gotchas and the merged AIO checkpoints: `references/setup-and-workflows.md §2`.

### diffusers

`QwenImagePipeline` (base, 2512), `QwenImageEditPipeline` (2508), **`QwenImageEditPlusPipeline`** (2509 and 2511, `image=[img1, img2]`), `QwenImageLayeredPipeline`, plus img2img, inpaint, edit-inpaint and two ControlNet pipelines. Floors: `diffusers>=0.35.0` for base and Edit, 0.36.0 for the Plus pipeline, 0.37.0 for Layered. Pass **`true_cfg_scale=4.0`** and `negative_prompt=" "`; `guidance_scale` is inert. The cards' `pip install git+…` line is launch-day legacy. The minimal call and the per-checkpoint parameters: `references/setup-and-workflows.md §3`. **2.1** is **`QwenImage21Pipeline`** (diffusers 0.41.0, `transformers>=5.17`), one class for T2I and edit: pass `image=` (one image or a list) to edit, `num_inference_steps=40`, no `true_cfg_scale` unless you also pass a negative. Output is RGBA when the prompt asks for transparency; save PNG. `references/setup-and-workflows.md §11`.

### Hosted surfaces

Alibaba Model Studio (DashScope) serves `qwen-image-3.0-pro`, `qwen-image-2.0[-pro]`, `qwen-image-max/plus` and `qwen-image-edit`, with **`prompt_extend` on by default** — the API rewrites your prompt unless you turn it off. ComfyUI reaches 3.0 Pro through the `QwenImageTextToImageApi` / `QwenImageEditApi` nodes; Replicate and fal host the open weights and a LoRA trainer. Pricing is unverified: `references/api-and-hosted.md`.

---

## Per-variant settings

### Qwen-Image and Qwen-Image-2512 (undistilled T2I)

- **Steps:** 20 (template) to 50 (cards, Lightning README). 2512's template ships 50. Treat 20 as a draft and 40–50 as the recommendation.
- **CFG:** 4.0 (template and cards). Realism-LoRA authors run `dpmpp_2m` / `beta`, 50 steps, guidance 2.5 `[community — Danrisi, Civitai]`.
- **Sampler:** `euler` / `simple`, shift **3.1**. RES4LYF (`res_2s` / `bong_tangent`) at 12 steps is a named realism recipe for T2I `[community — Top_Buffalo1668, 000TSC000]`. The same samplers are reported to *hurt* on Edit (two-bar section).
- **Resolution:** the **1328 class** — 1328 × 1328, 1664 × 928, 1472 × 1104 and their transposes, 1584 × 1056. This is a ~1.5 MP band, not 1 MP, and training buckets should respect it (`references/lora-training.md §3`).
- **Negatives:** live. 2512's card ships a Chinese negative against waxy skin, over-smoothing and "AI look"; the earlier cards pass a single space.
- **Seed:** high diversity; no Qwen-specific seed pathologies are reported.
- **LoRAs:** `LoraLoaderModelOnly`; realism LoRAs at 1.0 `[community — FortranUA]`. **2512 is a different base.** LoRAs trained on Qwen-Image "destroy the image" on 2512, and lowering strength does not help `[community — Friendly-Fig-6015; single report]`. Use `-2512` retrains.

### Lightning (distilled path, any variant)

- **Steps / CFG:** 8 / 1.0 (base V1.0 8-step) or 4 / 1.0 (everything else). Exactly 1.0. Negatives inert.
- **Version:** V2.0 "produces images with reduced over-saturation, improved skin texture", and the base template still ships V1.0 `[official — Lightning README]`. A colour cast is a version question before it is a LoRA question.
- **Quality:** the maintainers concede that dense text and hair-like detail come out "blurred or excessively sharpened". Three named authors rate Lightning-loaded Edit as plastic; one finds the 4-step LoRA *better* than 50 base steps for text-heavy edits `[community — Furacao__Boey; single report]`. Contested; see the two-bar section.
- **Base file:** scaled fp8 or bf16, never plain `fp8_e4m3fn` (the grid, above).

### Qwen-Image-Edit 2509 / 2511 (undistilled edit)

- **Steps / CFG:** 2509 — 20 / 4.0 (template), 40 / 4.0 (card). 2511 — 40 / 4.0. The best-documented 2511 author runs **20 steps, 30 for maximum sharpness**, `euler` / `simple`, and finds RES4LYF reduces quality on Edit `[community — nsfwVariant, r/StableDiffusion 1tqm8ic; strong]`.
- **Shift:** 3.0 (2509), 3.1 (2511). Do not carry shift between workflows: the Edit upscale LoRA wants **< 0.3**, and the node is parameterised differently across graphs.
- **Resolution:** **2–3 MP single-image, 1–2 MP multi-image**, dimensions **divisible by 16** (ComfyUI rounds to 8, and the mismatch ruins one edge). Anatomy fails above ~3 MP.
- **Inputs:** 1–3 images is the trained range. Inputs should share at least one edge dimension for transfer tasks.
- **Precision and LoRAs:** bf16 holds input colour better than fp8. 2511 has a contrast regression and broke 2509 LoRA compatibility; the Rapid-AIO 2509+2511 merge exists to fix both `[community — FluffyQuack, Phr00t]`. Otherwise use `-2511` retrains.

### Qwen-Image-Layered

- 640 max dimension, shift **1.0**, 20 / 2.5, `layers: 4` and `cfg_normalize: True` in diffusers, its **own VAE file**. That is the whole published settings surface; no first-hand craft report exists. **2.1 supersedes it for new work** (transparency is native there), but Layered is Apache 2.0 and 2.1 is not.

### Qwen-Image-2.1 (unified T2I + edit) `[official — card, diffusers, Comfy PR and templates, 2026-09-20; no community craft yet]`

- **Steps / CFG:** 40 / 1.0 (card, diffusers). Comfy templates 25 / 1.0 with a note that the official pipeline uses 40–50 on `euler`. Raise CFG only with a negative prompt; it doubles per-step cost.
- **Sampler:** `euler` / `simple`. No shift node; the scheduler is FlowMatchEuler with dynamic shift (`base_shift 0.5` at 256 tokens, `max_shift 0.9` at 8192, `shift_terminal 0.02`), which Comfy bakes in as 0.69 at 1 MP.
- **Resolution:** native 2K. Official sizes: 2048², 2400 × 1792, 2528 × 1696, 2752 × 1536 and transposes. Multiples of 32. The T2I template defaults to **1 MP**; set 4 MP for 2K. Official demo cases also ran 1664 × 2496 and 1760 × 2368, so the grid is loose.
- **Edit sizing:** the `resolution` field is a **total-pixel budget** (≈ res², aspect kept; 0 keeps each reference's own size, rounded to 32). The output canvas follows **`image_1`**; the node emits a matching empty latent because "any other size shifts the edit". Extra references may differ in size and aspect. The edit template ships `resolution 0`.
- **Inputs:** 0–10 references (the node exposes 16). `image_1` is the edit target, the rest are references. A separate mask goes in as a second image. Alpha in a reference: the vision tower sees it composited over white, the VAE keeps all four channels.
- **Prompt grammar:** `<image1>`, `<image2>` … in the prompt, not `Picture 1:`. The RGBA wrapper is a fixed sentence pair. Local edits name the annotation ("in the red circle", "the white-masked area") and, for drawn annotations, say the marks must not be rendered. `references/prompting-guide.md §9`.
- **KV cache:** `QwenImage21Cache` default `auto` / `default` is lossless; `int8` halves it at ~bf16 accuracy; `int4` quarters it and roughly doubles per-step error; `off` recomputes every step (debug only). Edits with many references are where it pays.
- **Negatives:** inert at CFG 1. DiffSynth coerces an empty prompt to a single space because Qwen has no BOS token.
- **LoRAs, Lightning, ControlNet:** none exist. Comfy's LoRA key mapping is in place for when they do.

---

## The Edit reference path — bypass the 1 MP downscale

This section is about the **20B Edit** node. 2.1's `TextEncodeQwenImage21` resizes with Lanczos to a pixel budget you set, rounds to 32, and keeps references at native size when `resolution` is 0, so the bypass below is not needed there.

The single largest quality lever on the 20B Edit is a ComfyUI node, not a model setting. `TextEncodeQwenImageEditPlus` force-downscales every reference to 1,048,576 px with **AREA** resampling, a box filter that destroys detail before the model sees it, and rounds to 8 where the patchifier wants 16. Two authors found the same fix a version apart, from different symptoms: blur, and edits that "unzoom" from the source `[community — nsfwVariant, danamir_; convergent]`.

1. **Disconnect the VAE input from `TextEncodeQwenImageEditPlus`**, or replace the node with plain `CLIPTextEncode`. The forced rescale only runs when the VAE input is filled.
2. **Scale the reference yourself with Lanczos**, to a multiple of 16, inside the 2–3 MP band.
3. **`VAEEncode` each reference and chain one `ReferenceLatent` per source** into the conditioning.
4. **Write the `Picture 1: … Picture 2: …` labels yourself.** The node normally splices the VL model's description of each image in that exact shape, and "Qwedit was trained on this exact format". A five-word label beats the VL model's paragraph.

Edits then land pixel-perfect at native size, and 1440–1920 px edits become routine. One refinement from the same sixteen-way A/B: **feed the reference in twice** ("double-ref") for sharper output and better off-angle likeness at ~50% more time, always better for single-image `[community — nsfwVariant; single report, reproducible]`. Node-by-node wiring: `references/setup-and-workflows.md §5`.

**The other grid.** The Qwen VAE prints a **faint halftone**, worse at high resolution and on Edit. It is a decode artefact, not LoRA overfit, and not the fp8 grid from the one-rule section (that one is fixed by the base file). The fix is a **0.5–0.75× downscale then re-upscale**; SeedVR2 removes it (`references/setup-and-workflows.md §8`).

---

## Text and editing are the strengths; realism is the tax

Qwen-Image's headline is **bilingual in-image text** and **instruction editing that preserves it**. 2509 edits the font, colour and material of existing text; 2511 folds relighting and novel viewpoints into the base; 2.1 claims improved typography, portrait lighting and product fidelity, edits text inside a transparent layer, and turns one reference into infographics, storyboards and 360° panoramas (official showcase only). Its default *look*, though, is plastic, from three causes that get conflated `[community — nsfwVariant, Top_Buffalo1668, Square_Empress]`:

1. **Distillation.** Lightning collapses the high-frequency tail. Fix: drop Lightning and run 20 steps on the base.
2. **Base-model bias.** Skin is "still a bit plastic" at 12 steps with no Lightning loaded; "ZIT is clearly superior in terms of realism". How much of the tax is Lightning's is contested.
3. **Edit patch mismatch.** An edited region comes back smoother than the surrounding skin.

The lever that moves it is **a realism LoRA**, not adjectives. No first-hand report says camera words alone defeat the bias. That is the reverse of [`z-image`](../z-image/), where the camera stack *is* the fix. The stack by downloads (Civitai, 2026-09-09): **Lenovo UltraReal** and **NiceGirls UltraReal** (Danrisi), **2000s Analog Core**, **Boreal**, and the Edit-side **qwen-edit-skin**. **SamsungCam UltraReal** runs at weight 1.0 and stacks with character LoRAs `[community — FortranUA]`. One wording rule carries over from Krea 2, on the same VAE family: say "photo", never "photorealistic", because the base saw a lot of artwork `[community — Ashen3; Krea 2 evidence]`.

**2.1 has official prompt rewriters, and they are the only prompt guidance it ships.** `Qwen/Qwen-Image-2.1-PE-T2I` and `-PE-I2I` are fine-tuned Qwen3.5-VL 9B models (~19 GB each, same research licence) that turn a brief into a long English paragraph plus a `wh_ratio`. Their system prompts are the useful part even if you never run them: describe the finished image as an observer, open with medium + style + subject + background, place every element with 8–14 positional phrases that reach the corners, quote every legible string in its own script, call unreadable text "blurred" rather than inventing it, never write a ratio or pixel count into the prompt. For edits: change exactly the named attribute and push it hard, name what stays by type and position **without describing its appearance** (a concrete preservation description reads as a generation instruction and drifts), and point at the reference image for identity rather than describing the face. `references/prompting-guide.md §9`.

**"Positive magic."** Qwen's own inference code appends `, Ultra HD, 4K, cinematic composition.` to every English prompt and most templates carry it; realism-LoRA authors' settings do not, and nobody has A/B'd it. "Cinematic composition" is a real style instruction to a VL encoder, so decide, don't inherit it. Vocabulary, the Edit instruction rules and the rewriter's text rules: `references/prompting-guide.md`.

---

## Structural control and Layered

Ranked by evidence of use, not by what exists. Two paths carry real practice: the **native ControlNet inside Edit-2509+** (depth, edge, keypoint; strength 0.9 in a published head-swap pipeline `[community — Substantial_Angle680]`), and **InstantX ControlNet-Union** for T2I (`ControlNetLoader` → `ControlNetApplyAdvanced`, scale 0.8–1.0; its card warns that small-font text is lost unless named in the prompt). DiffSynth's canny / depth / inpaint are **model patches**, loaded through `ModelPatchLoader` → `QwenImageDiffsynthControlnet` from `models/model_patches/`, the same node pair [`z-image`](../z-image/) uses. **EliGen and Blockwise-ControlNet show zero monthly downloads**; routing a reader there routes them to abandoned code. **Layered** is adopted and trainable, and no author has published a settings-level write-up. Files and wiring: `references/setup-and-workflows.md §6`.

**2.1 changes the mask story.** It has no ControlNet, but region control is built in three ways: draw coloured circles on the input and name them ("remove the watch in the blue circle … the annotation lines must not be rendered"), paint a region and name it ("the white-masked area on the right"), or pass the untouched original as `image_1` and a mask as `image_2` and say "at the circled place add …". Transparency is native: an RGB photo in, "extract the subject as a transparent layer" out, and a transparent input can be edited while staying transparent. The background-removal template is literally `"Remove the background, and output a PNG image"`. Whether alpha edges are usable without matting is unmeasured; DiffSynth warns hair, ribbons and water produce wide semi-transparent alpha and suggests "clear silhouette / clean edges" in the prompt.

---

## LoRA training & characters (summary — full treatment in references)

**There is no Raw/Turbo split, no distillation to train against, and no training adapter.** Qwen-Image is one undistilled base per generation, and Lightning is an inference-time LoRA. The doctrine is simply **train on the bf16 base, stack Lightning at inference**. That is what every named author does and what every official config implies. Grid the checkpoints on the base at 20 steps *and* on your deploy graph, never only through Lightning.

**Trainers and defaults** `[official — ai-toolkit, musubi and DiffSynth configs]`: ai-toolkit rank 16, LR 1e-4, `[512, 768, 1024]`, 24 GB via `uint3` plus a per-variant recovery adapter. musubi-tuner dim 16, **LR 5e-5**, `--discrete_flow_shift 2.2`, **bf16 files only**. DiffSynth rank 32 and the only documented **`--zero_cond_t`** for Edit-2511. "Kohya" here means musubi. The community character recipe runs rank 32, LR 2e-4, `sigmoid` timesteps, 40–60 images at 60/30/10 close/half/full `[community — FarTable6206]`. Caption length is four-way contested, and 60/30/10 sits against the suite's one-third rule. **The trigger is a plain name in a sentence**, not a rare token.

**2.1 training** `[official — DiffSynth-Studio, 2026-09-20]`: DiffSynth is the **only trainer** with support today (ai-toolkit, musubi-tuner and SimpleTuner have none `[flagged — re-verify weekly]`). Its example: rank 32, LR 1e-4, `--max_pixels 1048576`, 5 epochs, gradient checkpointing, default target modules. Images load as **RGBA**, so a dataset with alpha trains transparent generation directly; edit LoRAs use `--data_file_keys "image,edit_image" --extra_inputs "edit_image"`. There is no Lightning to stack and no community LoRA to compare against. The licence's §4(b) requires "Built with Qwen" on any published derivative, and §2 makes selling a LoRA or its outputs non-compliant without a commercial licence. `references/lora-training.md §11`.

**[`krea-2`](../krea-2/) doctrine does not transfer**, even though Krea 2 is Qwen-derived and shares this VAE: shift 2.5 (Qwen's is 2.2), the 1024-native source rule (Qwen's band is the 1328 class), LoKr, grad-accum 2 and Raw/Turbo are Krea findings, and nothing in this skill is `[live-use]` (`references/lora-training.md §9`).

**Characters.** There is **no PuLID, InstantID or IP-Adapter-FaceID for Qwen-Image**. Edit-2511 *is* the identity tool: "I don't think there's a better open-weight model out there than Qwedit for making new shots of a character without loras" `[community — nsfwVariant; strong]`. The protocol: a nude or plain-underwear reference even for SFW work, several zoom levels by prompting the zoom, several angles via the Multiple-Angles LoRA, then ~90% of new shots off a single reference. Re-anchor on it rather than chaining edits; how many sequential edits identity survives is **unpublished**. Edit also builds the LoRA dataset: one headshot → 20 angle variants in ~130 s `[community — acekiube]`. **2.1 claims stronger identity and product fidelity and accepts ten references** (the official showcase fuses six portraits into one group photo), but as of 2026-09-20 nobody outside Qwen has tested likeness, and its licence bars the commercial character work most readers do. Keep Edit-2511 as the identity engine until a named author reports otherwise.

**Ecosystem, 2026-09-09:** 1,637 Qwen LoRAs on Civitai by full pagination — 273 character (16.7%), 773 style, 422 adult-flagged (25.8%); the earlier 1,162 / 192 figure excluded NSFW. The encoder cannot refuse; it is consumed as hidden states, not run as a chat model. Abliterated encoders are reported to reduce quality rather than unlock anything `[community — -p-e-w-, nsfwVariant]`. Full treatment: `references/lora-training.md`, `references/characters.md`, and [`character-lora-training`](../character-lora-training/).

---

## Production pipelines & mixing models

Qwen's ladder is short. The edit model does the work most families give to detailers and inpaint, and it runs natively at 2–3 MP.

1. **Compose** — Qwen-Image or 2512 on the base regime at a 1328-class size; Lightning for drafts. Judge layout, reroll freely.
2. **Edit passes** — Edit-2511 through the bypass path, double-ref on, one instruction per pass, each re-anchored on the *original* reference. This replaces img2img: Edit has no denoise dial, and no published denoise ladder exists for Qwen img2img.
3. **Halftone round trip** — 0.5–0.75× downscale, then re-upscale.
4. **Detailer** — FaceDetailer with the character LoRA swapped in here, for distant faces.
5. **Upscale** — the Edit-2509 Upscale LoRA (`"Enhance image quality"` plus a scene description, 8-step Lightning, shift < 0.3) or SeedVR2 `[community — vafipas663]`.

Every stage after 1 is bypassable. **2.1 in this ladder:** for a research job it collapses stages 1 and 2 into one checkpoint and adds native transparency; it cannot appear anywhere in a commercial ladder. **Mixed-model roles:** Qwen-Image-Edit is the suite's **dataset factory and edit engine for other families' stills**. [`z-image`](../z-image/) and [`krea-2`](../krea-2/) both send their anchor images here to be multiplied into a LoRA set. The reverse handoff is [`z-image`](../z-image/) as the realism finisher for a Qwen composition, because Z-Image wins on skin `[community — Top_Buffalo1668]`. **Decode to pixels between VAE families**; identity-preserving refines live at denoise ~0.2–0.5. Cross-family craft: [`image-production-workflows`](../image-production-workflows/). Renting the GPU: [`comfyui-on-runpod`](../comfyui-on-runpod/).

**Making it move:** a still locked here goes to image-to-video on [`wan-2-2`](../wan-2-2/), whose VAE is the same family. Multiply the character's angles on Edit before handing a frame over; the video model inherits whatever likeness the still carries.

---

## Failure modes & QC

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| Every Edit output is soft or blurry | `TextEncodeQwenImageEditPlus` downscales to 1 MP with AREA resampling — a box filter — before the model sees the reference | Bypass: VAE-encode refs yourself into chained `ReferenceLatent`, Lanczos-scale first |
| Edit "unzooms" or does not register with the source | Same forced rescale; output is generated at a different scale | Same bypass; edits then land pixel-perfect |
| Ruined strip along one edge | Node rounds to 8; the patchifier wants 16, so the last partial patch is garbage | Size to a multiple of 16 |
| Faint halftone / grid, worse at high res | The Qwen VAE decoder, strongest on Edit | Downscale 0.5–0.75× and re-upscale; SeedVR2 removes it |
| Regular grid after adding Lightning | Downcast `fp8_e4m3fn` base × bf16-trained LoRA — no calibrated scaling | Scaled fp8 base, the fp8-distilled Lightning LoRA, or bf16 |
| Plastic skin | Distillation collapsing the high-frequency tail; base-model bias; edit-region texture mismatch | Drop Lightning, 20 base steps; stack a realism LoRA (UltraReal, SamsungCam, qwen-edit-skin) |
| Blocky artefacts on 2511, absent on 2509 | The 2511 4-step Lightning LoRA specifically; persists across samplers and 8–24 steps | Use the 2509 Lightning on 2511, or the 8-step 2511, or none `[community — MastMaithun, DrinksAtTheSpaceBar]` |
| Lightning LoRA no-ops (returns input in 0.01 s) or gives garbage | Version skew — the adapter's keys do not bind to a different generation | Match the Lightning file to the exact base (base / 2509 / 2511 / 2512) |
| Colour or tone shifts across edits | fp8 precision; 2511's contrast regression; residual per-edit drift | bf16 or fp8mixed; the Rapid-AIO merge; **re-edit from the reference, never edit the edit** |
| Pose changes when you only asked for an outfit change | Under-specified instruction — unmentioned attributes are treated as free | Add "Leave their pose unchanged" — ~99% correct `[community — nsfwVariant]` |
| Output inherits grain or blur from the source | Appearance conditioning comes straight from the VAE-encoded source | Clean or upscale the input first, even for "entirely new shots" |
| Likeness degrades in distant shots | Face occupies too few latent tokens | FaceDetailer pass, needed more often than on Z-Image |
| Hands and anatomy break above ~3 MP | Anatomy priors learned near 1–2 MP | Stay ≤ 3 MP or re-roll |
| Any LoRA destroys the image on 2512 or 2511 | LoRA trained for a different generation; 2511/2512 folded community LoRAs into the base, so old deltas double-apply | `-2511` / `-2512` retrains; lowering strength does **not** fix it |
| 2.1 graph errors at load, or a 20B LoRA / Lightning file does nothing on it | Different architecture: 7B single-stream, 64-channel latent, Qwen3-VL-8B encoder; no 20B file binds | Use only `Comfy-Org/Qwen-Image-2.1` files; there are no 2.1 LoRAs or Lightning yet |
| 2.1 output is opaque when you asked for transparency, or alpha vanishes on save | Transparency is prompt-switched; `.convert("RGB")`, JPEG, or an RGB-only save node discards the fourth channel | Use the RGBA wrapper sentence verbatim; save PNG (`SaveImageAdvanced` png in the templates) |
| 2.1 transparent subject has wide soft fringes | Hair, ribbons, water and glow produce mid-value alpha by design | Prompt "clear silhouette, clean edges"; avoid describing environment light, which fills the canvas `[official — DiffSynth doc]` |
| 2.1 edit shifts or misregisters against `image_1` | Latent size differs from the resized `image_1` grid | Use the latent the text-encode node emits; with `custom_size` on, keep the canvas near the resized `image_1` |
| 2.1 twice as slow after adding a negative | CFG engaged (`true_cfg_scale` > 1 needs a negative and runs the DiT twice per step) | Drop the negative, or accept the cost knowingly; the model is tuned for CFG off |
| 2.1 renders your circles or paint marks into the output | Annotations are pixels the model can copy | State "the annotation lines must not be rendered in the image", or pass the original plus a separate mask as two inputs |

---

## Pre-flight checklist

1. Which regime? Base: 20–50 steps, CFG 2.5–4.0, real negative. Lightning: 4/8 steps, CFG exactly 1.0, no negative. Never crossed?
2. Flipped the template's `PrimitiveBoolean`, not the KSampler widgets?
3. Lightning file matches the exact base generation, and the base is scaled fp8 or bf16 — not plain `fp8_e4m3fn` under a bf16 LoRA?
4. `CLIPLoader` type `qwen_image`; `ModelSamplingAuraFlow` shift 3.1 (T2I, 2511, 2512), 3.0 (2508, 2509), 1.0 (Layered)?
5. Edit: `TextEncodeQwenImageEditPlus` VAE input disconnected, Lanczos scale to a multiple of 16 inside 2–3 MP, one `ReferenceLatent` per source, double-ref on, source cleaned first?
6. Prompt is a sentence, simple and direct; `Picture 1:` labels written; what must *not* change is named?
7. Text to render in straight double quotes with position stated, kept in its original language?
8. Realism: a realism LoRA loaded, and the "positive magic" suffix a deliberate choice?
9. Halftone round trip and a FaceDetailer in the graph for distant faces?
10. LoRA: trained on the bf16 base for *this* generation, evaluated on the base at 20 steps and on the deploy graph, trigger a plain name?
11. Character: re-anchoring on the original reference each edit, not chaining?
12. **2.1:** is the job research or evaluation, not commercial? 40 steps, CFG 1.0, no negative; `<image1>` tags; RGBA wrapper verbatim and PNG out; `image_1` is the target; nothing from the 20B shelf in the graph?

---

## Where Qwen-Image sits in the suite

Choose the model for the job. Defaults like realism direction and prompting dialect are model-specific, not universal:

| Job | Qwen-Image | Reach for instead |
|---|---|---|
| Consistent characters | **Edit-2511 is the suite's no-training identity engine** and the dataset factory the other image skills point at; character LoRAs work too (`references/characters.md`) | [`flux-2`](../flux-2/) for PuLID-class adapters; [`krea-2`](../krea-2/) for Identity Edit and multi-character DOP; [`z-image`](../z-image/) when the face must be photoreal in close-up |
| Style / character LoRA ecosystem | Large (1,637 LoRAs, style-heavy), full trainer support, Edit LoRAs on paired data | [`sdxl`](../sdxl/) for the deepest mature ecosystem; [`character-lora-training`](../character-lora-training/) for the craft that transfers |
| In-image typography | **Strong**: bilingual rendering, and editing existing text while preserving its font is unique in the suite | [`ideogram-4`](../ideogram-4/) for typography-led layout and dense lettering |
| Structural control (pose / depth / canny) | Native 2509 ControlNet, InstantX Union, DiffSynth patches | [`sdxl`](../sdxl/) for the most complete stack |
| Photoreal faces and skin | Workable only with the realism-LoRA stack; the plastic default is the tax | [`z-image`](../z-image/): better skin with nothing loaded, and the standard finisher for a Qwen composition |
| Instruction editing / multi-image composition | **The headline**: 1–3 references on Edit-2511, product placement, relight, novel angles, text edits; **up to 10 references, mask-guided local edits and transparent layers on 2.1** (research only) | [`krea-2`](../krea-2/) + Identity Edit for one-sentence scene-preserving edits; [`flux-2`](../flux-2/) for its multi-reference path |
| Aesthetic range | One realism-leaning look | [`krea-2`](../krea-2/): no house look, style as a control surface |
| Commercial use under the licence | **20B family: Apache-2.0 on code and weights, no output clause, no gate** — tied with Z-Image as the least encumbered. **2.1: Qwen Research License, non-commercial, PRC law** — out on the first rung of the atlas ladder | Nothing is freer than the 20B family; for the jobs only 2.1 does, [`flux-2`](../flux-2/) [klein] or [`z-image`](../z-image/) with a matting pass are the commercial routes |
| Mixed-model pipelines | The edit engine and dataset factory for other families' stills; Z-Image finishes its skin | [`image-production-workflows`](../image-production-workflows/) for the cross-model craft |
| Making it move | Still images only — but the still that carries angle and outfit coverage into video is built here | [`wan-2-2`](../wan-2-2/): image-to-video from a still locked here; same VAE family |
| **Choosing between all of these in the first place** | — this table is one model's view of the suite | [`generative-media-atlas`](../generative-media-atlas/): the whole suite ranked by job, the elimination ladder, and end-to-end routes across several skills |

---

## Licence & limitations

| Artefact | Licence | Commercial use |
|---|---|---|
| Inference code (`QwenLM/Qwen-Image`) | Apache 2.0 (verbatim `LICENSE`) | Yes |
| All six open weights (base, Edit, 2509, 2511, 2512, Layered) | Apache 2.0 — `license: apache-2.0` on every card | Yes; derivative LoRAs and merges permitted; no acceptable-use addendum, no output-rights clause |
| **Qwen-Image-2.1** weights, and the PE-T2I / PE-I2I rewriters | **Qwen Research License Agreement** (LICENSE dated 2026-09-20, verbatim): §1(i) "Non-Commercial" = research or evaluation; §2(b) commercial use needs a separate licence from `model-business@notice.qwencloud.com`; §4(b) derivative models must display "Built with Qwen" / "Improved using Qwen"; §4(c) "Qwen" may not be a derivative's primary name; §5(c) patent-retaliation clause; §8 PRC law, Hangzhou courts. The Comfy-Org repackage inherits it | **No** without a separate licence. Outputs are "Materials … or any outputs" under §4 and the grant in §2 is non-commercial |
| Qwen-Image 2.0 / 3.0 | No weights published; Alibaba Model Studio terms | Per the hosted terms |
| Third-party pieces — Lightning LoRAs (lightx2v), InstantX and DiffSynth ControlNets, PAI's 2512 Fun ControlNet, community merges | **Not covered by Qwen's grant**; read each card | Check each |

**Timeline:** six Apache-2.0 releases from 2025-08-04 to 2025-12-31 (dates in the selector); 2.0 announced 2026-02-10 and 3.0 Pro on the API by 2026-08, both hosted-only; **2.1 open weights 2026-09-20 under the research licence**, after a 50-seat ModelScope early-access programme announced 2026-09-17. 2.0 and 3.0 remain without weights. The HF Space demo for 2.1 calls a hosted `pre-qwen-image-2.1-pro` endpoint with server-side prompt rewriting, so demo quality is not local-weights quality. Whether a commercial licence path or a re-licence follows is unknown `[flagged — re-verify]`.

**Known limitations:** plastic skin without a realism LoRA, no identity adapter of the PuLID class, no official Nunchaku past 2509, broken 2509 LoRA compatibility on 2511, anatomy degrading above ~3 MP on Edit, no denoise ladder for img2img, a Layered model with adoption and no craft, and a 2.1 that is non-commercial, unquantised beyond int8, untrainable outside DiffSynth and untested by anyone but Qwen. Publishing gates (Civitai's real-person ban, the TAKE IT DOWN Act) bind harder than capability. Owned by [`character-lora-training`](../character-lora-training/references/publishing-and-likeness.md).

---

## How to read the claims in this skill — two bars, by claim type

This skill holds two kinds of claim to two different standards, because they fail in two different ways.

**Hard facts — must be exact or it breaks.** This covers the architecture (20B dual-stream MMDiT, `guidance_embeds: false`, Qwen2.5-VL 7B, the Wan-2.1-family VAE; 2.1's 7B single-stream DiT, Qwen3-VL-8B, 64-channel RGBA VAE, `causal_condition`), the Apache-2.0 licence on the 20B family and the research licence on 2.1, exact filenames and sizes, the Comfy-Org and Lightning repos, the node names and the CLIPLoader `qwen_image` type, every template number, the diffusers classes and version floors, the `true_cfg_scale` semantics, the trainer defaults, the Lightning fp8 correctness table, and the hosted model strings. **The source of truth is official**: the template JSON, the Comfy-Org HF trees, the six model cards and their `config.json` files, the QwenLM README and `prompt_utils.py`, the Lightning README, the trainer docs, and the Model Studio docs, read verbatim on 2026-09-09; for 2.1 the HF card, LICENSE and `config.json` files, the `QwenLM/Qwen-Image-2.1` README, diffusers PR #14804, ComfyUI PR #16400 and templates v0.11.65–66, DiffSynth's 2.1 doc and the PE system prompts, read verbatim on 2026-09-20. A wrong filename 404s; a wrong shift silently degrades. The volatile ones are the packaging: Lightning and GGUF filenames, quant maintainers, template defaults, hosted pricing, and the closed status of 2.0 and 3.0. **Re-verify these before relying on them, regardless of who said it.**

**Craft — what actually makes a good image.** This covers the two-regime settings ladders, the Edit bypass and double-ref, the 2–3 MP band, the halftone round trip, the plastic-skin diagnosis and the realism-LoRA stack, the `Picture N:` and relight grammar, the nude-reference protocol, the VRAM table, the LoRA hyperparameters, and the failure-mode fixes. **The authoritative source here is the community**: named, reproducible authors who ran the generations — nsfwVariant (the sixteen-way reference A/B and the 2509/2511 write-ups), danamir_, FluffyQuack, Danrisi, Top_Buffalo1668, 000TSC000, FarTable6206, Icy_Upstairs3187, acekiube, AwakenedEyes, vafipas663, Phr00t, kingroka, dx8152, Substantial_Angle680, gebba, Epictetito, MastMaithun, and the Civitai model API — across Reddit, Civitai and Hugging Face. It is stated with confidence. Ranges mean "your quant, resolution or dataset differ from the author's", not "unreliable". Two blind spots are named rather than hidden. Chinese-language venues (Bilibili, LiblibAI, ModelScope) were not sampled, which bears directly on Chinese text rendering. Banodoco is Discord-only.

Nothing in this skill is `[live-use]`. This lab has not trained a Qwen LoRA, and every media-lab finding a sibling cites is Krea 2 evidence. **Every 2.1 claim is official-only.** The weights were hours old when this pass ran; Reddit, Civitai and Banodoco carried nothing, and the fifty ModelScope early-access testers publish by 2026-09-28. Until then 2.1 has hard facts and no craft.

Contested and unresolved points, each greppable:
- **Lightning quality.** Three named authors call it a loss (plastic skin). One finds the 4-step LoRA cleaner than 50 base steps for text. And 2511's own 4-step LoRA is reported blocky by one author, with its 8-step build reported as the fix by another `[contested]`.
- **Samplers.** `euler` / `simple` beats everything on Edit for one author; RES4LYF is "needed for realism" on T2I for two others. Different models, possibly both true `[contested]`.
- **Quant floor** — "fp8 is much higher quality than Q8 GGUF" against "no difference Q4–Q8" `[contested]`.
- **Abliterated encoders** — the mechanism and the leading craft author say no; a second-hand minority says yes `[contested]`.
- **"Positive magic"** — in the official code and most templates, absent from every realism author's settings, never A/B'd `[contested]`.
- **LoRA hyperparameters** — LR 5e-5 / 1e-4 / 2e-4, rank 16 / 32, training resolution 1024 vs the 1328 class, `weighted` vs `sigmoid`, caption length, and 60/30/10 vs one-third, all held by named authors with no same-dataset A/B `[contested]` (`references/lora-training.md §8`).
- **Negatives on the base.** The suite's authoring spec records an unattributed report of negatives ignored across CFG 1–7. This pass found no named author saying so, and 2512's own card ships a working negative. Treated here as live at CFG > 1 `[flagged — re-verify]`.
- **Packaging that moves** — Lightning and GGUF filenames and maintainers, the Nunchaku gap at 2509, the closed status of 2.0 and 3.0, and hosted pricing `[flagged — re-verify]`.
- **2.1 unknowns** — whether identity holds across ten references, the plastic-skin status, usable alpha edges, real consumer-GPU VRAM, the edit output-size rule (Comfy follows `image_1`, vLLM-Omni the last reference), a tagged ComfyUI release, a GGUF loader, any trainer beyond DiffSynth, any distillation, and any commercial-licence path `[flagged — re-verify after 2026-09-28]`.

**Facts dated 2026-09-09; community craft refreshed 2026-09-09.** What moves fastest is the packaging around the weights and the open/closed line, which could move if Qwen releases 2.0 or 3.0 weights. The two-regime rule, the Edit bypass and the identity protocol age slowly.

---

## Reference files

| File | When to read it |
|---|---|
| `references/prompting-guide.md` | Writing or fixing a prompt: the sentence dialect, the Edit instruction rules, the `Picture N:` format, the official rewriter's rules, text in both languages, the aspect table and 2512's negative, drop-in templates |
| `references/setup-and-workflows.md` | Installing, sizing a card, wiring an Edit graph or **using** a LoRA: every template's files and numbers, the quant and GGUF ladders and their maintainers, diffusers per checkpoint, the bypass node by node, ControlNet and model patches, the upscale and halftone chains, cross-generation LoRA compatibility, the ladder and handoffs |
| `references/lora-training.md` | **Making** a LoRA: the trainer matrix and each trainer's defaults, the undistilled-base doctrine, the 1328-class question, captions and the plain-name trigger, composition, rank and LR, Edit LoRAs on paired data, adult work and the encoder question, evaluation, what does not transfer from Krea 2 |
| `references/characters.md` | A character has to survive more than one image: Edit-2511 as the identity engine, the reference protocol, the adapter status table, Edit as the dataset factory, character LoRA plus Edit, multi-subject fusion, failure modes, when to route to a sibling |
| `references/api-and-hosted.md` | Renting rather than running: the DashScope model strings and parameters, the ComfyUI API nodes for 3.0 Pro, Replicate and fal, what is and is not known about pricing, how hosted output differs from open weights |
