---
name: flux-2
description: >
  Authoritative guide for FLUX.2 by Black Forest Labs (BFL) across all variants and surfaces. Use this
  whenever the user touches FLUX.2 in any way, even obliquely: choosing a variant ([dev] vs [klein] 4B vs
  [klein] 9B vs the API-only [pro]/[max]/[flex]) and understanding the licence split (Non-Commercial dev/9B
  weights, Apache 2.0 klein-4B weights, commercially-usable API outputs). Installing in ComfyUI: the new
  Flux2Scheduler, EmptyFlux2LatentImage, CLIPLoader type "flux2", FluxGuidance+BasicGuider vs CFGGuider, and
  the exact file names and folders from the official templates. Writing or fixing prompts: the Mistral-24B
  and Qwen3 LLM encoders reward the four-part Subject→Action→Style→Context structure rather than tag lists;
  hex color control; JSON for production consistency; no negative prompts, so phrase constraints positively.
  Getting photoreal results: camera gear stacking, and avoiding the over-AI'd over-sharpened look on klein.
  Multi-reference image editing (ReferenceLatent nodes, up to ~8–10 references). Pose control (Alibaba PAI
  Fun Union ControlNet, covering Pose, Depth, Canny, HED and more, and why Flux.1 ControlNets are
  incompatible). Face identity preservation (PuLID via iFayens nodes; InsightFace + EVA-CLIP pipeline).
  Calling the BFL hosted API (flux-2-pro, flux-2-max, flux-2-flex, flux-2-klein endpoints, async polling
  pattern, pricing). Setting up diffusers (Flux2Pipeline, Flux2KleinPipeline, Flux2KleinKVPipeline).
  Training a LoRA with AI-Toolkit/Kohya, including style LoRAs (rank ablations, prose captions, the captionless debate, XY-grid
  evaluation). Creating a consistent character (the ReferenceLatent multi-reference character engine, PuLID,
  the character-LoRA dataset factory and detailer deployment, multi-character scenes via attention masking).
  Building production pipelines (refine passes, tiled upscale for DiTs, Klein as the img2img refiner for
  SDXL renders in mixed-model workflows). Debugging spatial logic failures, missing elements,
  over-sharpening, wrong hex colours, or text rendering issues. Use this for any question about FLUX.2 in
  any context. Choosing between models, comparing them, or working out which skills and install commands a
  job needs is [`generative-media-atlas`](../generative-media-atlas/)'s job. Start there when the model is
  not already settled.
---

# FLUX.2

FLUX.2 is Black Forest Labs' (BFL) second-generation text-to-image family. It shipped in two waves: the **32B flagship [dev]** model on 25 November 2025, and the lightweight **[klein]** sub-family (4B and 9B) on 15 January 2026. [dev] is a **32B rectified-flow MM-DiT** (multimodal diffusion transformer), with 8 double-stream and 48 single-stream parallel transformer blocks. It uses a single **Mistral Small 3.2 24B** vision-language model as its text encoder, replacing Flux.1's dual T5+CLIP. The [klein] family shrinks to 4B and 9B with **Qwen3** encoders, distilled to 4-step inference. All variants share a new **`AutoencoderKLFlux2` VAE** (Apache 2.0, a full retrain). Resolution: 256–2048 px, multiples of 16, up to ~4 MP.

The defining trait of FLUX.2 is that **Mistral 3.2 and Qwen3 are instruction-following LLMs**. They parse clause structure, word order, and semantic context. The model does not want tags; it wants a sentence. FLUX.2 was trained on natural language, so JSON is an *optional* production tool, not a schema requirement.

**Two surfaces:** open weights (ComfyUI / diffusers / CLI, on your GPU or rented cloud) and BFL's hosted API (`api.bfl.ai`). The licence decides which variants you can use commercially, and that split is the most important axis in this skill.

> **A `../link/` on this page that doesn't resolve is a skill you have not installed, not a broken
> page.** [`generative-media-atlas`](../generative-media-atlas/) is the map of this suite: which
> model fits a job, which skills that job needs, and the commands to install them. It works on its
> own, so it is the one to add first — `npx skills add ryannel/skills --skill generative-media-atlas`

---

## Variant selector

| Variant | Params | Text encoder | Distilled | Licence | VRAM (FP8/quant) | Open weights? | Use when… |
|---|---|---|---|---|---|---|---|
| **[dev]** | 32B | Mistral Small 3.2 24B | guidance-distilled | **Non-Commercial** | ~20 GB (fp8); 80 GB+ full | Yes (gated) | Highest quality; research / personal / non-production; LoRA training base |
| **[klein] 4B** | 4B | Qwen3 4B | step+guidance-distilled | **Apache 2.0 ✓** | ~8 GB (fp8) / ~13 GB (fp16) | Yes | Commercial work on open weights; rapid iteration; consumer GPU |
| **[klein] 4B Base** | 4B | Qwen3 4B | no (20-step) | **Apache 2.0 ✓** | ~13 GB | Yes | Fine-tuning / LoRA training with commercial rights |
| **[klein] 9B** | 9B | Qwen3 8B | step+guidance-distilled | Non-Commercial | ~14–16 GB (fp8) | Yes (gated) | Better quality than 4B; non-commercial |
| **[klein] 9B KV** | 9B | Qwen3 8B | step+guidance-distilled | Non-Commercial | ~14–16 GB (fp8) | Yes (gated) | Multi-reference editing with KV-caching for repeated reference tokens |
| **[klein] 9B Base** | 9B | Qwen3 8B | no (20-step) | Non-Commercial | ~24+ GB | Yes (gated) | Fine-tuning / LoRA training |
| **[pro] / [max] / [flex]** | Closed | — | — | API-only | — | No | Commercial production; [max] adds web grounding; [flex] exposes steps/guidance |

> **[klein] 4B is the only Apache-2.0 open-weight model in the family.** It is the only variant you can run locally and use for commercial purposes without a separate BFL licence. [dev] and all [klein] 9B variants follow the FLUX Non-Commercial License v2.0 wherever you run them. Your GPU, RunPod, or any cloud does not change that.

---

## The one rule that changes everything

Mistral 3.2 24B ([dev]) and Qwen3 ([klein]) are full instruction-following LLMs. They parse **clause structure, word order, and semantic context**, the same way any language model would. Quality-tag chains (`masterpiece, 8k, best quality, ultra-realistic, highly detailed`) are Stable Diffusion 1.5 habits, and the encoder reads them as near-zero-signal noise. This is a property of the encoder class, not folklore: the same sentence rule governs Z-Image's Qwen-3. The opposite holds for CLIP models like SDXL, which want weighted tags and verbatim trigger tokens. Prompt dialect, LoRA triggers, and training captions all follow the encoder **here**.

**One refinement makes this rule portable.** The encoder class sets the *ceiling*: what a dialect can express at all. The **training caption corpus** sets what the model is actually fluent in. FLUX.2 was captioned in natural language, so the two line up, and "follow the encoder" is exactly right for this model. Elsewhere in the suite the two come apart: [`anima`](../anima/) has an LLM encoder but wants weighted Danbooru tags, and [`ideogram-4`](../ideogram-4/) has one but wants JSON. Keep the encoder rule as your default, because it is right far more often than not. But **check what a model was captioned on before you infer its dialect from its encoder name.**

**Write a sentence. Use BFL's official four-part structure: Subject → Action → Style → Context**

| Don't | Do |
|---|---|
| `masterpiece, 8k, woman, professional photo, cinematic lighting, photorealistic` | *A woman in her early 30s with short-cropped silver hair sits at a wet café counter, hand wrapped around an espresso cup, staring past the camera. Shot on a Hasselblad X2D, 80mm f/2.8, available window light, rain-grey afternoon.* |

**Sweet spot: 30–80 words.** For quick concept sketches, 10–30 is fine. Above 80 words adds nuance. The hard cap is 512 tokens, well above typical prompts. Front-load the subject and key details, because Mistral weights earlier tokens more heavily.

**No negative prompts. Phrase constraints positively.** FLUX.2 uses rectified flow. [dev] bakes guidance into a single forward pass via `FluxGuidance=4` + `BasicGuider`, so no CFG path exists. [klein] distilled runs `CFGGuider` at CFG=**1**, which is guidance-off. In a ComfyUI KSampler, guidance-off is 1, never 0. Neither variant applies negative conditioning. Instead of "no motion blur", write "sharp, still frame". Instead of "no text", write "clean background, no signage".

**Hex color control.** This is a FLUX.2 capability absent from Flux.1. Signal hex with the word "color" or "hex" before the code:
- `"An apple in color #0047AB"`, not `"An apple #0047AB"`
- `"Logo text 'ACME' in color #FF5733 on a white background"`

**JSON for production (optional, not required).** For multi-subject scenes needing brand consistency and automation, BFL's official format:
```json
{
  "scene": "brief overall description",
  "subjects": [{"description": "detailed subject", "position": "location in frame"}],
  "style": "photographic spec or artistic style",
  "color_palette": ["#hex1", "#hex2"],
  "lighting": "source, direction, quality",
  "camera": {"angle": "bird's eye / eye level / low", "lens": "85mm f/1.4"}
}
```
FLUX.2 treats JSON as a production convenience for repeatability, not a schema it was optimised for. Plain natural language is the primary training distribution.

Full prompt anatomy, drop-in templates, and multi-reference editing: **`references/prompting-guide.md`**.

---

## Setup & ecosystem

FLUX.2 runs in **ComfyUI core**, with no custom nodes needed for the standard templates. One thing matters most: FLUX.2 introduced new node names. Using the wrong Flux.1 nodes silently produces degraded output, so always start from the official templates.

### FLUX.2 [dev] — text-to-image file layout

Source: `Comfy-Org/workflow_templates/image_flux2_text_to_image.json` (read verbatim from raw JSON):

| File | ComfyUI folder | Loader node |
|---|---|---|
| `flux2_dev_fp8mixed.safetensors` | `models/diffusion_models/` | `UNETLoader` |
| `mistral_3_small_flux2_bf16.safetensors` | `models/text_encoders/` | `CLIPLoader` (type **`flux2`**) |
| `full_encoder_small_decoder.safetensors` | `models/vae/` | `VAELoader` |
| `Flux_2-Turbo-LoRA_comfyui.safetensors` *(optional)* | `models/loras/` | `LoraLoaderModelOnly` |

All four files download from `Comfy-Org/flux2-dev` on Hugging Face. The [dev] **image-edit** template swaps the VAE to `flux2-vae.safetensors` and the text encoder to `mistral_3_small_flux2_fp8.safetensors`, and adds `ReferenceLatent` nodes. Details: **`references/setup-and-workflows.md`**.

**Stock node settings** from `image_flux2_text_to_image.json`:

| Node | Setting | Value |
|---|---|---|
| `Flux2Scheduler` | steps, width, height | **20**, 1024, 1024 |
| `FluxGuidance` | guidance | **4** |
| `KSamplerSelect` | sampler | **`euler`** |
| `EmptyFlux2LatentImage` | width × height | 1024 × 1024 |
| `ComfySwitchNode` | turbo mode | **off** (8-step via turbo LoRA when on) |

### FLUX.2 [klein] 4B — file layout

Source: `Comfy-Org/workflow_templates/image_flux2_klein_text_to_image.json` (raw JSON):

| File | ComfyUI folder | Loader node |
|---|---|---|
| `flux-2-klein-4b.safetensors` (distilled) *or* `flux-2-klein-base-4b.safetensors` (base) | `models/diffusion_models/` | `UNETLoader` |
| `qwen_3_4b.safetensors` | `models/text_encoders/` | `CLIPLoader` (type **`flux2`**) |
| `flux2-vae.safetensors` | `models/vae/` | `VAELoader` |

Downloads from `Comfy-Org/flux2-klein`. **Stock settings for distilled:** steps **4**, `CFGGuider` **1**. **For base:** steps **20**, `CFGGuider` **5**. Sampler: `euler` / `Flux2Scheduler` for both variants.

> Klein 9B uses `qwen_3_8b_fp8mixed.safetensors` + `full_encoder_small_decoder.safetensors`. Details and the 9B KV template: **`references/setup-and-workflows.md`**.

### Key ComfyUI node changes from Flux.1

| Component | Flux.1 node (wrong for FLUX.2) | FLUX.2 node (correct) |
|---|---|---|
| Scheduler | `FluxScheduler` | **`Flux2Scheduler`** |
| Latent | `EmptyLatentImage` / `EmptySD3LatentImage` | **`EmptyFlux2LatentImage`** |
| CLIP type | `"flux"` | **`"flux2"`** |
| Guider ([dev]) | — | **`FluxGuidance` + `BasicGuider`** |
| Guider (klein base) | — | **`CFGGuider`** |
| Image edit | — | **`ReferenceLatent`** |

Mixing old and new nodes still compiles, but it produces degraded results, and no error message warns you.

### Quantisation

**Official** (BFL / Comfy-Org HF repos):
- [dev] `flux2_dev_fp8mixed.safetensors`: 35.5 GB, ~20 GB VRAM, in `Comfy-Org/flux2-dev`.
- [dev] fp4 text encoder `mistral_3_small_flux2_fp4_mixed.safetensors`: 12.3 GB, same repo.
- [dev] NVFP4 weights: `black-forest-labs/FLUX.2-dev-NVFP4` (~2.7× faster, ~55% less VRAM vs bf16).
- [klein] 4B FP8: `black-forest-labs/FLUX.2-klein-4b-fp8` (~8 GB).
- [klein] 9B FP8: `flux-2-klein-base-9b-fp8.safetensors` / `flux-2-klein-9b-fp8.safetensors` (~14–16 GB).

**Community.** GGUF quants of [dev] run from Q2_K (~13 GB) to Q8_0 (~35 GB) `[community — city96, unsloth; re-verify]`. This requires the `city96/ComfyUI-GGUF` custom node. GGUF files go in `models/unet/`, not `models/diffusion_models/`.

### Pose control (ControlNet)

FLUX.2's 8-double-stream block architecture is incompatible with Flux.1 ControlNets. The only FLUX.2-native ControlNet is **Alibaba PAI's Fun Union**: `FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors` (~8.3 GB, `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` on HF). It supports Pose (DWPose), Depth, Canny, HED, Scribble, and more through a single Union model.

Install to `models/model_patches/`. It requires custom ComfyUI nodes: either the official VideoX-Fun nodes (`github.com/aigc-apps/VideoX-Fun/comfyui/flux2`) or the community `bryanmcguire/comfyui-flux2fun-controlnet` pack. Full node names, parameters, and preprocessor recommendations: **`references/controlnet-and-identity.md`**.

### Face identity (PuLID)

**`iFayens/ComfyUI-PuLID-Flux2`** (`github.com/iFayens/ComfyUI-PuLID-Flux2`) is the only FLUX.2-specific PuLID implementation. It needs three things. PuLID weights from `Fayens/Pulid-Flux2` go in `models/pulid/`. AntelopeV2 ONNX files go in `models/insightface/models/antelopev2/`. EVA-CLIP auto-downloads. It supports all FLUX.2 variants (the weights are natively klein-trained). Strength 1.0–1.4 recommended `[official — iFayens/ComfyUI-PuLID-Flux2]`.

No FLUX.2-native IP-Adapter face model exists. Full setup, node wiring, and a PuLID vs LoRA comparison: **`references/controlnet-and-identity.md`**. **Building a consistent character** is covered in **`references/characters.md`**: choosing between multi-reference, PuLID, and the character-LoRA pipeline, plus the dataset factory and multi-character scenes.

### diffusers

`Flux2Pipeline` ([dev]) and `Flux2KleinPipeline` ([klein] 9B base) are the primary classes. diffusers **0.40.0** is a stable PyPI release (2026-08-20, verified 2026-09-09), so a plain `pip install "diffusers>=0.40"` is enough. The git-install path is no longer needed `[official — diffusers 0.40.0 docs]`.

0.40.0 also adds **`Flux2KleinKVPipeline`**, the diffusers counterpart of the ComfyUI 9B KV template. On the first denoising step it runs the reference-image tokens through the forward pass and caches their attention K/V projections. Later steps reuse the cache instead of recomputing it. Use it for repeated multi-reference editing against a fixed reference bundle. For a single generation it behaves like `Flux2KleinPipeline` `[official — diffusers 0.40.0 docs]`.

```python
from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
image = pipe("A woman in her early 30s with short-cropped silver hair...",
             num_inference_steps=50, guidance_scale=4.0, height=1024, width=1024).images[0]
```

Detailed diffusers params, hardware options (cpu_offload, group offloading, 4-bit quant): **`references/setup-and-workflows.md`**.

---

## Per-variant settings

### [dev] — 32B guidance-distilled

- **Steps:** 20 in the official ComfyUI template. Use 28–50 for highest quality in diffusers.
- **Guidance:** `FluxGuidance = 4`. It is baked into the forward pass, not classifier-free guidance.
- **Sampler / scheduler:** `euler` / `Flux2Scheduler`.
- **Resolution:** 1024×1024 default, up to ~4 MP (2048×2048 max).
- **Negative prompts:** none. The model is guidance-distilled with no CFG path. Phrase positively.
- **Turbo LoRA:** `Flux_2-Turbo-LoRA_comfyui.safetensors` reduces to **8 steps** (guidance stays at 4). Toggle it via `ComfySwitchNode`.

Distilled and base are separate checkpoints with separate numbers. Mixing them up is the most common klein mistake: a base checkpoint at 4 steps is mush, and a distilled one at 20 steps burns edges `[community]`. That is why they get separate blocks here.

### [klein] 4B — distilled (Apache 2.0)

- **Steps:** 4. This is the number the distillation was trained for, not a floor to raise.
- **Guidance:** `CFGGuider` **1** (guidance-off).
- **Sampler / scheduler:** `euler` / `Flux2Scheduler`.
- **Resolution:** 1024×1024 default, with the same 256–2048 px / multiple-of-16 range as [dev].
- **Negative prompts:** CFG=1 is guidance-off, so negatives are inert. Never type 0.0 in KSampler.
- **Not a training base.** Distillation removes the texture diversity that fine-tuning needs. The same rule holds across Z-Image and Krea 2: train on the undistilled variant `[community — convergent]`.

### [klein] 4B Base — undistilled (Apache 2.0)

- **Steps:** 20.
- **Guidance:** `CFGGuider` **5**.
- **Sampler / scheduler:** `euler` / `Flux2Scheduler`.
- **Resolution:** as above.
- **Negative prompts:** a real CFG path exists here, but FLUX.2's prompting doctrine still says phrase positively. The encoder is an LLM, not a tag matcher.
- **This is the LoRA-training target** for commercially-deployable work (Apache 2.0 weights).

### [klein] 9B — distilled / 9B KV

- **Steps / guidance:** 4 steps, `CFGGuider` **1**. These are the same as 4B distilled.
- **Sampler / scheduler:** `euler` / `Flux2Scheduler`.
- Better quality than 4B on skin and fine detail, but still susceptible to over-sharpening `[community — re-verify]`.
- **[klein] 9B KV:** the same distilled settings, but use it for repeated multi-reference editing. It KV-caches reference tokens, so a fixed reference bundle re-renders faster across many prompts. In diffusers the same path is `Flux2KleinKVPipeline` (0.40.0+).

### [klein] 9B Base — undistilled

- **Steps / guidance:** 20 steps, `CFGGuider` **5**.
- The Non-Commercial training base. Pick it over 4B Base only when quality outweighs losing commercial rights.

---

## Realism — the FLUX.2 approach

FLUX.2's default rendering, especially on [klein], skews toward over-processed, "over-AI'd" sharpness: too-perfect skin, slightly synthetic hair, over-saturated edges. The fix is **camera gear stacking**. It works because Mistral/Qwen3 treat camera vocabulary as semantic context about the type of image being produced.

Stack two or three:
1. **Real camera body:** "Shot on a Hasselblad X2D" / "Sony A7R IV" / "Canon EOS R5".
2. **Lens + aperture:** "80mm f/2.8" / "85mm f/1.4" / "35mm f/1.8".
3. **Film stock or sensor emulation:** "Kodak Portra 400" / "Fujifilm Pro 400H" / light grain.

"Realistic", "high quality", "8K" are legacy booru tokens, and they add nothing here. One non-idealised human feature (visible pores, a freckle, slight under-eye shadow) breaks the perfection signal. The over-AI'd look is worst on [klein] 4B distilled, because guidance-off at 4 steps leaves the least room to argue with the model's prior. [dev] is more measured. The camera/lens/film-stock vocabulary, and the judgement about where the look bites hardest, both come from the same body of community practice that the reference file tabulates `[community — fal.ai prompting guide; convergent]`.

---

## Production pipelines & mixing models

FLUX.2 generates 1–2 MP natively, so it does not need the classic SD pattern of generating low-res first and upscaling everything. The production ladder that fits it:

The denoise bands in stages 2–4 are convergent ComfyUI-author practice, not anything BFL publishes. They are the numbers most worth re-testing against your own base gen `[community — re-verify]`.

1. **Base gen** at 1024²–4 MP ([dev] 20 steps, or [klein] 4-step for drafts). Judge composition, then reroll.
2. **Refine pass** (optional): img2img on itself at denoise ~0.3–0.45 for detail without re-composition.
3. **Detailers:** Impact Pack FaceDetailer is model-agnostic and runs FLUX.2 fine. Use denoise ~0.4. The character-LoRA swap happens here (`references/characters.md §3`).
4. **Tiled upscale:** for DiT models prefer **TTP Toolset**. It captions each tile for per-tile conditioning, which is the anti-hallucination trick. The alternative is `UltimateSDUpscale` at low denoise (~0.2–0.3) with a simplified prompt `[community — convergent]`.
5. **Finish:** ColorMatch against the pre-upscale image, then a SeedVR2-class restorer for the final push.

**FLUX.2's roles in mixed-model pipelines.** Each pattern below names its author `[community]`:
- **Quality refiner:** [klein] img2img over SDXL/Pony renders for "more natural rendering, better anatomical consistency, reduced SDXL artifacts" (Enzino's Flux Klein IMG2IMG workflow, Civitai). Denoise ~0.25–0.4, and **[klein] 4B's Apache licence makes this the commercially-clean refine path**.
- **Composition front-end:** [dev]'s prompt comprehension builds the scene, then an SDXL photoreal finetune img2img pass (~0.3–0.55) adds its texture character afterward. This pattern has no single canonical author behind it `[community — re-verify]`.

The handoff rule is simple: **always VAE-decode to pixels between model families**. FLUX.2's VAE is its own, and [dev] and [klein] 4B even use different VAE files internally, so foreign latents produce garbage `[community — convergent]`. Full cross-model craft (denoise bands, resolution matching, color management, workflows-as-code): the [`image-production-workflows`](../image-production-workflows/) skill.

---

## Failure modes & QC

| Symptom | Cause | Fix |
|---|---|---|
| Key element missing or underweighted | Mid-prompt placement; Mistral/Qwen3 front-weight | Front-load subject; use 4-part structure; keep prompt ≤80 words |
| Spatial logic failure, overlapping subjects | Multi-constraint prompt overloads flow | One spatial relationship per sentence; use JSON `subjects` with `position` fields |
| Over-sharp / "over-AI'd" skin or hair ([klein]) | Distilled klein's default rendering prior; guidance-off removes diversity | Stack camera body + lens + film stock; one non-idealised feature; switch to [dev] for skin-critical work |
| Negatives ignored | [dev]: no CFG path (guidance-distilled). [klein] distilled: CFG=1 = guidance-off | Phrase all constraints positively in the main prompt |
| CFG=0 typed in KSampler | 0.0 outputs the unconditional and ignores the prompt | Use **1.0** for guidance-off, never 0.0 in ComfyUI KSampler |
| Garbled or missing text in image | Improved vs Flux.1 but text rendering has high variance | Wrap text in quotes; ≤10 words per block; generate 3–5 candidates and select |
| Wrong/ignored hex colour | Missing "color"/"hex" keyword before the code | Write `"...in color #0047AB"`, not bare `#0047AB` |
| Flux.1-style artefacts or degraded output | Wrong node names: `FluxScheduler`, `EmptyLatentImage`, CLIPLoader type `"flux"` | Replace with `Flux2Scheduler`, `EmptyFlux2LatentImage`, type `"flux2"` |
| Over-saturated / edge-burned ([klein] distilled at extra steps) | Distilled model optimised for exactly 4 steps | Keep at 4 steps; use base variant for longer step runs |

---

## Pre-flight checklist

1. Sentence, not a tag list? Subject front-loaded?
2. 30–80 words? (10–30 for quick sketches)
3. All constraints phrased positively — no negative prompt field?
4. [dev] ComfyUI: `Flux2Scheduler`, `EmptyFlux2LatentImage`, CLIPLoader type `"flux2"`, `FluxGuidance` + `BasicGuider`?
5. [klein] distilled: `CFGGuider` at **1** (not 0)?
6. VAE file correct: `full_encoder_small_decoder` (dev t2i / klein 9B) or `flux2-vae` (dev image-edit / klein 4B)?
7. Camera body + lens + film stock named for photoreal subjects?
8. Hex colours signalled with `"in color #XXXXXX"` — not bare hex codes?
9. Commercial use routed right? [klein] 4B (Apache 2.0) for local commercial, the API for [pro]/[flex], and **not** [dev] or [klein] 9B weights for commercial production.
10. Multi-reference: image count within the supported range (~4–10: marketing states 10, the prompting guide states 8 `[contested]`, so verify per model card at time of use)?

---

## Where FLUX.2 sits in the suite

| Job | FLUX.2 | Reach for instead |
|---|---|---|
| Consistent characters | **Strongest no-training path** — native multi-reference (ReferenceLatent) + PuLID; full LoRA pipeline too (`references/characters.md`) | [`sdxl`](../sdxl/) for the deepest adapter toolbox (InstantID/HyperLoRA) and `[SEP]` multi-character routing; [`character-lora-training`](../character-lora-training/) for the dataset and captioning craft that transfers across every model here; [`qwen-image`](../qwen-image/) — the Apache-2.0 open-weights alternative for multi-reference editing and re-shooting a character with no adapter at all, and the dataset factory `references/characters.md` names |
| Style LoRAs | Supported, young ecosystem (`references/lora-training.md`) | [`sdxl`](../sdxl/) for mature recipes and years of accumulated craft |
| In-image typography | Good, high variance | [`ideogram-4`](../ideogram-4/) — the typography leader |
| Structural control | Fun Union ControlNet (custom nodes) | [`sdxl`](../sdxl/) for the most complete, mature control stack |
| Commercial local use | **[klein] 4B is the family's Apache-2.0 path** — the one FLUX.2 variant you may run locally for commercial work without a separate BFL licence | [`z-image`](../z-image/) — Apache-2.0 across its whole family, weights *and* outputs, and the least encumbered licence in the suite; [`sdxl`](../sdxl/) (OpenRAIL++-M) for the mature alternative, though its use-restrictions travel downstream; [`krea-2`](../krea-2/) is free commercial only under $1M revenue |
| Photoreal faces & skin | Strong on [dev] with the camera/lens/film-stock stack; [klein] skews over-sharpened (see *Aesthetic range* below) | [`z-image`](../z-image/) — the suite's owner of faces and skin, and the standard face-pass finisher over a FLUX.2 render |
| Anime / booru illustration | Not its dialect: the encoder was trained on prose and wants sentences, so tag vocabularies land as noise | [`anima`](../anima/), the anime-native base (LLM encoder, booru corpus — the caption-corpus point above, in one model); [`sdxl`](../sdxl/) for the mature Illustrious/Pony finetunes |
| Aesthetic range / anti-AI-look | Default rendering (especially [klein]) skews over-sharpened "AI look" | [`krea-2`](../krea-2/) — tuned *against* the AI look, style-reference system, widest stylistic space (its hosted Large even renders through the FLUX.2 VAE) |
| Mixed-model pipelines | Quality refiner ([klein] img2img) and composition front-end | [`image-production-workflows`](../image-production-workflows/) for the cross-model craft |
| Making it move | Still images only | [`wan-2-2`](../wan-2-2/) — image-to-video. Wan's I2V path is much stronger than its text-to-video, so the still you lock here controls the shot; multi-reference identity work here is the upstream half of a consistent character on screen |
| **Choosing between all of these in the first place** | — this table is one model's view of the suite | [`generative-media-atlas`](../generative-media-atlas/) — the whole suite ranked by job (realism, identity, LoRA trainability, control, licence, video), the elimination ladder that settles most choices, and end-to-end routes across several skills |

---

## FLUX 3 — announced, not available, and not an image model in the usual sense

BFL announced **FLUX 3** on **23 July 2026** `[official — bfl.ai/blog/flux-3]`. It matters to anyone using FLUX.2 today, and it is important not to over-read it.

**What it is.** A *multimodal* foundation model trained jointly on **images, video and audio** in one architecture. BFL calls the approach **Self-Flow**: it aligns multimodal generation and understanding within the same architecture. The framing is explicitly world-modelling rather than image generation. The argument is that each modality is a lossy projection of one underlying reality, and their mutual constraints teach more than any one modality alone. Stated video capabilities: up to **20 seconds with native audio** in a single generation, text-to-video, image-to-video (as animation or as visual reference), and video-to-video carrying a character into a new scene. The list continues with keyframe-to-video, multilingual dialogue, agentic chaining of clips into multi-shot sequences, and strong typography. There is also an **action-prediction** branch (FLUX-mimic, with mimic robotics), which has nothing to do with content creation.

**What you can actually use.** As of this writing, FLUX 3 is **Early Access via API and private weight access only.** FLUX 3 Image had not opened even to early access at announcement. The launch plan does promise **"FLUX 3 Dev"**, open-weight access to a multimodal backbone covering video, audio, image and action. But there is **no date**, and every prior capability is gated behind an early-access phase first. `[pending release]`

**How to read the numbers.** BFL's preference rates are their own, preliminary, and taken while the model was still in development. They show FLUX 3 preferred over Grok Imagine Video in up to 69% of comparisons, and over Kling v3 Pro in 60%. Against Seedance 2.0 and Gemini Omni Flash the rate is 52%, against Runway Gen-4.5 it is 77%, and against Luma Ray 3.2 it is 93%. Treat them as a vendor claim about a moving target.

**What this changes for you today: nothing, except planning.** FLUX.2 [dev] and [klein] remain the open FLUX models. But note the direction: BFL's next generation is a video-and-audio model, the same bet MiniMax made with [`minimax-h3`](../minimax-h3/) and Lightricks made with [`ltx-2-5`](../ltx-2-5/). If FLUX 3 Dev ships as described, it will not slot into this skill. It will need its own skill, and it will land in the *video* half of the suite rather than the image half.

---

## Licence & limitations

| Asset | Licence | Commercial use |
|---|---|---|
| Inference code (`github.com/black-forest-labs/flux2`) | Apache 2.0 | Yes |
| VAE (`AutoencoderKLFlux2`) | Apache 2.0 | Yes |
| **[klein] 4B / 4B Base weights** | **Apache 2.0** | **Yes** |
| [dev] / [klein] 9B / 9B KV / 9B Base weights | FLUX Non-Commercial License v2.0 | No |
| API outputs ([pro], [max], [flex], [dev] via API) | You own your outputs | Yes |

**FLUX Non-Commercial License v2.0** permits personal research, experimentation, hobby, charitable, and internal non-production testing. It prohibits commercial production use, military use, surveillance, biometric processing, training competing models from outputs, and circumventing safety measures. Commercial weight use requires a separate BFL licence. The restriction is about *purpose*, not *place*: running [dev] weights on RunPod does not grant commercial rights.

The [klein] 4B's Apache 2.0 status is a deliberate BFL decision: they made the fastest, most accessible variant the commercially-free one. **For any locally-run commercial workflow, [klein] 4B is the correct path.**

**Safety.** The reference inference repo ships a `SafetyChecker` using the Mistral encoder (threshold 0.85). It is active in the API.

---

## How to read the claims in this skill — two bars, by claim type

This skill holds two kinds of claim to two different standards, because they fail in two different ways.

**Hard facts — must be exact or it breaks.** These are the architecture (32B MM-DiT, 8+48 blocks, Mistral 3.2 24B for [dev], Qwen3 4B/8B for [klein] 4B/9B). They are the licence terms (Apache 2.0 for [klein] 4B, Non-Commercial for [dev]/9B). They include the ComfyUI file layout and stock node settings (verbatim from the template JSON), all node names, and the official quantised filenames and sizes (Comfy-Org HF repos). They also include the no-negative mechanism (guidance-distilled for [dev], CFG=1 for [klein]), the 4-part prompting structure, hex color syntax, and the API model slugs and async polling pattern. **Source of truth is official**: BFL GitHub/model cards/docs, the raw ComfyUI template JSON, and diffusers. A wrong quant filename 404s. A misread licence (NC vs Apache) is a legal problem. **The weights have settled, but the packaging around them has not.** These stay volatile: quant filenames/sizes, the diffusers pipeline API (0.40.0 verified on PyPI 2026-09-09; a later breaking release could rename or re-cut the classes), and ComfyUI template details. All of these are republished independently of the model itself. **Re-verify before relying on them, regardless of who said it.**

**Craft — what actually makes a good image.** This covers the photoreal camera/lighting vocabulary, LoRA weights and stacking, multi-reference editing technique, GGUF VRAM trade-offs, and the ControlNet/PuLID identity tooling. **The authoritative source here is the community**: the ComfyUI workflow authors and the people running [dev]/[klein] daily. For a model this new, *they are often ahead of BFL's own docs*. This is stated with confidence. Ranges and "verify at time of use" flags mark where the community layer is still forming, not where it is untrustworthy. The third-party and community tooling to verify before downloading: the Alibaba PAI **Fun Union ControlNet** repo/node names/filenames, the **iFayens PuLID** weights, and the **bryanmcguire** community nodes.

**One genuinely-unresolved fact:** the multi-reference image count. BFL marketing says **10**, the prompting guide says **8**. This discrepancy is unresolved across official sources. Treat ~8 as the safe working number and test if you need more `[contested]`.

**Facts dated 2026-09-09.** Both release dates are in the intro paragraph. This line dates the *claims*, not the model. What moves fastest, and must be re-verified before you rely on it: BFL API pricing and model slugs, the quantised filenames and their sizes, the diffusers stable version, the third-party LoRA / ControlNet / PuLID tooling, and the ComfyUI template details every settings number above is read from.

---

## Reference files

| File | When to read it |
|---|---|
| `references/prompting-guide.md` | Full 4-part prompt anatomy; hex color control with examples; JSON production format; camera vocabulary for photoreal; multi-reference image editing; typography/text-in-image guidance; drop-in templates |
| `references/api-and-hosted.md` | BFL hosted API: endpoints (global/EU/US), auth, model slugs, parameters, async polling pattern, pricing note, API model capability comparison ([pro] vs [max] vs [flex] vs [klein]) |
| `references/setup-and-workflows.md` | All ComfyUI templates (dev image-edit / klein 9B / klein 9B KV); full diffusers setup (`Flux2Pipeline` / `Flux2KleinPipeline` / `Flux2KleinKVPipeline`, 0.40.0+) and VRAM table; GGUF setup (city96 loader); **§7 Using LoRAs** (`LoraLoaderModelOnly` model-only loading, the [dev]↔[klein] variant-incompatibility rule, weight ranges, why FLUX.2 dislikes trigger words, stacking, the Turbo accel-LoRA); multi-reference workflow patterns |
| `references/lora-training.md` | **Making** a LoRA (using is setup-and-workflows §7): training bases (base-not-distilled; [klein] 4B for commercial rights), AI-Toolkit YAML, the Civitai klein recipe and its dim-2 floor warning, hyperparameters by target (Herbst's style ablation), prose caption-the-residual and the **contested captionless debate**, style-LoRA specifics (diversity maxim, color-cast lock-in, acceptance test), XY-grid evaluation |
| `references/characters.md` | Creating a **consistent character**: the path decision (multi-reference vs PuLID vs character LoRA and how they compose), **ReferenceLatent as the character engine** (reference bundles, KV batching), the dataset factory (generate ~60 → curate ~30, 8-point rotation), the detailer LoRA swap, multi-character scenes via core attention masking, failure modes |
| `references/controlnet-and-identity.md` | Pose control: Alibaba PAI Fun Union ControlNet (why Flux.1 won't work, model files, ComfyUI nodes, per-type strength settings, preprocessors); face identity: PuLID setup (files, nodes, dependencies, integration); IP-Adapter status; ReferenceLatent native tool; choosing between ControlNet vs PuLID vs ReferenceLatent |
