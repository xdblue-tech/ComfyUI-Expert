# FLUX.2 — Setup & Workflows Reference

Source tier: the ComfyUI template JSONs are primary sources, read verbatim. The Comfy-Org Hugging Face repos are also primary. The diffusers docs and the HF blog are official-via-docs. The community GGUF repos and the Civitai training guide are community-tier and labelled as such.

---

## Contents

1. [VRAM requirements table](#1-vram-requirements-table)
2. [ComfyUI — [dev] image-edit template](#2-comfyui--dev-image-edit-template)
3. [ComfyUI — [klein] 9B templates](#3-comfyui--klein-9b-templates)
4. [ComfyUI — [klein] 9B KV template](#4-comfyui--klein-9b-kv-template)
5. [ComfyUI — GGUF quants (community)](#5-comfyui--gguf-quants-community)
6. [diffusers — detailed setup](#6-diffusers--detailed-setup)
7. [Using LoRAs](#7-using-loras)
8. [LoRA training → moved to `references/lora-training.md`](#8-lora-training--referenceslora-trainingmd)

---

## 1. VRAM requirements table

Values are approximate. They depend on resolution (1024×1024 unless noted) and batch size 1. Source: HF blog (official-via-docs) + deepwiki architecture analysis (community).

| Config | VRAM (approx) | Notes |
|---|---|---|
| [dev] bf16 full | ~80 GB | Multi-GPU (2× A100/H100) or offload |
| [dev] fp8 mixed (`flux2_dev_fp8mixed.safetensors`) | ~20 GB | Fits single A100/H100 40GB or 3090/4090 24GB |
| [dev] NVFP4 (`FLUX.2-dev-NVFP4`) | ~11 GB | Requires Ada/Hopper GPU (3090+ or 4090); ~2.7× faster, ~55% less VRAM vs bf16 |
| [dev] GGUF Q4_K_M (community) | ~16 GB | `city96/ComfyUI-GGUF` required |
| [dev] GGUF Q2_K (community) | ~13 GB | Quality degradation vs fp8 |
| [dev] text encoder bf16 (Mistral 3.2) | ~35.6 GB | Large — use fp8 (18 GB) or fp4 (12.3 GB) in practice |
| [dev] text encoder fp8 | ~18 GB | Default for most Comfy setups |
| [dev] text encoder fp4 | ~12.3 GB | Lowest quality reduction, highest compression |
| [klein] 4B fp16 | ~13 GB | — |
| [klein] 4B fp8 | ~8 GB | Fits RTX 3080/4070Ti |
| [klein] 9B fp8 | ~14–16 GB | Fits 3090/4090 |
| [klein] 9B bf16 full | ~32+ GB | Needs offload or 40 GB+ |

**GGUF note (community-tier):** Exact sizes change as new quants ship. Check `city96/FLUX.2-dev-gguf` and `unsloth/FLUX.2-dev-GGUF` on Hugging Face before you plan storage.

---

## 2. ComfyUI — [dev] image-edit template

Source: `Comfy-Org/workflow_templates/image_flux2_image_editing.json` (verbatim).

**File layout:**

| File | Folder | Loader | Notes |
|---|---|---|---|
| `flux2_dev_fp8mixed.safetensors` | `models/diffusion_models/` | `UNETLoader` | Same as t2i template |
| `mistral_3_small_flux2_fp8.safetensors` | `models/text_encoders/` | `CLIPLoader` type `"flux2"` | fp8 (not bf16) in edit template |
| `flux2-vae.safetensors` | `models/vae/` | `VAELoader` | Full encoder — different from t2i's `full_encoder_small_decoder` |
| `Flux_2-Turbo-LoRA_comfyui.safetensors` *(optional)* | `models/loras/` | `LoraLoaderModelOnly` | Same turbo LoRA as t2i |

**Key nodes added for image editing:**
- `LoadImage` feeds into `VAEEncode`, which encodes the reference image to latent space
- `ReferenceLatent` is a new FLUX.2 node. It takes a latent plus an optional mask, and outputs a reference conditioning token
- Multiple `ReferenceLatent` nodes are wired in sequence. Each one accepts one reference image
- `BasicGuider` and `FluxGuidance=4` stay unchanged. The reference latents are injected into the conditioning stream, not into the guider

**Stock settings (image-edit template defaults):**

| Setting | Value |
|---|---|
| Steps | 20 |
| FluxGuidance | 4 |
| Sampler | euler |
| Scheduler | Flux2Scheduler |
| Edit strength (noise) | Controlled by denoising start/end sliders |

**Practical tip:** Start with denoising strength 0.7–0.85 to keep the content while changing style. Use 0.9–1.0 for near-full regeneration, with the reference as a composition seed.

---

## 3. ComfyUI — [klein] 9B templates

Source: `Comfy-Org/workflow_templates/image_flux2_text_to_image_9b.json` (verbatim).

**File layout:**

| File | Folder | Loader |
|---|---|---|
| `flux-2-klein-base-9b-fp8.safetensors` (base) or `flux-2-klein-9b-fp8.safetensors` (distilled) | `models/diffusion_models/` | `UNETLoader` |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` | `CLIPLoader` type `"flux2"` |
| `full_encoder_small_decoder.safetensors` | `models/vae/` | `VAELoader` |

Downloads from `Comfy-Org/flux2-klein-9B` on Hugging Face.

**Stock settings:**

| Variant | Steps | Guider | CFG | Notes |
|---|---|---|---|---|
| 9B distilled | 4 | CFGGuider | 1 | guidance-off; sampler euler |
| 9B base | 20 | CFGGuider | 5 | sampler euler |

**Note on 9B vs 4B VAE:** 9B uses `full_encoder_small_decoder.safetensors` (same as [dev] t2i). 4B uses `flux2-vae.safetensors`. Do not mix them. The two VAEs have different architectures.

---

## 4. ComfyUI — [klein] 9B KV template

Source: `Comfy-Org/workflow_templates/image_flux2_klein_9b_kv*.json` (verbatim, exact filename may vary — check the Comfy-Org templates repo for current name).

**Purpose:** KV-caching variant. It caches the reference image's key-value attention states, so repeated inference with the same set of reference images runs much faster. The reference encoding is computed once and reused. This matters most for multi-reference workflows with many reference images.

**File layout:** Same files as the standard 9B template. No additional model files.

**Additional nodes:**
- `Flux2KleinKVCache` node — caches reference latent K/V states
- Standard `ReferenceLatent` nodes for the actual reference images

**When to use KV vs standard 9B:**
- Use KV when you run batch jobs with the same reference image set and varying prompts. The first call pays the reference encoding cost, and later calls reuse the cache
- For single-shot generation with one reference, standard 9B is equivalent
- KV caching gives roughly 1.5–3× speedup on repeated-reference batches (community-tier estimate)

---

## 5. ComfyUI — GGUF quants (community)

**Required custom node:** Install `city96/ComfyUI-GGUF` via ComfyUI Manager before using GGUF models.

**Important path difference:** GGUF `.gguf` files go in `models/unet/`, not `models/diffusion_models/`. The `UnetLoaderGGUF` node (from the custom node) reads from `models/unet/`.

**Repositories (community-tier):**
- `city96/FLUX.2-dev-gguf` produces Q2_K through Q8_0 quants
- `unsloth/FLUX.2-dev-GGUF` is an alternative quant series, and may include iQ (importance-weighted) variants

**File naming convention:** `flux2-dev-Q4_K_M.gguf`. This is approximate, so verify exact filenames in the HF repo before downloading.

**Node swap from official templates:**

| Official template node | GGUF equivalent |
|---|---|
| `UNETLoader` with `.safetensors` | `UnetLoaderGGUF` with `.gguf` |
| Everything else | Unchanged |

Text encoder and VAE files stay `.safetensors`. GGUF quantisation applies only to the UNet/DiT; the text encoder and VAE keep running at their usual precision.

**Quality expectation:** Q8_0 is close to fp8 quality. Q4_K_M gives a mild quality reduction and works well for most subjects. Q2_K shows noticeable degradation on fine detail and faces. For portrait and skin-heavy work, use Q4_K_M or higher.

---

## 6. diffusers — detailed setup

**Recommended install:**
```bash
pip install "diffusers>=0.40"
```
diffusers 0.40.0 is a stable PyPI release (2026-08-20, verified 2026-09-09). It carries all three FLUX.2 pipeline classes below, and the official docs link source at the `v0.40.0` tag with no git-install caveat. The older `git+https://github.com/huggingface/diffusers` install is no longer needed `[official — diffusers 0.40.0 docs]`.

**[dev] — `Flux2Pipeline`**

```python
import torch
from diffusers import Flux2Pipeline

# Standard load (requires ~20 GB VRAM)
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# OR: CPU offload (works on <16 GB VRAM, slower)
pipe.enable_model_cpu_offload()

# OR: Group offloading (finer-grained memory management)
# pipe.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="block_level")

image = pipe(
    "A woman in her early 30s with silver-grey cropped hair...",
    num_inference_steps=50,
    guidance_scale=4.0,
    height=1024,
    width=1024,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
image.save("output.jpg")
```

**[klein] 9B base — `Flux2KleinPipeline`**

```python
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

image = pipe(
    "A café counter in morning light...",
    num_inference_steps=20,
    guidance_scale=5.0,
    height=1024,
    width=1024,
).images[0]
```

**[klein] 9B KV-cached — `Flux2KleinKVPipeline`** (diffusers 0.40.0+)

```python
from diffusers import Flux2KleinKVPipeline
# Load exactly as Flux2KleinPipeline but with the KV class
pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B",
    torch_dtype=torch.bfloat16
)
```

This is the diffusers counterpart of the ComfyUI 9B KV template (§4). On the first denoising step the pipeline runs the reference-image tokens through the forward pass and caches their attention K/V projections. Every later step reuses that cache instead of recomputing the references. The saving scales with the number of references and the number of steps, so it matters most for a fixed multi-reference bundle re-rendered across many prompts. For a single one-reference generation it behaves like `Flux2KleinPipeline`. Check the class docstring for the reference-image argument name before you wire it in `[official — diffusers 0.40.0 docs]`.

**[klein] 4B:** At research time, the diffusers docs had no dedicated named pipeline class for the 4B variant. Load it via the same `Flux2KleinPipeline` pattern, or directly from the 4B HF repo — check the model card for the recommended pipeline when you use it. Community reports say `Flux2KleinPipeline` works with the right config changes.

**Hardware minimums for diffusers (community-tier):**

| Config | Min VRAM | Notes |
|---|---|---|
| [dev] fp8 with cpu_offload | 12 GB | Slow; model layers shuttle CPU↔GPU |
| [dev] fp8 full | 20 GB | RTX 3090/4090 or A100 |
| [klein] 4B fp8 | 8 GB | RTX 3080Ti/4070Ti |
| [klein] 9B fp8 | 14–16 GB | RTX 3090/4090 |

**4-bit quantisation (bitsandbytes, community):**
```python
from transformers import BitsAndBytesConfig
from diffusers import Flux2Pipeline

nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    text_encoder=None,  # load separately to apply bnb config
    transformer=None,
    torch_dtype=torch.bfloat16
)
# See HF bitsandbytes integration docs for full pattern
```

---

## 7. Using LoRAs

This section covers the generic path for any downloaded FLUX.2 LoRA. Training your own is covered in §8.

> **Sourcing:** the loader node and the frozen-encoder fact are verified against the official template and the AI-Toolkit configs. The weight ranges and the claim that "FLUX.2 dislikes trigger words" are community craft from named Flux LoRA trainers (apatero, RunComfy, fal.ai, bghira/SimpleTuner). FLUX.2 is new, so treat the numbers as starting points.

**Node wiring is model-only.** A FLUX.2 LoRA patches the **DiT transformer only**. The text encoders (Mistral 3.2 for [dev], Qwen3 for [klein]) stay **frozen** in both training and inference, so there are no encoder weights to apply. Load it with **`LoraLoaderModelOnly`** on the model path:

```
(model loader) → LoraLoaderModelOnly → Flux2 sampler chain
```

If a LoRA *does* ship text-encoder weights (rare for FLUX.2), switch to the full `LoraLoader` so they apply. Model-only is still the norm.

**LoRAs are variant-specific: a [dev] LoRA does not load on [klein], and vice-versa.** This is unlike Z-Image, where Base and Turbo share one architecture. FLUX.2's variants are **different model sizes**: [dev] is 32B, and [klein] is 4B or 9B. Their LoRAs are therefore **not interchangeable**, so match the LoRA to the exact variant it was trained on. The Klein sizes matter too. **Klein 9B needs the Qwen3-8B encoder**. If you run it against the 4B encoder, it fails outright. Check the LoRA's stated base model before you download it.

**Weight.** Start around **0.8**, and sweep **0.6–1.2**. Go lower for style LoRAs that flatten texture, and higher to force a stubborn concept. Read the LoRA's card for the author's tested weight `[community — consistent with Flux.1 conventions]`.

**Trigger words: FLUX.2 mostly doesn't want them.** The encoder is a full LLM (Mistral/Qwen3), so FLUX.2 reads **natural-language description** far better than bare trigger tokens. Trainers report that trigger words "confuse the model," and that semi-long descriptive captions activate a LoRA best. If a LoRA defines a trigger, include it verbatim. Otherwise, simply *describe* what you want in prose. This is the opposite of tag-based SDXL, where the literal trigger token is mandatory.

**Stacking.** Chain `LoraLoaderModelOnly` nodes (MODEL out → MODEL in), or use the rgthree **Power Lora Loader**. Run **3–4 LoRAs** max `[community]`. Lower each LoRA's strength as you add more, so they don't fight or over-bake. For example, run a character plus a style plus an effect, each around 0.5–0.8. Use `strength_model` to make one dominant.

**The Turbo LoRA** (`Flux_2-Turbo-LoRA_comfyui.safetensors`) is a special case. It is an *acceleration* LoRA that cuts [dev]/[klein] to 8 steps (guidance stays 4), toggled via `ComfySwitchNode`. It stacks with content LoRAs like a speed LoRA, not a style one.

**Ecosystem (early 2026, fast-moving).** Civitai is the main LoRA source. Filter by the **exact FLUX.2 variant** ([dev] vs [klein]), because a Flux.1 LoRA won't load. Most are trained with the **Ostris AI-Toolkit**. BFL published an official "fine-tune [klein] in under 60 min" LoRA guide, and fal.ai, RunComfy, and bghira's SimpleTuner all support FLUX.2 training. The published pool was still small and growing close to release.

---

## 8. LoRA training → `references/lora-training.md`

Training moved to its own file, matching the suite's layout. **`references/lora-training.md`** covers the supported training bases (train on base, never distilled; [klein] 4B Base for commercial rights), the AI-Toolkit YAML, and the Civitai klein recipe, including its dim-2 floor warning. It covers hyperparameters by target — character versus style, based on Herbst's 50+-run ablation. It also covers caption-the-residual in prose, the contested captionless debate, **style-LoRA specifics** (the diversity maxim, color-cast lock-in, and the out-of-set acceptance test), and XY-grid evaluation. The full character pipeline is **`references/characters.md`**. Once you have trained a LoRA, **§7** above covers loading, weights, stacking, and variant compatibility.
