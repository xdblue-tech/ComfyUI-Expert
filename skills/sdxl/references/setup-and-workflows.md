# SDXL Setup & Workflows

All ComfyUI node settings in this file are read from the official `Comfy-Org/workflow_templates` JSONs and from comfyanonymous's `ComfyUI_examples`. The diffusers code comes from the Stability HF model cards. Community items are labelled.

## Contents
1. File layout & the VAE gotcha
2. ComfyUI — base-only graph
3. ComfyUI — base + refiner ensemble
4. ComfyUI — Turbo / Lightning / LCM / Hyper
5. diffusers — t2i, img2img, inpaint, the ensemble
6. Hires-fix & tiled upscale
7. ControlNet & IP-Adapter
8. Quantisation & VRAM

---

## 1. File layout & the VAE gotcha

| File | `models/` folder | Loader node |
|---|---|---|
| `sd_xl_base_1.0.safetensors` (~6.94 GB fp16) | `checkpoints/` | `CheckpointLoaderSimple` |
| `sd_xl_refiner_1.0.safetensors` (~6.08 GB) | `checkpoints/` | `CheckpointLoaderSimple` |
| `sd_xl_turbo_1.0_fp16.safetensors` | `checkpoints/` | `CheckpointLoaderSimple` |
| `sdxl_lightning_{N}step.safetensors` (full) | `checkpoints/` | `CheckpointLoaderSimple` |
| `sdxl_lightning_{N}step_lora.safetensors` | `loras/` | `LoraLoader` |
| `sdxl_vae.safetensors` / fp16-fix | `vae/` | `VAELoader` |
| finetunes (Juggernaut, RealVisXL, Pony…) | `checkpoints/` | `CheckpointLoaderSimple` |

**The checkpoint is one file.** It bundles the UNet, CLIP-L, OpenCLIP-bigG, and the VAE together. `CheckpointLoaderSimple` outputs `MODEL`, `CLIP`, and `VAE`, so the stock graph needs no separate loaders.

**VAE gotcha** `[community; strong]`: SDXL's original VAE **overflows in fp16 and produces black or NaN images**. The baked-in VAE used by the stock templates is fine. If you decode in fp16 with a *standalone* VAE, use **`madebyollin/sdxl-vae-fp16-fix`** via a `VAELoader` wired into `VAEDecode`, or run the VAE in fp32. If you get black outputs, this is almost always the cause.

---

## 2. ComfyUI — base-only graph

```
CheckpointLoaderSimple (sd_xl_base_1.0.safetensors)
  ├─ MODEL ─────────────► KSampler
  ├─ CLIP ──► CLIPTextEncode (positive) ─► KSampler.positive
  │           CLIPTextEncode (negative) ─► KSampler.negative
  └─ VAE ───────────────────────────────► VAEDecode
EmptyLatentImage (1024×1024) ────────────► KSampler.latent
KSampler ─ LATENT ─► VAEDecode ─► SaveImage
```

Stock `KSampler` settings for base are: **steps 25–40, cfg 5–8, sampler `euler`, scheduler `normal`, denoise 1.0**. The simple-example template uses an advanced sampler split because it feeds the refiner. For base-only work, a plain `KSampler` at 30 steps / cfg 7 is the common default.

Use the plain `CLIPTextEncode` node, which sends the same text to both encoders. Swap to `CLIPTextEncodeSDXL` only when you want the `text_g`/`text_l` split (see prompting-guide §3).

---

## 3. ComfyUI — base + refiner ensemble

This graph is taken verbatim from `sdxl_simple_example.json`. It uses two `CheckpointLoaderSimple` nodes (base + refiner), each with its own positive and negative `CLIPTextEncode`, chained through two `KSamplerAdvanced` nodes by latent:

**BASE — `KSamplerAdvanced`:**
`add_noise=enable, noise_seed=<seed>, steps=25, cfg=8, sampler_name=euler, scheduler=normal, start_at_step=0, end_at_step=20, return_with_leftover_noise=enable`

**REFINER — `KSamplerAdvanced`:**
`add_noise=disable, steps=25, cfg=8, sampler_name=euler, scheduler=normal, start_at_step=20, end_at_step=10000, return_with_leftover_noise=disable`

The refiner's `LATENT` output then goes to `VAEDecode` and `SaveImage`.

**Why these exact numbers:** both samplers share the **same 25-step schedule**. The base model runs steps 0→20 (a fraction of **0.8**) and returns its latent *with leftover noise*. The refiner continues from step 20 to the end **without adding fresh noise** (`add_noise=disable`). This is the "ensemble of experts" hand-off: the base builds the structure, and the refiner adds high-frequency detail at low noise. The refiner uses **only the bigG encoder**. The `sdxl_refiner_prompt_example.json` variant gives the refiner its own prompt.

**Is the refiner worth it?** Often not, once you use a good finetune, because the finetune bakes the detail in. Keep the refiner for last-mile sharpness on base SDXL, and skip it on Juggernaut or RealVis.

---

## 4. ComfyUI — Turbo / Lightning / LCM / Hyper

**Turbo** uses a custom-sampler graph, taken verbatim from `sdxlturbo_example.json`:
- `CheckpointLoaderSimple` = `sd_xl_turbo_1.0_fp16.safetensors`
- `KSamplerSelect` = **`euler_ancestral`**
- `SDTurboScheduler` = **steps 1, denoise 1**
- `SamplerCustom` = `add_noise=True, noise_seed=0, **cfg=1**`
- `EmptyLatentImage` = **512×512**, because Turbo is a 512px model. It works at 1–10 steps.

**Lightning** (HF `ByteDance/SDXL-Lightning`):
- **Full checkpoint or UNet:** `CheckpointLoaderSimple`, then standard `KSampler` at **sampler `euler`, scheduler `sgm_uniform`, cfg 1, steps = the checkpoint's step count (1/2/4/8), denoise 1**, 1024².
- **LoRA:** load a base-SDXL *finetune*, add a `LoraLoader` with `sdxl_lightning_{N}step_lora.safetensors` at strength 1.0, and use the same sampler settings. Use the LoRA route when you want to make a *custom finetune* fast. Use the full checkpoint for best quality on plain SDXL. **Match the step count to the file**: a 4-step LoRA run at 8 steps degrades. 1-step is experimental; 2-step is the floor and 4-step is the default.

**LCM** (comfyanonymous LCM examples):
- Patch the model with **`ModelSamplingDiscrete`** set to **`lcm`**, then run a `KSampler` at **sampler `lcm`, scheduler `sgm_uniform`, cfg 1–2, steps 4–8**, at 1024². **LCM-LoRA** (via `LoraLoader`) applies the same settings to any finetune. Forgetting the `ModelSamplingDiscrete` patch, or leaving CFG high, "blows things up" `[community]`.

**Hyper-SDXL** (ByteDance): runs as a LoRA or a full checkpoint, at **sampler `euler`, scheduler `sgm_uniform`, cfg ~1, steps 1/2/4/8**. It is the best-rated 1-step option. Some 1-step modes ship a unified-guidance LoRA, so follow the model card.

**Composability (the key workflow):** Lightning, LCM, and Hyper **LoRAs stack onto any photoreal finetune**, which gives you fast and photoreal at the same time. Chain a `LoraLoader` (Lightning/LCM/Hyper) after the finetune checkpoint and set the matching sampler, scheduler, cfg, and steps. You can then draft in 4 steps at Juggernaut quality.

---

## 5. diffusers — t2i, img2img, inpaint, the ensemble

The pipelines are: `StableDiffusionXLPipeline` for t2i, `StableDiffusionXLImg2ImgPipeline` for img2img and the refiner, and `StableDiffusionXLInpaintPipeline` for inpaint. You need **diffusers ≥ 0.19.0.**

**Text-to-image:**
```python
import torch
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
img = pipe(prompt, num_inference_steps=30, guidance_scale=7.0,
           height=1024, width=1024).images[0]
```

**Base + refiner ensemble (80/20 split):**
```python
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
base = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16").to("cuda")
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16, variant="fp16",
    text_encoder_2=base.text_encoder_2, vae=base.vae).to("cuda")

n_steps, high_noise_frac = 40, 0.8
latents = base(prompt=prompt, num_inference_steps=n_steps,
               denoising_end=high_noise_frac, output_type="latent").images
image = refiner(prompt=prompt, num_inference_steps=n_steps,
                denoising_start=high_noise_frac, image=latents).images[0]
```

**Turbo in diffusers:** use `guidance_scale=0.0` and `num_inference_steps=1` (up to 4), at 512×512. **Lightning/LCM/Hyper:** use `guidance_scale=0.0` (for LCM, ~1.0–2.0 via `LCMScheduler`), with steps set to the variant's count. **Remember that ComfyUI `cfg=1` equals diffusers `guidance_scale=0.0`**: both mean guidance is off. Never set `guidance_scale` between 0 and 1 expecting "a little" guidance, because it is effectively off below ~1.

**VAE in diffusers:** if you hit black images in fp16, pass `vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)` to the pipeline.

---

## 6. Hires-fix & tiled upscale

SDXL is 1024-native, so you do not *need* the SD1.5 low-res-first dance. But to exceed ~1.5 MP without duplication or anatomy artefacts, generate in a 1024-area bucket first and then upscale:

- **Latent hires-fix:** `KSampler` → `LatentUpscaleBy` (×1.5, `bislerp`) → second `KSampler` at **denoise 0.3–0.5** → decode. Low denoise preserves the composition while adding detail. There is also a pixel-space variant: decode → `ImageUpscaleWithModel` → re-encode → re-sample at **0.25–0.35**. It tolerates lower denoise than latent interpolation does.
- **Tiled upscale** `[community]`: use `UltimateSDUpscale` (a custom node) with a 4× ESRGAN model and **denoise ~0.2–0.35**, tile size 1024, for large prints. When seams show, use its **seam-fix modes** (half-tile is the usual pick) plus tile overlap. A tile only sees its own patch, so give the upscale pass a *simpler* prompt than the base generation. Otherwise localized details get stamped onto every tile. For a clean, non-hallucinating enlarge, use a model-only upscale (`UpscaleModelLoader` + `ImageUpscaleWithModel`, e.g. `4x-UltraSharp` or `Remacri`) with no re-diffusion.
- **Face/hand repair:** inpaint the region, or use a detailer node (`FaceDetailer` from Impact Pack) at denoise ~0.4. This is the SDXL-era equivalent of SD1.5's aDetailer. Settings that recur across published detailer workflows: `guide_size` 512, `max_size` 1024, and `bbox_crop_factor` ~1.3–2 for tighter face context `[community]`. This is also the stage where a character LoRA gets swapped in (`references/characters.md §3`).
- **Color drift:** VAE round-trips and second samplers shift color. Fix it once at the end with a **ColorMatch** node (KJNodes) against the pre-upscale image, rather than fixing it at every stage.
- The full production ladder — per-stage denoise bands, finishers like SeedVR2, cross-model refine passes — lives in the **`image-production-workflows`** skill in this suite. SDXL's specific role there is the controllable front-end and the texture-refine back-end.

---

## 7. ControlNet & IP-Adapter

Both are mature for SDXL, and that maturity is its biggest practical edge.

**ControlNet** comes from a community model zoo: Stability's official SDXL ControlNets plus the `xinsir`, `diffusers`, and `kohya` releases.
- Load with `ControlNetLoader` → `ControlNetApplyAdvanced` (set `strength`, `start_percent`, `end_percent`) between the conditioning and the sampler. To chain multiple ControlNets, stack Apply nodes with per-CN strength, start, and end.
- Common types: **canny, depth, openpose, tile, scribble, lineart, softedge**. **`xinsir/controlnet-union-sdxl-1.0` (ProMax)** is the consensus best. It packs 10+ control types plus tile, inpaint, and outpaint into one checkpoint. xinsir's further training is stalled for GPU funding, so the project is frozen, but it is stable and still SOTA for SDXL.
- Preprocess the source image with the `comfyui_controlnet_aux` nodes (depth-anything, openpose, canny, etc.).

**IP-Adapter** handles image prompting, style transfer, and identity transfer.
- Use the `ComfyUI_IPAdapter_plus` custom nodes. The variants are: base, **Plus** (more detail), and **FaceID** / **FaceID Plus v2** (identity via InsightFace embeddings).
- The pipeline is `IPAdapterUnifiedLoader` → `IPAdapter` node, with a reference image and a `weight` of ~0.5–0.8. FaceID needs the InsightFace model installed.
- **Maintenance status:** cubiq's `ComfyUI_IPAdapter_plus` went **maintenance-only in April 2025**. Comfy-Org maintains a reference implementation (`comfyorg/comfyui-ipadapter`). Both work, so check which one your other nodes expect.
- For *face identity* specifically, **InstantID** has displaced FaceID as the community go-to. **HyperLoRA** (ByteDance) generates LoRA weights zero-shot from a face photo. The full identity-tool decision table is in **`references/characters.md §1`**.

These tools let SDXL do the pose, structure, and identity control that newer DiT models still lack mature tooling for. That is exactly why mixed-model pipelines compose with SDXL first and refine elsewhere.

---

## 8. Quantisation & VRAM

- **There is no useful GGUF for SDXL.** SDXL is a **Conv2D-heavy UNet**, while GGUF/DiT quantisation targets transformers. The `city96/ComfyUI-GGUF` author says explicitly *don't quantise SDXL*, because quality collapses. fp16 is the format.
- **fp8 weight-casting:** ComfyUI's `--fp8_e4m3fn-unet` flag casts the UNet weights to fp8 to save VRAM, at a small quality cost. The model still computes in higher precision.
- **VRAM** `[community]`: ComfyUI auto-offloads, so a 1024² render runs on **~4 GB** in low-VRAM mode, and **6–8 GB** is comfortable. Base+refiner keeps both checkpoints resident, so budget **8 GB+** for it. Fast variants do not reduce VRAM, because they use the same UNet; they only reduce time. Stability and Comfy publish no single hard minimum.
- **Speed:** the distilled variants (Turbo, Lightning, LCM, Hyper) cut a 30-step render down to 1–8 steps, which is near-real-time on a mid-range GPU. Distillation, not quantisation, is the lever for low-end hardware.
