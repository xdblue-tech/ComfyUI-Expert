---
name: sdxl
description: >
  Authoritative guide for Stable Diffusion XL (SDXL 1.0, Stability AI) and its ecosystem: base + refiner, the
  distilled fast variants (Turbo, Lightning, LCM, Hyper-SDXL), and the community finetunes (Juggernaut,
  RealVisXL, DreamShaper, Pony, Illustrious/NoobAI), in ComfyUI or the diffusers API. Use this whenever the
  user touches SDXL in any way, even obliquely: choosing a checkpoint or fast variant (base vs Turbo vs
  Lightning vs LCM vs Hyper, or which photoreal/anime finetune); installing or setting it up in ComfyUI
  (single-checkpoint loader, the fp16-fix VAE, file layout, base+refiner ensemble graph); writing or fixing
  prompts — SDXL is a dual-CLIP 77-token model, so it wants weighted comma-separated keyword phrases, not LLM
  sentences and not generic "masterpiece 8k" — including matching the prompt dialect to the checkpoint, the
  text_g/text_l split, negative prompts and CFG; getting photoreal results (use a photoreal finetune, then
  stack camera body + film stock + lens + lighting vocabulary); choosing steps/CFG/sampler/scheduler/resolution
  per variant; running ControlNet / IP-Adapter / LoRA; training a LoRA (kohya_ss / OneTrainer) including style
  LoRAs (the Illustrious recipe, dataset diversity, XY-grid evaluation); creating a consistent character
  (InstantID vs IP-Adapter FaceID vs HyperLoRA vs a trained character LoRA, the detailer LoRA swap, ADetailer
  [SEP] multi-character routing, block-weighted LoRA to stop style bleed); building multi-stage production
  pipelines (hires-fix, tiled upscale, detailers); using SDXL as the controllable front-end or texture
  back-end in mixed-model workflows; and debugging artefacts (fried colours, cut-off heads, plastic skin,
  mangled hands, unreadable text). It also covers the licence picture (OpenRAIL++-M is commercially clean;
  Turbo's licence is contested) and when to leave the SDXL family entirely — Anima, the anime-native base
  speaking the same booru dialect but whose *weights* (not outputs) are non-commercial, and Pony V7 (AuraFlow,
  not SDXL). Use this for any question about SDXL in any context. Choosing between models, comparing them, or
  working out which skills and install commands a job needs is
  [`generative-media-atlas`](../generative-media-atlas/)'s job — start there when the model is not already
  settled.
---

# Stable Diffusion XL (SDXL)

SDXL 1.0 is Stability AI's open-weights latent-diffusion model, released July 2023. It is a **convolutional UNet, not a DiT**, with a **2.6B-parameter UNet backbone**. The full **base + refiner "ensemble of experts" pipeline is ~6.6B parameters**. It uses **two fixed CLIP text encoders working together**: **CLIP-ViT/L** ("CLIP-L") and **OpenCLIP-ViT/bigG** ("CLIP-G"). Their penultimate hidden states are concatenated into a 2048-dim conditioning, and the pooled bigG embedding feeds the timestep. Native resolution is **1024×1024**. The licence is **CreativeML Open RAIL++-M**, which permits commercial use.

One trait defines SDXL, and it is also the thing that dates it: **it is a CLIP-conditioned UNet, not an LLM/T5-conditioned transformer.** That gives it a hard **77-token** prompt window. You write keywords and tags, not sentences. It cannot render reliable in-image text, and it understands complex compositions worse than transformer-based models with LLM text encoders do. In exchange, it has the **deepest ecosystem of any open model**: thousands of finetunes, LoRAs, ControlNets, and IP-Adapters. It runs on **6–8 GB VRAM**, and it is fast. You almost never run base SDXL raw; you run a finetune of it.

## Two orthogonal axes — they compose

SDXL choices fall on **two independent axes that stack**:

- **Speed** — base (full quality) versus a distilled fast variant (Turbo / Lightning / LCM / Hyper-SDXL), which trades steps for seconds.
- **Style / dialect** — which checkpoint: raw base, a photoreal finetune (Juggernaut, RealVisXL), an art generalist (DreamShaper), or an anime/booru model (Pony, Illustrious/NoobAI).

These axes are **composable**. Lightning, LCM, and Hyper-SDXL all ship as **LoRAs** that drop onto *any* SDXL finetune. So "Juggernaut + 4-step Lightning LoRA" gives you fast *and* photoreal in one stack. Pick a point on each axis.

### Speed variants (the fast axis)

| Variant | Method | Steps | CFG (ComfyUI / diffusers) | Sampler · Scheduler | Native res | Licence | Use when… |
|---|---|---|---|---|---|---|---|
| **Base 1.0** | — | 25–40 | 5–8 / 5.0–8.0 | `euler` · `normal` | 1024² | OpenRAIL++-M | final quality, max control, LoRA training base |
| **Base + Refiner** | ensemble of experts | 25 (split ~0.8) | 8 / 5.0 | `euler` · `normal` | 1024² | OpenRAIL++-M | squeeze extra high-frequency detail (optional; finetunes make it redundant) |
| **Turbo** | ADD | 1–4 | **1** / **0.0** | `euler_ancestral` · `SDTurboScheduler` | 512² | contested ¹ | real-time preview / drafting |
| **Lightning** | progressive + adversarial distil | 2 / 4 / 8 | **1** / **0.0** | `euler` · `sgm_uniform` | 1024² | OpenRAIL++-M | fast at 1024², commercially clean; **LoRA or full ckpt** |
| **LCM** | latent consistency | 4–8 | **1–2** / **1.0–2.0** | `lcm` · `sgm_uniform` | 1024² | OpenRAIL-ish | fast on *any* finetune (LCM-LoRA); needs `ModelSamplingDiscrete`→`lcm` |
| **Hyper-SDXL** | TSCD + reward | 1 / 2 / 4 / 8 | ~1 / ~0.0 | `euler` · `sgm_uniform` | 1024² | OpenRAIL++-M | best-rated 1-step; **LoRA or full ckpt** |

> ¹ **Turbo's licence is contested** — see *Licence & limitations*. **CFG note (read this):** for every distilled variant, "guidance off" means **CFG `1.0` in a ComfyUI sampler** and **`guidance_scale=0.0` in diffusers**. Those are the same setting expressed in two conventions. **Never type CFG `0.0` into a ComfyUI sampler** — that outputs the unconditional result and ignores your prompt. At guidance-off, **negative prompts are inert** (see *The one rule* and *Failure modes*).

### Checkpoints (the style axis) — you want a finetune, not raw base

Base SDXL 1.0 is a strong *foundation*, but it looks undertrained and plasticky next to its finetunes. For real work, start from one of these. Verify current versions and licences on the model page:

| Checkpoint `[community — Civitai model pages]` | Use when… | Prompt dialect |
|---|---|---|
| **Juggernaut XL** | all-purpose **photoreal** — the default "just works" pick | descriptive photo-keywords (this skill's default) |
| **RealVisXL** | **maximum photorealism**, portraits, skin/hair | descriptive photo-keywords |
| **DreamShaper XL** | artistic / fantasy / concept-art generalist | descriptive keywords, looser |
| **Pony Diffusion V6 XL** | anime/furry/cartoon, very flexible, own sub-ecosystem. **The last SDXL Pony** — V7 is a different architecture (callout below) | **`score_9, score_8_up, …` + `source_*` + booru tags** — normal prompts fail |
| **Illustrious / NoobAI XL** | anime/illustration powerhouse | **Danbooru booru tags** |

> **Dialect follows the checkpoint.** Pony and Illustrious were trained on tag vocabularies, so they need their own dialect. Their LoRAs are a **separate pool** and do not work with base-SDXL LoRAs. Everything else in this skill assumes the photoreal/base dialect unless noted. Details and current-version notes: `references/checkpoints-and-loras.md`.

> **The anime axis now has an option that is not an SDXL checkpoint.** **Anima** (CircleStone Labs, 2B, derived from NVIDIA's Cosmos-Predict2) has its own architecture, its own loaders, and its own LoRA pool, so nothing on this page loads it. But it takes **Danbooru tags and attention weighting**, which means it reads to a prompter like a fifth anime "checkpoint". It is now one of the largest base-model ecosystems on Civitai, competing directly with Pony/Illustrious/NoobAI `[flagged — re-verify]`. **The licence differs, but not in the way the phrase "non-commercial" suggests, so read it before you route around it.** Anima's CircleStone Labs NC licence (plus NVIDIA's Open Model License, inherited through Cosmos) restricts the **weights**, not the pictures. §1(a) puts Outputs outside the definition of Derivative. §2(e) grants use of them *"for any purpose (including for commercial purposes)"*, and the card names sold images, paid commissions and game/VN assets as allowed, for companies as much as for individuals. What you may not do is **ship the weights**. Hosting Anima behind a paid API, or embedding it in a monetised product, needs a separate licence. §2(c)'s carve-out lets an individual sell derivative weights, but only *"solely to the model weights, and not to any larger product"*. So SDXL wins when the **model** is what you are shipping. For paid work whose deliverable is a **picture**, Anima is not disqualified. The other trap for an SDXL reader is weighting: Anima needs weights pushed well past SDXL's ~1.05–1.3 band. `(chibi:2)` is ordinary there, and it would fry SDXL. Full treatment: [`anima`](../anima/).

> **Pony has left SDXL too.** **Pony Diffusion V7** shipped on 2025-10-08 as a ~7B **AuraFlow** model, not an SDXL finetune, so nothing on this page loads it either. It has its own LoRA pool, it trains on SimpleTuner rather than kohya_ss or OneTrainer, and it drops the `score_9, score_8_up…` ladder for a tag-plus-description prompting scheme `[official — purplesmartai/pony-v7-base card; Civitai article 6309]`. Its own team says V7 prompting is still inconsistent, with a fix promised for V7.1. So **V6 XL stays the practical Pony pick for SDXL-ecosystem work.** V6 is the version this skill's dialect notes, LoRA pool and stacking advice describe, and V7's tooling is still young `[flagged — re-verify]`. A V6 LoRA will not load on V7, because the architecture is different. When a user says "Pony" with no version, ask which one. The anime axis now has two non-SDXL challengers, Anima and Pony V7, not one.

> **A `../link/` on this page that doesn't resolve is a skill you have not installed, not a broken
> page.** [`generative-media-atlas`](../generative-media-atlas/) is the map of this suite: which
> model fits a job, which skills that job needs, and the commands to install them. It works on its
> own, so it is the one to add first — `npx skills add ryannel/skills --skill generative-media-atlas`

---

## The one rule that changes everything

SDXL is conditioned by CLIP, and CLIP matches **tokens and short phrases**, not syntax. So **write a weighted, comma-separated list of keyword phrases — front-loaded, within ~77 tokens — and match the dialect to your checkpoint.** This is the opposite of the LLM-encoder models. Do not write a flowing sentence, because CLIP cannot parse clause structure the way LLM-encoder models can. Do not lean on a generic `masterpiece, best quality, 8k` booster block either; it is near-useless on SDXL. Earn quality with concrete photographic terms instead.

The encoder class also flips the LoRA rules. On SDXL, triggers are **verbatim rare tokens**, because CLIP does tag-matching, and training captions are tags. Flux and Z-Image instead fold triggers into prose or omit them. Dialect, triggers, and captions all follow the encoder here, with the refinement that **dialect follows the checkpoint**, as the note above already implies. The encoder is standing in for the **caption corpus**, and the corpus is the real rule underneath: the encoder class sets the ceiling on what a dialect *can* express, while what a model was captioned on decides what it is fluent in. The two usually agree, because CLIP arrived with tag corpora and LLM encoders arrived with prose. That is exactly why Pony and Illustrious can change dialect without changing encoder, why [`anima`](../anima/) takes weighted booru tags behind an LLM encoder, and why [`ideogram-4`](../ideogram-4/) takes JSON behind one. The encoder is still the right first guess; check the corpus before betting on it.

| Don't (LLM-style) | Don't (empty booster) | Do (SDXL keyword phrases) |
|---|---|---|
| *A candid documentary photograph of a young woman standing alone in a sunlit kitchen…* | `1girl, masterpiece, best quality, 8k, ultra detailed` | `candid documentary photo, young woman, detailed skin, standing in a sunlit kitchen, soft window light, 35mm, shot on Fujifilm X-T4, Kodak Portra 400` |

Four mechanics follow from the CLIP encoder:

1. **Position = emphasis.** CLIP weights earlier tokens more. Put the subject and the 2–3 highest-value concepts first.
2. **Weighting syntax** is `(phrase:1.2)` — raise toward 1.3, lower toward 0.9. **SDXL fries faster than SD1.5.** Keep weights in **~1.05–1.3** `[community]`. The 1.5–1.8 values common in SD1.5 guides over-saturate and posterise SDXL.
3. **77 tokens per chunk.** Past ~77 tokens, prompt-weight falls off a cliff, so keep it tight. If you genuinely need more, split with `BREAK` (ComfyUI/A1111) so each chunk is encoded separately.
4. **Dialect by checkpoint** — photoreal keywords for base/Juggernaut/RealVis; `score_*`/`source_*` for Pony; booru tags for Illustrious/anime.

**Advanced lever — the dual-encoder split.** SDXL has two text encoders, and the **`CLIPTextEncodeSDXL`** node lets you feed them different text. `text_g` goes to OpenCLIP-bigG, the global/scene channel, and `text_l` goes to CLIP-L, the local/detail channel. The **stock ComfyUI base and base+refiner templates use the plain `CLIPTextEncode` node**, which sends the same text to both. The split is an *optional* power-user technique, not the default path. Use it when you want scene and detail steered separately — for example, global mood in `text_g` and specific object or texture detail in `text_l`.

Full prompt anatomy, the complete photoreal vocabulary (camera bodies, film stocks, lenses, lighting, photographer names), the Pony/booru dialects, negatives, and worked SDXL-calibrated examples: **`references/prompting-guide.md`**.

---

## Setup & ecosystem

SDXL runs in **ComfyUI core** with no custom nodes. Unlike the DiT models, the checkpoint is a **single file that bundles UNet + both CLIP encoders + VAE**. Load it with one **`CheckpointLoaderSimple`** node, which outputs `MODEL`, `CLIP`, and `VAE`.

**File layout** (download from the Stability HF repos):

| File | ComfyUI folder | Loader node |
|---|---|---|
| `sd_xl_base_1.0.safetensors` (~6.94 GB) | `models/checkpoints/` | CheckpointLoaderSimple |
| `sd_xl_refiner_1.0.safetensors` (~6.08 GB) | `models/checkpoints/` | CheckpointLoaderSimple |
| `sdxl_vae.safetensors` / `sdxl-vae-fp16-fix` (~335 MB) | `models/vae/` | VAELoader (optional override) |
| any finetune / fast-variant checkpoint | `models/checkpoints/` | CheckpointLoaderSimple |
| LoRAs (incl. Lightning/LCM/Hyper LoRAs) | `models/loras/` | LoraLoader |

**The VAE gotcha:** SDXL's original VAE **overflows in fp16 and produces black/NaN images**. The VAE baked into the checkpoint is fine, and the stock templates use it. But if you decode in fp16 with a standalone VAE, point a `VAELoader` at **`madebyollin/sdxl-vae-fp16-fix`**, or run the VAE in fp32. Black outputs mean this is what happened `[community; strong]`.

**Stock node settings (verbatim from the official `sdxl_simple_example.json` template):**
- **Base:** `EmptyLatentImage` **1024×1024**; `KSamplerAdvanced` → **steps 25, cfg 8, sampler `euler`, scheduler `normal`**, `start_at_step 0`, `end_at_step 20`, `return_with_leftover_noise enable`.
- **Refiner (ensemble):** a second `KSamplerAdvanced` with **`add_noise disable`**, **steps 25, cfg 8, `euler`/`normal`**, `start_at_step 20`, `end_at_step 10000`. The **20/25 = 0.8 split** means the base runs 0→0.8 and the refiner finishes 0.8→1.0 on the *same* 25-step schedule. The refiner continues the base's leftover-noise latent — no fresh noise, no decode in between.
- Recommended SDXL resolutions (from the template's note): **1024×1024 · 1152×896 · 896×1152 · 1216×832 · 832×1216 · 1344×768 · 768×1344 · 1536×640 · 640×1536** (all ≈1 MP, multiples of 64). 512² gives badly degraded output. Unlike SD1.5, do not go below the 1024-area buckets.

**diffusers:** the pipelines are **`StableDiffusionXLPipeline`** (t2i), **`StableDiffusionXLImg2ImgPipeline`** (img2img / the refiner pass), and **`StableDiffusionXLInpaintPipeline`** (inpaint). Minimum **diffusers ≥ 0.19.0**. The base+refiner ensemble is wired with `denoising_end=0.8` on the base and `denoising_start=0.8` on the refiner, handing off latents. Full code: `references/setup-and-workflows.md`.

**Quantisation — there is effectively no GGUF path for SDXL.** SDXL is a **Conv2D-heavy UNet**, and GGUF/DiT quantisation is built for transformer models. The `ComfyUI-GGUF` author explicitly says *don't quantise SDXL*. fp16 is the standard format (~6.5 GB), with optional `--fp8_e4m3fn-unet` weight-casting in ComfyUI for tight VRAM. ComfyUI auto-offloads, so 1024² runs on ~**4 GB** in low-VRAM mode, and **6–8 GB is comfortable**. Base+refiner keeps both checkpoints resident, so budget **8 GB+**. Stability and Comfy publish no hard minimum, so these are practitioner figures `[community]`.

ControlNet, IP-Adapter (incl. FaceID), hires-fix and tiled-upscale workflows are all mature for SDXL — this control depth is its biggest practical edge. See `references/setup-and-workflows.md` and `references/checkpoints-and-loras.md`.

---

## Per-variant settings

One block per variant. Numbers are from the official templates and model cards (primary), except where a marker says otherwise. **Seed behaviour is stated per block because the distilled variants change it.** At 25 steps you can nudge the schedule and keep the image. But at 1–4 steps, the seed, the step count and the scheduler act as a single joint input: change any one of them and you have re-rolled, not refined.

### Base 1.0

Steps **25–40** (30–40 sweet spot), **CFG 5–8** (the template uses 8; finetunes often prefer 3–7), `euler`/`normal`, 1024-area bucket. **Use negative prompts.** Best LoRA-training base. **Seed:** fully deterministic at a fixed sampler, scheduler and step count, and stable *across* small step changes. You can tune steps without losing a composition you like.

### Base + Refiner

Total **steps 25**, hand off at **0.8** (base 0→20, refiner 20→25), **CFG 8**, `euler`/`normal`. The refiner adds high-frequency detail only. It is *optional*, and it becomes largely redundant once you use a good finetune. **Seed:** one seed governs both passes, because the refiner continues the base's leftover-noise latent on the same 25-step schedule. Reproducing an image means reproducing the split point (20/25) too, not just the seed.

### Turbo

**1–4 steps**, **CFG 1** (ComfyUI) / **0.0** (diffusers), **`euler_ancestral`** via `SamplerCustom`, scheduler **`SDTurboScheduler`** (`steps`, `denoise 1`), **512²**. Negatives are inert. 1 step is usable; 4 is better. **Seed:** reproducible only when the step count *and* scheduler match exactly. `euler_ancestral` injects fresh noise at every step, and with only one to four steps, a step-count change rewrites the image.

### Lightning

**2 / 4 / 8 steps** (match the checkpoint/LoRA to the step count), **CFG 1** / **0.0**, **`euler`**, **`sgm_uniform`**, 1024². The full ckpt or UNet beats the LoRA on quality; use the **LoRA** to add speed onto a *custom finetune*. 1-step is experimental; 2-step is the floor and 4-step the popular default. **Seed:** deterministic, but bound to the step count the checkpoint or LoRA was distilled for. Running a 4-step Lightning LoRA at 8 steps gives a different image, not a more detailed one.

### LCM

**4–8 steps**, **CFG 1–2** / **1.0–2.0**, sampler **`lcm`**, **`sgm_uniform`**. **LCM-LoRA** applies to any SDXL finetune, but it requires a **`ModelSamplingDiscrete`** node set to **`lcm`** and patched onto the model. It is softer than Lightning/Hyper but the most portable. **Seed:** deterministic given the same steps, scheduler *and* the `ModelSamplingDiscrete` patch. The patch is part of the sampling identity, so re-running the same seed without it silently produces a different — and worse — image rather than an error.

### Hyper-SDXL

**1 / 2 / 4 / 8 steps**, **CFG ~1** / **~0.0**, **`euler`**, **`sgm_uniform`**, 1024². LoRA or full ckpt; often the best **1-step** quality of the fast variants. **Seed:** same rule as Lightning — reproducible only at the step count the variant was distilled for.

---

## Realism: pick a finetune, then stack the gear

SDXL reaches photoreal by the **gear-stacking route**: concrete photographic vocabulary piled on top of a photoreal finetune. Two levers, in order of impact:

1. **Use a photoreal finetune, not base.** This is the single biggest lever. **Juggernaut XL** or **RealVisXL** outclass raw base SDXL on skin, light, and anatomy before you change a single prompt word.
2. **Stack concrete photographic vocabulary.** Earn realism with specifics, not adjectives. Build the prompt from these slots (full vocabulary tables in `references/prompting-guide.md`):
   - **Style tag first:** `candid photo`, `documentary photography`, `glamour photography`, `analog photo`, `polaroid`…
   - **`detailed skin`** — the highest-value realism keyword on SDXL finetunes. It adds pores and texture and kills the plastic look.
   - **Camera body:** `shot on Canon EOS 5D`, `Fujifilm X-T4`, `Hasselblad X1D II`, `ARRI ALEXA 65`, `RED digital cinema`…
   - **Film stock:** `Kodak Portra 400` (flattering skin), `Cinestill 800T` (halation), `Ektar 100` (saturated), `Tri-X 400` (B&W), `Fujicolor Pro`…
   - **Lighting:** name the source, quality, and direction — `soft window light from camera-left`, `golden hour rim light`, `harsh direct flash`, `chiaroscuro single source`.
   - **One imperfection / texture anchor:** `film grain`, `slight vignette`, `subsurface scattering`, `visible skin pores`.

   Keep effect/style weights modest (`(detailed skin:1.2)`, not `:1.6`). `8k`, `masterpiece`, and `realistic` do almost nothing. Drop them for the concrete terms above.

SDXL's strengths are **ecosystem depth (LoRAs, ControlNet, IP-Adapter), speed, low-VRAM accessibility, artistic/anime range, and photoreal-via-finetune**, plus clean commercial licensing on base. Its weaknesses are **in-image text (it basically can't render legible words), complex/compositional prompts (77-token CLIP, no LLM encoder), and hands/anatomy**. Use SDXL when you want control tooling, a specific finetune/LoRA look, fast iteration, or to run on a small GPU. When SDXL's CLIP limitations are the bottleneck — reliable text, compositional prompts, natural-language description — you need a model with an LLM text encoder instead.

---

## Production pipelines & mixing models

For production output, SDXL runs the **full multi-stage ladder**. It is the family with the most rungs worth climbing. The denoise bands below are convergent community settings rather than official ones `[community — Civitai workflow authors]`:

1. **Base gen** in a 1024-area bucket (finetune + optional speed LoRA). Judge composition only; reroll freely.
2. **Hires second pass** — latent ×1.5 at denoise 0.3–0.5, or pixel-space upscale + re-sample at 0.25–0.35.
3. **Detailers** — FaceDetailer/ADetailer at denoise ~0.4. Swap the character LoRA in *here* rather than loading it in the base pass `[community — MyAIForce; strong]` (`references/characters.md §3`).
4. **Tiled upscale** — `UltimateSDUpscale` with a 4× ESRGAN model, denoise 0.2–0.35, and a simpler prompt than the base gen.
5. **Finish** — ColorMatch against the pre-upscale image; optional SeedVR2-class restorer.

Per-stage settings: `references/setup-and-workflows.md §6`.

**SDXL's role in mixed-model pipelines** is defined by its ecosystem: it has the control tooling the DiT models lack, and they have the rendering quality it lacks. Two named, mainstream patterns `[community — Civitai workflow authors]`:
- **Controllable front-end:** compose with SDXL's ControlNet/IP-Adapter/regional stack → decode to pixels → refine in Z-Image-Turbo or Flux.2 Klein img2img at denoise ~0.25–0.4 for natural rendering and cleaner anatomy.
- **Texture back-end:** generate in a DiT model → img2img through a photoreal SDXL finetune (RealVis-class) at ~0.3–0.55 to add its skin/texture character.

The handoff rule between families: **always VAE-decode to pixels first**. SDXL's latent space is incompatible with Flux/Z-Image latents. The full cross-model craft (denoise bands, resolution matching, color management, workflows-as-code) is the **[`image-production-workflows`](../image-production-workflows/)** skill.

---

## Failure modes & QC

| Symptom | Cause | Fix |
|---|---|---|
| Plastic / waxy / undertrained look | Running **base** SDXL raw | Switch to a photoreal finetune (Juggernaut/RealVisXL); add `detailed skin` |
| Fried, over-saturated, posterised colours | Prompt weights too high (SD1.5-style 1.5–1.8), or CFG too high | Keep weights ~1.05–1.3; drop CFG toward 5–7 |
| Black / NaN image | SDXL VAE overflowing in fp16 | Use `sdxl-vae-fp16-fix` via VAELoader, or decode the VAE in fp32 |
| Subject's head / feet cut off at frame edge | Non-zero crop-conditioning (`crops_coords_top_left`) | Set crop coords to **(0,0)** (centered); the `CLIPTextEncodeSDXL` `crop_w/crop_h` = 0 |
| Output looks low-res / soft even at 1024² | Low size-conditioning (`original_size`) | Set `original_size`/`target_size` to your actual resolution (e.g. 1024×1024) |
| Duplicated subjects, warped anatomy | Rendering far outside a 1024-area bucket (e.g. native 2048²) | Generate in a listed bucket, then upscale (hires-fix / tiled) |
| Negative prompt seems ignored | Distilled variant at **CFG 1 / guidance 0.0** — negatives are inert | Phrase constraints positively; or use base/finetune at CFG > 1 for negative control |
| Garbled / absent in-image text | CLIP-UNet can't render reliable typography | Don't ask SDXL for text — composite it in post-production instead |
| Mangled hands, extra fingers | Hands vary hugely from every angle and constantly occlude themselves, so training never gave the UNet a stable, reliable form the way well-lit, unoccluded faces did | Negative `extra fingers, deformed hands`; inpaint/ControlNet the hands; re-roll |
| Prompt past ~77 tokens ignored | CLIP token window | Tighten, or split with `BREAK` so each chunk encodes separately |
| Pony/Illustrious output is mush | Wrong dialect (photoreal prompt on a tag-trained model) | Use `score_*`/`source_*` (Pony) or booru tags (Illustrious) |

---

## Pre-flight checklist

Before hitting Queue Prompt:

1. Using a **finetune** suited to the job (photoreal → Juggernaut/RealVis; anime → Pony/Illustrious), not raw base?
2. Prompt is **weighted keyword phrases**, front-loaded, not an LLM sentence or an empty `8k masterpiece` block?
3. Weights in **~1.05–1.3** (not SD1.5's 1.5–1.8)?
4. Prompt **dialect matches the checkpoint** (photo-keywords vs `score_*` vs booru)?
5. Resolution is a **1024-area bucket**; crop-conditioning at **(0,0)**?
6. Photoreal: `detailed skin` + camera body + film stock + named lighting?
7. Right **CFG for the variant** — base 5–8; distilled at **CFG 1 (ComfyUI) / 0.0 (diffusers)**, never 0.0 in ComfyUI?
8. Negative prompt present **only where it works** (base/finetune at CFG > 1), inert on distilled?
9. fp16 decode → `sdxl-vae-fp16-fix` loaded (no black images)?
10. Not asking SDXL to render in-image text?

---

## Where SDXL sits in the suite

| Job | SDXL | Reach for instead |
|---|---|---|
| Consistent characters | **Deepest toolbox** — InstantID/HyperLoRA adapters, mature LoRA training, `[SEP]` routing, block-weight control (`references/characters.md`) | [`flux-2`](../flux-2/) for native multi-reference editing; [`qwen-image`](../qwen-image/) for the no-training edit engine (Edit-2511) and the dataset factory `references/characters.md` routes through |
| Style LoRAs | **The mature ecosystem** — years of recipes, two trainers, separate Pony/Illustrious pools (`references/lora-training.md`) | a DiT model when the style needs prompt comprehension SDXL lacks; [`character-lora-training`](../character-lora-training/) for the dataset and captioning craft that transfers across every base here |
| Structural control | **The most complete stack** — union ControlNet, IP-Adapter, regional prompting | — (this is SDXL's edge) |
| Photoreal faces & skin | **Strong once you leave raw base — but the strength is control, not skin.** A photoreal finetune (Juggernaut/RealVisXL) plus `detailed skin` and the camera/film/lens vocabulary gets you a convincing frame. Then SDXL's adapters and ControlNets let you dictate pose, composition and identity inside it in a way no newer model matches. Base SDXL alone will not get you there | [`z-image`](../z-image/) — the suite's owner of faces and skin themselves, and its standard face-pass finisher. The two compose rather than compete: block out and control the shot here, then finish the face there (~0.2 denoise) `[community — nsfwVariant, Civitai]`. [`krea-2`](../krea-2/) gives stylistic breadth without checkpoint-hopping, but budget its two taxes (soft default, muted expressions). The camera/lens/film-stock stack does **not** go away there |
| Anime / booru illustration | **Deep, and commercially clean end to end** — Pony V6 XL, Illustrious and NoobAI, each with its own dialect and its own LoRA pool, with no restriction on shipping the weights. Pony V7 is AuraFlow, not SDXL, and is routed out (checkpoint callout above) | [`anima`](../anima/) when you want the anime-native base rather than an SDXL finetune. Its *outputs* are commercially free, so paid illustration work is fine. SDXL wins when you must **ship or host the model itself** |
| In-image typography | Basically can't | [`ideogram-4`](../ideogram-4/) |
| Compositional / long prompts | 77-token CLIP ceiling | [`flux-2`](../flux-2/) or [`z-image`](../z-image/) (LLM encoders) |
| Stylistic range without checkpoint-hopping | The look lives in the checkpoint — switching styles means switching finetunes | [`krea-2`](../krea-2/) — one model spanning a wide visual space via style refs / official style LoRAs |
| Commercial use under the licence | Base, Lightning and Hyper-SDXL are clean OpenRAIL++-M with no revenue cap — quietly one of SDXL's strongest advantages over most of the newer models. [`z-image`](../z-image/)'s Apache-2.0 is cleaner still, since OpenRAIL++-M's use-restrictions travel downstream with every redistribution. **Turbo is the exception** `[contested]` | — (verify Turbo directly with Stability before shipping on it) |
| Mixed-model pipelines | **Front-end (control) and back-end (texture)** roles | [`image-production-workflows`](../image-production-workflows/) for the cross-model craft |
| Making it move | Still images only | [`wan-2-2`](../wan-2-2/) — image-to-video from a still locked here. SDXL's mature identity stack (InstantID, IP-Adapter FaceID) is a strong upstream for the still that anchors each shot |
| **Choosing between all of these in the first place** | — this table is one model's view of the suite | [`generative-media-atlas`](../generative-media-atlas/) — the whole suite ranked by job (realism, identity, LoRA trainability, control, licence, video), the elimination ladder that settles most choices, and end-to-end routes across several skills |

---

## Licence & limitations

**Commercial-use picture (verify on each model page before relying on it):**

| Variant | Licence | Commercial? |
|---|---|---|
| **SDXL 1.0 base / refiner** | CreativeML **Open RAIL++-M** | ✅ Yes, no revenue cap |
| **SDXL Lightning** (ByteDance) | Open RAIL++-M (inherits base) | ✅ Yes, no cap — the cleanest fast variant |
| **Hyper-SDXL** | Open RAIL++-M | ✅ Yes |
| **LCM-LoRA** | OpenRAIL/permissive | ✅ Generally yes |
| **SDXL Turbo** | **contested** — see below `[contested]` | ⚠️ Verify |

**OpenRAIL++-M** permits royalty-free commercial use, and you own your outputs. But it is a *responsible-AI* licence, not OSI-open. It carries **use-restrictions** (no illegal use, CSAM, harassment, disinformation, unlicensed medico-legal advice, etc.), and you must **pass them downstream** when you redistribute the model or derivatives.

**Turbo's licence is genuinely contested — don't state a flat verdict.** The current `LICENSE.md` in `stabilityai/sdxl-turbo` is the **Stability AI Community License** (updated 5 July 2024), which **permits free commercial use for entities under US $1M annual revenue** (paid membership above that). *But* the same repo's metadata tag still reads `sai-nc-community`, and the model-card prose still says "non-commercial / research." The actual LICENSE.md governs. Still, **anyone relying on commercial Turbo use should confirm with Stability directly** rather than trust one field.

**Architectural limitations** (primary, from the encoder design): the **77-token CLIP window** and the absence of an LLM/T5 encoder cap compositional and long-prompt comprehension. There is **no reliable in-image text**, and hands/anatomy need negatives, inpainting, or ControlNet. Don't import **SD1.5 negative-embedding** crutches (`UnrealisticDream`, etc.) — they don't load on SDXL.

**Release & stability:** SDXL 1.0 shipped 26 July 2023. By image-model standards it is **old and stable**, so the core facts here move slowly. The **fast-moving parts** are the community layer: finetune versions (Juggernaut and RealVis release new versions regularly), the fast-variant LoRAs, and Turbo's licence status. Re-verify those before relying on them.

---

## How to read the claims in this skill — two bars, by claim type

This skill holds two kinds of claim to two different standards, because they fail in two different ways.

**Hard facts — must be exact or it breaks.** This covers the architecture (2.6B UNet, dual CLIP-L + OpenCLIP-bigG, 2048-dim concat, micro-conditioning), the ~6.6B ensemble figure, native 1024² and the resolution buckets, the node names (`CLIPTextEncodeSDXL`, `CLIPTextEncodeSDXLRefiner`, `LoraLoader`), the stock node settings (1024², 25 steps, CFG 8, euler/normal, the 0.8 base/refiner split), the diffusers pipeline classes and ensemble code, the distillation methods, the OpenRAIL++-M / Lightning licence terms, and the fp16-VAE overflow. **The source of truth is official** — the SDXL paper (arXiv 2307.01952), the Stability + ByteDance HF model cards, the licence files, and the official ComfyUI templates — and these claims are verified there. A wrong node name won't wire. A misread licence is a legal problem. SDXL is old, so these facts move slowly, but Turbo's licence (above) is the exception `[contested]`. Re-verify before relying on them, regardless of who said it.

**Craft — what actually makes a good image.** This covers which finetune to start from and at what version (Juggernaut, RealVisXL, DreamShaper, Pony V6, Illustrious/NoobAI), the Pony/Illustrious prompt dialects (score tags, booru ordering), the photoreal camera/film/lens/lighting vocabulary, LoRA weights and how the fast-variant LoRAs stack onto finetunes, VRAM thresholds, and GGUF (un)suitability. **The authoritative source here is the community**: the finetune authors, the Civitai model pages, and practitioners who have run these checkpoints for years (neonkisu, QuantumBogoSort, MyAIForce, WeirdWonderfulAI, Khanykov01, Ainara and L3n4 are the most-cited across this skill's references). It is *not* the base model card, which describes the undertrained base SDXL that nobody ships for real work. This is the deep, battle-tested layer, and it is stated with confidence. Where a claim is a range or says "verify the current version," that is because the community layer moves (new finetune versions land monthly), not because it is unreliable. One calibration note: the photoreal *vocabulary* is CLIP-understood, but the SD1.5-era **weight values** from older guides don't transfer — those were dropped.

**Contested / unresolved points.** There are five, and they are all in the community bar:

- **Turbo's licence.** The `stabilityai/sdxl-turbo` repo's `LICENSE.md` permits free commercial use below US $1M revenue, while the same repo's metadata tag and model-card prose still read non-commercial. The licence file governs, but the contradiction is unresolved on Stability's side. So commercial Turbo is unsettled, not merely undocumented `[contested]`.
- **Character-LoRA rank.** The classic 8–16 ladder and the 48/48 default (48–64 when a LoRA "forgets" below weight 0.5) `[community — neonkisu, QuantumBogoSort]` are both defensible. This skill deliberately does not average them. The choice turns on whether you need a well-behaved stacking citizen or maximum identity retention at reduced weight `[contested]`. Both positions, with sources: `references/lora-training.md §3`.
- **Anima against SDXL's anime finetunes.** It is growing fast, and named trainers call it "the new Illustrious." But its LoRA pool is young, its quality varies widely, and the weights-side licence restriction caps who can build a *product* on it. Whether it displaces Pony/Illustrious/NoobAI is genuinely open `[flagged — re-verify]`. See [`anima`](../anima/) and `references/lora-training.md §1`.
- **Pony V7 against Pony V6 XL.** V7 is AuraFlow, so it sits outside SDXL entirely, and its LoRA pool and prompting are still settling. This skill keeps V6 XL as the Pony recommendation because that is where the compatible LoRAs and the settled dialect are. Once V7's pool matures, the Pony rows here become a routing note like Anima's `[flagged — re-verify]`. See `references/checkpoints-and-loras.md §3` and `references/lora-training.md §1`.
- **Which finetune currently leads.** Juggernaut, RealVisXL, Pony, Illustrious and NoobAI ship new versions faster than this skill is re-checked. The picks named here are stable *families*, not current-version verdicts. Treat any version number in this skill as an example, not a recommendation `[flagged — re-verify]`.

**Facts dated 2026-09-09.** The architecture, node names, template settings and diffusers classes above have not moved since 2023, and are unlikely to. Everything that moves is in the community bar: finetune version numbers first, then Turbo's licence status, then the two non-SDXL challengers on the anime axis, Anima and Pony V7. Re-verify those before relying on them.

---

## Reference files

| File | When to read it |
|---|---|
| `references/prompting-guide.md` | Full prompt anatomy; the complete photoreal vocabulary (style tags, camera bodies, film stocks, lenses, lighting, photographer names, filters); weighting and 77-token/BREAK economy; the dual-encoder text_g/text_l split; negative-prompt guidance (variant-aware); the Pony score-tag and Illustrious booru dialects; worked SDXL-calibrated example prompts |
| `references/setup-and-workflows.md` | ComfyUI graphs in full (base, base+refiner ensemble, Turbo, Lightning, LCM, Hyper); every stock node setting; the fp16-fix VAE wiring; diffusers code (t2i, img2img, inpaint, the ensemble); hires-fix and tiled upscale; ControlNet and IP-Adapter setup; quantisation/VRAM |
| `references/checkpoints-and-loras.md` | The finetune ecosystem in depth (Juggernaut, RealVisXL, DreamShaper, Pony V6, Illustrious/NoobAI) with dialects and licence notes; the separate Pony/Illustrious LoRA pools; **§4 Using LoRAs** (loading any LoRA — the full `LoraLoader` patches UNet + both CLIP encoders, `strength_model`/`strength_clip`, dialect-pool matching, weight-by-type, triggers, stacking); the fast-variant speed LoRAs; ControlNet and IP-Adapter model catalog |
| `references/lora-training.md` | **Making** a LoRA (using is checkpoints-and-loras §4) — the deepest treatment in the suite. **Base selection** (Pony V6 vs Illustrious vs NoobAI v-pred vs WAI vs Anima, why it dominates every other choice, and why Pony V7 routes out to SimpleTuner), kohya_ss/OneTrainer/ai-toolkit and the Civitai on-site trainer, the convergent recipes and the **contested rank question** (8–32 ladder vs 48–64 for low-weight retention), the ~80–100 steps-per-image anchor, **the 0.33 identity ratio** dataset architecture, **`caption_dropout` as the generalisation lever**, why you cannot caption your way out of missing variety, **dataset traps** (reference-sheet cutups, colour padding baked into the weights, aspect-ratio mismatch), style-LoRA specifics, **weight noising + depth anchoring**, **stacking several LoRAs of one character at reduced strength**, XY-grid evaluation and a concrete ship criterion, and adult/NSFW work on SDXL |
| `references/characters.md` | Creating a **consistent character**: the identity-tool decision table (InstantID vs FaceID vs HyperLoRA vs ReActor vs character LoRA), the character LoRA pipeline (edit-model dataset factory, 8-point rotation in keyword dialect), the detailer LoRA swap, **`[SEP]` multi-character routing**, the **block-weight style-bleed fix** (SDXL-unique), multi-outfit LoRAs, failure modes |
