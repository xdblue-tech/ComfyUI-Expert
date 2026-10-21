# Krea 2 — Setup & workflows

This file covers local setup (ComfyUI, diffusers, the reference CLI), quantisation and VRAM strategy, using LoRAs, and the multi-stage and mixed-model workflows. Facts were verified against the template JSON, HF repo listings, GitHub README, diffusers docs, and musubi-tuner docs on 2026-07-07; the diffusers release, the quant table and the VAE tooling were re-checked 2026-09-09. Community recipes are attributed inline.

## Contents
1. [ComfyUI: the official template, node by node](#1-comfyui-the-official-template-node-by-node)
2. [Quantisation & VRAM](#2-quantisation--vram)
3. [The reference CLI](#3-the-reference-cli)
4. [diffusers](#4-diffusers)
5. [The VAE decision](#5-the-vae-decision)
   - [5a. The Wan 2.1 VAE swap](#5a-the-wan-21-vae-swap)
   - [5b. Qwen Image VAE Sharp / Sharp Plus](#5b-qwen-image-vae-sharp--sharp-plus)
   - [5c. Colour grading in latent space](#5c-colour-grading-in-latent-space)
6. [Using LoRAs](#6-using-loras)
7. [Multi-stage workflows](#7-multi-stage-workflows)
8. [Krea 2 in mixed-model pipelines](#8-krea-2-in-mixed-model-pipelines)

---

## 1. ComfyUI: the official template, node by node

The template is **`image_krea2_turbo_t2i.json`** (from Comfy-Org/workflow_templates; it appears in-app under Templates → Image once ComfyUI is updated). The tutorial lives at docs.comfy.org/tutorials/image/krea/krea-2. It needs no custom nodes. Only a Turbo t2i template exists. There is no official Raw template, because Raw is not recommended for inference.

**Downloads** (all from `Comfy-Org/Krea-2`):

| File | Size | → folder |
|---|---|---|
| `diffusion_models/krea2_turbo_fp8_scaled.safetensors` | 13.1 GB | `models/diffusion_models/` |
| `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` | 5.2 GB | `models/text_encoders/` |
| `vae/qwen_image_vae.safetensors` | 0.25 GB | `models/vae/` |
| `loras/krea2_<style>.safetensors` (optional) | 0.47 GB | `models/loras/` |

**The graph.** A subgraph wraps the pipeline; double-click to open it. The nodes are:

- `UNETLoader` → `krea2_turbo_fp8_scaled.safetensors`, weight_dtype `default`
- `CLIPLoader` → `qwen3vl_4b_fp8_scaled.safetensors`, type **`krea2`**, device `default`
- `VAELoader` → `qwen_image_vae.safetensors`
- `LoraLoaderModelOnly` → strength **0.8**, behind an `enable_lora?` boolean switch. It is model-only because Krea 2 LoRAs never touch the text encoder.
- `CLIPTextEncode` for the positive prompt. The negative path runs through **`ConditioningZeroOut`**, which is the mechanism that makes negatives have no effect at cfg 1.0.
- `EmptyLatentImage` at 1024×1024. This is the *plain* latent node, not an SD3/Flux one, and there is **no shift node**: no `ModelSamplingAuraFlow`, because the model config handles the resolution-aware time shift. A `ResolutionSelector` offers 1K–2K aspect presets.
- `KSampler`: **steps 8, cfg 1.0, sampler `euler`, scheduler `simple`, denoise 1.0**
- `VAEDecode` → `SaveImage` (prefix `Krea2_turbo`)

**Subgraph switches.** `prompt_enhance` defaults to **true** and runs the `TextGenerate` LLM expander; see `prompting-guide.md §6` for when and why to turn it off, including the documented refusal bug Comfy-Org/ComfyUI#14631. `enable_lora?` defaults to false. A `CustomCombo` on the main canvas picks the style LoRA and auto-appends its trigger phrase to your prompt via `StringConcatenate`.

**Your ComfyUI has to be newer than 2026-06-22, and an old build does not tell you so.** `krea2` is a CLIP *type*, so an older build simply does not list it in the `CLIPLoader` dropdown. The encoder file is sitting there, the model loads, and the one option you need is quietly missing from a menu. This hurts most on rented pods and prebuilt containers, because their images are often pinned months back — a container built 2026-05-01 is older than the open-weights release itself. Check that dropdown before you debug anything else `[community — production run on a RunPod ComfyUI template, 2026-08-25]`.

Updating a container's bundled ComfyUI has two snags worth knowing about first:

- **`git pull` can fail because the branches have diverged**, since the image build edited the working tree. `git fetch origin && git reset --hard origin/master` works reliably. But check for local changes you care about first, because the reset throws them away.
- **Run `pip install -r requirements.txt` after the pull, not just the pull.** A current ComfyUI needs a newer `comfy_kitchen` than an older image ships with. You find out through a startup `ImportError` about a missing name (`int8_attention_is_available`, in the case we saw). Nothing mentions version numbers.

One more container trap: restart ComfyUI with **the same interpreter that started it**. If you kill a working `python3` process and relaunch under a `python` that points at a different environment, you get a torch build that does not match the card. It reads like a hardware fault.

**Hosted Medium/Large in ComfyUI** is a separate path entirely: the "Krea 2 Image" API/partner node, with API-key billing, style refs, moodboard IDs, and creativity Raw/Low/Medium/High. See `api-and-hosted.md §4`.

## 2. Quantisation & VRAM

**Official precisions** (Comfy-Org/Krea-2 repo listing, sizes exact):

| File | Size | Notes |
|---|---|---|
| `krea2_turbo_bf16` / `krea2_raw_bf16` | 26.3 GB | full precision; ~46 GB unified memory in practice `[community — liutyi]` |
| `krea2_turbo_fp8_scaled` / `krea2_raw_fp8_scaled` | 13.1 GB | the template default |
| `krea2_turbo_int8_convrot` / `krea2_raw_int8_convrot` | 13.5 GB | **~2× faster than fp8** — replicated down to a 1050 Ti, and quality disputed — "same if not better" `[community — nsfwVariant, YeahYeah2992]` vs **worse complex-prompt adherence than fp8** `[community — ganrocks007, 3060 12 GB]`. Needs a recent ComfyUI |
| `krea2_turbo_mxfp8` | 13.5 GB | Turbo only |
| `krea2_turbo_nvfp4` | 7.7 GB | Turbo only; Blackwell-class hardware |
| `qwen3vl_4b_bf16` / `_fp8_scaled` | 8.9 / 5.2 GB | text encoder |
| `qwen_image_vae` | 0.25 GB | |

**Community GGUF.** This time the ecosystem formed without city96. The repos are `gguf-org/krea-2-gguf`, `vantagewithai/Krea-2-Turbo-GGUF` and `-Raw-GGUF`, `molbal/krea2-gguf`, and `realrebelai/KREA-2_GGUFs`. The vantagewithai sizes are: Q2_K 4.9 GB, Q4_K_M 7.5 GB, Q6_K 10.6 GB, Q8_0 13.7 GB. GGUF requires the `ComfyUI-GGUF` custom node, because GGUF DiTs load via its loader rather than `UNETLoader`. **A per-quant comparison now exists, but read it before you lean on it.** A 150-image benchmark across BF16 / FP8 / INT8-ConvRot / INT4 / GGUF, scored on perceptual, semantic and latent-space metrics, is reported to recommend INT8-ConvRot as the primary pick for 16–24 GB and near-lossless against BF16 `[community — instasd formats guide, lilcheaty and Winnougan HF cards, 2026-08; re-verify]`. The benchmark itself could not be read for this pass — the HF cards only say "near-lossless in testing" and the write-up sits behind a wall — so it settles the overall-quality question provisionally and does **not** settle the complex-prompt-adherence dispute in the int8 row above `[contested]`. The table below is still size arithmetic plus the general GGUF experience from sibling models:

| VRAM | Working setup |
|---|---|
| 8–12 GB | GGUF Q2–Q4 DiT + fp8 encoder (or encoder on CPU) — expect quality loss at Q2/Q3. Or full `fp8_scaled` on weight offload if you have the system RAM to hold it: Turbo `fp8_scaled` is reported running on an 8 GB RTX 3070 Ti with 64 GB host RAM `[community — niechta]` — same weights, slower |
| 12–16 GB | Q4_K_M–Q6_K, or nvfp4 (7.7 GB) on Blackwell |
| 16–24 GB | fp8_scaled or int8_convrot (13.1–13.5 GB) + fp8 encoder — the comfortable tier |
| 24 GB+ | fp8/int8 with full headroom; bf16 wants ~46 GB (unified-memory / multi-GPU territory) |

The memory pattern that matters is documented for musubi, and the same physics apply in ComfyUI. The DiT stays resident on the GPU. The ~5–9 GB encoder and the VAE shuttle on and off the GPU around it. On a 24 GB card, fp8 (or block offloading) buys the headroom for the encode and decode steps; it does not work by evacuating the DiT. That pattern assumes a card with room to keep the DiT. Where there isn't one, the DiT stops being resident too, and its blocks stream in per step. That streaming is what makes 8 GB viable at all, and it is also why, on a small card, the number to check first is *host* RAM, not VRAM. Offload trades bus time for capacity, while GGUF trades precision for capacity. Prefer GGUF when system RAM is the scarce resource, and offload when it isn't. Koboldcpp's rolling build also runs Krea 2 (with Qwen3-VL + a Wan 2.1 VAE) `[community — u/Eisenstein, HN]`.

## 3. The reference CLI

The CLI lives at `github.com/krea-ai/krea-2` (Apache-2.0; install with `uv sync`). Set the `OSS_RAW` / `OSS_TURBO` env vars to the downloaded `raw.safetensors` / `turbo.safetensors` from the `krea/Krea-2-Raw` / `krea/Krea-2-Turbo` HF repos:

```bash
# Raw — full sampler with CFG; trained to 1K
uv run inference.py "a fox walking in the snow" --checkpoint oss_raw --steps 52 --cfg 3.5
# Turbo — 8 steps, CFG off, pinned mu; 1K–2K
uv run inference.py "a fox walking in the snow" --checkpoint oss_turbo --steps 8 --cfg 0.0 --mu 1.15 --width 2048 --height 2048
```

The flag defaults are: `--steps 28`, `--cfg 4.5` (0 disables it), and `--y1 0.5` / `--y2 1.15` for the resolution-interpolated time shift, where `--mu` pins a constant instead (1.15 is the recommended value for Turbo). Dimensions are padded to multiples of 16. `--seed 0` is the default, with image *i* using seed+*i*. Note that the CFG numbers follow the Krea convention, where 0 means off.

## 4. diffusers

**`Krea2Pipeline` ships in the stable release since diffusers 0.40.0 (PyPI, 2026-08-20)**, with the modular pipeline, docs and tests in-tree `[official — diffusers 0.40.0 release notes]`. The source-only install that early guides describe is no longer needed:

```python
# pip install "diffusers>=0.40.0"
import torch
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Raw", torch_dtype=torch.bfloat16).to("cuda")
image = pipe("a fox in the snow", height=1024, width=1024,
             num_inference_steps=52, guidance_scale=3.5,      # Raw HF-card settings; diffusers class default is 28 / 4.5
             generator=torch.Generator("cuda").manual_seed(0)).images[0]
```

- The classes are `Krea2Pipeline` / `Krea2Transformer2DModel` / `AutoencoderKLQwenImage` / `Qwen3VLModel`, plus `text_encoder_select_layers`, which holds the 12-layer tap indices.
- **Turbo:** `is_distilled=True` in the pipeline config gives a fixed `mu=1.15`; run `num_inference_steps=8, guidance_scale=0.0`.
- **Guidance convention:** velocity = `cond + g·(cond − uncond)`. Guidance is enabled when g>0, and this equals classic CFG at `1+g`. `negative_prompt` is ignored when `g ≤ 0`.
- Scheduler: `FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting=True`, `base_shift=0.5`, `max_shift=1.15`, `base_image_seq_len=256`, `max_image_seq_len=6400`.
- `max_sequence_length=512`; dims are rounded to ×16. It is **t2i only** — there are no img2img/inpaint/edit pipeline classes yet.
- The HF repos carry a full diffusers layout (`transformer/`, `text_encoder/`, `vae/`, `scheduler/`, `model_index.json`) alongside the single-file checkpoints.

## 5. The VAE decision

Three decode paths now exist for the same latents, and the choice matters more on Krea 2 than on most models. A large share of what people call the model's softness is decided in the decode, not in the sampler. That makes the VAE a tuning knob, and one you can A/B on a latent you have already paid for. The stock `qwen_image_vae` is the conservative baseline. The Wan 2.1 FP32 swap is the standard cure for the soft default. The Qwen Image VAE Sharp line is the cure that does not move your colour.

### 5a. The Wan 2.1 VAE swap

The single highest-leverage quality fix for the soft/airbrushed default and the halftone/dark-noise artefacts is to decode through the **Wan 2.1 VAE (FP32)** instead of `qwen_image_vae`. It was reported independently as "solves this" for the blur complaint `[community — mobiuscog, HN]` and adopted in the best-documented realism workflow `[community — nsfwVariant, Civitai]`. The mechanics are simple: drop the Wan 2.1 VAE file into `models/vae/` and point the `VAELoader` at it. The latent spaces are compatible enough for decode; you are changing the renderer, not the model. Keep the Qwen VAE for encode-side operations (img2img-style passes) to stay conservative. A/B the swap on your own content, because this is community craft, not an official configuration. It is also not a cure-all. Moiré/halftone artefacts on hair and clothing are reported *on the Wan VAE too*, notably on community checkpoint merges (Fascium-class) even with all LoRAs off `[community — derTommygun, r/StableDiffusion]`. When that happens on a merge, re-test the stock checkpoint before debugging settings.

**Know what the swap does not fix.** It cures the *soft* default. It does nothing for texture that comes from rendering outside the checkpoint's resolution band: at 2048, the Wan 2.1 VAE and the Qwen VAE decoded to identical crumpled skin in a four-cell sweep, and the fix was rendering at 1024 native `[live-use — media lab, Ciara band sweep, 2026-09-08]`. If the artefact is a reticulated or "scale" skin rather than blur, the VAE is the wrong lever — see `lora-training.md §9`.

### 5b. Qwen Image VAE Sharp / Sharp Plus

These are retuned decoders for Krea 2 Turbo and Raw. They lift fine-edge response, micro-contrast and high-frequency detail *without* shifting colour, composition or character. That is the whole reason they exist alongside the Wan swap, because the Wan VAE does move colour. There are two grades:

| Decoder | Character | Reach for it when |
|---|---|---|
| `Qwen Image VAE Sharp` | conservative crisp-up | you want the stock look, just less mushy |
| `Qwen Image VAE Sharp Plus` | a real sharpening decode | hair, fabric, machinery and architecture need to separate visibly |

The mechanics are the same drop-in as §5a: put the file into `models/vae/` and point the `VAELoader` at it. **FP32 builds need ComfyUI started with `--fp32-vae`**, and without that flag you get a silently worse decode rather than an error. This is a clarity trade, not a strict upgrade: keep the stock VAE for softer or painterly work, where the sharpening reads as harshness `[community — Merserk13]`.

A decision shortcut: if the image is soft *and* the colour is fine, use Sharp or Sharp Plus. If it is soft *and* you also dislike the Qwen VAE's rendering character, use Wan 2.1. For painterly or stylised work, leave it stock.

### 5c. Colour grading in latent space

The **exposure, temperature, tint, detail/clarity and contrast vectors** have been extracted from Krea 2's ([Qwen-Image](../../qwen-image/)) VAE. In principle, that makes Camera-Raw-style grading available *inside* the diffusion process rather than in post. Two things follow from that which post-processing cannot do. One is a higher dynamic range than a graded PNG allows. The other is the ability to steer generations into territory the model resists on its own — very dark or very bright frames, and reportedly even the morphology of objects. This works because the grade conditions the sample instead of correcting it afterwards.

**The node has shipped.** `muerrilla/ComfyUI-Colorcraft` supports the Qwen-Image VAE family, Krea 2 included, and the author tested it on both Krea 2 and Z-Image as intended `[community — muerrilla, github.com/muerrilla/ComfyUI-Colorcraft, 2026-08]`. It is a new capability class — grading inside sampling — not a replacement for the decode choice above. It does not change the §5a recommendation: the Wan swap and the Sharp decoders decide *how* the latent is rendered, Colorcraft decides *what* the sampler is steered toward. A/B it on a latent you have already paid for, the same way you would a decoder, and pin the version, because it is a single-maintainer pack weeks old.

## 6. Using LoRAs

- **Node:** `LoraLoaderModelOnly`. Krea 2 LoRAs are DiT-only, because the encoder is never trained (encoder-class doctrine). GGUF DiT + LoRA works through the same node.
- **Official style LoRAs:** run them at strengths 0.8–1.0. Their natural-phrase triggers are auto-appended by the template; `prompting-guide.md §5` has the verbatim table.
- **The official Turbo LoRA:** `loras/krea2_turbo_lora_rank_64_bf16.safetensors` in Comfy-Org/Krea-2 is the Turbo distillation *as a rank-64 LoRA*. Applied over **Raw**, it turns Raw into a few-step model; at partial strength it blends distillation speed with Raw's diversity. This is the enabling piece of the two-stage recipe below.
- **Trained LoRAs:** train on Raw, apply on Turbo — that is the official doctrine, covered in `lora-training.md`. Two numbers exist for character LoRAs. A 474-image LoRA holds identity at ~0.8 while stacking with style LoRAs `[community — JahJedi]`. Protocol-size LoRAs (13–32 images) ship at **1.0**, and a LoRA that only reads right at 1.1 is underfit rather than strong `[live-use — media lab, 21 runs, 2026-08 → 09]`. Sweep 0.8–1.1 per LoRA; there is still no per-type weight table beyond that, eleven weeks in. Deploy strength, the different-body-per-checkpoint effect and the inference rules that go with a character LoRA are in `lora-training.md §10`.
- **Slider/utility LoRAs** are already appearing (for example a Detail Slider on Civitai). Treat their weights per the author's card.

## 7. Multi-stage workflows

**Give any refine pass its own step budget.** A partial-denoise pass only runs the tail of the schedule. At denoise 0.30 on Turbo's stock 8 steps, that tail is about 2 effective steps. Two steps cannot resolve the re-noised image, so the pass produces artifacts. Raise the step count on the refine sampler itself — around 20 steps — so the denoised tail still has enough steps to work with. Do not copy the 2-step finish in §7a as a counter-example: that recipe tunes its finish stage deliberately, and it is not a default for a generic img2img refine.

### 7a. The two-stage Raw+Turbo-LoRA recipe (the best-documented local workflow)

The author publishes it as "Krea 2 simple gen workflow for high quality realism" with a full write-up, and claims it is "WAY better" for photoreal than stock Turbo `[community — nsfwVariant, Civitai]`.

| Stage | Model | Settings | Purpose |
|---|---|---|---|
| 1 — compose | **Raw + Turbo-LoRA @ 0.6** | 6 steps, `res_2s` / `beta`, full denoise | deliberately *undercooked* — keeps Raw's expressiveness and texture diversity, dodges the safety-tuned polish |
| 2 — finish | same | 2 steps, `deis_3m` / `bong_tangent`, **denoise 0.2** | resolves the undercooked noise without re-imposing the default look |
| decode | **Wan 2.1 FP32 VAE** | — | the §5a swap |

The author's companion numbers: the int8 convrot quant, and a cfg 2.0 variant for when negatives are needed (at 2× the time).

### 7b. Alternative sampler ladders (stock Turbo)

`[community — RaymondLuxuryYacht, Civitai "RLY Basic Photorealism"]` recommends `res_2s`/`beta` at **4–5 steps** for maximum texture, and `er_sde`/`simple` at **4–9 steps** for a cleaner look. Past 8 steps, gains are minimal at 1024 `[community — liutyi, tested to 12]`. But at higher resolutions the step economy shifts: `euler_ancestral`/`simple` at **15 steps, 1536×1792, cfg 1** (alternate: `uni_pc_bh2`, seeds batch of 2) is a named daily-driver `[community — m0ran1's sampler thread, r/StableDiffusion]`. A LoRA loading note: each active LoRA adds ~5–10 s per generation on Turbo (reported on int8 + rgthree Power Lora stacking). That overhead is more noticeable against an 8-step base than it ever was on slower models `[community — rarezin, r/comfyui]`.

### 7b'. Low-VRAM pixel-space ladder

For weak GPUs, generate at **512×512 and upscale in pixel space with realESRGAN 2×**, instead of rendering at native resolution. Mr Flow-style nodes adapted for Krea-2/ZIT keep detail while cutting compute `[community — MFGREBEL, RealRebelAI/Rebels_MrFlow]`.

### 7c. The full production ladder

`[community — lonecatone23, "Pro Grade" workflow]` chains: caption/enhance (abliterated local LLM) → base gen → detailer-daemon sampling → **SAM3 face/eye detailers** → `UltimateSDUpscale` at low denoise with a simplified prompt → post FX. The character-LoRA swap belongs at the detailer stage, not the base gen (`characters.md §4`). Note the author's own honesty about a limitation: the workflow's image-edit stage underperforms, because there is no real edit model yet.

### 7d. Turbo img2img: the denoise ladder

Community practice for img2img and repaint on stock Turbo, at its native 8 steps / cfg 1.0, has settled on three rungs: **0.4 stays close to the source, 0.55 is the balanced default, 0.7 is a strong restyle.** A packaged Civitai workflow adds face preservation on top `[community — Civitai 2768351, kombitz.com, 2026-07-19; convergent]`. Read the rungs against the step-budget rule at the top of §7: at 8 steps, denoise 0.4 leaves only ~3 effective steps, so these numbers assume you either accept the community's 8-step result or raise the refine sampler's own step count. Once a character LoRA is in the graph the numbers change — a face pass wants LoRA 1.0 at denoise 0.5 into a 1024² crop, 0.3 is too timid and 0.6+ drifts the hair (`lora-training.md §10`).

## 8. Krea 2 in mixed-model pipelines

Krea 2's role in a mixed pipeline is the **aesthetics/composition front-end**: it has the widest stylistic range and is strong on anatomy, animals and wide-aspect frames `[community — nsfwVariant's comparison]`. Its finishing partners:

- **Z-Image as the face/detail finisher and repair-inpainter.** Z-Image Base beats Krea 2 on facial expressiveness and hair, at ~8× the generation time. The efficient split is Krea 2 for the scene and Z-Image for the face pass. Krea 2's characteristic artefact zones (hair strands, fine repeating patterns, halftone-prone fabric, dark-area noise) inpaint cleanly with **Z-Image at denoise ~0.2** `[community — nsfwVariant]`. Paired LoRA releases — the same style in Krea-2 and Z-Image-Turbo versions — are already a Civitai pattern, and the community treats the two as a standard pairing.
- **The Wan 2.1 VAE** (§5) is itself a cross-family graft: a renderer from one family serving a generator from another.
- **The handoff rule** (suite-standard): **VAE-decode to pixels between model families**, because Qwen-Image-VAE latents are not Z-Image/SDXL/Flux latents. Identity-preserving refines live at denoise ~0.2–0.5, and 0.2 is the Krea2→Z-Image repair number above.
- No hosted↔open handoff subtlety is documented yet. But remember that hosted Large renders through the FLUX.2 VAE, so hosted and local outputs of "the same model" will not match pixel-level.

Cross-model craft in depth — denoise bands, resolution matching, color management, workflows-as-code — lives in the **[`image-production-workflows`](../../image-production-workflows/)** skill.
