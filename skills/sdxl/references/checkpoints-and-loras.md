# SDXL Checkpoints, LoRAs & Control Tooling

The defining practical fact about SDXL is that **you almost never run base 1.0 raw. You run a finetune of it.** This file maps that ecosystem. Everything here is **community tier** unless a licence is cited from a model card. Verify current versions and licences on the model page, because these move even though the base model doesn't.

## Contents
1. Why finetunes, and the two dialect families
2. The photoreal / general finetunes
3. The anime / booru finetunes (Pony, Illustrious/NoobAI)
4. Using LoRAs — loading any style / character / concept LoRA
5. The fast-variant LoRAs (stacking speed onto any finetune)
6. LoRA training → moved to `references/lora-training.md`
7. ControlNet & IP-Adapter catalog

---

## 1. Why finetunes, and the two dialect families

Base SDXL 1.0 is a strong foundation, but it looks undertrained, or "plasticky," next to community finetunes that trained it harder on curated data. Finetunes split into **two dialect families**, and the families share neither prompt style nor LoRA pools:

- **Photoreal/general family** — base, Juggernaut, RealVisXL, DreamShaper. Its **dialect** is descriptive photo-keywords (prompting-guide §1–5). LoRAs trained on base SDXL work here.
- **Anime/booru family** — Pony V6, Illustrious, NoobAI. Its **dialect** is tag-based (score tags / booru tags). It has a **separate LoRA pool**: base-SDXL LoRAs don't transfer, because these models drifted far enough from base that they're effectively their own ecosystems.

Match both the **dialect** and the **LoRA pool** to the checkpoint family.

---

## 2. Photoreal / general finetunes

| Checkpoint | Maker | For | Notes |
|---|---|---|---|
| **Juggernaut XL** | RunDiffusion | all-purpose **photoreal**, the default "just works" pick | strong skin/lighting/anatomy; frequent version bumps — check current. The v10 / "Juggernaut X" generation ships as **two separate checkpoints, SFW and NSFW**, so pick the file that matches the job `[official — rundiffusion.com/juggernaut-xl, 2026-09-09]` |
| **RealVisXL** | SG161222 | **maximum photorealism**, portraits, skin/hair | the realism specialist; v5 current line |
| **DreamShaper XL** | Lykon | artistic / fantasy / concept-art generalist | handles stylised scenes; looser prompts |
| **ZavyChroma XL** | — | general, vivid colour, realism/art balance | |
| **NightVision / epiCRealism XL / CrystalClear** | — | further photoreal options | similar dialect; try-and-compare |

All of these use the **photo-keyword dialect**. Most carry an OpenRAIL++-M licence or a permissive Civitai licence. Commercial use is often allowed, but with restrictions such as no model-merge resale. **Verify per model**, because Civitai licences vary by uploader.

---

## 3. Anime / booru finetunes

**Pony Diffusion V6 XL** is wildly popular and very flexible (anime/furry/cartoon, strong concept flexibility). It is its own sub-ecosystem.
- **Dialect:** start every prompt with the **score ladder** `score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up`, then a **source tag** (`source_anime` / `source_pony` / `source_furry` / `source_cartoon`), a **rating** (`rating_safe` / `rating_questionable` / `rating_explicit`), then Danbooru content tags.
- Normal photoreal prompts produce poor results on it. Pony **LoRAs are a separate pool**.

**Pony Diffusion V7 is not an SDXL model, and this file does not cover it.** It shipped on 2025-10-08 as a ~7B **AuraFlow** base with its own loaders and its own LoRA pool. It replaces the `score_9, score_8_up…` ladder with a tag-plus-description prompting scheme, and it trains on SimpleTuner rather than kohya_ss or OneTrainer `[official — purplesmartai/pony-v7-base card; Civitai article 6309]`. The V7 team itself calls V7 prompting inconsistent, with a fix promised for V7.1. **V6 XL remains the practical pick for SDXL-ecosystem work** while V7's LoRA pool and tooling mature `[flagged — re-verify]`. A V6 LoRA will not load on V7. When a Civitai page or a user says "Pony", check which generation before matching dialect or LoRA pool.

**Illustrious XL / NoobAI XL** are the leading **anime/illustration** bases, trained on Danbooru tag vocabularies. They are the current go-to for anime finetuning and LoRAs.
- **Dialect:** comma-separated **booru tags** (`1girl, solo, <character>, <series>, <attributes>`), plus the model's quality tags (`masterpiece, best quality` — these *do* work here, unlike on photoreal SDXL). NoobAI is an Illustrious-derived community continuation. Check its card for the exact quality-tag convention and any model-specific tags.
- This pool is separate from both base SDXL and Pony.

---

## 4. Using LoRAs — loading any style / character / concept LoRA

This is the generic path for any downloaded LoRA. The speed LoRAs in §5 and the ControlNet catalog are special cases; this section covers the everyday "I grabbed a style LoRA off Civitai" flow.

**Node wiring — SDXL LoRAs patch the UNet *and* the text encoders.** The newer DiT models (Flux, Z-Image) use LoRAs that touch only the diffusion model. An SDXL LoRA is different: it usually carries weights for the **UNet *and* both CLIP encoders** (CLIP-L + OpenCLIP-bigG). So use the **full `LoraLoader`** (MODEL + CLIP in/out), **not** `LoraLoaderModelOnly`:

```
Load Checkpoint → LoraLoader (model + clip) → KSampler (model) + CLIPTextEncode (clip)
```

Place it right after the checkpoint loader. Feed the patched MODEL to the sampler and the patched CLIP to your text-encode nodes. `LoraLoader` exposes **`strength_model`** and **`strength_clip`** separately. Start with both equal (for example, 0.8). Lowering `strength_clip` relative to `strength_model` reduces how hard the LoRA's vocabulary hijacks your prompt interpretation, which helps when a LoRA "takes over" the composition.

**Match the LoRA to the checkpoint's dialect family. This is the SDXL-specific rule that trips everyone.** SDXL has *separate LoRA pools* (§1). A LoRA trained on base or photoreal SDXL will misbehave on Pony or Illustrious, and the reverse is true too, because those finetunes drifted too far from base. **Check the LoRA's "base model" on its Civitai page and match the family:** base/photoreal ↔ photoreal finetunes; Pony ↔ Pony; Illustrious/NoobAI ↔ the Illustrious pool. A mismatched pool is the number one reason a LoRA "does nothing."

**Weight by LoRA type** `[community]` — always read the LoRA's own card, because authors publish a tested weight and trigger:

| Type | Typical `strength_model` | Notes |
|---|---|---|
| **Style** | 0.6–1.0 | lower if it flattens detail or fries color |
| **Character / subject** | 0.7–1.0 | higher to carry identity |
| **Concept / slider** | 0.3–0.8 (sliders go ±) | often subtle by design |
| **Detail / "add-detail"** | ±0.5–1.0 | many are bipolar — negative = smoother |

**Trigger words** go in the prompt, usually near the front, and must appear **verbatim**. SDXL is a CLIP tag-matcher, so the trigger is a literal token that the LoRA bound its concept to. Many style LoRAs need no trigger; most character or concept LoRAs do. This is the opposite of the LLM-encoder models (Flux/Z-Image), where a trigger folds into natural-language prose.

**Stacking multiple LoRAs.** Chain `LoraLoader` nodes (MODEL/CLIP out → next LoRA's in), or use the rgthree **Power Lora Loader** to hold several in one node with per-LoRA toggles. Keep the *combined* strength down. Two LoRAs at 1.0 each commonly fry the image, so drop each to about 0.5–0.8. Stacking across dialect pools — a Pony LoRA plus a photoreal LoRA — fights, so stay within one family. The speed LoRAs (§5) are the deliberate exception: they are built to stack onto any same-family checkpoint.

**Two failure signatures.** A LoRA that *does nothing* points to the wrong dialect pool, or a missing or misspelled trigger. A LoRA that *fries everything* points to `strength_model`/`strength_clip` set too high, or stacked totals too high. Lower them.

---

## 5. The fast-variant LoRAs (stacking speed onto any finetune)

The speed axis composes with the style axis because Lightning, LCM, and Hyper-SDXL all ship as **LoRAs**:

| LoRA | Steps | Sampler · Scheduler | CFG (ComfyUI / diffusers) | Notes |
|---|---|---|---|---|
| **SDXL Lightning LoRA** | 2 / 4 / 8 | `euler` · `sgm_uniform` | 1 / 0.0 | match LoRA file to step count; 4-step popular |
| **LCM-LoRA (SDXL)** | 4–8 | `lcm` · `sgm_uniform` | 1–2 / 1.0–2.0 | needs `ModelSamplingDiscrete`→`lcm`; most portable |
| **Hyper-SDXL LoRA** | 1 / 2 / 4 / 8 | `euler` · `sgm_uniform` | ~1 / ~0.0 | best 1-step; some modes use a unified-guidance LoRA |

**Workflow:** load your finetune → `LoraLoader` (the fast LoRA, strength about 1.0) → set the matching sampler, scheduler, CFG, and steps. For example, **RealVisXL + 4-step Lightning LoRA** gives a photoreal draft in 4 steps. You can also stack a *content* LoRA on top (character or style) by chaining multiple `LoraLoader` nodes. Keep the content-LoRA strength around 0.6–0.9, so it doesn't fight the speed LoRA. Note that distilled means guidance is off, so **negatives are inert** in these stacks.

---

## 6. LoRA training → `references/lora-training.md`

Training moved to its own file, matching the suite's layout. **`references/lora-training.md`** covers kohya_ss/OneTrainer, the convergent hyperparameter recipes (rank-by-type ladder, Prodigy, the alpha=2×rank myth), and caption-the-residual in the target dialect. It also covers **style-LoRA specifics** (the Illustrious recipe, dataset diversity, color-cast lock-in, the out-of-set acceptance test), LoKr, the good-citizen principle, and XY-grid evaluation. The full character pipeline (dataset factory, detailer deployment, multi-character) is in **`references/characters.md`**.

---

## 7. ControlNet & IP-Adapter catalog

Setup and wiring are in `references/setup-and-workflows.md` §7. This section is the model roster.

**ControlNet models for SDXL:**
- Stability's official SDXL ControlNets (canny, depth), plus the broader community set from **`xinsir`** (high-quality canny/openpose/scribble/depth), **`diffusers`**, **`kohya`** (the "control-lora" slim variants), and **`thibaud`** (openpose).
- **`xinsir/controlnet-union-sdxl-1.0`** — one model, many control types (the convenient default).
- Preprocessors: `comfyui_controlnet_aux` (DepthAnything, OpenPose, Canny, Lineart, SoftEdge, Normal, MLSD, etc.).

**IP-Adapter models for SDXL** (`h94/IP-Adapter`):
- `ip-adapter_sdxl` / `ip-adapter-plus_sdxl` — image-prompt / style transfer (Plus = finer detail).
- `ip-adapter-faceid_sdxl` / `ip-adapter-faceid-plusv2_sdxl` — identity transfer (needs InsightFace embeddings).
- Strength about 0.5–0.8; combine with ControlNet for pose plus identity. This is SDXL's standout capability: identity and structure control with no training, far ahead of most newer DiT models' tooling.

---

**Reminder on volatility:** finetune *versions* and Civitai *licences* change often, and new fast-variant LoRAs keep appearing. The base SDXL facts — architecture, the OpenRAIL++-M base licence, the ComfyUI graph — have been stable since July 2023. Treat the roster above as a snapshot, and check the model page before committing, especially for commercial use.
