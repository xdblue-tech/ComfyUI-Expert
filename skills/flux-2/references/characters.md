# FLUX.2 Characters — creating a consistent character

This file covers how to invent an original character and keep them consistent across poses, outfits, scenes, and multi-character images. FLUX.2 holds a distinctive position here: among open models, it is the one where the **no-training path is strongest**. It has native multi-reference conditioning (`ReferenceLatent`, up to ~8–10 images) plus a working PuLID. So the first question is whether you need to train at all.

Tool wiring lives in `controlnet-and-identity.md`. LoRA training mechanics live in `lora-training.md`. Loading and stacking live in `setup-and-workflows.md §7`.

---

## 1. Choose the path

| Path | Tool | When it wins |
|---|---|---|
| **Multi-reference editing** (no training) | `ReferenceLatent` chains — feed 1–10 images of the character, and FLUX.2 [dev] re-renders them into new scenes | When one to a few images of the character exist, when the scene or outfit changes, and when you want the fastest iteration. Community tip: prefer [dev] over Flux.1 Kontext for character consistency, lock seeds, and drop conflicting references. The multi-reference mechanics are official template behaviour; the preference and the two tips are craft `[community]` |
| **PuLID** (no training) | iFayens `ComfyUI-PuLID-Flux2`, strength 1.0–1.4 | When you need face identity from a single portrait, especially one you can't train on. It supports [dev] and both [klein] sizes. The natively-trained weights are **klein-first** (`pulid_flux2_klein_v1/v2`), and calibration differs between base and distilled variants. It is text-to-image only as of v0.6.2 (checked 2026-09-09; edit mode is still on the roadmap) `[official — iFayens/ComfyUI-PuLID-Flux2]` |
| **Character LoRA** (training) | AI-Toolkit on a **base** (non-distilled) variant | When a recurring character must survive angle extremes, carry a body, outfit, and mannerisms, and stack with style LoRAs |
| **Combinations** | Multi-ref or a LoRA for the character, plus PuLID or a detailer pass to lock the face. ControlNet (Fun Union pose) plus PuLID for posed identity shots | Production work. The paths compose because they all output standard conditioning |

**The cross-model decision rule** `[community — MyAIForce]`: adapters and reference conditioning excel at one-shot likeness with zero setup, but they drift at extreme angles and under heavy restyling. A trained LoRA is what carries the *whole* character robustly. FLUX.2 softens this rule, because multi-reference covers many cases that needed a LoRA on other models. It does not repeal it.

---

## 2. Multi-reference as the character engine

The native workflow uses the ComfyUI [dev] image-edit template (`setup-and-workflows.md §2`):

1. **Build a small reference set** of 2–6 images: a clean front portrait, a three-quarter, a full-body, and optionally the signature outfit, either flat or worn. A character sheet generated once — front, side, and angled on one canvas, or as separate frames — doubles as this bundle. The Mickmumpitz-class "consistent character creator" workflows are the established pattern for this `[community — Mickmumpitz-class workflows; strong]`.
2. **Chain one `ReferenceLatent` per image**, then prompt the new scene. Label the references semantically in the prompt ("the woman from the reference images sits at…"). BFL's marketing says 10 references, and the prompting guide says 8. Treat ~8 as the safe number `[contested]`.
3. **Iterate with fixed seeds.** If a reference fights the prompt — for example, its background keeps leaking — bypass it. `ReferenceLatentPlus` (shootthesound) adds per-image strength and timestep gating when you need one reference to dominate.
4. **[klein] 9B KV** is the batch variant. It caches reference K/V states, so a fixed character bundle re-renders against many prompts at roughly 1.5–3× speed (`setup-and-workflows.md §4`).

Multi-reference is also the **dataset factory** for the LoRA path. One anchor image becomes many consistent variants — angles, outfits, expressions, lighting — which you curate into a training set. Generate about 60, keep the best 30, and cut every frame where the face drifted. [Qwen-Image-Edit 2509/2511](../../qwen-image/) is the other widely-used factory tool: a different model used purely for data manufacturing. WeirdWonderfulAI's Qwen-Edit dataset writeup is the canonical version of this pipeline `[community — WeirdWonderfulAI; strong]`.

---

## 3. The character LoRA pipeline

1. **Generate the anchor image** in FLUX.2 itself: front three-quarter, neutral light, plain background, and specific nameable features ("silver-grey cropped hair, a freckle above the left brow"). These features are what the LoRA will absorb.
2. **Build the dataset:** 20–50 images via the factory above. Coverage should include the **8-point rotation** — front, both three-quarters, both profiles, both back three-quarters, and back — with identical descriptions and only the angle clause varying. Add one high- and one low-elevation close-up, 3+ expressions, and 2–3 outfits. Mix shot sizes: close-up about 30%, full-body about 20%. Vary exactly one thing per image. Back views need explicit back-of-hair and outfit description, plus retries. The full clause-by-clause protocol is in [`z-image`](../../z-image/references/prompting-guide.md) §3.3–3.5. It is written as natural-language sentences, which is exactly FLUX.2's dialect, so reuse it directly.
3. **Caption in prose, and caption the residual:** describe pose, clothing, scene, lighting, and angle in natural sentences. Leave the identity undescribed so it binds to the concept. Do not use a bare trigger token. If you want one, fold it into the sentence: "a photo of TRIGGER, a woman with…" FLUX.2's LLM encoders read description better than made-up tokens (`setup-and-workflows.md §7`).
4. **Train on a base variant.** Use [klein] 4B Base for commercially-deployable LoRAs (Apache 2.0), or [dev] for max quality. Never train on the 4-step distilled variants. Start low, with character rank ~16 and LR 1e-4. Details are in `lora-training.md`.
5. **Evaluate** with an XY grid (checkpoint × strength) on out-of-set prompts: a new outfit, a new setting, a profile view. The LoRA passes when the face holds while everything else obeys the prompt.
6. **Deploy with the detailer swap** `[community — established cross-model technique; strong]`. Generate the base image *without* the character LoRA, because at full strength in the base pass it drags composition toward its training data. Then apply the LoRA inside a FaceDetailer pass at denoise ~0.4, with the detailer prompt matched to the image. Impact Pack detailers are model-agnostic and run FLUX.2 fine.

---

## 4. Beyond the face

- **Signature outfit:** leave the outfit uncaptioned in images where the character wears it, so it binds to the identity. Caption it in images where it should stay swappable. For one-off wardrobe changes, multi-reference — image 1 as the character, image 2 as the garment — replaces wardrobe training entirely.
- **Multi-outfit LoRAs:** use a unique trigger phrase per outfit, keep the outfits visually distinct, and balance the image counts. The practical ceiling is **~6 outfits**. Khanykov01's guide is written for SDXL, but the capacity logic is architecture-general `[community — Khanykov01]`.
- **Multi-character scenes:** FLUX-class models have one real regional tool: **mask-based attention masking**, merged into ComfyUI core (PR #5942). The SD-era Regional Prompter approach does *not* work on the DiT. Use a layered defense: (1) regional attention masks per character, (2) per-face detailer passes, each loading its own character LoRA, and (3) one distinguishing feature per character in the scene prompt. The core PR is official `[official]`. The craft layer on top of it comes from the community and is still forming. How well per-region *LoRA* application works on DiT attention masking is still contested `[contested]`, so expect retries.

---

## 5. Failure modes & fixes

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| Identity collapses at profiles/back views | The dataset is front-heavy, or an adapter is at its angle limit | Add targeted angle images via the factory. For adapters, switch to the LoRA path — angle robustness is its core advantage |
| Multi-ref output blends the references into a stranger | Conflicting references (different haircuts/ages) average out | Curate the bundle so the era and haircut are consistent; drop the weakest reference; raise the cleanest portrait's strength (ReferenceLatentPlus) |
| A chain of edits (face, then outfit, then background) slowly warps the parts you never touched | Each pass re-encodes the previous output, so small deviations compound across iterations | Keep the original reference in the bundle on every pass, so each edit is anchored to the source rather than to the last output. A [klein] 9B edit-mode "Consistency LoRA" is reported to lock unmodified structure across passes; one Civitai write-up so far `[community — Civitai article 27410; single report]` |
| PuLID face looks pasted on or ignores scene lighting | The identity embedding overrides local context at high strength | Drop strength toward 1.0, then finish with a low-denoise detailer pass to re-blend |
| Same-face rigidity, dataset poses reproduced | Near-duplicate shots; overtraining | Pick an earlier checkpoint; add more variety; re-read the XY grid |
| Character LoRA restyles every image | Style is entangled with identity in a one-look dataset | Vary dataset lighting and medium. The SDXL block-weight fix has **no established FLUX.2 equivalent**, so prevention at dataset time is the lever `[flagged — no DiT block map yet]` |
| Two characters swap attributes | A single conditioning stream with no isolation | Regional attention masks plus per-face detailer LoRAs (§4) |
| LoRA won't load at all | Variant mismatch — [dev] vs [klein] 4B vs 9B LoRAs are not interchangeable | Match the LoRA to its exact training variant (`setup-and-workflows.md §7`) |

---

## Sources & confidence

The hard facts — ReferenceLatent/KV templates, PuLID files and limits, the attention-masking core PR, variant incompatibility — are `[official]` from ComfyUI templates and the respective repos. PuLID-Flux2 is a fast-moving single-maintainer project, so re-verify its current weights and limits before you install it. The craft — reference-bundle curation, the dataset factory, the detailer swap, multi-character layering — is **named-community** (Mickmumpitz, WeirdWonderfulAI, MyAIForce, Civitai authors), convergent across models, and stated with confidence. FLUX.2-specific numbers are early and flagged where contested.
