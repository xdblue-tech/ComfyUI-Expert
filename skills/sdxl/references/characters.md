# SDXL Characters — creating a consistent character

This file covers how to invent an original character and keep them consistent across poses, outfits, scenes, and multi-character images. SDXL stands out here. It has the **deepest no-training identity toolbox of any open model** (InstantID, IP-Adapter FaceID, HyperLoRA, ReActor). It also has the most mature LoRA training stack. As a result, the craft is mostly about choosing the right tool rather than working around missing ones.

Training mechanics live in `lora-training.md`. Loading and stacking live in `checkpoints-and-loras.md §4`. ControlNet and IP-Adapter wiring lives in `setup-and-workflows.md §7`.

---

## 1. Choose the path: adapter, LoRA, or both

| Tool | What it does | Strengths | Weaknesses |
|---|---|---|---|
| **InstantID** | one reference photo → identity-locked generation (InsightFace embedding + its own ControlNet) | best one-shot face fidelity on SDXL `[community — displaced FaceID as the go-to]` | face only; degrades at extreme angles; adds its own ControlNet pass |
| **IP-Adapter FaceID / FaceID Plus v2** | identity via InsightFace embedding into cross-attention | lighter than InstantID; composes with other IP-Adapters | widely held to underperform InstantID; h94's own card concedes imperfect ID consistency |
| **HyperLoRA** (ByteDance, CVPR 2025) | **zero-shot generation of LoRA weights** from a face photo — adapter convenience, LoRA-grade quality | official ComfyUI nodes; "fidelity" vs "edit" checkpoints | newer, smaller install base `[official — ByteDance HyperLoRA repo]` `[community — validation growing]` |
| **ReActor** | post-hoc face *swap* on a finished image | fixes identity after the fact; the 2025 rewrite **dropped InsightFace** (removing its non-commercial licence problem) and added ReSwapper/HyperSwap models | swap quality ceiling; doesn't help body/outfit |
| **Character LoRA** | trained from 20–50 images | the only tool that carries the **whole character** — body, outfit, mannerisms — and survives angle extremes | needs a dataset and a training run |

**The decision rule** `[community — MyAIForce comparisons; convergent]`: adapters excel at one-shot *face* identity and need no training. However, they drift at extreme angles, and they capture nothing below the neck. **A trained LoRA wins for a recurring character; an adapter wins for a one-off.** The strongest results combine both approaches. Use a character LoRA for the character, and use InstantID or a detailer pass to lock the face in difficult shots.

---

## 2. The character LoRA pipeline

The 2026 cross-model consensus applies to SDXL directly `[community — WeirdWonderfulAI, Mickmumpitz, Civitai dataset guides; strong]`:

1. **Anchor image** — generate the character once in your target checkpoint (Juggernaut/RealVis for photoreal; Pony/Illustrious for anime). Give the character specific, nameable features, a plain background, and neutral light.
2. **Dataset factory** — multiply the anchor into **20–50 varied images**. The modern route uses an edit model ([Qwen-Image-Edit 2509/2511](../../qwen-image/)) as a data tool only. Its look does not contaminate the LoRA, because the captions name everything except the identity. The classic route is img2img plus inpainting inside SDXL itself. Generate about 60 images, then keep the best 30. Cut every frame where the face drifted.
3. **Coverage** — vary exactly one thing per image: the 8-point rotation (below), 3+ expressions, 2–3 outfits, several lighting setups, and a shot-size mix (close-up about 30%, full body about 20%, the rest medium or cowboy). Near-duplicates are the number one cause of same-face overfit.
4. **Caption in the checkpoint's dialect** — caption the residual, and add a **verbatim rare trigger token**. SDXL is a CLIP tag-matcher, so the trigger must be a literal token. For the photoreal family, use descriptive keywords. For Pony/Illustrious, use booru tags. Booru tags already include ready-made angle tags (`from_side`, `from_behind`, `from_above`, `looking_back`) that slot straight into the rotation series.
5. **Train** (`lora-training.md`) — use character rank 8–16. Train on base SDXL for cross-finetune compatibility, or on Pony/Illustrious if that family is the target.
6. **Evaluate** — run an XY grid (epoch × strength) on out-of-set prompts: a new outfit, a new setting, a profile view. The test passes when the face holds while everything else obeys the prompt.

**The 8-point rotation, keyword dialect.** Generate all eight images with identical subject and lighting keywords, and vary only the angle phrase: `front view, facing camera` · `three-quarter view from the right` · `right side profile` · `back three-quarter view from the right, sliver of cheek visible` · `view from behind, back of head and shoulders` · (mirror the three right-side phrases for left). Add one high-angle and one low-angle close-up per lighting setup. Back views are the weakest-trained angles in every diffusion model, so describe the back of the hair and the back of the outfit explicitly, and expect retries. The full protocol with sentence-form clauses is in the z-image skill's prompting guide §3.3–3.5; translate its clauses into keywords for SDXL.

---

## 3. Deploying: the detailer LoRA swap and `[SEP]` routing

**The detailer swap** `[community — MyAIForce ADetailer/FaceDetailer writeups; strong]`: generate the base image **without** the character LoRA. At full strength in the base pass, the LoRA drags composition and body proportions toward its training data. Instead, apply the LoRA only inside the **FaceDetailer/ADetailer** inpaint pass, where it gets the whole sampling budget on the face region. Set detailer denoise to about 0.4–0.5, and match the detailer prompt to the image. If the prompts don't match, you get the LoRA's default face.

**Multi-character via `[SEP]` routing** — ADetailer processes detected faces left-to-right and splits its prompt on the `[SEP]` token. Each face therefore gets its own prompt *and its own LoRA*:

```
Bobby <lora:bobby:0.9> [SEP] Tracy <lora:tracy:0.9>
```

The `[SEP]` syntax is documented upstream `[official — ADetailer discussion #533]`. In ComfyUI you build the same pattern manually: run per-detection detailer passes (Impact Pack SEGS), and load a different character LoRA in each one.

---

## 4. Style bleed — the block-weight fix (SDXL's unique lever)

Character LoRAs absorb their dataset's *style* along with the identity, and then stamp that style on every image. SDXL has a fix no DiT model has yet: **block-weighted LoRA application**. In the SDXL UNet, style concentrates in the **OUT1** (+IN08) attention blocks, while subject and identity live mostly in **OUT0**. Zeroing the style-carrying blocks at inference therefore strips the bleed *without retraining*. In ComfyUI, use Inspire Pack's **`LoraLoaderBlockWeight`** node, which supports per-block strength vectors and a preset sweep syntax. In A1111-family UIs, use the `lbw=` syntax. The named writeup argues that inference-time block editing beats lossy block-restricted *training*: you keep the full LoRA and choose per-image `[community — Civitai 5301, Inspire Pack docs; strong]`.

Prevention at dataset time still matters. Vary the dataset's lighting, background, and medium so that less style gets entangled in the first place.

---

## 5. Beyond the face

- **Signature outfit:** leave the outfit uncaptioned in the shots where it appears, and it binds to the trigger. Caption it in the shots where you want it swappable.
- **Multi-outfit LoRAs:** use a unique trigger tag per outfit, make the outfits visually distinct, and balance the per-outfit image counts. The practical ceiling is **~6 outfits per LoRA** before quality collapses `[community — Khanykov01, Civitai 6990; strong]`.
- **Multi-character scenes:** SDXL is the best-equipped open model here. Use three layers of defense, applied in order. (1) **Regional prompting / attention-couple masking** with per-region LoRA application. The classic Regional-Prompter approach works on SDXL's UNet, but it does *not* transfer to Flux-class DiTs. (2) **`[SEP]`-routed detailer passes** (§3) to re-assert each identity. (3) Keep the scene prompt to one distinguishing feature per character, which reduces bleed pressure. Attribute bleed — hair or clothing colors migrating between figures — is the expected failure. The layered defense manages it rather than eliminating it.

---

## 6. Failure modes & fixes

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| Identity collapses at profiles/back views | Front-heavy dataset | Add targeted angle images (booru angle tags make this easy on Pony/Illustrious); adapters drift hardest here — switch to LoRA for angle-critical work |
| Adapter face looks pasted-on / lighting mismatch | Face embedding overrides local lighting context | Lower adapter weight (~0.5–0.7); finish with a low-denoise detailer pass to re-blend |
| Same-face rigidity, training poses reproduced | Near-duplicate dataset shots; overtraining | Earlier epoch; more pose/expression variety; check the XY grid |
| Character LoRA restyles every image | Style entangled in the LoRA | Block-weight fix (§4); vary dataset style next training run |
| Two characters swap attributes | One conditioning stream without regional isolation | Regional prompting + `[SEP]` detailer routing (§5) |
| LoRA does nothing on Pony/Illustrious | Wrong dialect pool — base-SDXL LoRA on a drifted finetune | Match the LoRA's base family (`checkpoints-and-loras.md §4`) |

---

## Sources & confidence

Tool facts (InstantID/FaceID/HyperLoRA/ReActor repos, ADetailer `[SEP]`, Inspire Pack nodes) are `[official]` from the respective repos. ReActor's InsightFace removal and HyperLoRA's checkpoints are 2025 changes, so re-verify the current state before install. The craft — the detailer swap, the block-weight map, the multi-outfit ceiling, and the dataset rules — is `[community — MyAIForce, Civitai 5301/6990, dataset guides; convergent]`, and this skill states it with confidence per its two-bar policy. SDXL's character stack is mature and slow-moving compared with the DiT models'.
