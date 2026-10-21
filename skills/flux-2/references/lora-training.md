# FLUX.2 LoRA Training

> **Shared craft lives in [`character-lora-training`](../../character-lora-training/)**. That skill covers dataset coverage, caption-the-residual, evaluation, adult/NSFW base selection, and the real-person likeness rules that decide whether a LoRA is publishable. This file covers what is specific to this model.


> **Using** a LoRA — the loader node, variant compatibility, weights, stacking, and the Turbo accel-LoRA — is covered in `setup-and-workflows.md §7`. This file is only about **making** one. The full character pipeline (dataset factory, deployment, multi-character) is `references/characters.md`.

**Supported training bases:**
- [dev] 32B (Non-Commercial)
- [klein] 4B Base (**Apache 2.0 — use this for commercially-deployable LoRAs**)
- [klein] 9B Base (Non-Commercial; local training wants 32–48 GB VRAM `[community]`)

**Do not train on the distilled (4-step) variants.** Distillation removes the texture diversity that fine-tuning needs. Train on the corresponding base, then deploy on either.

## Tooling

**Ostris AI-Toolkit** is the default trainer. It supports [dev] and both klein bases. Here is a working YAML skeleton:

```yaml
job: extension
config:
  name: my_flux2_lora
  process:
    - type: sd_trainer
      training_folder: "path/to/your/images"
      trigger_word: "my_concept"      # optional on FLUX.2 — see captioning below
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      train:
        batch_size: 1
        steps: 2000
        train_unet: true
        train_text_encoder: false      # always false — encoders are frozen on FLUX.2
        learning_rate: 1e-4
        optimizer: adamw8bit
        lr_scheduler: cosine
      model:
        name_or_path: "black-forest-labs/FLUX.2-dev"
        is_flux2: true                 # verify this flag in current AI-Toolkit
        quantize: true
```

Other live options exist. **Civitai's official orchestration recipe** covers [klein] (`developer.civitai.com/orchestration/recipes/training-flux2-klein`). It runs on the ai-toolkit engine and includes an edit-training mode that uses `control_N/` reference folders. SimpleTuner works, and fal.ai and RunComfy offer hosted trainers. kohya/sd-scripts FLUX.2 support was still pending close to release, so verify it exists before relying on it.

> ⚠ **The Civitai klein-4B recipe defaults to dim 2 / alpha 1.** That is a cost-optimized floor for their hosted trainer, not a quality recommendation. Community ablation finds that rank 4–8 LoRAs barely move FLUX.2's fused attention/MLP blocks. For local training, start at 16 for a character and 32+ for a style. This is an official-platform default sitting against a named community ablation, and neither side is obviously wrong: Civitai is optimising their own GPU bill, and the ablation is optimising output `[contested]`.

## The official reference config

BFL publishes its own Klein training documentation. That makes this the one model in the suite with a **vendor-stated** LoRA recipe rather than a purely community-derived one:

| Parameter | Value |
|---|---|
| Rank | **64** |
| Alpha | **128** (i.e. `2×rank` — double effective LR, and a legitimate config despite the old "alpha must not exceed rank" folklore) |
| Batch (total) | **4** |
| Learning rate | **1e-5** |

`[official — BFL Klein training docs]`

BFL also describes the expected shape of a run: **15–40 images sharing one look, roughly 60 minutes** on a single GPU.

**Train on a Base (undistilled) variant.** Klein Base is the intended fine-tuning target precisely because it is undistilled. This is the same train-on-the-undistilled-variant rule that governs Z-Image and Krea 2. Guidance-distilled variants fight training.

**Klein 9B has a reported collapse pattern.** ostris/ai-toolkit#654 ("Flux Klein 9b training collapse", opened 2026-01-19) describes a 9B run collapsing by step ~250 and fully degenerating by step 500. As of 2026-09-09 it is one report with no maintainer response and no fix. If a 9B run degenerates rather than converging, treat it as a known class of problem rather than a dataset fault. Sample early on 9B: the reported collapse begins before the first checkpoint a 500-step interval would save `[community — ostris/ai-toolkit#654; single report, re-verify]`.

The `ai-toolkit-perceptual` character-training fork **defaults to the Klein 9B checkpoint** in its quickstart, so Flux.2 is the best-supported target for weight-noising and depth-anchoring experiments. For background, see [`character-lora-training`](../../character-lora-training/) §8 territory, and [`sdxl`](../../sdxl/references/lora-training.md) §8 for the full method.

## Hyperparameters

| Parameter | Character | Style | Notes |
|---|---|---|---|
| Rank (linear) | 16 (32 high end) | **32–128** | Herbst's 50+-run [klein]/[dev] ablation landed on **128/64/64/32** (linear/alpha/conv/conv-alpha, a 4:2:2:1 ratio) as "universally strong" for style, versus tool defaults of 16. The direction (style ≥ character) is agreed; the magnitude is contested `[community — Calvin Herbst, Medium]` |
| Alpha | = rank | = rank or rank/2 | Effective LR scales as `alpha ÷ rank`. `alpha = 2×rank` is legitimate — it just doubles effective LR. The "never exceed rank" rule is an SDXL-era myth |
| LR | 1e-4, drop to 5e-5 if frying | same | FLUX.2's big DiT is **LR-hypersensitive — start low**. [klein] tolerates 1e-4–5e-4. Herbst found weight decay 1e-5 is a surprisingly load-bearing color/tonality knob |
| Steps | 1500–3000 | 2000 (Civitai default) — Herbst's style sweet spot was **7000**, degrading past 10k | Almost certainly dataset-size and LR dependent — contested, so run checkpoints and read the grid |
| Batch | 1–4 | 1–4 | 1 + gradient accumulation is the safe floor |
| Dataset | 20–50 images | **20–30** (10–50 range) | DiT-class models need far fewer style images than SDXL folklore suggests; curation beats quantity |
| Text encoder | never | never | Mistral/Qwen3 are frozen; FLUX.2 LoRAs are model-only |

**How the knobs interact** (this is architecture-general): total steps ≈ images × repeats × epochs ÷ batch, and effective LR = `alpha ÷ rank` × LR.

## Dataset & captioning — caption the residual, in prose

FLUX.2's encoders are LLMs, so captions are **descriptive natural-language sentences, not tags**. The residual principle is unchanged: whatever you don't describe is what the LoRA absorbs.

- **Character:** describe everything that is *not* the identity — pose, clothing, scene, lighting, angle. Avoid a bare trigger token, because it can confuse the LLM encoder. If you use one, embed it naturally ("a photo of TRIGGER, a woman with…"). The dataset protocol (rotation, expressions, factory) is in `references/characters.md §3`.
- **Style:** invert the rule. Describe the **content** of each image across **diverse subjects** and never mention the style. The shared look becomes the residual.
- ⚠ **Captionless training is genuinely contested on Flux-class models.** The no-caption camp shows strong style replication from raw images alone. The pro-caption camp has the better-documented evidence for *generalization*: recris's "clown test" in kohya discussion #1497 shows that detailed captions transfer beyond the training distribution, while captionless LoRAs tend to replicate the dataset. The practical synthesis is **short natural-language scene descriptions that never mention the style** — caption-the-residual in prose. Captionless is defensible for a pure-replication style; captioned generalizes better. Both camps have named evidence, which is why this is presented as open rather than settled `[contested]`.

## Style LoRAs — the specifics

- **The diversity maxim:** keep consistency in the style and diversity in everything else — people, objects, interiors, landscapes. Otherwise the LoRA learns "this style = these subjects."
- **Palette discipline:** cover the style's tonal range, and keep B&W out of a color set. Narrow color statistics cause **color-cast lock-in**, where every output takes the dataset's average palette.
- **Resolution:** 1024² is standard. Training at 512 demonstrably works on Flux-class models for some named users, but "512 beats 1024" is unresolved `[contested — kohya #1497]`.
- **Inference strength:** FLUX.2 style LoRAs run hotter than SDXL's. Typical strength is 0.7–1.0. Herbst's ablated optimum for his style was 0.73, with a usable range of 0.4–0.75.
- **Ethics flag:** single-living-artist styles without consent are the community fault line, and Civitai requires real-artist disclosure. Prefer self-made, licensed, or historic/aggregate aesthetics.

**The style acceptance test:** the LoRA passes when the style is recognizable on subjects *not* in the training set. Point the trainer's sample prompts at out-of-set subjects, and include one prompt *without* any trigger to catch leakage early.

## Adult / NSFW work

**BFL filtered the pre-training data** for NSFW and unlawful content. That is a data decision, not a refusal mechanism. The model does not decline; it simply has thin coverage and produces poor anatomy. No conditioning trick changes that. In particular, swapping in an abliterated text encoder does nothing but perturb your conditioning — the mechanism is explained in [`character-lora-training/references/nsfw-training.md`](../../character-lora-training/references/nsfw-training.md) §1.

Consequences for training:

- **Community NSFW finetunes exist**, and they are the practical base if this is the target. Verify each one's licence and lineage separately: Flux's variant licences differ sharply, and a finetune inherits the constraints of whatever it was built on.
- **Training the capability in from a filtered base is expensive.** You are teaching coverage the model largely lacks, not adjusting a bias it already has. If adult work is the primary goal, [`sdxl`](../../sdxl/) and its purpose-built finetunes are a far shorter path.
- The licence split across Flux.2 variants makes this a **commercial-use question as well as a capability one**. Check which variant you trained on before distributing anything.

## Assessing fit — judge by images, not loss

Save **multiple checkpoints** — every 200–500 steps, since the best one is usually well before the final. Then generate an **XY grid of checkpoint × LoRA strength (0.1–1.0)** on fixed prompts spanning in-domain to out-of-domain subjects, and pick the "Goldilocks" cell.

**Then judge the grid blind, on a probe set you wrote before seeing results.** Tooling and protocol: [`character-lora-training/references/evaluation-and-tooling.md`](../../character-lora-training/references/evaluation-and-tooling.md).

| Signal | What you see | Fix |
|---|---|---|
| **Good fit** | Concept reproduced *with flexibility* — pose/scene/outfit all remain promptable | ship it |
| **Overfit** | Drift toward the training images: rigid poses, baked backgrounds. Style tells: composition memorization, training subjects appearing unprompted, color-cast lock-in | earlier checkpoint; more variety; lower rank |
| **Underfit** | Weak likeness / style won't transfer | more steps; check captions actually isolate the concept |

A sweet spot **below 1.0 is normal**, not a sign of overcooking. A modest, not-over-trained LoRA is also the one that **stacks** cleanly — one style plus one character is the reliable pairing (`setup-and-workflows.md §7`). If a style keeps hijacking composition, training-time layer targeting (`only_if_contains` / `ignore_if_contains` in AI-Toolkit) is the DiT-world analogue of SDXL's inference-time block-weight editing.

> **FLUX.2 training tooling is new (2026) and fast-moving.** The relationships above are stable, but the exact defaults are not. Verify against current AI-Toolkit FLUX.2 examples and the BFL "[klein] in 60 min" guide before a long run.
