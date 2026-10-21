# SDXL LoRA Training (kohya_ss / OneTrainer / ai-toolkit)

> **Shared craft lives in [`character-lora-training`](../../character-lora-training/)**. That skill covers dataset coverage, caption-the-residual, evaluation, adult/NSFW base selection, and the real-person likeness rules that decide whether a LoRA is publishable. This file covers what is specific to SDXL and its finetune families.

> **Using** a LoRA (loading, dialect pools, weights, stacking) is covered in `checkpoints-and-loras.md §4–5`. This file is only about **making** one. The full character pipeline is `references/characters.md`.

SDXL LoRA training is the most mature corner of open-source image generation. It has years of accumulated recipes, several solid trainers, stable behaviour, and by far the deepest body of published craft. The numbers below are convergent recipes drawn from named guides. Note that LR and **rank in particular are genuinely contested** across reputable sources, so treat them as starting points rather than fixed answers.

## Contents

1. [Choosing the base — the decision that dominates](#1-choosing-the-base)
2. [Tools](#2-tools)
3. [Hyperparameters](#3-hyperparameters)
4. [Dataset architecture — the identity ratio](#4-dataset-architecture--the-identity-ratio)
5. [Captioning](#5-captioning)
6. [Dataset traps](#6-dataset-traps)
7. [Style LoRAs](#7-style-loras)
8. [Advanced: weight noising and depth anchoring](#8-advanced-weight-noising-and-depth-anchoring)
9. [Stacking several LoRAs of the same character](#9-stacking-several-loras-of-the-same-character)
10. [Assessing fit](#10-assessing-fit)
11. [Adult / NSFW work on SDXL](#11-adult--nsfw-work-on-sdxl)

---

## 1. Choosing the base

**This choice decides your quality ceiling, your tagging dialect, and which existing LoRAs you can stack with.** It matters more than any hyperparameter. It is also not reversible, because a LoRA belongs to the family it was trained on.

| Base | Dialect | Position |
|---|---|---|
| **SDXL 1.0 / photoreal finetunes** | Descriptive keyword phrases | Maximum compatibility across the photoreal family. The right choice for realism work |
| **Pony Diffusion V6 XL** | Booru tags **plus `score_X` quality tags** | Enormous ecosystem. The score-tag system is divisive: it becomes part of the training captions and **does not transfer** to other families `[community]`. **The last SDXL Pony** — see the V7 row |
| **Pony Diffusion V7** | Tags plus natural-language descriptions | **Not an SDXL finetune.** A ~7B AuraFlow model, shipped 2025-10-08, with its own LoRA pool. **It trains on SimpleTuner**, not kohya_ss or OneTrainer, so none of the recipes in this file apply as written. The V7 team flags its prompting as still inconsistent pending a V7.1 fix `[official — purplesmartai/pony-v7-base card; Civitai article 6309]`. **Train on V6 XL for SDXL-ecosystem work** until V7's pool matures; a V6 LoRA does not load on V7 `[flagged — re-verify]` |
| **Illustrious** (v2.0 as a finetune base) | Booru tags | **The largest character-LoRA library.** The default choice for character work if you want it to work with existing community LoRAs |
| **NoobAI-XL V-Pred 1.0** | Booru tags | Reported as the most anatomically accurate, with the best tag comprehension. **It requires v-prediction sampler settings and Euler specifically — other samplers will not work.** Budget an evening for that alone |
| **WAI-NSFW v17** | Booru tags | The usual easier-setup runner-up to v-pred NoobAI |
| **Anima** | Booru tags | **Not an SDXL finetune.** Anima is a separate 2B model that happens to speak the same booru dialect, so it needs its own trainer config and its own LoRA pool. Named trainers describe it as becoming "the new Illustrious," and they note that LoRA quality there varies widely because the pool is still young `[community — re-verify]`. **The weights are non-commercial**, unlike everything else in this table, and your LoRA is a Derivative that carries that restriction too. It is not a flat bar, though: §2(c) lets a person **operating in an individual capacity** sell the LoRA weights themselves. What no one may do is ship those weights inside a larger product or serve them behind a paid API. The images the LoRA makes stay unrestricted for everyone, individual or company. The full shape is in [`anima`](../../anima/) `[official — CircleStone NC licence §1(a)/§2(c)/§2(e)]` |

**Cross-family transfer is partial at best.** Some NoobAI LoRAs are reported to work acceptably on Illustrious, because the two lineages share ancestry `[community — re-verify]`. Pony V6 is the outlier: its score-tag conditioning means a Pony-trained LoRA carries assumptions that no other family shares. Pony V7 is not a transfer question at all, because it is a different architecture.

**The honest heuristic is simple: train on the checkpoint you will actually generate with.** If you do not know yet, pick Illustrious for anime or stylised character work, and a photoreal SDXL finetune for realism. Both have the largest pools of compatible LoRAs to stack with, which makes them the safe defaults.

---

## 2. Tools

- **kohya_ss** (`sd-scripts`) is the de-facto standard, with both a GUI and a CLI. Its GitHub **discussions** are a primary craft source, because named users publish reproducible experiments there.
- **OneTrainer** has a friendlier UI, good defaults, and is the most-requested target when new techniques get ported.
- **ai-toolkit** (Ostris) is increasingly the default for newer architectures, and it is the base for the experimental fork in §8.
- **kohya_ss now spans SDXL and Anima natively.** Its sd-scripts changelog lists LoKr/LoHa support for "SDXL/Anima" together, plus multi-resolution bucketing, so the two booru bases in §1 share one trainer `[official — kohya-ss/sd-scripts changelog, 2026]`. Pony V7 is the exception: it trains on **SimpleTuner** against AuraFlow (§1).
- **Civitai's on-site trainer** is the no-local-GPU path for SDXL and Illustrious. It bills in buzz per step, and its defaults differ from kohya's in three ways that matter `[community — Civitai 21257; single report; re-verify]`: set **repeats to 1 and scale batch size instead**, because inflating repeats "lies to the optimizer"; the only augmentation on offer is **flip**; and **Prodigy always trains the text encoder there**, whatever the TE setting says. Do not copy this file's kohya numbers into it verbatim.

**Text-encoder training is contested.** SDXL LoRAs *can* train the CLIP encoders; that is why `LoraLoader` exposes `strength_clip`. kohya's default TE LR is 5e-5, which is half the UNet's. The Illustrious style recipe runs TE at 0.5 Prodigy-relative. Other trainers disable TE entirely for styles. Training the TE binds vocabulary harder, which helps trigger-heavy characters, but it costs more prompt-hijacking when the LoRA is loaded. Both camps have named, reproducible advocates. `[contested]`

---

## 3. Hyperparameters

**The convergent photoreal/general recipe**, which ViewComfy, Bieler and the Civitai cheatsheets independently land on, is: **rank 32 / alpha 32, AdamW or AdamW8bit, constant scheduler, 0% warmup, LR ~3e-5** (usable ~3e-6 to 8e-5), batch 1–4, 1024-area bucketed images.

**Prodigy** is the default across Pony/Illustrious training guides. Set LR to 1.0 and it self-tunes; use a d-coef of ~0.5 for styles. It is the best answer to "I don't want to tune LR" `[community]`.

**Rank is genuinely contested. The disagreement is worth understanding, not averaging away:**

| Source | Position |
|---|---|
| The classic rank-by-type ladder `[community]` | Simple character **8**, complex/realistic character **16**, concept 16–32, style **32** (up to 128/α64 for very detailed styles) |
| Named character-LoRA trainers | **48/48** as a working default, with **48–64** prescribed specifically when a LoRA "forgets" at weights below 0.5 `[community — neonkisu, QuantumBogoSort]` |

Both positions are defensible. Lower rank soaks up less unwanted background and style, so it makes a better-behaved stacking citizen. Higher rank captures fine identity detail and holds up at reduced weight. **If your LoRA needs 1.0 to work, that is the signal to raise rank**, not to accept 1.0.

**Steps.** There are two anchors, and they agree better than they look:

- **~80–100 steps per image** is the community rule of thumb, corroborated across SDXL and newer architectures `[community — QuantumBogoSort and others]`. Twenty images therefore means roughly 1600–2000 steps.
- **~3000 total** is the SDXL/Illustrious *style* anchor `[community — Civitai 25645, L3n4]`. Characters usually converge sooner.

**How the knobs interact:** total steps ≈ images × repeats × epochs ÷ batch. **Effective LR scales as `alpha ÷ rank`**, so alpha = rank means no scaling. The old rule that "alpha must never exceed rank or it burns" is a myth. `alpha = 2×rank` is a legitimate config that simply doubles the effective LR.

**VRAM and batch.** Around 12 GB works if you use gradient checkpointing plus an 8-bit optimizer; 16–24 GB is comfortable. When you drop batch size to fit in memory, **hold `batch × gradient_accumulation` constant.** Batch 1 with accum 4 behaves like batch 4, just at roughly a third of the speed `[community — QuantumBogoSort]`.

**LyCORIS.** **LoKr** has the strongest *style* reputation: it gives better texture fidelity at much smaller file sizes, at some cost in trainability and portability. **LoKr factor 8** is the setting used in the current experimental character recipes (§8). kohya's sd-scripts now lists LoKr and LoHa as supported for SDXL and Anima alike `[official — kohya-ss/sd-scripts changelog, 2026]`. DoRA outperforms plain LoRA in academic benchmarks but stays niche. Plain LoRA and LoKr cover nearly all practical 2026 work.

**Resolution.** 1024 is the point of diminishing returns. Training at 1536 and 2048 is analytically better, but for most work it is not perceptibly better `[community — QuantumBogoSort]`.

---

## 4. Dataset architecture — the identity ratio

This is the most useful published structure for character sets. It turns "dataset variety" into something you can actually count `[community — neonkisu, Civitai]`.

**Every image plays one of two roles:**

| Role | Share | What it is | What the LoRA learns |
|---|---|---|---|
| **Identity-emphasis** | **~33%** | Close-up portraits, head-and-shoulders, clean backgrounds. The character fills the frame | *Who this person is* |
| **In-context** | **~67%** | Full-body across varied environments, poses, outfits, scenes. The character is doing something somewhere | *How this person renders in the wild* |

Anywhere from **0.25 to 0.4** works. What matters is that both categories are present and meaningfully sized. The two failure directions tell you what went wrong:

- **Mostly close-ups** produces great faces, but body type, outfit consistency and pose flexibility are lost. The LoRA "knows the face and treats everything below the neck as a stranger." This is the classic selfie-style character LoRA.
- **Mostly in-context** holds body and outfit, but the face drifts toward generic when the prompt zooms in.

**Verify the split by counting.** Sort the set into the two categories and count them. This takes a minute, and it catches the most common structural dataset fault.

Size: **20–30 images for a single-outfit character** is the current Illustrious-era figure `[community — NanashiAnon]`; use 30–60 across both categories for a fuller character. New trainers routinely overestimate this, because older guides quoted much larger numbers.

---

## 5. Captioning

Caption in the **target dialect**: booru tags for Pony/Illustrious/NoobAI/Anima, descriptive keyword phrases for photoreal SDXL. SDXL's CLIP needs the **trigger verbatim**: a rare literal token, placed deliberately (`skw man` binds the man, `skw suit` binds the suit).

**The rule, stated precisely:** caption the things you want to remain **promptable later**. Leave out what should fuse into the trigger.

**But captioning is downstream of the dataset, and this is the part most guides miss.** Here is a hard-won correction from a published post-mortem:

> Identity features were stripped from the captions so they would bind to the trigger — that worked, the face locked. Hair colour was captioned in every single image so it would stay free — and it fused to the trigger anyway, because **all sixteen images had the same hair colour**. *"Describing it is not enough when it never varies in the data."* `[community — Ainara, Civitai]`

**So you cannot caption your way out of missing variety.** Captioning something only makes it separable if it actually varies across the set. If you want hair colour promptable, the dataset needs more than one hair colour. Otherwise, accept that it belongs to the character.

**Use consistent identity tokens across both dataset categories.** The same identity tags should appear on identity-emphasis and in-context images alike. That repetition is the signal that says "this character is always this person" `[community — neonkisu]`.

**Put off-spec negatives in the captions**, not only in the inference-time negative prompt. If the character has ice-blue eyes, tagging `brown_eyes, green_eyes` as negatives during training reduces colour flipping later `[community — neonkisu]`.

**Caption dropout is the strongest single lever for generalisation, and most character LoRAs skip it.** Setting `caption_dropout ≈ 0.3` randomly drops 30% of caption tokens each step. Without it, the LoRA learns the *whole caption* as its trigger condition. That is why a character trained on "library, reading" struggles at "beach, swimming": the environment gets tangled up with the identity. With dropout, the model learns each token's meaning more independently, and the trigger alone activates the character in combinations it never saw `[community — neonkisu]`.

**Prepend / append structure**, for trainers that expose it (Civitai's own trainer does) `[community — RONK234, Civitai]`:

| Slot | Put here |
|---|---|
| **Trigger** | The character's name, and nothing else. Add other traits *after* training, once tested |
| **Prepend** | Permanent physical traits only — hair colour, eye colour, hairstyle, **body type** |
| **Append** | Genuinely minor details. **Never put important character details here** — appended tags are pushed to the back and deprioritised |

There are two traps in that structure:

- **If the dataset mixes colour and monochrome images, keep colour terms out of the prepend.** They confuse how the model interprets monochrome inputs, which is a real problem for manga/manhwa sourcing.
- Use the append slot for something you want to toggle at inference. Tagging source styling (`manga`, `manhwa`) there lets you switch panel styling in or out of the positive prompt later.

Auto-tagger settings: **max ~30 tags, minimum threshold ~0.4**, plus a standard quality blacklist (`bad quality`, `worst quality`, `deformed`, `mutation`, `blurry`) `[community — convergent across SDXL captioning guides]`.

**Body tokens belong in the captions.** Height, build and proportions are part of an identity. Omitting them is a documented cause of a LoRA whose face is right and whose body is wrong, and that matters most where the body is actually visible. Put them in the **identity-emphasis** captions too, not only the in-context ones `[community — neonkisu, Civitai 31467]`.

**Pony-specific:** its `score_X` quality tags participate in training captions and must **honestly match** the image's quality, or they destabilise training.

**Subject masking**, where the trainer supports it, masks out the background. If you use it, captions must describe **only the character, never the setting**. Mixing masked training with setting descriptions gives the model contradictory signal `[community — QuantumBogoSort]`.

---

## 6. Dataset traps

Here are three traps that have cost people whole training runs, all from published post-mortems:

**A reference sheet cut into pieces is not a dataset.** Sixteen crops from one character sheet is still one lighting setup, one background, one camera, one moment. *"A LoRA cannot learn variety it has never seen."* Everything else on this list follows from that `[community — Ainara]`.

**Never pad to square with a solid colour.** Padding non-square crops to 1024 with white teaches the model the white. The result is a pale border baked into every generation. This is not a rendering artefact, and **no negative prompt removes it**, because from the model's point of view the bar is simply part of what this character looks like. **Crop square; do not pad.**

**Train at the aspect ratios you will generate at.** Training exclusively on 1:1 and then generating portrait degrades the output. Use bucketing across the ratios you actually want.

Also: **dedupe hard.** Twenty frames pulled from one video clip look like twenty images and behave like one.

---

## 7. Style LoRAs

The governing maxim: **consistency in the thing you're training, diversity in everything else.** A style set must show the style on varied subjects — people, objects, interiors, landscapes — or the LoRA learns "this style = these subjects."

- **Size:** ~50 minimum is the Illustrious-era recommendation. The legacy "300–500 ideal" figure is early-SDXL folklore that curation has since replaced. **A well-curated 30–50 beats a poorly curated 500** `[community — L3n4's crash course, Civitai 25645]`.
- **The Illustrious style recipe** `[community — Civitai 25645]`: dim 32 / alpha 32, Prodigy (UNet 0.5 / TE 0.5), cosine scheduler, ~3000 steps, ~50 images, booru captions capped at ~30 tags, no style tags in captions.
- **Invert the captioning:** caption the **content** of each image and never mention the style. The shared look then becomes the residual that the LoRA learns.
- **Palette discipline:** include the style's full tonal range, and keep B&W out of a colour style set. Narrow colour statistics cause **colour-cast lock-in**, where every output takes the dataset's average palette.
- **Ethics flag:** a single living artist's style trained without consent is the community's sharpest fault line, and Civitai requires real-artist disclosure. Prefer self-made, licensed, or historic/aggregate aesthetics.

**The style acceptance test:** the style is recognisable on subjects *not* in the training set. Point sample prompts at out-of-set subjects, and include one sample **without** the trigger to catch style leakage early.

---

## 8. Advanced: weight noising and depth anchoring

This is an experimental method with published results and an active feedback loop. It is worth knowing because it targets the exact failure that plagues character runs `[community — QuantumBogoSort, ai-toolkit-perceptual fork]`.

**Weight noising** injects a small Gaussian disturbance directly into the LoRA weights at every training step. The intuition is that it helps the model *forget* what is inconsistent and keep only what the data agrees on. Mechanically, it biases training toward flatter loss minima and spreads learning across more singular directions of the LoRA factorisation (a measured +20% stable rank on the same config). The practical effect is **resistance to the memorisation that overcooks character runs**, with better likeness at the same step count.

The reference config from the published comparison: batch 4, LR 5e-5, buckets 512/768/1024, **LoKr factor 8**, AdamW8bit, 1200 steps with the best checkpoint at **750**, on an **8-image** dataset. Noise **sigma 0.0125** gave the best results. The right value is believed to scale with dataset and batch size, and it is not yet mapped.

**Expect mid-training weirdness.** Body horror and extra limbs during the run are *normal* here, because the noise explores latent space more aggressively before converging. The heuristic: if you sample every 25 steps and see continuous body horror for more than ~20% of the run, sigma is too high. Lower it in 0.0025 increments.

**Status on SDXL:** the ai-toolkit-perceptual repo now lists SDXL as supported for both weight noising and depth anchoring. But it gives only the generic sigma range, **0.01–0.017**, with no SDXL-tuned recipe distinct from Flux or Z-Image `[official — ai-toolkit-perceptual README, 2026-09-09]`. Start inside that range and sweep; the 0.0125 above was not established on SDXL `[flagged — re-verify]`.

One more thing is emerging from the same discussion: the **Rose optimizer** (stateless, lower VRAM, better reported generalisation) needs a **much higher LR — around 1e-3 rather than 1e-4.** Its overfitting also shows up as minor artefacting rather than the usual memorisation, so save checkpoints more frequently when using it. `[community — ECF630; contested, early]`

---

## 9. Stacking several LoRAs of the same character

This is a widely-reproduced finding worth knowing before you train at all: **for a popular character with several existing community LoRAs, stacking them at reduced strength beats any one of them at 1.0** `[community — featherless_fiend]`.

| LoRAs stacked | Strength each |
|---|---|
| 2 | ~0.55 |
| 3 | ~0.425 |
| 4 | ~0.35 |
| 5 | ~0.275 |

The reported gain is **flexibility.** A single LoRA pushed to 1.0 follows prompt actions less, drops quality, and makes more mistakes. The averaging effect has theoretical support in the **Model Soups** line of work (arXiv 2203.05482).

Two consequences follow. First, if your character already has good community LoRAs, **try stacking before training.** Second, if you *are* training one, expect it to be used this way, so build a good citizen (§10).

---

## 10. Assessing fit

**Judge by images, not loss.** The loss curve barely predicts image quality.

Save **checkpoints throughout** — every 200–500 steps on a conventional run, every 25 on an experimental one — because the best checkpoint is usually well before the final one. Then build an **XY grid of epoch × strength (0.1–1.0)** on fixed prompts spanning simple to complex and in-domain to out-of-domain, and pick the Goldilocks cell. Generating the grid in ComfyUI gives a truer picture than trainer preview samples.

**Then judge the grid blind.** The grid is labelled by design, so you already know which cell trained longer before you look at it. For which comparison tool to use, the shuffled-candidate pass, a reusable probe set, and how to score likeness with numbers, see [`character-lora-training/references/evaluation-and-tooling.md`](../../character-lora-training/references/evaluation-and-tooling.md).

**A concrete ship criterion** `[community — neonkisu]`: if the LoRA holds identity at **0.7+ weight across three or more clearly different environments**, ship it. Do not over-optimise the first version. Feedback from real use is faster than another isolated round of iteration.

**Train a good citizen if it will be stacked.** Use modest rank, do not over-train, and accept a sweet spot **below 1.0.** That is what lets a LoRA coexist with speed LoRAs and other content LoRAs. Sub-1.0 strength is normal, not a sign of undercooking.

| Signal | What you see | Fix |
|---|---|---|
| **Good fit** | Concept reproduced *with flexibility* — pose, clothing, scene remain promptable | ship that checkpoint |
| **Overfit** | Drift toward training images: rigid poses, baked backgrounds, fried colour. Style tells: composition memorisation, training subjects appearing unprompted, colour-cast lock-in | earlier epoch; more dataset variety; lower rank |
| **Underfit** | Weak likeness, style won't transfer | more steps or higher LR; check captions actually isolate the trigger |
| Face right, body proportions drift | Too little in-context — identity ratio too high | Add full-body shots; lower ratio toward 0.25 |
| Body consistent, face generic on close-ups | Too little identity-emphasis | Add close-ups; raise ratio toward 0.4 |
| Renders only in training environments | Caption dropout too low or zero | Set `caption_dropout` ~0.3 |
| A captioned attribute fuses to the trigger anyway | That attribute never varied in the dataset | Add variety; captioning alone cannot fix it |
| Pale border or edge artefact in every output | Padded images in the dataset | Recrop square; retrain |
| Forgets the character below ~0.5 weight | Rank too low, or under-converged | Rank 48–64; or more epochs |

**ControlNet and IP-Adapter often substitute for training** when you need pose or identity on a one-off, with no training run needed (`setup-and-workflows.md §7`, `characters.md §1`).

---

## 11. Adult / NSFW work on SDXL

SDXL is where this is deepest in the whole open-weights field, and base choice does nearly all the work. See §1 and [`character-lora-training/references/nsfw-training.md`](../../character-lora-training/references/nsfw-training.md) for the cross-model treatment.

SDXL-specific points:

- **Pick a base with the training data in mind.** Use NoobAI-XL V-Pred for anatomical accuracy (with the v-pred/Euler constraint), Illustrious for the largest compatible character-LoRA pool, Pony V6 XL for breadth (V7 is not SDXL; §1), or WAI-NSFW for an easier setup. Training an explicit character on vanilla SDXL 1.0 fights the base model the whole way.
- **Booru dialect means explicit tagging is natural.** These finetunes were trained on tagged data, so tag the explicit content plainly and specifically. Vague captions cause the content to fuse into the character, producing a LoRA that can only ever be explicit.
- **Include clothed images** if the character needs to be renderable clothed. This follows from the same rule.
- **Body tokens belong in the identity-emphasis captions too.** Build and proportions are part of the identity, and omitting them is a documented cause of "outfit and body type wrong" `[community — neonkisu, Civitai 31467]`.
- **Publishing is the binding constraint, not capability.** Real-person likeness is prohibited outright on Civitai regardless of rating, and NCII of real people is federally enforced in the US. See [`character-lora-training/references/publishing-and-likeness.md`](../../character-lora-training/references/publishing-and-likeness.md) before sourcing a dataset.
