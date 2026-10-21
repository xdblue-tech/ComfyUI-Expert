# Qwen-Image — LoRA training

> **Shared craft lives in [`character-lora-training`](../../character-lora-training/)** — dataset coverage, the identity ratio, caption-the-residual, evaluation, adult base selection, and the real-person likeness rules that decide whether a LoRA is publishable. This file covers what is specific to Qwen-Image: the trainers and their Qwen defaults, why there is no Raw/Turbo doctrine here, the 1328-class resolution question, the plain-name trigger, Edit LoRAs on paired data, adult work, and what does not transfer from Krea 2.

This file is about **making** a LoRA. Loading and stacking one is `setup-and-workflows.md §7`; the character pipeline end to end is `characters.md`. Trainer facts were read from each repo's docs and example configs on 2026-09-09; community recipes are attributed inline. **Nothing here is `[live-use]`.** This lab has not trained a Qwen LoRA, and every media-lab number cited by [`krea-2`](../../krea-2/) is Krea evidence (§9).

## Contents

1. Trainers — the support matrix and each trainer's Qwen defaults
2. The doctrine — one undistilled base, Lightning stacked at inference
3. Dataset resolution — the 1328-class question
4. Captions and the plain-name trigger
5. Size, composition, rank, LR, steps
6. Edit LoRAs on paired data
7. Adult / NSFW work and the encoder-refusal question
8. Hyperparameter table and the contested points
9. What transfers from Krea 2, and what does not
10. Evaluation and debugging
11. Qwen-Image-2.1 — DiffSynth only, RGBA datasets, and the licence

---

## 1. Trainers — the support matrix and each trainer's Qwen defaults

| Trainer `[official — each repo's docs and examples, 2026-09-09]` | Qwen-Image | 2512 | Edit 2508 | 2509 | 2511 | Layered |
|---|---|---|---|---|---|---|
| **ai-toolkit** (ostris) | yes | yes | yes | yes | yes | — |
| **musubi-tuner** (kohya) | yes | not named | yes | yes | yes | **yes** |
| **diffusion-pipe** | yes | not named | yes | not named | not named | — |
| **SimpleTuner** | yes | not named | "Qwen Edit" under reference-input training | | | — |
| **OneTrainer** | listed; no per-model doc found | | | | | |
| **DiffSynth-Studio** | yes | yes | yes | yes | yes | yes |
| **kohya sd-scripts** | **not a Qwen trainer** — its only Qwen artefact is a 2D VAE flag | | | | | |

When people say "kohya" for Qwen they mean **musubi-tuner**.

### 1.1 ai-toolkit

Three shipped configs: `train_lora_qwen_image_24gb.yaml`, `train_lora_qwen_image_edit_32gb.yaml`, `train_lora_qwen_image_edit_2509_32gb.yaml`. **No 2511 or 2512 example exists** despite both being supported; adapt the 2509 file or use the UI. T2I config, condensed:

```yaml
network:  { type: lora, linear: 16, linear_alpha: 16 }
datasets: [{ caption_dropout_rate: 0.05, cache_latents_to_disk: true, resolution: [512, 768, 1024] }]
train:    { batch_size: 1, cache_text_embeddings: true, steps: 2000, gradient_checkpointing: true,
            train_text_encoder: false, noise_scheduler: flowmatch, optimizer: adamw8bit, lr: 1e-4, dtype: bf16 }
model:    { name_or_path: "Qwen/Qwen-Image", arch: qwen_image, low_vram: true,
            quantize: true,    qtype: "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors",
            quantize_te: true, qtype_te: qfloat8 }
```

- **`qtype: uint3` plus an accuracy-recovery adapter.** A 20B DiT on 24 GB means 3-bit weights and a matching recovery adapter. **Each variant names its own adapter** (`qwen_image_edit_torchao_uint3`, `qwen_image_edit_2509_torchao_uint3`); the wrong one is a silently degraded run, not an error. Whether uint3 costs LoRA quality against bf16 is unmeasured.
- **Rank 16 / alpha 16** in all three configs; `train_text_encoder: false` ("probably won't work with qwen image").
- The Edit configs add `steps: 3000`, **`timestep_type: "weighted"`**, and a dataset `control_path` (a **list of up to three** folders for `qwen_image_edit_plus`, i.e. 2509). `QwenImageEditPlusModel` does not resize control images by default.
- Raise `max_step_saves_to_keep` above the shipped 4, or you lose the rungs you need to grid.

### 1.2 musubi-tuner — the fullest documented path

```bash
accelerate launch --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
  --dit … --vae … --text_encoder … --model_version original --dataset_config …toml --sdpa \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.2 \
  --optimizer_type adamw8bit --learning_rate 5e-5 --gradient_checkpointing \
  --network_module networks.lora_qwen_image --network_dim 16 \
  --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 --output_dir out --output_name my-lora
```

- `--model_version` ∈ `original` / `edit` / `edit-2509` / `edit-2511` / `layered`. It must be right at **cache** time too, because Edit's text cache is built with the control images in the prompt. Two pre-cache scripts, both required.
- **`--discrete_flow_shift 2.2`, deliberately low**: Qwen's inference shift "is set quite low… so a lower value than other models may be preferable". **`--timestep_sampling qwen_shift`** reproduces the resolution-dependent inference shift per sample, "typically around 2.2 for 1328x1328 images". Use it for bucketed sets.
- **LR 5e-5**, half the ai-toolkit figure; dim 16.
- **bf16 files only.** The fp8 repacks are rejected: `fp8_scaled` encoder, `fp8_e4m3fn` Edit DiT, `fp8mixed` Layered DiT. Quantise inside the trainer with `--fp8_base --fp8_scaled` and `--fp8_vl`.
- **VRAM at 1024², batch 1:** 42 GB → fp8 **30 GB** → `+ --blocks_to_swap 16` **24 GB** → `+ --blocks_to_swap 45` **12 GB**; 64 GB system RAM recommended when swapping.
- The docs' own disclaimer: "The appropriate settings for each parameter are unknown. Feedback is welcome."

### 1.3 diffusion-pipe, SimpleTuner, DiffSynth

**diffusion-pipe:** `type = 'qwen_image'`, `transformer_dtype = 'float8'`, **`timestep_sample_method = 'logit_normal'`** (a third schedule family), rank 32, example resolution **640**. "Use bf16 files even if you are casting the transformer to float8; fp8_scaled weights won't work at all." Edit is "the same as Flux-Kontext"; references must roughly match the target aspect. Output is **ComfyUI format**.

**SimpleTuner:** `lora_rank` **8 or lower** at 24 GB, `standard` or `lycoris` (LoKr), LR 1e-4, **`flow_schedule_shift 1.73`**, and int2-quanto or nf4-bnb base precision at 24 GB. The blunt fact: **the Qwen2.5-VL encoder alone is ~16 GB before quantisation**, so on a 24 GB card the encoder, not the DiT, breaks naive setups.

**DiffSynth-Studio** ships a script per variant, all with `--max_pixels 1048576` (an area cap = 1024²), `--learning_rate 1e-4`, **`--lora_rank 32`**, and this target list:

`to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1`

That is the clearest published statement of what a Qwen LoRA touches: a dual-stream MMDiT, image stream and text stream, plus two output projections. Edit training adds `--data_file_keys "image,edit_image" --extra_inputs "edit_image"`. **`--zero_cond_t` is required for Edit-2511 and only 2511.** The script's own comment is the whole documentation: "This is a special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model." It has no counterpart in ai-toolkit's or musubi's documented surface `[flagged — re-verify]`. It matches `zero_cond_t: true` in 2511's transformer config.

### 1.4 Hosted trainers and LoRA formats

Replicate lists `qwen/qwen-image-lora-trainer`; fal hosts Qwen endpoints and a LoRA shelf. Hyperparameters are managed; the dataset doctrine below still applies (`api-and-hosted.md §4`).

Formats: diffusion-pipe → ComfyUI keys by design. musubi → sd-scripts keys, which ComfyUI loads; it does **not** emit diffusers keys (issue #442, open). ai-toolkit → diffusers-style PEFT keys, which named authors ship and load in ComfyUI. DiffSynth → its own keys with the `pipe.dit.` prefix stripped. **Most "key mismatch" reports are variant mismatches** (`setup-and-workflows.md §7`).

---

## 2. The doctrine — one undistilled base, Lightning stacked at inference

**Qwen-Image is undistilled.** `guidance_embeds: false` on every variant; there is one base per generation plus an optional inference-time speed LoRA from a different team. So the Krea question "which checkpoint do I train on?" has no referent here. There is **no Raw/Turbo split, no distillation to train against, and no de-distillation training adapter**. No trainer documents one, and no source states a "train on base, infer with Lightning" rule in so many words. It is what every official config implies and what every named author does `[community — Nyao, Substantial_Angle680, acekiube; convergent]`. Nyao's manga-style LoRA (ai-toolkit, 2,750 steps, LR 2e-4, no trigger) shows every sample "generated in 4 steps (with the lightning lora by lightx2v)". Substantial_Angle680's style LoRA "works with lightning loras aw, working weight is 0.8–1.2". acekiube generates the training set itself on Edit-2509 + Lightning. Top_Buffalo1668 runs character LoRAs at 12 steps, CFG 1.

**The dissent is about quality, not doctrine.** "All the lightning loras / distils for Qwedit (that I've tested) are terrible… it makes people's skin look like plastic." That author's alternative is fewer base steps, "just set it to 10 steps instead of 20" while iterating `[community — nsfwVariant]`. A second author sees plastic skin at 12 steps with **no** Lightning loaded, so plastic skin is partly a base property that Lightning worsens (contested; SKILL.md two-bar section). **Practice:** train on the bf16 base, and grid the checkpoints on the base at 20 steps *and* on your real deploy graph. Never judge a character LoRA only through a Lightning graph.

**The fp8 grid is a LoRA hazard, not only a Lightning one.** Any LoRA on the downcast `qwen_image_fp8_e4m3fn.safetensors` can grid, because the interaction is base-quantisation × LoRA (`setup-and-workflows.md §4`). **Training your own Lightning LoRA:** asked publicly for Edit-Plus, zero replies.

---

## 3. Dataset resolution — the 1328-class question

| Source | Value | Framing |
|---|---|---|
| ai-toolkit | `[512, 768, 1024]` | shipped default, all three configs |
| DiffSynth | `--max_pixels 1048576` (= 1024² area) | shipped default |
| musubi | 1024² | the VRAM benchmark size |
| diffusion-pipe | **640** | "a resolution the model was trained on"; also the 24 GB lever |
| SimpleTuner | 512/768 → 1024 | explicitly VRAM-driven |
| Icy_Upstairs3187 | **1440 px @ 4:5**, a 1440 bucket added | matched the subject's Instagram size |
| FarTable6206 | "assuming your dataset is 1024px or higher" | the rank-32 verdict is conditioned on it |

**The Qwen-specific fact underneath:** the supported aspect set is the **1328 class (~1.5 MP)**, and musubi's `qwen_shift` quotes "around 2.2 for 1328×1328". Qwen's inference band sits *above* 1024, unlike Krea 2 Raw, which is 1K-native. That is why 1440 is defensible here and would be an error on Krea 2. Every shipped default is at or below 1024, and the low end is always framed as VRAM, never as quality. **Nobody has A/B'd 1024 against a 1328-class bucket** `[contested]`. 1024 is the safe, tested floor; a 1328-class bucket is the untested match to the model's band.

**Photographic sources want maximum sharpness.** "Always use the highest resolution and sharpest images you can… blurry, compressed, or low-resolution images will poison the LoRA" `[community — AwakenedEyes, LoRA primer v2]`. This does not conflict with Krea 2's "render at 1024 native, never above": that rule governs *generated* sources on a 1K-native base. No Qwen report reproduces the hi-res-source hazard, and none refutes it (§9).

---

## 4. Captions and the plain-name trigger

**Length is four-way contested**, all from authors who shipped something `[community — Icy_Upstairs3187, FarTable6206, acekiube, Nyao]` `[contested]`:

- **Verbose** — "longer/descriptive ones worked best"; 79 images, GPT-5 wrote them (Icy_Upstairs3187).
- **30–50 words** — "I tried everything from only tag & 10-word tags to 100-word descriptions. Medium-length works best", with a named trap: "Qwen VL (the text encoder) can get confused by too much noise" (FarTable6206, the only author who claims to have tested the range).
- **One word**, the character's name — "verbose captioning doesn't seem to be necessary to get good likeness" across dozens of LoRAs on Flux, Qwen and Wan (acekiube).
- **None** — two shipped *style* LoRAs (Nyao, Substantial_Angle680). A different job, not direct counter-evidence.

**Everyone training a character agrees on the residual rule.** FarTable6206's Qwen-specific version: never write `face`, `head`, `eyes`, `mouth`, `lips`, `nose`, because "the text encoder tries to learn what a mouth is instead of just learning 'this person's mouth'". Background as one word, clothing as style plus colour, and composition only when unusual, because "if you label 'front view' on every single photo, the LoRA will get stuck in that view". The same author avoids `woman` / `girl` / `man` / `person` to stop the base's generic prior pulling in. That contradicts this suite's shipped practice ("a woman named zciara" on Krea 2), so it is part of the caption dispute above, not adopted.

**The trigger is a plain name inside a sentence.** "Unlike Flux, Qwen doesn't really vibe with the single trigger word → description thing… it works better as a natural human name inside a normal sentence. Good: 'A beautiful Chinese woman named Kayan.' Bad: 'TOK01 woman'" `[community — Icy_Upstairs3187]`. Hearmeman98's shipped prompts use `Sydney01` in ordinary prose; acekiube writes the bare name as the whole caption. This is the encoder-class rule, reached independently on Qwen. The Krea 2 report that rare tokens surface as watermarks has **no Qwen equivalent**.

**Concept LoRAs invert it.** Adult concept LoRAs use short leet tokens (`l1ck`, `creamp1e`) and they work. A concept needs a handle the base has no prior for; a character needs the base's person-prior to cooperate.

---

## 5. Size, composition, rank, LR, steps

| Author | Images | Composition | Steps | Trainer |
|---|---|---|---|---|
| FarTable6206 (character) | **40–60** (tested 30–100) | **60 / 30 / 10** close / half / full; 70% front | 80–100 repeats/img → 3,000–6,000 | ai-toolkit, H200 |
| Icy_Upstairs3187 (character) | **79** | **33 / 33 / 33** | **6,000** | ai-toolkit / fal / Replicate |
| AI_Characters (realism style) | 18 | — | custom polynomial scheduler with a minimum LR | ai-toolkit, H100 |
| acekiube (character) | 20, all Edit-generated | face angles only | — | musubi / ai-toolkit |
| Nyao (style) | 44 | — | 2,750 | ai-toolkit |
| ProGamerGov (360° concept) | ~100k + 64k regularisation | — | 2.3M; rank 128 | — |

Everything below is FarTable6206's character recipe unless another author is named `[community — FarTable6206, r/StableDiffusion, 2026-01]`.

**Composition is the sharpest conflict with this suite.** "Many people tell you to include lots of full-body shots. Don't. If you have too many full-body shots (20–30%), your LoRA will produce blurry or distorted faces when you try to generate full-body images." 60/30/10 is close to the inverse of [`character-lora-training`](../../character-lora-training/)'s one-third identity / two-thirds context, and of Icy_Upstairs3187's 33/33/33 on the same model. It agrees with the sibling's *mechanism* (identity is learned per face-scale) and with AwakenedEyes' "at least 50% headshots". No A/B exists `[contested]`.

**Rank.** "Rank 32 is the consensus. Why not 16? It skips fine details… Why not 64? It over-generalizes." DiffSynth ships 32. **But every shipped default is 16** (ai-toolkit ×3, musubi), or 8 on SimpleTuner at 24 GB. Rank failure on Qwen is described as *over-generalisation*, not overfit. Rank 128 is in use for concepts.

**LR.** 1e-4 everywhere except musubi (5e-5) and the character specialists (2e-4): "0.0002 is actually more efficient for real people… this high LR works best when paired with the sigmoid timestep". Nyao's style LoRA also used 2e-4. The safest framing: 2e-4 is conditioned on `sigmoid`.

**`timestep_type: sigmoid` plus a `constant` scheduler for characters.** "Sigmoid is miles ahead of weighted… for character LoRAs, sigmoid is the only way to go", attributed to the toolkit author's own video. Ostris's shipped Edit configs use `weighted`, so this is a character-specific override.

**Batch 1, grad-accum 1** — "larger batches tend to generalize the character too much". This contradicts a well-argued Krea 2 report that grad-accum 2 is "severely underrated". Different models; possibly both right.

**Steps.** "80–100 repeats per image is the gold standard… under 50 repeats and it doesn't look like the person; over 100 and the quality starts to break or get 'fried'." That is a band with both edges named. Count is not the lever: named authors span 18 → 100 images with no monotone relationship.

**LoKr** is available (SimpleTuner `lycoris`, ai-toolkit) and **no named Qwen character author uses it**. The one prominent Qwen LoKr is `SNOFS`, an adult concept LoRA. LoKr enthusiasm in this suite is Krea 2 and Z-Image evidence.

---

## 6. Edit LoRAs on paired data

On Qwen an Edit LoRA is usually a **capability** LoRA, a new verb (swap, repose, relight, colourise, upscale) built from synthetically paired data, not a character LoRA. The two recipes with real numbers run far longer than a character run, which is what teaching an operation rather than a face should cost.

**Paired-data layout.** musubi uses parallel directories matched by filename `[official — musubi dataset_config.md]`:

```toml
[[datasets]]
image_directory    = "/targets"    # the AFTER images
control_directory  = "/controls"   # the BEFORE images
resolution         = [1024, 1024]
control_resolution = [1024, 1024]  # "We strongly recommend specifying this value."
```

Multiple controls take numeric suffixes (`a_0.png`, `a_1.png`). **Only one control on plain Edit (2508)**; multi-control needs `edit-2509` or `edit-2511`. `control_resolution` matters because musubi otherwise resizes controls to the *target's* resolution and aspect, a silent quality loss. ai-toolkit: `control_path`, a list of up to three folders. DiffSynth: two metadata columns (`image,edit_image`) plus `--zero_cond_t` for 2511. diffusion-pipe: "same as Flux-Kontext".

One community claim has no official confirmation. **Edit-2511 in ai-toolkit requires a 1024×1024 solid black image as the control map**, and uses 30–50% more VRAM than 2512 because it processes that map even when blank `[community — FarTable6206; single report]` `[flagged — re-verify before a paid run]`.

**What people train Edit LoRAs for:**

| LoRA | Job | Numbers |
|---|---|---|
| **BFS — Best Face Swap** (NRDX) | face swap ("Focus Faces") and head swap ("Focus Head") | **5,500+ steps**; the 2511 build ships a pure train and a 2509 × 2511 merge the author rates better for expression range; ~71k downloads |
| **Multiple-Angles (2511)** (fal) | 96 camera poses | **3,000+ pairs from Gaussian Splatting**; trigger `<sks> [azimuth] [elevation] [distance]` |
| **Next Scene (2509)**, **InScene** (PetersOdyssey) | next shot, same character; consistent shots in a scene | prefix `"Next scene:"`; navigate by green rectangles |
| **AnyPose (2511)**, **Pose Transfer V2** (kingroka) | ControlNet-free posing | the prompt is the whole instruction sentence |
| **PanelPainter (2509)** | manga colourisation | **~7k steps on ~7.5k pairs**; deploy 0.45–0.6 |
| **ICEdit LoRA (2511)**, **Qwen-Image-i2L** (DiffSynth) | infer A→B and apply to C; image → LoRA | official; i2L's own authors flag weak generalisation |

**The prompt is part of the dataset.** BFS ships its trigger as a whole paragraph of instruction, and `Picture 1: / Picture 2:` is the format Edit was trained on (`prompting-guide.md §3`). Edit-LoRA captions must be written in that format, or the LoRA will not fire the way the base expects.

---

## 7. Adult / NSFW work and the encoder-refusal question

- **Base behaviour: safety-tuned, not stripped.** The community frames it as "uncensoring": `MCNL` is "a convenient way to uncensor Qwen with many concepts included". A large working adult LoRA layer means the base weights carry the anatomy; the reticence is tuning. Same read the suite records for Krea 2.
- **26% of Qwen LoRAs are adult-flagged** (422 of 1,637; the character-tagged subset 31% explicit) against Krea 2's 52% `[community — Civitai API, 2026-09-09]`.
- **The encoder cannot refuse.** It is loaded "the SAME way that any other model loads any other text encoder… purely processing, with absolutely none of the typical Qwen chat format personality being 'alive'" `[community — ZootAllures9111]`. Written about Z-Image, and true of how Qwen-Image consumes Qwen2.5-VL. No report exists of Qwen2.5-VL refusing a prompt. What users experience as censorship is **data suppression in the diffusion weights**: vague anatomy, not an error.
- **Abliterated encoders do not help.** They exist (`dummy9996/Qwen2.5-VL-7B-abliterated_nvfp4_int8convrot_comfyui`, `huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated`). The author of the leading abliteration tool: perturbed hidden states "either [have] no effect at all, or the effect of reducing prompt adherence and potentially introducing artifacts. But it will never, ever remove censorship from the output" `[community — -p-e-w-, creator of Heretic]`. The leading Qwen-Edit craft author agrees: "Use only the normal FP8 text encoder with Qwedit; abliterated/GGUF encoders will reduce your output quality" `[community — nsfwVariant]`. A second-hand minority says the opposite, with no A/B (SKILL.md two-bar section).
- **Named artefacts** `[community — Civitai API, 2026-09-09]`: `SNOFS` (Ashen3, ~249k downloads, the most-downloaded Qwen LoRA of any kind, a LoKr), `MCNL` (leet triggers), `Mystic XXX`, `QWEN 4 PLAY`, `SexGod NSFW … 2511` (15k monthly HF pulls), per-anatomy enhancers. Several ship **cross-base lines** (Qwen, Krea 2, Z-Image, Flux, Chroma) from one dataset. `Qwen-Rapid-AIO-LiteNSFW` (Phr00t) is an AIO Edit checkpoint with the capability baked in. Check for an existing LoRA or AIO checkpoint before training.
- **The nude-reference protocol is a craft recommendation for SFW work too**, and needs no LoRA (`characters.md §3`).
- **Publishing gates bind harder than capability**: Civitai's real-person ban, the TAKE IT DOWN Act, and BFS's own card asking users not to post swaps of public figures. Owned by [`character-lora-training/references/publishing-and-likeness.md`](../../character-lora-training/references/publishing-and-likeness.md).

---

## 8. Hyperparameter table and the contested points

What each named source states, not a recommendation. `—` = silent.

| Source | Base | Rank / alpha | LR | Optimizer | Resolution | Steps | Shift / timestep |
|---|---|---|---|---|---|---|---|
| ai-toolkit example | Qwen-Image | 16 / 16 | 1e-4 | adamw8bit | `[512, 768, 1024]` | 2,000 | flowmatch, no `timestep_type` |
| ai-toolkit example | Edit, Edit-2509 | 16 / 16 | 1e-4 | adamw8bit | `[512, 768, 1024]` | 3,000 | `timestep_type: weighted` |
| musubi-tuner | any | dim 16 | **5e-5** | adamw8bit | 1024² | 16 epochs | `shift` 2.2, or `qwen_shift` |
| DiffSynth | every variant | **32** | 1e-4 | — | `max_pixels 1048576` | 5 epochs × repeat 50 | `--zero_cond_t` for 2511 |
| diffusion-pipe | Qwen-Image | 32 | — | automagic | 640 | — | `logit_normal` |
| SimpleTuner | Qwen-Image | ≤ 8 at 24 GB | 1e-4 | optimi-lion / adamw-bf16 | 512/768 → 1024 | — | `flow_schedule_shift 1.73` |
| FarTable6206 (character) | 2512, Edit-2511 | **32** | **2e-4** | — | ≥ 1024 | 80–100 repeats/img | `sigmoid` + `constant`; batch 1, GA 1 |
| Icy_Upstairs3187 (character) | Qwen-Image | — | — | — | 1440 @ 4:5 | 6,000 | — |
| Nyao (style) | Qwen-Image | — | 2e-4 | — | — | 2,750 | — |
| ProGamerGov (concept) | Qwen-Image | 128 | — | — | 2048×1024 | 2.3M | nf4 32 ep → int8 16 ep |

The first six rows are the trainers' shipped defaults; the last four are named community authors (§5).

**The contested points**, held by named authors with no same-dataset A/B `[contested]`:

- caption length (§4)
- LR 5e-5 / 1e-4 / 2e-4, with 2e-4 conditioned on sigmoid
- rank 16 (shipped) vs 32 (character consensus) vs 128 (concepts)
- training resolution 512/640/768 vs 1024 vs 1440, with the 1328-class bucket untested
- composition 60/30/10 vs 33/33/33 vs the suite's one-third
- grad-accum 1 (Qwen) vs 2 (Krea 2)
- `weighted` vs `sigmoid`
- how much of plastic skin is Lightning's
- whether uint3 plus the recovery adapter costs quality
- one same-dataset Qwen-vs-Z-Image comparison, in which ZIT wins realism and adherence while Qwen wins concept bleed and multi-LoRA stacking `[community — Top_Buffalo1668; single report]`

---

## 9. What transfers from Krea 2, and what does not

Krea 2 is Qwen-Image-derived and decodes through the same `qwen_image_vae.safetensors`, so *setup* transfers further than *doctrine*. Anything measured in the media lab's Krea runs is **Krea evidence** and stays marked as such in [`krea-2`](../../krea-2/).

**Transfers.** DiT-only with the encoder frozen. Plain-name-in-prose triggers. Caption-the-residual, sharpened here to "never write face / eyes / mouth / nose". A distilled inference path costs skin and hair detail. The Qwen VAE decode is an artefact source you must not blame on the LoRA. **Validate on the deploy graph with a strength-0 control**, reinforced, because Qwen has *three* base-artefact sources (halftone, plastic skin, fp8 grid) that one strength-0 render separates. Adult work is a data question. Count is not the lever. And the literal VAE file.

**Does not transfer:**

| Krea 2 doctrine | Why it has no referent, or is wrong, on Qwen |
|---|---|
| "Train on Raw, run on Turbo" | No Raw/Turbo split; Lightning is an inference-time LoRA |
| The Ostris turbo *training* adapter | No Qwen equivalent; do not invent one by analogy |
| `discrete_flow_shift 2.5` / `krea2_shift` | Qwen's figure is **2.2** and deliberately low; the sampler is `qwen_shift`; SimpleTuner uses 1.73. Carrying 2.5 across is a real error |
| LR 1e-4 as settled | Qwen's band is 5e-5 → 2e-4 |
| Rank/alpha 32 as "the" setting | Every shipped Qwen default is 16 (8 at 24 GB on SimpleTuner); 32 is a community character finding |
| "Render sources at 1024-native, never above" | Measured on *generated* sources against a 1K-native base. Qwen's band is the **1328 class**, and a named author trained at 1440. No Qwen measurement of the hi-res-source hazard exists either way |
| "Rare tokens surface as text and watermarks" | Krea 2, single report; Qwen reaches the plain-name conclusion by a different route |
| `gradient_accumulation: 2` as a free win | The strongest Qwen character report prescribes GA 1 |
| LoKr as the character default | Krea 2 / Z-Image evidence; no named Qwen character author uses it |
| One-third identity / two-thirds context | The named Qwen recipe is 60/30/10 and argues against full-body-heavy sets by name |
| Adult-finetune resolution bands | Krea 2 checkpoint craft; Qwen's adult layer is LoRAs and AIO Edit checkpoints |

---

## 10. Evaluation and debugging

The shared method (blind pairs, a strength-0 control in every comparison, the held-out probe set, why loss is a weak signal) is [`character-lora-training/references/evaluation-and-tooling.md`](../../character-lora-training/references/evaluation-and-tooling.md). Nothing published prescribes a Qwen matrix. This one is synthesis:

- **Deploy checkpoint × regime** — base at 20 steps *and* Lightning at 4 or 8 — on **scaled fp8 or bf16**, never plain `fp8_e4m3fn`.
- **Two face scales**, close-up and full-body, with and without a FaceDetailer.
- **A strength-0 control in every grid**, to separate the three base artefacts from LoRA artefacts.
- **The trigger-removed prompt**, for bleed. Qwen's bleed is comparatively mild: "the concept bleeding on ZIT is worse than Qwen", and Qwen tolerated three stacked LoRAs where ZIT managed two `[community — Top_Buffalo1668]`.

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| A regular grid on every render | fp8 grid — the downcast base × your LoRA, or × Lightning | Scaled fp8 or bf16 base; not a training problem |
| A faint halftone, worse at high res | The VAE decoder | Downscale / re-upscale; not overfit |
| Colour cast | Lightning V1.x over-saturates, by the maintainers' own admission | Check the Lightning version before blaming the LoRA |
| Stuck in one view | "front view" captioned on every photo | Caption composition only when unusual |
| Blurry or distorted faces on full-body prompts | Full-body over-representation in the set | Rebalance toward close-ups (60/30/10 is the named recipe) |
| Not the person under 50 repeats/img; fried over 100 | The FarTable6206 band | Stay inside 80–100 repeats per image |
| Rank 64 loses the person's specificity | Over-generalisation, the Qwen rank failure | Rank 32 for characters |
| LoRA destroys the image on 2512 / 2511 | Trained for another generation; the base absorbed community LoRAs | Retrain on the deploy generation; strength does not fix it |

Once trained: `setup-and-workflows.md §7` for loading and weights; `characters.md` for deploying it with Edit.

---

## 11. Qwen-Image-2.1 — DiffSynth only, RGBA datasets, and the licence

`[official — DiffSynth-Studio commits and `docs/zh/Model_Details/Qwen-Image-2.1.md`, `examples/qwen_image_21/`, 2026-09-20; ai-toolkit, musubi-tuner and SimpleTuner trees checked the same day]`. Nothing here is community-tested; no 2.1 LoRA exists on Civitai or Hugging Face.

**Support matrix.** DiffSynth-Studio: full fine-tune and LoRA, day 0. ai-toolkit: none (last Qwen commit 2026-09-16, pre-release). musubi-tuner: none. SimpleTuner: none. `[flagged — re-verify weekly]`. None of §1's Qwen defaults apply; 2.1 is a different transformer, encoder and latent space.

**The DiffSynth example, verbatim numbers:** `train.py` with `--lora_base_model dit --lora_rank 32 --lora_target_modules ""` (framework default set), `--learning_rate 1e-4`, `--num_epochs 5`, `--dataset_repeat 50`, `--max_pixels 1048576` (dynamic resolution, 1 MP cap — the model's native band is 2K, so this is a compute choice, not a doctrine), `--use_gradient_checkpointing`, `--find_unused_parameters`, `--remove_prefix_in_ckpt "pipe.dit."`. Loads all three components from `Qwen/Qwen-Image-2.1` by origin path; `--processor_path` and `--initialize_model_on_cpu` are the two 2.1-specific flags. `--fp8_models` can hold the frozen encoder in fp8.

**RGBA is the default training format.** Images are loaded as RGBA, so a dataset with real alpha trains transparent generation directly; an RGB dataset trains opaque output. This is the one genuinely new lever: a sticker, icon or cut-out character LoRA can be trained on transparent PNGs without a matting stage.

**Edit LoRAs** use the same script with `--data_file_keys "image,edit_image" --extra_inputs "edit_image"`; the shipped example points at the Edit-2511 paired dataset, so paired data made for the 20B Edit transfers as data (§6's pairing rules hold).

**No Lightning, no doctrine yet.** §2's "train on the base, stack Lightning at inference" has no second half here. Train and evaluate at 40 steps, CFG 1.

**Licence gate before you start.** §4(b) of the Qwen Research License: anything trained on or with 2.1 that is distributed must "prominently display 'Built with Qwen' or 'Improved using Qwen'". §2: the grant is non-commercial; a paid LoRA, a paid dataset generated with 2.1, or a commercial deployment needs a separate licence from Qwen. The publishing gates in [`character-lora-training`](../../character-lora-training/references/publishing-and-likeness.md) apply on top.
