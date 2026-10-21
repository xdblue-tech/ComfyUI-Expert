# Krea 2 — LoRA training

> **Shared craft lives in [`character-lora-training`](../../character-lora-training/)** — dataset size and coverage, the identity ratio, caption-the-residual, checkpoint selection, synthetic data, evaluation, adult/NSFW base selection, and the real-person likeness rules that decide whether a LoRA is publishable. This file covers what is specific to this model: the trainer flags, the resolution band, Raw vs Turbo, rank, steps and rungs, deploy strength, and the measured numbers behind them.

This file is about **making** a LoRA for Krea 2. Using or stacking one is covered in `setup-and-workflows.md §6`. The end-to-end character pipeline is in `characters.md`. Open weights landed 2026-06-22. The official doctrine is unusually explicit, several full named recipes exist, and the biggest question is still contested. Verified 2026-07-07; community sections refreshed 2026-08-22. **§3a, §3b, §9 and §10 were written 2026-09-09 from twenty-one media-lab runs on two characters; the run-by-run ledger behind them is [`media-lab-runs.md`](media-lab-runs.md).**

**How to read the live-use claims.** Anything marked `[live-use — media lab, <run>, <date>]` comes from one lab's rented-GPU runs, judged by one person. The two characters are a real person with a scarce photo pool and a fully synthetic character. These claims are first-hand and measured, which most community sources are not. They are also *one lab*, not consensus. Where the lab and the community disagree, both are given and the point is marked `[contested]`.

## Contents
1. [The doctrine — and the dispute](#1-the-doctrine--and-the-dispute)
2. [musubi-tuner (the fullest documented path)](#2-musubi-tuner-the-fullest-documented-path)
   - [2a. Training on 12 GB](#2a-training-on-12-gb--the-low-vram-configuration)
   - [2b. Adult / NSFW work](#2b-adult--nsfw-work)
   - [2c. 16 GB, measured — and four corrections](#2c-16-gb-measured--and-four-corrections-that-came-out-of-it)
3. [AI-Toolkit and the Ostris turbo-adapter path](#3-ai-toolkit-and-the-ostris-turbo-adapter-path)
   - [3a. The AI-Toolkit Raw recipe, held across twenty-one runs](#3a-the-ai-toolkit-raw-recipe-held-across-twenty-one-runs)
   - [3b. Rank, steps and the rung — measured](#3b-rank-steps-and-the-rung--measured)
4. [Hosted trainers](#4-hosted-trainers)
5. [Captioning — the Krea-specific part](#5-captioning--the-krea-specific-part)
6. [Character LoRAs: three named recipes](#6-character-loras-three-named-recipes)
7. [Style LoRAs](#7-style-loras)
8. [Evaluation](#8-evaluation)
9. [Rendering a synthetic dataset: the three resolutions](#9-rendering-a-synthetic-dataset-the-three-resolutions)
10. [Deployment and inference rules](#10-deployment-and-inference-rules)

---

## 1. The doctrine — and the dispute

**The official position is "TRAIN on Raw and RUN on Turbo"** (the capitals are theirs) `[official — GitHub FAQ]`. Raw is the undistilled checkpoint. Krea describes it as "diverse and highly malleable… what you should use for fine-tuning, post-training, and LoRA training", and adds that "LoRAs trained on RAW are designed to express strongly on Turbo". Every distilled family follows this same principle of training on the base rather than the distilled model (Z-Image, FLUX.2 klein), and by September 2026 Krea, BFL and kohya all say so in their own docs. But Krea is the first lab to turn the cross-checkpoint transfer into an explicit contract that the models are designed for. They even ship the distillation itself as a LoRA (`krea2_turbo_lora_rank_64_bf16.safetensors`). That is why Raw-trained LoRAs compose with Turbo so directly.

**The dispute:** Ostris (AI-Toolkit) ships a **de-distillation training adapter** (`ostris/krea2_turbo_training_adapter`) for training *directly on Turbo*. The adapter is itself a LoRA, trained at LR 1e-5 on thousands of Turbo generations. It holds the step-distillation together during fine-tuning and is removed at inference, so your LoRA runs on Turbo at full speed. Ostris's position is that for short runs (styles, concepts, characters), "training directly on the turbo model could yield better results" `[official — Ostris HF adapter card]`. Read the card's own limit too: the adapter "slows down" the breakdown of distillation rather than preventing it, so it is a bounded hack for short runs, not a second base. Kohya's musubi docs and Krea's own docs stay Raw-first.

**What has moved since August is framing, not evidence.** Several secondary write-ups now call AI-Toolkit plus the turbo adapter the "de facto standard" for Krea 2 fine-tuning, on popularity and convenience grounds `[community — ostrisai on X, buildfastwithai; convergent]`. musubi-tuner's own `krea2.md` still calls its support experimental (re-read 2026-08-15). That is a tilt in what people reach for first. It is not an A/B: **no first-hand comparison of a Raw-trained LoRA against a Turbo+adapter-trained one on the same dataset has been published** `[contested]`. The largest first-hand evidence in this file — twenty-one media-lab runs (§3a) — is all Raw-path on AI-Toolkit. That shows the Raw path works at scale. It says nothing about the adapter.

**How to choose today:** both paths have named end-to-end successes. Raw-path examples include characters (JahJedi and the media lab, §6) and styles (urabewe, §7; the Arthemy Comics author "training a highly specialized LoRA on the RAW version, as the Krea team suggested"). Turbo/AI-Toolkit successes include multi-character LoKr runs. Raw-first remains the doctrine-backed default, and it is the path both trainers document. The Turbo-adapter path is lighter on VRAM (§3) and fine for short runs. A/B it if the run matters.

Two structural facts hold on either path (encoder-class doctrine):
- **DiT-only.** The Qwen3-VL encoder is never trained. Krea 2 LoRAs are model-only, so you load them with `LoraLoaderModelOnly`.
- **No rare-token triggers.** Fold a natural descriptive phrase into your captions and prompts. The official style LoRAs use trigger phrases like `monochrome ink wash style` (`prompting-guide.md §5`). On Krea 2 specifically, random strings are reported to be ignored or to surface as text and watermarks `[community — promptdexter, 2026-07-24]`. The rule is per-model: BFL prescribes made-up tokens for FLUX.2 klein, which also has an LLM encoder, so do not generalise it across the encoder class.

## 2. musubi-tuner (the fullest documented path)

Kohya's musubi-tuner added day-0 experimental Krea 2 support, and the docs still say experimental `[official — musubi docs/krea2.md, re-read 2026-08-15]`. This is the most completely documented trainer. It is also the source of several load-bearing architecture facts: 28 blocks, GQA 48Q/12KV, and the resolution-aware shift schedule.

**Models needed:** the Raw DiT (`raw.safetensors` from `krea/Krea-2-Raw`); optionally the Turbo DiT for sampling; the **[Qwen-Image](../../qwen-image/) VAE** (the same file ComfyUI uses); and **Qwen3-VL-4B-Instruct as a single safetensors file** (the Comfy-Org `qwen3vl_4b_bf16.safetensors` works and can be shared with ComfyUI).

**Pre-cache both stages.** Cache image latents with `krea2_cache_latents.py --vae …`, and text-encoder outputs with `krea2_cache_text_encoder_outputs.py --text_encoder …`. Krea 2 caches the 12-layer hidden-state stack, so the encoder is not needed during training itself.

**Training** (`krea2_train_network.py` — the authors'-default configuration, verbatim-adjacent from the docs):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  src/musubi_tuner/krea2_train_network.py \
  --dit raw.safetensors --vae qwen_image_vae.safetensors --dataset_config data.toml \
  --sdpa --mixed_precision bf16 \
  --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.5 \
  --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
  --network_module networks.lora_krea2 --network_dim 32 --network_alpha 32 \
  --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
  --output_dir out --output_name my-lora
```

Key facts from the docs:

- **rank/alpha 32 all-Linear "reproduces the model authors' recommended default".** That covers 264 Linear layers: attention, MLPs, the text-fusion transformer, and projections. The authors' **"long training run" config** takes the opposite trade. It trains attention-only (140 Linears — `wq/wk/wv/wo/gate`, via an `exclude_patterns` network-arg) at *higher* rank, in order to preserve prompt adherence over long runs.
- **Flow shift:** `--discrete_flow_shift 2.5` matches K2's inference-time shift at 1024². The schedule is resolution-aware: roughly 1.6 at 256² and 3.2 at 1280². For bucketed multi-resolution datasets, use `--timestep_sampling krea2_shift` instead. It reproduces the per-sample resolution-aware schedule exactly, so you do not need a fixed shift. (`flux_shift` is close but saturates at 1024px instead of 1280px.) The docs carry their own caveat: "the optimal settings are not yet established."
- **Memory:** `--fp8_base --fp8_scaled` (they must be set together; fp8 covers the 28 main blocks, while the text-fusion stage stays bf16); `--blocks_to_swap` up to **26**; `--gradient_checkpointing`; `--compile` for the main blocks. **12 GB is enough in practice.** A named style-LoRA config on an RTX 3060 12 GB with 48 GB RAM runs `--fp8_base --fp8_scaled --blocks_to_swap 18 --block_swap_h2d_only --block_swap_ring_size 1 --split_attn --gradient_checkpointing_cpu_offload`. That gives ~1,200 steps in ~2 h at ~5.9 s/it on 30-image datasets `[community — urabewe, r/StableDiffusion, full command published]`.
- **Sample on Turbo while training Raw:** `--turbo_dit turbo.safetensors` applies the in-training LoRA on top of Turbo weights for previews (`--l 1 --s 8` in the sample prompt — CFG off, 8 steps). This lets you preview on the checkpoint you will actually run on, which is the doctrine made practical. Raw-side samples need CFG, because CFG-off Raw output is blurry by design. `--turbo_dit` is incompatible with `--blocks_to_swap`.
- **Inference script** (`krea2_generate_image.py`) uses the **classic CFG scale** (≤1 = off): Turbo = `--steps 8 --guidance_scale 1 --mu 1.15`; the Raw default `--guidance_scale 5.5` ≙ official guidance 4.5. It fits a 24 GB card with `--fp8_scaled` and/or block swap. LoRAs merge into base weights at load, which is the only correct route under fp8.

## 2a. Training on 12 GB — the low-VRAM configuration

A fully documented 12 GB path exists. It is worth stating precisely, because the naive attempt fails `[community — SirMick, Civitai, 2026-07]`. It was verified on an **RTX 3060 12 GB, 64 GB system RAM, Windows**, with musubi-tuner.

**Why it is hard:** the full model stack far exceeds the card.

| Component | Approx. size |
|---|---|
| Krea 2 Raw | ~24.5 GB |
| Qwen3-VL 4B text encoder | ~8.3 GB |
| Qwen Image VAE | ~1.1 GB |

**FP8 alone is not enough.** With FP8 only, training stayed unstable, slow, and prone to CUDA crashes. The working solution is FP8 **plus CPU block swapping**:

```
--fp8_base --fp8_scaled          # base-model quantisation
--blocks_to_swap 20              # move inactive transformer blocks to system RAM
--block_swap_h2d_only            # keep a CPU master copy; stream host→device only,
                                 #   skipping the needless device→host copy in LoRA training
--block_swap_ring_size 2         # two GPU ring buffers so the next block prefetches
                                 #   while the current one is processed (also musubi's default)
```

**System RAM is the hidden requirement.** 64 GB was tested. **32 GB can become tight**, depending on dataset, OS overhead, caching, and page-file settings. On a 12 GB card, the constraint that bites is usually host memory rather than VRAM.

Validated bucket geometries at this size: **512×512 and 1024×256**, batch size 1.

The original author flags an important limit. Their benchmark tests the block-swapping flags as a combination, and does not isolate `--block_swap_h2d_only`. So no independent speedup is claimed for that flag alone. That level of care is worth preserving when you cite these numbers.

## 2b. Adult / NSFW work

**Krea 2 supports this well.** Roughly **52% of published Krea 2 LoRAs are adult-flagged**, the highest share of any image model in this suite `[community — Civitai model API, sampled 2026-08-13]`. That is consistent with `nsfwVariant` already being one of this skill's most-cited craft sources for general Krea 2 technique.

A few Krea-specific points:

- **The doctrine is unchanged: train on Raw, run on Turbo.** Adult LoRAs follow the same path as any other, and several published ones ship exactly that way.
- **Several LoRA families span bases.** The same named adult LoRA lines appear built for Krea 2, Z-Image Turbo, and Flux Klein. If you are already running one family on another base, check whether a Krea 2 build exists before training your own.
- **Stock Turbo will not render nudes, at any LoRA strength** `[community — week-long character-LoRA run on stock Turbo]`. A character LoRA cannot add content the base refuses to draw, so run it over an adult checkpoint (next bullet) rather than fighting the stock base. Before blaming a LoRA for a failing stack, generate once with the LoRA at strength 0. That probes the base alone and costs cents. General base-selection doctrine is in [`character-lora-training/references/nsfw-training.md`](../../character-lora-training/references/nsfw-training.md).
- **Check for a finetuned checkpoint before training anything.** By August 2026 the adult Krea 2 checkpoint ecosystem is mature and easily the busiest on Civitai. `LUSTIFY!` was the single most-downloaded model on the site over the month sampled, and `FinePorn v3 TURBO`, `Moody Krea 2 Mix (uncensored)`, and several Stable Yogi realism finetunes are in wide use. A later sweep found the same names in circulation with no rival base contesting them: **Krea 2 is the dominant adult *image* model by a wide margin** `[community — r/unstable_diffusion, 2026-08-23]`. Reported workflows are unremarkable (Euler or ER SDE, 10 steps, guidance 1.0). That is the point — the checkpoint is doing the work.
- **A finetune has its own resolution band, and it is not Turbo's.** The media lab deploys character LoRAs over `fineporn_v4_int8`. At **1024×1536 it is unusable** — cheeks break up, and LoRA strength is not the lever (1.0, 0.85, 0.70 and 0.55 were all bad). At **≥1344×2016 it is good**. Stock Turbo is fine at 1024. So a LoRA evaluated only on Turbo at 1024 can ship broken on the checkpoint you actually use for the adult work (§8) `[live-use — media lab, krea2-v3 Ciara, 2026-09-08]`.
- **The `krea2filterbypass` line is past circulating and into routine production use.** The form seen in the wild is **`<lora:krea2filterbypass3:2>`** — version 3, at **weight 2** — stacked on top of `MysticXXX_KREA2_V4` rather than used alone `[community — KlitoriaPierce]`. Read both halves of that. A LoRA can only re-expose behaviour the base weights already encode. It can never supply what was never trained. So the fact that a bypass line works at all tells you something about the base model: its reticence comes from tuning, not from missing data. And the bypass takes *double* the usual LoRA strength, on a checkpoint that is itself already a finetune. That tells you how hard the tuning was applied — a normal-weight counter-LoRA does not clear it. Plan to stack rather than substitute. The same weight-2 figure is quoted in SKILL.md under the muted-expression tax, where the bypass line is also the SFW expressiveness fix.
- **You may not need to train at all.** For a *consistent* character in adult scenes, the named production pattern is Krea 2 + the Identity Edit LoRA + an NSFW LoRA `[community — Clone-Protocol-66]`. Identity Edit carries the likeness, the NSFW LoRA carries the content, and neither has to do the other's job. Train a character LoRA when likeness must survive close-ups; see [`characters.md §3`](characters.md).
- Krea 2's soft, airbrushed default and its **two taxes** (see SKILL.md) apply here as everywhere. The Wan 2.1 VAE swap and texture anchoring matter more for anatomy work, not less — with one exception. Once a LoRA carries the skin, texture anchors turn harmful (§10).

General doctrine — why base coverage rather than refusal is the limit, explicit captioning, anatomy failure modes — is in [`character-lora-training/references/nsfw-training.md`](../../character-lora-training/references/nsfw-training.md). Publishing constraints, which bind harder than capability, are in [`publishing-and-likeness.md`](../../character-lora-training/references/publishing-and-likeness.md).

## 2c. 16 GB, measured — and four corrections that came out of it

A second fully-measured run exists, this time on an **RTX 5080 16 GB with 32 GB system RAM**, using musubi-tuner `[community — Economy_Cucumber_702, 2026-07]`. It is worth reading not for the numbers, but for what the author got wrong and then corrected in public.

**Measured:** 1152 steps in **67 minutes at 3.42 s/it**, peak **15,284 of 16,303 MiB VRAM**, and 17.5 GB of 32 GB system RAM. Turbo inference afterwards ran at ~13 s per 768×1024 image at 8 steps.

Here is the drop-in config, with the resolution corrected:

```toml
dit = "/path/to/krea2_raw_bf16.safetensors"
vae = "/path/to/qwen_image_vae.safetensors"
sdpa = true
mixed_precision = "bf16"
fp8_base = true            # must be set together —
fp8_scaled = true          # plain fp8 is rejected on purpose
blocks_to_swap = 16        # max 26
block_swap_h2d_only = true # avoids the copy doubling that eats host RAM
block_swap_ring_size = 1
gradient_checkpointing = true
max_data_loader_n_workers = 0
timestep_sampling = "krea2_shift"
weighting_scheme = "none"
network_module = "networks.lora_krea2"
network_dim = 32
network_alpha = 32
optimizer_type = "adamw8bit"
learning_rate = 1e-4
max_grad_norm = 1.0
max_train_epochs = 16
save_every_n_epochs = 1
seed = 42
```

Dataset toml: `resolution = [1024, 1024]`, `batch_size = 1`, `enable_bucket = true`, `caption_extension = ".txt"`.

**The four corrections are the actually useful part:**

1. **Train at 1024, not 768.** The author ran at 768 and was corrected. The technical report says pretraining spanned **256, 512 and 1024 px stages**, so 768 was never a trained resolution. Every number above was measured at 768, so **expect to raise `blocks_to_swap` and re-check VRAM at 1024.** This correction applies to §2a's 12 GB config too. The validated bucket geometries there are 512×512 and 1024×256, and both sit on the trained lattice. The question has since become livelier, not settled. AI-Toolkit ships `resolution: [512, 768, 1024]` for Flux and Qwen-Image. Two Krea-2 community sources bucket 512+768+1024 as a composition→detail progression, and a dataset tool drops to 768-only purely as a *memory* lever `[community — chengyansen-ai, lora-dataset-studio]`. No published run shows a quality win from a 768 bucket on Krea 2, and the twenty-one media-lab runs in §3a all trained at `[1024]` alone `[contested]`. Train at 1024; treat multi-res as an experiment. The *training bucket* is only one of three resolutions that matter. The dataset's source render and the deploy render are the other two, and §9 is about keeping them apart.
2. **16 GB is enough.** That contradicts the widely-linked musubi-tuner issue thread, which says it is not. The author names the thread in their write-up, and their measurement is the only counter-evidence published `[community — Economy_Cucumber_702; single report]`.
3. **The official `krea/Krea-2-*` repos are gated. `Comfy-Org/Krea-2` is not, and carries a byte-identical RAW checkpoint.** This is useful only for trainers that take a file path rather than a repo id — but that covers musubi.
4. **A LoRA bleeding into prompts that omit the trigger is normal, and the fix is regularisation images, not caption surgery.** The author initially guessed at a captioning change. That was the wrong lever. This is consistent with §5's residual doctrine: captions control what stays *promptable*, not what leaks. The media lab's trigger A/B (§5) shows the same mechanism from the other side.

`timestep_sampling = "krea2_shift"` is the right default here for the same reason as in §2. It reproduces Krea's resolution-aware schedule per sample and survives aspect-ratio bucketing. At a fixed 1024 you can equally use `shift` with `discrete_flow_shift = 2.5`.

**On citing this:** it is one run, on one machine, with no ablations. The writeup is AI-assisted, with the measurements taken from the author's own machine. Treat the numbers as a plausibility check for your own hardware, and treat the four corrections as the durable content.

## 3. AI-Toolkit and the Ostris turbo-adapter path

Krea's README lists Ostris AI-Toolkit as a recommended trainer. The architecture key is `krea2`. There is still **no krea2 example YAML in the repo's `config/examples/`** — the directory's newest DiT entries remain the Qwen-Image and Wan 2.2 configs `[official — repo listing, re-verified 2026-09-09]`. So configure through the UI, adapt a nearby DiT config, or start from §3a's working file:

- **Raw path:** a standard AI-Toolkit LoRA run against Raw. There is a hardware gotcha with a named fix: Raw training **OOMs early on a 24 GB RTX 3090 even in Low-VRAM mode** (it fails around 3 GB allocated, with 32 GB system RAM) until **Layer Offloading is set to ~10%** (5% also works and is slightly faster) `[community — Fast-Cash1522, r/StableDiffusion, marked SOLVED]`.
- **The best-replicated community character recipe** runs on this path: **LoKr factor 4, Automagic3 optimizer, sigmoid scheduling, "Balanced", LR 1e-4 + weight decay, 1024-only, ~50-image datasets, 2–3k steps**. The author rated the likeness above their Z-Image results across multiple characters `[community — Any_Tea_3499]`. Note that it is LoKr, not classic LoRA — factor 4 is the capacity knob that stands in for rank.
- **LoKr is becoming the community's preferred decomposition for Krea 2 characters, and its numbers are not settled.** A dedicated guide gives LoKr rank **512 or 1024 — never 768**, after ai-toolkit Discord reports of failed 768 runs. The reported benefit is noticeably less identity bleed when two characters share a frame. The factor is genuinely contested: 16 in that guide, 4 in the ai-toolkit community, 8 as a compromise `[community — instasd LoKr guide, 2026-08-20; contested factor]`. LoKr is also reported *more* sensitive to source resolution and JPEG compression than plain LoRA `[community — chengyansen-ai]`. Set that against §3a: plain LoRA at rank 32 shipped single-character likeness across twenty-one runs. Reach for LoKr when the job is several characters in one frame (`characters.md §2`). Plain LoRA is the proven single-character path.
- **Turbo-adapter path:** load `ostris/krea2_turbo_training_adapter` as the training adapter over Turbo, then train your LoRA. The adapter is dropped at inference. It is made for short runs — styles, concepts, characters. It is lighter than Raw on the same hardware: the 3090 user above trained Turbo+adapter "without any issues" on the settings that OOM'd Raw.
- Ostris also ships a Krea 2 **edit-training** stack (paired-data edit LoRAs plus a 3-reference-image ComfyUI node). That belongs to `characters.md §3`, not classic LoRA training.

## 3a. The AI-Toolkit Raw recipe, held across twenty-one runs

`[live-use — media lab, krea2-v1…v16 (Amy) and krea2-v1…v5 (Ciara), 2026-08-24 → 2026-09-09]`

Two sources arrived at the same plain-LoRA recipe independently in August. One is a **community-calibrated workflow** published as a repo `[community — chengyansen-ai/krea2-lora-training v0.4.0]`. The other was a finished private character run. That run has since become **twenty-one runs on one config**: 16 on a real person (13–74 photos) and 5 on a fully synthetic character (32–62 rendered cells). The config was never varied and never needed to be. Every arm changed the *dataset*, not the recipe. That is the strongest evidence in this file that the numbers below are a safe first attempt. What each arm changed and measured is in [`media-lab-runs.md`](media-lab-runs.md) §1.

| Setting | Value | Why |
|---|---|---|
| `arch` / base | `krea2`, **Krea 2 Raw** (`krea2_raw_bf16.safetensors` as a local single file) | Doctrine (§1). All twenty-one runs are Raw-path |
| Network | **LoRA, `linear` 32 / `linear_alpha` 32** | The all-Linear rank and alpha both sources use; matches the musubi default in §2. Rank 16 loses slightly and rank 64 gains nothing (§3b) |
| Optimizer / LR | `adamw8bit`, **1e-4** | Drop to 5e-5 if you want to run past about 3k steps without overfitting `[community — chengyansen-ai]` |
| `noise_scheduler` | `flowmatch` | — |
| `timestep_type` | **`linear`** | Called out specifically as *not* Flux's sigmoid `[community — chengyansen-ai]`. §3's LoKr recipe uses sigmoid; pick per recipe `[contested]` |
| `train_text_encoder` | **`false`** | Qwen3-VL stays frozen |
| `quantize` / `qtype` | `true` / `qfloat8`, encoder too | fp8 on both sides is what makes 32 GB comfortable. Measured VRAM in use: **18.5–19.2 GB** |
| `resolution` | **`[1024]`** | §2c. Krea 2 needs no upscaling of sources: 1024 is native |
| `low_vram` | `false` | Not needed at 32 GB; §3's offload report is a 24 GB problem |
| `cache_text_embeddings` | `true` | Safe only with the trigger literal in the sidecars — see below |
| `caption_dropout_rate` | `0.05` | Both sources. One community article calls **0.3** "the strongest single fix for generalization" `[community — neonkisu, Civitai 31467]`; an order of magnitude apart and untested in the lab `[contested]` |
| Steps | **2250–3500**, `save_every` 250, samples disabled | 2250 on 25 images, 2500 on 13–16, 3000 on 32, 3500 on 54–91. That is ~90–170 per image; §3b on why the rung matters more than the total |

**Hardware, in one line.** An RTX 5090 runs 2500 steps in about two hours for about $2.30; the RTX PRO 4500 is half the throughput at 0.73× the price; the A100 has no fp8 units, so on this recipe it is slower *and* dearer. Check the datatype before the price. The measured table is in [`media-lab-runs.md`](media-lab-runs.md) §3.

**`cache_text_embeddings: true` is only safe if the trigger word is written into the captions themselves.** The cache encodes each caption once and reuses that encoding. So anything the trainer would normally swap in later — such as a `trigger_word` field it injects per step — gets frozen at the wrong value. Type the trigger into the `.txt` sidecars yourself, leave `trigger_word` unset, and the cache is free speed. Caching also takes the text encoder out of the step loop, which is what frees the VRAM.

**You do not need layer offloading at 32 GB.** With fp8 on both sides, text embeddings cached, and gradient checkpointing on, the measured run used 15.8 of 32 GB with `low_vram` off. Turning offloading on gained nothing either — the card was the limit, not memory traffic. So treat offloading as a way to fit a job on a smaller card, not as something that slows you down.

**The trainer wants HuggingFace-format folders for the encoder and VAE, not the ComfyUI single files.** Those single files — `qwen3vl_4b_*.safetensors` and `qwen_image_vae.safetensors` — are for generating images. AI-Toolkit will still fetch its own diffusers-format `Qwen3-VL-4B-Instruct` (about 9 GB) and Qwen-Image VAE on top of them. Budget disk for both, or point at folders you already have:

```yaml
model:
  arch: "krea2"
  name_or_path: "/workspace/models/diffusion_models/krea2_raw_bf16.safetensors"
  model_kwargs:
    text_encoder_path: "/path/to/Qwen3-VL-4B-Instruct"
    vae_path: "/path/to/Qwen-Image"
```

`[community — chengyansen-ai/krea2-lora-training]`. On rented hardware this is also how you rescue a nearly-full network volume: send the encoder and the HF cache to the pod's throwaway container disk, and keep the volume for checkpoints — see [`comfyui-on-runpod`](../../comfyui-on-runpod/references/volume-and-models.md) §6.

**Disable in-training samples.** Previews on Raw render on the undistilled model at guidance 4, about 90 seconds each, and a 6-prompt set every 250 steps burns more time than the training it watches. They are also washed out in exactly the fine skin and bone detail a face is recognised by; "Raw previews look plastic" is listed as *expected* by the community workflow. The lab turned samples off from its seventh run and never turned them back on. Keep `save_every` at 250 to build the checkpoint series, and judge on a Turbo grid afterwards (§8). **This is the one place musubi beats AI-Toolkit on this model:** `--turbo_dit` (§2) previews the in-training LoRA on Turbo directly.

Two failures specific to this path come from the community workflow's pitfall list `[community — chengyansen-ai]`. First, **the character bleeds into prompts that never mention the trigger** — pick an earlier checkpoint, do not retrain. Second, **`res_2s`-family samplers drift** when you generate.

## 3b. Rank, steps and the rung — measured

`[live-use — media lab, krea2-v2/v4/v5 (Ciara), krea2-v1…v13 (Amy), 2026-08-26 → 2026-09-09]`. The per-run evidence is in [`media-lab-runs.md`](media-lab-runs.md) §1–2; this section keeps the rules.

**Rank 32 beats rank 16, slightly. Rank 64 buys nothing.** On one 32-cell set r32 won every table against r16; on an unchanged real-photo set r64 with more steps lost a blind grid to the r32 control — facial *shape* improved mid-run and never converted into identity. Capacity and length were exhausted as levers before the dataset was. Rank/alpha 32 is the setting; change the dataset next. Nor is rank the fix for a texture artefact: the r16 arm was run partly to see whether a lower rank would stop reticulated "scale" skin, and both arms came out equally clean, which proved the cause was the source render (§9). The numbers are in the ledger §2.

**The rung is per-dataset.** Measured peaks on the one recipe ran from c1750 on a 13-image face set — where 2500 collapses the three-quarter view and 1250 undercooks the front — to `final` at 3000 on the 32-cell synthetic set. Steps-per-image is a loose guide at best, and BFL now quotes 1,500–3,000 for a character across 15–40 images with no per-image formula `[official — docs.bfl.ml]`. So build the 250-step series and pick by grid. The two rules that decide the pick are the suite's, not Krea's: **when the grid cannot separate two rungs, take the lower one**, and expect an all-phone-JPEG set to peak a rung earlier, because past the peak the *source medium* trains in as skin. Both are in [`character-lora-training/references/evaluation-and-tooling.md`](../../character-lora-training/references/evaluation-and-tooling.md) §3, with the Krea 2 numbers in the ledger. The community version is "watch the samples, not the loss": `avr_loss` moved only 0.0774→0.0735 across a 3,960-step Krea 2 run `[community — Zaytron40k, HF]`.

**One report runs far shorter.** A usable Krea 2 character LoRA from **10–15 tightly curated images at 600–1,200 steps**, with every off-looking image culled and captions naming only what should vary `[community — r/StableDiffusion 1wacvcm, 2026-09; single report]`. That is a third of the lab's 2,500 on 13–16 images, and nobody has run both on one set. Treat it as the low end of a contested step range — [`character-lora-training`](../../character-lora-training/) lists steps per image as open — and let the checkpoint series decide rather than the number.

**Krea 2 is not H3.** Rank 32 collapsed MiniMax H3 under AI-Toolkit, in a ladder that trained with distillation handling off; r16 is the ceiling there until r32 is shown surviving with handling on. The rank ceiling is model-specific. Do not carry this file's 32 into [`minimax-h3`](../../minimax-h3/).

## 4. Hosted trainers

- **fal:** `fal-ai/krea-2-trainer` (train) + `fal-ai/krea-2/turbo/lora` (run). Zero setup, hyperparameters managed, weights come back as standard safetensors LoRAs usable locally.
- **Krea's own trainer** (Max/Business tiers, in beta as of its announcement): auto-captions, and requires "at least three" images `[official — krea.ai blog, 2026-05-21]`. The floor is far below practice; treat it as a minimum the UI enforces, not a recommendation. Krea's own guidance agrees with everything else here — "a cleaner set that repeats the same subject usually teaches the model more clearly than a large, mixed dataset". `api-and-hosted.md §3`.

## 5. Captioning — the Krea-specific part

The encoder-class rule (LLM/VLM encoder) is **prose captions, caption-the-residual**: describe what varies and should stay promptable, leave the identity uncaptioned so it binds to the trigger phrase. The full doctrine — what to caption, the exception rule, register, length, dropout, the trigger A/B and its three consequences, which traits become promptable — is model-agnostic and lives in [`character-lora-training/references/dataset-and-captioning.md`](../../character-lora-training/references/dataset-and-captioning.md) §4. What follows is only what was measured *on Krea 2*.

- **The trigger is redundant on a single-subject set, and that is expected.** Base + `zamy` draws an unrelated woman; the LoRA draws the same person with or without the token, because every image is the same subject and every caption carries the trigger `[live-use — media lab, Amy dataset v13 A/B, 2026-08-31]`. The levers are regularisation images (other people, captioned *without* the trigger — JahJedi's 348, §6) or reduced strength, not a caption rewrite. §2c's fourth correction is the same mechanism.
- **Trigger dialect that shipped:** `z<name>` folded into prose — *"a woman named zciara"*. That literal trigger is what makes `cache_text_embeddings: true` safe (§3a). One visible cost: the phrase **prints the trigger onto signage and name badges** in rendered output. Prompt around signage, or drop "named" and use the bare token `[live-use — media lab, krea2-v5 Ciara flexibility sheet, 2026-09-09]`.
- **Hair colour and length become promptable when captioned; age and freckle density do not.** Naming freckle density made freckles a prompt variable that fought the LoRA, so both were stripped from every later set `[live-use — media lab, krea2-v13 vs v14 flex probe, 2026-09-05]`. The cost of captioning hair: prompts must name it from then on.
- **Caption in the register you intend to prompt in.** A multi-character LoRA captioned with literal scene descriptions held up when prompted that way, and bled under free-form creative prompts `[community — krigeta1's documented failure case]`.
- **Captionless training** has no Krea-2-specific evidence either way. Flag any strong claim you meet as unverified.

## 6. Character LoRAs: three named recipes

Three full recipes exist, at three dataset scales. Pick by what you have.

| | Media lab (plain LoRA) | Any_Tea_3499 (LoKr) | JahJedi (musubi, heavyweight) |
|---|---|---|---|
| Trainer / base | AI-Toolkit on **Raw** (§3a) | AI-Toolkit on Raw | musubi-tuner on **Raw** |
| Dataset | **13–32** curated real photos, or 32 synthetic cells | ~50 images | 474 character + 348 regularisation |
| Network | LoRA 32/32 | LoKr factor 4 | dim/alpha 32 |
| LR / optimizer | 1e-4, adamw8bit | 1e-4 + weight decay, Automagic3, sigmoid | 1e-4, AdamW, fp32 |
| Steps | 2250–3500, pick the rung (§3b) | 2–3k | ~13,000 (~4.6 h) |
| Result | shipped at **strength 1.0** on Turbo and on a fineporn finetune (§10) | likeness rated above the author's Z-Image results | identity holds at **0.8** under heavy style mixing, on Turbo |
| Source | `[live-use — media lab, 21 runs, 2026-08 → 09]` | `[community — Any_Tea_3499]` | `[community — JahJedi, HF krea2-character-lora-recipe]` |

Two things the lab's runs settled that the community recipes leave open. **Count is not the lever**: 25 real photos beat a 74-image strict superset of themselves blind, and 15 coverage-built images came out "very close" to 32, which is why the lab's band is 13–32 `[live-use — media lab, krea2-v1 vs v4, krea2-v8, 2026-08-27/30]`. Read JahJedi's 474 as what he had, not what Krea 2 needs. **A fully synthetic dataset works**: a character with no real photos reached triage medians of 0.155 against a real-photo calibration p50 of 0.112, with clean skin, on 32 cells rendered as §9 describes `[live-use — media lab, krea2-v5 Ciara, 2026-09-08]`. The dataset craft that decides these outcomes — coverage over count, the identity ratio, crops-are-duplicates, the three-quarter view as the canary — is owned by [`character-lora-training`](../../character-lora-training/), and the runs are in the ledger.

A community wrapper trainer exists (`bongobongo2020/krea2-character-lora-trainer`). It is unexamined — verify before trusting.

**If you intend to load several character LoRAs at once, consider Differential Output Preservation** on a LoKr run — class `"woman"`, 1500 steps rather than 750. Up to **four** characters then reportedly coexist with minimal bleed. Five falls apart. This comes from a single author's runs and has not been replicated, so budget a test run before you commit a project to it `[community — MASilverHammer; single report]`. The full account, including the same author's report that the technique fails on Z-Image Base, is in [`characters.md §2`](characters.md#2-the-character-lora-pipeline).

## 7. Style LoRAs

- Krea's own style-LoRA line (nine on Comfy-Org plus the `krea/krea-2-loras` HF collection) demonstrates the format: modest rank, DiT-only, a natural-phrase trigger, and strength 0.8–1.0. No official training write-up accompanies them. The community layer is already large — **1,500+ style LoRAs from a single named trainer** (ilker's `fal-Krea-2-Style-LoRAs`, highlighted in Krea's own community roundup) plus a steady Civitai stream.
- **Named Raw-path style runs with published settings:** urabewe's Garbage Pail Kids / Ren & Stimpy LoRAs used musubi defaults (§2), 30-image datasets, ~1,200–1,250 steps, and ~2 h on a 12 GB 3060. They are used at strength **1.0 with no trigger word** (nudge with "cartoon"/"animation" when needed) `[community — urabewe; full command and dataset-builder tool published]`. Philosopher_Jazzlike's anime-style LoRA (config attached on Civitai; run at 0.85 stacked with a second LoRA at 1.0) reports Krea 2 Turbo "absolutely brilliant at adopting styles while still executing the prompt". Strength-1.0-no-trigger is emerging as a common style pattern, in contrast to the official LoRAs' appended trigger phrases — so read each author's card.
- Style dataset craft is the suite-shared kind: diverse subjects so the style does not bind to content; composition-memorisation and colour-cast lock-in as overfit signals; an out-of-set subject as the acceptance test. See the [`sdxl`](../../sdxl/) / [`z-image`](../../z-image/) lora-training references for the full treatment.
- Style rank on Krea 2: rank/alpha 32 (the musubi default) is what the named style runs above used. The rank-64 Turbo LoRA brackets the high end. LoKr-factor-4 is the AI-Toolkit-side equivalent anchor (§3).

## 8. Evaluation

The shared method — blind pairs, probe design, a strength-0 control in every comparison, reproducing a known-good render before forming a hypothesis, the wall test, face-embedding triage and its calibration — is in [`character-lora-training/references/evaluation-and-tooling.md`](../../character-lora-training/references/evaluation-and-tooling.md). The lab learned the strength-0 rule the hard way here: a freckle-clumping artefact blamed on a LoRA came from the *base*, found by a no-LoRA render that "should have been the second image made, not the twentieth" (`media-lab-runs.md` §4). The Krea-2-specific rules:

- **Validate on the checkpoint *and resolution* you will deploy on — not just "on Turbo".** The old rule, "validate on Turbo, not Raw", still holds: a LoRA that looks great on Raw at cfg 3.5 and falls apart at 8-step guidance-off has failed. It is not enough. A LoRA evaluated only on Turbo shipped `final`, and on the adult finetune it was meant for, 1024×1536 was unusable while ≥1344×2016 was good (§2b); **c2500 was the only rung that survived the hardest case.** Raw is not an inference target. The lab's matrix is now **{Turbo, finetune} × {1024, 1344+} × ≥2 seeds at cfg 1.0, with a strength-0 control** `[live-use — media lab, krea2-v3→v5 Ciara, 2026-09-08]`. The run is in the ledger §1.
- **cfg > 1 on a guidance-distilled Krea 2 checkpoint is a grain source with no LoRA loaded.** cfg 1.0 at 8 and 12 steps is clean; cfg 1.5, with or without a negative, is grainy, and the grain covers the background. Run the eval at cfg 1.0. The lab keeps one exception: face-hidden body cells may run cfg 1.5 with a negative, because mass words do not move at cfg 1.0 on the adult finetune `[live-use — media lab, no-LoRA sweep, 2026-09-07]`.
- Add a Krea-2-specific grid axis: **with and without the Wan-VAE swap**, since much of the perceived quality difference lives in the decode (`setup-and-workflows.md §5`).
- The Krea-2-specific confound for "expression lock-in" is the **muted-expression tax**. Do not diagnose it in your LoRA before checking that the base model produces the expression at all (`SKILL.md`, *two taxes*).
- Overfit tells are the standard ones: same-face, composition memorisation, bleed into untriggered prompts. The cheapest test is the same prompt minus the trigger, which "must stay close to base; if polluted, you overtrained" `[community — chengyansen-ai]`.

## 9. Rendering a synthetic dataset: the three resolutions

`[live-use — media lab, Ciara dataset v1→v5 and krea2-v1→v5, 2026-09-06 → 2026-09-09]`

This section applies when the dataset is *generated* rather than photographed: a fully synthetic character, or synthetic cells filling gaps in a real pool. It is the most expensive lesson in the file. The general doctrine — why extra pixels cannot help (trainers downscale into buckets and never upscale), the recursion hazard, how much synthetic is safe, and the wrong turn told in full — is owned by [`character-lora-training`](../../character-lora-training/): `dataset-and-captioning.md` §3 for resolution and `synthetic-datasets.md` for seeding. What follows is the Krea-2-specific rule and the numbers.

**Three resolutions have to be kept apart. A week was lost conflating them:**

| Resolution | Krea 2 value | Why |
|---|---|---|
| **Dataset source render** — the size you generate a training image at | **1024-wide, native** (1024×1024, ×1280, ×1536, ×1792 cells) | Inside the base's trained band. Rendering above it bakes in a texture that survives downscaling (below) |
| **Training bucket** | **1024** (§2c) | Krea 2 was pretrained at 256/512/1024 only |
| **Deploy render** | Turbo 1024–2048; an adult finetune **≥1344×2016** (1024×1536 unusable, §2b) | Each checkpoint has its own band; evaluate on it (§8) |

**Render the dataset at 1024 native. Never render high and downscale.** Hi-res sources downsampled into the 1024 bucket do not give the LoRA real skin texture; they give the *wrong* texture, and it survives the downscale. A four-cell sweep at LoRA strength 0 settled it on 2026-09-08: 1024×1280 native clean, 2048 and 2560 crumpled, and both still speckled after downscaling to 1024. Two decoders were identical at 2048, so it was never the VAE. The cause was that Turbo's band is 1K–2K, Raw is 1K-native, and the set had been rendered at 2048–2816. Two LoRAs retrained on a 1024-native set, at rank 16 and 32, were clean at every rung on both deploy checkpoints: *"the fix was the 1024-native dataset, not rank."* The sweep table and the wrong turn as it happened are in [`media-lab-runs.md`](media-lab-runs.md) §4. **Before you diagnose any subtle skin artefact, verify you are inside the base model's trained resolution band and step band. Then look at the wall.**

**Step count is a separate, independent lever, and 20 is wrong.** At 2048, **20 steps over-cooks skin into a harsh pore grid; 12 gives natural skin; 8 is a touch soft.** Two runs on identical 62 cells changed only the source step count, 20 against 12: the 20-step set trained a fine speckle in, the 12-step set did not, and its LoRA won every in-distribution prompt (all-cell median 0.199 against 0.232). This is a *source-render* setting, invisible on a contact sheet `[live-use — media lab, krea2-v2 vs v3 Ciara, 2026-09-07]`.

**cfg stays at 1.0 for dataset renders on a distilled checkpoint.** cfg 1.5 grains the whole frame with no LoRA loaded (§8). A research note in the lab once prescribed 1.5–2.5 for dataset generation, "never Turbo's 1.0". The measurement overturned it for face-visible cells.

**Textured or patterned wall wording prints its pattern onto the skin — at 1024 native, on every seed.** "White studio cyclorama" rendered as terrazzo speckle on hip, thigh and torso; "bare studio" became patterned wallpaper whose cells printed across the abdomen; a bedroom wallpaper printed as orange-peel. All three failed on every seed, so it is the wording, not seed luck, and every cell against a *smooth, plain painted wall* was clean. Name the wall that way, or reuse a setting already proven clean. Observed on the `fineporn_v4_int8` finetune `[live-use — media lab, Ciara dataset v4 inspection, 2026-09-09]`.

**Generate at LoRA strength 0.0, always.** The dataset comes from the base checkpoint, never from the character's own previous LoRA — *"we can't use the lora to train the next lora"*. That is what makes a synthetic set a clean source rather than a recursion. Settled render settings for the synthetic character: `fineporn_v4_int8`, 12 steps, cfg 1.0, `wan_2.1_vae`, LoRA 0.0, one baseline seed. Dataset prompts ran 243–434 words, far past the ~45-word inference budget in §10, because they describe a cell rather than fight a LoRA.

**How much synthetic is safe is contested, inside the lab's own documents.** The only *measured* points on a real-person pool are 12–14% fine and **64% a failure** ("body learned, face regressed" — every synthetic carried one generator's rendering of the face). Nothing between has been run; a later arm ran at ≤40% on the stated direction "almost entirely synthetic", and a high-variety 65% set is prepared but unlaunched `[live-use — media lab, krea2-v15 vs DATASET-DESIGN, 2026-09-06]` `[contested]`. The fully synthetic character is a different case: 100% synthetic *from the base*, not from a predecessor LoRA, and it worked. The hazard is recursing generations, not the presence of renders. The literature cap and the seeding rules are in the sibling's `synthetic-datasets.md`.

**None of this transfers to Qwen-Image**, even though Krea 2 is Qwen-derived and shares this VAE. The 1024-native source rule (Qwen's band is the 1328 class), flow shift 2.5 (Qwen's is 2.2), LoKr as the character default, grad-accum 2 and train-on-Raw/run-on-Turbo are all Krea findings; Qwen has one undistilled base per generation and stacks Lightning at inference. The Qwen side of the line is [`qwen-image/references/lora-training.md`](../../qwen-image/references/lora-training.md) §9.

## 10. Deployment and inference rules

`[live-use — media lab, krea2-v6/v9/v13 (Amy) and krea2-v5 (Ciara), 2026-08-27 → 2026-09-09]`

**Deploy at strength 1.0.** Early runs shipped at 1.1, and the read was correct: needing above-normal strength to assert identity is the weak-likeness signature. It also means there is no stacking headroom. Later, better-fit runs ship at 1.0. On the adult finetune the band is 1.0–1.1: *"0.7 face clearly not her; 0.9 sort of there; I wouldn't go below 1."* JahJedi's 0.8 (§6) is the community number for a 474-image LoRA under heavy style mixing. The two are not in conflict; they are different LoRAs. A FLUX.2 style trainer's rule that "a LoRA needing 1.0 is a red flag" `[community — Herbst]` is the style-LoRA form of the same signature. A monotone rise in likeness up to 1.1 is a diagnosis, not a setting.

**The same LoRA gives a different body on each deployment checkpoint.** Turbo renders the synthetic character slim and small-chested. The adult finetune renders her noticeably bustier with wider hips, from the same LoRA and prompt. Record this in the LoRA's sidecar so nobody expects one body from both.

**Checkpoint identity is not portable.** Turbo and Raw render *a different woman* from the same prompt — no freckles, different face. The synthetic character exists only on the finetune she was rendered and evaluated on. The workaround "dressed shots on Raw, nudes on the finetune, face-swap between" fails, because the face is exactly what differs. Pick the deploy checkpoint before you build the dataset, and render the dataset on it.

**Prompting a LoRA render.** The rules are in [`prompting-guide.md` §3](prompting-guide.md), where the texture stack inverts once a LoRA is in the graph; the anchors are: no face words (the LoRA owns face, skin, freckles and build, and a long face block *claims the frame*); **~45 words**, since the same ideas at ~95 render freckles as reptile scale; no base-model realism tail — `"an unretouched film photograph."` is the whole tail; never describe anything meant to be off camera (trigger, state of dress, then only what is in frame landed 18/18 where framing words had failed four sweeps); and against a LoRA that learned a crop, a taller canvas (832×1216) beats any wording. Any slimness word shrinks body features without narrowing the torso; "petite" in the lead sentence was the one word that narrowed the frame.

**Wording is only valid at a stated LoRA strength.** At 1.0 the character LoRA pulled the body small, and only exaggerated wording landed on the reference. At 0.3 the *same* words read literally and overshot. Record the strength with the prompt, or the recipe is unreproducible.

**The repaint stack for getting the real identity onto a generated body.** Every variant at 832×1216 came back soft, waxy and over-freckled. What settled it: **body pass at 1216×1792 with the LoRA at 0.3** (below ~0.5 the base owns the shape, size words work again, and the face is no longer her), then **one** face pass into a 1024² crop at **LoRA 1.0, denoise 0.5**, and **no full-image pass**. Denoise 0.30 is too timid. 0.6+ drifts the hair. Chaining several 0.5 face passes drifts rounder each step. What fixed the waxiness was more pixels under the face crop, not more passes. For plain single-image work the lab's default is simpler, and it beat the face-pass route: **the LoRA at full strength in a single pass with in-frame-only prompting.** A face pass at denoise 0.5 reskins the base's face structure rather than replacing it, and 0.7+ brings back the double-pass softness `[live-use — media lab, Amy repaint sweeps, 2026-09-05]`. The community denoise ladder for Turbo img2img without a LoRA — 0.4 / 0.55 / 0.7 — is in `setup-and-workflows.md §7d`.

**Face identity passes on video frames are face-only.** Never run a full-image identity pass on a frame bound for a video model. The frame's job is to match the driving clip (`characters.md §7`).
