# Krea 2 — the media-lab run ledger

This file is the evidence behind the rules in [`lora-training.md`](lora-training.md) §3a, §3b, §9 and §10. It records what one lab changed and what it measured, run by run, across **twenty-one AI-Toolkit Raw-path runs on one config**. Sixteen were on a real person with a scarce photo pool (Amy). Five were on a fully synthetic character (Ciara). They ran between 2026-08-24 and 2026-09-09. Read it when you want to know *why* a rule says what it says, or when your own run looks like one of these. The rules themselves stay in `lora-training.md`. The model-agnostic craft they produced is owned by [`character-lora-training`](../../character-lora-training/): count is not the lever, the identity ratio, crops are duplicates, the three-quarter canary, the resolution rule.

**How to read it.** Everything here is `[live-use — media lab, <run>, <date>]`. That means one lab's rented-GPU runs, judged by one person, with most arms at n=1. These results are first-hand and measured, which most community sources are not. They are also one lab, not consensus. A run id is `krea2-vN` per character. A dataset id is `dataset vN`, and a dataset is frozen once a run references it.

## Contents

1. [The runs, in order](#1-the-runs-in-order)
2. [Rank, steps and rungs — the numbers](#2-rank-steps-and-rungs--the-numbers)
3. [Hardware and cost](#3-hardware-and-cost)
4. [The resolution sweep, and the wrong turn](#4-the-resolution-sweep-and-the-wrong-turn)

---

## 1. The runs, in order

The recipe (`lora-training.md` §3a) was never varied. It is Raw, LoRA 32/32, `adamw8bit` at 1e-4, `flowmatch` with `linear` timesteps, fp8 on both sides, `[1024]`, dropout 0.05, `save_every` 250, samples off. Every arm changed the dataset, or changed one training variable against an otherwise identical arm.

### Amy — a real person, 13–74 photos `[live-use — media lab, krea2-v1…v16, 2026-08-24 → 2026-09-07]`

| Run | Dataset | What changed | What was measured | Rule it produced |
|---|---|---|---|---|
| krea2-v1 | 25 real photos, 56% close-up (14 close-ups) | First run; 2250 steps | Peak c1750/2250. Best faces of the early sets; weak bodies; refused nudity. Likeness kept rising to strength 1.1 | 1.1 is the underfit signature, not a setting (§10); 0.33 is necessary, not sufficient |
| krea2-v2 | Same 25 | Rank 32→64, steps 2250→3000 | Lost a blind Turbo grid to v1 hidden as the control; every 0.6/0.8 cell "not even the same person". Facial *shape* improved mid-run and never converted into identity | Rank/alpha 32; capacity and length were exhausted before the dataset was |
| krea2-v4 | 74 photos, a strict superset of v1's 25 | Count ×3 | v1 the dominant winner on a 5-model × 5-seed blind face probe | Count is not the lever |
| krea2-v5 | 32% close-up set, 9 close-ups | Identity ratio | Inside the 0.25–0.40 band and still fails the tight crop; best bodies | Absolute close-up count matters; identity is learned per face-scale |
| krea2-v6 | 32 images, 43% close-up | Identity ratio | Between v1 and v5 on both faces and bodies | — |
| krea2-v8 | 15 coverage-built images | Count, against v6's 32 | "Very close" at under half the count | The lab's band is 13–32 |
| krea2-v9…v13 | Four real-photo arms | One dataset change per arm | The three-quarter probe lost every time (0/6, 0/4); the front close-up and full body moved independently of it. Over-trained rungs collapse the three-quarter first | Which probe regressed is the finding, not one likeness score |
| krea2-v10 | — | Captions *and* images changed in one arm | The loss could not be attributed | One variable per arm |
| krea2-v11 | 15 phone-JPEG frames; two flat, front-lit portraits removed | Source quality | Peaked c1750; later rungs "bake in a bit weird skin texture" — JPEG grain learned as skin. Removing the flat frames recovered the front close-up | Ties break downward; the source medium bakes in past the peak |
| krea2-v12 | dataset v16: two chest-up crops of photos already in the set | Crops as coverage | Subject read tanner and older (those frames now counted four times); heads cut off on head-and-shoulders prompts | A crop is a duplicate, and it teaches its framing |
| krea2-v13 vs v14 | dataset v18: hair colour/length, age and freckle density captioned on every image | Caption content | Hair became promptable ("auburn", "bob", "long" all obey). Age did not. Freckle density did not, and became a prompt variable that fought the LoRA | Caption the variable, never the identity; both stripped from later sets |
| krea2-v15 | dataset v19: 64% synthetic — 56 of 88 entries rendered by the previous LoRA | Synthetic share | "Body learned, face regressed": every synthetic carried one generator's rendering of the face, weakest at small scale, and training amplified it | The measured failure point; 12–14% had been uneventful |
| krea2-v16 | dataset v20: 65% synthetic with enforced variety — a seed per cell, twelve scenes, six camera styles | Synthetic share, gated | Prepared, not launched | The open question (`lora-training.md` §9) |

Three side experiments ran on the same character. The **trigger A/B** on dataset v13 compared base + trigger, LoRA without trigger, and LoRA with trigger. The LoRA drew the same person either way (2026-08-31). The **repaint sweeps** settled the body-pass and face-pass stack (2026-09-05). The **hi-res arm** rendered a 62-cell synthetic set at 2048×2560, 2048×3008 and 1536×2304. It produced "a weird grain that is very uncanny valley", and the wall diagnosed it, not the face (2026-09-07, §4).

### Ciara — a fully synthetic character, 32–62 rendered cells `[live-use — media lab, krea2-v2…v5, dataset v1→v5, 2026-09-06 → 2026-09-09]`

| Run | Dataset | What changed | What was measured | Rule it produced |
|---|---|---|---|---|
| krea2-v2 vs v3 | 62 identical cells, rendered at 2048 | Source-render step count, 20 against 12 | The 20-step set trained a fine speckle in; the 12-step set did not and won every in-distribution prompt (all-cell median 0.199 against 0.232) | 12 steps for dataset renders; 20 over-cooks skin into a pore grid |
| krea2-v3 | The 62-cell set rendered at 2048–2816 | Evaluated on Turbo only; shipped `final` | Reticulated "scale" skin. On the adult finetune, 1024×1536 was unusable at every strength (1.0 / 0.85 / 0.70 / 0.55) and ≥1344×2016 was good; c2500 was the only rung that survived the hard case where the Turbo-only grid had picked `final` | Evaluate on every deploy checkpoint × resolution (§8); the finetune has its own band |
| krea2-v4 (r16) and krea2-v5 (r32) | dataset v4: 32 cells at 1024 native, identity ratio 0.31 | Rank, on two pods in parallel, 3000 steps | Both clean at every rung on the finetune at 1024 and 1344 — the scale skin was the source render, not capacity. v5 won every table: Turbo triage medians 0.155–0.196 against 0.164–0.216, strict profiles 4/4 against 2/4; files 229 MB and 114 MB. `final` stopped overtraining. Both arms rendered "red hair" as *a different woman* on every checkpoint, because dark brown held 29 of 32 cells | Rank 32; the 1024-native rule; a majority binds to the trigger |
| — | dataset v5 | Colour redistributed across the same 32 cells so nothing has a majority (dark brown 10, auburn 8, copper 4, dark blonde 4, chestnut 3, black 3); jewellery on 16, makeup on 4 | Not yet trained at the time of writing | Redistribute; do not add cells |

Ciara's krea2-v5 is the evidence that a fully synthetic dataset works. It reached triage medians of 0.155 against a real-photo calibration p50 of 0.112, with clean skin, from cells drawn at LoRA strength 0 on the base. Its flexibility sheet also showed the trigger phrase printed on signage and name badges (`lora-training.md` §5).

---

## 2. Rank, steps and rungs — the numbers

**Steps by dataset size, as run.** 2250 on 25 images. 2500 on 13–16. 3000 on 32. 3500 on 54–91. That is roughly 90–170 steps per image, and the lab treats it as a loose guide only.

**Peak rung by dataset**, on the one recipe, picked by blind grid:

| Dataset | Peak rung | Note |
|---|---|---|
| 25 real photos | c1750 / c2250 | — |
| 28-image single-era set | c2750 / c3000 | A set with more views of the same photos tolerated one rung more |
| 13-image face set | c1750 | 2500 collapses the three-quarter view; 1250 undercooks the front |
| Two 15-image sets | c2250 / c2500 | — |
| 15 phone-JPEG frames | c1750 | Later rungs bake in JPEG grain as skin |
| 32-cell synthetic set | `final` (3000), c2500 equivalent | On the 1024-native set `final` no longer overtrains |

The table shows two things. The rung is per-dataset, and steps-per-image predicts it badly: 13 images peaked at c1750 and 28 at c2750. And **the peak tracks source quality, not size.** An all-phone-JPEG set peaks a rung earlier than one with DSLR frames, because past the peak the source medium trains in.

**The rank A/B, in full.** Ciara v4 (r16) against v5 (r32) used the same 32 cells and the same 3000 steps. r32 was better on every table (medians above). Amy v2 (r64, 3000 steps) against v1 (r32, 2250) put the r32 control first in a blind grid. Rank 32 is the setting on Krea 2. The ceiling is model-specific: 32 collapsed MiniMax H3 under the same trainer ([`minimax-h3`](../../minimax-h3/references/lora-training.md)).

---

## 3. Hardware and cost

All of these ran on the §3a config: fp8 on the DiT and the encoder, text embeddings cached, gradient checkpointing on, `low_vram` off, samples off.

| Card | 2500 steps | Speed | Cost | Note |
|---|---|---|---|---|
| RTX 5090 | 2h02 | ~2.9 s/it | ~$2.30 | The lab's rule became "5090 or higher for training" |
| RTX PRO 4500 | 3h49 | ~5.5 s/it | ~$2.90 | Half the throughput at 0.73× the hourly price |
| A100 | — | est. ~5 s/it | $1.39/hr | No fp8 units: slower *and* dearer on this recipe. Check the datatype before the price |

VRAM in use ran **15.8–19.2 GB of 32 GB** with `low_vram` off. Turning layer offloading on gained nothing. The card was the limit, not memory traffic. Raw previews cost about 90 seconds each at guidance 4, so a 6-prompt set every 250 steps burned more time than the training it watched. In the first measured run the face was still generic at step 750 and unmistakable by 1500. It was also sharper on Turbo than in any Raw preview. Samples were off from the seventh run on.

---

## 4. The resolution sweep, and the wrong turn

**The sweep** `[live-use — media lab, Ciara resolution sweep, 2026-09-08]`. It ran at LoRA strength 0 on the deploy finetune, with the seed fixed, 12 steps and cfg 1.0. Cheek patches were cropped as a fixed fraction of inter-eye distance, so every size showed the same patch of skin.

| Render size | At native size | After downscale to 1024 |
|---|---|---|
| 1024×1280 (inside the band) | Clean — discrete freckles | — |
| 2048 | Moderately crumpled | Still speckled; loses to 1024 native |
| 2560 | Heavily crumpled | Still speckled; loses to 1024 native |

The Wan 2.1 and Qwen decoders gave identical texture at 2048, so it was never the VAE. Training confirmed it the next day. krea2-v4 and v5 on the 1024-native set were clean at every rung (§1).

**The cfg check** `[live-use — media lab, no-LoRA sweep at seed 707, 2026-09-07]`. cfg 1.0 at 8 and 12 steps was clean. cfg 1.5, with or without a negative, was grainy across the background as well as the skin. The lab keeps one exception: face-hidden body cells may run cfg 1.5 with a negative, because mass words do not move at cfg 1.0 on the adult finetune.

**Wall wording** `[live-use — media lab, Ciara dataset v4 inspection, 2026-09-09]`. "White studio cyclorama" rendered as terrazzo speckle, and the speckle reappeared on hip, thigh and torso. "Bare studio" became patterned wallpaper whose cells printed across the abdomen. A bedroom wallpaper printed as orange-peel. All three failed on every seed, on `fineporn_v4_int8`. Every cell against a smooth, plain painted wall was clean.

**The wrong turn, as it happened** `[live-use — media lab, Amy hi-res arm and Ciara V4-PLAN, 2026-09-07/08]`. The real-person arm rendered a 62-cell synthetic set at 2048×2560, 2048×3008 and 1536×2304 to match the sibling character's regime, and got the uncanny grain. The first read was a freckle stipple, the subject's dense freckling meeting the base's 2048 regime. It was nearly written up as a hard limit for that character. Four "sharpeners" were then stacked on top of the over-band renders: the Wan VAE swap, texture-anchor words, a detail-heavy adult finetune, and yet more resolution. None was the cause. The synthetic arm inherited the diagnosis as a per-image freckle-placement problem. It built a plan around a canonical-face pilot, which was superseded before it ran. One observation on the grain broke the loop: *"you can even see it on the wall actually, so it's not a skin stipple."* The decisive control was a no-LoRA render that "should have been the second image made, not the twentieth". The rule that came out of it is in `lora-training.md` §9. Verify the resolution band and the step band, then look at the wall, before you diagnose skin.
