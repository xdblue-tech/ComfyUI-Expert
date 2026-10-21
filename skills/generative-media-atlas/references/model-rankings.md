# Model rankings — the evidence, and the trade behind each verdict

SKILL.md states the verdicts. This file carries the reasoning, the second-place cases and the
axis-by-axis matrix. **It owns comparison only.** Settings, filenames, node wiring and licence clause
text belong to the model skills. If anything here conflicts with a model skill, the model skill wins.

## Contents

1. [How to read a ranking here](#1-how-to-read-a-ranking-here)
2. [Photoreal faces and skin](#2-photoreal-faces-and-skin)
3. [Consistent characters](#3-consistent-characters)
4. [Character LoRA trainability — the three-way split](#4-character-lora-trainability--the-three-way-split)
   · [4.5 the Civitai census](#45-what-the-ecosystem-actually-looks-like--a-civitai-census)
5. [Structural control](#5-structural-control)
6. [Typography, anime and aesthetic range](#6-typography-anime-and-aesthetic-range)
7. [Hardware and VRAM](#7-hardware-and-vram)
8. [Licence, ranked by what it lets you ship](#8-licence-ranked-by-what-it-lets-you-ship)
9. [Video, by job](#9-video-by-job)
10. [The axis matrix](#10-the-axis-matrix)

**Adult work has its own file** — [`adult-work.md`](adult-work.md).

---

## 1. How to read a ranking here

**A ranking is a starting position, not a result.** Three things make these rankings softer than
they look:

- **The gaps are small at the top and large at the bottom.** On realism, first and second place
  are separated by a single pipeline pass. On typography, the gap between first and last is the
  difference between "works" and "impossible".
- **Most axes are won by composition, not by a model.** The suite's most-repeated finding is that
  the right answer is usually a chain of models. Compose where control is deepest, render where
  quality is highest, and finish where the finisher is best `[community — convergent across the suite]`.
- **Rankings inherit the model skills' confidence.** A model skill sometimes marks a claim
  `[contested]` or `[flagged — re-verify]`. Any ranking built on that claim carries the same weight,
  even when the ordering here reads as clean.

**Convergence is the strongest signal available.** Sometimes four independent model skills route to
the same model on one axis, as they do for [`z-image`](../../z-image/) on faces. That is a settled
finding, not one author's opinion. Where a ranking rests on a single named test, this file says so.

---

## 2. Photoreal faces and skin

**Order:** [`z-image`](../../z-image/) → [`flux-2`](../../flux-2/) [dev] → [`sdxl`](../../sdxl/)
photoreal finetune → [`krea-2`](../../krea-2/). [`ideogram-4`](../../ideogram-4/) and
[`anima`](../../anima/) are not contenders. Anima's card says realism is *"intended"* to be out of
scope.

| Model | The case for it | The trade |
|---|---|---|
| **Z-Image** | This is the axis the rest of the suite routes *here* for: [`sdxl`](../../sdxl/), [`krea-2`](../../krea-2/), [`ideogram-4`](../../ideogram-4/) and [`anima`](../../anima/) all send you here. It is also the suite's standard face-pass finisher over another model's render | It has one strong realism-leaning look, and that realism prior fights stylisation. ControlNet is **Turbo-only**. There is an anti-gloss tax to pay before it looks right |
| **FLUX.2 [dev]** | Strong when prompted with the camera/lens/film-stock vocabulary stack | [klein] skews over-sharpened into the AI look; [dev] is non-commercial and heavy |
| **SDXL photoreal finetune** | **The strength is control, not skin.** Juggernaut/RealVisXL plus the camera/film vocabulary gets a convincing frame. Inside that frame you can dictate pose, composition and identity in a way no newer model matches | Base SDXL alone will not get there. 77-token window, no in-image text |
| **Krea 2** | Good anatomy, animals and wide-aspect composition `[community — nsfwVariant]` | Expressions are the weak point. The default look is soft, and each image costs roughly 8× what Z-Image charges |

[`qwen-image`](../../qwen-image/) is not a contender here without a realism LoRA. Its own skill calls the
plastic default the tax and routes to Z-Image as the finisher, so it joins the table below as a composer,
not a renderer of skin.

**One finding matters more than the order: realism is a *pipeline position*.** Across the suite,
the same pattern keeps showing up. Practitioners compose and control in SDXL or Krea 2, then finish
the face in Z-Image at ~0.2 denoise `[community — nsfwVariant, Civitai]`. Ranking the models against
each other on this axis answers a question most practitioners stopped asking. The ladder and the
denoise bands live in [`image-production-workflows`](../../image-production-workflows/).

---

## 3. Consistent characters

"Consistent characters" hides two different questions, and they have different winners.

**Without training anything:**

| Rank | Model | Mechanism |
|---|---|---|
| 1 | [`qwen-image`](../../qwen-image/) | Edit-2511: one to three references, no adapter, and the dataset factory every image skill here routes through. *"I don't think there's a better open-weight model out there than Qwedit for making new shots of a character without loras"* `[community — nsfwVariant; strong]`. Apache-2.0, 273 character-tagged LoRAs. The trap is ComfyUI's edit node downscaling references to 1 MP; the model skill's bypass is the fix |
| 2 | [`flux-2`](../../flux-2/) | Native multi-reference (`ReferenceLatent`) plus PuLID — the only PuLID-class face adapter in the suite, which an edit model cannot replace. No likeness head-to-head against Qwen-Image-Edit has been published `[contested]` |
| 3 | [`sdxl`](../../sdxl/) | The deepest adapter toolbox — InstantID, HyperLoRA, IP-Adapter FaceID — plus `[SEP]` routing for several characters |
| 4 | [`krea-2`](../../krea-2/) | Identity Edit LoRA v1.2 is mature in adoption, and it is **the standard tool for prepping a video character swap**. But it is *unofficial*: a community fine-tune of Krea 2 Raw by conradlocke, *"not affiliated with or endorsed by Krea.ai"*, and it needs the `ComfyUI-Krea2Edit` node pack for its dual conditioning `[community — Enshitification, 611 pts]` |
| — | [`z-image`](../../z-image/) | No adapter shortcut exists. The path is LoRA + FaceDetailer or nothing |
| **see §3.1** | [`ideogram-4`](../../ideogram-4/) | No adapter, no edit variant, one character-tagged LoRA on Civitai (36 total, 2026-09-09) — **and a working no-training path anyway** |
| — | [`anima`](../../anima/) | **Knowledge-first**: it knows thousands of characters by tag, but identity transfer by reference is immature |

### 3.1 The Ideogram canvas trick — and why the suite got this wrong

**The finding.** A published workflow gets consistent characters out of
[`ideogram-4`](../../ideogram-4/) with no adapter, no edit model and no training
`[community — reality_comes, 402 pts, ~2026-06]`. It works like this. Put the reference on the
**left half of a wide canvas** and lock that half so it cannot be altered. Then prompt the model to
complete the canvas as **two photos of the exact same person**, describing the new scene for the
right half only. Cut the canvas and keep the right side.

**The mechanism is the part that generalises.** The model is composing *one* image that contains two
depictions. It can see the locked half the entire time it paints the other half, so it does what any
generator does when asked to draw a character twice in one frame: it keeps them consistent. The
workflow adds no identity mechanism. It **borrows one the model already had**, and turns an
intra-image consistency behaviour into an image-to-image one.

**Why this suite missed it — the transferable lesson.** Every fact in the old verdict was
right: no identity adapter, no edit variant, one character LoRA on Civitai (§4.5). The error was
**inferring incapability from absent tooling**. Tooling is what gets counted and catalogued, so a
tooling-shaped survey ends up silently answering a tooling-shaped question. Before you accept "model
X can't do Y" anywhere in this suite — including here — ask whether the evidence is about the model
or about its ecosystem, and go look for a *workflow*, not a node.

**What it does not settle.** Nobody has compared this trick against PuLID or a trained LoRA on
likeness. It spends a wide canvas of resolution per generation. And Ideogram's weights remain
non-commercial, so the path exists but gate 1 may still rule it out.

### 3.2 Where the practice actually is

A year-sorted r/StableDiffusion sweep on 2026-08-23 for `pulid`, `instantid`, `identity edit`,
`character consistency` and `consistent character` returned **no top post naming PuLID or InstantID
at all**. The identity conversation is now about **edit models, mixed freely**. Practitioners
describe reaching for Krea 2 Identity Edit, Flux [klein] 9B and Qwen-Image-Edit inside a single job,
and taking whichever one lands the frame `[community — blackmixture 1796 pts, DeerWoodStudios 461 pts]`.
Alongside that sit reference-*sheet* approaches: build a 360° sheet with a video model, then feed it
back as multi-image reference `[community — bstr3k, 1482 pts]`.

**Read this carefully.** Absence from a search is not evidence of inferiority. The adapters still do
things edit models cannot, and the capability ordering above stands. What the sweep shows is where
the workflows, the troubleshooting and the help are, and that is a real cost when you get stuck.
**If you choose an adapter-based route today, you will debug it alone.**

**All three edit models now have a skill, and the largest by ecosystem is the newest.**
[`qwen-image`](../../qwen-image/) covers Qwen-Image-Edit. The family carries 1,637 Civitai LoRAs, 273
character-tagged, against [klein] 9B's ~670 — counted by full pagination with `nsfw=true` on 2026-09-09,
where the 1,162 / 192 figure this file carried earlier the same day excluded NSFW
`[official — Civitai API, 2026-09-09]`. The open line shipped between 2025-08 and 2025-12 and then
froze, since Qwen-Image 2.0 and 3.0 are hosted-only, so the skill will age slowly. What moved the
ranking above is not the count. It is that Qwen-Image-Edit is where the image skills already send
their anchors, and that it needs no adapter. The handoff is unchanged:
[`image-production-workflows`](../../image-production-workflows/)'s pixels-in, pixels-out rule applies.

### 3.3 Several characters, and identity across a pipeline

**Several named characters in one image** is a separate and less settled problem. Krea 2's
Differential Output Preservation on a LoKr run is the most promising answer in the suite: up to four
characters with minimal bleed. But it **rests on one author's report**, and the same author found it
fails on Z-Image Base. Above four characters, or if DOP does not replicate, SDXL's regional
prompting is the fallback. Details and the caveat: [`krea-2`](../../krea-2/) `references/characters.md`.

**Identity across a whole pipeline** is owned by
[`character-lora-training`](../../character-lora-training/). The deciding rule is sequencing,
not model choice: **re-assert identity after the last whole-image pass**, in the detailer stage,
because every refine above ~0.35 denoise erodes identity.

---

## 4. Character LoRA trainability — the three-way split

**"Easiest" is three different questions.** Answering it as one question is the most common mistake
on this axis. The three questions have different winners, and the trade between them is the actual
decision.

### 4.1 Best likeness out of the run

One named cross-model test exists: **MesmerTools, published 2026-07-14**. It trained the same two
subjects — a face the field finds easy, and a South Asian man the field historically does not —
across six bases. That took ~30 runs and four days of GPU time on two 16 GB cards, using fp8 for
most models, scene-only captions and seed-locked evaluation prompts. Their ordering, verbatim in
substance:

| Rank | Base | The author's note |
|---|---|---|
| 1 | **Ideogram 4** | Best likeness on both faces; held up on novel prompts using its native JSON caption format |
| 2 | **Krea 2** | Mid-pack when sampled on Raw — *"jumps to #2 once you sample with Turbo"* |
| 3 | **FLUX.1 dev** | *"The old reliable, now hitting its ceiling"* at fp8 on 16 GB |
| 4 | **Z-Image** | *"Fastest by a mile. Punches above its size"* — ~90 min/run, good by step ~900 |
| 5 | **FLUX.2 [klein]** | The FLUX.2 that fits (~4 GB); showed skin-lightening on the darker subject until captions and rank were adjusted |
| 6 | **FLUX.2 [dev]** | *"Couldn't tame it. The only DNF"* — ~90 GB system RAM to quantise, abandoned after two paid attempts |

`[community — MesmerTools, 2026-07-14; single source]` **Treat this as one data point, not a
benchmark.** It is one author, one dataset pair, one hardware tier. The author's own caveat is that
the first two evaluation rounds used prompts too basic to expose overfitting. Even so, it is the only
controlled cross-model comparison published, and it agrees with the one other cross-model
report the suite carries, which rated Ideogram 4 above Krea 2 and Z-Image at learning tattoos
`[community — Any_Tea_3499; single report]`.

### 4.2 Fastest loop

Two models win here. **[`z-image`](../../z-image/)** takes ~90 minutes per run, with useful results
by step ~900 rather than 3,000. **[`anima`](../../anima/)** trains a LoRA from roughly **6 GB at
768 px** `[community — citronlegacy, Civitai 26217; convergent]`.

This axis is undervalued. **The bottleneck in learning to train is not the run. It is the three
failed runs it takes to discover what a dataset is missing.** Maybe a rear angle is missing, or a
jacket shows up in twenty of thirty images, or one expression runs throughout. At 6 GB, those
failures cost nothing. At 24 GB rented, they cost real money, so you take fewer of them.

### 4.3 Most documented

**[`sdxl`](../../sdxl/)**, by a wide margin. It has years of settled recipes, two mature trainers,
and separate Pony/Illustrious/NoobAI LoRA pools with their own conventions. Nothing else in the
suite already has an answer written for the question you are about to hit.

### 4.5 What the ecosystem actually looks like — a Civitai census

Measured directly against the Civitai API on **2026-08-23**, with the rows marked **‡** re-counted by
full cursor pagination on **2026-09-09** `[official — Civitai /api/v1/models, counted this pass]`.
Capped counts (`+`) are lower bounds from paginating 2,200 deep. Unmarked figures are exact, reached
by exhausting the cursor. **Once these numbers are weeks old, re-run the count instead of trusting
it:** `python scripts/civitai_census.py --pages 22` (add `--tag character` for the character column).
The API has no total field and answers 403 to a bare Python User-Agent; the script handles both, and
its header says how.

| Base | LoRAs total | tagged `character` | What it means |
|---|---|---|---|
| [`sdxl`](../../sdxl/) family — SDXL 1.0 / Illustrious / NoobAI / Pony V6 XL | 2,200+ each | 1,200+ each | The largest absolute pools, as expected |
| [`z-image`](../../z-image/) **Turbo** | **2,191+** | **1,198+** | **A peer to the SDXL family.** The "young ecosystem" framing is wrong |
| [`z-image`](../../z-image/) **Base** | 671 | 201 | **The ecosystem lives on Turbo**, not Base — see the trap below |
| [`krea-2`](../../krea-2/) | 2,199+ | 1,166 | Enormous for a model this new |
| [`anima`](../../anima/) | 2,197+ | 1,199+ | Same, and it is the newest model in the suite |
| [`qwen-image`](../../qwen-image/) family ‡ | **1,637** (1,162 without `nsfw=true`) | **273** (192) | Above [klein] 9B, an order of magnitude above H3 and Ideogram. Apache-2.0 throughout. **The only row counted with `nsfw=true`** (full pagination, 2026-09-09); the script behind the other rows does not pass the flag, so compare by the same method — see the bullet below |
| FLUX.1 dev | 700+ | 1,197+ | The legacy pool, still deep |
| FLUX.2 **[klein] 9B** ‡ | ~670 (653 on 08-23) | 178 (08-23) | Modest growth; character column not re-run |
| FLUX.2 **[klein] 4B** | **133** | **16** | The Apache-2.0 variant has ~5× fewer LoRAs and ~11× fewer characters |
| [`ideogram-4`](../../ideogram-4/) ‡ | **36** | **1** | Exact. **One** character-tagged LoRA on the main host (34 / 1 on 08-23; essentially unchanged) |
| [`wan-2-2`](../../wan-2-2/) 2.2 (I2V + T2V + 5B) ‡ | 501 | 6 (I2V, 08-23) | A real ecosystem — but almost none of it is character work. Flat since August |
| [`minimax-h3`](../../minimax-h3/) ‡ | **86** (22 on 08-23) | **29** (14) | **Quadrupled in seventeen days.** Still character-weighted; the big downloads are NSFW concept LoRAs at 40–47k |
| [`ltx-2-5`](../../ltx-2-5/) — 2.5 / 2.3 | **3 / 168** (08-22, not resampled) | — | Reproduced the 2.3-vs-2.5 adapter split; the *download* lead has since flipped to 2.5 (see §4.4) |
| [`scail-2`](../../scail-2/) | **none** | — | No `baseModel` entry exists, which confirms there is no training path |

**Velocity is a separate measurement.** Sampling the 400 most-downloaded LoRAs of the past month and
the 400 newest: [`anima`](../../anima/) took **208** and **167** of them, [`krea-2`](../../krea-2/)
**137** and **77**, the SDXL family **37** and **129**, [`z-image`](../../z-image/) **8** and **10**.
Anima also held **130 of the top 337 checkpoints**. So the SDXL family is still where most *new*
uploads land after the two hot models, while Anima and Krea 2 hold current attention.

**Three things this changes:**

- **[`z-image`](../../z-image/)'s pool is on the wrong variant for its own doctrine.** The skill's
  training doctrine is *train on Base, deploy via the detailer swap*. But 2,191+ published LoRAs
  target **Turbo**, against Base's 671. So downloading someone else's Z-Image LoRA means getting a
  Turbo LoRA. Those generally load on Base at reduced strength, but that is a convenience, not a
  plan. Check the variant before you assume a LoRA fits your graph.
- **The FLUX.2 licence escape hatch costs you the ecosystem.** [klein] 4B is the Apache-2.0 variant,
  so it is the one to reach for under gate 1. But it has only 133 LoRAs to 9B's 653, and sixteen of
  them are character LoRAs. **Choosing the licence here means training your own.**
- **Ideogram 4's trainability win is one test against one published character LoRA.** §4.1 ranks it
  first on likeness, but the host has exactly **one** character-tagged LoRA on it, per the 2026-09-09
  pull (the [`ideogram-4`](../../ideogram-4/) skill says none; the difference is the `character` tag
  on one upload, and the tag is author-applied). Both facts are true. Together they say the model
  may well train well, and nobody has done it in public.
- **The `nsfw=true` query flag moves a count by ~40%.** Qwen-Image's pool is 1,162 without it and 1,637
  with it, 192 against 273 on the character tag `[official — Civitai API, 2026-09-09]`.
  [`scripts/civitai_census.py`](../scripts/civitai_census.py) does not pass the flag, so the other exact rows
  above are floors for the NSFW-inclusive pool, not counts. Compare rows by the same method, and re-run with
  the flag before quoting a total `[flagged — method matters here]`.

**Read these numbers for what they are.** Civitai is one host — an important one, but it **bans
real-person likeness entirely**, so it systematically undercounts a whole category. The `character`
tag is author-applied and noisy. "Most downloaded this past month" favours recent models mechanically.
And a capped figure is a floor, not a count.

### 4.4 The trade, stated plainly

- **Ideogram 4 wins likeness and loses shipping.** Its open weights are non-commercial with no
  escape variant. Its LoRA ecosystem is 36 models with **exactly one** tagged `character` (§4.5). And
  a LoRA is only as useful as the base you may deploy.
- **Krea 2's win depends on one detail**: train on Raw, *sample* on Turbo. Miss it and the same
  weights read as mid-pack.
- **Z-Image is the pragmatic default** for a synthetic character you will render commercially. It has
  a fast loop, Apache-2.0 on weights and outputs, and it already owns the realism axis.
- **The base you train on should be the base you will render on.** A LoRA usually loads on a
  distilled sibling at reduced strength, but that is a convenience, not a plan.
- **Video is a different world.** [`wan-2-2`](../../wan-2-2/) is mature — **two LoRAs, one per MoE
  expert, from one dataset** — with a first-class trainer. [`minimax-h3`](../../minimax-h3/) has
  moved from "young and unsettled" to **converging**: four trainers (ai-toolkit, musubi-tuner,
  diffusion-pipe, SimpleTuner) plus fal.ai hosted, a shared doctrine for the base's CFG distillation
  that every trainer now handles, 86 Civitai LoRAs, and one lab's measurement that a stills-only
  character LoRA carries identity into motion at thigh-up framing (0.200 median face distance vs
  0.920 without it) `[live-use — media lab, Ciara h3-v1, 2026-09]`. The limits are still specific:
  rank 32 collapsed under ai-toolkit, so r16 is the ceiling there; H3's weakness is likeness, not
  adherence (its stills LoRA sat ~1.5× further from canon than the same dataset's Krea 2 LoRA); and
  the whole doctrine is weeks old. The model skill's `references/lora-training.md` owns it.
  [`ltx-2-5`](../../ltx-2-5/) has a capable first-party trainer, but 168 of ~171 community LoRAs sat
  on 2.3 on 2026-08-22, even though 2.5 now leads 2.3 on Hugging Face downloads (1.64M vs 1.18M a
  month on 2026-09-09) — the adapters have not followed the downloads yet — and **your LoRA inherits
  the licence and carries the obligation to whoever you give it to**. [`scail-2`](../../scail-2/) has
  no training path at all. Outside the suite, **HunyuanVideo-1.5** has musubi-tuner LoRA support and
  the EU/UK/KR exclusion (§8); it is not covered here.

Everything about datasets, captioning, hyperparameters, evaluation and publishing lives in
[`character-lora-training`](../../character-lora-training/).

---

## 5. Structural control

**Order:** [`sdxl`](../../sdxl/) ≫ [`z-image`](../../z-image/) / [`flux-2`](../../flux-2/) /
[`qwen-image`](../../qwen-image/) → [`anima`](../../anima/) ≈ [`krea-2`](../../krea-2/) →
[`ideogram-4`](../../ideogram-4/).

| Model | Stack | Gap |
|---|---|---|
| **SDXL** | Union ControlNet, IP-Adapter, regional prompting — all mature | xinsir's SDXL ControlNet training has stalled: frozen, still SOTA |
| **Z-Image** | Fun Union ControlNet | **Turbo only** |
| **FLUX.2** | Fun Union ControlNet via custom nodes | Younger ecosystem |
| **Qwen-Image** | Native ControlNet inside Edit-2509+ (depth, edge, keypoint), InstantX ControlNet-Union for T2I, DiffSynth canny/depth/inpaint as model patches through the same node pair Z-Image uses | Ranked by evidence of use, not by what exists: EliGen and Blockwise-ControlNet show zero monthly downloads. No published head-to-head against the Fun Union stacks |
| **Anima** | LLLite: lineart, depth, scribble, inpainting | **No pose, no canny, no HED** — pose is the weak one |
| **Krea 2** | Two community ControlNet-LoRAs: **depth** and **OpenPose** (thedeoxen, 2026-08-04 — DWPose skeleton in, body pose followed) | No canny or union, and the two are separate adapters. Having pose is what moves it level with Anima |
| **Ideogram 4** | `bbox` layout only | No control or identity adapter exists from anyone |
| **SenseNova U1.5** — *not covered* | Region-controlled edit via masks, bounding boxes and visual markers, in ComfyUI core since 2026-09-01 | A different kind of control (edit-time region, not generation-time structure). Licence unverified here |

**This ranking is why the control front-end pattern exists.** Compose in SDXL, where the pose, depth
and regional stack actually works, then render or refine in the DiT that has the quality. Regional
prompting does **not** transfer: SD-era regional tooling fails on DiTs, and core attention-masking is
the only approach that works there. Whether per-region *LoRA* application works on DiT regional
setups is unsettled `[contested]`.

---

## 6. Typography, anime and aesthetic range

**In-image typography.** [`ideogram-4`](../../ideogram-4/) leads on layout and dense lettering: JSON
captions, `bbox` layout, text layers, transparency. [`qwen-image`](../../qwen-image/) is the second, and
the first for two jobs Ideogram does not do — bilingual Chinese/English rendering, and editing text
already in an image while keeping its font (Edit-2509+) — under Apache-2.0. FLUX.2 is "good, high
variance". Z-Image is workable for short bilingual text. SDXL basically cannot do it. Krea 2 is
unreliable. The trade at the top is licence. Ideogram's open weights are non-commercial *and* have no adapter ecosystem, so the typography
plate is usually a **stage in someone else's pipeline** rather than the pipeline.

**Anime and booru illustration.** [`anima`](../../anima/) is the anime-native base with a modern
encoder, and the community reads it as Illustrious's successor. [`sdxl`](../../sdxl/)'s
Illustrious/NoobAI/Pony V6 XL finetunes are the mature alternative — deeper LoRA pools, real
ControlNets, and **the only path where you may ship the weights**. Two things about that pool moved.
**Pony V7** (2025-10-08) is a ~7B **AuraFlow** model, not an SDXL finetune: its own LoRA pool, its
own trainer (SimpleTuner), no `score_9` ladder, a licence that bars inference services and >$1M
companies, and V6 LoRAs do not load on it. When someone says "Pony", ask which; the
[`sdxl`](../../sdxl/) skill keeps V6 XL as its Pony. And the **NoobAI** line has shipped no
checkpoint since 2024-12; its team published a LoRA trainer for **SenseNova U1** instead
(`Laxhar/sensenova-u1-lora-trainer`, 2026-05-09), which is the best current answer to "is there a
NoobAI successor" — not a model, a migration. SenseNova is not covered here. Krea 2 covers anime
*looks* but not the booru tag vocabulary or the character-by-name knowledge that ecosystem runs on.
Z-Image is not in this contest at all. It is a photoreal-leaning sentence-prompted generalist, and it
fights both the dialect and the aesthetic.

**Aesthetic range without checkpoint-hopping.** [`krea-2`](../../krea-2/) is deliberately built with
no house look. It offers style references, moodboards, official style LoRAs and 1,500+ community
ones. The trade is its two taxes (soft default, muted expressions) and the ~8× per-image cost
against Z-Image. SDXL gets range too, but by *switching finetunes*, which is a different workflow.

---

## 7. Hardware and VRAM

These are practitioner figures. **Several vendors publish no inference VRAM number at all.** Treat
them as where the practice lands, not as thresholds.

| Model | Comfortable | Notes |
|---|---|---|
| [`sdxl`](../../sdxl/) | **6–8 GB** fp16 (~6.5 GB); ~4 GB with offload | No usable GGUF path — it is a Conv2D-heavy UNet and the GGUF author says don't. Budget 8 GB+ for base+refiner |
| [`anima`](../../anima/) | 8 GB floor, 12 GB comfortable | 4.18 GB checkpoint. The floor is **one report on one AMD card** `[flagged — re-verify]` |
| FLUX.2 **[klein] 4B** | ~8 GB fp8 / ~13 GB fp16 | The Apache-2.0 variant |
| [`z-image`](../../z-image/) Turbo | 6–8 GB quantised `[community — re-verify]` | Official NVFP4 DiT ~4.5 GB, **Turbo only** |
| [`krea-2`](../../krea-2/) | 16–24 GB | 13.1 GB fp8_scaled + 5.2 GB encoder; GGUF Q2_K 4.9 GB → Q8_0 13.7 GB below that |
| [`z-image`](../../z-image/) Base | ~16 GB | Inferred from parameter count, not measured `[flagged — re-verify]` |
| FLUX.2 **[dev]** | ~20 GB fp8mixed (35.5 GB file) | GGUF Q2_K ~13 GB → Q8_0 ~35 GB |
| [`ideogram-4`](../../ideogram-4/) | 24 GB (nf4, CUDA-only) | Gated on HF — authenticate or downloads 404 |
| [`wan-2-2`](../../wan-2-2/) | 12 GB (14B at Q4_K_M) / 8 GB (5B) | **Below ~12 GB, 5B fp16 beats 14B at Q3_K** `[community — re-verify]` |
| [`scail-2`](../../scail-2/) | 16 GB fp8_scaled; 8–12 GB GGUF + chunking | No vendor baseline exists `[contested]` at 16 GB |
| [`ltx-2-5`](../../ltx-2-5/) | **32 GB** per documentation | The same vendor's marketing says 16 and 12 in the same week `[contested]`. Clip length and decoder choice dominate card size |

**Training is the real constraint.** The suite's mainstream models need 16–24 GB to train, which is
why people rent ([`comfyui-on-runpod`](../../comfyui-on-runpod/)). [`anima`](../../anima/)'s ~6 GB
is the one exception.

---

## 8. Licence, ranked by what it lets you ship

Cleanest first. **The ranking changes depending on what leaves the building.** That is why SKILL.md
puts licensing in a gate rather than a table.

| Model | Sell pictures | Ship a pipeline / host the weights | The catch |
|---|---|---|---|
| [`z-image`](../../z-image/) | ✅ | ✅ | None — Apache-2.0 on weights *and* outputs. The least encumbered in the suite |
| [`qwen-image`](../../qwen-image/) | ✅ | ✅ | None — Apache-2.0 on code and every open weight, no output clause, no gate; tied with Z-Image. Open through 2512 and Edit-2511; 2.0 and 3.0 are hosted-only under Model Studio terms. Third-party Lightning LoRAs and ControlNets are outside Qwen's grant |
| [`wan-2-2`](../../wan-2-2/) | ✅ | ✅ | None — Apache-2.0, no territory, revenue, field-of-use or acceptable-use clause |
| [`scail-2`](../../scail-2/) | ✅ | ✅ | Apache-2.0 code, MIT weights card. But it cannot originate a shot |
| FLUX.2 **[klein] 4B** | ✅ | ✅ | Apache-2.0 — and it is the *only* FLUX.2 variant that is |
| [`sdxl`](../../sdxl/) | ✅ | ✅ | OpenRAIL++-M: no revenue cap, but its use-restrictions **travel downstream with every redistribution**. Turbo is the exception `[contested]` |
| [`krea-2`](../../krea-2/) | ✅ | ✅ under $1M | Community Licence revenue line |
| [`anima`](../../anima/) | ✅ | ❌ | The split *is* the story: outputs are explicitly outside the non-commercial term; the Model is not. Individuals may sell weights, but not inside a larger product |
| [`ideogram-4`](../../ideogram-4/) | ❌ from weights | ❌ | Purpose-restricted, with **no escape variant** — the fallback is the hosted API, not another checkpoint |
| FLUX.2 **[dev] / 9B** | ❌ | ❌ | Non-commercial wherever you run them; the licence travels with anything derived |
| [`ltx-2-5`](../../ltx-2-5/) | ✅ under $10M | ⚠️ | ¶20 bars competing with Lightricks' commercial products **at any revenue**, unqualified. The AUP bars explicit content universally. Your LoRA is a Derivative and the obligation travels |
| [`minimax-h3`](../../minimax-h3/) | ❌ in US/EU/UK/KR | ❌ there | A territory exclusion, not a limitation — it removes the model |

**The chain rule.** Selling *pictures* asks whether you may use each model. Shipping the *pipeline*
asks whether each model may be distributed. One non-commercial rung stops the whole chain, whether
it sits first or last. This rule is owned by
[`image-production-workflows`](../../image-production-workflows/).

**Outside the suite, five rows worth knowing before you add a model to a chain** `[official — model
cards and licence files, read 2026-09-09]`:

| Model | Sell pictures | Ship a pipeline | The catch |
|---|---|---|---|
| **Bernini-R** | ✅ | ✅ | Apache-2.0, no territory clause. A video-editing path that survives gates 1 and 3 where H3 does not. Not covered |
| **nvidia/PiD** | ❌ | ❌ | **NSCLv1, non-commercial.** A decoder/upscaler rung that drops into FLUX, FLUX.2, SD3, SDXL and Qwen-Image graphs, with a Comfy-Org template. The chain rule means one PiD rung makes the whole pipeline non-commercial |
| **Pony V7** | ✅ | ⚠️ | Custom licence: no inference services, no companies over $1M revenue, no professional video. A third revenue gate beside Krea 2 and LTX-2.5 |
| **HunyuanImage 2.1 / 3.0**, **HunyuanVideo-1.5** | ❌ in EU/UK/KR | ❌ there | Tencent Community Licence: territory exclusion plus a 100M-MAU cap. A second territory gate beside H3's, one row wider |
| **SenseNova U1.5** | ? | ? | Apache-2.0 per the GitHub repo; the Hugging Face card is gated and was not read. Verify before relying on it (flagged in [`ecosystem-map.md`](ecosystem-map.md) §6.1) |

One principle from the same sweep: **run the ladder against the checkpoint, not the lab.** Qwen-Image
3.0 went hosted-only with no weights, no licence and no technical report in the same four months
that Ideogram 4, Krea 2, MiniMax H3 and LTX-2.5 all released weights. "Historically open" is a
description of the past, not a licence.

---

## 9. Video, by job

| The job | The answer | The alternative, and its cost |
|---|---|---|
| **Animate a still** | [`wan-2-2`](../../wan-2-2/) — the strongest I2V under a licence with no conditions | [`minimax-h3`](../../minimax-h3/) or [`ltx-2-5`](../../ltx-2-5/) for higher raw fidelity, if their licences clear you |
| **Originate from text** | Any of the three generalists — but **lock a still first** with an image model. Wan's T2V gives up all spatial control | — |
| **Sound in the same pass** | [`minimax-h3`](../../minimax-h3/) (generates) or [`ltx-2-5`](../../ltx-2-5/) (generates *and* consumes) | The choice is a licence fork, not a quality one: territory exclusion vs revenue cap + no-compete + NSFW ban |
| **Lip-sync to an existing track** | [`wan-2-2`](../../wan-2-2/) S2V — it consumes audio rather than making it | H3 makes its own audio and will not follow yours |
| **Several cuts in one generation** | [`ltx-2-5`](../../ltx-2-5/), alone — and it is a prompting technique, not a node | — |
| **Replace a person, tracking their motion** | **A task split, not one winner.** [`scail-2`](../../scail-2/) — tracked frame-for-frame with SAM3 — for whole-body replacement, non-human subjects, complex action, multi-character scenes and low-res stability. Wan **Animate 2** ([`wan-2-2`](../../wan-2-2/), native ComfyUI 2026-08-08, no pose extraction, `WanAnimate2Cache`) for close-up faces, eye movement and lip sync `[community — Wensleydale on X, dreamerland.ai; two sources]` | The split was measured against the original Animate and has not been re-run against Animate 2 `[flagged — re-verify]`. H3's editing mode **re-generates** the motion and its identity latch gives out around 5–7 s |
| **Edit a clip — relight, restyle, insert a subject, remove an object** | **Nothing in the suite.** The open model that does it is ByteDance's **Bernini-R** (Apache-2.0, six modes t2i→rv2v, native ComfyUI 2026-06-14, GGUF and 4-step LoRA derivatives already exist) — **not covered here** as of 2026-09-09 | The boundary is clean: SCAIL-2 tracks a person, Bernini re-renders a scene. The suite's only partial answers are narrow LTX-2.3 style LoRAs |
| **Camera control** | [`wan-2-2`](../../wan-2-2/) Fun Camera — discrete moves | **Nothing does a freeform camera path.** That is a named gap, not an omission |
| **Post: upscale, restore, interpolate** | [`ltx-2-5`](../../ltx-2-5/) doubles as the generative video upscaler | **Restore before you interpolate, never after** — [`image-production-workflows`](../../image-production-workflows/) |
| **Train a video LoRA** | [`wan-2-2`](../../wan-2-2/) — mature, Apache-2.0, publishable | LTX's derivatives inherit its licence; H3 now has four trainers and a converging doctrine, r16 ceiling under ai-toolkit (§4.4), and its territory gate travels with the LoRA; SCAIL has no path |
| **Fit a 14 GB card** | [`wan-2-2`](../../wan-2-2/) 5B, or 14B at Q4_K_M | **HunyuanVideo-1.5** (Tencent, 8.3B, ~14 GB, native ComfyUI 2025-11-24, musubi-tuner LoRAs) is the other open answer and is **not covered** — and its licence excludes the EU, UK and South Korea |

**Out of scope, named so the silence is not ambiguous.** Every hosted video launch since May 2026 is
closed: **FLUX 3 / FLUX 3 Video** (BFL, GA 2026-08-04, 20 s clips with native audio, keyframe and
v2v continuation — the vendor behind [`flux-2`](../../flux-2/), but no weights; "FLUX 3 Dev" is a
roadmap item, tracked in [`ecosystem-map.md`](ecosystem-map.md) §4), **Wan 3.0** (API beta 2026-08-06, no weights), **Seedance 2.5**,
**Kling 3.0**, **Gemini Omni** (which supersedes Veo — Comfy archived Veo 3 on 2026-08-28),
**Grok Imagine 2.0**, **Runway Characters**, **Pixverse V6**. The native-audio open-weight field is
still exactly H3 and LTX-2.5. Dates and the full list are in
[`ecosystem-map.md`](ecosystem-map.md) §7.

---

## 10. The axis matrix

One table, eleven models, the axes that decide. **●** best in suite · **○** capable · **–** weak ·
**✗** cannot.

| | realism | identity (no train) | LoRA pool | control | text | anime | range | licence |
|---|---|---|---|---|---|---|---|---|
| [`z-image`](../../z-image/) | ● | ✗ | ● (Turbo) | – | ○ | – | – | ● |
| [`flux-2`](../../flux-2/) | ○ | ○ (● adapter) | ○ | ○ | ○ | ✗ | – | – (klein 4B ●) |
| [`sdxl`](../../sdxl/) | ○ | ○ | ● | ● | ✗ | ● | ○ | ○ |
| [`krea-2`](../../krea-2/) | ○ | ○ | ● | – | – | – | ● | ○ |
| [`qwen-image`](../../qwen-image/) | – | ● (edit) | ○ (1,637) | ○ | ○ (● bilingual, text edits) | ✗ | – | ● |
| [`ideogram-4`](../../ideogram-4/) | – | ✗ | ✗ (34) | ✗ | ● | ✗ | – | ✗ |
| [`anima`](../../anima/) | ✗ | – | ● | – | – | ● | – | ○ images / ✗ weights |
| [`wan-2-2`](../../wan-2-2/) | video | ○ | ● video | ● video | ✗ | ○ | ○ | ● |
| [`minimax-h3`](../../minimax-h3/) | video | ○ | – | – | ✗ | ○ | ○ | ✗ US/EU/UK/KR |
| [`ltx-2-5`](../../ltx-2-5/) | video | – | – | ○ | ✗ | ○ | ○ | – |
| [`scail-2`](../../scail-2/) | video | ● (replace) | ✗ | – | ✗ | ○ | – | ● |

**The `LoRA pool` column measures size, not recipe maturity**, and the two came apart in the §4.5
census. [`z-image`](../../z-image/) Turbo, [`krea-2`](../../krea-2/) and [`anima`](../../anima/) now
match the SDXL family on published LoRAs, but [`sdxl`](../../sdxl/) still owns the *settled recipes*
— the years of accumulated craft that tell you what to do when a run goes wrong. Size tells you what
you can download. Maturity tells you what you can train without inventing the method.

**Read the columns, not the rows.** No row wins. Every column has one clear answer. That is the
argument for the elimination ladder in SKILL.md, and for chaining models rather than choosing one.

**What the matrix leaves out, deliberately.** Only published skills get a row, because a row is a
summary of a model skill and there is nothing to summarise for the others. Three open models would
earn one on capability alone — **Bernini-R** (video editing), **SenseNova U1.5** (native 4K, region edit) and
**HunyuanVideo-1.5** (14 GB video, territory-gated). They are listed as *not yet covered* in
SKILL.md's suite map and in [`ecosystem-map.md`](ecosystem-map.md) §6, with where to route until a
skill exists. Read an axis where one of them would win as "the suite's answer, among what it
covers", not as "the best available".
