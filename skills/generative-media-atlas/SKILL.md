---
name: generative-media-atlas
description: >
  The entry point to the generative-media suite. It tells you which open image or video model to
  use for a job, how they rank and trade off, which skills to install, and in what order to use
  them. Use this whenever the user is choosing,
  comparing or planning rather than operating a model they have already settled on, even when they
  never say the words "which model". Reach for it early: the constraints it screens for (licence,
  territory, VRAM, dialect) are cheap to check now and unrecoverable later. Triggers:
  "which model is best for photoreal faces / anime / typography / text in images", "which is
  easiest to train a character LoRA on", "rank these for realism", "Z-Image vs Krea 2 vs FLUX.2",
  "what can I run on 12 GB", "which video model gives me audio", "can I sell what this makes",
  "which licence lets me ship the pipeline", "best model for NSFW / adult / uncensored work" and
  which ones bar it, "I want realistic photos of a character I invented, in ComfyUI on RunPod —
  where do I start", or any multi-step goal crossing model choice, LoRA training, pipeline design
  and GPU deployment. It also owns getting the skills onto the machine: what to install with
  `npx skills add`, which siblings a job needs, the canonical skills published outside this
  suite by RunPod, Comfy-Org, Hugging Face and Black Forest Labs, and which strong open models
  have no skill here yet. Installed alone, it tells you which others to pull and when. Per-model
  settings, filenames, node wiring, prompt dialects and licence
  clause detail live in the model skills. This skill owns the comparison between them, the route
  through them, and the decision of which ones you need. When a question could plausibly be
  answered by a model skill or by this one, start here. Routing costs one read, and choosing the
  wrong model costs a week.
---

# Generative media atlas

This is the **map** of the suite: the skill you read *before* you know which skill you need.

Every other skill here answers "how do I do this with model X". This one answers the questions that come first:
**which X, why that one, what it costs you, which skills to install, and in what order.** Then it hands you off.
It is deliberately the one skill in the suite that works on its own, because it is the one people install first.

Two ideas organize everything below:

1. **Model choice is an elimination, not a beauty contest.** The constraints that rule a model out are binary
   and public. The quality differences between the survivors are small and come down to taste.
2. **A finished result is a route through several skills, not one.** Locking a character is one job. Training
   its LoRA, rendering it well, and running it on a rented GPU are three more. Each is a different skill's job,
   and the expensive mistakes happen at the joins.

---

## What this owns, and what it doesn't

**Routing outward is the whole job.** If the answer changes when you change the model, it is not this skill's
answer to give.

| Question | Where it belongs |
|---|---|
| **Which model for this job, and how do they rank** | **here** |
| **What a ranking costs you — the trade behind the winner** | **here** |
| **Which skills to install for a stated goal, and in what order to use them** | **here** |
| **The end-to-end route from "I want X" to a finished file** | **here** — the playbooks |
| **Which canonical skills exist outside this suite, and what each is actually for** | **here** |
| Node settings, filenames, prompt dialect, sampler, resolution, that model's licence text | the model skill — [`flux-2`](../flux-2/), [`ideogram-4`](../ideogram-4/), [`z-image`](../z-image/), [`sdxl`](../sdxl/), [`krea-2`](../krea-2/), [`anima`](../anima/), [`qwen-image`](../qwen-image/), [`wan-2-2`](../wan-2-2/), [`minimax-h3`](../minimax-h3/), [`ltx-2-5`](../ltx-2-5/), [`scail-2`](../scail-2/) |
| The stage ladder, denoise bands, cross-family handoffs, mixing models | [`image-production-workflows`](../image-production-workflows/) |
| Dataset, captioning, hyperparameters, evaluating a run, publishing a LoRA | [`character-lora-training`](../character-lora-training/) |
| Volume layout, `extra_model_paths.yaml`, ComfyUI as a serverless endpoint | [`comfyui-on-runpod`](../comfyui-on-runpod/) |
| Provisioning a pod, GPU selection, `runpodctl`, pod lifecycle, cost guards | **RunPod's own skills** — see [The ecosystem beyond this suite](#the-ecosystem-beyond-this-suite) |
| Actually running a job on Comfy Cloud | **Comfy-Org's skills** — same section |

**One rule about disagreement.** This skill will occasionally be wrong before its siblings are. Where a ranking
here conflicts with the model skill it summarises, **the model skill wins.** The model skill is closer to the
evidence, and it is checked on its own freshness cadence. Report the conflict.

---

## Before you pick anything: three gates that can end the project

These come first because none of them is recoverable later, and this is the only skill that sees all of them at
once. Each is owned in full elsewhere; this section is the triage.

**1. What are you shipping?** The licences in this suite do not split into "free" and "paid". They split by
*what leaves the building*.

| You are selling… | The question the licence asks |
|---|---|
| **Pictures or clips** | May you *use* the model? Almost everything here clears this — including [`anima`](../anima/), whose non-commercial term explicitly excludes Outputs |
| **A pipeline, product or hosted API** | May you *ship or serve* the weights? One non-commercial rung stops the whole chain, whether it comes first or last. [`image-production-workflows`](../image-production-workflows/) owns the chain question |
| **Anything, above a revenue line** | [`krea-2`](../krea-2/) is free under $1M and [`ltx-2-5`](../ltx-2-5/) under $10M. LTX also bars competing with Lightricks' products **at any revenue** |
| **…and the licence you pick has an ecosystem cost** | FLUX.2's Apache-2.0 [klein] 4B carries 133 published LoRAs against 9B's 653. Clearing gate 1 can mean training your own `[official — Civitai API, 2026-08-23]` |
| **Adult work** | [`ltx-2-5`](../ltx-2-5/)'s AUP bars it outright; [`wan-2-2`](../wan-2-2/) has no acceptable-use clause at all |

**2. Is the person real?** Suppose a character resembles an identifiable living or deceased individual. **Then
Civitai will not host the LoRA — SFW or NSFW, with no consent exception.** The TAKE IT DOWN Act is in force too,
and it reaches AI-generated intimate imagery of real people. **The test is resemblance, not provenance.** A
character you invented is clean. A lookalike is the case the rule was written for. Settle this before you build
a dataset. The enforcement date, the penalty figure and the quoted policy text live in
[`character-lora-training`](../character-lora-training/), which owns them. Read them there rather than from this
summary.

This gate is where **adult work** gets decided, and it is the one place in this skill where the answer is not a
trade-off. Adult generation with invented adult characters is a first-class use of these models, and it is
covered as such throughout. Two lines are absolute, and they are not licence questions: **sexual content
depicting minors**, and **sexual imagery of real, identifiable people without their consent**. Neither is a
capability gap to work around. Everything else on this axis is craft, and the craft is below.

**3. Where are you?** [`minimax-h3`](../minimax-h3/)'s licence **excludes the US, EU, UK and South Korea**. It
is the only territory gate *inside* the suite, and it rules the model out entirely rather than just limiting it.
It is no longer the only one you can walk into. Tencent's Hunyuan Community Licence **excludes the EU, UK and
South Korea**, and it covers **HunyuanVideo-1.5** and **HunyuanImage 2.1 / 3.0**, none of which this suite
covers `[official — HunyuanVideo-1.5 card, read 2026-09-09]`. If you are in those territories,
strike both families before you compare anything.

Then comes the ordinary question: **what hardware do you have**, or are you renting? See *Hardware* below.

---

## The one rule that changes everything

**Eliminate on constraints, then rank on quality — in that order, and never the reverse.**

Constraints are binary, published, and fatal. Quality gaps between the surviving models are small, contested,
and usually fixable with a second pass. People pick a model from a comparison thread, build for a week, and then
discover a licence or dialect problem that was knowable on day one.

**The elimination ladder**, in order. Each rung removes models; only the last one ranks them:

| # | Rung | What it removes |
|---|---|---|
| 1 | **Licence** — what leaves the building | Shipping or serving the weights kills [`anima`](../anima/) (whose *outputs* stay free), [`ideogram-4`](../ideogram-4/)'s open weights and FLUX.2 [dev]/9B. Adult work kills [`ltx-2-5`](../ltx-2-5/) |
| 2 | **Territory** | [`minimax-h3`](../minimax-h3/) if you are in the US/EU/UK/KR. Outside the suite, **HunyuanVideo-1.5** and **HunyuanImage 2.1 / 3.0** if you are in the EU/UK/KR |
| 3 | **Hardware** | Under ~12 GB the field narrows to [`sdxl`](../sdxl/), [`anima`](../anima/), FLUX.2 [klein] 4B, quantised [`z-image`](../z-image/) and [`wan-2-2`](../wan-2-2/) 5B. See the table below |
| 4 | **Capability** — can it do the job at all | In-image typography → only [`ideogram-4`](../ideogram-4/). Tracked person-replacement → only [`scail-2`](../scail-2/). Cuts inside one generation → only [`ltx-2-5`](../ltx-2-5/). Native audio → only [`minimax-h3`](../minimax-h3/) and [`ltx-2-5`](../ltx-2-5/). **Adult work is a capability axis in its own right**; see below |
| 5 | **Dialect** — will it understand you | Booru tags into a prose-trained encoder land as noise, and sentences into CLIP hit a 77-token wall. This is an encoder-class fact, not a preference |
| 6 | **Quality** | *Now* rank. See below, and expect the gaps to be smaller than the threads suggest |

Rung 5 is the one people skip. [`sdxl`](../sdxl/) and [`anima`](../anima/) want tags. [`flux-2`](../flux-2/),
[`z-image`](../z-image/) and [`krea-2`](../krea-2/) want sentences. [`ideogram-4`](../ideogram-4/) wants a JSON
document. A model prompted in the wrong dialect looks like a bad model.

**Three traps on rungs 1 and 2, all outside the suite, all easy to pick up by accident.** They are here because
each one looks clean from where people meet it. **`nvidia/PiD`** is a latent-to-pixel decoder and 4× upscaler
that drops into FLUX, FLUX.2, SD3, SDXL and [Qwen-Image](../qwen-image/) graphs, and Comfy-Org ships a template for it. Its
licence is **NSCLv1, non-commercial**. A template is not a licence check. **Pony V7** (AuraFlow, not SDXL)
carries a custom licence that bars inference services, companies over $1M revenue and professional video, which
makes it a third revenue gate beside [`krea-2`](../krea-2/)'s $1M and [`ltx-2-5`](../ltx-2-5/)'s $10M.
**HunyuanImage 2.1 / 3.0** sit under the Tencent Community Licence: EU, UK and South Korea excluded, 100M-MAU
cap `[official — model cards and licence files, read 2026-09-09]`. One more rule from the same sweep: run the
ladder against the **specific checkpoint**, never against the vendor's reputation. Qwen-Image 3.0 went
hosted-only with no weights and no licence in the same four months that Ideogram 4, Krea 2, MiniMax H3 and
LTX-2.5 all released weights, and **Qwen-Image-2.1 (open weights 2026-09-20) arrived under a non-commercial
Qwen Research License** while every 20B Qwen-Image checkpoint stays Apache 2.0. "Historically open lab"
predicts nothing, and neither does the family name.

---

## Hardware — what actually fits

This is elimination rung 3. These are practitioner figures for a comfortable run at native resolution, not
vendor minimums. Several vendors publish no minimums at all. Per-model detail is in each skill's *Quantisation &
VRAM* section, and the full table with quant filenames is
[`references/model-rankings.md`](references/model-rankings.md) §7.

| Card | What runs |
|---|---|
| **6–8 GB** | [`sdxl`](../sdxl/) fp16 (its long-standing advantage), [`anima`](../anima/) `[flagged — re-verify]`, FLUX.2 [klein] 4B fp8, quantised [`z-image`](../z-image/) — reported running on an **RTX 2060** `[community — Royal_Carpenter_1338, 1084 pts]` — [`krea-2`](../krea-2/) Turbo fp8_scaled on an **8 GB 3070 Ti** `[community — niechta, 600 pts]`, [`wan-2-2`](../wan-2-2/) 5B |
| **12–16 GB** | [`krea-2`](../krea-2/) fp8_scaled, [`wan-2-2`](../wan-2-2/) 14B at Q4_K_M — but below ~12 GB prefer the purpose-built 5B over a deeply-quantised 14B |
| **16–24 GB** | [`z-image`](../z-image/) Base, FLUX.2 [dev] fp8, [`ideogram-4`](../ideogram-4/) nf4, [`wan-2-2`](../wan-2-2/) 14B, [`scail-2`](../scail-2/) fp8_scaled |
| **32 GB+ / rent** | [`ltx-2-5`](../ltx-2-5/) `[contested]`, FLUX.2 [dev] unquantised, and **most LoRA training** — [`comfyui-on-runpod`](../comfyui-on-runpod/) |

**The bands are conservative at the bottom.** Two well-received reports put models a tier below where this table
places them: Z-Image on a 2060, and Krea 2 Turbo on 8 GB. Read a row as *where the model is comfortable*, not as
where it becomes impossible. **Training moves the floor up**, which is why renting is the usual answer. One
exception is worth knowing even if you never make anime: [`anima`](../anima/) trains in roughly **6 GB at 768
px**. The real cost of learning to train is the three failed runs, and those runs are free there.

---

## The rankings — and what each one costs you

The verdicts are here. The evidence, the disagreements and the second-place cases are in
[`references/model-rankings.md`](references/model-rankings.md). Every ranking is a *summary of the model
skills*, which own their own claims.

**Photoreal faces and skin.** [`z-image`](../z-image/) → [`flux-2`](../flux-2/) [dev] → [`sdxl`](../sdxl/)
photoreal finetune → [`krea-2`](../krea-2/). The suite routes *here* on this axis from four directions, which is
what makes it settled rather than argued. **The cost:** Z-Image has one strong realism-leaning look, it fights
stylisation, and its ControlNet is Turbo-only. So the standard move is not "pick the winner". Instead, **compose
and control in [`sdxl`](../sdxl/) or [`krea-2`](../krea-2/), then finish the face in Z-Image at ~0.2 denoise**.
Realism is a pipeline position, not a model.

**Consistent characters without training anything.** [`qwen-image`](../qwen-image/) (Edit-2511: one to three
references, no adapter, and the dataset factory every image skill here routes through) → [`flux-2`](../flux-2/)
(multi-reference + PuLID, the only PuLID-class face adapter in the suite) → [`sdxl`](../sdxl/) (InstantID/HyperLoRA,
`[SEP]` routing for several characters) → [`krea-2`](../krea-2/) (Identity Edit). [`z-image`](../z-image/) has
no adapter shortcut at all. Qwen-Image-Edit leads because that is where the practice is and where the other
skills already send their anchors; FLUX.2 keeps the one thing an edit model cannot do, a face adapter. No
head-to-head likeness test between the two exists `[contested]`. **The cost:** these routes are faster to start
but weaker under pressure. A trained LoRA still wins on a character you will render a thousand times.

**Practice says the same.** A year-sorted r/StableDiffusion sweep on 2026-08-23 returned **no top post naming
PuLID or InstantID at all**. The work has moved to **edit models**, meaning Qwen-Image-Edit, Krea 2 Identity Edit
and Flux [klein] 9B, mixed freely in one job. The adapters still do things edit models cannot, but an adapter
route today is one you debug alone ([`references/model-rankings.md`](references/model-rankings.md) §3.2). **Two
traps fail silently in a stock graph.** Krea 2 Identity Edit is an *unofficial* community fine-tune; it needs
the `ComfyUI-Krea2Edit` node pack for its dual conditioning `[community — Enshitification, 611 pts]`. And
ComfyUI's Qwen edit node downscales every reference to 1 MP before the model sees it, which is where "Qwen Edit
is blurry" comes from; the bypass is the first thing [`qwen-image`](../qwen-image/) teaches. **The largest of
the three by ecosystem is the newest skill here.** The **Qwen-Image 20B family** (Qwen-Image, Edit 2509/2511, 2512,
Layered) is Apache-2.0, ComfyUI-native, and carries **1,637 Civitai LoRAs, 273 of them character-tagged**,
counted by full pagination with `nsfw=true` on 2026-09-09; the 1,162 / 192 figure this suite carried earlier
the same day excluded NSFW `[official — Civitai API, 2026-09-09]`. **Qwen-Image-2.1** (2026-09-20) takes up to
ten references and claims stronger identity, but it is research-only by licence and has no independent
likeness report; it does not move this ordering `[flagged — re-verify after 2026-09-28]`.

**Easiest to train a character LoRA on — and this splits three ways.** There is no single winner. Treating it as
one question is the mistake:

| "Easiest" meaning | Winner | Why |
|---|---|---|
| **Best likeness out of the run** | [`ideogram-4`](../ideogram-4/), then [`krea-2`](../krea-2/) sampled through Turbo | One named cross-model test, six bases, ~30 runs `[community — MesmerTools, 2026-07-14; single source]` |
| **Fastest loop — most attempts per day** | [`z-image`](../z-image/) (~90 min/run, good by step ~900), [`anima`](../anima/) (~6 GB, so failures are free) | Iteration count is what actually teaches you what a dataset is missing |
| **Most documented — least to invent** | [`sdxl`](../sdxl/) | Years of settled recipes, two trainers, separate Pony/Illustrious pools |

**Ecosystem size is a fourth axis, and it does not track the first three.** These counts are from Civitai on
2026-08-23, with the fast-moving rows re-run on 2026-09-09. They are **re-measurable**, which matters more than
the snapshot: run [`scripts/civitai_census.py`](scripts/civitai_census.py) rather than trusting these figures
once they are a few weeks old. [`z-image`](../z-image/) Turbo, [`krea-2`](../krea-2/), [`anima`](../anima/) and
the SDXL family each carry **2,000+** published LoRAs. [`qwen-image`](../qwen-image/) sits next at **1,637**
(273 character-tagged; counted with `nsfw=true`, which the earlier 1,162 was not). FLUX.2 [klein] 9B has ~670, and **[klein] 4B just 133**. [`ideogram-4`](../ideogram-4/) has 36. Two traps
fall out of this. First, **Z-Image's pool is on Turbo (2,191+) not Base (671)**, and Turbo is the variant its
own training doctrine trains on. Second, **picking [klein] 4B for its Apache-2.0 licence costs you most of the
FLUX.2 LoRA pool**. The full census, method and caveats are in
[`references/model-rankings.md`](references/model-rankings.md) §4.5.

**The cost of the podium.** Ideogram 4's win rests on one author and one dataset. Its **weights are
non-commercial**, so a LoRA trained there may be unshippable for the reason gate 1 names. Its ecosystem is **36
LoRAs on Civitai, exactly one of them tagged as a character LoRA** `[official — Civitai API, counted 2026-09-09]`.
The model may well train beautifully. Nobody has done it in public. Krea 2 trains slowly and pays off only if you
remember to *sample* on Turbo. FLUX.2 [dev] is the one to skip: it needed ~90 GB of system RAM to quantise, and
that test never finished a usable LoRA on it. On video, [`wan-2-2`](../wan-2-2/) is the mature ecosystem (two
LoRAs, one per expert, from one dataset). [`minimax-h3`](../minimax-h3/) has stopped being the young one: four
trainers now support it (ai-toolkit, musubi-tuner, diffusion-pipe, SimpleTuner) plus a hosted fal.ai option, its
Civitai pool went from 22 to **86 LoRAs, 29 character-tagged**, in seventeen days, and one lab has shown a
stills-only H3 character LoRA holding identity in motion — 0.200 median face distance against 0.920 for the
no-LoRA control `[live-use — media lab, Ciara h3-v1, 2026-09]`. The doctrine is weeks old and the model skill
owns it; rank 32 collapses under ai-toolkit, so r16 is the ceiling there. [`scail-2`](../scail-2/) has **no
training path at all**.

**Adult and NSFW work.** This is a dominant use of open-weights models, and it is treated here as a capability
axis, not just a licence one. The model that wins it is not the model that wins realism, and the difficulty is
routinely misdiagnosed.

| | The answer | The catch |
|---|---|---|
| **Image** | [`krea-2`](../krea-2/) — the busiest adult checkpoint ecosystem on Civitai (`LUSTIFY!`, `FinePorn v3 TURBO`, Moody Krea 2 Mix), typically Euler or ER SDE, 10 steps, guidance 1.0 | The checkpoint does the work, not your settings. Krea 2's safety tuning also drives its muted-expression tax — the same tuning `krea2filterbypass`-class LoRAs exist to undo |
| **Anime / illustration** | [`sdxl`](../sdxl/)'s Pony V6 XL, Illustrious and NoobAI — still the deepest pool — with [`anima`](../anima/) rising fast and running where nothing else will | Anima's weights are non-commercial; Pony V6 XL/Illustrious are not. **Pony V7 is AuraFlow, not SDXL**, and its licence bars inference services and >$1M companies (rung 1) |
| **Video** | **[`minimax-h3`](../minimax-h3/), decisively** — *"far above LTX and Wan"*, *"the most powerful open source model"* `[community — AidenAizawa, Revolutionary-Bar766; convergent]` | **Its licence excludes the US, EU, UK and South Korea.** The capability leader is one that many readers may not lawfully use — gate 3 is not a formality here |
| **Video, licence-clean** | [`wan-2-2`](../wan-2-2/) — Apache-2.0, **no acceptable-use clause at all**, the only unencumbered adult path in the suite | Weaker prompt adherence; anatomy needs an NSFW-merged checkpoint (below) |
| **Ruled out** | [`ltx-2-5`](../ltx-2-5/) — its AUP bars explicit content universally, local weights included. Adult LTX work happens on **2.3**, which is practice, not permission | — |

**One boundary on this table: it ranks adult *scene* generation, not training bases.** Suppose the deliverable
is a character LoRA used in adult work. Then the photoreal ranking above wins the base choice. Train on
[`z-image`](../z-image/), and let the checkpoint ecosystem re-enter at scene time. The reconciliation is in
[`references/adult-work.md`](references/adult-work.md) §2.

**The one reframe that saves the most time: the limit is training data, not refusal.** Open-weights models do
not refuse. They render poor anatomy because the base saw little of it. This is why swapping in an abliterated
("heretic") text encoder does nothing: refusal lives in output layers a text encoder never uses. Change the base
model or the checkpoint instead. Craft, captioning and anatomy failure modes are in
[`character-lora-training`](../character-lora-training/) `references/nsfw-training.md`.

**A diagnostic follows from this, and it generalises well beyond adult work: when seed is the only variable that
moves the result, stop rolling and change the checkpoint.** One 27-generation single-variable study looked at
Wan I2V anatomy collapse. It found that prompt, steps, LoRA weight, clip length, shift and source image all made
no difference, and only 2 of 28 seeds were usable. Swapping to an NSFW-merged checkpoint fixed it in one run
`[community — RedMimicStudios]`. The numbers and the measurable QC proxy are in
[`references/adult-work.md`](references/adult-work.md) §3.

**Structural control (pose, depth, canny, regional).** [`sdxl`](../sdxl/) wins, and it is not close: union
ControlNet, IP-Adapter and regional prompting are all mature there. Then come
[`z-image`](../z-image/)/[`flux-2`](../flux-2/) (Fun Union, custom nodes, Turbo-only on Z-Image), then
[`anima`](../anima/) (LLLite: lineart/depth/scribble, **no pose, no canny**) and [`krea-2`](../krea-2/) (two
community ControlNet-LoRAs, **depth and OpenPose**, no canny or union — the pose one landed 2026-08-04, which
lifts it level with Anima on the axis people most often need), then [`ideogram-4`](../ideogram-4/) (`bbox` layout
only). **The cost:** SDXL's control comes with a 77-token CLIP window and no in-image text. That is why it so
often acts as the *front end* of a chain rather than the whole of it.

**And a correction that shows this skill's own failure mode.** The suite rates [`ideogram-4`](../ideogram-4/)
weak on characters because it has no identity adapter, no edit variant and one character LoRA. **Those clauses
are true, and the conclusion was wrong.** A published workflow gets identity from it with no adapter and no
training. It asks for the character *twice in one canvas* and crops the result `[community — reality_comes, 402
pts]`. **"No adapter exists" describes tooling, not capability**
([`references/model-rankings.md`](references/model-rankings.md) §3.1).

**Everything else, one line each.** Typography goes to [`ideogram-4`](../ideogram-4/) for layout and dense
lettering; [`qwen-image`](../qwen-image/) is the second, and the first for bilingual Chinese/English rendering and
for editing text already in an image without losing its font.
Anime goes to [`anima`](../anima/), or to [`sdxl`](../sdxl/)'s Illustrious/NoobAI/Pony V6 XL finetunes when you
must ship the weights (Pony V7 is AuraFlow and is not an SDXL checkpoint). Widest aesthetic range:
[`krea-2`](../krea-2/). Cleanest licence: [`z-image`](../z-image/), tied with [`qwen-image`](../qwen-image/)'s **20B family** (Apache-2.0 on code and
weights, open through 2512 and Edit-2511; 2.0 and 3.0 are hosted-only; **2.1 is research-only** and fails the
first rung). Instruction-based editing, multi-image
fusion and layered RGBA output: [`qwen-image`](../qwen-image/) — Edit-2511 for the edits, Layered for RGBA,
which is adopted but has no settings-level craft yet; for research work, 2.1 does all three in one
checkpoint with native transparency, ten references and mask-guided local edits. Native 4K with region-controlled
edits: **SenseNova U1.5**, ComfyUI core since 2026-09-01, not yet covered, and its licence is verify-gated.

**Video, by what you are actually doing.** Animating a still is [`wan-2-2`](../wan-2-2/)'s job. Its I2V is far
stronger than its T2V, so lock the still with an image model first. For sound in the same pass, use
[`minimax-h3`](../minimax-h3/) or [`ltx-2-5`](../ltx-2-5/); which one is a licence question, not a quality one.
Several cuts in one generation: [`ltx-2-5`](../ltx-2-5/), alone. Replacing a person in footage is now a **task
split**, not a single answer: [`scail-2`](../scail-2/) for body motion, non-human subjects and multi-character
scenes, tracked frame-for-frame; Wan **Animate 2** (in [`wan-2-2`](../wan-2-2/), native ComfyUI since
2026-08-08) for close-up faces and lip sync `[community — Wensleydale on X, dreamerland.ai; two sources]`. **Two named gaps.** Nothing in the suite does a freeform camera path. And nothing *edits* a clip —
relight, restyle, insert a subject, remove an object. The open model that does is ByteDance's **Bernini-R**
(Apache-2.0, native ComfyUI since 2026-06-14), **not yet covered**; SCAIL-2 tracks a person, Bernini re-renders a
scene. A fourth open video model, Tencent's **HunyuanVideo-1.5** (8.3B, ~14 GB VRAM, native ComfyUI, musubi-tuner
LoRAs), is also uncovered and carries the EU/UK/KR exclusion from gate 3.

---

## Pick your job, get your stack

This is the routing table. "Install" is the set of skills that job needs. Run the command in [Installing what
you need](#installing-what-you-need). Full step-by-step routes, with what to read at each step, are in
[`references/playbooks.md`](references/playbooks.md).

| The goal | Route | Install |
|---|---|---|
| **Realistic photos of a character I invented, on rented GPUs** | Playbook A — lock anchor → dataset → train → evaluate → deploy → production ladder | `generative-media-atlas` `z-image` `qwen-image` `character-lora-training` `comfyui-on-runpod` `image-production-workflows` + RunPod's |
| **An anime character, on my own card** | Playbook B — the 6 GB loop | `anima` `character-lora-training` `image-production-workflows` |
| **A design or marketing image with real text in it** | Playbook C — typography plate → composite | `ideogram-4` `image-production-workflows` |
| **Edit an image by instruction, or re-shoot a character in new scenes without training** | Compose, then Edit-2511 one instruction per pass, re-anchored on the original each time; pixels out per the handoff rules | `qwen-image` `image-production-workflows` |
| **Turn a still into a shot** | Playbook D — the image-to-video handoff, and the audio licence fork | an image skill + `wan-2-2` (or `ltx-2-5` / `minimax-h3`) |
| **Put my character into footage I already have** | Playbook E — edit frame 0, then track | `krea-2` `scail-2` `character-lora-training` |
| **Run all this as an API** | Playbook F — API-format workflows on serverless | `comfyui-on-runpod` + RunPod's `runpod`, `flash` |
| **Adult work, image or video** | Playbook G — checkpoint first, then licence, then anatomy | `krea-2` or `sdxl` (image) / `wan-2-2` or `minimax-h3` (video) + `character-lora-training` |
| **What's deployed on my RunPod account, and what is it costing me** | `comfyui-on-runpod` *Cost guards that actually work* (burn check, the clock-outside-the-pod guard, agent-free teardown) → RunPod's `runpod-mcp`/`runpodctl` to act on it | `comfyui-on-runpod` + RunPod's |
| **I just want to know which model** | The elimination ladder above, then the rankings | this skill alone |

---

## Installing what you need

**The skills CLI has no dependency mechanism.** A skill cannot declare that it needs another, and nothing is
pulled in transitively. That is why this skill carries install commands instead of relying on its links. **The
relative links above resolve only when the sibling is also installed, and dangle otherwise.** A dangling link is
not a bug report. It is the install command you have not run yet.

```bash
# The atlas alone — enough to choose a model and plan a route
npx skills add ryannel/skills --skill generative-media-atlas

# A playbook's stack, in one command
npx skills add ryannel/skills --skill z-image --skill qwen-image --skill character-lora-training --skill comfyui-on-runpod --skill image-production-workflows

# Browse the catalogue first
npx skills add ryannel/skills --list
```

Add `-g` for global rather than project scope, `-a claude-code` to target one agent, and `-y` to skip prompts.
`npx skills update` refreshes installed skills. It is worth running against this suite, whose subjects move
weekly. To read a skill once without installing it, use `npx skills use ryannel/skills@z-image`.

**Ask before installing globally or into a repo you do not own.** Installing writes files into the user's agent
directories. That is a visible change to their machine, not a lookup.

Scopes, agent targeting, symlink-vs-copy, private repos and troubleshooting are covered in
[`references/installing-skills.md`](references/installing-skills.md).

---

## The ecosystem beyond this suite

This suite deliberately does not restate what its vendors already publish well. Four canonical sources matter,
and knowing what each one is *not* is as useful as knowing what it is.

| Source | Install | What it owns | What it is **not** |
|---|---|---|---|
| **RunPod** — `runpod/runpod-plugins-official` | `npx skills add runpod/runpod-plugins-official` | 8 skills — `runpod` (router), `runpod-usage`, `runpodctl`, `runpod-mcp`, `flash`, `companion-clis`, `runpod-migrate`, `runpod-templates` — plus 24 golden paths including `02-comfyui-pod`, `07-network-volume-handoff`, `20-model-caching-endpoint`, `21-storage-tiers`, `25-bake-vs-mount` | Not ComfyUI-aware. Where models must sit so a fresh instance finds them is [`comfyui-on-runpod`](../comfyui-on-runpod/)'s job |
| **Comfy-Org** — `Comfy-Org/comfy-skills` | `/plugin marketplace add Comfy-Org/comfy-skills` then `/plugin install comfy-cloud@comfy-skills` | 12 skills wrapping the **Comfy Cloud MCP** — `comfy-generate-image`, `comfy-generate-video`, `comfy-search-models`, `comfy-search-nodes`, `comfy-search-templates`, `comfy-upscale-image`, and others | **Mostly command wrappers, not craft.** They execute a job on Comfy Cloud; they carry no per-model settings, prompt dialects or licence analysis. One exception: `technique-combine-people` is a real face-consistency compositing recipe on hosted partner nodes. They pair with this suite rather than replacing it |
| **Hugging Face** — `huggingface/skills` | `hf skills add <name>` | `hf-cli` (fetching weights), `hf-mem` (estimating VRAM from safetensors/GGUF), and `huggingface-lora-space-builder`, which publishes a trained LoRA (Qwen-Image, LTX, Wan, FLUX, SDXL) as a Gradio ZeroGPU demo | Still no diffusion-*training* skill; its trainers are LLM and vision-classification shaped |
| **Black Forest Labs** — `black-forest-labs/skills` | `npx skills add black-forest-labs/skills` (standard `skills/` layout; the shorthand is inferred, not run here — see [`references/ecosystem-map.md`](references/ecosystem-map.md) §4) | 10 skills: `bfl-api`, `flux-image-best-practices`, and eight `flux-3-*` skills for the **hosted** FLUX 3 API (image, 20 s video with native audio, keyframes, v2v continuation, prompt doctor) | Hosted API only. **FLUX 3 has no open weights**; "FLUX 3 Dev" is on BFL's roadmap with no date, licence or size. Open-weight FLUX.2 craft stays in [`flux-2`](../flux-2/) |

**Three things changed recently enough to catch people out.** RunPod ships **8** skills, not the 6 or 7 widely
cited: `runpod-migrate` and then `runpod-templates` were added `[official — repo tree, read 2026-09-09]`. Black
Forest Labs became the first model vendor in this suite's field to publish agent skills, which retires the claim
this section used to make that none did. And `metadata.internal: true` hides a skill from discovery unless
`INSTALL_INTERNAL_SKILLS=1` is set, which is how a repo keeps authoring machinery out of its published listing.

No agent skills from Stability, Alibaba/Tongyi, Lightricks, MiniMax, Z.ai or Civitai were findable as of
**2026-09-09** (a negative search result, flagged for re-verification in the ecosystem map). That is still the
gap this suite exists to fill. Full inventories, the BFL repo, the open models with no skill yet, and how to judge a third-party skill
before trusting it, are in [`references/ecosystem-map.md`](references/ecosystem-map.md).

### The compute stack is pluggable — RunPod is the current inventory, not the architecture

Every route in this suite runs on a three-layer stack, and only the bottom layer is vendor-shaped:

1. **Model skills** — what to run and how to prompt, train and judge it. Platform-agnostic.
2. **Execution surface** — ComfyUI or diffusers. Every model skill carries **both** as first-class surfaces, so
   "not ComfyUI" is already covered: the diffusers path needs no UI shell at all.
3. **Compute platform** — where the GPU lives. Today the suite ships one platform skill,
   [`comfyui-on-runpod`](../comfyui-on-runpod/). Its scoping, *the layer between the vendor's platform skills
   and the model skills*, is the plug socket, not a commitment to RunPod.

**The platform-skill contract.** A skill for any other platform (Modal, Vast.ai, fal, Lambda, a homelab) fills
the same slot. It owns the same four things `comfyui-on-runpod` proved out: where models live so a fresh
instance finds them out of the box; cost guards and the session-ending burn check; deploy-and-smoke-test for
both interactive and API execution; and **routing to the vendor's own skills for provisioning** rather than
restating them. Everything above the slot (model choice, prompting, LoRA craft, pipeline design) transfers
unchanged.

**Pick one platform, install its one skill.** The skills CLI resolves no dependencies, so pluggable means
something concrete: running on RunPod means `npx skills add ryannel/skills --skill comfyui-on-runpod` plus
RunPod's own suite, per the table above. A second platform's skill, when it exists, slots into this sentence and
nothing else in the suite changes.

**Why there is exactly one platform skill today.** `comfyui-on-runpod` was generalised from a real production
deployment, and that is the bar. A platform skill written from vendor docs alone would be the paraphrase this
suite refuses to be. So new platform skills get authored the first time real work runs on that platform, not
speculatively. If you run this suite somewhere else and want the skill to exist, that experience is the missing
ingredient.

---

## Failure modes & QC

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| A week of work, then the licence blocks delivery | Ranked on quality before eliminating on constraints — rung 6 before rung 1 | Run the ladder in order; settle gate 1 against the *deliverable*, not the picture |
| Links in this skill 404 | Siblings not installed; the CLI resolves no dependencies | Install the playbook's stack — [Installing what you need](#installing-what-you-need) |
| The trained LoRA cannot be published anywhere | Dataset was a real person — resemblance, not provenance, is the test | Synthetic character; [`character-lora-training`](../character-lora-training/) gate |
| The model ignores half the prompt | Wrong dialect for its encoder class — tags into an LLM encoder, or sentences past CLIP's 77 tokens | Rung 5; the model skill's `prompting-guide.md` |
| Great model, wrong country | [`minimax-h3`](../minimax-h3/)'s territory exclusion is a licence term, not a geoblock you can ignore | [`wan-2-2`](../wan-2-2/) |
| GPU bill with nothing to show | Rented before the graph ran end to end once — or trusted a CLI timer. `runpodctl` 2.12.0 (2026-08-27) removed `--stop-after`/`--terminate-after`, and its own PR #330 says they were forwarded but **never enforced** | Smoke-test cheap. Put the clock **outside the pod**: a scheduler or shell `trap` that calls `runpodctl pod remove "$POD_ID"` unconditionally, plus an on-pod watchdog. Then the burn check — [`comfyui-on-runpod`](../comfyui-on-runpod/) and RunPod's `runpod-usage` |
| Works locally, breaks on serverless | The dual mount root — `/workspace` vs `/runpod-volume` | [`comfyui-on-runpod`](../comfyui-on-runpod/) |
| Chose the "best realism" model and the render still looks AI | Realism is a pipeline position, not a model — one pass rarely gets there | The production ladder in [`image-production-workflows`](../image-production-workflows/) |
| Swapped in an abliterated encoder to "unlock" anatomy; nothing improved and adherence got worse | Refusal lives in LLM output layers a **text encoder never uses** — there is no refusal path to remove, only perturbed conditioning | Change the base or checkpoint. Use abliterated models only for a refusing *prompt-expander* stage |
| Trained a character LoRA and the anatomy is still wrong | **Two different jobs at different scales** were confused — teaching *who someone is* (15–30 images) vs teaching anatomy the base lacks (1,500+, rank 128+) | Pick a base that already has the coverage, or stack a capability LoRA under the character one |
| Endless rerolling on a shot the model keeps breaking | Seed is the only variable moving the result — the base cannot hold that composition | **Stop rolling, change the checkpoint.** See the adult-work section |
| A model is written off as incapable, and someone posts a workflow doing it anyway | **"No adapter exists" was read as "the model cannot."** Ideogram 4's character reference is the worked example — a prompting construction, not tooling | Ask what the *capability* is before accepting a tooling-shaped verdict; check for a published workflow |
| This skill and a model skill disagree | The atlas summarises; the model skill is the source | **The model skill wins.** Report the conflict |

---

## Pre-flight checklist

1. Gate 1 settled — is the deliverable a picture, a pipeline, or the model itself?
2. Gate 2 settled — is the character synthetic, or does it resemble a real person?
3. Gate 3 settled — does any candidate's licence exclude your territory or your revenue band?
4. Hardware known: your card's VRAM, or the decision to rent, made before model choice?
5. Ladder run **in order** — licence, territory, hardware, capability, dialect, *then* quality?
6. Capability checked against what only one model can do — typography, tracked replacement, multishot, native
   audio?
7. Prompt dialect matched to the encoder class you picked?
8. Route written down as a sequence of skills, with the handoffs named — and every one installed?
9. If training: base chosen as the model you will *render* on, not the one that trains best?
10. If renting: a terminate call scheduled from **outside** the pod (no CLI flag does this any more), and the
    workflow proven cheaply first?
11. If the model you want is one of the "not yet covered" rows in the suite map: licence read from its own card,
    not inferred from the vendor?

---

## The suite map

Every published skill, and the question it answers. This table is the suite keyed by *question*.
[`image-production-workflows`](../image-production-workflows/) keeps the suite map keyed by *pipeline role*
(who composes, who refines, who finishes), and the two are meant to be read together.

| Skill | The question it answers |
|---|---|
| [`z-image`](../z-image/) | "Why do the faces still look plastic?" — the suite's realism owner and standard face-pass finisher; Apache-2.0 throughout |
| [`flux-2`](../flux-2/) | "How do I keep this character without training anything?" — multi-reference identity, PuLID; [klein] 4B is the Apache-2.0 escape hatch |
| [`sdxl`](../sdxl/) | "How do I control the pose/composition exactly?" — the deepest control, adapter and LoRA ecosystem, on 6–8 GB |
| [`krea-2`](../krea-2/) | "How do I get a look that isn't the AI look?" — widest aesthetic range, style references, Identity Edit |
| [`qwen-image`](../qwen-image/) | "How do I edit this image by instruction, or re-shoot this character without training anything?" — Edit-2511 identity engine and dataset factory, multi-image fusion, bilingual text; Apache-2.0 through 2512/2511, 2.0/3.0 hosted-only; **2.1** (2026-09-20) unified T2I + edit with native RGBA and ten references, **non-commercial licence** |
| [`ideogram-4`](../ideogram-4/) | "How do I put real text in the image?" — typography, layout, JSON captions. Open weights are non-commercial |
| [`anima`](../anima/) | "How do I make anime that understands booru tags?" — 2B, ~6 GB, outputs commercially free, weights not |
| [`wan-2-2`](../wan-2-2/) | "How do I make this still move?" — the strongest unencumbered I2V path, plus the camera and motion rigs |
| [`minimax-h3`](../minimax-h3/) | "How do I get video with sound in one pass?" — **and may I, where I live?** |
| [`ltx-2-5`](../ltx-2-5/) | "How do I get several cuts out of one generation?" — plus the suite's generative video upscaler |
| [`scail-2`](../scail-2/) | "How do I put a different person into this footage?" — tracked, not re-generated |
| [`character-lora-training`](../character-lora-training/) | "How do I make an identity that survives prompts it never saw — and may I publish it?" |
| [`image-production-workflows`](../image-production-workflows/) | "How do I get from a render to something shippable?" — the ladder, mixing models |
| [`comfyui-on-runpod`](../comfyui-on-runpod/) | "Why can't ComfyUI find my model?" — volumes, mount roots, serverless |
| **this skill** | "Which of the above, and in what order?" |

**Strong open models with no skill here yet.** Plain bold, not a link, because there is nothing to link to. Each
row says where to go instead. The status is as of **2026-09-09**; if a row below has become a link in a later
version of this table, the skill exists. The Qwen-Image family left this table the same day, when
[`qwen-image`](../qwen-image/) shipped.

| Model | The question it would answer | Not yet covered — route to |
|---|---|---|
| **Bernini-R** (ByteDance, Apache-2.0, Wan-derived renderer) | "How do I relight, restyle, insert into or remove from a clip I already have?" — six modes, t2i through rv2v; native ComfyUI since 2026-06-14; 14B is H100-class, 1.3B is the small one | `github.com/bytedance/Bernini`. Boundary against the suite: [`scail-2`](../scail-2/) tracks a person, Bernini re-renders a scene |
| **SenseNova U1.5-8B-MoT** (SenseTime) | "How do I get native 4K, or edit a region by mask or box?" — ComfyUI core since 2026-09-01; the NoobAI team published its LoRA trainer | `github.com/OpenSenseNova/SenseNova-U1`. **Licence and parameter count are unverified here**: the HF card is gated and "8B" sits against ~18B total (flagged in the two-bar section below) |
| **HunyuanVideo-1.5** (Tencent, 8.3B) | "Which open video model fits ~14 GB?" — native ComfyUI since 2025-11-24, musubi-tuner LoRAs | `huggingface.co/tencent/HunyuanVideo-1.5`. **Gate 3 applies: the Tencent Community Licence excludes the EU, UK and South Korea** |

Held back rather than uncovered: **HiDream-O1-Image** (MIT) and **Mage-Flow** (Microsoft, MIT) both have core
ComfyUI support and trainers, and 47 and 0 Civitai LoRAs respectively. Strong on paper, absent in practice;
re-check date and thresholds are in [`references/ecosystem-map.md`](references/ecosystem-map.md) §6.

---

## How to read the claims in this skill — two bars, by claim type

This skill holds two kinds of claim to two different standards, because they fail in two different ways. There
is also a third kind, peculiar to a router, worth naming on its own.

**Hard facts — must be exact or it breaks.** This covers the names and install commands of every skill here and
in the four external sources. It also covers RunPod's eight-skill inventory and its golden-path filenames;
Comfy-Org's twelve; Black Forest Labs' ten; the `skills` CLI's flags, scopes and the absence of a dependency
mechanism; `metadata.internal`; the shape of each licence gate (territory, revenue line, weights-vs-outputs
split), including the three traps outside the suite; the licence and ComfyUI status of each "not yet covered"
model; the removal of `runpodctl`'s timer flags; and the fact that a Civitai likeness ban and live TAKE IT DOWN
enforcement exist at all. The dates and figures behind those two are
[`character-lora-training`](../character-lora-training/)'s to state, not this skill's. **The source of truth is
official.** That means the repository trees read directly, the CLI's README and release notes, Comfy-Org's
`workflow_templates` index (not `blog.comfy.org`, which is now behind a subscription wall and returns nothing to
a checker), and the model cards and licence texts via the model skills. A wrong install command fails loudly. A
misread licence gate is a legal problem. Vendor skill repositories add and rename skills without notice, so
**re-verify before relying on them, regardless of who said it.**

**Craft — what actually makes a good choice.** This covers the elimination ladder and its ordering; every
ranking and the trade attached to it; the hardware bands; the playbook routes and where their handoffs fail; and
the three-way split in what "easiest to train on" means. **The authoritative source here is the community and
the practitioners the sibling skills cite.** Two pieces of first-hand evidence were gathered for this skill as
well. The first is one named cross-model test for LoRA trainability, MesmerTools' six-base comparison of
2026-07-14. It is **a single author on a single dataset**, and it is marked as such wherever it is used. The
second is a **census of the Civitai API taken 2026-08-23 and re-run for the fast-moving rows on 2026-09-09**, a
direct measurement rather than a report. It is bounded by being one host, one that bans real-person likeness
outright and so undercounts a whole category. There are also **primary Reddit sweeps from 2026-08-23** across
r/StableDiffusion and r/unstable_diffusion. These were top-sorted over the past year and month, and read for
the realism, identity and adult-work axes specifically. The adult stack in particular is reported from where
that work is actually discussed, rather than inferred. Where a community claim here carries a point count, that
is the sweep. A **landscape sweep on 2026-09-09** read Comfy-Org's template index, the Hugging Face API and
Civitai for every open launch since May; it is the source for the "not yet covered" rows and the three ladder
traps, and it did *not* measure Reddit or Banodoco, so community sentiment on those models is unmeasured, not
low. Where a claim carries `[live-use — media lab, …]`, it is one lab's measured run in this suite's own
testbed: firmer than a forum report, weaker than consensus, and always dated. Rankings are stated with
confidence, and they mean "this is where the practice lands", not "this is measured".

**Derived claims — the third bar.** Almost every comparative verdict here is *synthesised from the sibling
skills*. Those are themselves two-bar documents with their own provenance and their own freshness cadence, and
this skill adds no new evidence for them. Two things follow: this skill inherits their confidence rather than
exceeding it, and **where it disagrees with a model skill, the model skill is right and this one is stale.**

**Contested / unresolved points:**

- The character-LoRA trainability podium rests on one author's test `[community — MesmerTools; single source]`.
- The no-training identity ordering puts Qwen-Image-Edit ahead of FLUX.2 on practice and ecosystem, not on a
  measured likeness test; none has been published `[contested]`.
- [`ltx-2-5`](../ltx-2-5/)'s VRAM floor is 32 GB in documentation against 16 and 12 in the same vendor's
  marketing, published the same week `[contested]`.
- [`anima`](../anima/)'s 8 GB inference floor is one report on one AMD card `[flagged — re-verify]`.
- Whether Ideogram 4's LoRA win survives a second independent test is unknown. Nobody has published a character
  recipe for it; the one style recipe that exists is one author's `[flagged — re-verify]`.
- The SCAIL-2 / Wan Animate task split rests on one same-scene comparison and one third-party guide, and both
  predate **Animate 2** `[flagged — re-verify]`.
- SenseNova U1.5's licence and parameter count are secondhand until its gated card is read `[flagged — re-verify]`.
- Whether `runpodctl`'s timer flags come back (runpodctl#331, blocked on RunPod#5718) is
  [`comfyui-on-runpod`](../comfyui-on-runpod/)'s watch, not this skill's; until they do, no CLI flag is a cost guard.

**Facts dated 2026-09-09** (Qwen-Image-2.1 rows 2026-09-20); the H3, Ideogram 4, Wan 2.2, FLUX.2 [klein] 9B and Qwen-Image Civitai rows re-counted
the same day (Qwen-Image by full pagination with `nsfw=true`), the rest on 2026-08-23. The fastest-moving parts are the external vendors' skill inventories (RunPod went
6→7→8 across this suite's last three passes, and BFL appeared from nothing), the "not yet covered" table, and
the `skills` CLI's flags. The Civitai counts move fast too, since Anima and Krea 2 add LoRAs weekly and H3
quadrupled in seventeen days. So does any ranking whose model shipped in the last quarter — which today is
most of them.

**What the sweeps changed.** Realism **held**: Z-Image owns that conversation, convergent with the suite's
routing. Identity did not hold. The capability ordering survives, but the practice has moved to edit models, and
the "Ideogram is weak on characters" verdict was **wrong**. On adult work, the Civitai census **inverts** the
ordering in this suite's own existing table, most likely because that metric reads preview images and so
undercounts video. All three are corrected above and filed as findings against the skills that own them. The
2026-09-09 landscape sweep's largest finding was an **omission, not a launch**: Qwen-Image had been named
across twelve skills as a peer the reader was assumed to have, with no skill behind it. That skill,
[`qwen-image`](../qwen-image/), shipped the same day, and the identity, typography, control and licence rankings
above were re-cut around it.

---

## Reference files

| File | When to reach for it |
|---|---|
| [`references/model-rankings.md`](references/model-rankings.md) | You have a ranking above and need the evidence behind it, the second-place case, or the axis-by-axis matrix — realism, identity, LoRA trainability, control, typography, anime, licence, video |
| [`references/playbooks.md`](references/playbooks.md) | You have picked a goal and want the ordered route: which skill at which step, what to read in it, where the handoffs silently fail, and what to check before paying for the next stage |
| [`references/installing-skills.md`](references/installing-skills.md) | You are getting skills onto a machine: scopes, agent targeting, bundles per playbook, updating, private repos, `metadata.internal`, and why nothing installs transitively |
| [`references/adult-work.md`](references/adult-work.md) | You are doing adult work and need the model choice settled: the Civitai explicit-share census and why it undercounts video, the image and video stacks people actually run, MiniMax H3's prompting rules, the anatomy-collapse study and its checkpoint-swap tell, and two open gaps |
| [`references/ecosystem-map.md`](references/ecosystem-map.md) | You are deciding whether an external skill covers something this suite does not — the full RunPod, Comfy-Org, Hugging Face and Black Forest Labs inventories, the open models with no skill yet (with re-check dates), the hosted-only frontier, what was checked and found unreleased, and how to judge a third-party skill before trusting it |
| [`scripts/civitai_census.py`](scripts/civitai_census.py) | **An ecosystem or adult-share number here looks stale, or you want one for a base this skill doesn't list.** Re-measures both from the Civitai API — run it rather than trusting the snapshot, and read its header first: it encodes the bitmask semantics and the dead `nsfw` field that make hand-rolled versions silently wrong |
