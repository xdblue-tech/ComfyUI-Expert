# Playbooks — ordered routes from a goal to a finished file

SKILL.md routes a goal to a set of skills. This file is the route *through* them. It tells you which
skill to use at each step, what to read in that skill, where the handoffs fail silently, and what to
check before you pay for the next stage.

**This file owns sequence and handoffs only.** The settings for every step belong to the skill that
step names. If a playbook and a skill disagree on a number, trust the skill.

## Contents

1. [How to run a playbook](#1-how-to-run-a-playbook)
2. [A — a realistic character you invented, in ComfyUI on RunPod](#2-a--a-realistic-character-you-invented-in-comfyui-on-runpod)
3. [B — an anime character, on your own card](#3-b--an-anime-character-on-your-own-card)
4. [C — a design image with real text in it](#4-c--a-design-image-with-real-text-in-it)
5. [D — turn a still into a shot](#5-d--turn-a-still-into-a-shot)
6. [E — put your character into footage you already have](#6-e--put-your-character-into-footage-you-already-have)
7. [F — run the whole thing as an API](#7-f--run-the-whole-thing-as-an-api)
8. [G — adult work](#8-g--adult-work)
9. [The joins that fail silently](#9-the-joins-that-fail-silently)

---

## 1. How to run a playbook

**Install first, then work.** Each playbook opens with its stack. Skills do not install their
siblings automatically — see [`installing-skills.md`](installing-skills.md). If a step points at a
skill you have not installed, the link will be dead.

**Three rules that apply to all of them:**

1. **Settle the gates before step 1.** Licence, likeness and territory are SKILL.md's three gates.
   Each one is cheap to check now, and impossible to fix later.
2. **Prove the route cheaply before you scale it.** Generate one image at low resolution, one
   3-second clip, or one 200-step training run. Every playbook below marks a checkpoint **STOP** for
   this reason.
3. **Save the intermediate output of every stage.** When a final output is bad, you can only trace
   the failure back to the stage that caused it if that stage left evidence behind.

---

## 2. A — a realistic character you invented, in ComfyUI on RunPod

**The goal:** a synthetic person you can render repeatedly, photoreal, on rented GPUs. The workflows
should open on a fresh instance without missing-model dialogs.

**Stack:**

```bash
npx skills add ryannel/skills --skill generative-media-atlas --skill z-image --skill character-lora-training --skill comfyui-on-runpod --skill image-production-workflows
npx skills add runpod/runpod-plugins-official
```

If the elimination ladder sends you to a different render model, substitute it here: use
[`sdxl`](../../sdxl/) if you need deep pose control, or [`krea-2`](../../krea-2/) if you want a
non-default look. **Whatever model you choose to render on is also the model you train on.** Make
this choice before step 2, not after.

### Step 0 — gates

- **Likeness.** The character must not resemble an identifiable real person. "Based on" a real
  person is not the test — it must not *resemble* one. See
  [`character-lora-training`](../../character-lora-training/), *Before anything: can you publish
  it?*
- **Licence.** If you will sell images, almost any model clears. If you will ship or host the
  pipeline itself, stay on [`z-image`](../../z-image/) (Apache-2.0, covering weights and outputs) or
  [`sdxl`](../../sdxl/). Also make sure every rung of the later ladder is clear.
- **Budget.** Set a cost guard before you create anything, and **put the clock outside the pod**.
  Use RunPod's `runpod-usage` for GPU selection. Do not reach for `--stop-after` or
  `--terminate-after`: `runpodctl` 2.12.0 (2026-08-27) removed both, and its PR #330 says they were
  forwarded to the API but never enforced, so a pod "guarded" that way kept billing. The guard that
  works is a scheduler or a shell `trap` on your own machine that calls
  `runpodctl pod remove "$POD_ID"` unconditionally when the job ends or the script dies, plus a
  detached watchdog on the pod (sized for the whole session, eval included) for the case where
  your machine goes away `[official — runpodctl v2.12.0 release notes; PR #330]`. A rented-GPU
  session must also **end with a burn check**: confirm nothing is `RUNNING` and every volume is
  accounted for before you call the session done. The guard shape, the watchdog and the check are
  in [`comfyui-on-runpod`](../../comfyui-on-runpod/), *Cost guards that actually work*. If you are
  an agent running these playbooks, treat the burn check as a STOP gate, not a suggestion.

### Step 1 — lock the anchor image

You need one image that *is* the character. Everything downstream inherits it, so put more attempts
into this step than it feels like it deserves.

- Render it in the model you will deploy on, at that model's native resolution bucket.
- Read that model's `prompting-guide.md`. The prompt dialect decides whether you get the face you
  asked for. [`z-image`](../../z-image/) and [`krea-2`](../../krea-2/) want sentences;
  [`sdxl`](../../sdxl/) wants tags inside 77 tokens.
- If the look is right but the skin is not, treat it as a pipeline problem, not a model problem.
  Finish the face through a Z-Image pass at ~0.2 denoise `[community — nsfwVariant, Civitai; via z-image and sdxl]`
  ([`image-production-workflows`](../../image-production-workflows/)).

**STOP:** do you actually like this face at 100% zoom? Every step after this one makes the face
*more* permanent, not better.

### Step 2 — build the dataset

15–30 curated images beat 100 mediocre ones, and coverage beats volume `[community — NanashiAnon, L3n4/Civitai 25645; convergent, via character-lora-training]`.

- **People skip the 8-point rotation**, and that is the number-one cause of a LoRA that collapses to
  a single pose. Also vary elevation, shot size, expression and lighting.
- **The chained approach is the standard method.** Hold the character description fixed, vary
  exactly one clause at a time, generate each variation from the anchor with an edit model — [`qwen-image`](../../qwen-image/)'s Edit-2511
  is the one the image skills route to — and curate hard.
- **A video turnaround solves rotation exactly rather than approximating it** `[community — via character-lora-training]`. Prompt a slow 360°
  with no cuts, then cut the clip into frames. Check the harvested model's licence first, because
  some licences bar using outputs to train.
- The full protocol is in [`character-lora-training`](../../character-lora-training/)
  `references/dataset-and-captioning.md` §2.

### Step 3 — caption

**Caption the residual.** Describe what varies between images. Never describe the face, because a
caption that names the identity teaches the model that the identity is optional. The form of the
trigger token depends on the encoder class: use a rare literal token on a CLIP encoder, and fold it
into a natural phrase on an LLM encoder.

### Step 4 — train

- For hyperparameters, check your model skill's `lora-training.md` first, then the cross-model
  starting ranges in [`character-lora-training`](../../character-lora-training/).
- **Save checkpoints throughout the run.** The best epoch is rarely the last one. A run without
  intermediates gives you a verdict instead of a choice.
- Some model-specific rules cannot be skipped: Z-Image trains on Base and deploys via the detailer
  swap. Krea 2 trains on Raw and *samples* on Turbo. For SDXL, the choice of base finetune
  dominates everything else.

**STOP:** run 200 steps and look at the training samples before committing to the full run.

### Step 5 — evaluate

1. Training samples are free, but they are only good for finding roughly where the useful region
   is.
2. Build a **checkpoint × strength grid** with fixed prompts and a fixed seed, generated in the tool
   you will ship from.
3. **Judge it blind.** Shuffle the outputs unlabelled, pick the best, then reveal the labels. This
   routinely reverses the answer the labelled grid gave.

**Probe out of distribution, or you have tested nothing.** Try a costume, a painted style, and a
wide shot where the face is small. Score likeness and prompt-adherence **separately**, because they
peak at different checkpoints. The tooling is in
[`character-lora-training`](../../character-lora-training/)
`references/evaluation-and-tooling.md`.

### Step 6 — deploy to RunPod

- **The dual mount root is the whole step.** The same volume appears as `/workspace/` on a pod and
  `/runpod-volume/` on a serverless worker. `extra_model_paths.yaml` must therefore declare **both
  roots with identical key sets**. Miss one and you get the signature failure: everything works in
  the studio and breaks in serverless.
- Lay the volume out by **which loader node reads each folder**, and fold LoRAs by base model.
- Move weights with a **CPU pod or the S3 API**, never a GPU pod.
- Keep a **manifest** so a new pod can reproduce the old one.
- All of this is covered in [`comfyui-on-runpod`](../../comfyui-on-runpod/). For provisioning, GPU
  choice and pod lifecycle, see RunPod's `runpod` and `runpod-usage`.

**STOP:** smoke-test on the cheapest GPU that fits, and check *every* output branch before scaling.

### Step 7 — the production ladder

The ladder runs: base gen → refine/hires → **FaceDetailer, where the character LoRA swaps in** →
tiled upscale → ColorMatch and finish. Every stage past the first can be bypassed, so preview
before the expensive passes.

**One identity rule decides this step: re-assert the character *after* the last whole-image pass.**
Any refine above ~0.35 denoise drifts the face `[community — convergent; via image-production-workflows]`, so the detailer stage goes last. See
[`image-production-workflows`](../../image-production-workflows/) for settings.

### Step 8 — optional: make it move

The still you just finished is what controls the shot. Wan's I2V is far stronger than its T2V. See
playbook D.

---

## 3. B — an anime character, on your own card

**The goal:** a consistent anime character without renting anything.

**Stack:** `anima`, `character-lora-training`, `image-production-workflows`.

**Why this is a different playbook, not a variant of A:** [`anima`](../../anima/) inverts three of
A's assumptions. First, it is a 2B model, so inference runs on ~8 GB and **LoRA training fits in
~6 GB**. Failed runs cost nothing, which changes how you learn. Second, it prompts in **Danbooru
tags** even though it has an LLM encoder, so A's sentence-prompt advice is actively wrong here.
Third, it already knows thousands of characters by tag, so **check whether your character needs a
LoRA at all** before you train one.

1. **Settle the licence first, and specifically.** Outputs are commercially free, because the card
   places them outside the non-commercial term. The **weights** are not free. Selling illustrations
   is fine; shipping a product or API that contains Anima is not. If you must ship the model, switch
   this playbook to [`sdxl`](../../sdxl/) with an Illustrious/NoobAI/Pony V6 XL finetune. Not Pony
   V7: it is AuraFlow, not SDXL, and its licence bars inference services and >$1M companies.
2. **Learn the dialect before anything else.** That means tag order, the `@artist` prefix (which is
   mandatory), the `score_*` and rating tags, and prompt weights pushed far past SDXL norms. A
   prompt in the wrong dialect looks like a bad model.
3. **Train, and do not train the LLM adapter** (`llm_adapter_lr 0`). Training the adapter rewrites
   prompt understanding globally. That failure presents as "Anima got worse", not as a broken LoRA,
   which makes it the single most confusing failure here.
4. **Control is the gap.** LLLite covers lineart, depth and scribble, but there is **no pose and no
   canny**. If the shot needs pose control, compose it in an SDXL anime finetune and refine through
   Anima at low denoise. This is a live community pattern, and at 2B it is cheap.
5. Finish on the ladder as in A, minus the realism passes.

---

## 4. C — a design image with real text in it

**Stack:** `ideogram-4`, `image-production-workflows`.

1. **Decide the surface first, because the licence decides it for you.**
   [`ideogram-4`](../../ideogram-4/)'s open weights are non-commercial with **no escape variant**.
   The commercial path is the hosted API or web app, not another checkpoint. Settle this before you
   build anything.
2. **Write the prompt as a JSON document**, not as prose. The caption schema and `bbox` layout are
   the model's actual control surface. See `references/json-caption-guide.md`.
3. **Treat the result as a plate, not a picture.** Its strongest use is as a typography or design
   layer composited into imagery that another model rendered. Ideogram is weak on photoreal faces,
   and nobody has published control or identity adapters for it.
4. **The composite step is reconstructed craft, not a published graph** `[flagged — no canonical
   workflow]`. Expect to invent it. See [`image-production-workflows`](../../image-production-workflows/).
5. **If the image must move, animate around the type, not through it.** Rendered text does not
   survive motion cleanly.

---

## 5. D — turn a still into a shot

**Stack:** an image skill (whichever one locked the still) + `wan-2-2`, or `ltx-2-5` / `minimax-h3`.

1. **Lock the still first.** This is the whole reason image and video live in one suite: I2V is much
   stronger than T2V, so the still carries the composition, the identity and the look. Spend the
   effort there.
2. **Pick the video model on your constraint, not on the demo reel:**
   - You need motion and camera control, under a licence with no conditions → [`wan-2-2`](../../wan-2-2/).
   - You need sound generated in the same pass → [`minimax-h3`](../../minimax-h3/) **if you are
     outside the US/EU/UK/KR**, otherwise [`ltx-2-5`](../../ltx-2-5/). Note that LTX bars explicit
     content and bars competing with Lightricks' products at any revenue.
   - You need several connected cuts from one generation → [`ltx-2-5`](../../ltx-2-5/), and only it.
   - You have ~14 GB and Wan 5B is not enough → **HunyuanVideo-1.5** is the other open answer, not
     covered here, and its licence excludes the EU, UK and South Korea. Read the card first.
3. **Match the handoff format.** Aspect ratio and resolution are per-model, and video models have
   frame lattices — LTX uses `8n+1`, Wan has per-variant counts. A mismatch silently drops the tail
   of the clip.
4. **Post-process in this order: restore or upscale first, interpolate second.** In the reversed
   order, the restorer inherits the interpolator's smears across double the frames. A per-frame
   image upscaler is wrong at either position, because it has no cross-frame consistency and
   therefore shimmers.
5. **If the clip has audio, expect most post nodes to drop it.** Re-mux deliberately.

---

## 6. E — put your character into footage you already have

**Stack:** `krea-2`, `scail-2`, `character-lora-training` — or `wan-2-2` in place of `scail-2` when
the shot is a close-up.

This is the suite's most sequence-sensitive route, and the ordering is not obvious.

0. **Pick the replacer by the shot, not by reputation.** [`scail-2`](../../scail-2/) wins whole-body
   replacement, non-human subjects, complex action and multi-character scenes. Wan **Animate 2**
   ([`wan-2-2`](../../wan-2-2/), native ComfyUI since 2026-08-08) wins close-up faces, eye movement
   and lip sync `[community — Wensleydale on X, dreamerland.ai; two sources]`. The split
   was measured against the original Animate, not Animate 2 (flagged in
   [`model-rankings.md`](model-rankings.md) §9). If you need to
   *edit* the footage rather than replace a person — relight it, restyle it, remove something —
   nothing in the suite does that; **Bernini-R** (ByteDance, Apache-2.0) does, and is not covered
   here.
1. **You need real footage.** [`scail-2`](../../scail-2/) has no T2V and no I2V. It tracks a driving
   clip; it cannot originate a shot.
2. **Edit the driving clip's own first frame into your character.** Do not use a reference image you
   like — use *that clip's frame 0*, edited. [`krea-2`](../../krea-2/)'s Identity Edit is the
   standard tool. This is the undocumented step that decides whether the whole job works.
3. **Masks are the real control surface**, not the prompt. The prompt barely matters here.
4. **Know the limits before you shoot.** Multi-person scenes are the known soft spot, clothing
   morphs past about five seconds, and **no SCAIL LoRA training path exists** — Wan LoRAs will not
   load either.
5. **If you need a specific likeness rather than just "a different person,"** build the identity
   upstream first ([`character-lora-training`](../../character-lora-training/)) and bake it into
   frame 0.

---

## 7. F — run the whole thing as an API

**Stack:** `comfyui-on-runpod` + RunPod's `runpod`, `flash`, `runpodctl`, and `runpod-templates` if
you start from one of RunPod's own images.

1. **Build in the GUI, but ship the API-format JSON.** Export (API) format is not the same as the UI
   format, and the `/prompt` endpoint takes the former. See
   [`image-production-workflows`](../../image-production-workflows/)
   `references/workflows-as-code.md`, which compares the four automation routes.
2. **Use `/run` plus polling, never `/runsync`,** for anything that takes real time.
3. **Validation errors arrive inside the status body**, not as an HTTP error. A job can "succeed"
   and produce nothing.
4. **Declare both mount roots** in `extra_model_paths.yaml`. This is where the studio/serverless
   split bites hardest, because it only shows up in production.
5. **Smoke-test every output branch.** A video model with audio can emit a perfectly valid silent
   file.
6. Provisioning, autoscaling, webhooks and streaming are RunPod's own golden paths —
   `20-model-caching-endpoint`, `13-autoscaling-tuning`, `16-serverless-webhooks`.

---

## 8. G — adult work

**Stack:** `krea-2` or `sdxl` (image), `wan-2-2` or `minimax-h3` (video), plus
`character-lora-training`, and [`comfyui-on-runpod`](../../comfyui-on-runpod/) if you are renting the
GPU. For model choice and the census, see [`adult-work.md`](adult-work.md).

**Start with the gate, and note that it is the one gate here with no trade-off.** Invented adult
characters only. Sexual content depicting minors, and sexual imagery of real identifiable people
without consent, are absolute lines. They are not licence questions, and they are not capability
gaps to route around. If a real person is anywhere near the dataset, stop and read
[`character-lora-training`](../../character-lora-training/) `references/publishing-and-likeness.md`
before anything else.

1. **Pick the checkpoint before you pick settings.** This inverts the normal order, and it is the
   single highest-leverage step. Adult results come from purpose-built finetunes and merges, not
   from a base model plus clever prompting. The reported settings on the leading Krea 2 adult
   checkpoints are deliberately plain (Euler or ER SDE, 10 steps, guidance 1.0), precisely because
   the checkpoint is doing the work. One boundary: checkpoint-first governs *scene generation*, not
   your training base. A photoreal character LoRA destined for adult work still trains on the
   photoreal winner, and the checkpoint re-enters at scene time ([`adult-work.md`](adult-work.md)
   §2).
2. **Settle the licence against what you are shipping**, not against the content.
   [`wan-2-2`](../../wan-2-2/) is the only video path with no acceptable-use clause at all.
   [`minimax-h3`](../../minimax-h3/) is the capability leader **and excludes the US/EU/UK/KR**.
   [`ltx-2-5`](../../ltx-2-5/) bars explicit content outright.
3. **If anatomy is wrong, do not reach for a conditioning trick.** The limit is training data, not
   refusal. An abliterated text encoder cannot help, because refusal lives in layers a text encoder
   never uses. Change the checkpoint instead.
4. **Do not try to fix missing anatomy with a character LoRA.** Teaching *who someone is* takes
   15–30 images. Teaching anatomy the base model lacks takes 1,500+ images at rank 128+. These are
   different jobs at a different scale. Stacking a capability LoRA under a character LoRA is the
   usual move, and it frequently destabilises the character — this is a live unsolved constraint,
   not a solved recipe.
5. **Caption explicitly when training.** Euphemism teaches the model that the explicit content *is*
   the character. Automated captioners fail here, so adult video datasets are captioned by hand —
   a real cost on top of frame count.
6. **STOP:** before a long run or a rented GPU, generate one clip or one set. If only the seed
   changes the result, change the checkpoint rather than the seed.
7. **For video, work from a reference.** H3's undressing and anatomy work depends on a nude
   reference image (or close-up anatomy references). Base nudity without one is unreliable. Ref2VA
   is the more flexible mode.
8. **Finish on the ordinary ladder** — detailers, restore-before-interpolate, ColorMatch. None of it
   changes for adult work.

---

## 9. The joins that fail silently

The handoffs between skills are where the expensive failures live, because each side of a handoff is
individually correct.

| Join | What fails | The rule |
|---|---|---|
| Image model → image model | A foreign latent produces black or deep-fried output, sometimes subtly | **Decode to pixels between families**, always |
| Refine pass → identity | The face drifts above ~0.35 denoise and nothing errors | Re-assert identity **after** the last whole-image pass |
| Training base → deployment base | The LoRA loads on the distilled sibling at reduced strength and reads as "weak LoRA" | Train on what you will render on; record real compatibility in the manifest |
| Still → video | Aspect ratio, resolution or frame lattice mismatch; the tail is dropped without an error | Match the video model's buckets and frame arithmetic before generating |
| Restore ↔ interpolate | Interpolating first bakes noise into double the frames | Restore first, always |
| Pod → serverless | Model-not-found for a file visibly on the volume | Declare **both** mount roots with identical keys |
| Model licence → chain licence | Everything renders; delivery is blocked at the end | Settle the chain's terms, not each model's, before building |
| Skill → skill | A link points nowhere | The sibling is not installed — see [`installing-skills.md`](installing-skills.md) |
