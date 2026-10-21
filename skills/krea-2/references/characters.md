# Krea 2 — Consistent characters

If you want a persistent, reusable character with Krea 2, you use the character-LoRA pipeline. It is the highest-fidelity production path. FLUX.2 has an adapter shortcut for this job, but Krea 2 does not. Ecosystem verified 2026-07-07, two weeks after open weights, and re-swept 2026-09-09 at eleven weeks: the tooling has moved (a pose ControlNet, a mature Identity Edit LoRA), and the LoRA path now has twenty-one first-hand runs behind it (`lora-training.md §3a`).

## Contents
1. [The state of identity tooling](#1-the-state-of-identity-tooling)
2. [The character-LoRA pipeline](#2-the-character-lora-pipeline)
3. [The nearest no-training tools (young)](#3-the-nearest-no-training-tools-young)
4. [Deployment: the detailer-stage swap](#4-deployment-the-detailer-stage-swap)
5. [Multi-outfit, multi-character](#5-multi-outfit-multi-character)
6. [Failure modes](#6-failure-modes)
7. [When to use another model](#7-when-to-use-another-model)

---

## 1. The state of identity tooling

| Tool class | Krea 2 status |
|---|---|
| PuLID | **does not exist** |
| InstantID / IP-Adapter (face) | **does not exist** |
| Native multi-reference generation | not in the open models; hosted style refs condition *style*, not identity |
| Official edit model | **announced, unreleased**. The tech report names "robust editing, image reference" as future work and calls it "coming" `[official — tech report; Krea CTO on HN]` `[pending release]` |
| Character LoRA | **works — the production path** (multiple named recipes; `lora-training.md §6`) |
| Community identity-edit LoRA | **Now the standard tool, not an experiment.** `conradlocke/krea2-identity-edit` reached **v1.2** as of August 2026. It does instruction-based, identity-preserving editing. It needs its own `ComfyUI-Krea2Edit` node pack, because it uses dual conditioning (in-context VAE tokens plus image-grounded Qwen3-VL encoding) that stock nodes cannot drive. It ships workflows. See §3 for the craft. `[community — NatalieCrypto, MelodicFuntasy; convergent]` |
| Community edit/reference nodes | Ostris edit stack + Conditioning-Rebalance, and a packaged reference-image edit workflow built on Rebalance-class nodes (§3) |
| Structural control (pose / depth) | **Two community ControlNet-LoRAs**: a depth CN (Tanmay Patil) and an **OpenPose CN** that takes a DWPose skeleton and follows body pose while the prompt drives identity, clothing and style `[community — thedeoxen, HF Krea-2-pose-controlnet, 2026-08-04]`. No canny or union CN yet. These hold *pose*, not identity — pair them with a character LoRA |
| Pure-prompt consistency | the "description-locked character sheet" technique (§3a). No training, no adapter |

Plan on the LoRA for production **when likeness must survive close-ups**. That advice has weakened since it was written. **Identity Edit reached v1.2 and became the community's default identity tool.** People now use it well outside Krea 2 work. It is the standard way to prepare a first frame for video character-swap models (see §7). It is also the standard undressing tool in the adult community. Treat the adapter layer as *working*. Treat the LoRA as the higher-fidelity option rather than the only one.

## 2. The character-LoRA pipeline

The chain is the suite-standard one. [`z-image`](../../z-image/)'s `characters.md` documents the full craft. This section covers only the Krea-2 deltas.

1. **Anchor image** — generate the character once, at your best quality. Krea 2 note: fight the two taxes *before* anchoring (Wan-VAE decode, texture anchors, expression coaxing — `SKILL.md`) so you do not lock a waxy default into the character.
2. **Dataset factory** — there is still no *official* edit model. Two routes exist. The community Identity Edit LoRA (§3) generates prompted rotations, expressions and lighting from one anchor in-family, or borrow another family's edit model (the z-image/[Qwen-Image-Edit](../../qwen-image/) trick); cross-family is fine because the LoRA learns from pixels, not latents. The route that has actually shipped a character, though, is **prompt-only rendering from the base at LoRA strength 0**: a 32-cell fully synthetic set, rendered at 1024 native on the deploy checkpoint, with rotation held by a description-locked identity block (§3a). The render settings that make or break that set are in `lora-training.md §9`.
3. **Coverage** is model-agnostic craft and lives in [`character-lora-training`](../../character-lora-training/). The Krea-2-specific additions: over-represent non-neutral expressions, because the base under-produces them (the muted-expression tax); and render every cell against a plain painted wall, because patterned-wall wording prints onto skin on the adult finetunes (`lora-training.md §9`).
4. **Captioning** — prose, caption-the-residual; the Krea-specific findings (the trigger is redundant, hair becomes promptable, age and freckles do not) are in `lora-training.md §5`.
5. **Train on Raw** (rank/alpha 32, LR 1e-4 — the recipe held across twenty-one runs, `lora-training.md §3a`), then **validate on every checkpoint and resolution you will deploy on**, not only Turbo at 1024 (`lora-training.md §8`).
6. **Deploy at strength 1.0** — at the detailer stage per the community recipe, or in a single full-strength pass per the lab's measured default (§4).

Dataset size: the earliest recipe used 474 images plus 348 regularisation images `[community — JahJedi]`. The now-better-replicated result, however, is that **protocol-size datasets work**: about 50 images, 2–3k steps, trained at 1024 only. The author rates the likeness *above* their Z-Image results from the same datasets. Settings: AI-Toolkit, **LoKr factor 4, Automagic3 optimizer, sigmoid scheduling, LR 1e-4** `[community — Any_Tea_3499, r/StableDiffusion]`. The same author notes one weakness: tattoos. Ideogram beats Krea 2 at learning them, and Krea 2 beats Z-Image. The first-hand band is smaller still: **13–32 images**, with 25 real photos beating a 74-image superset of themselves blind, and a fully synthetic 32-cell set landing likeness with clean skin `[live-use — media lab, krea2-v1 vs v4 (Amy), krea2-v5 (Ciara), 2026-08-27 / 2026-09-08]`. Start at protocol size, and change *which* images before you add more.

Multi-character LoRAs are possible but caption-sensitive. In one documented failure, two characters held up under training-style captions, then *bled into each other* under creative free-form prompts `[community — krigeta1, r/StableDiffusion]`. Keep each character's identity block verbatim-stable in your prompts (see §3a, where description discipline is the same lever). Keep per-character detailer passes as the fallback (§5).

**There is a training-side answer, and if it replicates it is the most consequential Krea-2 training finding of the last month: Differential Output Preservation.** Train each character LoRA with DOP enabled against a **class** (`"woman"`), on top of a LoKr config, at **1500 steps** rather than 750. Previews stabilise around 1500. Multiple character LoRAs then load together with minimal bleed. The author reports accidentally leaving a character LoRA active and seeing almost no contamination. Read the confidence honestly: this is **one author's runs**, on top of LilBrownBebeShoes's LoKr config, which is itself unreplicated. Nobody has published a second set.

The same author states what it does not do, so these limits carry exactly the same single-source confidence as the result:

- **Hard cap of 4 characters.** A five-character generation fell apart. Four works reliably.
- Characters **borrow features from each other**, lips especially, so similar-looking characters drift toward looking related. The greater the visual difference between characters, the better it holds.
- **Prompt the distinguishing features.** If one character has a long nose, say so. Naming what separates them is what keeps them separated.
- The author's datasets were captioned lazily, with the trigger word only. Better captioning is expected to strengthen likeness: the well-captioned dataset in their set produced the more resilient LoRA.

**The cross-model detail that matters for this suite:** the same technique **worked on Krea 2** but **failed on Z-Image Base**. On Z-Image Base, DOP essentially prevented the character from being learned at all. This suite otherwise treats the two models as near-interchangeable for character work. If the finding holds, it is a real, testable difference between them, and a reason to pick Krea 2 when the job is several characters in one frame. [`z-image`](../../z-image/) records the failure half from the same source and hedges it the same way. Treat the pair as one report, not two. A/B it on your own characters before you plan a shoot around it `[community — MASilverHammer; single report]`.

## 3. The nearest no-training tools (young)

- **Identity-edit LoRA** — `conradlocke/krea2-identity-edit` (unofficial fine-tune of Raw), **v1.2**. Give it an image plus a plain-language instruction, and it edits while preserving what you did not ask to change, *including the person*. It requires the `ComfyUI-Krea2Edit` node pack, because the LoRA's dual conditioning (in-context VAE tokens plus image-grounded Qwen3-VL encoding) needs nodes that stock ComfyUI does not have. Ready-made workflows ship with it. It is the most direct identity tool that exists, and the obvious dataset-factory engine (§2.2).

  **Craft that emerged from a month of use** `[community — NatalieCrypto, MelodicFuntasy, r/unstable_diffusion]`:

  - **One short sentence beats a paragraph.** This is the single most-repeated finding, and it inverts normal Krea 2 prompting doctrine. *"Turn her to profile."* *"Change her outfit to a singlet and a short skirt."* Paragraphs degrade the edit.
  - Background and lighting genuinely **do not move**. Edits that would need masking and inpainting on other stacks are one sentence here. No masks, no inpainting.
  - Demonstrated: profile turns, object swaps, taking outfit from one reference and pose from another, expression-only changes.
  - **Posing is the weak axis**, and a pose change will shift the face somewhat. If the pose is the point, this is the wrong tool.
  - **It stacks with other LoRAs.** Krea 2 + Identity Edit + an NSFW LoRA is a named production pattern for putting a *consistent* character into adult scenes `[community — Clone-Protocol-66]`. The same stack is the community's standard undressing pipeline. The edit LoRA holds identity, and the other LoRA supplies the content the base model is thin on. Neither can do the other's job. Where the base model's own reticence is what is in the way, the `krea2filterbypass3` line goes on the same stack (`lora-training.md §2b`).
  - Its **fastest-growing use is outside image work entirely**: prepping a first frame for a video character-swap model, so the reference already matches the driving clip's pose and framing. See §7.
- **Ostris edit stack** — `ostris/ComfyUI-Krea2-Ostris-Edit`. It feeds up to **3 reference images** through the Qwen3-VL encoder plus a model patch, driving paired-data **edit LoRAs** (a concept-edit LoRA trains in ~1,750 steps) `[community — Ostris; comfyui-wiki write-up]`. Third parties are already training on the method. One example is sktksm's detail-enhancer edit LoRA, trigger "enhance this image", with its own author's caveats: experimental, weaker on horizontal aspect ratios, can shift lighting and colors `[community — sktksm]`. It is not an identity guarantee. It is an edit conditioner, not a face embedder.
- **Conditioning-Rebalance** — `nova452/ComfyUI-Conditioning-Rebalance`. This is "IP-Adapter-like" per-layer conditioning manipulation that exploits the 12-layer text tap. It does image-reference editing plus the safety-tuning bypass, and a late-June update added negative prompting for edits `[community — nova452]`. It is powerful and experimental in equal measure.
- **A packaged reference-image edit workflow** — a RunComfy workflow that takes a reference image plus a plain-English instruction and returns an edit that keeps subject, pose and framing. Mechanically it runs open **Turbo** through `UNETLoader` with Rebalance-class custom nodes (`TextEncodeKrea2` for full-image or masked edits, `Krea2EditRebalance`, `Krea2SystemPrompt`), optional style LoRAs on top, and no Identity Edit LoRA `[community — RunComfy workflow, 2026-07-15; single report]`. It is a third route beside Identity Edit and the Ostris stack, not a new model. Treat its preservation claims as its author's and check the identity on your own subject.

All are single-maintainer projects days-to-weeks old. Pin versions and expect churn.

### 3a. Pure-prompt: the description-locked character sheet

Krea 2's prompt adherence is strong enough that an **exhaustively specific character description, repeated verbatim across scene prompts, holds a near-identical character with no training at all**. One author generates unlimited "panels" this way. Each character gets a character-sheet prompt covering age, height, build, skin, hair, eyes, and wardrobe down to belt buckle and earrings. Every scene prompt then re-states the full block, and a local VLM splits a master prompt into per-scene prompts `[community — aurelm, r/comfyui "Infinite panels" workflow]`. The mechanism is that the 12-layer Qwen3-VL tap rewards precise, repeated description, so identity lives in the text. The costs: prompts run very long (watch the 512-token ceiling per scene), and likeness is "near-identical", not pixel-locked. This is good for storyboards, keyframes, and dataset bootstrapping. Use a LoRA when likeness must survive close-ups.

## 4. Deployment: the detailer-stage swap

This is suite-standard, and it applies verbatim. Generate the base image from a detailed prompt with **no** character LoRA, so composition stays free. Then apply the LoRA in the FaceDetailer-class pass where the face is re-rendered, at ~0.8 strength for the published recipe. Match the detailer's prompt to the actual image, or you get the LoRA's default face. On Krea 2 specifically, run the detailer on the two-stage recipe's output or post-Wan-VAE decode (`setup-and-workflows.md §7`), so the detailer is not fighting the soft default. SAM3 face/eye detailers are the community's current pick `[community — lonecatone23]`.

**The lab's measured default is the opposite: one pass, LoRA at full strength, in-frame-only prompting.** A face pass at denoise 0.5 reskins the base's face rather than replacing it, so identity needs 0.7+, and that brings the double-pass softness back. The single-pass route beat it. The detailer-stage swap is kept only for putting identity onto a *generated body* `[live-use — media lab, Amy repaint sweeps, 2026-09-05]`. Two working recipes, opposite defaults, no head-to-head on one subject `[contested]`. Numbers in `lora-training.md §10`.

## 5. Multi-outfit, multi-character

There is no Krea-2-specific evidence yet, so the suite's shared limits are the planning assumption. Multi-outfit LoRAs cap out around ~6 outfits before bleed. Multi-character scenes have **no regional tooling** here: no regional prompter, and no attention masking like FLUX.2's. The working method is therefore per-face detailer passes. Run one pass per character, each with its own LoRA and each prompted to its own face. Character style-bleed rules follow the LoRA-stacking norms (identity ~0.8, drop style LoRAs to 0.3–0.6 when stacking). The JahJedi recipe's "holds at 0.8 under heavy style mixing" is the one measured point.

## 6. Failure modes

| Failure | Mechanism | Fix |
|---|---|---|
| Expression lock-in (character always neutral) | Base model's muted-expression tax compounding a neutral-heavy dataset | Over-represent expressions in the dataset; verify base can produce the expression first (`SKILL.md` taxes); bypass/Rebalance at inference |
| Angle collapse (only frontal views resolve) | Dataset missing rotation coverage | 8-point protocol; regenerate the missing angles via an edit model |
| Same-face overfit (every generated person is the character) | Over-trained, or captions did not isolate identity | Fewer steps or lower weight; caption-the-residual properly; regularisation images (JahJedi used 348) |
| Waxy/soft character at deploy time | The VAE/softness tax, not the LoRA | Wan-VAE decode + detailer pass before blaming training |
| Style bleed from stacked style LoRAs | Weight budget exceeded | Identity 0.8, styles lower; re-test each style added |
| Identity great on Raw, unstable on Turbo | Validated on the wrong checkpoint | Always evaluate at Turbo 8-step guidance-off (musubi `--turbo_dit`) |
| Identity fine on Turbo, cheeks break up on the adult finetune | Validated at one resolution; the finetune's band starts higher | Evaluate on every deploy checkpoint × resolution; ≥1344×2016 on the finetune (`lora-training.md §8`) `[live-use — media lab, krea2-v3, 2026-09-08]` |
| Reticulated / "scale" skin that no rank or step change removes | Dataset rendered above the base's band (2048+) or at 20 steps; the texture survives downscaling | Re-render the dataset at 1024 native, 12 steps, cfg 1.0 (`lora-training.md §9`) |
| Freckles turn to reptile scale at render time | Prompt too long (~95 words) or a realism tail re-added over the LoRA | ~45 words, one clause per idea, `"an unretouched film photograph."` as the whole tail (`lora-training.md §10`) |
| Heads cut off on head-and-shoulders prompts | The LoRA learned a chest-centred crop from doubled crops in the set | Taller canvas (832×1216); drop the crops from the dataset — framing words lose |
| One hair colour on every prompt; a different woman when you ask for another | A colour majority in the set (29 of 32 cells) bound to the trigger | Spread colours with no majority across the same cells; caption every one `[live-use — media lab, krea2-v5 red-hair probe, 2026-09-09]` |
| The trigger word appears on signs and badges | "a woman named z…" in every caption is read as a name to render | Prompt around signage, or use the bare token |

## 7. When to use another model

- **No-training identity today:** [`flux-2`](../../flux-2/). Its native multi-reference (ReferenceLatent) plus PuLID is the suite's strongest adapter path.
- **Expressive close-up faces:** [`z-image`](../../z-image/) has better *base-model* facial expressiveness and hair. It is also the natural *finisher* for a Krea-2 scene: generate the scene here, run the face pass there (`setup-and-workflows.md §8`). Note the distinction that emerged in July 2026 testing. Z-Image still wins on un-trained expressiveness. *Trained character likeness* on Krea 2, however, is now reported at or above Z-Image's (`lora-training.md` §5 carries the run).
- **Mature multi-character tooling:** [`sdxl`](../../sdxl/) has regional prompting and the deepest adapter toolbox, and it is still the safe choice. §2's Differential Output Preservation result would close much of that gap on Krea 2, allowing up to **4** character LoRAs with minimal bleed where Z-Image fails outright. It is, however, a single unreplicated report (§2). Treat it as the promising option to test, not the settled one to plan around.
- **Editing a character into a new scene, pose or state:** now genuinely contested. [`minimax-h3`](../../minimax-h3/) run at **one frame** is reported to beat Krea 2 + Identity Edit on character fidelity, 3D scenes, mirrors and composition. See that skill's edit-mode coverage. Krea 2 + Identity Edit remains far cheaper to set up and is the better default for scene-preserving one-sentence edits.

Krea 2 earns the character job when the character lives inside its aesthetic strengths: wide-aspect scenes, stylised looks driven by style LoRAs/refs, and anatomy-heavy compositions. It also asks that you are willing to run the LoRA pipeline.

### Krea 2 as the front half of a video pipeline

This is worth stating separately, because it is now the single most common thing people do with Identity Edit, and it is not an image job at all. **Video character-replacement models want a reference that already matches the driving clip's pose and framing.** So the pipeline runs in three steps. Take the actual first frame of the driving video. Edit it into your character with Identity Edit (or Flux Klein 9B). Then hand *that* frame to the video model as the reference. Doing this rather than supplying a generic portrait is reported as the difference between mediocre and excellent character swaps `[community — blackmixture]`. This applies to [`scail-2`](../../scail-2/), which documents this handoff from the video side, and to [`minimax-h3`](../../minimax-h3/)'s Ref2VA video-edit mode alike.
