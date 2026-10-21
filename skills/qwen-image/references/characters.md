# Qwen-Image — Consistent characters

Qwen-Image's answer to identity is unusual in this suite. The **edit model is the identity tool**. The character LoRA is the second path, not the first. This file owns path selection, the reference protocol, Edit as the dataset factory for a LoRA, deployment, multi-subject work, what is unmeasured, and when to route to a sibling. Training mechanics are `lora-training.md`. The bypass graph that makes high-resolution identity edits work is `setup-and-workflows.md §5`. Ecosystem verified 2026-09-09.

## Contents

1. The state of identity tooling
2. Which path — Edit alone, Edit plus LoRA, or LoRA
3. The reference protocol — from one image to a working set
4. Edit as the dataset factory for a character LoRA
5. Deploying a character LoRA with Edit
6. Multi-subject and multi-outfit
7. Failure modes
8. When to use another model
9. Qwen-Image-2.1 and identity — ten references, no evidence, no commercial use

---

## 1. The state of identity tooling

| Tool class | Qwen-Image status |
|---|---|
| PuLID | **does not exist** (either generation) |
| InstantID / IP-Adapter-FaceID | **does not exist** (searched 2026-09-09; that family remains SD1.5 / SDXL / Flux) |
| Native multi-reference editing | **yes — the headline.** Edit-2509/2511 take 1–3 references through Qwen2.5-VL (semantics) and the VAE (appearance) |
| Vendor statement on identity | Edit-2511: "Character consistency has been significantly improved… preserving the identity and visual characteristics"; "high-fidelity fusion of two separate person images into a coherent group shot" `[official — Edit-2511 card]` |
| Community identity Edit LoRAs | **BFS — Best Face Swap** (face and head swap, ~71k downloads); **Multiple-Angles** (2509 by dx8152; 2511 port by fal, 64k monthly pulls); Next Scene; InScene; AnyPose |
| Character LoRA | works; full trainer support; the second path (§2) |
| Image → LoRA | DiffSynth's `Qwen-Image-i2L`, whose own authors flag weak generalisation |
| Regional / multi-character tooling | none beyond Edit's own multi-person fusion; per-face detailer passes remain the fallback |

State the absence plainly, because readers arriving from SDXL will look for it. There is no adapter that embeds a face. What Qwen has instead is a model that reads the reference image with the same encoder that reads the prompt. The community's verdict on that is strong: "I don't think there's a better open-weight model out there than Qwedit for making new shots of a character without loras, for now" `[community — nsfwVariant, r/StableDiffusion 1tqm8ic; strong]`.

---

## 2. Which path — Edit alone, Edit plus LoRA, or LoRA

| Path | Reach for it when | Cost |
|---|---|---|
| **Edit-2511 alone** | new shots, outfits, angles, relights of a character you have one good image of; storyboards; a video first frame | no training; one reference; ~50–130 s per shot at 1–2 MP on a 5090 |
| **Edit builds the dataset → character LoRA** | the character must survive close-ups, many scenes, T2I composition without a reference in the graph, or stacking with style LoRAs | ~130 s to generate the set, then a training run |
| **Character LoRA + Edit** | you already have the LoRA and want Edit's angle and outfit control on top | test the LoRA on Edit individually; many T2I LoRAs load and some work |
| **Face/head swap after the fact** | put an identity onto a generated body with no training | BFS or the head-swap pipeline (§5) |

Two facts decide most cases. **Sequential-edit identity retention is unpublished.** Nobody has reported how many chained edits a face survives. Edit-alone work must therefore re-anchor on the original reference every time (§3). And **distant shots degrade**. "For distant shots, Qwen LoRAs often require FaceDetailer to make the likeness look better. ZIT sometimes needs FaceDetailer too, but not as often as Qwen" `[community — Top_Buffalo1668]`. Budget a detailer pass on either path.

---

## 3. The reference protocol — from one image to a working set

This section is one author's protocol, from the sixteen-way reference A/B, unless another source is named `[community — nsfwVariant; single report, reproducible]`.

1. **Make a nude (or plain-white-underwear) reference even for SFW work.** It gives the model maximal information about proportions and how clothing should sit. Re-dressing from it is far easier than undressing a clothed reference. **No LoRA is needed.** The sensitive regions lack detail, which does not matter for a reference.
2. **Make several zoom levels by prompting the zoom** (`prompting-guide.md §2`). Use the **head-to-thighs** framing as the everyday reference, and a closer one for close-ups.
3. **Make several angles** with the Multiple-Angles LoRA. The fal 2511 port takes a structured trigger, `<sks> [azimuth] [elevation] [distance]`. Its eight azimuths are exactly the suite's rotation checklist, with four elevations and three distances, at strength 0.8–1.0 `[community — fal]`. The older dx8152 LoRA targets 2509: "I've never seen any model get new subject angles this well… the success rate is over 90%" `[community — LeoKadi]`. Identity hold at the back azimuths is claimed, not benchmarked. Curate those hardest.
4. **Drive ~90% of new shots off a single reference, and re-anchor on it for every edit.** Colour drifts per edit, from fp8 precision, 2511's contrast regression and residual drift `[community — FluffyQuack, Phr00t, DiagramAwesome]`. Identity is assumed to drift as well.

**The four things that decide identity quality on Edit:**
- **Bypass `TextEncodeQwenImageEditPlus`'s forced 1 MP AREA downscale.** It is "the primary reason all ComfyUI qwen edits give blurry images out", and bypassing it is what enables native 1440–1920 px references (`setup-and-workflows.md §5`).
- **Double-ref.** Feed the reference twice for "better resemblance of characters at different angles", at ~50% more time. "For single image edits it's ALWAYS better."
- **`Picture 1: <five words>` at the start of the prompt**, because that is the trained format.
- **The VAE halftone is a decode artefact.** Fix with the 0.5–0.75× round trip. Do not diagnose it as overfit.

Cost on a 5090, double-ref, single-image: 1 MP 52 s, 2 MP 131 s, 5.3 MP 550 s. Stay at 2–3 MP.

**Officially supported reference count is three.** Trained combinations: person + person, person + product, person + scene `[official — Edit-2509 card]`. Clothing transfer between two people is the hardest. Expect retries, match an edge dimension, or use the Clothes Try On / Outfit Extractor LoRAs (kingroka).

**If the character does adult work,** the nude reference above is already the protocol, and vanilla Edit does not refuse it. The weights are local and there is no runtime filter. Explicit *anatomy* degrades on the stock model, so the merged variants (Rapid-AIO NSFW splits) are the fallback for explicit edits. Publishing gates bind regardless (`lora-training.md §7`).

---

## 4. Edit as the dataset factory for a character LoRA

This is Qwen's answer to the v0-LoRA chicken-and-egg problem, and it is cheaper than the usual answer. **The Edit model *is* the v0.**

- A published workflow takes **one upper-body headshot** and generates **20 angle variants** on Edit-2509 fp8 + 4-step Lightning in ~130 s on a 5090. It writes the character name into 20 `.txt` sidecars as the whole caption `[community — acekiube, r/StableDiffusion 1o6xjwu]`. A Nunchaku fp4 variant does 12 captioned images in ~4 min under 16 GB `[community — The-ArtOfficial]`.
- The sibling protocol (generate ~60, curate to ~30, cut anything where the face drifted) is [`z-image`](../../z-image/)'s `characters.md §3`, which already uses Qwen-Image-Edit as its factory. The coverage checklist is model-agnostic and lives in [`character-lora-training`](../../character-lora-training/).

Two Qwen-specific notes. **Generate the set with Lightning if you like, but train on the base**, and grid the LoRA on the base at 20 steps as well as on the Lightning graph (`lora-training.md §2`). And **clean the source first.** Appearance conditioning comes straight from the VAE-encoded reference, so grain or blur in the headshot is inherited by every variant.

Then train per `lora-training.md`. The named Qwen character recipe is 40–60 images at 60/30/10 close/half/full, captions a plain name in a sentence, rank 32 and LR 2e-4 with sigmoid `[community — FarTable6206]`; the shipped alternative is 16 / 1e-4. The contested points are in `lora-training.md §8`.

---

## 5. Deploying a character LoRA with Edit

- **Train on T2I, use with Edit.** It usually loads and sometimes works, and "a dedicated training for each version will always give you better results". 2511 broke 2509 LoRA compatibility, so a character LoRA for Edit-2511 wants a `-2511` train or the Rapid-AIO merge (`setup-and-workflows.md §7`).
- **The detailer-stage swap is suite-standard and applies here.** Generate the base image without the LoRA from a detailed description. Then apply the LoRA in the FaceDetailer pass, with a prompt matched to the image. Qwen needs this pass on distant shots more often than Z-Image does.
- **Swap the identity in afterwards, with no training.** BFS ("Focus Faces" keeps head shape and hair; "Focus Head" swaps the head), or a head-swap workflow: Edit + 4-step Lightning, auto face detect and mask, ControlNet 0.9, SeedVR2 final `[community — Substantial_Angle680]`.
- **Stacking**: a character LoRA stacks with SamsungCam UltraReal at 1.0 `[community — FortranUA]`. Qwen tolerated three LoRAs where Z-Image Turbo managed two; 2512 collapses with 2–3.
- **Grid on the deploy graph**, including the Lightning regime if that is what you ship on. Keep a strength-0 control to separate the halftone, the plastic default and the fp8 grid from the LoRA.

---

## 6. Multi-subject and multi-outfit

**Multi-person fusion is native.** 2511 "enables high-fidelity fusion of two separate person images into a coherent group shot" `[official — Edit-2511 card]`. The 2509-era naming convention (`Jane is in image1. Forrest is in image2.`) plus the `Picture N:` format are how you address each subject (`prompting-guide.md §3`). Three references is the trained ceiling. Beyond that, or when fusion bleeds attributes between people, fall back to the suite's structural method. Generate the scene with generic figures, then run **per-face detailer passes, each with its own character LoRA and prompt**. There is no regional-prompting tool for Qwen-Image; Edit's own fusion is the regional tool.

**Outfits are an Edit job, not a LoRA job.** "Change her outfit to …" with "Leave her pose unchanged" appended is ~99% correct. Product placement, clothes try-on and outfit extraction have dedicated LoRAs (Futurlunatic, kingroka). A multi-outfit LoRA's ~6-outfit ceiling is the suite's shared limit, with no Qwen-specific evidence either way.

---

## 7. Failure modes

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| Every edit is soft; likeness is mushy at 1 MP | The node's AREA downscale destroyed the reference before the model saw it | Bypass; Lanczos; 2–3 MP; double-ref (`setup-and-workflows.md §5`) |
| Likeness slips after a few edits | Per-edit drift; chain length unmeasured | Re-anchor on the original reference every edit |
| Likeness fine in close-up, generic at distance | Face occupies too few latent tokens | FaceDetailer with the LoRA; closer framing; the head-to-thighs reference |
| Pose or expression changes you did not ask for | Unmentioned attributes are free | "Leave their pose and expression unchanged" |
| Back and profile angles drift | Weakest-trained angles everywhere; identity hold at back azimuths is unbenchmarked | Multiple-Angles LoRA; curate those cells hardest; add targeted dataset images |
| Generated variants inherit grain, blur or JPEG artefacts | Appearance conditioning from the VAE-encoded source | Clean or upscale the headshot first |
| A faint grid on every reference | VAE halftone, or the fp8 base under a LoRA | Round trip; scaled fp8 or bf16 |
| Character LoRA drags composition and body toward its set | Full-strength LoRA in the base pass | Detailer-stage swap; describe the character in the base prompt instead |
| Two subjects swap hair or clothing in a fusion | Attribute bleed within one conditioning stream | Name each subject and what distinguishes them; per-face detailer passes |
| A specific limb placement will not obey | Model-level composition limit; one operator failed 16 times | Pose ControlNet, or another model `[community — Mean_Ship4545]` |
| LoRA does nothing or wrecks the image on 2511 / 2512 | Generation mismatch; the base absorbed community LoRAs | Retrain for the generation; strength is not the lever |

---

## 8. When to use another model

- **Adapter-style identity** (PuLID, ReferenceLatent-class): [`flux-2`](../../flux-2/). Qwen has no face embedder; it has an edit model.
- **Photoreal skin in close-up**: [`z-image`](../../z-image/). It has better skin with nothing loaded and is the standard finisher for a Qwen composition. Z-Image sends its anchor *here* to be multiplied, then finishes the face there. The pairing runs both ways.
- **One-sentence scene-preserving edits with a mature community LoRA, or several character LoRAs in one frame**: [`krea-2`](../../krea-2/), for Identity Edit v1.2 and the single-author Differential Output Preservation result. Krea 2 also sends anchors here for its dataset factory.
- **Mature multi-character and regional tooling**: [`sdxl`](../../sdxl/).
- **The video first frame**: build it here, then hand it to [`wan-2-2`](../../wan-2-2/). Compose, multiply the angles, and pick the frame that matches the driving clip's pose. The video model inherits whatever likeness the still carries.

Qwen-Image earns the character job when the job is *editing*: many shots of one person from one good image, group fusion, outfit and angle changes, text that must survive the edit. It is not the model for the finished close-up skin.

---

## 9. Qwen-Image-2.1 and identity — ten references, no evidence, no commercial use

`[official — card and blog, 2026-09-20]`. 2.1 accepts up to ten references and its launch showcase fuses six separate portraits into one group photograph, builds a five-input virtual try-on (model, garment, shoes, bag, hat) and claims it "better preserves facial features so that a person's identity remains more consistent across edits". The rewriter's own rule is the useful craft hint: **point at the reference image for identity, never describe the face** (`prompting-guide.md §9.4`).

Three things stop it replacing Edit-2511 in §2's path selection today:

1. **Zero independent likeness reports.** Weights were hours old at this pass; no author has run the nsfwVariant-style A/B on it. Whether identity holds across ten references, across sequential edits, or at 2K, is unknown.
2. **The licence.** Research or evaluation only. Every commercial character pipeline this suite describes is out of scope without a separate licence from Qwen.
3. **No LoRA path.** Nothing on Civitai, DiffSynth the only trainer, no Lightning for the dataset-factory speed that makes §4 practical (20 angles in ~130 s on Edit-2509).

Where it may earn a place once tested: research-only group composition beyond three subjects; transparent character cut-outs straight from the model (`prompting-guide.md §9.2`); mask-guided local identity edits without an inpaint graph. Revisit after the 2026-09-28 early-access reviews.
