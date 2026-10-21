# Z-Image Characters — creating a consistent character

This file explains how to invent an original character and keep them consistent across poses, outfits, scenes, and multi-character images. It is the orchestration layer. Dataset craft lives here. Training mechanics live in `lora-training.md`. Deployment wiring lives in `setup-and-workflows.md` (§6 LoRA usage, §10 identity methods).

**Start with the Z-Image reality.** Z-Image has no PuLID, no IP-Adapter face model, and no released edit model. Z-Image-Edit is announced but unreleased. So **the character LoRA is not one option among several here — it is the path**, and the craft below is about making that path excellent. On models where adapters exist, adapters trade against LoRAs; see the flux-2 and sdxl skills. That comparison matters when you are choosing a model, not after.

---

## 1. The two paths (and how they chain)

The 2026 community consensus across model families settles on two complementary approaches. The same picture appears independently in Mickmumpitz's workflows, WeirdWonderfulAI's writeups, and the Civitai dataset guides `[community — Mickmumpitz, WeirdWonderfulAI; convergent]`:

| Path | What it is | Z-Image status |
|---|---|---|
| **Edit-model character engine** | No training: a character sheet + a multi-reference edit model ([Qwen-Image-Edit 2511](../../qwen-image/), FLUX.2 ReferenceLatent) re-renders the character into new scenes directly | Not available *in* Z-Image — no released edit variant. But see the chain below. |
| **Character LoRA pipeline** | Train a LoRA from 20–50 varied images; deploy at the detailer stage | **The Z-Image path.** Fully supported (Ostris AI-Toolkit). |

**The chain: use an edit model as the dataset factory.** The two paths are not rivals. The strongest current practice chains them. You design the character once, then use an edit model to multiply that one image into the varied dataset the LoRA needs. The usual choice is **Qwen-Image-Edit 2509/2511**, running in the same ComfyUI install. The edit model never touches inference. It only manufactures training data, so its own look does not contaminate your Z-Image outputs: the LoRA learns the *identity residual*, and your captions name everything else. The canonical writeup is WeirdWonderfulAI's "QWEN Image Edit can create Character Consistent LoRA Dataset" (Oct 2025). Mickmumpitz's Consistent Character Creator v3 is the most-cited turnkey version `[community — WeirdWonderfulAI, Mickmumpitz; strong]`.

---

## 2. Designing the character: the anchor image

Everything starts from **one anchor portrait** that defines the identity envelope. Generate it in Z-Image itself, so the identity is native to the model's distribution:

- Front three-quarter view, neutral expression, plain background, soft north-window light, 50–85 mm, 1024×1024.
- Write the identity with **named, specific, non-idealised features** — for example "copper shoulder-length hair, sun-freckled olive skin, small mole below the left jaw". These are the markers you will later *omit* from captions, so the trigger absorbs them.
- Template A in `prompting-guide.md §8` is exactly this shot. Reroll until the face is one you can describe and recognize.

Keep the anchor's **identity description text**. You will reuse it byte-identical in every dataset prompt, and later in base-generation prompts when the LoRA is not loaded.

---

## 3. Building the dataset

Target **20–50 images**. A set of 15–25 is workable, and the marginal value past ~50 is low. The governing rule: **keep the identity consistent, and make everything else deliberately varied** — angle, expression, shot size, outfit, lighting, background. Near-duplicate shots are the #1 cause of same-face overfit `[community — Civitai guides 7777/21257/21114; convergent]`.

There are three ways to multiply the anchor, in order of fidelity:

1. **[Qwen-Image-Edit factory](../../qwen-image/references/characters.md) (best).** Feed the anchor plus ~20–60 edit prompts covering the 8-point rotation, expressions, outfits, lighting setups, and settings. Generate ~60, then **curate the best ~30**, cutting anything where the face drifted. 2511 (Dec 2025) is the release character-LoRA builders reach for when consistency matters `[community — single report]`. It is not "one version behind": Qwen-Image-2.0 (Feb 2026) is API-only, so 2511 is the newest edit model you can run locally `[official — QwenLM, 2026-08-23]`. Setup below.
2. **Z-Image img2img at low denoise** from the anchor. This is the original method in `lora-training.md §2`. It works, but it drifts identity faster than an edit model at large angle changes.
3. **Pure txt2img with the identity description + locked clauses.** Use this to fill specific gaps, such as a missing back view, when the other two stall. Expect retries.

**Factory setup (ComfyUI).** The factory uses the same install as Z-Image but a different loader stack: three files plus an optional speed LoRA. All come from `Comfy-Org/Qwen-Image-Edit_ComfyUI` (diffusion) and `Comfy-Org/Qwen-Image_ComfyUI` (encoder + VAE), in the `split_files/` trees. You need ComfyUI ≥ 0.6.0 for 2511 `[official — Comfy-Org repos + docs.comfy.org 2511 template, 2026-08-23]`:

| File | Size | Goes in | Loader |
|---|---|---|---|
| `qwen_image_edit_2511_fp8mixed.safetensors` (or `..._bf16`, ~40.9 GB) | ~20.5 GB | `models/diffusion_models/` | `UNETLoader` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | ~9.4 GB | `models/text_encoders/` | `CLIPLoader`, type `qwen_image` |
| `qwen_image_vae.safetensors` | ~254 MB | `models/vae/` | `Load VAE` |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` (optional, lightx2v) | — | `models/loras/` | `LoraLoaderModelOnly`, strength 1.0 |

There is **no 2511 e4m3fn fp8 file**. Third-party posts claiming one are misremembering the 2509 generation; `fp8mixed` is the 2511 quant. The official template wires **`TextEncodeQwenImageEditPlus`** (the edit-conditioning node), `ModelSamplingAuraFlow` at **shift 3.1**, `CFGNorm`, and `FluxKontextImageScale`. It samples at **40 steps / CFG 4.0 / euler / simple**, or **4 steps / CFG 1.0** with the Lightning LoRA `[official — Comfy-Org template JSON, read 2026-09-09]`. VRAM: fp8 is comfortable on 24 GB and workable on 16 GB with offloading, because the fp8 encoder offloads to system RAM; bf16 wants ~40 GB+ `[community — convergent; no official figure]`. Licence: Apache-2.0 weights, no AUP, no output-or-training clause. Nothing restricts feeding the factory's outputs to your LoRA trainer `[official — HF card + QwenLM README, 2026-08-23]`.

**Coverage checklist.** The rotation and elevation craft is in `prompting-guide.md §3.3–3.5`; use those clauses verbatim.
- All **eight horizontal rotations**, with identical subject and lighting text and only the angle clause varying. Back views need explicit back-of-hair and outfit description, plus retries. They are the weakest-trained angles in every diffusion model.
- **One high + one low elevation** close-up per lighting setup (30–45°, gaze anchored away from camera).
- **Shot-size mix:** close-up ~30%, full body ~20%, the rest medium/cowboy.
- **3+ expressions** beyond neutral. Otherwise the LoRA absorbs the dataset's dominant expression (expression lock-in).
- **2–3 outfits** if you want wardrobe flexibility. Caption the clothing in those shots so it stays promptable.
- **Vary one thing per image.** Change outfit *or* lighting *or* angle, never two at once. If two things change together, the LoRA cannot disentangle which features are the character.

**The coverage shortcut:** fal's **Qwen-Image-Edit-2511-Multiple-Angles LoRA** (Lovis Odin @ fal, Apache-2.0) solves the rotation protocol by construction. It was trained on 3,000+ Gaussian-splat renders and takes a structured trigger — `<sks> [azimuth] [elevation] [distance]` — whose **eight azimuths are exactly this checklist's rotations** (front, the four quarters, both sides, back), with four elevations (−30/0/30/60°) and three distances (0.6/1.0/1.8×). Run it at strength 0.8–1.0 `[community — fal, 2026-08-23]`. Identity hold at the back azimuths is claimed via 2511's own consistency but unbenchmarked, so curate those shots hardest, per character. The older dx8152 *Multiple-angles* LoRA (natural-language camera prompts, far larger download numbers, Comfy-Org rehosts it) targets **2509**. Its 2511 compatibility is unverified, and 2511 natively absorbed much of that multi-angle capability, so prefer the fal 2511 LoRA.

**If the character does adult work, split the dataset by tool.** Vanilla Qwen-Image-Edit does not refuse: the weights are local, there is no runtime filter, and nothing suggests SFW character or angle editing is gated at all. But explicit anatomy *degrades*. The model is undertrained there, and the community's artifact trail says LoRAs alone hit a ceiling. The working paths are merged or finetuned variants: Phr00t's Rapid-AIO builds ship NSFW/SFW splits, on 2511 since V15, and Civitai's "abliterated" edit model is actually a LoRA merge, not an abliteration `[community — HF/Civitai artifact trail, 2026-08-23]`. So the factory doctrine stands unchanged. Multiply the clothed/coverage subset through the factory as normal, and generate the explicit subset **natively in Z-Image**, whose anatomy coverage is the stronger of the two (`lora-training.md §6`). A merged edit variant is the fallback only if you specifically need explicit *edits*.

**Captioning:** caption-the-residual, in prose, because Qwen-3 reads sentences, not tags. Name the pose, clothing, background, lighting, and angle in every caption. Leave the identity to the trigger. Full reasoning in `lora-training.md`.

---

## 4. Train, evaluate, deploy

- **Train on Z-Image Base, generate on Turbo.** `lora-training.md` covers the why, the Turbo training-adapter requirement, and hyperparameters (rank 8–16, LR 1e-4, 2000–3000 steps for this dataset size).
- **Evaluate with the XY grid** (epoch × strength) on out-of-set prompts, including at least one *new outfit and setting* the dataset never showed. A character LoRA passes when the face holds while everything else obeys the prompt.
- **Deploy with the detailer swap** (`setup-and-workflows.md §6`, end). Generate the base image *without* the LoRA using the identity description, then apply the LoRA at the FaceDetailer stage, where it gets the full sampling budget on the face. Load it on both ZIB and ZIT passes for maximum likeness. MyAIForce's writeups are the clearest named treatment of this established cross-model technique `[community — MyAIForce; strong]`. The swap exists because a character LoRA at full strength in the base pass drags composition and body proportions toward its training data.

---

## 5. Beyond the face: outfits, props, multiple characters

- **Signature outfit:** keep it *uncaptioned* in the shots where the character wears it, so it binds to the trigger and appears by default. Caption it where you want it swappable.
- **Multi-outfit LoRAs:** one LoRA can carry several outfits. You need a **unique trigger tag per outfit**, visually distinct outfits, and balanced per-outfit image counts. The practical ceiling is **~6 outfits** before quality collapses `[community — Khanykov01, Civitai 6990; strong]`. For one-off wardrobe changes, the edit-model factory (image 1 = character, image 2 = garment) is replacing wardrobe training entirely.
- **Multiple characters in one frame.** The honest status: same-prompt multi-character scenes bleed attributes, because one conditioning stream feeds the whole canvas. Hair and clothing colors migrate between figures. Regional tooling for Z-Image is **new and experimental** (next paragraph), so the reliable mitigation is still structural, not promptal. Generate the scene with *generic* figures, then run **per-face detailer passes, each loading a different character LoRA** with its own prompt. Impact Pack detailers operate per-detection, so each face gets its own pass. Plan for retries; two characters is practical, three is pushing it. The per-face-detailer pattern is the same per-subject routing idea that ADetailer's `[SEP]` syntax implements on SDXL — see [`sdxl`](../../sdxl/references/characters.md) `[community — convergent]`.

  **Two regional-prompting tools appeared in June 2026, and both are early.** Until then Z-Image had none. The SD-era regional nodes hook cross-attention, and the single-stream S3-DiT has no separate cross-attention step to hook. **zit-regions** is an SD Forge extension that adds a bias mask inside the joint self-attention: each region's caption tokens attend to their own pixels, and cross-region image attention is penalised. Its author reports it working on Z-Image-Turbo at 9 steps `[community — MisterLotto; single report, early]`. **ZIT-Ideogram** is a ComfyUI custom node that reuses the KJNodes Ideogram-4 box editor to draw rectangles and route a separate prompt and mask into each `[community — bbc-s; single report, early]`. The caveats reported across the two: results are seed-sensitive, region boundaries blur, per-region LoRA behaviour is inconsistent, and three or more regions are lightly tested. Treat them as a way to place two characters *before* the detailer passes, not as a replacement for them. If a region holds, run the per-face detailer on top. If it does not, fall back to the generic-figure method above. Re-verify both before building a pipeline on them `[flagged — early tooling; re-verify]`.

  **The training-side fix used elsewhere does not work here.** In August 2026 the community found that **Differential Output Preservation** — training each character LoRA against a class so several load together without bleeding — lets up to four character LoRAs coexist on Krea 2. The same technique was tried on **Z-Image Base and essentially failed to learn the character at all** `[community — MASilverHammer; single report, re-verify]`. Until someone finds a Z-Image configuration that takes, per-face detailer passes remain the answer here. A multi-character job is a genuine reason to render the scene on [`krea-2`](../../krea-2/) instead — see [`character-lora-training`](../../character-lora-training/references/dataset-and-captioning.md) for the technique itself.

---

## 6. Failure modes & fixes

| Symptom | Cause (mechanism) | Fix |
|---|---|---|
| Identity collapses at profiles/back views | Front-heavy dataset — the LoRA never learned those angles | Add 5+ targeted images of the failing angles (the edit-model factory makes this cheap); retrain |
| Every output is the same pose/framing ("same face" rigidity) | Near-duplicate training shots; too many steps | Cut steps or pick an earlier epoch; increase pose and shot-size variety; re-check the XY grid for the pre-rigid checkpoint |
| Character always wears the dataset's expression | Expression lock-in — the dominant expression was absorbed into the trigger | Add deliberate expression variety; caption expressions so they stay promptable |
| Character LoRA drags a style/color cast into every image | Dataset shared one look; style entangled with identity | Vary lighting, background, and medium across the set; prevention at dataset time is still the reliable lever. An inference-time option now exists: two community per-layer LoRA loaders for the 30-block S3-DiT let you dial down the blocks carrying the cast (`setup-and-workflows.md §6`). They give you the knob but not the map, since nobody has published which Z-Image blocks hold style versus identity, so expect to sweep. Both loaders are weeks old and unvalidated `[flagged — per-layer loaders early; re-verify]` |
| Two characters swap hair/clothing in one scene | Attribute bleed — one conditioning stream, no regional isolation | Per-face detailer passes with per-character LoRAs (§5); simplify to one distinguishing feature per character in the scene prompt |
| LoRA "barely does anything" | Not a training failure. On a ComfyUI older than March 2026, the diffusers-format QKV loading bug; on a current build, a hot weight or a Base/Turbo mismatch | Check the ComfyUI version floor, then the variant the LoRA was trained on and its card weight (`setup-and-workflows.md §6`) |

---

## Sources & confidence

The pipeline shape — edit-model factory, then LoRA, then detailer deploy — is **named-community craft, convergent across independent authors** (Mickmumpitz, WeirdWonderfulAI/Harmeet, MyAIForce, Civitai dataset guides). Per this skill's two-bar policy, it is stated with confidence. The Z-Image-specific hard facts (no PuLID/IP-Adapter, Z-Image-Edit unreleased, the training-adapter requirement) are verified against the repos and `setup-and-workflows.md §10`. The §3 factory-setup files and template wiring are official-bar, read from the Comfy-Org repos and the docs.comfy.org 2511 template on 2026-08-23. The fast-moving parts are: Qwen-Image-Edit versions and the factory filenames; Z-Image-Edit's eventual release, which would absorb much of the factory role natively, so re-check before building new pipelines; and the multi-character and per-layer LoRA tooling, which went from absent to experimental between June and September 2026 (checked 2026-09-09) and is the least settled thing in this file.
