# Qwen-Image — Prompting guide

This file owns the prompt: the dialect the Qwen2.5-VL 7B encoder wants, the Edit instruction rules the community converged on, the multi-image format the model was trained on, the official rewriter's rules, and text rendering in both languages. It does not own settings (SKILL.md) or training captions (`lora-training.md §4`). Official rules are read from `QwenLM/Qwen-Image/src/examples/tools/prompt_utils.py` and the model cards. Community craft is attributed inline. Verified 2026-09-09; §9 (2.1) from the card, templates, the HF Space `cases.json` and the PE `system_prompt.txt` files, 2026-09-20.

## Contents

1. The dialect — a sentence to a vision-language model
2. Edit instructions — four rules from the sixteen-way A/B
3. Multi-image — `Picture 1:` is a trained format
4. Negatives, "positive magic" and the aspect table
5. The official rewriter and enhancer — what they enforce
6. Text rendering, English and Chinese
7. Common mistakes
8. Drop-in templates
9. Qwen-Image-2.1 — `<image1>` tags, the RGBA wrapper, local-edit grammar, and the PE rewriters

---

## 1. The dialect — a sentence to a vision-language model

Qwen-Image conditions on **Qwen2.5-VL 7B**, a vision-language model, not CLIP or T5. It parses clause structure and word order, so the prompt is a sentence, never tag soup. The one contributor who answered the official request for a prompt guide on the 2511 repo said the model "uses a llm as clip" and is "a LOT more flexible than all the various guides suggest" `[community — Andyx1976, HF Edit-2511 discussion #7]`.

Two lengths apply, and they differ by variant. **T2I tolerates long prompts.** The official rewriter targets under 200 words, as a paragraph of subject, style, spatial relationships and shot composition (§5). **Edit wants short prompts.** Long, careful instructions hurt; the rule is "pretend you're talking to a child" (§2).

The encoder class also settles triggers and captions: a LoRA trigger is a plain name inside a sentence, not a rare token (`lora-training.md §4`). And the encoder cannot refuse. It is consumed as hidden states, not run as a chat model, so a prompt that comes back vague is the diffusion weights declining to draw (`lora-training.md §7`).

**Prompt anatomy for T2I**, in the order the official rewriter assembles it:

| Part | What to include | Why |
|---|---|---|
| **Subject** | who or what, with concrete characteristics | the rewriter's first job is "subject characteristics" |
| **Text to render** | the exact words in straight double quotes, plus position and style | the rewriter enforces quotes and position (§6) |
| **Style / medium** | one precise, niche style; the official default is realistic photography | the rewriter picks "a precise, niche style" when none is given |
| **Spatial relationships** | where things sit relative to each other | the rewriter refines these explicitly |
| **Shot composition** | framing, lens, angle | same |
| **Lighting** | source, direction, quality, colour | the highest-leverage variable on every LLM-encoder model |

Keep one medium and one lighting scheme per prompt. Contradictions become uncanny blends, not averages.

---

## 2. Edit instructions — four rules from the sixteen-way A/B

The most detailed reproducible craft document on this family is one author's 2511 write-up, the sequel to the 2509 one the community treated as canonical. Everything in this section is that source `[community — nsfwVariant, r/StableDiffusion 1tqm8ic; strong]`.

**1. Simple and direct.** Say what should happen, in one clause.

| Bad | Good |
|---|---|
| `Place a red apple on the table, ensuring it's in the center and removing the plate that was in the same spot.` | `Replace the middle plate with a red apple.` |

**2. Name what must not change.** Unmentioned attributes are treated as free. `Change her outfit to a bikini top and short shorts.` often changes the pose too. Adding `Leave her robot arm and pose unchanged.` makes it correct "99% of the time". This is the highest-value habit on Edit, and the official enhancer bakes the same rule in for people (§5).

**3. `relight` is a keyword with a grammar, not a description.** The shape is `Relight to <strength> <colour> <direction>.`:

```
Relight to white diffuse.
Relight to warm backlit.
Relight to bright cool frontlit.
```

It can be the whole prompt. 2511 folded the community relight LoRA into the base, which is why the keyword works without an adapter.

**4. Framing is promptable.** Zoom is how you build a reference set at several scales (`characters.md §3`):

```
Zoom in on the person's upper body. The composition should frame their head and thighs.
Zoom out to a full body shot.
Zoom in for a close up portrait.
```

A pixel-perfect edit prompt from a second author mixes subject reference, scene and grade in three short clauses `[community — danamir_, r/StableDiffusion 1o01e6i]`:

```
The blonde girl from image 1 in a dark forest under a thunderstorm, a tornado in the distance,
heavy rain in front. Change the overall lighting to dark blue tint. Bright backlight.
```

Two operational notes. The Edit upscale LoRA wants `"Enhance image quality"` plus a scene description, and "the more descriptive it is, the better the upscale effect" `[community — vafipas663]`. And Edit used as a restorer of a degraded source only engages if the prompt also asks for a semantic change, such as a background change `[community — Agile-Role-1042; single report]`.

---

## 3. Multi-image — `Picture 1:` is a trained format

ComfyUI's `TextEncodeQwenImageEditPlus` runs the VL model over each input and splices the descriptions into the conditioning as literally `Picture 1: <desc>  Picture 2: <desc>`. If you bypass that node (`setup-and-workflows.md §5`), you write the labels yourself in that exact shape, because "Qwedit was trained on this exact format". **Shorter labels beat richer ones**: "a 5 word description wins over whatever BS the VL model spews out, every time" `[community — nsfwVariant]`.

```
Picture 1: a man wearing a t-shirt. Picture 2: a top hat.
Make the man in Picture 1 wear the top hat from Picture 2.

Picture 1: a living room. Picture 2: a woman.
Put the woman from Picture 2 into the living room in Picture 1.
```

A complementary 2509-era convention **names** the subjects and refers to them by name `[community — goddess_peeler, r/StableDiffusion 1o1zsny]`:

```
Jane is in image1.  Forrest is in image2.  Bonzo is in image3.
Jane sits next to Forrest.  Bonzo sits on the ground in front of them.
All other details from image2 remain unchanged.
```

Both work. The `Picture N:` form has the stronger justification, because it is what the model's own pipeline emits. The trained range is **1–3 input images**, and the trained combinations are person + person, person + product, person + scene `[official — Edit-2509 card]`. Clothing transfer is the hardest of these. "All models failed… you have to expect multiple attempts", and the working hypothesis is mismatched aspect ratio and crop between the inputs `[community — FluffyQuack]`. That agrees with the independent advice to make inputs share at least one edge dimension.

---

## 4. Negatives, "positive magic" and the aspect table

**Negatives** are live on the base regime (CFG > 1) and inert under Lightning (CFG 1.0); the mechanism is SKILL.md's one rule. Every card except 2512 passes `negative_prompt=" "`, a single space. **Qwen-Image-2512 ships a real negative**, in Chinese, in both its card and its ComfyUI template:

```
低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。
```

That reads: low resolution, low quality, deformed limbs, deformed fingers, over-saturated, waxy look, faceless detail, over-smoothed, AI-looking; chaotic composition; blurry, distorted text. Note what Qwen chose to negate on its realism refresh: waxy, over-smoothed, AI look. That is the plastic default, named by its own authors.

**"Positive magic."** `prompt_utils.py` and the base card append a fixed suffix:

```python
positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图."
}
```

The same constants survive into `prompt_utils_2512.py`, and most ComfyUI templates and Civitai workflow JSONs carry it, so many users run it unknowingly. Realism-LoRA authors' published settings do not reproduce it `[community — FortranUA, r/StableDiffusion 1odsid9]`. Nobody has A/B'd it. "Cinematic composition" is a real style instruction to a VL encoder, biasing toward shallow depth of field and warm grading; that objection is inference, not a sourced result `[contested]`. Decide, and do not inherit it from a template.

**Aspect table**, verbatim from the cards. Every size is in the ~1.5 MP "1328 class", which training buckets should also respect (`lora-training.md §3`):

| Ratio | Qwen-Image (2025-08) | Qwen-Image-2512 |
|---|---|---|
| 1:1 | 1328 × 1328 | 1328 × 1328 |
| 16:9 | 1664 × 928 | 1664 × 928 |
| 9:16 | 928 × 1664 | 928 × 1664 |
| 4:3 | 1472 × 1140 | **1472 × 1104** |
| 3:4 | 1140 × 1472 | **1104 × 1472** |
| 3:2 | 1584 × 1056 | 1584 × 1056 |
| 2:3 | 1056 × 1584 | 1056 × 1584 |

1472 × 1104 is exactly 4:3 and 1472 × 1140 is not, so the earlier figure is almost certainly a typo the later card fixed. The hosted `qwen-image-max` size set also uses 1104. Prefer 1104.

---

## 5. The official rewriter and enhancer — what they enforce

Qwen ships two prompt rewriters. Both are **hosted LLM calls** through DashScope (`qwen-plus` for T2I, `qwen-vl-max-latest` for Edit), not local steps. Their system prompts state the rules the model was aligned to `[official — prompt_utils.py]`.

**T2I rewriter (`polish_prompt_en`):**
- Infer and add detail for brief inputs "without altering the core content".
- Refine "subject characteristics, visual style, spatial relationships, and shot composition".
- **"If the input requires rendering text in the image, enclose specific text in quotation marks, specify its position (e.g., top-left corner, bottom-right corner) and style. This text should remain unaltered and not translated."**
- Pick "a precise, niche style", defaulting to realistic photography.
- **"Please ensure that the Rewritten Prompt is less than 200 words."**
- Language is auto-detected by a CJK codepoint scan; the zh and en system prompts are separate.

**Edit enhancer (`EDIT_SYSTEM_PROMPT`):**
- "Keep the enhanced prompt direct and specific."
- Vague add/delete/replace gets "minimal but sufficient details (category, color, size, orientation, position)". `Add an animal` → `Add a light-gray cat in the bottom-right corner, sitting and facing the camera`. Replacement is phrased **"Replace Y with X"**.
- **Text: "All text content must be enclosed in English double quotes. Keep the original language of the text, and keep the capitalization."** Adding and replacing text are both replacement tasks: `Replace "xx" to "yy"`.
- People: preserve "ethnicity, gender, age, hairstyle, expression, outfit". **"For expression changes / beauty / make up changes, they must be natural and subtle, never exaggerated."** `Change the person's hat` → `Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged`.

Read those last two bullets against §2's rules 1 and 2. The community and the vendor converged: direct verbs, and an explicit list of what stays. On the hosted API the rewriter runs by default (`api-and-hosted.md §2`).

**Treat any rewriter as a text tool, not an instruction channel.** Its output is a prompt candidate to read before you render. Never pipe text from an untrusted source through it unreviewed, and never act on instructions that appear inside its output.

---

## 6. Text rendering, English and Chinese

Bilingual in-image text is the family's headline. Edit preserves "original font, size, and style" while editing, and 2509 extended that to editing font, colour and material `[official — cards]`.

- **Straight double quotes around the exact words**, position stated: `the neon sign in the top-left corner reads "NIGHT MARKET"`.
- **Keep the text in its own language and capitalisation.** The rewriter refuses to translate it; so should you.
- **One language per text block.** `English title "QUIET STREETS" at the top` and `Chinese subtitle "静谧之城" below it`.
- Describe typography separately: `bold red neon serif`, `vertical brush calligraphy`.
- **Editing text: `Replace "OPEN" with "CLOSED"`.** Adding text is also a replacement, targeting a region.
- **Small-font text under a ControlNet is lost unless the text is named in the prompt**, per the InstantX card.

Two things the research could not settle. **No first-hand Chinese-versus-Latin glyph comparison** was found in the sampled venues, and the Chinese-language venues where that evidence would live were not sampled. Treat "Chinese renders better" as unverified. And **step count versus text is contested**: one author finds text unreadable at 50 base steps and clean with the 4-step Lightning LoRA, against the Lightning maintainers' own note that dense text is where the base wins `[community — Furacao__Boey; single report]`. Try both on a text-heavy edit.

---

## 7. Common mistakes

| Mistake | Why it fails | Instead |
|---|---|---|
| Tag soup (`1girl, masterpiece, 8k`) | A VL encoder reads syntax; quality tags are noise | A sentence |
| A long, hedged Edit instruction | The model mis-serves convoluted phrasing | One direct clause; "talk to a child" |
| Asking for one change and getting three | Unmentioned attributes are free | Name what must not change |
| Describing the light instead of relighting | Description is weaker than the trained keyword | `Relight to <strength> <colour> <direction>.` |
| Letting the node write `Picture N:` | The VL model's paragraph is worse than five words | Bypass and write short labels |
| Text without quotes, or translated | The alignment rules expect quotes, position, original language | `"exact words"`, position, own language |
| Negative prompt under Lightning | CFG 1.0 never samples the unconditional branch | Phrase positively, or run the base regime |
| Inheriting `Ultra HD, 4K, cinematic composition` | A style instruction you did not choose | Decide per project; realism authors drop it |
| "photorealistic" | Pulls toward the artwork the base saw | "photo" / "photograph" `[community — Ashen3; Krea 2 evidence]` |
| Chaining edits | Colour and identity drift per edit | Re-anchor on the original reference |
| Two inputs with different crops for a transfer | The concatenation-trained path is framing-sensitive | Share an edge dimension; expect retries |

---

## 8. Drop-in templates

### A. T2I, realism with rendered text
> Editorial photograph of a small ramen stall on a narrow back alley in Shibuya at midnight, viewed from across the street. A red noren curtain over the entrance reads "夜市" in bold white brush calligraphy; a smaller pink neon sign in the window reads "OPEN". Steam rising from the open kitchen, one customer hunched over a bowl at the counter, back to camera. Wet pavement reflecting the signage. Available neon and warm interior light, 35 mm lens, slight grain.

### B. Edit, single reference, outfit change
> Change her outfit to a charcoal wool coat over a white shirt. Leave her face, pose, hair and the background unchanged.

### C. Edit, relight only
> Relight to warm backlit.

### D. Edit, two references, product placement
> Picture 1: a wooden desk by a window. Picture 2: a green ceramic mug.
> Put the mug from Picture 2 on the desk in Picture 1, front left, with the window light falling on it. Leave everything else in Picture 1 unchanged.

### E. Edit, zoom for a reference set
> Zoom in on the person's upper body. The composition should frame their head and thighs. Leave their face, expression and clothing unchanged.

### F. Edit, text replacement
> Replace "OPEN" with "CLOSED" on the sign. Keep the font, colour and size.

---

## 9. Qwen-Image-2.1 — `<image1>` tags, the RGBA wrapper, local-edit grammar, and the PE rewriters

`[official — HF card, Comfy templates, `Qwen/Qwen-Image-2.1` Space `examples/cases.json` (26 official cases), PE-T2I / PE-I2I `system_prompt.txt`, 2026-09-20]`. The encoder is **Qwen3-VL-8B**, not Qwen2.5-VL, so §1's dialect notes are inherited by analogy, not evidence. Nothing in §2's Edit rules (short prompts, "talk to a child") has been tested on 2.1; the official demo prompts are, if anything, long.

### 9.1 References are `<image1>` … `<image10>`

The trained format is an angle-bracket tag per slot, four tokens each, inserted by the pipeline as `<image1><|vision_start|>…<|vision_end|>` before your text. **Not `Picture 1:`.** `image_1` is the edit target; the rest are references; order is read order (block-causal attention lets later blocks see earlier ones). Chinese demo prompts use `【图1】` interchangeably.

> Keep the character and pose in `<image1>` unchanged, put this light blue denim shirt from `<image2>` on the character, preserve the original facial features, hair, body shape and pose, the denim shirt fits naturally on body, realistic denim fabric texture, natural clothing folds, keep the original background and original lighting. `[official — edit template]`

> Using the six people in `<image1>`, `<image2>`, `<image3>`, `<image4>`, `<image5>` and `<image6>` as identity references, generate a brand-new vertical group portrait in the style of a 1980s sitcom promotional photo … `[official — demo case, 6 inputs]`

The name-what-must-not-change rule from §2 carries over verbatim in every official example ("preserve the original facial features … keep the original background and original lighting").

### 9.2 The RGBA wrapper

Transparency is decided by the prompt; there is no flag. The card's recommended form, verbatim:

> This is an RGBA image with transparency. `⟨your description⟩`. The image has alpha channel and the background is transparent.

Chinese equivalent from the demo: `这是一张带有透明度的RGBA格式图像，… 该图像具有alpha通道，背景是透明的。` DiffSynth adds that "isolated on a fully transparent background", "alpha matte" and "die-cut sticker" also trigger it, and gives the two craft rules that exist: **do not describe environment or ambient light** ("underwater", "indoor", "light streaming through") or the model fills the canvas — recast atmosphere as properties of the subject and add "no background elements besides the subject"; and for hair, ribbons, water and glow, which produce wide mid-value alpha, add **"clear silhouette, clean edges"** if you need a hard cut. Save PNG; `.convert("RGB")` or JPEG drops the channel.

Background removal is an edit with the same mechanism: `"Remove the background, and output a PNG image"` (template). Extraction: "extract the ⟨subject⟩ as a transparent RGBA layer".

### 9.3 Local edits — three grammars

1. **Drawn circles**, several colours at once, plus an instruction that the marks are not content: "Remove the brown-leather metal watch in the **blue circle** and fill the wrist with skin matching the arm; change the boy's blond hair in the **red circle** to black …; replace the clothing in the two **green circles** with grey short-sleeved linen pyjamas …; **the blue, red and green annotation lines must not be rendered in the image.**" `[official — demo case, translated]`
2. **Painted region**: "In the **white-masked area on the right** add a scuba diver in a black wetsuit, mask and back-mounted tank, hovering slightly tilted, facing the camera, a stream of bubbles rising from the regulator."
3. **Original + separate mask as two inputs** (keeps the original unobscured): image_1 the photo, image_2 the mask, prompt "At the **circled place** add a mounted cowboy: brown wide-brim hat, thick beard, face turned left, brown canvas jacket over a dark-blue denim shirt …"

All three describe the *inserted* content in full detail and the *region* by its marker only.

### 9.4 What the official rewriters enforce (PE-T2I, PE-I2I)

Two Qwen3.5-VL 9B fine-tunes, thinking mode on, output `{"rewritten_prompt", "wh_ratio"}` (edits add `"ratio_follow": "<image1>"` when the output should inherit an input's aspect). Their system prompts are the best prompt-writing guidance Qwen has published for any generation:

**T2I** — one long English paragraph "describing the finished image as if you were looking at it"; never address the renderer. Opening sentence ≈ 20 words: `The image is a ⟨vertical/wide/square⟩ ⟨style⟩ ⟨photograph | poster | illustration | infographic | …⟩ of ⟨subject⟩, ⟨background and palette⟩`; the medium noun is never omitted. Then background first, then the top band, then body left → centre → right, then the bottom band; a single subject is walked background → pose → head → body → held objects → edges. **8–14 positional phrases** ("in the upper-left corner", "across the lower third") reaching corners and edges, about a third of sentences opening on one. Every legible string in straight double quotes, in its own script, with weight, colour, case and relative size; a line break is "a second line", never a newline; unreadable text is "blurred / indistinct", never invented; about a third of images should have **no** text and inventing signage is a named mistake. **Never write a ratio, resolution or pixel count into the prompt** — it goes in `wh_ratio` (defaults 3:2 landscape, 2:3 portrait). Job instructions ("4K", "sharp text", "use double quotes") are obeyed silently and never echoed. A three-word brief and a three-hundred-word brief both become a description of the same size.

**Edit** — "edit exactly the attribute(s) the user named, push each to a strong and unmistakable degree, and hold everything else at input fidelity"; the two symmetric failures are *leakage* (touching the unnamed) and *under-editing*. Name what stays **by type, position and role, not appearance** — "a preservation description reads to the model as a generation instruction: the more concretely you describe something you meant to keep, the more likely it drifts." Prefer one blanket preservation clause over walking the frame. Identity: "point at that image rather than describing features in words — verbal descriptions make the model regenerate and degrade the likeness." Rendering medium (photo, anime, sketch) survives every edit unless targeted. Rendered-text language: user-specified → the image's dominant existing text language → the instruction's language; monolingual inside quotes; genre never switches labels to English.

Everything in §9.4 is what the rewriters *produce*; whether hand-writing to the same rubric beats a short instruction on 2.1 is untested.
