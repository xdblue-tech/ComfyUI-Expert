# FLUX.2 — Prompting Guide

Source tier: BFL official prompting guide (primary), BFL blog, HF blog (official-via-docs), fal.ai prompting guide and community consensus (community).

---

## Contents

1. [Anatomy of a FLUX.2 prompt](#1-anatomy)
2. [What the encoder actually parses](#2-what-the-encoder-actually-parses)
3. [Hex color control](#3-hex-color-control)
4. [JSON for production (optional, not required)](#4-json-for-production-optional-not-required)
5. [Realism vocabulary: camera, lens, film stock](#5-realism-vocabulary-camera-lens-film-stock)
6. [Multi-reference image editing](#6-multi-reference-image-editing)
7. [Text-in-image guidance](#7-text-in-image-guidance)
8. [Drop-in prompt templates](#8-drop-in-prompt-templates)
9. [Common mistakes and corrections](#9-common-mistakes-and-corrections)

---

## 1. Anatomy

**The official BFL four-part structure** (from `docs.bfl.ml/guides/prompting_guide_flux2`):

```
[Subject] [Action/Setting] [Style] [Context]
```

| Part | What goes here | Example |
|---|---|---|
| **Subject** | Who or what the image shows. Be specific and concrete, and avoid generics like "person" or "scene". | `"A woman in her early 30s with silver-grey cropped hair and faint freckles"` |
| **Action / Setting** | What the subject is doing, and/or where. Environmental specifics belong here. | `"sits at a rain-wet counter in a narrow Kyoto kissaten, hands around a ceramic cup"` |
| **Style** | The photographic spec, artistic medium, or aesthetic genre. Camera gear lives here. | `"Hasselblad X2D, 55mm f/2.8, Fujifilm Pro 400H emulation, overcast afternoon diffuse"` |
| **Context** | Secondary elements: background details, atmosphere, secondary figures. | `"rain-blurred street visible through the window, steam rising from the coffee"` |

**Length sweet spots:**

| Use case | Recommended length |
|---|---|
| Concept sketch / iteration | 10–30 words |
| Standard generation | 30–80 words |
| Multi-element scene | 50–100 words |
| Multi-reference / structured JSON | Any length, but count subjects |
| Hard cap | 512 tokens (~380 words) |

Put the subject and key elements first. Mistral 3.2 and Qwen3 give more weight to earlier tokens in their attention mechanism, so the front of the prompt carries the most influence.

---

## 2. What the encoder actually parses

[dev] uses **Mistral Small 3.2 24B** (a 24-billion-parameter vision-language model). [klein] uses **Qwen3** (4B for [klein] 4B, 8B for [klein] 9B). Both are instruction-following LLMs. They belong to the same model families that are used for code generation and text summarisation.

**What they read well:**
- Grammatical clause structure (`"a woman who is holding"` > `"woman holding"`).
- Compositional prepositions and spatial language (`"in front of"`, `"partially hidden behind"`).
- Proper nouns for brands, places, films, periods (`"Kodak Portra 400"`, `"1970s Tokyo"`, `"Bauhaus-era poster"`).
- Colour descriptions in natural language (`"deep indigo"`, `"burnt-sienna matte"`).
- Specific names for known cultural artefacts (`"a Leica M-A body"`, `"Rolleiflex twin-lens"`).
- Hex color codes when signalled correctly (see section 3).

**What they parse as noise:**
- Quality-adjective chains: `masterpiece`, `8k`, `best quality`, `ultra-realistic`, `highly detailed`, `photorealistic`, `trending on artstation`. These are Stable Diffusion 1.5 booru tags, and they carry almost no meaning for an LLM-class encoder.
- Comma-separated tokens with no predicate: `woman, hair, eyes, blue, rain, city, night`. A list like this describes a pile of nouns, not a scene.
- Parenthetical emphasis tokens: `(masterpiece:1.2)`, `((eyes))`. This is AUTOMATIC1111 syntax, and it means nothing here.
- Negative prompts. They cannot work because there is no CFG path in [dev] and no active guidance in [klein] distilled (see section 9).

---

## 3. Hex color control

FLUX.2 introduced explicit hex-color conditioning, which Flux.1 did not have. The mechanism needs a keyword trigger before the hex code, because that trigger is what tells the encoder to route the code to the color-conditioning pathway.

**Format: `"...in color #XXXXXX"` or `"...in hex #XXXXXX"`**

| Correct | Incorrect |
|---|---|
| `"An apple in color #0047AB"` | `"An apple #0047AB"` |
| `"Logo text 'ACME' in color #FF5733"` | `"bright red #FF5733 logo"` |
| `"Background in hex #1A1A2E, foreground in color #E94560"` | `"#1A1A2E background with #E94560 text"` |

Multiple hex codes work in one prompt. Each needs its own `"in color"` or `"in hex"` signal.

**Use cases:**
- Brand colour consistency across a campaign: define a `color_palette` array in the JSON format (section 4), then reference elements with hex codes inline.
- Logo and type-on-image generation: pair hex color conditioning with the JSON `subjects` format, and wrap text in quotes.
- Replicating a film still's specific colour grading: hex-code the primary midtone and shadow colour.

---

## 4. JSON for production (optional, not required)

FLUX.2 was trained on natural language, so JSON is never required. It is a workflow tool for situations where:
- You need multi-subject scenes with explicit positional control.
- You want machine-readable, version-controllable prompts in an API pipeline.
- You need strict colour consistency across a batch.

**Official BFL schema:**

```json
{
  "scene": "brief one-sentence description of the overall scene",
  "subjects": [
    {
      "description": "detailed description of this subject",
      "position": "where in the frame: center / left / right / foreground / background / lower-left / etc."
    }
  ],
  "style": "photographic specification or artistic style",
  "color_palette": ["#hex1", "#hex2", "#hex3"],
  "lighting": "light source, direction, quality, colour temperature",
  "camera": {
    "angle": "eye level / bird's eye / low angle / dutch / ...",
    "lens": "focal length and aperture: e.g. 85mm f/1.4"
  }
}
```

**Rules:**
- Each distinct subject needs its own entry in `subjects[]`.
- `position` is a freeform string. The model reads it semantically, not as pixel coordinates.
- `color_palette` hex codes compose with the inline `"in color #XXXXXX"` syntax.
- The JSON can be sent directly as the prompt string in ComfyUI or the API, with no special wrapper.

**When to use JSON vs natural language:**

| Situation | Best format |
|---|---|
| Single subject, photoreal | Natural language sentence |
| Quick concept iteration | Short natural language |
| Multi-subject compositional scene | JSON with `subjects[]` + `position` |
| Brand asset pipeline (batch) | JSON with `color_palette` |
| Text-heavy layout with specific placement | JSON, each text block as a subject |

---

## 5. Realism vocabulary: camera, lens, film stock

By default, FLUX.2 tends toward over-processed sharpness, especially in [klein]. Camera vocabulary counteracts this because Mistral and Qwen3 have strong priors about what images each setup produces. These terms are not just keywords; they activate the encoder's world model.

### Camera bodies (photoreal signal strength: high) `[community — fal.ai prompting guide; convergent]`

| Camera | Typical use / signal |
|---|---|
| Hasselblad X2D | Medium-format, rich tonal range, ultra-fine detail, studio/landscape |
| Leica M-A / M11 | Street, reportage, subtle grain, clinical sharpness |
| Sony A7R V | Editorial, fashion, fine skin gradation |
| Canon EOS R5 / R3 | News, sports, editorial, full-frame authority |
| Nikon Z9 | Documentary, photojournalism |
| Fujifilm GFX 100S | Medium-format landscape, fashion |
| Ricoh GR IIIx | Street, 40mm equivalent, compact, film-noir undertone |
| Mamiya RB67 | Vintage medium-format, square format, smooth bokeh |

### Lenses (signal: aperture controls background separation) `[community — fal.ai prompting guide; convergent]`

| Spec | What it reads as |
|---|---|
| 85mm f/1.4 | Portrait classic, creamy background separation |
| 35mm f/1.8 | Environmental, slight context, still background-separated |
| 50mm f/2 | "Normal" field of view, documentary |
| 80mm f/2.8 | Medium-format standard, studio quality |
| 135mm f/2 | Compressed background, telephoto compression |
| 24mm f/2.8 | Wide environmental portrait, architecture |

### Film stocks (signal: tonal and colour character) `[community — fal.ai prompting guide; convergent]`

| Stock | Character |
|---|---|
| Kodak Portra 400 | Warm skin tones, creamy shadows, fine grain, best for people |
| Kodak Ektar 100 | Vivid saturated colours, fine grain, best for landscapes/objects |
| Fujifilm Pro 400H | Cool midtones, subtle blues, even skin, editorial |
| Fujifilm Velvia 50 | High saturation, punchy, landscape slide film |
| Ilford HP5 | B&W, medium contrast, versatile, reportage |
| Kodak Tri-X 400 | B&W, gritty grain, street photography |
| CineStill 800T | Cinematic halation, tungsten colour shift, night scenes |

### Lighting vocabulary `[community — fal.ai prompting guide; convergent]`

| Term | Effect |
|---|---|
| "available light" / "window light" | Natural, uncontrolled, believable |
| "overcast diffuse" | Even lighting, no harsh shadows, editorial |
| "golden hour back-light" | Warm rim light, slightly flared, lifestyle |
| "mixed tungsten and fluorescent" | Colour-cast authenticity, interior environment |
| "harsh midday sun" | High contrast, deep shadows, documentary |
| "practical lights only" | Sources visible in frame, moody, cinematic |
| "softbox / octabox" | Studio, even skin, fashion |

### Breaking the "over-AI'd" look

Add one or two of the following:
- One non-idealised human feature: "slight under-eye shadow", "visible pores on cheeks", "fine hairline scar above left eyebrow".
- Environment imperfections: "dust on the window", "slightly chipped enamel mug", "worn leather shoulder bag".
- Motion / life: "exhaled breath visible in cold air", "hair caught by a gust", "jacket wrinkled from the bus ride".

Do **not** add `"realistic"`, `"ultra-realistic"`, `"photorealistic"`, `"8K"`, or `"high quality"`. These are booru tokens that carry zero signal.

---

## 6. Multi-reference image editing

FLUX.2 [dev] and [klein] 9B KV support **reference-based image editing**. You provide one or more reference images, and the model integrates them into the generated composition. In ComfyUI this capability uses `ReferenceLatent` nodes, which are FLUX.2-native. It does not use the older IPAdapter approach.

**Supported reference count:** the sources disagree. Marketing documentation states up to 10, the official prompting guide states up to 8, and both 4B and 9B documentation mention approximately 4 in some configurations. Use the lower bounds as safe targets until you verify your specific variant's model card.

**Suggested supported counts by variant (verify at time of use):**
- [dev]: up to ~8–10 reference images.
- [klein] 9B: up to ~8 (9B KV adds KV-caching to speed repeated reference processing).
- [klein] 4B: up to ~4 (marketing may overstate this for 4B specifically).

### ComfyUI reference workflow (dev image-edit template)

The `image_flux2_image_editing.json` template introduces:
- The `ReferenceLatent` node takes a reference image plus an optional mask, and outputs a latent reference.
- Multiple `ReferenceLatent` nodes chain into the conditioning.
- Reference images encode separately from the main latent, which is why 9B KV's KV-caching speeds up repeated reference processing.

The full image-edit template notes and node graph are in **`setup-and-workflows.md`**.

### Prompting for multi-reference

When you use reference images, describe each reference's role explicitly:
- `"[Subject from reference 1] wearing the jacket shown in reference 2, standing in front of the building from reference 3"`.
- The model reads semantic labels in the prompt and maps them to the provided reference images in order.
- Positional language such as `"first reference"` or `"the jacket image"` helps the model disambiguate.

---

## 7. Text-in-image guidance

FLUX.2 improved text rendering significantly over Flux.1. It is suitable for short text overlays (1–5 words), typographic accents, decorative scripts, and sign text in scenes. It is not suitable for multi-line body copy, dense typographic layouts, or logos with complex internal geometry.

**Rules that improve success:**
1. Wrap the exact text in `'single quotes'` or `"double quotes"` in the prompt: `"A shop sign reading 'CLOSED'"`.
2. Keep each text block to ≤10 words; shorter is more reliable.
3. Specify typography style: `"hand-lettered in white chalk"`, `"serif metal engraving"`, `"neon tube lettering in color #FF3300"`.
4. Use JSON `subjects` to position text blocks precisely:
   ```json
   {"description": "Sign reading 'WELCOME' in brushed copper lettering", "position": "upper-center on the door frame"}
   ```
5. Combine hex color control for text colour: `"Sign text 'ACME' in color #FF5733"`.
6. Generate 3–5 candidates and select the best, because text rendering has higher variance than scene composition.

For anything heavier, such as labels on product packaging, business cards, poster layouts, or UI mockups, FLUX.2 text rendering is not reliable. Dense multi-line copy and layout-driven design need a model purpose-built for typography instead.

---

## 8. Drop-in prompt templates

### Portrait — photoreal

```
A [age + specific descriptor] [subject noun] [specific action] in [specific environment]. 
[Camera body], [focal length] [aperture], [film stock or lighting qualifier], [time of day or atmosphere].
[One non-idealised physical detail].
```

Example:
```
A man in his late 40s with grey-streaked temples and a weathered face adjusts fishing line at dawn on a deserted pier.
Leica M11, 35mm f/2, Kodak Portra 400, cool pre-dawn fog.
Slight purple under-eye shadow, collar of a worn flannel shirt.
```

### Product still-life

```
[Product with specific descriptor] on [precise surface and context].
Shot overhead with [camera body], [lens], [lighting description].
[Color palette or mood]. [Background detail].
```

Example:
```
A ceramic espresso cup with a hairline crack on the handle rests on a weathered cypress-wood counter, surrounded by scattered coffee grounds.
Shot overhead with a Hasselblad X2D, 80mm f/4, soft north-facing window light.
Muted greys and cream, slightly underexposed.
```

### Architecture / exterior

```
[Building or structure with period/style/location], [time of day], [weather].
Shot with [camera body], [focal length] [aperture].
[Specific lighting condition]. [Human or life element to give scale/life].
```

Example:
```
A 1960s Tokyo shotgun house sandwiched between two glass towers, early evening, light rain.
Shot with a Ricoh GR IIIx, 28mm f/2.8.
Tungsten glow from the interior windows against blue dusk. A bicycle leaning against the entrance gate.
```

### Commercial / brand (JSON format)

```json
{
  "scene": "A lifestyle product shot of a skincare serum bottle",
  "subjects": [
    {"description": "A 30ml amber glass dropper bottle with the label 'AURIC BOTANICS' in minimal sans-serif type", "position": "center-foreground, slightly right of center"},
    {"description": "Three dried botanical stems and two smooth white river stones", "position": "left-foreground, partially behind the bottle"}
  ],
  "style": "editorial beauty photography, clean and minimal",
  "color_palette": ["#F5F0E8", "#2C3E2D", "#C8A96E"],
  "lighting": "single large soft-box from upper left, slight catch-light on the bottle",
  "camera": {"angle": "eye level, slight downward tilt", "lens": "100mm f/2.8 macro"}
}
```

### Multi-reference editing

```
[Subject from reference 1] wearing [clothing/item from reference 2], standing in [environment from reference 3].
[Camera spec]. [Lighting]. [Atmosphere].
```

Example:
```
The woman from the first reference wearing the deep indigo coat shown in the second reference, standing at a rain-blurred intersection at night.
Sony A7R V, 50mm f/1.8, available street lighting, puddles on the pavement.
```

---

## 9. Common mistakes and corrections

| Mistake | Why it's wrong | Correct |
|---|---|---|
| `"photorealistic, 8K, ultra detailed"` | These are SD1.5-era booru tokens, and they carry near-zero signal for Mistral/Qwen3 | Use a camera body + lens + film stock instead |
| Negative prompt filled in | [dev] has no CFG path, and [klein] distilled runs with guidance off | Phrase constraints positively in the main prompt |
| CFG=0 in KSampler | A value of 0.0 outputs the unconditional result, which ignores the prompt entirely | Use **1** for guidance-off in [klein] distilled |
| `"The image should show…"` opening | This is reflexive meta-language. The encoder interprets it literally and may generate an image of an image | Open directly with the subject: `"A woman…"` |
| Bare hex code without keyword | `"...#0047AB"` routes as text noise and does not trigger color conditioning | `"...in color #0047AB"` |
| Subject buried mid-prompt | LLM encoders give more weight to early tokens | Put the subject in the first clause |
| Treating JSON as required ("plain text fails") | FLUX.2 was trained on natural language. JSON is optional, and is a workflow tool | Use plain language for single subjects, and JSON for complex multi-subject scenes |
| Dropping camera gear to avoid the "AI look" | Some models benefit from removing DSLR markers. FLUX.2 is the opposite, because Mistral/Qwen3 treat camera vocabulary as semantic context | Keep the camera body + lens + film stock in FLUX.2 prompts for photoreal results |
| `(text:1.5)` parenthetical weight syntax | This is AUTOMATIC1111 syntax, and it means nothing here: FLUX.2's Mistral/Qwen3 conditioning has no per-token weight channel. This is not a rule about LLM encoders as a class, since [`anima`](../../anima/) has one and ships officially documented weighting | Do not use parenthetical weighting; rephrase the sentence instead |
