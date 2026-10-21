# Krea 2 — API & hosted surfaces

This file covers the hosted side of Krea 2. That means Krea's own API and web app, which serve Medium and Large (the variants you cannot download), fal's endpoints for the open models, and the ComfyUI partner nodes. Every figure in this file comes from Krea's developer docs and fal's pages, verified 2026-07-06/07 `[official — Krea developer docs, fal pages]`. Community additions are marked where they appear. **Pricing and the 1K resolution cap are exactly the kind of fact that moves. Re-verify both before quoting them in a budget.**

## Contents
1. [Hosted vs open — what's actually different](#1-hosted-vs-open--whats-actually-different)
2. [The Krea API](#2-the-krea-api)
3. [The web app](#3-the-web-app)
4. [ComfyUI partner nodes](#4-comfyui-partner-nodes)
5. [fal (and other hosts) for the open models](#5-fal-and-other-hosts-for-the-open-models)

---

## 1. Hosted vs open — what's actually different

| | Open (Raw/Turbo) | Hosted (Medium/Large) |
|---|---|---|
| Weights | downloadable | no |
| VAE | [Qwen-Image](../../qwen-image/) VAE | **Large trained with the FLUX.2 VAE** `[official — team statement, HN]` — this is a real fidelity difference, not just a scale difference |
| Resolution | 1K (Raw) / 1–2K (Turbo) | **1K only via API, currently** `[official — API docs]` |
| Style references / moodboards | not in the open stack (style LoRAs instead) | yes — the flagship feature |
| Creativity dial | no (analogue: enhancer on/off) | raw / low / medium / high |
| Sliders (intensity/complexity/movement) | no | −100…100 each |
| Cost | your GPU | $0.030–0.070/img (§2) |
| Licence | Community License (revenue-gated) | Krea ToS; you own outputs |

No independent hosted-vs-open quality shootout has been published yet. Krea's own positioning ("richest output", photorealism) suggests Large beats Turbo on fidelity, so assume that it does. But treat the size of the gap as unknown.

## 2. The Krea API

The base URL is `https://api.krea.ai`, and the docs live at krea.ai/docs/developers. The API uses an async job pattern: your `POST` returns a `job_id`, and then you either poll `GET /jobs/{job_id}` or register a webhook.

**Endpoints:** `POST /generate/image/krea/krea-2/medium` and `…/krea-2/large`.

**Parameters:**

| Param | Values | Notes |
|---|---|---|
| `prompt` | string | same prompting doctrine as local (`prompting-guide.md`): content goes in the prompt, style goes in the controls below |
| `aspect_ratio` | `1:1, 4:3, 3:2, 16:9, 2.35:1, 4:5, 2:3, 9:16` | |
| `resolution` | `1K` | "1K only currently" |
| `creativity` | `raw / low / medium / high` | default `medium`; `raw` renders only what's written, with no expansion |
| `seed` | int | |
| `image_style_references` | up to **10**, each with `strength` | style conditioning, not identity |
| `moodboards` | max **1** (moodboard ID) | created in the app |
| `styles` | up to 10 preset/LoRA style IDs | |
| `intensity`, `complexity`, `movement` | −100…100, default 0 | global aesthetic sliders |

**Pricing** `[official — Krea pricing page, 2026-07-06; re-verify]`:

| | base | + style refs | + moodboards |
|---|---|---|---|
| Medium | $0.030 | $0.035 | $0.040 |
| Large | $0.060 | $0.065 | $0.070 |

## 3. The web app

The krea.ai image generator lets you pick **Medium / Large / Turbo**. It offers up to **4 style references, each with a strength slider**, moodboards ("the most precise way to set a visual direction"), batches up to 4, and 1K output. The app is also where you create the moodboards that the API uses. Krea also runs its **own LoRA trainer** on the Max/Business tiers (beta at announcement): auto-captioning, and a floor of "at least three" images `[official — krea.ai blog, 2026-05-21]`. Managed hyperparameters, like fal's; the training doctrine in `lora-training.md §4` still applies to what you feed it. Krea's broader editor (realtime canvas, upscaler, etc.) wraps the same models, but it is out of scope here.

## 4. ComfyUI partner nodes

Hosted Medium/Large run inside ComfyUI via the official **"Krea 2 Image" API node**, launched 2026-05-27. The node exposes prompt, style refs, moodboard IDs, and creativity Raw/Low/Medium/High, and it bills through API credits. This is a *different integration* from the local Turbo template (`setup-and-workflows.md §1`). Partner nodes call Krea's servers, so nothing loads on your GPU.

## 5. fal (and other hosts) for the open models

fal is the launch API partner (2026-05-27). It offers four endpoints:

- `fal-ai/krea-2/turbo` — the open Turbo model, hosted. It takes `image_size` presets or a custom WxH, `enable_prompt_expansion` (this is the same expander, so the same considerations as `prompting-guide.md §6` apply), acceleration `none/regular`, a safety checker toggle, and png/jpeg output.
- `fal-ai/krea-2/turbo/lora` — Turbo plus your LoRA weights.
- `fal-ai/krea-2-trainer` — hosted LoRA training (`lora-training.md §4`).
- Hosted Medium/Large are also proxied on fal as `krea/v2/medium|large/text-to-image`.

The other listed inference partners for the open weights are SGLang (cookbook recipe), Replicate, Cloudflare, Together, GCP, AWS, and Runware. Partner param schemas count as official-via-host. Where fal and Krea disagree on a default, the GitHub README/CLI is the tiebreaker.
