# Qwen-Image — API & hosted surfaces

This file covers renting rather than running: Alibaba Cloud Model Studio (DashScope), which is the only place Qwen-Image 2.0 and 3.0 exist; the ComfyUI API nodes that call it; and the third-party hosts (Replicate, fal) that serve the open weights and a LoRA trainer. Model strings and parameters were read from Alibaba's docs and the ComfyUI API templates on 2026-09-09 `[official-via-docs — Model Studio; ComfyUI API templates]`. **Pricing could not be verified against Alibaba's own price list.** Re-verify both pricing and the open/closed status before quoting either.

## Contents

1. Hosted vs open — what is actually different
2. Alibaba Model Studio / DashScope
3. The ComfyUI API nodes
4. Replicate and fal
5. Pricing — what is known and what is not

---

## 1. Hosted vs open — what is actually different

| | Open (Qwen-Image … 2512, Edit … 2511, Layered; **2.1 since 2026-09-20**) | Hosted (2.0, 2.0 Pro, 3.0, 3.0 Pro, max, plus; the 2.1 demo's `pre-qwen-image-2.1-pro`) |
|---|---|---|
| Weights | downloadable — Apache 2.0 for the 20B family, **Qwen Research License (non-commercial) for 2.1** | **none** — no `Qwen/Qwen-Image-2.0` or `-3` repo exists on Hugging Face; the `Qwen/Qwen-Image-2.1` Space calls a hosted `pre-qwen-image-2.1-pro-yunqi` model with `prompt_extend`, not the open weights |
| Model | 20B MMDiT, same geometry across variants; 2.1 is a 7B single-stream DiT (`setup-and-workflows.md §11`) | 2.0 is "a next-generation foundational image generation model" with a "lighter model architecture"; 3.0 is undocumented by Qwen on GitHub `[official — QwenLM README News]` |
| Resolution | the 1328 class (~1.5 MP); Edit natively 2–3 MP via the bypass | **native 2K**; 2.0 takes any size with total pixels between 512×512 and 2048×2048 |
| Prompt handling | what you type is what the encoder sees | **`prompt_extend` defaults to `true`** — a hosted LLM rewrites the prompt before rendering |
| Instruction length | Edit wants short prompts | 2.0 advertises "1k-token instructions" |
| Regions | your GPU | Singapore (`ap-southeast-1`) and Beijing (`cn-beijing`) |
| Licence | Apache 2.0 (20B); research-only (2.1) | Model Studio terms of service |

**`prompt_extend` on by default is the single most important hosted-vs-local difference.** Local inference never rewrites your prompt; the API does unless you set the flag false. That matters for reproducibility, for text-to-render, and for anything you A/B against a local render. The rewriter's rules are in `prompting-guide.md §5`. No independent hosted-vs-open shootout has been published. Assume 3.0 Pro beats 2512 on fidelity and treat the size of the gap as unknown.

---

## 2. Alibaba Model Studio / DashScope

Model strings on the image-generation endpoint, per Alibaba's model list:

| Model string | Snapshots | Note |
|---|---|---|
| `qwen-image-3.0-pro`, `qwen-image-3.0` | — | the flagship; the ComfyUI API templates target `qwen-image-3.0-pro` |
| `qwen-image-2.0-pro` | `2026-06-22`, `2026-04-22`, `2026-03-03` | |
| `qwen-image-2.0` | `2026-03-03` | |
| `qwen-image-max` | `2025-12-30` | defaults to `1664*928`, fixed size set |
| `qwen-image-plus` | `2026-01-09` | same |
| `qwen-image` | — | |
| `qwen-image-edit` | — | the editing endpoint |

**Parameters common to the family:**

| Param | Values | Note |
|---|---|---|
| `prompt` | string | rewritten by default |
| `negative_prompt` | max 500 characters | live: the hosted models run real CFG |
| `prompt_extend` | boolean, **default `true`** | set `false` for verbatim prompts |
| `watermark` | boolean, default `false` | |
| `seed` | 0 – 2147483647 | |
| `n` | int | images per call |
| size | 2.0 series: any total between 512×512 and 2048×2048, default 2048×2048; max/plus: `1664*928, 1472*1104, 1328*1328, 1104*1472, 928*1664` | the max/plus set is the open model card's aspect table |

The max/plus size set uses `1472*1104`, the corrected 4:3 figure, which supports reading the earlier card's 1140 as a typo (`prompting-guide.md §4`). The official prompt rewriters in `QwenLM/Qwen-Image` also call DashScope (`qwen-plus` for T2I, `qwen-vl-max-latest` for Edit, via `DASHSCOPE_API_KEY`). Even "local" official prompt expansion is a hosted call.

---

## 3. The ComfyUI API nodes

Comfy-Org shipped two API templates on **2026-08-06**: `api_qwen3_t2i.json` ("Qwen Image 3.0 Pro: Text to Image") and `api_qwen3_image_edit.json` ("Qwen Image 3.0 Pro: Image Edit") `[official — templates/index.json]`. An API node is a remote call to Alibaba's service billed through ComfyUI API credits. Nothing loads on your GPU, and this is the **only** first-party way to reach 3.0 from ComfyUI.

- **`QwenImageTextToImageApi`** — widgets `["qwen-image-3.0-pro", <prompt>, <negative>, 1024, 1024, 1, 42, "randomize", true, false]`, with a `ResolutionSelector` node feeding width and height.
- **`QwenImageEditApi`** — widgets `["qwen-image-3.0-pro", <prompt>, <negative>, "match input", 1, <seed>, "randomize", true, false]`, with `model.images.image_1` and `image_2` inputs. The hosted edit node takes at least two references.

The two trailing booleans are unlabelled in the template. Given the DashScope defaults they are almost certainly `prompt_extend` and `watermark`, in that order, but the template does not say so `[flagged — re-verify]`. If your hosted render ignores the wording you typed, that first boolean is the suspect.

---

## 4. Replicate and fal

**Replicate** (`replicate.com/qwen`) lists the open weights and two closed ones: `qwen/qwen-image`, `qwen/qwen-image-2`, `qwen/qwen-image-2-pro`, `qwen/qwen-image-2512`, `qwen/qwen-image-layered`, `qwen/qwen-image-edit`, `qwen/qwen-image-edit-plus`, `qwen/qwen-image-edit-plus-lora`, `qwen/qwen-edit-multiangle`, and **`qwen/qwen-image-lora-trainer`** (plus a `-legacy`). No prices are shown on the index `[official-via-docs — replicate.com/qwen]`. The `edit-plus-lora` and `edit-multiangle` slugs are the hosted forms of the paired-data Edit LoRAs in `lora-training.md §6`.

**fal** exposes `alibaba/qwen-image-3/text-to-image`, "resolutions up to 2048×2048, with automatic prompt rewriting and prompt-guided resolution selection" `[official-via-docs — fal.ai]`, alongside the open models and a Qwen LoRA shelf. The fal Multiple-Angles 2511 port is the most-pulled Qwen LoRA on Hugging Face. Where a host and Alibaba's docs disagree on a default, Alibaba's docs are the tiebreaker for the hosted models and the model card for the open ones.

**Hosted LoRA training** on Replicate manages the hyperparameters. The dataset doctrine in `lora-training.md §3–5` still governs what you upload, and the trained weights come back as standard safetensors usable locally. Ask which generation (2509 / 2511 / 2512) the trainer targets before uploading.

---

## 5. Pricing — what is known and what is not

Alibaba's model-list page carries no per-image figures, and the separate "Model pricing" page could not be read for this pass. What circulates in secondary reporting `[flagged — secondary, unverified; re-verify before quoting]`:

- Qwen-Image-3.0 Pro: roughly $0.04 per image at 1K and $0.075 at 2K.
- Qwen-Image-3.0 (standard): a flat ~$0.03 per image.
- A ¥0.18-per-image figure also circulates.

None of these was confirmed against Alibaba's own price list. Do not put a number in a budget from this file. The same caution applies to the launch timeline. Secondary sources say 3.0 was announced 2026-07-21, invite-only, then generally available on the API around 2026-08-04/05, which is consistent with the ComfyUI templates dated 2026-08-06. The QwenLM README's News block stops at 2.0 (2026-02-10) and never mentions 3.0. Qwen-Image-2.1 has its own repo (`QwenLM/Qwen-Image-2.1`) and, as of 2026-09-20, no Model Studio model string, no ComfyUI API node and no third-party host; the 'Pro' variant behind the demo is unannounced and unpriced `[flagged — re-verify]`.
