---
name: comfyui-workflow-builder
description: Generate, build, create, or design ComfyUI workflow JSON from natural language descriptions. Produces valid node graphs with correct class_types, connections, output indices, and model-appropriate settings. Handles txt2img, img2img, inpainting, ControlNet, LoRA stacking, upscaling, and face detailing pipelines. Does NOT cover ComfyUI installation, custom node development, Python scripting, model training, hardware advice, or architectural explanations.
user-invocable: true
metadata: {"openclaw":{"emoji":"🔧","os":["darwin","linux","win32"],"requires":{"anyBins":["curl","wget"]},"primaryEnv":"COMFYUI_URL"}}
---

# ComfyUI Workflow Builder

Translates natural language requests into executable ComfyUI workflow JSON. Always validates against inventory before generating.

## Workflow Generation Process

### Step 1: Understand the Request

Parse the user's intent into:
- **Output type**: Image, video, or audio
- **Source material**: Text-only, reference image(s), existing video
- **Identity method**: None, zero-shot (InstantID/PuLID), LoRA, Kontext
- **Quality level**: Draft (fast iteration) vs production (maximum quality)
- **Special requirements**: ControlNet, inpainting, upscaling, lip-sync

### Step 2: Check Inventory

Read `state/inventory.json` to determine:
- Available checkpoints → select best match for task
- Available identity models → determine which methods are possible
- Available ControlNet models → enable pose/depth control if available
- Custom nodes installed → verify all required nodes exist
- VRAM available → optimize settings accordingly

### Step 3: Select Pipeline Pattern

Based on request + inventory, choose from:

| Pattern | When | Key Nodes |
|---------|------|-----------|
| Text-to-Image | Simple generation | Checkpoint → CLIP → KSampler → VAE |
| Identity-Preserved Image | Character consistency | + InstantID/PuLID/IP-Adapter |
| LoRA Character | Trained character | + LoRA Loader |
| Image-to-Video (Wan) | High-quality video | Diffusion Model → Wan I2V → Video Combine |
| Image-to-Video (AnimateDiff) | Fast video, motion control | + AnimateDiff Loader + Motion LoRAs |
| Krea 2 (K2) text-to-image | Flux-derived DiT with Qwen3-VL-4B encoder | `UNETLoader` + `CLIPLoader(type="krea2")` + `VAELoader(krea2RealVae)` → `KSampler` |
| Talking Head | Character speaks | Image → Video → Voice → Lip-Sync |
| Upscale | Enhance resolution | Image → UltimateSDUpscale → Save |
| Inpainting | Edit regions | Image + Mask → Inpaint Model → KSampler |

### Step 4: Generate Workflow JSON

**ComfyUI workflow format:**

```json
{
  "{node_id}": {
    "class_type": "{NodeClassName}",
    "inputs": {
      "{param_name}": "{value}",
      "{connected_param}": ["{source_node_id}", {output_index}]
    }
  }
}
```

**Rules:**
- Node IDs are strings (typically "1", "2", "3"...)
- Connected inputs use array format: `["source_node_id", output_index]`
- Output index is 0-based integer
- Filenames must match exactly what's in inventory
- Seed values: use random large integer or fixed for reproducibility

### Step 5: Validate

Before presenting to user:

1. Every `class_type` exists in inventory's node list
2. Every model filename exists in inventory's model list
3. All required connections are present (no dangling inputs)
4. VRAM estimate doesn't exceed available VRAM
5. Resolution is compatible with chosen model (512 for SD1.5, 1024 for SDXL/FLUX)

### Step 6: Output

**If online mode**: Queue via `comfyui-api` skill
**If offline mode**: Save JSON to `projects/{project}/workflows/` with descriptive name

## Workflow Templates

### Basic Text-to-Image (FLUX)

```json
{
  "1": {
    "class_type": "LoadCheckpoint",
    "inputs": {"ckpt_name": "flux1-dev.safetensors"}
  },
  "2": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "{positive_prompt}", "clip": ["1", 1]}
  },
  "3": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "{negative_prompt}", "clip": ["1", 1]}
  },
  "4": {
    "class_type": "EmptyLatentImage",
    "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
  },
  "5": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 25,
      "cfg": 3.5,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["1", 0],
      "positive": ["2", 0],
      "negative": ["3", 0],
      "latent_image": ["4", 0]
    }
  },
  "6": {
    "class_type": "VAEDecode",
    "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
  },
  "7": {
    "class_type": "SaveImage",
    "inputs": {"filename_prefix": "output", "images": ["6", 0]}
  }
}
```

### Krea 2 Text-to-Image (Qwen3-VL-4B)

Krea 2 (K2) is a Flux-derived DiT (signature key `txtfusion.projector.weight` in the diffusion-model state dict) trained on a 12-layer Qwen3-VL-4B text-encoder tap and conditioned on 30720-D hidden stacks. It is *not* a vanilla Flux model — the loader chain, VAE, sampler, and CFG are all distilled. The standard Flux chain (`DualCLIPLoader(clip_l + t5xxl)` + `ae.safetensors` + `karras`) will load but fail at conditioning time or produce grid noise.

```json
{
  "1": {
    "class_type": "UNETLoader",
    "inputs": {"unet_name": "darkBeastKREA2_fp8.safetensors", "weight_dtype": "default"}
  },
  "2": {
    "class_type": "CLIPLoader",
    "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2"}
  },
  "3": {
    "class_type": "VAELoader",
    "inputs": {"vae_name": "krea2RealVae_v10.safetensors"}
  },
  "4": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "{positive_prompt}", "clip": ["2", 0]}
  },
  "5": {
    "class_type": "CLIPTextEncode",
    "inputs": {"text": "{negative_prompt}", "clip": ["2", 0]}
  },
  "6": {
    "class_type": "EmptyLatentImage",
    "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
  },
  "7": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 12,
      "cfg": 1.0,
      "sampler_name": "er_sde",
      "scheduler": "simple",
      "denoise": 1.0,
      "model": ["1", 0],
      "positive": ["4", 0],
      "negative": ["5", 0],
      "latent_image": ["6", 0]
    }
  },
  "8": {
    "class_type": "VAEDecode",
    "inputs": {"samples": ["7", 0], "vae": ["3", 0]}
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {"filename_prefix": "k2", "images": ["8", 0]}
  }
}
```

### With Identity Preservation (InstantID + IP-Adapter)

Extends basic template by adding:
- Load reference image node
- InstantID Model Loader + Apply InstantID
- IPAdapter Unified Loader + Apply IPAdapter
- FaceDetailer post-processing

See `references/workflows.md` for complete node settings.

### Video Generation (Wan I2V)

Uses different loader chain:
- Load Diffusion Model (not LoadCheckpoint)
- Wan I2V Conditioning
- EmptySD3LatentImage (with frame count)
- Video Combine (VHS)

See `references/workflows.md` Workflow 4 for complete settings.

## VRAM Estimation

| Component | Approximate VRAM |
|-----------|-----------------|
| FLUX FP16 | 16GB |
| FLUX FP8 | 8GB |
| SDXL | 6GB |
| SD1.5 | 4GB |
| InstantID | +4GB |
| IP-Adapter | +2GB |
| ControlNet (each) | +1.5GB |
| Wan 14B | 20GB |
| Wan 1.3B | 5GB |
| AnimateDiff | +3GB |
| FaceDetailer | +2GB |

## Common Mistakes to Avoid

1. **Wrong output index**: CheckpointLoader outputs `[model, clip, vae]` at indices `[0, 1, 2]`
2. **CFG too high for InstantID**: Use 4-5, not default 7-8
3. **Wrong resolution for model**: FLUX/SDXL=1024, SD1.5=512
4. **Missing VAE**: FLUX needs explicit VAE (`ae.safetensors`)
5. **Wrong model in wrong loader**: Diffusion models need `LoadDiffusionModel`, not `LoadCheckpoint`

## Krea 2 Gotchas

- **VAE mismatch is the #1 silent failure.** The standard Flux `ae.safetensors` decodes K2 latents as a regular grid/tiled noise pattern — every pixel is high-variance but no subject is visible. Use the K2-specific `krea2RealVae_v10.safetensors` (≈485 MB). When in doubt: `safe_load()` it and look for the `modelspec.architecture` field — K2-Real-VAE lists `conv1`/`conv2` keys rather than `decoder.*`/`encoder.*`.
- **Text encoder is Qwen3-VL, not Flux T5/CLIP-L.** Use `CLIPLoader(clip_name="qwen3vl_4b_fp8_scaled.safetensors", type="krea2")`. `DualCLIPLoader` with `clip_l`+`t5xxl` will load but produce `RuntimeError: Krea2 expects conditioning with 12x2560=30720 features but got 4096`.
- **Distilled sampling is non-negotiable.** `er_sde` sampler + `simple` scheduler + cfg=1.0 + steps=4–12. `karras` scheduler and CFG≥2 desync the timestep schedule and blow out the image.
- **Detect K2 architecture by signature.** `model.diffusion_model.txtfusion.projector.weight` present in the state dict → K2. Handled automatically by `comfy/model_detection.py`; you only need to know this when reading a fresh safetensors file.
- **The checkpoint's metadata is the recipe.** Many K2 fine-tunes embed a ComfyUI workflow JSON in the safetensors `metadata.workflow` field. Extract it (`safetensors.torch.safe_open(...).metadata()['workflow']`) and copy the author's exact `KSampler`/`VAELoader` settings before guessing.
- **Resolution baseline is 1024×1024.** K2 fine-tunes are trained on 1024 and produce coherent 1024×1024 outputs; 832×1216 or 1280×720 work but check the checkpoint's embedded workflow for the canonical resolution.

## Reference Files

- `references/workflows.md` - Detailed node-by-node templates
- `references/models.md` - Model files and paths
- `references/prompt-templates.md` - Model-specific prompts
- `state/inventory.json` - Current inventory cache
