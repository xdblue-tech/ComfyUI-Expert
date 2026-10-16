# SDXL worker for RunPod serverless ComfyUI (runpod/worker-comfyui, queue API).
#
# Build context is the local ComfyUI models directory from config/rp.yaml, so the
# COPY sources are model files as they sit on disk. Build and push with:
#
#   python3 scripts/rp_images.py build images-sdxl
#   python3 scripts/rp_images.py push images-sdxl
#
# COPY lines mirror models.yaml; `rp_images.py build --check` fails if they drift.
# Keep the base image in sync with base_image in config/rp.yaml.

FROM runpod/worker-comfyui:5.10.0-base-cuda12.8.1

COPY checkpoints/unholyDesireMixSinister_v80.safetensors /comfyui/models/checkpoints/

COPY ["loras/cinematic photography detailed illu xl v5.safetensors", "/comfyui/models/loras/"]
