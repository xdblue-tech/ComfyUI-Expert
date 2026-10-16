# Wan 2.1 I2V + FramePack worker for RunPod serverless ComfyUI (queue API).
#
# Build context is the local ComfyUI models directory from config/rp.yaml, so the
# COPY sources are model files as they sit on disk. Build and push with:
#
#   python3 scripts/rp_images.py build video-wan
#   python3 scripts/rp_images.py push video-wan
#
# COPY lines mirror models.yaml; `rp_images.py build --check` fails if they drift.
# Keep the base image in sync with base_image in config/rp.yaml.
# Video jobs need the long endpoint timeout from config/rp.yaml (video profile,
# execution_timeout_ms 1800000) — Wan 720p/81 frames runs 10-20 min per clip.

FROM runpod/worker-comfyui:5.10.0-base-cuda12.8.1

COPY diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors /comfyui/models/diffusion_models/
COPY diffusion_models/FramePackI2V_HY_fp8_e4m3fn.safetensors /comfyui/models/diffusion_models/

COPY text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors /comfyui/models/text_encoders/

COPY clip_vision/clip_vision_h.safetensors /comfyui/models/clip_vision/

COPY vae/wan_2.1_vae.safetensors /comfyui/models/vae/
