# Hardware Profile

## GPU

- **Model**: NVIDIA GeForce RTX 5070 Ti
- **VRAM**: 16.7 GB (reported by CUDA; plan for ~16 GB usable)
- **ComfyUI**: 0.33.1 at <http://127.0.0.1:8188>, source at /home/xdblue/ComfyUI
- **OS**: Linux, Python 3.14

## Capabilities at 16.7 GB VRAM

| Task class | 16.7 GB verdict |
| ------------ | ----------------- |
| SDXL / Pony / Illustrious checkpoints | Full quality, no offload |
| Flux dev (fp8 or GGUF Q8) | Viable, expect offload with large batches |
| SD1.5 / SDXL ControlNet stacks | Viable |
| LoRA training (SDXL) | Viable with small batch + gradient checkpointing |
| 14B-class video models (Wan etc.) | Not installed; only GGUF quantized + offload, slow |
| Simultaneous video + upscaler pipelines | Avoid; run sequentially |

> If you move this repo to a 32 GB card, restore a 32 GB capability table.

## Recommended Launch Flags

```bash
python main.py --listen
```

- Use the default flags. `--highvram` is not appropriate for a 16 GB card: it keeps models resident and causes OOM.
- `--fp8_e4m3fn-unet` is optional, for Flux workflows.

## Performance Tips

- Queue jobs sequentially. Parallel batches of 4x 1024x1024 do not fit in 16.7 GB.
- Enable tiled VAE for large upscales.
- Use fp8 checkpoints for Flux when other models are resident.
- cuDNN 8800+ recommended for maximum throughput.
