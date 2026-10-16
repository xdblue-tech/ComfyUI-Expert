# RunPod Serverless ComfyUI — the full pipeline

How this repo turns a ComfyUI API workflow into a live RunPod endpoint and runs it.
All commands run from the repo root. Every script is stdlib Python — no MCP tools, no
runpodctl required.

```
workflow.json ──► bake image ──► push to Hub ──► create endpoint ──► submit jobs ──► outputs
                    rp_images.py                  rp_endpoints.py     rp_submit.py    state/output/
```

## One-time setup (already done on this machine)

1. `docker login -u xdblue` — Docker Hub personal access token, Read & Write scope
2. RunPod API key in `~/.runpod/api_key`
3. `config/rp.yaml` — namespace, GPU profiles, timeouts; `repo_names:` overrides repo names

Re-check state of the world at any time:

```bash
python3 scripts/rp_images.py list        # images built/pushed
python3 scripts/rp_endpoints.py list     # endpoints on the account
python3 scripts/rp_endpoints.py health unholy-sdxl   # live worker counts
```

## Process: workflow → running endpoint (about 15–25 min, mostly build+push)

### 1. Extract the workflow's model files (~2 min)

Read the workflow JSON. Every node whose `class_type` contains `Loader`
(`CheckpointLoaderSimple`, `LoraLoader`, `UNETLoader`, `CLIPLoader`, `VAELoader`, ...)
holds file references. The harley workflow has two:

- `CheckpointLoaderSimple.inputs.ckpt_name` → `unholyDesireMixSinister_v80.safetensors`
- `LoraLoader.inputs.lora_name` → `cinematic photography detailed illu xl v5.safetensors`

Rules:

- Enumerate every loader's actual inputs — no fixed field list (`clip_name` on a Krea
  workflow is easy to miss and fails at runtime).
- The VAE comes free from the checkpoint (output 2) unless a `VAELoader` node exists.
- Bake **only** these files. Each extra checkpoint is ~6.5 GB of build time, push time,
  and cold-start pull time.

### 2. Record them in `images/images-sdxl/` (~2 min)

`models.yaml` and the Dockerfile `COPY` lines must stay in sync — `build --check` fails
on drift. Filenames with spaces need the JSON form:

```dockerfile
COPY checkpoints/unholyDesireMixSinister_v80.safetensors /comfyui/models/checkpoints/
COPY ["loras/cinematic photography detailed illu xl v5.safetensors", "/comfyui/models/loras/"]
```

The Docker repo name comes from `config/rp.yaml` (`docker_namespace` + `repo_names:`
override; default is `comfyui-<short-spec>`).

Also declare the image's default workflow(s) in `models.yaml` under a `workflows:` key
(e.g. `workflows/unholy-sdxl.json`). `build --check` then verifies every model file the
workflow's Loader nodes reference is actually baked — a workflow that needs a model the
image lacks fails the check, not the endpoint at runtime.

### 3. Build and push (~10–20 min)

```bash
python3 scripts/rp_images.py build --check    # must print [OK] for the spec — no docker call
python3 scripts/rp_images.py build images-sdxl   # ~5–10 min, no login needed
python3 scripts/rp_images.py push images-sdxl    # ~10–30 min depending on upload speed
```

Tags are semantic and immutable (`v1.0.0`, `v1.1.0`, ...; `latest` is never used —
RunPod caches images per host). Records land in `state/images.json`.

### 4. Create the endpoint (~1 min)

```bash
python3 scripts/rp_endpoints.py create xdblue/unholy-sdxl:v1.0.0 --profile image-gen --name unholy-sdxl
```

Profile `image-gen` = RTX 4090, $1.12/hr, workers 0→1, 10 s idle timeout. Idle cost: $0.

### 5. Run jobs

```bash
# standard submit — polls to completion, saves outputs
python3 scripts/rp_submit.py --endpoint unholy-sdxl \
  --workflow /path/to/workflow.json --wait

# with a project folder and input images (whole request capped at 10 MB for /run)
python3 scripts/rp_submit.py --endpoint unholy-sdxl \
  --workflow proj.json --image in1.png --project myproj --wait

# just see the envelope, no network
python3 scripts/rp_submit.py --endpoint unholy-sdxl --workflow wf.json --dry-run
```

Outputs land in `state/output/<job-id>/` (or `projects/<p>/output/<job-id>/` with
`--project`), next to `job.json` and `result.json`.

Timing: first job on a fresh host pays a ~4 min cold start (image pull); after that,
image jobs take ~30 s to start + ~17 s to generate at 832×1216, 30 steps.

## Costs

- Image-gen (RTX 4090): ~$1.12/hr, billed only while a worker is up
- `workersMin: 0` — an idle endpoint costs nothing
- A typical 832×1216 image job: ~$0.01–0.05
- Video (A40/A6000, 720p/81 frames): ~$1.22/hr, 10–20 min per clip — use profile
  `video` and never send video to an endpoint with the 600 s image timeout

## Failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `create` → HTTP 500 `workers quota` | account caps total workers across all endpoints (10) | delete an endpoint or lower its `workersMax` in the Console; lowering never stops running workers |
| `create` → `Template name must be unique` | a previous failed `create` left a template behind | `DELETE https://rest.runpod.io/v1/templates/<id>`, then retry |
| `build --check` drift error | models.yaml and Dockerfile COPY lines disagree | make them match exactly, rebuild check |
| Endpoint dead after a week idle | RunPod silently drops idle workersMax (2 after ~3 days, 0 after ~7) | `python3 scripts/rp_endpoints.py sync` before any session |
| Job fails `file not found` | a loader's model file wasn't baked into the image | add the file to models.yaml + Dockerfile, rebuild, push, new tag + endpoint |
| Broken-symlink error on check | a models.yaml file is a dangling symlink locally | restore the file or remove it from the manifest |

## State files

| File | What it tracks |
|---|---|
| `state/images.json` | built/pushed tags, digests, sizes |
| `state/endpoints.json` | endpoint ids, worker bounds, drift status |
| `state/output/<job-id>/` | job metadata, raw response, extracted images/videos |

## Related

- `skills/runpod-comfyui-ops/SKILL.md` — the agent-facing version of this process
- `config/rp.yaml` — every deployment value (GPUs, timeouts, disk, namespace)
- `images/images-sdxl/`, `images/video-wan/` — image specs (Dockerfile + models.yaml)