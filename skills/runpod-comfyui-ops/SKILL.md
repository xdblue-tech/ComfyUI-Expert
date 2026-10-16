---
name: runpod-comfyui-ops
description: Run the RunPod ComfyUI production system for images and videos — convert a ComfyUI API workflow into a baked worker image (extract its model files, bake exactly those), build and push the images (images-sdxl, video-wan), register and sync serverless endpoints (state/endpoints.json), and submit ComfyUI API workflows to a live endpoint with scripts/rp_submit.py. Use when deploying a ComfyUI workflow to RunPod serverless, when an endpoint's workersMax has drifted, when create fails on workers quota or template-name collision, or when a video job needs an execution timeout long enough for Wan.
user-invocable: true
metadata: {"openclaw":{"emoji":"☁️","os":["linux","darwin"]}}
---

# RunPod ComfyUI Ops

Three loops over one local system. Everything is stdlib Python and file state; no MCP
tools and no runpodctl required.

| Loop | Tool | State it writes |
| --- | --- | --- |
| Build/push images | `scripts/rp_images.py` | `state/images.json` |
| Register/sync endpoints | `scripts/rp_endpoints.py` | `state/endpoints.json` |
| Produce (submit workflows) | `scripts/rp_submit.py` | `output/<job-id>/` or `projects/<p>/output/<job-id>/` |
| Export (third-party agents) | `scripts/rp_distribute.py` | `distribute/` (templates + manifest + standalone tool + packaged skill) |

Config: `config/rp.yaml` (Docker namespace, `comfyui_models_path`, base image, API key
file, per-profile GPU ids, worker bounds, timeouts). Read it before acting; every
deployment value comes from there, not from memory.

## One-time setup

1. Fill `docker_namespace` in `config/rp.yaml` (it ships as `PLACEHOLDER_DOCKERHUB_NS`).
2. `docker login` with a Docker Hub PAT (`read,write,delete` scope, not your password).
3. Put the RunPod key in `~/.runpod/api_key` (one line) or export `RUNPOD_API_KEY`.
   The key is read at runtime and never written into a script or state file.

## Loop 0 — turn a ComfyUI API workflow into a running endpoint

The recipe that goes from "user has a workflow JSON + models on disk" to "job COMPLETED".

1. Read the workflow JSON. For every model-loading node — any `class_type` containing
   `Loader` (`CheckpointLoaderSimple`, `LoraLoader`, `UNETLoader`, `CLIPLoader`/
   `DualCLIPLoader`, `VAELoader`, `ControlNetLoader`, `UpscaleModelLoader`, ...) — collect
   every string-valued file input (`ckpt_name`, `lora_name`, `unet_name`, `clip_name`,
   `vae_name`, `model_name`, ...). Do not trust a fixed field list either: enumerate each
   loader node's actual inputs, or a text encoder (e.g. `clip_name` on a Krea/Flux
   workflow) gets missed and the endpoint produces `file not found` at runtime.
   VAE comes free from the checkpoint (`CheckpointLoaderSimple` output 2) unless a
   `VAELoader` node exists.
2. Verify each file exists under `comfyui_models_path` (broken symlinks fail `build --check`).
   Resolve exact filenames — model names with spaces are common.
3. Write `images/<spec>/models.yaml` with exactly those files, mirror them as `COPY` lines
   in the spec's `Dockerfile`. Filenames with spaces need the JSON form:
   `COPY ["loras/file with spaces.safetensors", "/comfyui/models/loras/"]`
   (the drift checker accepts both forms).
4. Bake only what the workflow needs unless the user asks for more — every extra
   checkpoint is ~6.5 GB of build time, push time, and cold-start pull time.
   Declare the workflow itself in the spec's `models.yaml` under a `workflows:` key;
   `build --check` then verifies every model its Loader nodes reference is baked
   (a missing model fails the check instead of failing on the endpoint).
5. Set `repo_names:` in `config/rp.yaml` if the Docker repo name should differ from
   `comfyui-<short-spec>` (default tag is `<ns>/comfyui-<spec>:vX.Y.Z`).
6. `python3 scripts/rp_images.py build --check` — must say `[OK]` for the spec.
7. `python3 scripts/rp_images.py build <spec>` then `push <spec>` (push needs
   `docker login`; build does not).
8. `python3 scripts/rp_endpoints.py create <tag> --profile <p>` — then submit with
   `rp_submit.py --wait`. First job pays a ~4 min cold start (image pull); later jobs
   on the same host start in seconds.

## Loop 1 — build and push images

Image specs: `images/images-sdxl/` (workflow-derived: bake exactly the models the target
workflow references — checkpoints, LoRAs, VAE — not a general collection) and
`images/video-wan/`
Each spec is a `Dockerfile` plus a `models.yaml` manifest; the build context is
`comfyui_models_path`, so COPY lines name model files as they sit on disk.

1. `python3 scripts/rp_images.py build --check` — validates every spec: Dockerfile and
   manifest agree, every model file exists, base image matches config. No docker call.
2. `python3 scripts/rp_images.py build video-wan` — builds `--platform linux/amd64`,
   tagged `<ns>/comfyui-video-wan:v1.0.0` (minor version bumped from `state/images.json`).
3. `python3 scripts/rp_images.py push video-wan` — pushes, then records tag, digest, size,
   and timestamps in `state/images.json`.
4. `python3 scripts/rp_images.py list` — what was built and pushed.

Rules that break deploys when skipped: `latest` is never used (RunPod caches images per
host), tags are semantic and immutable in practice, `--version` is validated as `vX.Y.Z`
on both build and push, and an ARM build without `--platform linux/amd64` produces an
image RunPod cannot run.

## Loop 2 — register and sync endpoints

`create` makes a template (image, container disk, env) and then an endpoint (template,
GPU type ids, `workersMin`/`workersMax`, `idleTimeout`, `executionTimeoutMs`).

**The workers-quota trap.** The account caps total workers across all endpoints (10 on a
standard account). `create` fails with HTTP 500 `workers quota` when the new endpoint
would exceed it; existing endpoints' caps are freed with `delete` or by lowering their
`workersMax` in the Console. Lowering a cap never stops running workers.

**The template-collision trap.** `create` makes a template first, then the endpoint; if
endpoint creation fails (quota), the template stays and a retry fails with `Template
name must be unique`. Delete the leftover template (`DELETE /v1/templates/<id>`) before
retrying, or `create` with a different `--name`.

1. `python3 scripts/rp_endpoints.py create <ns>/comfyui-video-wan:v1.0.0 --profile video`
   — the tag must be `vX.Y.Z` (`:latest` and untagged images are refused before any
   remote resource is created) and the default endpoint name drops the namespace
   prefix: `comfyui-video-wan`
2. `python3 scripts/rp_endpoints.py list` — registry status per endpoint.
3. `python3 scripts/rp_endpoints.py health <id-or-name>` — endpoint config plus live worker
   counts from the v2 worker read.
4. `python3 scripts/rp_endpoints.py sync` — diff against the account and repair drift.
5. `python3 scripts/rp_endpoints.py delete <id-or-name>` — deletes the endpoint, marks the
   registry row `deleted`; the template stays.

**The idle-drop trap.** RunPod silently lowers an idle endpoint's max workers — to 2
after about 3 idle days, to 0 after about 7 — so a healthy endpoint one week is a dead
one the next. `sync` detects the drop and restores the recorded `workersMax` with a
`PATCH` on the REST v2 serverless route (`PATCH /v2/serverless/<id>`,
`{"workers":{"max":N}}` — v1 documents no endpoint-update route); run it before any
production session. A PATCH response is not a read, so `sync` re-reads the endpoint and
only then marks the row `active`; if the repair fails or the read-back still disagrees,
the row stays `drifted` (`unknown` when RunPod reports no `workersMax` at all) and
`sync` exits 1. Fix those in the Console or with
`runpodctl serverless update <id> --workers-max N`.

**GPU ids.** `gpuTypeIds` takes display names (`NVIDIA A40`, `NVIDIA RTX A6000`,
`NVIDIA GeForce RTX 4090`), not serverless pool ids (`AMPERE_48`). Verify with
`runpodctl gpu list` and change the one line in `config/rp.yaml` if a name is wrong.

Schema of `state/endpoints.json`: each row is `endpoint_id, name, template_id,
image_tag, gpu_types, workers_min, workers_max, exec_timeout_ms, status
(active|drifted|unknown|deleted), created_at, last_synced_at`, plus an optional
`drift_note`.
Schema of `state/images.json`: each row is `spec, repo, version, tag, digest, size_mb,
created_at, pushed_at, status (built|pushed)` — `build` records the tag, `push` fills in
the digest, size, and image creation time.

## Loop 3 — produce

1. Build the workflow with `comfyui-workflow-builder` (save it under
   `projects/<p>/workflows/`), validated against `state/inventory.json` and, on the
   remote side, against the models baked into the image.
2. `python3 scripts/rp_submit.py --endpoint comfyui-video-wan --workflow <path.json> --wait`
3. Outputs land in `output/<job-id>/` (or `projects/<p>/output/<job-id>/` with
   `--project`), next to `job.json` (job metadata) and `result.json` (the worker's raw
   response, written before extraction, so an unrecognized output shape or a failed
   download is still recoverable). Failed extraction exits 1; a response with no
   recognizable image/video files warns and exits 0.

Envelope (worker-comfyui input contract — `images` only when inputs are attached):

```json
{"input": {"workflow": {"1": {"class_type": "...", "inputs": {}}},
           "images": [{"name": "in.png", "image": "data:image/png;base64,..."}]}}
```

- `--dry-run` prints the envelope and the target URL without a network call: endpoint
  names resolve against `state/endpoints.json` only, and anything else is printed as typed.
- `/run` + `/status` polling is the default with `--wait` (5 s interval, 30 min timeout);
  `--sync` uses `/runsync` for one held request. Cold starts outlive `runsync`.
- Input images share one whole-request limit with the workflow: 10 MB for `/run`, 20 MB
  for `/runsync`. Base64 is 4/3 the file size, and several images accumulate in the same
  body, so `rp_submit.py` checks the serialized envelope before POSTing (about a 7.5 MB
  image alone fits the 10 MB `/run` limit). Larger inputs must be passed as http(s) URLs,
  which the worker fetches itself.
- Video timing: 720p/81 frames on a 48 GB card runs 10–20 min per clip, so video
  endpoints carry `execution_timeout_ms: 1800000`. Never launch a video job against an
  endpoint still on the 600000 ms image default.
- Cost: image-gen is ~$1.12/hr on an RTX 4090, video ~$1.22/hr on A40/A6000, billed while
  a worker is up. `workersMin: 0` keeps idle cost at zero.

## Lanes to the Runpod plugin skills

Do not duplicate those skills and do not edit them (they are vendored):

| Need | Lane |
| --- | --- |
| Route an unfamiliar Runpod task, or pick a golden path | `skills/runpod/SKILL.md` |
| HuggingFace/docker/GitHub/S3 CLI details, auth, image push traps | `skills/companion-clis/SKILL.md` |
| Templates, network volumes, registry credentials, secrets | `skills/lifecycle-crud/SKILL.md` |
| Diagnose a queued job, dying worker, or endpoint scaling | `skills/endpoint-ops/SKILL.md` |
| Deploy some other model/source to serverless | `skills/serverless-deploy/SKILL.md` |
| Wan/FramePack model prerequisites and settings | `skills/comfyui-video-pipeline/SKILL.md` |
