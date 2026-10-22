"""Opt-in real vendor smoke test: one job, one artifact, no secrets recorded.

This is the only place the prototype touches the vendor. It is disabled unless
the caller passes ``--enable-live-vendor`` and supplies ``RUNPOD_API_KEY`` in the
environment. The key never enters the report and the artifact is decoded locally
with Pillow under byte, dimension and pixel caps: a complete image stream that
produces real pixels, never a header or a chunk list that only looks complete. The
decoded image is not displayed, saved elsewhere or sent to a model.
"""

import io
import json
import os
import time
from pathlib import Path

from PIL import Image

from .bridge import submit_plan

DEFAULT_KEY_ENV = "RUNPOD_API_KEY"
DEFAULT_PLAN = {
    "template": "unholy-sdxl",
    "prompt": (
        "masterpiece, best quality, photorealistic smoke test of a red apple on a "
        "wooden table, soft window light, no text"
    ),
    "seed": 42,
    "width": 832,
    "height": 1216,
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
IMAGE_FORMATS = {"PNG", "JPEG"}
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_PIXELS = 16 * 1024 * 1024
MAX_ARTIFACT_DIMENSION = 8192


class VendorSmokeError(RuntimeError):
    """The smoke test could not run or the vendor result was not usable."""


def run_vendor_smoke(
    *,
    enable_live_vendor: bool,
    package_root,
    output_dir,
    plan_data=None,
    api_key_env: str = DEFAULT_KEY_ENV,
    env=None,
    runner_prefix=None,
    budget_seconds: float = 900.0,
) -> dict:
    """Submit one real job and verify its artifact, or return ``status: not-run``."""
    if not enable_live_vendor:
        return {
            "status": "not-run",
            "reason": (
                "the real vendor smoke test is opt-in: pass --enable-live-vendor with "
                "an explicit budget and RUNPOD_API_KEY in the environment. No request "
                "was sent and no live result is claimed"
            ),
        }
    environment = dict(os.environ if env is None else env)
    if not environment.get(api_key_env):
        raise VendorSmokeError(
            f"{api_key_env} is not set — export the RunPod API key before running the "
            "smoke test. The key is read from the environment and never written to "
            "artifacts"
        )
    from .package import Package

    package = Package.load(package_root)
    output = Path(output_dir)
    started = time.monotonic()
    result = submit_plan(
        package,
        dict(DEFAULT_PLAN if plan_data is None else plan_data),
        output_dir=output,
        api_key_env=api_key_env,
        env=environment,
        runner_prefix=runner_prefix,
        wait_timeout_s=max(60, int(budget_seconds) - 120),
        timeout=budget_seconds,
    )
    duration = time.monotonic() - started
    failures = []
    artifact = None
    if not result.job_id:
        failures.append("the runner did not report a job id")
    if result.output_dir is None or not Path(result.output_dir).is_dir():
        failures.append("the runner did not report an output directory")
    else:
        artifact_path = _find_image(Path(result.output_dir))
        if artifact_path is None:
            failures.append(f"no image artifact under {result.output_dir}")
        else:
            artifact = _describe_image(artifact_path, failures)
    if duration > budget_seconds:
        failures.append(f"job took {duration:.0f}s, over the {budget_seconds:.0f}s budget")
    return {
        "status": "passed" if not failures else "failed",
        "job_id": result.job_id,
        "duration_seconds": round(duration, 2),
        "artifact": artifact,
        "failures": failures,
        "limits": {"budget_seconds": budget_seconds, "jobs": 1},
    }


def _find_image(root: Path):
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return candidates[0] if candidates else None


def _describe_image(path: Path, failures: list):
    size = path.stat().st_size
    descriptor = {"path": str(path), "bytes": size}
    if size == 0:
        failures.append(f"{path.name} is empty")
        descriptor["format"] = "unknown"
        return descriptor
    if size > MAX_ARTIFACT_BYTES:
        failures.append(
            f"{path.name} is {size} bytes; the artifact cap is {MAX_ARTIFACT_BYTES}"
        )
        descriptor["format"] = "unknown"
        return descriptor
    try:
        image_format, width, height = sniff_image(path.read_bytes())
    except VendorSmokeError as exc:
        failures.append(f"{path.name}: {exc}")
        descriptor["format"] = "unknown"
        return descriptor
    descriptor.update({"format": image_format, "width": width, "height": height})
    if width <= 0 or height <= 0:
        failures.append(f"{path.name}: non-positive dimensions {width}x{height}")
    return descriptor


def sniff_image(data: bytes):
    """Return (format, width, height) after verifying and decoding the image.

    A header alone is not evidence of a usable image: a truncated download has
    one, a structurally valid PNG can carry an empty IDAT, and a JPEG can stop
    after its frame header. ``verify()`` walks the container (chunk structure,
    CRCs, terminator) and a second open inflates every pixel with ``load()``, so a
    broken or truncated stream raises instead of passing. The canvas is bounded
    before that allocation, which also refuses a decompression bomb. Nothing is
    displayed and nothing leaves this process.
    """
    if len(data) > MAX_ARTIFACT_BYTES:
        raise VendorSmokeError(
            f"artifact is {len(data)} bytes; the cap is {MAX_ARTIFACT_BYTES}"
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in IMAGE_FORMATS:
                raise VendorSmokeError(
                    f"artifact format {image_format!r} is not PNG or JPEG"
                )
            if width <= 0 or height <= 0:
                raise VendorSmokeError(f"artifact has non-positive dimensions {width}x{height}")
            if width > MAX_ARTIFACT_DIMENSION or height > MAX_ARTIFACT_DIMENSION:
                raise VendorSmokeError(
                    f"artifact is {width}x{height}; each side is capped at "
                    f"{MAX_ARTIFACT_DIMENSION}"
                )
            if width * height > MAX_ARTIFACT_PIXELS:
                raise VendorSmokeError(
                    f"artifact is {width * height} pixels; the cap is "
                    f"{MAX_ARTIFACT_PIXELS} pixels"
                )
            image.verify()
        # verify() leaves the image unusable, so pixels need a fresh open.
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except VendorSmokeError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise VendorSmokeError(
            f"artifact does not decode to pixels ({exc}); it is truncated or corrupt"
        ) from exc
    return image_format.lower(), width, height


def write_smoke_report(report: dict, output_dir) -> Path:
    """Persist a redacted summary next to the job artifacts."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / "vendor-smoke.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
