"""Bounded PNG/JPEG artifact validation shared by the smoke test and the service.

The bytes here come from a remote worker, so the check is deliberately complete:
a header alone is not proof of an image. ``verify()`` walks the container and a
second open inflates every pixel, under byte, dimension and pixel caps. The
decoded image is never displayed, saved elsewhere or sent to a model.
"""

import io

from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
IMAGE_FORMATS = {"PNG", "JPEG"}
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_PIXELS = 16 * 1024 * 1024
MAX_ARTIFACT_DIMENSION = 8192


class ArtifactError(RuntimeError):
    """The artifact bytes are empty, corrupt, oversized or not a usable image."""


def sniff_image(data: bytes):
    """Return (format, width, height) after verifying and decoding the image.

    A header alone is not evidence of a usable image: a truncated download has
    one, a structurally valid PNG can carry an empty IDAT, and a JPEG can stop
    after its frame header. ``verify()`` walks the container (chunk structure,
    CRCs, terminator) and a second open inflates every pixel with ``load()``, so a
    broken or truncated stream raises instead of passing. The canvas is bounded
    before that allocation, which also refuses a decompression bomb.
    """
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"artifact is {len(data)} bytes; the cap is {MAX_ARTIFACT_BYTES}"
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            if image_format not in IMAGE_FORMATS:
                raise ArtifactError(
                    f"artifact format {image_format!r} is not PNG or JPEG"
                )
            if width <= 0 or height <= 0:
                raise ArtifactError(f"artifact has non-positive dimensions {width}x{height}")
            if width > MAX_ARTIFACT_DIMENSION or height > MAX_ARTIFACT_DIMENSION:
                raise ArtifactError(
                    f"artifact is {width}x{height}; each side is capped at "
                    f"{MAX_ARTIFACT_DIMENSION}"
                )
            if width * height > MAX_ARTIFACT_PIXELS:
                raise ArtifactError(
                    f"artifact is {width * height} pixels; the cap is "
                    f"{MAX_ARTIFACT_PIXELS} pixels"
                )
            image.verify()
        # verify() leaves the image unusable, so pixels need a fresh open.
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except ArtifactError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ArtifactError(
            f"artifact does not decode to pixels ({exc}); it is truncated or corrupt"
        ) from exc
    return image_format.lower(), width, height
