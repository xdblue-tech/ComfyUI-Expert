"""The shared decoder keeps the vendor-smoke contract and caps untrusted bytes."""

import struct
import unittest
import zlib

from distribute_agent import artifacts, vendor_smoke

from service_test_support import PNG_BYTES, build_png


class VendorCompatibilityTests(unittest.TestCase):
    def test_vendor_smoke_reexports_the_shared_decoder_and_constants(self):
        self.assertIs(vendor_smoke.MAX_ARTIFACT_BYTES, artifacts.MAX_ARTIFACT_BYTES)
        self.assertIs(vendor_smoke.MAX_ARTIFACT_PIXELS, artifacts.MAX_ARTIFACT_PIXELS)
        self.assertIs(
            vendor_smoke.MAX_ARTIFACT_DIMENSION, artifacts.MAX_ARTIFACT_DIMENSION
        )
        self.assertTrue(issubclass(vendor_smoke.VendorSmokeError, artifacts.ArtifactError))
        self.assertEqual(vendor_smoke.sniff_image(PNG_BYTES), ("png", 4, 6))

    def test_vendor_smoke_still_raises_its_own_error_type(self):
        with self.assertRaises(vendor_smoke.VendorSmokeError) as caught:
            vendor_smoke.sniff_image(b"not an image")
        self.assertIn("decode", str(caught.exception))

    def test_shared_decoder_rejects_structure_without_pixels(self):
        header_only = PNG_BYTES[:24]
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.sniff_image(header_only)
        jpeg = bytes.fromhex(
            "ffd8ffc0001108000100010301110002110003110000ffd9"
        )
        with self.assertRaises(artifacts.ArtifactError):
            artifacts.sniff_image(jpeg)

    def test_shared_decoder_bounds_bytes_pixels_and_each_side(self):
        oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * artifacts.MAX_ARTIFACT_BYTES
        with self.assertRaises(artifacts.ArtifactError) as caught:
            artifacts.sniff_image(oversized)
        self.assertIn("cap", str(caught.exception))

        bomb_header = struct.pack(">IIBBBBB", 8192, 8192, 8, 2, 0, 0, 0)
        bomb = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", bomb_header)
            + _chunk(b"IDAT", _zlib(b"\x00" * 16))
            + _chunk(b"IEND", b"")
        )
        with self.assertRaises(artifacts.ArtifactError) as caught:
            artifacts.sniff_image(bomb)
        self.assertIn("pixels", str(caught.exception))

        wide_header = struct.pack(">IIBBBBB", 8193, 1, 8, 2, 0, 0, 0)
        wide = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", wide_header)
            + _chunk(b"IDAT", _zlib(b"\x00" * 4))
            + _chunk(b"IEND", b"")
        )
        with self.assertRaises(artifacts.ArtifactError) as caught:
            artifacts.sniff_image(wide)
        self.assertIn("capped", str(caught.exception))

    def test_accepted_png_and_jpeg_are_reported_with_real_dimensions(self):
        self.assertEqual(artifacts.sniff_image(build_png(2, 3)), ("png", 2, 3))
        self.assertEqual(artifacts.sniff_image(PNG_BYTES), ("png", 4, 6))


def _chunk(kind, payload):
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _zlib(data):
    return zlib.compress(data)


if __name__ == "__main__":
    unittest.main()
