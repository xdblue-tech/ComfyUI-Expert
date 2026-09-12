import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import scan_inventory as si


class OfflineScanTests(unittest.TestCase):
    def test_scan_offline_finds_models_and_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models" / "checkpoints").mkdir(parents=True)
            (root / "models" / "checkpoints" / "a.safetensors").write_bytes(b"x")
            (root / "models" / "loras").mkdir(parents=True)
            (root / "models" / "loras" / "b.safetensors").write_bytes(b"x")
            (root / "models" / "loras" / "ignore.txt").write_text("x")
            (root / "custom_nodes" / "SomeNode").mkdir(parents=True)
            (root / "custom_nodes" / "__pycache__").mkdir()
            (root / "custom_nodes" / "__pycache__" / "junk.pyc").write_bytes(b"x")

            data = si.scan_offline(root)

            self.assertEqual(data["mode"], "offline")
            self.assertEqual(data["models"]["checkpoints"], ["a.safetensors"])
            self.assertEqual(data["models"]["loras"], ["b.safetensors"])
            self.assertEqual(data["custom_nodes"], ["SomeNode"])
            self.assertEqual(data["node_classes"], [])
            self.assertEqual(data["comfyui_path"], str(root))


class OnlineScanTests(unittest.TestCase):
    def test_scan_online_shape(self):
        def fake_fetch(url, timeout=10):
            if url.endswith("/system_stats"):
                return {
                    "system": {"comfyui_version": "9.9.9"},
                    "devices": [
                        {
                            "name": "cuda:0 Fake",
                            "vram_total": 1000000000,
                            "vram_free": 500000000,
                        }
                    ],
                }
            if "/models/" in url:
                return ["m.safetensors"] if url.endswith("/checkpoints") else []
            if url.endswith("/object_info"):
                return {"KSampler": {}, "CLIPTextEncode": {}}
            raise AssertionError("unexpected url " + url)

        data = si.scan_online("http://example:8188", fetch=fake_fetch)

        self.assertEqual(data["mode"], "online")
        self.assertEqual(data["comfyui_version"], "9.9.9")
        self.assertEqual(data["system"]["gpu"], "cuda:0 Fake")
        self.assertEqual(data["system"]["vram_total_gb"], 1.0)
        self.assertEqual(data["models"]["checkpoints"], ["m.safetensors"])
        self.assertEqual(data["node_classes"], ["CLIPTextEncode", "KSampler"])
        self.assertEqual(data["custom_nodes"], [])


class PathDetectionTests(unittest.TestCase):
    def test_detect_uses_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models").mkdir()
            with mock.patch.dict(os.environ, {"COMFYUI_PATH": tmp}):
                self.assertEqual(si.detect_comfyui_path(), Path(tmp))

    def test_detect_skips_non_comfyui_dirs(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertIsNone(si.detect_comfyui_path(candidates=[Path(tmp)]))


if __name__ == "__main__":
    unittest.main()
