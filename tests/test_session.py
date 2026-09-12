import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import session as sess  # pyright: ignore[reportMissingImports]


class StalenessTests(unittest.TestCase):
    def test_fresh_report(self):
        text = "# Staleness Report\n**Date**: 2026-09-01\n"
        result = sess.parse_staleness(text, today=date(2026, 9, 12))
        self.assertEqual(result["last_run"], date(2026, 9, 1))
        self.assertEqual(result["days"], 11)
        self.assertFalse(result["stale"])

    def test_old_report(self):
        text = "**Date**: 2026-08-01\n"
        result = sess.parse_staleness(text, today=date(2026, 9, 12))
        self.assertTrue(result["stale"])

    def test_never_run(self):
        result = sess.parse_staleness("Not yet run\n", today=date(2026, 9, 12))
        self.assertIsNone(result["last_run"])
        self.assertIn("never", result["note"].lower())

    def test_missing_date(self):
        result = sess.parse_staleness("no date here", today=date(2026, 9, 12))
        self.assertIsNone(result["last_run"])


class SessionFileTests(unittest.TestCase):
    def test_write_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            sess.write_session("my-video", "http://127.0.0.1:8188", path=path)
            data = json.loads(path.read_text())
            self.assertEqual(data["active_project"], "my-video")
            self.assertEqual(data["comfyui_url"], "http://127.0.0.1:8188")
            self.assertIn("started", data)


class ConnectivityTests(unittest.TestCase):
    def test_unreachable_returns_none(self):
        self.assertIsNone(sess.check_comfyui("http://127.0.0.1:9"))


if __name__ == "__main__":
    unittest.main()
