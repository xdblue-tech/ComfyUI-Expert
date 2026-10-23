"""Service settings that the transport depends on."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from distribute_agent.service.config import (
    DEFAULT_AUTH_AUDIENCE,
    DEFAULT_AUTH_ISSUER,
    DEFAULT_AUTH_SCOPE,
    ConfigError,
    ServiceConfig,
)

PACKAGE = Path(__file__).resolve().parents[2] / "distribute"


class ServiceConfigTests(unittest.TestCase):
    def config(self, tmp, **overrides):
        return ServiceConfig(
            package_root=PACKAGE,
            state_dir=Path(tmp) / "state",
            output_dir=Path(tmp) / "output",
            **overrides,
        )

    def test_only_loopback_hosts_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for host in ("127.0.0.1", "::1", "localhost"):
                with self.subTest(host=host):
                    self.assertEqual(self.config(tmp, host=host).host, host)
            for host in ("0.0.0.0", "192.168.1.10", "media.internal"):
                with self.subTest(host=host):
                    with self.assertRaises(ConfigError) as caught:
                        self.config(tmp, host=host)
                    self.assertIn("loopback", str(caught.exception))
            with self.assertRaises(ConfigError):
                self.config(tmp, host="")

    def test_host_and_origin_allowlists_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(
                tmp,
                allowed_hosts=["media.internal"],
                allowed_origins=("http://media.internal:8000",),
            )
            self.assertEqual(config.allowed_hosts, ("media.internal",))
            self.assertEqual(config.allowed_origins, ("http://media.internal:8000",))
            for label, value in (("allowed_hosts", [""]), ("allowed_origins", [3])):
                with self.subTest(field=label):
                    with self.assertRaises(ConfigError):
                        self.config(tmp, **{label: value})
            with self.assertRaises(ConfigError):
                self.config(tmp, allowed_hosts="media.internal")

    def test_auth_defaults_match_the_documented_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp)
        self.assertEqual(config.auth_issuer, DEFAULT_AUTH_ISSUER)
        self.assertEqual(config.auth_audience, DEFAULT_AUTH_AUDIENCE)
        self.assertEqual(config.auth_scope, DEFAULT_AUTH_SCOPE)
        self.assertEqual(config.auth_token_ttl_seconds, 7 * 24 * 60 * 60)
        self.assertFalse(config.generation_enabled)

    def test_auth_token_ttl_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            for ttl in (0, -1, True):
                with self.subTest(ttl=ttl):
                    with self.assertRaises(ConfigError):
                        self.config(tmp, auth_token_ttl_seconds=ttl)

    def test_state_and_output_stay_outside_the_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                ServiceConfig(
                    package_root=PACKAGE,
                    state_dir=PACKAGE / "state",
                    output_dir=Path(tmp) / "output",
                )
            with self.assertRaises(ConfigError):
                ServiceConfig(
                    package_root=PACKAGE,
                    state_dir=Path(tmp) / "state",
                    output_dir=Path(tmp) / "state",
                )
            with self.assertRaises(ConfigError):
                replace(self.config(tmp), output_dir=self.config(tmp).snapshot_dir)


if __name__ == "__main__":
    unittest.main()
