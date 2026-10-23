"""JWT issuance and verification, including the native verifier's exp behavior."""

import asyncio
import time
import unittest

from joserfc import jwt
from joserfc.jwk import OctKey

from distribute_agent.service.auth import (
    AuthConfigError,
    MIN_SECRET_BYTES,
    build_verifier,
    issue_token,
    load_secret,
    validate_secret,
)

SECRET = "unit-test-secret-" + "a" * 32


def verify(token, secret=SECRET):
    return asyncio.run(build_verifier(secret).verify_token(token))


class IssueTokenTests(unittest.TestCase):
    def test_issued_token_carries_every_required_claim(self):
        before = int(time.time())
        token = issue_token(SECRET, subject="owner", ttl_seconds=3600)
        access = verify(token)
        self.assertIsNotNone(access)
        self.assertEqual(access.subject, "owner")
        self.assertEqual(access.client_id, "owner")
        self.assertEqual(access.scopes, ["media:access"])
        self.assertEqual(access.claims["iss"], "mediagen")
        self.assertEqual(access.claims["aud"], "mediagen-mcp")
        self.assertEqual(access.claims["scope"], "media:access")
        self.assertGreaterEqual(access.claims["iat"], before)
        self.assertEqual(access.claims["exp"] - access.claims["iat"], 3600)
        self.assertEqual(access.expires_at, access.claims["exp"])
        self.assertEqual(len(access.claims["jti"]), 16)

    def test_default_token_lifetime_is_seven_days(self):
        token = issue_token(SECRET)
        claims = verify(token).claims
        self.assertEqual(claims["exp"] - claims["iat"], 7 * 24 * 60 * 60)

    def test_expired_token_is_rejected(self):
        token = issue_token(SECRET, ttl_seconds=3600, now=time.time() - 7200)
        self.assertIsNone(verify(token))

    def test_missing_scope_is_rejected(self):
        token = issue_token(SECRET, scope="media:read")
        self.assertIsNone(verify(token))

    def test_wrong_issuer_is_rejected(self):
        self.assertIsNone(verify(issue_token(SECRET, issuer="someone-else")))

    def test_wrong_audience_is_rejected(self):
        self.assertIsNone(verify(issue_token(SECRET, audience="another-service")))

    def test_signature_from_another_secret_is_rejected(self):
        other = "another-secret-" + "b" * 32
        self.assertIsNone(verify(issue_token(other)))

    def test_bad_subject_and_ttl_are_refused_before_minting(self):
        for subject, ttl in (("", 3600), ("owner", 0), ("owner", -5), ("owner", True)):
            with self.subTest(subject=subject, ttl=ttl):
                with self.assertRaises(AuthConfigError):
                    issue_token(SECRET, subject=subject, ttl_seconds=ttl)

    def test_native_verifier_accepts_a_hand_built_token_without_exp(self):
        # FastMCP 4.1.3 checks exp only when present. issue_token always writes
        # one; this test pins the native behavior the module docstring warns about.
        claims = {
            "sub": "owner",
            "iss": "mediagen",
            "aud": "mediagen-mcp",
            "scope": "media:access",
            "iat": int(time.time()),
        }
        token = jwt.encode(
            {"alg": "HS256", "typ": "JWT"},
            claims,
            OctKey.import_key(SECRET),
            algorithms=["HS256"],
        )
        self.assertIsNotNone(verify(token))


class SecretTests(unittest.TestCase):
    def test_secret_must_come_from_the_named_environment_variable(self):
        with self.assertRaises(AuthConfigError) as caught:
            load_secret(env={}, secret_env="MEDIAGEN_JWT_SECRET")
        self.assertIn("MEDIAGEN_JWT_SECRET", str(caught.exception))
        with self.assertRaises(AuthConfigError):
            load_secret(env={"MEDIAGEN_JWT_SECRET": "   "}, secret_env="MEDIAGEN_JWT_SECRET")

    def test_short_secret_is_refused(self):
        with self.assertRaises(AuthConfigError) as caught:
            load_secret(env={"S": "too-short"}, secret_env="S")
        self.assertIn(str(MIN_SECRET_BYTES), str(caught.exception))
        with self.assertRaises(AuthConfigError):
            validate_secret("é" * 8)
        with self.assertRaises(AuthConfigError):
            build_verifier("short")

    def test_long_secret_is_accepted_unchanged(self):
        secret = "s" * MIN_SECRET_BYTES
        self.assertEqual(load_secret(env={"S": secret}, secret_env="S"), secret)
        self.assertIsNotNone(verify(issue_token(secret), secret=secret))


if __name__ == "__main__":
    unittest.main()
