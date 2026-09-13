# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""An imported image's reference: parsed, allowed or refused, and resolved to a digest (E3-04)."""

from __future__ import annotations

import httpx
import pytest

from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.image_import import (
    DEFAULT_ALLOWED_REGISTRIES,
    ImageReference,
    image_registry_allowed,
    parse_image_reference,
    refuse_unless_allowed,
    resolve_image_digest,
)

DIGEST = "sha256:" + "a" * 64


class TestParsingAReference:
    def test_a_bare_name_and_tag_is_docker_hub_library(self) -> None:
        parsed = parse_image_reference("python:3.12-slim-bookworm")
        assert parsed == ImageReference("docker.io", "library/python", "3.12-slim-bookworm")

    def test_an_organizations_repository_stays_on_docker_hub(self) -> None:
        """`nginx/nginx-prometheus-exporter` names an org, not a registry: no dot, no port."""
        parsed = parse_image_reference("nginx/nginx-prometheus-exporter:1.1.0")
        assert parsed.registry == "docker.io"
        assert parsed.repository == "nginx/nginx-prometheus-exporter"

    def test_a_host_with_a_dot_is_a_registry(self) -> None:
        parsed = parse_image_reference("ghcr.io/owner/repo:v1")
        assert parsed == ImageReference("ghcr.io", "owner/repo", "v1")

    def test_a_host_with_a_port_is_a_registry(self) -> None:
        parsed = parse_image_reference("localhost:5000/repo:v1")
        assert parsed == ImageReference("localhost:5000", "repo", "v1")

    def test_localhost_with_no_port_is_a_registry(self) -> None:
        parsed = parse_image_reference("localhost/repo:v1")
        assert parsed == ImageReference("localhost", "repo", "v1")

    def test_a_reference_pinned_by_digest_carries_no_tag(self) -> None:
        parsed = parse_image_reference(f"python@{DIGEST}")
        assert parsed == ImageReference("docker.io", "library/python", "", DIGEST)
        assert parsed.pinned is True

    def test_no_tag_and_no_digest_means_latest(self) -> None:
        assert parse_image_reference("python").tag == "latest"

    def test_an_empty_reference_is_invalid(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            parse_image_reference("")
        assert raised.value.detail["field"] == "spec.build.image.reference"

    def test_a_malformed_digest_is_invalid(self) -> None:
        with pytest.raises(EnvironmentsError):
            parse_image_reference("python@not-a-digest")

    def test_str_prefers_the_digest_when_pinned(self) -> None:
        assert str(ImageReference("docker.io", "library/python", "", DIGEST)) == (
            f"docker.io/library/python@{DIGEST}"
        )

    def test_str_falls_back_to_the_tag(self) -> None:
        assert str(ImageReference("docker.io", "library/python", "3.12")) == (
            "docker.io/library/python:3.12"
        )


class TestTheAllowlist:
    def test_every_default_registry_is_allowed(self) -> None:
        for registry in DEFAULT_ALLOWED_REGISTRIES:
            assert image_registry_allowed(ImageReference(registry, "repo")) is True

    def test_an_unlisted_registry_is_not_allowed(self) -> None:
        assert image_registry_allowed(ImageReference("evil.example.com", "repo")) is False

    def test_refusing_one_names_the_registry_and_the_field(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            refuse_unless_allowed(ImageReference("evil.example.com", "repo"))
        assert raised.value.code.code == "DL_ENV_POLICY_DENIED"
        assert raised.value.detail["registry"] == "evil.example.com"
        assert raised.value.detail["field"] == "spec.build.image.reference"

    def test_an_allowed_one_refuses_nothing(self) -> None:
        refuse_unless_allowed(ImageReference("docker.io", "library/python"))

    def test_a_narrower_allowlist_can_be_passed_in(self) -> None:
        """The bootstrap default is not the only list a caller may check against —
        an organization's own, once E3-06 exists, is the same shape."""
        with pytest.raises(EnvironmentsError):
            refuse_unless_allowed(ImageReference("ghcr.io", "owner/repo"), allowlist=("docker.io",))

    def test_a_credential_lets_an_unlisted_registry_through(self) -> None:
        """E3-04's private half, spec-only for now: a credential reference is
        presumed a deliberate private registry, not something to block on a
        bootstrap list meant only for the public default."""
        refuse_unless_allowed(
            ImageReference("registry.example.com", "team/env"), has_credential=True
        )

    def test_with_no_credential_the_allowlist_still_applies(self) -> None:
        with pytest.raises(EnvironmentsError):
            refuse_unless_allowed(
                ImageReference("registry.example.com", "team/env"), has_credential=False
            )


class TestResolvingADigest:
    def test_a_pinned_reference_needs_no_network_call(self) -> None:
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise AssertionError("a pinned reference must never be looked up")

        image = ImageReference("docker.io", "library/python", "", DIGEST)
        transport = httpx.MockTransport(unreachable)
        assert resolve_image_digest(image, transport=transport) == DIGEST

    def test_an_anonymous_registry_answers_the_header_directly(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "manifests/1.1.0" in str(request.url)
            return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})

        image = ImageReference("ghcr.io", "owner/repo", "1.1.0")
        transport = httpx.MockTransport(handler)
        assert resolve_image_digest(image, transport=transport) == DIGEST

    def test_a_challenge_is_answered_with_an_anonymous_token(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if "auth.docker.io" in str(request.url):
                assert request.url.params["scope"] == "repository:library/python:pull"
                return httpx.Response(200, json={"token": "anon-token"})
            if "Authorization" not in request.headers:
                return httpx.Response(
                    401,
                    headers={
                        "WWW-Authenticate": (
                            'Bearer realm="https://auth.docker.io/token",'
                            'service="registry.docker.io",'
                            'scope="repository:library/python:pull"'
                        )
                    },
                )
            assert request.headers["Authorization"] == "Bearer anon-token"
            return httpx.Response(200, headers={"Docker-Content-Digest": DIGEST})

        image = ImageReference("docker.io", "library/python", "3.12-slim-bookworm")
        transport = httpx.MockTransport(handler)
        assert resolve_image_digest(image, transport=transport) == DIGEST
        assert any("registry-1.docker.io" in url for url in calls)

    def test_a_missing_image_is_a_provider_error_naming_it(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        image = ImageReference("ghcr.io", "owner/nothing-here", "v1")
        transport = httpx.MockTransport(handler)
        with pytest.raises(EnvironmentsError) as raised:
            resolve_image_digest(image, transport=transport)
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"
        assert "owner/nothing-here" in raised.value.message

    def test_no_digest_header_falls_back_to_hashing_the_manifest(self) -> None:
        body = b'{"schemaVersion":2}'
        import hashlib

        expected = "sha256:" + hashlib.sha256(body).hexdigest()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body)

        image = ImageReference("ghcr.io", "owner/repo", "v1")
        transport = httpx.MockTransport(handler)
        assert resolve_image_digest(image, transport=transport) == expected

    def test_an_unanswerable_challenge_is_a_provider_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401)

        image = ImageReference("ghcr.io", "owner/repo", "v1")
        transport = httpx.MockTransport(handler)
        with pytest.raises(EnvironmentsError) as raised:
            resolve_image_digest(image, transport=transport)
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"
