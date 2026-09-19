# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""An organization's own policy, enforced on the spec at `validate` (PLAN_ENV.md E3-06).

The box's own `Done when`: "an index an organization admin denied yields
`DL_ENV_POLICY_DENIED` at `validate`". Bases, registries and packages are the
same shape and covered alongside it; licences are checked once an artifact's
SBOM exists, not here — see `test_environment_attest.py` and
`test_environment_organization_policy.py`.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.policy import UNRESTRICTED_POLICY, EnvironmentsPolicy
from code_sandboxes.environments.spec import (
    parse_environment,
    spec_findings,
    validate_environment,
)

EXAMPLE = """
apiVersion: environments.datalayer.io/v1alpha1
kind: Environment
metadata:
  name: geospatial-analysis
spec:
  contract: sandbox-contract/v1
  language:
    name: python
    version: "3.13"
  base:
    ref: datalayer/python-cpu
    channel: "2026.08"
  platform:
    architecture: linux/amd64
  packages:
    python:
      manager: uv
      dependencies:
        - geopandas==1.1.1
      indexes:
        - https://pypi.org/simple
  compatibility:
    variants:
      required: [datalayer]
  build:
    source: packages
"""


def document() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE)


def mutated(path: str, value: Any) -> dict[str, Any]:
    data = document()
    target: Any = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    target[parts[-1]] = value
    return data


class TestUnrestrictedByDefault:
    """No organization has written this section: every check here is a no-op."""

    def test_the_example_validates_with_no_policy_at_all(self) -> None:
        environment = validate_environment(EXAMPLE)
        assert spec_findings(environment) == []

    def test_the_default_policy_argument_is_unrestricted(self) -> None:
        environment = validate_environment(EXAMPLE, policy=UNRESTRICTED_POLICY)
        assert environment.metadata.name == "geospatial-analysis"


class TestAnIndexAnOrganizationDenies:
    """The box's own `Done when`, verbatim."""

    def test_an_undenied_index_still_validates(self) -> None:
        policy = EnvironmentsPolicy(allowed_indexes=("https://pypi.org/simple",))
        environment = validate_environment(EXAMPLE, policy=policy)
        assert spec_findings(environment, policy=policy) == []

    def test_an_index_not_on_the_allowlist_is_policy_denied_at_validate(self) -> None:
        policy = EnvironmentsPolicy(allowed_indexes=("https://pypi.org/simple",))
        doc = mutated("spec.packages.python.indexes", ["https://evil.example/simple"])
        with pytest.raises(EnvironmentsError) as raised:
            validate_environment(doc, policy=policy)
        assert raised.value.code.code == "DL_ENV_POLICY_DENIED"
        assert "https://evil.example/simple" in raised.value.message

    def test_the_finding_is_on_the_right_field_and_names_the_allowed_ones(self) -> None:
        policy = EnvironmentsPolicy(allowed_indexes=("https://pypi.org/simple",))
        doc = mutated(
            "spec.packages.python.indexes",
            ["https://pypi.org/simple", "https://evil.example/simple"],
        )
        environment = parse_environment(doc)
        findings = spec_findings(environment, policy=policy)
        [finding] = [f for f in findings if f.code.code == "DL_ENV_POLICY_DENIED"]
        assert finding.field == "spec.packages.python.indexes[1]"
        assert "https://pypi.org/simple" in finding.message


class TestABaseAnOrganizationDenies:
    def test_an_approved_base_the_organization_also_allows_passes(self) -> None:
        policy = EnvironmentsPolicy(allowed_bases=("datalayer/python-cpu",))
        environment = validate_environment(EXAMPLE, policy=policy)
        assert spec_findings(environment, policy=policy) == []

    def test_an_approved_base_the_organization_does_not_list_is_denied(self) -> None:
        """The platform approves it; the organization has not, and its
        policy narrows what the platform allows rather than widening it."""
        policy = EnvironmentsPolicy(allowed_bases=("datalayer/python-cuda",))
        with pytest.raises(EnvironmentsError) as raised:
            validate_environment(EXAMPLE, policy=policy)
        assert raised.value.code.code == "DL_ENV_POLICY_DENIED"
        assert "datalayer/python-cpu" in raised.value.message

    def test_an_image_or_dockerfile_source_is_not_checked_against_it(self) -> None:
        """`spec.base` names nothing for either source (the platform's own
        rule, above this one) — an organization's base policy has nothing to
        apply to and must not invent a refusal for a field these sources
        never use."""
        policy = EnvironmentsPolicy(allowed_bases=("datalayer/python-cuda",))
        doc = mutated("spec.build.source", "dockerfile")
        doc["spec"]["build"]["dockerfile"] = {"content": "FROM datalayer/python-cpu:2026.08\n"}
        environment = parse_environment(doc)
        findings = spec_findings(environment, policy=policy)
        assert not any(f.field == "spec.base.ref" for f in findings)


class TestARegistryAnOrganizationAdds:
    """The one dimension that widens rather than narrows (`_image_findings`'
    own docstring): the platform's public bootstrap stays reachable, and an
    organization's own private registries are added to it, never instead of
    it."""

    def _image_doc(self, reference: str) -> dict[str, Any]:
        doc = mutated("spec.build.source", "image")
        doc["spec"]["build"]["image"] = {"reference": reference}
        del doc["spec"]["packages"]
        return doc

    def test_the_platform_bootstrap_still_passes_with_no_organization_list(self) -> None:
        environment = parse_environment(self._image_doc("docker.io/library/python:3.12-slim"))
        assert not any(f.code.code == "DL_ENV_POLICY_DENIED" for f in spec_findings(environment))

    def test_an_organizations_own_private_registry_is_added_not_substituted(self) -> None:
        policy = EnvironmentsPolicy(allowed_registries=("registry.acme.internal",))
        environment = parse_environment(self._image_doc("docker.io/library/python:3.12-slim"))
        findings = spec_findings(environment, policy=policy)
        assert not any(f.code.code == "DL_ENV_POLICY_DENIED" for f in findings)

    def test_a_registry_on_neither_list_is_denied_naming_both(self) -> None:
        policy = EnvironmentsPolicy(allowed_registries=("registry.acme.internal",))
        environment = parse_environment(
            self._image_doc("registry.other.example/someone/image:latest")
        )
        findings = spec_findings(environment, policy=policy)
        [finding] = [f for f in findings if f.code.code == "DL_ENV_POLICY_DENIED"]
        assert "registry.acme.internal" in finding.message
        assert "docker.io" in finding.message  # the bootstrap default, still named


class TestAPackageAnOrganizationDenies:
    def test_an_allowed_package_passes(self) -> None:
        policy = EnvironmentsPolicy(allowed_packages=("geopandas",))
        environment = validate_environment(EXAMPLE, policy=policy)
        assert spec_findings(environment, policy=policy) == []

    def test_a_package_off_the_allowlist_is_denied_naming_it(self) -> None:
        policy = EnvironmentsPolicy(allowed_packages=("numpy",))
        with pytest.raises(EnvironmentsError) as raised:
            validate_environment(EXAMPLE, policy=policy)
        assert raised.value.code.code == "DL_ENV_POLICY_DENIED"
        assert "geopandas" in raised.value.message

    def test_constraints_are_not_checked_the_dependency_list_is(self) -> None:
        """`spec.packages.python.constraints` narrows a version, never adds a
        package that is actually installed on its own — nothing to deny
        there that is not already covered by the dependency it constrains."""
        policy = EnvironmentsPolicy(allowed_packages=("geopandas",))
        doc = mutated("spec.packages.python.constraints", ["numpy<2"])
        environment = parse_environment(doc)
        findings = spec_findings(environment, policy=policy)
        assert not any(
            f.code.code == "DL_ENV_POLICY_DENIED" and "numpy" in f.message for f in findings
        )
