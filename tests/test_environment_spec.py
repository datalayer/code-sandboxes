# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The canonical Environment specification, checked rule by rule.

The example is PLAN_ENV.md §4.1's, with a concrete build secret id where the
plan elides one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from code_sandboxes.environments import errors
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.schema import main as schema_main
from code_sandboxes.environments.schema import schema_text
from code_sandboxes.environments.spec import (
    Environment,
    parse_environment,
    spec_digest,
    spec_findings,
    validate_environment,
)

REPOSITORY = Path(__file__).resolve().parents[1]

EXAMPLE = """
apiVersion: environments.datalayer.io/v1alpha1
kind: Environment
metadata:
  name: geospatial-analysis
  title: Geospatial analysis
  labels:
    team: research
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
        - rasterio==1.4.3
      constraints: []
      indexes:
        - https://pypi.org/simple
    system:
      apt:
        - gdal-bin
  files:
    - path: /home/datalayer/content/.jupyter/jupyter_config.py
      contentRef: blob://environments/jupyter_config.py
  env:
    GDAL_DATA: /usr/share/gdal
  commands:
    postInstall:
      - "python -c 'import geopandas'"
  buildSecrets:
    - id: dlsec_01J8ZK
      mountAs: env
      name: PIP_EXTRA_INDEX_TOKEN
  resources:
    accelerator: none
    hints:
      cpu: 2
      memoryGi: 8
      diskGi: 10
  compatibility:
    variants:
      required: [datalayer]
      optional: [e2b, daytona, modal]
    regions: [eu-west, us-east]
  build:
    source: packages
"""


def document() -> dict[str, Any]:
    return yaml.safe_load(EXAMPLE)


def mutated(path: str, value: Any) -> dict[str, Any]:
    """The example with one field set; a path of dotted keys and list indexes."""
    data = document()
    target: Any = data
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if part.isdigit() else target[part]
    last = parts[-1]
    if last.isdigit():
        target[int(last)] = value
    else:
        target[last] = value
    return data


def _reversed_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reversed_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reversed_keys(item) for item in value]
    return value


def test_the_plan_example_is_a_valid_environment() -> None:
    environment = validate_environment(EXAMPLE)
    assert environment.metadata.name == "geospatial-analysis"
    assert spec_findings(environment) == []


def test_the_example_round_trips_with_its_camel_case_names() -> None:
    environment = parse_environment(document())
    dumped = environment.model_dump(by_alias=True, mode="json")
    for name in ("apiVersion", "buildSecrets"):
        assert name in dumped or name in dumped["spec"]
    assert "postInstall" in dumped["spec"]["commands"]
    assert "memoryGi" in dumped["spec"]["resources"]["hints"]
    assert "contentRef" in dumped["spec"]["files"][0]
    again = parse_environment(yaml.safe_dump(dumped))
    assert again == environment
    assert again.model_dump(by_alias=True, mode="json") == dumped


def test_a_json_document_parses_like_yaml() -> None:
    assert parse_environment(json.dumps(document())) == parse_environment(EXAMPLE)


def test_the_digest_does_not_depend_on_key_order_or_written_defaults() -> None:
    digest = spec_digest(parse_environment(document()))
    assert digest.startswith("sha256:")
    assert spec_digest(parse_environment(_reversed_keys(document()))) == digest
    without_defaults = document()
    del without_defaults["spec"]["platform"]
    del without_defaults["spec"]["build"]
    del without_defaults["spec"]["contract"]
    assert spec_digest(parse_environment(without_defaults)) == digest


def test_metadata_is_not_part_of_the_digest() -> None:
    renamed = mutated("metadata.title", "Something else")
    assert spec_digest(parse_environment(renamed)) == spec_digest(parse_environment(document()))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("spec.packages.python.dependencies.0", "geopandas==1.1.2"),
        ("spec.packages.system.apt.0", "gdal-data"),
        ("spec.env", {"GDAL_DATA": "/opt/gdal"}),
        ("spec.compatibility.variants.optional", ["e2b"]),
        ("spec.resources.hints", {"cpu": 4}),
        ("spec.base.channel", "2026.09"),
        ("spec.commands.postInstall", []),
    ],
)
def test_any_change_of_value_changes_the_digest(path: str, value: Any) -> None:
    assert spec_digest(parse_environment(mutated(path, value))) != spec_digest(
        parse_environment(document())
    )


def _codes(data: dict[str, Any]) -> dict[str, str]:
    return {finding.field: finding.code.code for finding in spec_findings(parse_environment(data))}


INVALID = "DL_ENV_SPEC_INVALID"
UNSUPPORTED = "DL_ENV_CAPABILITY_UNSUPPORTED"


@pytest.mark.parametrize(
    ("path", "value", "field", "code"),
    [
        ("metadata.name", "Geospatial", "metadata.name", INVALID),
        ("metadata.name", "x" * 64, "metadata.name", INVALID),
        ("metadata.name", "dl-geo", "metadata.name", INVALID),
        ("spec.contract", "sandbox-contract/v9", "spec.contract", UNSUPPORTED),
        ("spec.base.ref", "python", "spec.base.ref", INVALID),
        ("spec.language.version", "3.9", "spec.language.version", INVALID),
        ("spec.build.source", "dockerfile", "spec.build.source", UNSUPPORTED),
        ("spec.packages.python.manager", "conda", "spec.packages.python.manager", UNSUPPORTED),
        (
            "spec.packages.python.dependencies.0",
            "geopandas>=>1",
            "spec.packages.python.dependencies[0]",
            INVALID,
        ),
        (
            "spec.packages.python.indexes.0",
            "http://pypi.org/simple",
            "spec.packages.python.indexes[0]",
            INVALID,
        ),
        (
            "spec.packages.python.indexes.0",
            "https://user:hunter2@pypi.acme.dev/simple",
            "spec.packages.python.indexes[0]",
            INVALID,
        ),
        ("spec.packages.system.apt.0", "GDAL BIN", "spec.packages.system.apt[0]", INVALID),
        ("spec.files.0.path", "content/config.py", "spec.files[0].path", INVALID),
        ("spec.files.0.path", "/opt/datalayer/bin/doctor", "spec.files[0].path", INVALID),
        ("spec.files.0.contentRef", "ftp://files/config.py", "spec.files[0].contentRef", INVALID),
        ("spec.files.0.sizeBytes", 2 * 1024 * 1024, "spec.files[0].sizeBytes", INVALID),
        ("spec.env", {"PIP_INDEX_TOKEN": "abc"}, "spec.env.PIP_INDEX_TOKEN", INVALID),
        ("spec.env", {"AWS_REGION": "AKIAIOSFODNN7EXAMPLE"}, "spec.env.AWS_REGION", INVALID),
        ("spec.env", {"bad name": "x"}, "spec.env.bad name", INVALID),
        (
            "spec.env",
            {"GDAL_DATA": "/opt/gdal\nRUN whoami"},
            "spec.env.GDAL_DATA",
            INVALID,
        ),
        ("spec.commands.postInstall", ["  "], "spec.commands.postInstall[0]", INVALID),
        (
            "spec.commands.postInstall",
            ["echo hi\nUSER root"],
            "spec.commands.postInstall[0]",
            INVALID,
        ),
        (
            "spec.buildSecrets",
            [{"id": "dlsec_1", "name": "A"}, {"id": "dlsec_1", "name": "B"}],
            "spec.buildSecrets",
            INVALID,
        ),
        ("spec.resources.sizeClass", "huge", "spec.resources.sizeClass", INVALID),
        ("spec.resources.sizeClass", "gpu-small", "spec.resources.accelerator", INVALID),
        (
            "spec.resources.accelerator",
            {"type": "nvidia-a100"},
            "spec.resources.sizeClass",
            INVALID,
        ),
        (
            "spec.compatibility.variants.required",
            [],
            "spec.compatibility.variants.required",
            INVALID,
        ),
        (
            "spec.compatibility.variants.optional",
            ["kaggle"],
            "spec.compatibility.variants.optional",
            INVALID,
        ),
        (
            "spec.compatibility.variants.optional",
            ["datalayer"],
            "spec.compatibility.variants",
            INVALID,
        ),
        ("spec.compatibility.regions", ["EU West"], "spec.compatibility.regions[0]", INVALID),
    ],
)
def test_each_rule_names_its_field_and_its_code(
    path: str, value: Any, field: str, code: str
) -> None:
    assert _codes(mutated(path, value)).get(field) == code


def test_a_gpu_class_needs_a_cuda_base() -> None:
    data = mutated("spec.resources.sizeClass", "gpu-small")
    data["spec"]["resources"]["accelerator"] = {"type": "nvidia-a100", "count": 1}
    assert _codes(data) == {"spec.base.ref": INVALID}
    data["spec"]["base"]["ref"] = "datalayer/python-cuda"
    assert _codes(data) == {}


def test_all_baked_files_together_are_capped() -> None:
    data = document()
    data["spec"]["files"] = [
        {
            "path": f"/home/datalayer/content/f{index}",
            "contentRef": f"blob://f{index}",
            "sizeBytes": 1024 * 1024,
        }
        for index in range(9)
    ]
    assert _codes(data) == {"spec.files": INVALID}


def test_an_invalid_field_outranks_something_unsupported() -> None:
    data = mutated("spec.build.source", "dockerfile")
    assert _codes(data) == {"spec.build.source": UNSUPPORTED}
    with pytest.raises(EnvironmentsError) as unsupported:
        validate_environment(data)
    assert unsupported.value.code is errors.CAPABILITY_UNSUPPORTED

    data["metadata"]["name"] = "Nope"
    with pytest.raises(EnvironmentsError) as invalid:
        validate_environment(data)
    assert invalid.value.code is errors.SPEC_INVALID
    assert {finding["field"] for finding in invalid.value.detail["findings"]} == {
        "metadata.name",
        "spec.build.source",
    }


# -- Dependency files (E3-01) --------------------------------------------------


def a_dependency_file_document(**dependency_file: Any) -> dict[str, Any]:
    data = mutated("spec.build.source", "dependencyFile")
    data["spec"]["build"]["dependencyFile"] = {
        "sourceFormat": "requirements",
        "content": "geopandas==1.1.1\n",
        **dependency_file,
    }
    return data


class TestDependencyFiles:
    def test_a_requirements_file_is_supported_now(self) -> None:
        assert _codes(a_dependency_file_document()) == {}

    def test_a_pyproject_file_with_its_lock_is_supported(self) -> None:
        data = a_dependency_file_document(
            sourceFormat="pyproject",
            content='[project]\nname = "x"\ndependencies = ["geopandas==1.1.1"]\n',
            lockContent="# a uv.lock\n",
        )
        assert _codes(data) == {}

    def test_naming_no_dependency_file_at_all_is_invalid(self) -> None:
        data = mutated("spec.build.source", "dependencyFile")
        assert _codes(data) == {"spec.build.dependencyFile": INVALID}

    def test_an_unsupported_source_format_is_invalid(self) -> None:
        """`sourceFormat` is a closed set: pydantic refuses it before a
        finding would even run, the same as any other field's literal type."""
        data = a_dependency_file_document(sourceFormat="setup.py")
        with pytest.raises(EnvironmentsError) as refused:
            parse_environment(data)
        assert refused.value.code is errors.SPEC_INVALID
        assert "spec.build.dependencyFile.sourceFormat" in {
            finding["field"] for finding in refused.value.detail["findings"]
        }

    def test_empty_content_is_invalid(self) -> None:
        data = a_dependency_file_document(content="   \n")
        assert _codes(data) == {"spec.build.dependencyFile.content": INVALID}

    def test_content_over_the_byte_cap_is_invalid(self) -> None:
        from code_sandboxes.environments.spec import MAX_DEPENDENCY_FILE_BYTES

        data = a_dependency_file_document(content="x" * (MAX_DEPENDENCY_FILE_BYTES + 1))
        assert _codes(data) == {"spec.build.dependencyFile.content": INVALID}

    def test_a_pyproject_source_with_no_lock_is_invalid(self) -> None:
        data = a_dependency_file_document(
            sourceFormat="pyproject", content='[project]\nname = "x"\n'
        )
        assert _codes(data) == {"spec.build.dependencyFile.lockContent": INVALID}

    def test_a_requirements_source_with_a_lock_is_invalid(self) -> None:
        """`lockContent` is only read for `pyproject`; a `requirements` source resolves fresh."""
        data = a_dependency_file_document(lockContent="# unexpected\n")
        assert _codes(data) == {"spec.build.dependencyFile.lockContent": INVALID}

    def test_an_unparseable_requirement_line_is_invalid_and_named(self) -> None:
        data = a_dependency_file_document(content="geopandas>=>1\n")
        assert _codes(data) == {"spec.build.dependencyFile.content[0]": INVALID}

    def test_a_comment_and_a_blank_line_are_not_requirements(self) -> None:
        data = a_dependency_file_document(content="# a comment\n\ngeopandas==1.1.1\n")
        assert _codes(data) == {}


# -- An imported image (E3-04) -------------------------------------------------


def an_image_document(**image: Any) -> dict[str, Any]:
    data = mutated("spec.build.source", "image")
    data["spec"]["build"]["image"] = {"reference": "python:3.12-slim-bookworm", **image}
    return data


DENIED = "DL_ENV_POLICY_DENIED"


class TestImageSources:
    def test_a_public_image_is_supported_now(self) -> None:
        assert _codes(an_image_document()) == {}

    def test_pinned_by_digest_is_supported_too(self) -> None:
        data = an_image_document(reference="python@" + "sha256:" + "a" * 64)
        assert _codes(data) == {}

    def test_naming_no_image_at_all_is_invalid(self) -> None:
        data = mutated("spec.build.source", "image")
        assert _codes(data) == {"spec.build.image": INVALID}

    def test_an_empty_reference_is_invalid(self) -> None:
        data = an_image_document(reference="   ")
        assert _codes(data) == {"spec.build.image.reference": INVALID}

    def test_an_unparseable_reference_is_invalid(self) -> None:
        data = an_image_document(reference="not a reference@@")
        assert _codes(data) == {"spec.build.image.reference": INVALID}

    def test_a_registry_off_the_allowlist_is_policy_denied(self) -> None:
        data = an_image_document(reference="evil.example.com/x:y")
        assert _codes(data) == {"spec.build.image.reference": DENIED}

    def test_the_approved_base_is_not_checked_for_this_source(self) -> None:
        """`spec.base` is meaningless for an import: the image is the base,
        so an unapproved or mismatched one refuses nothing here."""
        data = an_image_document()
        data["spec"]["base"] = {"ref": "not-approved-at-all", "channel": "n/a"}
        assert _codes(data) == {}

    def test_a_registry_off_the_allowlist_with_a_credential_is_accepted(self) -> None:
        """A credential reference is E3-04's private half: spec-only for now
        (E3-05 resolves nothing yet), but it is what lets the registry
        through `spec_findings` at all."""
        data = an_image_document(
            reference="registry.example.com/team/env:v1",
            credentialSecretId="dlsec_01J8ZK",
        )
        assert _codes(data) == {}

    def test_a_malformed_credential_id_is_invalid(self) -> None:
        """`credentialSecretId` follows `BuildSecret.id`'s own pattern."""
        data = an_image_document(credentialSecretId="not-a-secret-id")
        with pytest.raises(EnvironmentsError) as refused:
            parse_environment(data)
        assert refused.value.code is errors.SPEC_INVALID
        assert "spec.build.image.credentialSecretId" in {
            finding["field"] for finding in refused.value.detail["findings"]
        }


class TestParsingARequirementsFile:
    def test_comments_and_blank_lines_are_dropped(self) -> None:
        from code_sandboxes.environments.spec import parse_requirements_txt

        text = "# a header\n\ngeopandas==1.1.1  # inline\n\nrasterio==1.4.3\n"
        assert parse_requirements_txt(text) == ["geopandas==1.1.1", "rasterio==1.4.3"]

    def test_a_pip_option_line_is_not_a_requirement(self) -> None:
        from code_sandboxes.environments.spec import parse_requirements_txt

        text = "-r other.txt\n--index-url https://example/simple\nsix==1.16.0\n"
        assert parse_requirements_txt(text) == ["six==1.16.0"]

    def test_an_empty_file_has_no_requirements(self) -> None:
        from code_sandboxes.environments.spec import parse_requirements_txt

        assert parse_requirements_txt("\n\n# only comments\n") == []

    def test_a_direct_references_own_fragment_is_not_a_comment(self) -> None:
        """`#egg=` and `#sha256=` have no whitespace in front of them: they
        are part of the requirement, not something to drop as a comment."""
        from code_sandboxes.environments.spec import parse_requirements_txt

        text = (
            "geopandas @ https://example.com/geopandas.whl#sha256=" + "a" * 64 + "\n"
            "git+https://example.com/rasterio.git#egg=rasterio  # inline note\n"
        )
        assert parse_requirements_txt(text) == [
            "geopandas @ https://example.com/geopandas.whl#sha256=" + "a" * 64,
            "git+https://example.com/rasterio.git#egg=rasterio",
        ]


@pytest.mark.parametrize(
    ("data", "field"),
    [
        (mutated("spec.packges", {}), "spec.packges"),
        (mutated("spec.buildSecrets.0.id", "secret-1"), "spec.buildSecrets[0].id"),
        (mutated("apiVersion", "environments.datalayer.io/v2"), "apiVersion"),
        (mutated("spec.language.version", "3"), "spec.language.version"),
    ],
)
def test_a_document_the_models_refuse_is_invalid_with_its_field(
    data: dict[str, Any], field: str
) -> None:
    with pytest.raises(EnvironmentsError) as refused:
        parse_environment(data)
    assert refused.value.code is errors.SPEC_INVALID
    assert field in {finding["field"] for finding in refused.value.detail["findings"]}


def test_the_published_schema_is_what_the_models_export() -> None:
    path = REPOSITORY / "schemas" / "environment-v1alpha1.json"
    assert path.read_text(encoding="utf-8") == schema_text()
    assert schema_main(["--check", str(path)]) == 0


def test_the_schema_check_fails_on_drift(tmp_path: Path) -> None:
    stale = tmp_path / "environment.json"
    stale.write_text(schema_text().replace('"Environment"', '"Environmint"', 1), encoding="utf-8")
    assert schema_main(["--check", str(stale)]) == 1
    assert schema_main(["--write", str(stale)]) == 0
    assert schema_main(["--check", str(stale)]) == 0


def test_the_example_validates_against_the_published_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(schema_text())
    jsonschema.validate(document(), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mutated("spec.packges", {}), schema)


def test_a_spec_is_digested_the_same_from_the_environment_or_the_spec() -> None:
    environment = parse_environment(copy.deepcopy(document()))
    assert isinstance(environment, Environment)
    assert spec_digest(environment) == spec_digest(environment.spec)
