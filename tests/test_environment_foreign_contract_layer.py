# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A foreign image made a sandbox: `sandbox-contract/v1` layered on at build.

What is rendered and refused, always. With `DATALAYER_ENVIRONMENT_DOCKER_TESTS`
set, the real thing: the resolver's and the builder's own Dockerfiles built by
Docker on real public images, the doctor run in the result, and a real Jupyter
server started by the image's own entrypoint, whose kernel runs a task's code.
A local image stands where the approved base does, since it is what the layer
copies from: built here from this package's doctor, constraints and wheelhouse,
and the plane's start script and config, which `PLANE_DOCKERFILES` names.
"""

# ruff: noqa: S603, S607
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import httpx
import pytest

from code_sandboxes.environments.adapters.datalayer import (
    CONTRACT_LAYER_FILES,
    TASK_WORKDIR_FILE,
    TINI_SHA256,
    Builder,
)
from code_sandboxes.environments.builders import BuildRequest
from code_sandboxes.environments.contract import (
    foreign_dockerfile_findings,
    pin_foreign_dockerfile_base,
)
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.image_import import DEFAULT_ALLOWED_REGISTRIES
from code_sandboxes.environments.resolve import (
    WHEELHOUSE_PATH,
    BuildkitResolveRunner,
    ResolveOutcome,
    ResolveRequest,
    contract_base_in,
    foreign_bootstrap_lines,
    lock_document,
    merge_requirements,
    resolve_environment,
)
from code_sandboxes.environments.spec import parse_environment, spec_findings

BASE = "datalayer/python-cpu@sha256:" + "a" * 64
FOREIGN = "docker.io/library/python@sha256:" + "b" * 64
LOCK = f"# datalayer-contract-base: {BASE}\nipykernel==7.3.0 --hash=sha256:{'c' * 64}\n"
ENVIRONMENTS = Path(__file__).parents[1] / "code_sandboxes" / "environments"


def plane_of(environ):
    """The plane's dockerfiles, when the variable names a directory that exists."""
    named = environ.get("PLANE_DOCKERFILES", "")
    return Path(named) if named and Path(named).is_dir() else None


PLANE = plane_of(os.environ)
REAL = bool(os.environ.get("DATALAYER_ENVIRONMENT_DOCKER_TESTS"))


_TRANSPORT = httpx.MockTransport(
    lambda request: httpx.Response(
        200, json={"token": "t"}, headers={"Docker-Content-Digest": "sha256:" + "b" * 64}
    )
)


def document(build):
    return {
        "apiVersion": "environments.datalayer.io/v1alpha1",
        "kind": "Environment",
        "metadata": {"name": "task-one"},
        "spec": {
            "language": {"name": "python", "version": "3.12"},
            "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
            "build": build,
        },
    }


def image_source(reference):
    return document({"source": "image", "image": {"reference": reference}})


def dockerfile_source(content):
    return document({"source": "foreignDockerfile", "dockerfile": {"content": content}})


def request_of(spec, *, lock=LOCK, base=FOREIGN):
    return BuildRequest(
        environment_uid="e1",
        version=1,
        build_uid="b1",
        owner_uid="o1",
        variant="datalayer",
        environment=parse_environment(spec),
        lock_text=lock,
        lock_digest="sha256:" + "d" * 64,
        resolved_base=base,
    )


def findings(spec):
    return [item.message for item in spec_findings(parse_environment(spec))]


class TestTheLayer:
    def rendered(self, spec=None):
        return Builder().dockerfile(request_of(spec or image_source("python:3.12-slim")))

    def test_it_copies_the_contract_from_the_approved_base_it_was_resolved_against(self):
        text = self.rendered()
        for path in CONTRACT_LAYER_FILES:
            assert f"COPY --from={BASE} {path} {path}" in text

    def test_it_installs_over_the_image_and_removes_nothing(self):
        text = self.rendered()
        assert "uv pip install --system --break-system-packages --require-hashes" in text
        assert "uv pip sync" not in text

    def test_pid_1_is_fetched_by_its_checksum(self):
        assert f"ADD --checksum=sha256:{TINI_SHA256} " in self.rendered()

    def test_it_ends_as_the_contracts_user_and_is_checked_by_the_doctor(self):
        text = self.rendered()
        assert text.index("usermod") < text.rindex("USER 1000:100")
        assert text.rstrip().endswith("doctor --json > /tmp/doctor.json")
        assert 'ENTRYPOINT ["/opt/datalayer/bin/tini", "-g", "--"]' in text

    def test_the_tasks_working_directory_is_kept(self):
        assert f"pwd > {TASK_WORKDIR_FILE}" in self.rendered()

    def test_a_dockerfile_keeps_what_it_says_and_gets_the_same_layer(self):
        spec = dockerfile_source("FROM python:3.12-slim AS task\nRUN echo hello > /hello\n")
        text = self.rendered(spec)
        assert f"FROM {FOREIGN} AS task" in text
        assert text.index("RUN echo hello") < text.index("COPY --from=" + BASE)

    def test_a_lock_with_no_contract_base_is_resolved_again_not_guessed(self):
        with pytest.raises(EnvironmentsError, match="resolve again"):
            Builder().dockerfile(request_of(image_source("python:3.12-slim"), lock="a==1\n"))

    def test_the_lock_carries_the_base(self):
        locked = lock_document(
            ResolveOutcome(lock_text="a==1\n"),
            python_version="3.12",
            base_reference=FOREIGN,
            merged=merge_requirements([]),
            contract_base=BASE,
        )
        assert contract_base_in(locked["content"]) == BASE


class TestTheReviewsFindings:
    """Six things a reader found, each pinned here."""

    @pytest.mark.parametrize(
        "content",
        [
            "FROM python:3.12\nCOPY --from=registry.example.com/team/tool:latest /tool /tool\n",
            "FROM python:3.12\nRUN --mount=type=bind,from=registry.example.com/t:1,target=/t a\n",
            "FROM python:3.12\nCOPY --from=later /a /a\nFROM python:3.12 AS later\n",
        ],
    )
    def test_nothing_is_pulled_past_the_base(self, content):
        said = " ".join(item.message for item in _foreign(content))
        assert "earlier stage" in said

    @pytest.mark.parametrize(
        "content",
        [
            "FROM a:1 AS tools\nRUN true\nFROM tools\nCOPY --from=tools /a /a\n",
            "FROM a:1 AS tools\nFROM tools\nCOPY --from=0 /a /a\n",
            "FROM a:1 AS tools\nFROM tools\nRUN --mount=type=bind,from=tools,target=/t true\n",
            "FROM a:1\nRUN --mount=type=cache,target=/c true\n",
        ],
    )
    def test_an_earlier_stage_is_fine(self, content):
        assert _foreign(content) == []

    @pytest.mark.parametrize(
        "content",
        [
            "FROM \\\n  python:3.12-slim\nRUN true\n",
            "FROM --platform=linux/amd64 \\\n  python:3.12-slim \\\n  AS task\nRUN true\n",
        ],
    )
    def test_a_base_on_a_continuation_line_is_pinned(self, content):
        text = Builder().dockerfile(request_of(dockerfile_source(content)))
        assert FOREIGN in text
        assert "python:3.12-slim" not in text

    def test_the_base_is_replaced_as_a_token(self):
        content = "FROM --platform=linux/amd64 a:1\nRUN echo a:1\n"
        pinned = pin_foreign_dockerfile_base(content, FOREIGN)
        assert f"FROM --platform=linux/amd64 {FOREIGN}" in pinned
        assert "RUN echo a:1" in pinned

    def test_the_reserved_tree_and_what_holds_it_stay_roots(self):
        text = Builder().dockerfile(request_of(image_source("python:3.12-slim")))
        assert "-path /opt/datalayer -prune" in text
        assert "! -path /opt" in text
        assert 'chown -R 1000:100 "$(pwd)"' not in text

    def test_the_lock_says_the_python_it_was_solved_for(self):
        class Solved:
            name = "fake"

            def solve(self, request, log=None):
                return ResolveOutcome(lock_text="ipykernel==7.3.0\n", python_version="3.12")

        spec = image_source("python:3.12-slim")
        spec["spec"]["language"]["version"] = "3.13"
        locked = resolve_environment(
            spec=spec, variants=["datalayer"], runner=Solved(), image_transport=_TRANSPORT
        )
        assert locked["python_version"] == "3.12"
        assert "# python: 3.12\n" in locked["content"]

    def test_the_solve_exports_the_python_it_saw(self):
        text = BuildkitResolveRunner(buildctl="/usr/bin/true").dockerfile(
            ResolveRequest(
                python_version="3.13",
                requirements=("ipykernel==7.3.0",),
                constraints=(),
                indexes=(),
                base_reference=FOREIGN,
                bootstrap_uv=True,
            )
        )
        assert "COPY --from=solve /solve/python-version /python-version" in text
        assert '--python-version "$(cat /solve/python-version)"' in text

    def test_pip_is_brought_to_the_python_the_image_kept(self):
        text = "\n".join(foreign_bootstrap_lines())
        assert "python3 -m ensurepip --default-pip" in text
        assert text.index("python3 -m ensurepip") < text.index("python3-pip")

    def test_an_unset_plane_is_no_plane(self):
        assert plane_of({}) is None
        assert plane_of({"PLANE_DOCKERFILES": ""}) is None
        assert plane_of({"PLANE_DOCKERFILES": "/no/such/plane"}) is None


class TestTheRegistries:
    def test_the_list_is_what_it_was(self):
        assert DEFAULT_ALLOWED_REGISTRIES == (
            "docker.io",
            "ghcr.io",
            "quay.io",
            "gcr.io",
            "registry.k8s.io",
            "public.ecr.aws",
        )

    def test_a_dockerfile_reaches_only_what_an_image_can(self):
        assert findings(dockerfile_source("FROM python:3.12-slim\n")) == []
        refused = findings(dockerfile_source("FROM registry.example.com/team/task:1\n"))
        assert "`registry.example.com` is not an allowed registry" in refused[0]

    def test_an_image_from_elsewhere_stays_refused(self):
        assert "is not an allowed registry" in findings(image_source("registry.example.com/a:1"))[0]

    @pytest.mark.parametrize(
        ("content", "said"),
        [
            ("FROM ${IMAGE}\n", "build argument"),
            ("FROM a:1 AS one\nFROM b:1 AS two\n", "second base"),
            ("FROM datalayer/python-cpu:2026.09\n", "approved base"),
            ("FROM python:3.12\nCOPY data.txt /data\n", "build context"),
            ("FROM python:3.12\nVOLUME /data\n", "VOLUME is refused"),
            ("RUN true\n", "names no base"),
        ],
    )
    def test_what_cannot_be_built_is_named(self, content, said):
        assert said in " ".join(item.message for item in _foreign(content))

    def test_a_stage_is_not_a_second_base(self):
        assert _foreign("FROM python:3.12 AS one\nFROM one AS two\n") == []

    def test_only_the_datalayer_builder_takes_it(self):
        from code_sandboxes.environments.adapters.modal import Builder as Modal

        assert "foreignDockerfile" not in Modal().capabilities().build_sources


def _foreign(content):
    return foreign_dockerfile_findings(content, registries=DEFAULT_ALLOWED_REGISTRIES)


def docker(*arguments, **options):
    done = subprocess.run(["docker", *arguments], capture_output=True, text=True, **options)
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def build(root, tag=None, output=None):
    command = ["buildx", "build", "--builder", "default", "--progress", "plain"]
    command += ["--output", f"type=local,dest={output}"] if output else ["--load", "-t", tag]
    docker(*command, str(root), timeout=900)


@pytest.fixture(scope="module")
def approved(tmp_path_factory):
    """What the layer copies from: the approved base's own paths, on a small image."""
    root = tmp_path_factory.mktemp("contract-source")
    shutil.copytree(WHEELHOUSE_PATH, root / "wheelhouse")
    shutil.copy(ENVIRONMENTS / "constraints" / "sandbox-contract-v1.txt", root)
    shutil.copy(ENVIRONMENTS / "doctor" / "build.py", root)
    shutil.copy(ENVIRONMENTS / "doctor" / "datalayer_sandbox.py", root)
    shutil.copy(PLANE / "jupyter-python" / "etc" / "jupyter" / "config" / "jupyter_config.py", root)
    shutil.copy(PLANE / "jupyter-python" / "etc" / "jupyter" / "scripts" / "start-jupyter.sh", root)
    (root / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "COPY . /c/\n"
        "RUN python3 /c/build.py /datalayer-sandbox\n"
        "RUN mkdir -p /opt/datalayer/bin /opt/datalayer/constraints /opt/datalayer/wheelhouse\n"
        "RUN cp /datalayer-sandbox /opt/datalayer/bin/ && cp /c/*.txt /opt/datalayer/constraints/ "
        "&& cp /c/wheelhouse/*.whl /opt/datalayer/wheelhouse/\n"
        "RUN mkdir -p /usr/local/sbin && cp /c/jupyter_config.py /usr/local/share/ "
        "&& cp /c/start-jupyter.sh /usr/local/sbin/\n"
        "RUN printf 'c.MappingKernelManager.root_dir = \"/home/datalayer\"\\n' "
        ">> /usr/local/share/jupyter_config.py\n"
    )
    build(root, tag="contract-source:test")
    return "contract-source:test"


def solve_and_build(approved, tmp_path, spec, base):
    """The real solve, then the real build, by the code that renders both."""
    merged = merge_requirements([])
    runner = BuildkitResolveRunner()
    solve = tmp_path / "solve"
    (solve / "wheelhouse").mkdir(parents=True)
    for wheel in WHEELHOUSE_PATH.glob("*.whl"):
        shutil.copy(wheel, solve / "wheelhouse")
    (solve / "requirements.in").write_text("\n".join(merged.requirements) + "\n")
    (solve / "constraints.txt").write_text("\n".join(merged.constraints) + "\n")
    request = ResolveRequest(
        python_version="3.12",
        requirements=tuple(merged.requirements),
        constraints=tuple(merged.constraints),
        indexes=(),
        base_reference=base,
        bootstrap_uv=True,
    )
    (solve / "Dockerfile").write_text(runner.dockerfile(request).replace("# syntax=", "# "))
    build(solve, output=tmp_path / "out")
    solved = (tmp_path / "out" / "python-version").read_text().strip()
    locked = lock_document(
        ResolveOutcome(lock_text=(tmp_path / "out" / "lock.txt").read_text()),
        python_version=solved,
        base_reference=base,
        merged=merged,
        contract_base=approved,
    )
    context = tmp_path / "context"
    context.mkdir()
    (context / "lock.txt").write_text(locked["content"])
    rendered = Builder().dockerfile(request_of(spec, lock=locked["content"], base=base))
    (context / "Dockerfile").write_text(rendered)
    tag = f"foreign-layer:{uuid.uuid4().hex[:8]}"
    build(context, tag=tag)
    return tag


KERNEL = """
import json, os, subprocess, sys, time, urllib.request
base = "http://127.0.0.1:2300/api/jupyter-server/api"
token = {"Authorization": "token t", "Content-Type": "application/json"}
def call(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    method = "POST" if body else "GET"
    request = urllib.request.Request(base + path, data=body, headers=token, method=method)
    return json.load(urllib.request.urlopen(request, timeout=30))
for _ in range(60):
    try:
        print(json.dumps(call("/status")))
        break
    except Exception as error:
        last = error
        time.sleep(1)
else:
    sys.exit(f"the server never answered: {last}")
print(json.dumps(call("/kernels", {"name": "python3"})))
"""


def served(tag):
    """The image's own entrypoint, the doctor beside it, a kernel's answer."""
    name = f"foreign-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--rm", "--name", name, "-e", "JUPYTER_TOKEN=t", tag)
    try:
        doctor = subprocess.run(
            ["docker", "exec", name, "/opt/datalayer/bin/datalayer-sandbox", "doctor", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        report = json.loads(doctor.stdout)
        status = docker("exec", name, "python3", "-c", KERNEL, timeout=120)
        code = (
            "import jupyter_client\n"
            "client = jupyter_client.BlockingKernelClient()\n"
            "from jupyter_client.manager import start_new_kernel\n"
            "manager, client = start_new_kernel(kernel_name='python3')\n"
            "seen = []\n"
            "client.execute_interactive('import os; print(os.getcwd(), 6 * 7)', "
            "output_hook=lambda m: seen.append(m['content'].get('text', '')), timeout=60)\n"
            "print(''.join(seen))\n"
            "manager.shutdown_kernel(now=True)\n"
        )
        answered = docker("exec", name, "python3", "-c", code, timeout=120)
        return report, status, answered, doctor.returncode
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.skipif(not (REAL and PLANE), reason="needs Docker, the network and the plane")
class TestARealImage:
    def _digest(self, image):
        docker("pull", "-q", image, timeout=300)
        return docker("image", "inspect", "--format", "{{index .RepoDigests 0}}", image).strip()

    def _settled(self, tag):
        report, status, answered, code = served(tag)
        assert code == 0 and report["ok"] is True, json.dumps(report)[:1500]
        assert len(report["rows"]) == 15 and all(row["ok"] for row in report["rows"])
        assert '"started"' in status
        assert '"execution_state"' in status
        assert "/home/datalayer 42" in answered
        return report

    def test_a_python_image_keeps_its_packages_and_is_a_sandbox(self, approved, tmp_path):
        base = self._digest("python:3.12-slim")
        spec = dockerfile_source(
            "FROM python:3.12-slim\nRUN pip install --no-cache-dir cowsay==6.1\n"
            "WORKDIR /app\nRUN echo task > /app/file\nUSER nobody\n"
        )
        tag = solve_and_build(approved, tmp_path, spec, base)
        self._settled(tag)
        assert (
            docker("run", "--rm", tag, "python3", "-c", "import cowsay; print('kept')").strip()
            == "kept"
        )
        assert docker("run", "--rm", tag, "cat", TASK_WORKDIR_FILE).strip() == "/app"
        assert docker(
            "run", "--rm", tag, "sh", "-c", "id -u; test -w /app/file && echo writable"
        ).split() == ["1000", "writable"]

    def test_a_working_directory_above_the_reserved_tree_leaves_it_alone(self, approved, tmp_path):
        base = self._digest("python:3.12-slim")
        spec = dockerfile_source("FROM python:3.12-slim\nWORKDIR /opt\nRUN echo t > /opt/file\n")
        tag = solve_and_build(approved, tmp_path, spec, base)
        self._settled(tag)
        owners = docker(
            "run", "--rm", tag, "stat", "-c", "%U", "/opt", "/opt/datalayer", "/opt/file"
        )
        assert owners.split() == ["root", "root", "datalayer"]

    def test_a_python_that_lost_its_pip_gets_it_back(self, approved, tmp_path):
        base = self._digest("python:3.12-slim")
        spec = dockerfile_source("FROM python:3.12-slim\nRUN python3 -m pip uninstall -y pip\n")
        self._settled(solve_and_build(approved, tmp_path, spec, base))

    def test_a_distribution_python_with_no_pip_gets_its_distributions(self, approved, tmp_path):
        base = self._digest("ubuntu:24.04")
        spec = dockerfile_source(
            "FROM ubuntu:24.04\nRUN apt-get update -qq && apt-get install -y python3\n"
        )
        self._settled(solve_and_build(approved, tmp_path, spec, base))

    def test_an_image_with_no_python_gets_its_distributions(self, approved, tmp_path):
        base = self._digest("ubuntu:24.04")
        tag = solve_and_build(approved, tmp_path, image_source("ubuntu:24.04"), base)
        self._settled(tag)

    def test_an_image_that_is_not_debian_derived_is_said_to_be(self, approved, tmp_path):
        base = self._digest("alpine:3.20")
        with pytest.raises(AssertionError, match="not Debian-derived"):
            solve_and_build(approved, tmp_path, image_source("alpine:3.20"), base)
