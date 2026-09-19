# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""``sandbox-contract/v1``: what an artifact must be to run as a Code Sandbox.

Every cross-provider promise of an Environment rests on this: an artifact is
usable as a Datalayer Code Sandbox on any variant if and only if it satisfies
the contract. The contract is data here — rows a person reads, values the
doctor checks, instructions a Dockerfile may not use — and the documentation
page is generated from it, so the page and the checks cannot say different
things.

Contract versions are additive. A launcher refuses an artifact whose contract
version is not in :data:`SUPPORTED_CONTRACTS`.

Run ``python -m code_sandboxes.environments contract --markdown PATH`` to
write the page, and ``contract --check PATH`` to fail when it has drifted.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .bases import APPROVED_BASES, ApprovedBase, is_approved_repository
from .errors import CAPABILITY_UNSUPPORTED, SPEC_INVALID, EnvironmentsError

__all__ = [
    "CONTRACT_V1",
    "MAX_CONTEXT_FILES",
    "MAX_CONTEXT_FILE_BYTES",
    "MAX_CONTEXT_TOTAL_BYTES",
    "SANDBOX_CONTRACT_V1",
    "SUPPORTED_CONTRACTS",
    "BuildContextEntry",
    "BuildContextFinding",
    "ContractRow",
    "DockerfileBase",
    "DockerfileFinding",
    "DockerfileInstruction",
    "SandboxContract",
    "check_build_context",
    "check_dockerfile",
    "contract_markdown",
    "dockerfile_base",
    "dockerfile_findings_for_build",
    "get_contract",
    "parse_dockerfile",
    "pin_dockerfile_base",
    "validate_build_context",
    "validate_dockerfile",
]

CONTRACT_V1 = "sandbox-contract/v1"

#: What a launcher accepts: the current version and the one before it, once
#: there is one (PLAN_ENV.md §18, question 12).
SUPPORTED_CONTRACTS: tuple[str, ...] = (CONTRACT_V1,)


class ContractRow(BaseModel):
    """One requirement of the contract, why it exists, and what checks it."""

    model_config = ConfigDict(frozen=True)

    area: str
    requirement: str
    reason: str
    #: ``doctor:<row>`` for a doctor row, ``conformance:<n>`` for an Appendix B check.
    checked_by: tuple[str, ...]


class SandboxContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    architecture: str
    os_family: str
    user: str
    uid: int
    gid: int
    home: str
    workdir: str
    reserved_path: str
    doctor_path: str
    python_executables: tuple[str, ...]
    kernel_packages: tuple[str, ...]
    graceful_shutdown_seconds: int
    locale: str
    timezone: str
    #: Dockerfile instructions refused, with why.
    forbidden_instructions: dict[str, str]
    rows: tuple[ContractRow, ...]


SANDBOX_CONTRACT_V1 = SandboxContract(
    version=CONTRACT_V1,
    architecture="linux/amd64",
    os_family="debian",
    user="datalayer",
    uid=1000,
    gid=100,
    home="/home/datalayer",
    workdir="/home/datalayer/content",
    reserved_path="/opt/datalayer",
    doctor_path="/opt/datalayer/bin/datalayer-sandbox",
    python_executables=("python3", "pip"),
    kernel_packages=("ipykernel", "jupyter_client"),
    graceful_shutdown_seconds=10,
    locale="C.UTF-8",
    timezone="UTC",
    forbidden_instructions={
        "VOLUME": "Modal's image builder does not implement `VOLUME`",
        "ONBUILD": "Modal's image builder does not implement `ONBUILD`",
        "STOPSIGNAL": "Modal's image builder does not implement `STOPSIGNAL`",
    },
    rows=(
        ContractRow(
            area="Architecture",
            requirement="`linux/amd64`.",
            reason="Modal pulls registry images for `linux/amd64` only; "
            "adopted everywhere for parity.",
            checked_by=("doctor:architecture",),
        ),
        ContractRow(
            area="Base OS",
            requirement="Debian-derived: Debian or Ubuntu.",
            reason="E2B builds templates only from Debian-derived images; "
            "adopted everywhere for parity.",
            checked_by=("doctor:os",),
        ),
        ContractRow(
            area="User",
            requirement=(
                "Non-root user `datalayer`, uid 1000, gid 100, home `/home/datalayer`, "
                "working directory `/home/datalayer/content`."
            ),
            reason=(
                "Runtime pods run as 1000:100 and home folders on the shared filesystem are owned "
                "that way. Modal ignores `USER`, so its adapter asserts the user in the run "
                "command."
            ),
            checked_by=(
                "doctor:user",
                "doctor:uid",
                "doctor:gid",
                "doctor:home",
                "doctor:workdir",
                "conformance:2",
            ),
        ),
        ContractRow(
            area="Python",
            requirement="`python3` and `pip` on `PATH`.",
            reason="The kernel, the doctor and the package installers all need them.",
            checked_by=("doctor:python", "doctor:pip", "conformance:3"),
        ),
        ContractRow(
            area="Kernel stack",
            requirement=(
                "`ipykernel`, `jupyter_client` and the Datalayer runtime agent, at versions inside "
                "the range Datalayer's protected constraints allow."
            ),
            reason="A pin outside that range builds, then gives a sandbox that never connects.",
            checked_by=("doctor:kernel", "conformance:4"),
        ),
        ContractRow(
            area="Entrypoint",
            requirement="A long-running process that `exec`s the arguments it is given.",
            reason=(
                "Modal requires it, Daytona defaults to `sleep infinity` and E2B uses a start "
                "command; the adapters make all three the same."
            ),
            checked_by=("conformance:8",),
        ),
        ContractRow(
            area="Signals",
            requirement="PID 1 reaps children and forwards `SIGTERM`; "
            "shutdown finishes within 10 seconds.",
            reason="A sandbox that ignores `SIGTERM` is killed with its work unsaved, "
            "and zombies pile up.",
            checked_by=("doctor:init", "conformance:8"),
        ),
        ContractRow(
            area="Filesystem",
            requirement=(
                "`/opt/datalayer` is reserved and read-only to the user; `/home/datalayer/content` "
                "is writable; nothing is assumed to persist across restarts."
            ),
            reason="Datalayer's tools live under `/opt/datalayer`; "
            "user code writes where its content is.",
            checked_by=("doctor:reserved_path", "doctor:workdir", "conformance:6"),
        ),
        ContractRow(
            area="Network",
            requirement=(
                "Egress is governed at runtime by policy, not baked into the image; no build-time "
                "credentials on disk."
            ),
            reason="A credential left in a layer is readable by anyone who can pull the image.",
            checked_by=("conformance:9", "conformance:10"),
        ),
        ContractRow(
            area="Locale and time",
            requirement="`C.UTF-8`, UTC, and current CA certificates.",
            reason="Tools behave the same on every provider, and HTTPS works.",
            checked_by=("doctor:locale", "doctor:timezone", "doctor:ca_certificates"),
        ),
        ContractRow(
            area="Health",
            requirement=(
                "`datalayer-sandbox doctor --json` exits 0 and reports the contract version, the "
                "Python version, the kernel versions, the user and the writable paths."
            ),
            reason="This command is the first check of every smoke test.",
            checked_by=("doctor", "conformance:1"),
        ),
    ),
)

_CONTRACTS = {SANDBOX_CONTRACT_V1.version: SANDBOX_CONTRACT_V1}


def get_contract(version: str) -> SandboxContract:
    """The contract of that version, or a refusal naming the supported ones."""
    try:
        return _CONTRACTS[version]
    except KeyError:
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"{version!r} is not a supported sandbox contract; supported: "
            + ", ".join(SUPPORTED_CONTRACTS),
            detail={"contract": version, "supported": list(SUPPORTED_CONTRACTS)},
        ) from None


# --- Dockerfiles -----------------------------------------------------------------


@dataclass(frozen=True)
class DockerfileInstruction:
    """One instruction, its continuation lines joined, and the line it starts on."""

    line: int
    keyword: str
    arguments: str


@dataclass(frozen=True)
class DockerfileFinding:
    """Something in a Dockerfile at least one provider cannot honor."""

    line: int
    instruction: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"line": self.line, "instruction": self.instruction, "message": self.message}


_DIRECTIVE = re.compile(r"^#\s*([a-zA-Z]+)\s*=\s*(\S+)\s*$")
_HEREDOC = re.compile(r"<<-?([\"']?)([A-Za-z_][A-Za-z0-9_]*)\1")


def parse_dockerfile(text: str) -> list[DockerfileInstruction]:
    """The instructions of a Dockerfile, the way BuildKit's parser reads them.

    Parser directives at the top (``# escape=`` changes the continuation
    character), comments, line continuations, and heredoc bodies — which are
    content, not instructions — are handled; the rest is left to BuildKit.
    """
    escape = "\\"
    instructions: list[DockerfileInstruction] = []
    directives_allowed = True
    pending: list[str] = []
    start = 0
    heredocs: list[tuple[str, bool]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if heredocs:
            terminator, strip_tabs = heredocs[0]
            candidate = raw.lstrip("\t") if strip_tabs else raw
            if candidate.rstrip() == terminator:
                heredocs.pop(0)
            continue
        stripped = raw.strip()
        if directives_allowed and not pending:
            directive = _DIRECTIVE.match(stripped)
            if directive:
                if directive.group(1).lower() == "escape":
                    escape = directive.group(2)
                continue
            directives_allowed = False
        if not stripped or stripped.startswith("#"):
            # A comment or a blank line, inside a continuation too: dropped.
            continue
        body = raw.rstrip()
        if not pending:
            start = number
        if body.endswith(escape):
            pending.append(body[: -len(escape)].strip())
            continue
        pending.append(body.strip())
        joined = " ".join(part for part in pending if part)
        pending = []
        keyword, _, arguments = joined.partition(" ")
        instruction = DockerfileInstruction(start, keyword.upper(), arguments.strip())
        instructions.append(instruction)
        if instruction.keyword in {"RUN", "COPY", "ADD"}:
            for match in _HEREDOC.finditer(instruction.arguments):
                heredocs.append((match.group(2), match.group(0).startswith("<<-")))
    if pending:
        joined = " ".join(part for part in pending if part)
        keyword, _, arguments = joined.partition(" ")
        instructions.append(DockerfileInstruction(start, keyword.upper(), arguments.strip()))
    return instructions


def _tokens(arguments: str) -> list[str]:
    try:
        return shlex.split(arguments, posix=True)
    except ValueError:
        return arguments.split()


def _check_from(
    instruction: DockerfileInstruction, stages: set[str], contract: SandboxContract
) -> list[DockerfileFinding]:
    findings: list[DockerfileFinding] = []
    tokens = _tokens(instruction.arguments)
    image = ""
    rest: list[str] = []
    for index, argument in enumerate(tokens):
        if argument.startswith("--platform="):
            platform = argument.split("=", 1)[1]
            if platform != contract.architecture and not platform.startswith("$"):
                findings.append(
                    DockerfileFinding(
                        instruction.line,
                        "FROM",
                        f"builds for {platform}; every provider runs {contract.architecture}",
                    )
                )
            continue
        if argument.startswith("--"):
            continue
        image, rest = argument, tokens[index + 1 :]
        break
    if not image:
        return findings
    if len(rest) >= 2 and rest[0].upper() == "AS":
        alias = rest[1].lower()
    else:
        alias = None
    if image.lower() in stages:
        pass
    elif "$" in image:
        findings.append(
            DockerfileFinding(
                instruction.line,
                "FROM",
                f"`{image}` is chosen through a build argument, so it cannot be checked against "
                "the approved bases; name the base itself",
            )
        )
    elif not is_approved_repository(image):
        findings.append(
            DockerfileFinding(
                instruction.line,
                "FROM",
                f"`{image}` is not an approved Datalayer base",
            )
        )
    if alias:
        stages.add(alias)
    return findings


def _mount_options(argument: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for part in argument.split("=", 1)[1].split(","):
        key, _, value = part.partition("=")
        options[key.strip().lower()] = value.strip()
    return options


def validate_dockerfile(
    text: str, *, contract: SandboxContract = SANDBOX_CONTRACT_V1
) -> list[DockerfileFinding]:
    """Everything in a Dockerfile the contract refuses, in line order.

    Refused: the instructions a provider's builder does not implement, a
    base that is not approved (or cannot be checked), a platform other than
    the contract's, privileged builds, the host network, Docker socket mounts
    and host bind mounts.
    """
    findings: list[DockerfileFinding] = []
    stages: set[str] = set()
    for instruction in parse_dockerfile(text):
        keyword = instruction.keyword
        reason = contract.forbidden_instructions.get(keyword)
        if reason:
            findings.append(
                DockerfileFinding(instruction.line, keyword, f"{keyword} is refused: {reason}")
            )
        if keyword == "FROM":
            findings.extend(_check_from(instruction, stages, contract))
        if "docker.sock" in instruction.arguments:
            findings.append(
                DockerfileFinding(instruction.line, keyword, "mounts the Docker socket")
            )
        if "--privileged" in instruction.arguments:
            findings.append(
                DockerfileFinding(instruction.line, keyword, "asks for privileged mode")
            )
        if keyword == "RUN":
            for argument in _tokens(instruction.arguments):
                if not argument.startswith("--"):
                    break
                if argument == "--security=insecure":
                    findings.append(
                        DockerfileFinding(
                            instruction.line, keyword, "runs privileged (`--security=insecure`)"
                        )
                    )
                elif argument == "--network=host":
                    findings.append(
                        DockerfileFinding(instruction.line, keyword, "uses the host network")
                    )
                elif argument.startswith("--mount="):
                    options = _mount_options(argument)
                    source = options.get("source") or options.get("src") or ""
                    if (
                        options.get("type", "bind") == "bind"
                        and "from" not in options
                        and source.startswith("/")
                        and "docker.sock" not in source
                    ):
                        findings.append(
                            DockerfileFinding(
                                instruction.line,
                                keyword,
                                f"bind-mounts the host path `{source}`",
                            )
                        )
    return sorted(findings, key=lambda finding: finding.line)


def check_dockerfile(text: str, *, contract: SandboxContract = SANDBOX_CONTRACT_V1) -> None:
    """Refuse a Dockerfile the contract does not allow, naming the first line at fault."""
    findings = validate_dockerfile(text, contract=contract)
    if findings:
        first = findings[0]
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"line {first.line}: {first.message}",
            detail={"findings": [finding.to_dict() for finding in findings]},
        )


# --- A Dockerfile source's base, and how it is built (E3-03) -----------------------------


@dataclass(frozen=True)
class DockerfileBase:
    """The approved base a Dockerfile builds on: its channel, and the `FROM` lines naming it."""

    ref: str
    channel: str
    lines: tuple[int, ...]


@dataclass(frozen=True)
class _ApprovedFrom:
    line: int
    image: str
    ref: str
    tag: str
    digest: str


def _from_image(instruction: DockerfileInstruction) -> tuple[str, str | None]:
    """A `FROM`'s image and its stage name, flags skipped."""
    tokens = _tokens(instruction.arguments)
    for index, argument in enumerate(tokens):
        if argument.startswith("--"):
            continue
        rest = tokens[index + 1 :]
        alias = rest[1].lower() if len(rest) >= 2 and rest[0].upper() == "AS" else None
        return argument, alias
    return "", None


def _approved_froms(text: str, bases: Mapping[str, ApprovedBase]) -> list[_ApprovedFrom]:
    """Every `FROM` naming an approved base, in order; a stage name is not a base."""
    stages: set[str] = set()
    found: list[_ApprovedFrom] = []
    for instruction in parse_dockerfile(text):
        if instruction.keyword != "FROM":
            continue
        image, alias = _from_image(instruction)
        if image and image.lower() not in stages and "$" not in image:
            repository, _, digest = image.partition("@")
            last = repository.rsplit("/", 1)[-1]
            name, _, tag = last.partition(":")
            repository = repository[: len(repository) - len(last)] + name
            for base in bases.values():
                if any(
                    repository == known or repository.endswith("/" + known)
                    for known in (base.ref, base.repository)
                ):
                    found.append(_ApprovedFrom(instruction.line, image, base.ref, tag, digest))
                    break
        if alias:
            stages.add(alias)
    return found


def dockerfile_findings_for_build(
    text: str, bases: Mapping[str, ApprovedBase] = APPROVED_BASES
) -> list[DockerfileFinding]:
    """What a Dockerfile source must also be to be built, beyond what the contract allows.

    - **One base, named by its channel.** Every `FROM` of an approved base
      names the same base, and its channel as the tag
      (`datalayer/python-cpu:2026.09`): the build pins that channel to its
      digest, the way a `packages` source's base is pinned (D-9), and the
      lock is solved in it. A digest is refused rather than trusted — which
      channel it belongs to is what the build must know.
    - **No build context yet.** A `COPY` or `ADD` from the context needs the
      upload this item also describes, which is not taken yet; without it
      the file is absent and the build fails halfway. From another stage
      (`--from=`), from a heredoc, and `ADD` of a URL need none.
    - **The default escape character.** The build appends its own lines,
      continued with a backslash.
    """
    findings: list[DockerfileFinding] = []
    for raw in text.splitlines():
        directive = _DIRECTIVE.match(raw.strip())
        if not directive:
            break
        if directive.group(1).lower() == "escape" and directive.group(2) != "\\":
            findings.append(
                DockerfileFinding(
                    1, "escape", "keep the default escape character: the build appends lines to it"
                )
            )
    named: list[_ApprovedFrom] = []
    for found in _approved_froms(text, bases):
        channels = bases[found.ref].channels
        if found.digest:
            message = f"name `{found.ref}` by its channel, not a digest: the build pins the channel"
        elif not found.tag:
            example = next(iter(channels), "2026.09")
            message = f"name `{found.ref}`'s channel as its tag, such as `{found.ref}:{example}`"
        elif found.tag not in channels:
            message = f"`{found.ref}` has no channel `{found.tag}`; channels: " + (
                ", ".join(channels) or "none"
            )
        else:
            named.append(found)
            continue
        findings.append(DockerfileFinding(found.line, "FROM", message))
    if len({(found.ref, found.tag) for found in named}) > 1:
        findings.append(
            DockerfileFinding(
                named[1].line,
                "FROM",
                "every approved base is the same base and channel: the build pins one",
            )
        )
    for instruction in parse_dockerfile(text):
        if instruction.keyword not in ("COPY", "ADD"):
            continue
        tokens = _tokens(instruction.arguments)
        if "<<" in instruction.arguments or any(t.startswith("--from=") for t in tokens):
            continue
        sources = [t for t in tokens if not t.startswith("--")][:-1]
        local = [s for s in sources if not re.match(r"^(https?|git)://|^git@", s, re.IGNORECASE)]
        if local:
            findings.append(
                DockerfileFinding(
                    instruction.line,
                    instruction.keyword,
                    f"copies `{local[0]}` from the build context, which this deployment does not "
                    "take yet: bake it with a `RUN` or a heredoc, or copy it from another stage",
                )
            )
    return sorted(findings, key=lambda finding: finding.line)


def dockerfile_base(
    text: str, bases: Mapping[str, ApprovedBase] = APPROVED_BASES
) -> DockerfileBase:
    """The approved base a Dockerfile source is built on, or a refusal naming the line."""
    refused = [
        finding
        for finding in dockerfile_findings_for_build(text, bases)
        if finding.instruction in ("FROM", "escape")
    ]
    if refused:
        raise EnvironmentsError(
            SPEC_INVALID,
            f"line {refused[0].line}: {refused[0].message}",
            detail={"field": "spec.build.dockerfile", "line": refused[0].line},
        )
    found = _approved_froms(text, bases)
    if not found:
        raise EnvironmentsError(
            SPEC_INVALID,
            "the Dockerfile builds on no approved base",
            detail={"field": "spec.build.dockerfile"},
        )
    return DockerfileBase(
        ref=found[0].ref,
        channel=found[0].tag,
        lines=tuple(item.line for item in found),
    )


def pin_dockerfile_base(
    text: str, reference: str, bases: Mapping[str, ApprovedBase] = APPROVED_BASES
) -> str:
    """The Dockerfile as it is built: each `FROM` of its base pinned to `reference`.

    Only the image of those lines changes; comments, the author's own
    continuations and the stage names stay as written. The parser directives
    go, because the build states its own frontend first (`# syntax=`), and a
    directive anywhere but the very top is only a comment.
    """
    images = {found.line: found.image for found in _approved_froms(text, bases)}
    rewritten: list[str] = []
    in_directives = True
    for number, raw in enumerate(text.splitlines(), start=1):
        if in_directives and _DIRECTIVE.match(raw.strip()):
            continue
        in_directives = False
        image = images.get(number)
        rewritten.append(raw.replace(image, reference, 1) if image else raw)
    return "\n".join(rewritten) + "\n"


# --- The build context (E3-03) -------------------------------------------------------

#: A `dockerfile` source uploads a build context to object storage. These bound
#: what Runtimes accepts before it issues a presigned URL, so a context cannot
#: be a way to smuggle a host file in (a symlink or `..`), or to fill a bucket.
MAX_CONTEXT_FILES = 2000
MAX_CONTEXT_FILE_BYTES = 50 * 1024 * 1024
MAX_CONTEXT_TOTAL_BYTES = 100 * 1024 * 1024

_CONTEXT_SEPARATOR = re.compile(r"[\\/]")


@dataclass(frozen=True)
class BuildContextEntry:
    """One member of an uploaded build context: its path, size, and whether it is a symlink."""

    path: str
    size_bytes: int = 0
    is_symlink: bool = False


@dataclass(frozen=True)
class BuildContextFinding:
    """Something in a build context that must not be uploaded."""

    path: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "message": self.message}


def validate_build_context(
    entries: Sequence[BuildContextEntry],
) -> list[BuildContextFinding]:
    """Everything in a build context the upload refuses, in the order given.

    Refused: an absolute path, a `..` that would escape the context, a symlink
    (which could point at a host file the build then reads), a file over the
    per-file limit, and — once — a context with too many files or too many
    bytes in all.
    """
    findings: list[BuildContextFinding] = []
    total = 0
    for entry in entries:
        path = entry.path
        components = _CONTEXT_SEPARATOR.split(path)
        if not path or all(part in ("", ".") for part in components):
            findings.append(BuildContextFinding(path, "is not a path inside the context"))
        elif path.startswith("/") or path.startswith("\\"):
            findings.append(BuildContextFinding(path, "is an absolute path, not a context path"))
        elif ".." in components:
            findings.append(BuildContextFinding(path, "escapes the context with `..`"))
        if entry.is_symlink:
            findings.append(BuildContextFinding(path, "is a symlink, which could read a host file"))
        if entry.size_bytes > MAX_CONTEXT_FILE_BYTES:
            findings.append(
                BuildContextFinding(
                    path, f"is over the {MAX_CONTEXT_FILE_BYTES}-byte per-file limit"
                )
            )
        total += entry.size_bytes
    if len(entries) > MAX_CONTEXT_FILES:
        findings.append(BuildContextFinding("", f"has more than {MAX_CONTEXT_FILES} files"))
    if total > MAX_CONTEXT_TOTAL_BYTES:
        findings.append(
            BuildContextFinding("", f"is over the {MAX_CONTEXT_TOTAL_BYTES}-byte total limit")
        )
    return findings


def check_build_context(entries: Sequence[BuildContextEntry]) -> None:
    """Refuse a build context the upload does not allow, naming the first fault."""
    findings = validate_build_context(entries)
    if findings:
        first = findings[0]
        where = f"`{first.path}`: " if first.path else ""
        raise EnvironmentsError(
            SPEC_INVALID,
            f"{where}{first.message}",
            detail={"findings": [finding.to_dict() for finding in findings]},
        )


# --- The documentation page ----------------------------------------------------------


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def contract_markdown(contract: SandboxContract = SANDBOX_CONTRACT_V1) -> str:
    """The contract as the documentation page shows it."""
    lines = [
        "---",
        "title: Sandbox contract",
        "description: What an artifact must be to run as a Datalayer Code Sandbox, "
        f"{contract.version}.",
        "---",
        "",
        f"# `{contract.version}`",
        "",
        "An artifact — an OCI image, an E2B template build, a Daytona snapshot, a Modal image — "
        "is usable as a Code Sandbox on any variant if and only if it satisfies this contract. "
        "Every Environment build ends by checking it, and a launcher refuses an artifact whose "
        "contract version it does not support.",
        "",
        "This page is generated from `code_sandboxes/environments/contract.py`: change the "
        "contract there, then run `python -m code_sandboxes.environments contract --markdown "
        "docs/docs/environments/contract.mdx`.",
        "",
        "## Requirements",
        "",
        "| Area | Requirement | Why | Checked by |",
        "|---|---|---|---|",
    ]
    for row in contract.rows:
        checks = ", ".join(f"`{check}`" for check in row.checked_by)
        lines.append(
            f"| {_cell(row.area)} | {_cell(row.requirement)} | {_cell(row.reason)} | {checks} |"
        )
    lines += [
        "",
        "`doctor:<row>` is a row of `datalayer-sandbox doctor --json`; "
        "`conformance:<n>` is a check "
        "of the conformance suite, which runs the doctor first.",
        "",
        "## Identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| User | `{contract.user}` |",
        f"| uid | `{contract.uid}` |",
        f"| gid | `{contract.gid}` |",
        f"| Home | `{contract.home}` |",
        f"| Working directory | `{contract.workdir}` |",
        f"| Reserved, read-only | `{contract.reserved_path}` |",
        f"| Doctor | `{contract.doctor_path}` |",
        f"| Locale | `{contract.locale}` |",
        f"| Time zone | `{contract.timezone}` |",
        f"| Graceful shutdown | {contract.graceful_shutdown_seconds} seconds |",
        "",
        "## Refused in a Dockerfile",
        "",
        "At least one provider cannot honor these, so a Dockerfile that uses them is refused "
        "before any build starts, with the line named.",
        "",
    ]
    for keyword, reason in contract.forbidden_instructions.items():
        lines.append(f"- `{keyword}`: {reason}.")
    lines += [
        "- A `FROM` that is not an approved Datalayer base, or that names its base "
        "through a build argument.",
        f"- A `--platform` other than `{contract.architecture}`.",
        "- Privileged builds: `--privileged` and `RUN --security=insecure`.",
        "- `RUN --network=host`.",
        "- Mounting the Docker socket.",
        "- Bind-mounting a host path.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m code_sandboxes.environments contract")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--markdown", metavar="PATH", nargs="?", const="-", help="write the page")
    group.add_argument("--check", metavar="PATH", help="fail when the page has drifted")
    arguments = parser.parse_args(argv)
    page = contract_markdown()
    if arguments.check:
        current = Path(arguments.check).read_text(encoding="utf-8")
        if current != page:
            print(  # noqa: T201 - a command's own output
                f"{arguments.check} is out of date; regenerate it with --markdown", file=sys.stderr
            )
            return 1
        return 0
    if arguments.markdown == "-":
        sys.stdout.write(page)
    else:
        Path(arguments.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(arguments.markdown).write_text(page, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
