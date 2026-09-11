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
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .bases import is_approved_repository
from .errors import CAPABILITY_UNSUPPORTED, EnvironmentsError

__all__ = [
    "CONTRACT_V1",
    "SANDBOX_CONTRACT_V1",
    "SUPPORTED_CONTRACTS",
    "ContractRow",
    "DockerfileFinding",
    "DockerfileInstruction",
    "SandboxContract",
    "check_dockerfile",
    "contract_markdown",
    "get_contract",
    "parse_dockerfile",
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
