# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Datalayer's protected constraints: the kernel stack, each package at one exact version."""

from __future__ import annotations

from importlib import resources

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from code_sandboxes.environments.contract import SANDBOX_CONTRACT_V1

#: PLAN_ENV.md, E1-04: the packages Datalayer reserves.
PROTECTED = {
    "ipykernel",
    "jupyter-client",
    "jupyter-server",
    "jupyter-server-nbmodel",
    "datalayer",
}


def _constraints() -> list[Requirement]:
    path = (
        resources.files("code_sandboxes.environments") / "constraints" / "sandbox-contract-v1.txt"
    )
    lines = [
        line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return [Requirement(line) for line in lines if line]


def test_the_constraints_protect_the_kernel_stack_and_nothing_else() -> None:
    names = [canonicalize_name(requirement.name) for requirement in _constraints()]
    assert len(names) == len(set(names))
    assert set(names) == PROTECTED
    assert {canonicalize_name(name) for name in SANDBOX_CONTRACT_V1.kernel_packages} <= set(names)


def test_each_constraint_is_one_exact_version() -> None:
    for requirement in _constraints():
        specifiers = list(requirement.specifier)
        assert len(specifiers) == 1, requirement
        assert specifiers[0].operator == "==", requirement
        assert "*" not in specifiers[0].version, requirement
        assert requirement.url is None, requirement
        assert requirement.marker is None, requirement
        assert not requirement.extras, requirement
