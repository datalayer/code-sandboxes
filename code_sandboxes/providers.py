# Copyright (c) 2023-2025 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""What each provider needs before it can run anything.

A sandbox variant is only usable when its credentials are on hand: Kaggle wants
an API token, Modal a token pair, Datalayer an account. Every caller that
offers sandboxes — the CLI, the web application, the JupyterLab extension —
has to answer the same question first, "which of these can this machine
actually use", and each of them was answering it on its own.

This module is that answer, once: a provider declares what it requires, how to
tell whether the requirement is met, and which environments it ships. Nothing
here starts a sandbox or reads a secret's value — it reports what is present,
so a caller can offer what will work and say why the rest is missing.

The environments themselves stay with the variants that own them
(``Sandbox.list_environments``): a provider ships environments the way
Datalayer ships ``ai-agents-env``, and only the variant knows its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .models import SandboxEnvironment, SandboxVariant

__all__ = [
    "ProviderRequirement",
    "SandboxProvider",
    "PROVIDERS",
    "available_providers",
    "get_provider",
]


@dataclass(frozen=True)
class ProviderRequirement:
    """One way of satisfying a provider's credentials.

    A provider may accept several: Kaggle takes a single API token, a
    username/key pair, or the file its own CLI writes. Any one of them is
    enough, which is why they are listed rather than merged.
    """

    #: Environment variables that must all be set for this way to be satisfied.
    env_vars: tuple[str, ...] = ()
    #: A file that satisfies it instead, e.g. what a provider's CLI writes.
    file: Optional[str] = None
    #: What to tell someone who has none of it.
    hint: str = ""

    def is_met(self) -> bool:
        """Whether this way of providing the credentials is satisfied here."""
        if self.env_vars and all(os.environ.get(name) for name in self.env_vars):
            return True
        if self.file and Path(self.file).expanduser().is_file():
            return True
        return False


@dataclass(frozen=True)
class SandboxProvider:
    """A place sandboxes can run, and what it takes to run there."""

    variant: SandboxVariant
    title: str
    description: str
    #: Any one of these satisfies the provider; empty means nothing is needed.
    requirements: tuple[ProviderRequirement, ...] = ()
    #: Extra packages needed, as the extra of this distribution.
    extra: Optional[str] = None
    #: Whether the provider can be used with no credentials at all.
    needs_credentials: bool = True
    #: Read the environments this provider ships, when it can be asked.
    list_environments: Optional[Callable[[], list[SandboxEnvironment]]] = field(
        default=None, repr=False
    )

    @property
    def name(self) -> str:
        """The identifier of the provider, which is that of its variant."""
        return self.variant.value

    def is_available(self) -> bool:
        """Whether this machine has what the provider requires."""
        if not self.needs_credentials or not self.requirements:
            return True
        return any(requirement.is_met() for requirement in self.requirements)

    def missing(self) -> tuple[ProviderRequirement, ...]:
        """The ways of satisfying it, when none of them is satisfied."""
        return () if self.is_available() else self.requirements

    def environments(self) -> list[SandboxEnvironment]:
        """The environments the provider ships, or none when it cannot say."""
        if self.list_environments is None:
            return []
        try:
            return self.list_environments()
        except Exception:  # noqa: BLE001
            # A provider that cannot be reached ships nothing rather than
            # taking down whoever asked what is available.
            return []


def _environments_of(variant: SandboxVariant) -> Callable[[], list[SandboxEnvironment]]:
    """Read a variant's own environments, lazily.

    Imported on the call rather than here: a provider whose package is not
    installed must cost nothing until someone asks about it.
    """

    def read() -> list[SandboxEnvironment]:
        from .base import Sandbox

        return Sandbox.list_environments(variant=variant)

    return read


#: Every provider, whether or not this machine can use it.
PROVIDERS: tuple[SandboxProvider, ...] = (
    SandboxProvider(
        variant=SandboxVariant.DATALAYER,
        title="Datalayer",
        description=(
            "Sandboxes of the Datalayer platform, in the environments the "
            "account may launch — `ai-agents-env` and the rest."
        ),
        requirements=(
            ProviderRequirement(
                env_vars=("DATALAYER_TOKEN",),
                hint="Sign in with `datalayer login`, or set DATALAYER_TOKEN.",
            ),
        ),
        list_environments=_environments_of(SandboxVariant.DATALAYER),
    ),
    SandboxProvider(
        variant=SandboxVariant.JUPYTER,
        title="Jupyter Server",
        description=(
            "Kernels of a Jupyter Server — the one this application is running "
            "against, or any other that is reachable."
        ),
        requirements=(
            ProviderRequirement(
                env_vars=("JUPYTER_SERVER_URL",),
                hint="Set JUPYTER_SERVER_URL (and JUPYTER_TOKEN if it is secured).",
            ),
        ),
        list_environments=_environments_of(SandboxVariant.JUPYTER),
    ),
    SandboxProvider(
        variant=SandboxVariant.KAGGLE,
        title="Kaggle",
        description=(
            "Kaggle notebook sessions, interactively against a running kernel "
            "or as a batch job."
        ),
        extra="kaggle",
        requirements=(
            ProviderRequirement(
                env_vars=("KAGGLE_API_TOKEN",),
                hint="Set KAGGLE_API_TOKEN to the contents of your kaggle.json.",
            ),
            ProviderRequirement(
                env_vars=("KAGGLE_USERNAME", "KAGGLE_KEY"),
                hint="Set KAGGLE_USERNAME and KAGGLE_KEY.",
            ),
            ProviderRequirement(
                file="~/.kaggle/kaggle.json",
                hint="Place kaggle.json in ~/.kaggle/, as the Kaggle CLI does.",
            ),
        ),
        list_environments=_environments_of(SandboxVariant.KAGGLE),
    ),
    SandboxProvider(
        variant=SandboxVariant.MODAL,
        title="Modal",
        description="Containers on Modal, with or without a GPU attached.",
        extra="modal",
        requirements=(
            ProviderRequirement(
                env_vars=("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
                hint="Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
            ),
            ProviderRequirement(
                file="~/.modal.toml",
                hint="Run `modal token new`, which writes ~/.modal.toml.",
            ),
        ),
        list_environments=_environments_of(SandboxVariant.MODAL),
    ),
    SandboxProvider(
        variant=SandboxVariant.DOCKER,
        title="Docker",
        description="Containers on the Docker daemon of this machine.",
        needs_credentials=False,
        list_environments=_environments_of(SandboxVariant.DOCKER),
    ),
    SandboxProvider(
        variant=SandboxVariant.EVAL,
        title="Eval",
        description=(
            "Code evaluated in this very process. For tests and examples; it "
            "isolates nothing."
        ),
        needs_credentials=False,
        list_environments=_environments_of(SandboxVariant.EVAL),
    ),
)


def get_provider(name: str) -> Optional[SandboxProvider]:
    """The provider of that name, if there is one.

    Args:
        name: Identifier of the provider, which is that of its variant.
    """
    wanted = (name or "").replace("-", "_").lower()
    for provider in PROVIDERS:
        if provider.name == wanted:
            return provider
    return None


def available_providers() -> tuple[SandboxProvider, ...]:
    """The providers this machine has the credentials for."""
    return tuple(provider for provider in PROVIDERS if provider.is_available())
