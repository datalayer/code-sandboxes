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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .models import SandboxEnvironment, SandboxVariant, normalize_variant

__all__ = [
    "PROVIDERS",
    "ProviderRequirement",
    "SandboxProvider",
    "available_providers",
    "get_provider",
    "provider_catalog",
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
    file: str | None = None
    #: What to tell someone who has none of it.
    hint: str = ""

    def is_met(self, secrets: Mapping[str, str] | None = None) -> bool:
        """Whether this way of providing the credentials is satisfied.

        Args:
            secrets: Where to look the variables up. The process environment
                by default; a service passes the secrets of an ACCOUNT here,
                so "enabled for this user" and "enabled on this machine" are
                the same question asked of different stores.
        """
        store: Mapping[str, str] = os.environ if secrets is None else secrets
        if self.env_vars and all(store.get(name) for name in self.env_vars):
            return True
        # A file only answers for the local machine: a remote store of
        # secrets has no files to offer.
        if secrets is None and self.file and Path(self.file).expanduser().is_file():
            return True
        return False


@dataclass(frozen=True)
class SandboxProvider:
    """A place sandboxes can run, and what it takes to run there."""

    variant: SandboxVariant
    title: str
    description: str
    #: The mark this provider is drawn with, as a slug of the Datalayer icon
    #: set — `daytona` for `DaytonaIcon`. Named here rather than by whoever
    #: draws it, so the CLI, the operator and the web all show one provider as
    #: one thing. None where the set has no mark for it yet; a reader then
    #: falls back to whatever it uses for the unknown.
    icon: str | None = None
    #: Any one of these satisfies the provider; empty means nothing is needed.
    requirements: tuple[ProviderRequirement, ...] = ()
    #: Extra packages needed, as the extra of this distribution.
    extra: str | None = None
    #: Whether the provider can be used with no credentials at all.
    needs_credentials: bool = True
    #: Credentials the environment listing takes as ARGUMENTS, as pairs of
    #: (keyword, the variable holding its value).
    #:
    #: Most providers ship a fixed list and need none. Datalayer asks its
    #: platform, so it needs the account to ask on behalf of — and reading it
    #: from the process environment is exactly wrong for a service, which
    #: holds the credentials of whoever is asking and not of itself.
    environment_secrets: tuple[tuple[str, str], ...] = ()
    #: Read the environments this provider ships, when it can be asked.
    list_environments: Callable[..., list[SandboxEnvironment]] | None = field(
        default=None, repr=False
    )

    @property
    def name(self) -> str:
        """The identifier of the provider, which is that of its variant."""
        return self.variant.value

    def is_available(self, secrets: Mapping[str, str] | None = None) -> bool:
        """Whether the provider's requirements are met, here or in `secrets`."""
        if not self.needs_credentials or not self.requirements:
            return True
        return any(requirement.is_met(secrets) for requirement in self.requirements)

    def missing(self) -> tuple[ProviderRequirement, ...]:
        """The ways of satisfying it, when none of them is satisfied."""
        return () if self.is_available() else self.requirements

    def environments(self, secrets: Mapping[str, str] | None = None) -> list[SandboxEnvironment]:
        """The environments the provider ships, or none when it cannot say.

        Args:
            secrets: Where to read the credentials the listing takes as
                arguments. The process environment by default; a service
                passes the secrets of an ACCOUNT, and then asking the platform
                what THAT account may launch is the whole point — asked
                without them, the provider reads as enabled and ships nothing.
        """
        if self.list_environments is None:
            return []
        store: Mapping[str, str] = os.environ if secrets is None else secrets
        arguments = {
            keyword: store[variable]
            for keyword, variable in self.environment_secrets
            if store.get(variable)
        }
        try:
            return self.list_environments(**arguments)
        except Exception:
            # A provider that cannot be reached ships nothing rather than
            # taking down whoever asked what is available.
            return []


def _environments_of(variant: SandboxVariant) -> Callable[..., list[SandboxEnvironment]]:
    """Read a variant's own environments, lazily.

    Imported on the call rather than here: a provider whose package is not
    installed must cost nothing until someone asks about it.
    """

    def read(**kwargs) -> list[SandboxEnvironment]:
        from .base import Sandbox

        return Sandbox.list_environments(variant=variant, **kwargs)

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
        environment_secrets=(
            ("token", "DATALAYER_TOKEN"),
            ("run_url", "DATALAYER_RUN_URL"),
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
        icon="kaggle",
        title="Kaggle",
        description=(
            "Kaggle notebook sessions, interactively against a running kernel or as a batch job."
        ),
        extra="kaggle",
        requirements=(
            ProviderRequirement(
                env_vars=("KAGGLE_API_TOKEN",),
                hint=(
                    "Set KAGGLE_API_TOKEN to an access token from "
                    "kaggle.com/settings (KGAT_…), or to the contents of "
                    "your kaggle.json."
                ),
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
        icon="modal",
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
        variant=SandboxVariant.DAYTONA,
        icon="daytona",
        title="Daytona",
        description=(
            "Sandboxes on Daytona, with a stateful Python interpreter and an optional GPU."
        ),
        extra="daytona",
        requirements=(
            ProviderRequirement(
                env_vars=("DAYTONA_API_KEY",),
                hint=("Create an API key at app.daytona.io and set DAYTONA_API_KEY."),
            ),
            ProviderRequirement(
                env_vars=("DAYTONA_JWT_TOKEN", "DAYTONA_ORGANIZATION_ID"),
                hint=("Set DAYTONA_JWT_TOKEN with the DAYTONA_ORGANIZATION_ID it belongs to."),
            ),
        ),
        list_environments=_environments_of(SandboxVariant.DAYTONA),
    ),
    SandboxProvider(
        variant=SandboxVariant.E2B,
        icon="e2b",
        title="E2B",
        description=(
            "Sandboxes on E2B, in Firecracker microVMs that start in about 150 ms, "
            "with a stateful Python kernel and rich outputs."
        ),
        extra="e2b",
        requirements=(
            ProviderRequirement(
                env_vars=("E2B_API_KEY",),
                hint="Create an API key at e2b.dev and set E2B_API_KEY.",
            ),
        ),
        list_environments=_environments_of(SandboxVariant.E2B),
    ),
    SandboxProvider(
        variant=SandboxVariant.COREWEAVE,
        title="CoreWeave",
        description=(
            "Containers on CoreWeave, with a stateful Python session and an optional GPU."
        ),
        extra="coreweave",
        requirements=(
            ProviderRequirement(
                env_vars=("CWSANDBOX_API_KEY",),
                hint=("Create an access token in the CoreWeave console and set CWSANDBOX_API_KEY."),
            ),
        ),
        list_environments=_environments_of(SandboxVariant.COREWEAVE),
    ),
    SandboxProvider(
        variant=SandboxVariant.CLOUDFLARE,
        title="Cloudflare",
        description=(
            "Containers on Cloudflare's edge, reached through a deployed sandbox "
            "bridge Worker. Each snippet runs in a process of its own."
        ),
        extra="cloudflare",
        requirements=(
            ProviderRequirement(
                env_vars=("CLOUDFLARE_SANDBOX_API_URL", "CLOUDFLARE_SANDBOX_API_KEY"),
                hint=(
                    "Deploy the sandbox bridge Worker, then set "
                    "CLOUDFLARE_SANDBOX_API_URL to where it answers and "
                    "CLOUDFLARE_SANDBOX_API_KEY to the secret it generated."
                ),
            ),
        ),
        list_environments=_environments_of(SandboxVariant.CLOUDFLARE),
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
            "Code evaluated in this very process. For tests and examples; it isolates nothing."
        ),
        needs_credentials=False,
        list_environments=_environments_of(SandboxVariant.EVAL),
    ),
)


def get_provider(name: str) -> SandboxProvider | None:
    """The provider of that name, if there is one.

    Args:
        name: Identifier of the provider, which is that of its variant.
    """
    # A provider is named by its variant, so the canonical name is what to
    # compare against, whatever spelling the lookup arrived in.
    wanted = normalize_variant(name or "")
    for provider in PROVIDERS:
        if provider.name == wanted:
            return provider
    return None


def available_providers(
    secrets: Mapping[str, str] | None = None,
) -> tuple[SandboxProvider, ...]:
    """The providers whose credentials are on hand, here or in `secrets`."""
    return tuple(p for p in PROVIDERS if p.is_available(secrets))


def provider_catalog(
    secrets: Mapping[str, str] | None = None,
) -> list[dict]:
    """Every provider as plain data, for a service to serve.

    The services of the platform — the operator first — answer "which
    environments exist, and which can this account use" over HTTP; they need
    the registry as JSON, not as dataclasses holding callables. Environments
    are read only for providers that are enabled: asking an unusable provider
    what it ships is a call that fails.
    """
    catalog: list[dict] = []
    for provider in PROVIDERS:
        enabled = provider.is_available(secrets)
        catalog.append(
            {
                "name": provider.name,
                "title": provider.title,
                "description": provider.description,
                "icon": provider.icon,
                "enabled": enabled,
                "needs_credentials": provider.needs_credentials,
                "requirements": [
                    {
                        "env_vars": list(requirement.env_vars),
                        "file": requirement.file,
                        "hint": requirement.hint,
                    }
                    for requirement in provider.requirements
                ],
                "environments": [
                    # What the environment runs on travels with it: a service
                    # listing environments is asked which has a GPU and which
                    # card it is, and that cannot be answered from a name.
                    # Keys the provider did not declare are left out rather
                    # than sent as null, so "did not say" stays tellable from
                    # "has none".
                    {
                        key: value
                        for key, value in {
                            "name": environment.name,
                            "title": environment.title,
                            "language": environment.language,
                            "cpu": environment.cpu,
                            "memory": environment.memory,
                            "gpu": environment.gpu,
                            "gpu_count": environment.gpu_count,
                            "gpu_memory": environment.gpu_memory,
                        }.items()
                        if value is not None
                    }
                    for environment in (provider.environments(secrets) if enabled else [])
                ],
            }
        )
    return catalog
