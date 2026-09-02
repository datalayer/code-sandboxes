# Copyright (c) 2023-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Describing some providers must not read the ones left out.

`provider_catalog` reads each enabled provider's environments, and reading
them is a network call. The `datalayer` provider's listing calls the
platform's own `/api/runtimes/v1/environments`, which the operator answers —
so an operator that described every provider in order to keep six of them
called the runtimes gateway, which called the operator, until every request
in the cycle timed out.

The property under test is therefore not that the filtered names are absent
from the result. It is that a name left out is never ASKED.

Launch the tests:
```
$ pytest tests/test_provider_catalog_names.py -v
```
"""

from __future__ import annotations

import pytest

# `P` because this file compares the module's constants against the catalog
# built from them, and both names appear on nearly every line; spelling the
# module out twice per assertion would bury what is being compared.
from code_sandboxes import providers as P  # noqa: N812
from code_sandboxes.providers import provider_catalog


@pytest.fixture
def asked(monkeypatch):
    """Record every provider whose environments get read.

    The registry is replaced with stand-ins that need no credentials, so
    every provider is enabled and the only thing deciding who gets asked is
    the filter under test — not which keys this machine happens to hold.
    """
    import dataclasses

    seen: list[str] = []

    def watched(provider):
        def listing(name=provider.name, **kwargs):
            seen.append(name)
            return []

        return dataclasses.replace(provider, needs_credentials=False, list_environments=listing)

    monkeypatch.setattr(P, "PROVIDERS", tuple(watched(p) for p in P.PROVIDERS))
    return seen


class TestWhoGetsAsked:
    def test_without_names_every_provider_is_asked(self, asked):
        provider_catalog()
        assert set(asked) == {p.name for p in P.PROVIDERS}

    def test_a_name_left_out_is_never_asked(self, asked):
        provider_catalog(None, ("e2b", "kaggle"))
        assert set(asked) == {"e2b", "kaggle"}

    def test_the_datalayer_provider_is_not_asked_for_an_external_listing(self, asked):
        """The cycle in one line: this is the provider whose listing calls
        the endpoint the operator answers."""
        provider_catalog(None, ("cloudflare", "coreweave", "daytona", "e2b", "kaggle", "modal"))
        assert "datalayer" not in asked

    def test_an_empty_selection_asks_nobody(self, asked):
        """Distinct from `None`, which means every provider. A service that
        computes its own list and comes up empty must get silence, not the
        whole catalogue."""
        assert provider_catalog(None, ()) == []
        assert asked == []


class TestWhatComesBack:
    def test_only_the_named_are_described(self):
        assert [e["name"] for e in provider_catalog(None, ("e2b", "kaggle"))] == [
            "kaggle",
            "e2b",
        ]

    def test_registry_order_is_kept_not_the_caller_s(self):
        """`PROVIDERS` order is what a listing shows; the argument is a
        selection, not a sort."""
        names = [e["name"] for e in provider_catalog(None, ("e2b", "kaggle"))]
        registry = [p.name for p in P.PROVIDERS]
        assert names == sorted(names, key=registry.index)

    def test_a_name_no_provider_has_is_simply_absent(self):
        assert [e["name"] for e in provider_catalog(None, ("e2b", "nobody"))] == ["e2b"]

    def test_names_are_normalised_like_every_other_lookup(self):
        """`normalize_variant` is what `provider_for` compares against, so a
        spelling that finds a provider there finds it here."""
        assert [e["name"] for e in provider_catalog(None, ("E2B",))] == ["e2b"]

    def test_entries_are_unchanged_by_filtering(self):
        whole = {e["name"]: e for e in provider_catalog()}
        assert provider_catalog(None, ("e2b",)) == [whole["e2b"]]
