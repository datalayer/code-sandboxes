# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The provider-specific Environment builders, one module per variant.

Not neutral: each module speaks its provider's SDK. They are loaded only by
:func:`code_sandboxes.environments.builders.get_builder`, and only where the
build runs — the durable worker — never by a service that merely describes
or launches an Environment.
"""
