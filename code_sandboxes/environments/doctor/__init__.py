# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""``datalayer-sandbox doctor``: the contract, checked from inside a sandbox.

:mod:`.datalayer_sandbox` is the checker. It uses nothing but the standard
library, because it runs inside images that have none of this package's
dependencies, and it ships into every image as a zipapp built by
:mod:`.build`:

    python -m code_sandboxes.environments.doctor.build dist/datalayer-sandbox
"""
