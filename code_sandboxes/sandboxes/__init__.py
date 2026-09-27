# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

"""The sandboxes, one subpackage per provider.

Each provider lives in its own package — its implementation in
``<provider>/<provider>.py``, re-exported by the package's ``__init__``, and
whatever else only it needs beside it (Kaggle's kernel client
and executors, Marimo's reactive graph and cells driver, Google Colab's
kernel client). The public names are re-exported from
:mod:`code_sandboxes`, which is the import path programs should use; these
modules are where the code lives.
"""
