# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Backward-compatible imports for Google Colab client symbols."""

from .google_colab import (  # noqa: F401
    COLAB_CLIENT_AGENT_HEADER,
    COLAB_RUNTIME_PROXY_TOKEN_HEADER,
    COLAB_RUNTIME_PROXY_TOKEN_PARAM,
    DEFAULT_COLAB_CLIENT_AGENT,
    ColabKernelClient,
    GoogleColabKernelClient,
    parse_colab_channels_url,
    parse_google_colab_channels_url,
)
