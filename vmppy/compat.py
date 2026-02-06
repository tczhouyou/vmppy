"""Small utilities to replace external dependencies."""

from __future__ import annotations

import inspect


def print_instantiation_arguments(cls, config_dict):
    """Best-effort pretty printer.

    The original upstream used `pyutils.functions_tools.print_instantiation_arguments`.
    Here we just print keys that match the __init__ signature.
    """
    try:
        sig = inspect.signature(cls.__init__)
        keys = set(sig.parameters.keys())
    except Exception:
        keys = set(config_dict.keys())

    use = {k: v for k, v in config_dict.items() if k in keys}
    print(f"[{cls.__name__}] {use}")
