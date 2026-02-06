"""Tiny type utilities to avoid extra dependencies."""

from __future__ import annotations

from typing import Any, Type


def check_type(x: Any, t: Type, name: str | None = None) -> None:
    if not isinstance(x, t):
        n = name or "value"
        raise TypeError(f"{n} must be {t}, got {type(x)}")
