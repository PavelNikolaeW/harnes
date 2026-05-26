"""Builtin tool implementations.

Регистрация всех built-in тулов в реестре через `register_builtins(registry)`.
"""
from __future__ import annotations

from harnes.tools.builtin import io as _io


def register_builtins(registry: "ToolRegistry") -> None:  # type: ignore[name-defined]
    """Registers read_file, write_file."""
    _io.register(registry)


# Avoid circular import at type-check time:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harnes.tools.registry import ToolRegistry  # noqa: F401
