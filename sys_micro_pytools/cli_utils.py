"""Shared helpers for Click CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Type, Union

import click

ItemType = Union[Type[int], Type[float], Type[str]]


def split_ws(
    item_type: ItemType = str,
    expected_count: Optional[int] = None,
    *,
    check_path_exists: bool = False,
) -> Callable[[click.Context, click.Parameter, Any], Optional[tuple]]:
    """Build a Click callback that splits a quoted, whitespace-separated string.

    Users pass a single option value, e.g. ``--ref_wells "A01 A02 B03"``.

    - Returns ``None`` if *value* is ``None`` or blank after stripping.
    - Otherwise returns a ``tuple`` of items, each cast with *item_type*.
    - If *expected_count* is set, raises :class:`click.BadParameter` unless the
      number of whitespace-separated tokens matches.
    - If *check_path_exists* is true, each token must be an existing path
      (file or directory).
    """

    def callback(
        ctx: click.Context, param: click.Parameter, value: Any
    ) -> Optional[tuple]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        stripped = value.strip()
        if not stripped:
            return None
        tokens = stripped.split()
        if expected_count is not None and len(tokens) != expected_count:
            raise click.BadParameter(
                f"expected {expected_count} whitespace-separated value(s) for "
                f"{param.name}, got {len(tokens)} ({tokens!r})"
            )
        out: list = []
        for raw in tokens:
            try:
                if item_type is str:
                    item: Any = raw
                else:
                    item = item_type(raw)
            except (ValueError, TypeError) as exc:
                raise click.BadParameter(
                    f"cannot convert {raw!r} to {getattr(item_type, '__name__', item_type)!r}"
                ) from exc
            if check_path_exists:
                p = Path(item)
                if not p.exists():
                    raise click.BadParameter(f"path does not exist: {p}")
                item = str(p)
            out.append(item)
        return tuple(out)

    return callback
