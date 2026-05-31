"""Shared CLI output helpers for the validate step.

Import from here instead of duplicating in every module::

    from mlip_pipeline.validate._cli import banner, step, ok, warn
"""
from __future__ import annotations

_W = 60  # total banner width


def banner(title: str) -> None:
    """Print a section separator banner."""
    pad = (_W - len(title) - 2) // 2
    right = _W - pad - len(title) - 2
    print(f"\n{'─' * pad} {title} {'─' * right}")


def step(tag: str, msg: str) -> None:
    """Print a plain informational line."""
    print(f"  [{tag:<8}]  {msg}")


def ok(tag: str, msg: str) -> None:
    """Print a success line."""
    print(f"  [{tag:<8}]  ✓  {msg}")


def warn(tag: str, msg: str) -> None:
    """Print a warning line."""
    print(f"  [{tag:<8}]  ⚠  {msg}")
