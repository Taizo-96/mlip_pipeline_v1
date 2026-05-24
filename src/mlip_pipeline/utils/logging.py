from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn,
        BarColumn, TaskProgressColumn, TimeElapsedColumn,
    )
    _RICH = True
except ImportError:
    _RICH = False

_console = None

def _get_console():
    global _console
    if _console is None:
        if _RICH:
            from rich.console import Console
            _console = Console()
        else:
            class _Fallback:
                def print(self, *a, **kw): print(*a)
            _console = _Fallback()
    return _console

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def step_header(name: str, generation: str | None = None) -> None:
    gen_tag = f" · gen_{generation}" if generation else ""
    if _RICH:
        _get_console().print(
            f"\n[bold cyan]{'─'*60}\n▶  {name}{gen_tag}\n{'─'*60}[/bold cyan]"
        )
    else:
        print(f"\n{'='*60}\n>>  {name}{gen_tag}\n{'='*60}")

def info(msg: str) -> None:
    if _RICH:
        _get_console().print(f"[dim]{_ts()}[/dim]  {msg}")
    else:
        print(f"[{_ts()}]  {msg}")

def warn(msg: str) -> None:
    if _RICH:
        _get_console().print(f"[dim]{_ts()}[/dim]  [bold yellow]WARNING[/bold yellow] {msg}")
    else:
        print(f"[{_ts()}] WARNING: {msg}")

def error(msg: str) -> None:
    if _RICH:
        _get_console().print(f"[dim]{_ts()}[/dim]  [bold red]ERROR[/bold red] {msg}")
    else:
        print(f"[{_ts()}] ERROR: {msg}")

def success(msg: str) -> None:
    if _RICH:
        _get_console().print(f"[dim]{_ts()}[/dim]  [bold green]✓[/bold green] {msg}")
    else:
        print(f"[{_ts()}] SUCCESS: {msg}")

@contextmanager
def make_progress(description: str, total: int):
    if _RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            yield progress
    else:
        class _FakeProgress:
            def add_task(self, *a, **kw): return 0
            def advance(self, *a): pass
        yield _FakeProgress()