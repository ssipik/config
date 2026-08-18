from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def resource_path(name: str) -> Path:
    """A yaml shipped inside this package, as a real filesystem path.

    The package is always installed from a directory or a wheel unpacked by pip,
    never from a zip, so as_file()'s temp-extraction case cannot arise here.
    """
    return Path(str(files("rag_config").joinpath(name)))
