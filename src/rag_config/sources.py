from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from rag_config.resources import resource_path


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: str  # reader type, key into READER_REGISTRY
    source_system: str
    url: str | None = None
    # local_files sources: base directory to scan and the glob within it
    # (recursive by default; the reader keeps only regular files).
    path: str | None = None
    glob: str = "**/*"
    doc_type: str | None = None
    default_language: str | None = None
    # Which PDF parser this source's PDFs get (PDF_PARSERS in the parse_chunk
    # stage's parsers.py): unset = the layout engine, 'fast' = the ~5x faster
    # mode that finds no tables. Only the parse_chunk stage reads it.
    pdf_parser: str | None = None
    # confluence sources: the Cloud site, the space, and the label that opts a
    # page into the corpus; include_attachments also lists each labelled page's
    # files as their own documents. Removing the label removes the page from the
    # corpus — the space+label listing is a complete snapshot, so the delta stage
    # reads the absence as a deletion.
    base_url: str | None = None
    space_key: str | None = None
    label: str | None = None
    include_attachments: bool = False
    # False for rolling-window sources (e.g. RSS feeds showing only the ~15
    # newest items): absence from a discovery run is not a deletion signal,
    # so the delta stage must exclude them from deletion detection.
    detect_deletions: bool = True


def sources_path(path: str | Path | None = None) -> Path:
    """Where sources.yaml is read from: the argument, then $RAG_SOURCES_PATH,
    then the copy shipped inside this package."""
    if path is not None:
        return Path(path)
    env = os.environ.get("RAG_SOURCES_PATH")
    if env:
        return Path(env)
    return resource_path("sources.yaml")


def load_sources(path: str | Path | None = None) -> dict[str, SourceConfig]:
    resolved = sources_path(path)
    raw = yaml.safe_load(resolved.read_text())
    sources = (raw or {}).get("sources")
    if not sources:
        raise ValueError(f"no 'sources' section in {resolved}")
    return {sid: SourceConfig(id=sid, **cfg) for sid, cfg in sources.items()}
