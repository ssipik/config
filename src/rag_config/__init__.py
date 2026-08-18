"""Shared configuration for the RAG pipeline: what to ingest, and with which
parameters.

Every stage depends on this package instead of carrying its own copy, so a
source definition or a tuning value has exactly one place to change. See
params.py for the precedence rule (env > yaml > code default).
"""

from rag_config.params import (
    AgentParams,
    EmbedParams,
    FinalizeParams,
    Params,
    ParseChunkParams,
    PlanParams,
    load_params,
    params_path,
)
from rag_config.sources import SourceConfig, load_sources, sources_path

__all__ = [
    "SourceConfig",
    "load_sources",
    "sources_path",
    "Params",
    "PlanParams",
    "ParseChunkParams",
    "EmbedParams",
    "FinalizeParams",
    "AgentParams",
    "load_params",
    "params_path",
]
