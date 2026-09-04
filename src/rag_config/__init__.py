"""Shared configuration for the RAG pipeline: what to ingest, and with which
parameters.

Every stage depends on this package instead of carrying its own copy, so a
source definition or a tuning value has exactly one place to change.

Two kinds of value, kept in separate types — see params.py for why, and for the
precedence rule (env > yaml > code default):

    Params    model parameters: chunking, embedding, retrieval, the LLM
    Settings  config variables: shared services, batching, thresholds
"""

from rag_config.params import (
    INGEST_FINGERPRINT_EXCLUDE,
    INGEST_FINGERPRINT_SECTIONS,
    AgentParams,
    EmbedParams,
    Params,
    ParseChunkParams,
    Settings,
    ingest_fingerprint,
    load_params,
    load_settings,
    params_path,
    set_param,
    settings_path,
)
from rag_config.sources import SourceConfig, load_sources, sources_path

__all__ = [
    "INGEST_FINGERPRINT_EXCLUDE",
    "INGEST_FINGERPRINT_SECTIONS",
    "AgentParams",
    "EmbedParams",
    "Params",
    "ParseChunkParams",
    "Settings",
    "SourceConfig",
    "ingest_fingerprint",
    "load_params",
    "load_settings",
    "load_sources",
    "params_path",
    "set_param",
    "settings_path",
    "sources_path",
]
