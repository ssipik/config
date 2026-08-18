"""Pipeline tuning parameters, shared by every stage.

The values live in ``parameters.yaml``; this module holds the schema and the
in-code fallbacks. Two layers, and the stages add a third on top:

    env var  >  parameters.yaml  >  the dataclass defaults below

``parameters.yaml`` is meant to be *generated* — the planned evaluation module
emits an optimized set and it gets blessed into this package under a version.
So the yaml stays values-only and every explanation of *why* a value is what it
is stays here, where a regenerated file cannot destroy it.

Deployment-specific values (DSNs, service URLs, credentials, MLflow URIs) are
deliberately absent: this file is committed, those differ per environment and
stay environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rag_config.resources import resource_path

__all__ = [
    "Params",
    "PlanParams",
    "ParseChunkParams",
    "EmbedParams",
    "FinalizeParams",
    "AgentParams",
    "load_params",
    "params_path",
]


@dataclass(frozen=True)
class PlanParams:
    """Stage 3. Changing these re-plans; no re-ingest."""

    # Bytes of source per batch, and the ceiling on documents in one batch.
    batch_target_cost: int = 15_000_000  # ~100 docs at the 150 KB default proxy
    batch_max_docs: int = 200
    # Stand-in size for a document whose byte count discovery could not read.
    default_doc_cost: int = 150_000
    default_cost_weight: float = 1.0


@dataclass(frozen=True)
class ParseChunkParams:
    """Stage 4. Changing these needs stages 4-5 re-run for the affected docs."""

    # Words, not tokens: the splitter runs with tokenizer=str.split so chunking
    # stays offline and deterministic (see the stage's CLAUDE.md).
    chunk_size: int = 700
    chunk_overlap: int = 100
    # Below this many characters of extracted text per page a PDF is treated as
    # scanned, and skipped rather than embedded as near-empty chunks.
    low_text_chars_per_page: int = 200


@dataclass(frozen=True)
class EmbedParams:
    """Stage 5, and the half of it the agent must match.

    dense_vector / sparse_vectors / sparse_model_id / sparse_avg_len are read by
    both stage 5 and the agent. They used to be duplicated constants in
    embed/qdrant.py and agent/qdrant.py; one definition here is the point of
    this package. Changing any of them invalidates the vectors already in
    Qdrant, so it is a re-embed, not a knob.
    """

    batch_size: int = 32  # chunks per TEI request; the server does the real batching
    qdrant_collection: str = "rag_chunks"
    # Named-vector ids — model-specific so a second model can be added without
    # ambiguity (e.g. dense_e5, sparse_splade).
    dense_vector: str = "dense_bge_m3"
    # Sparse field -> BM25 analyzer language. Every chunk gets both, whatever
    # language it is in: the analyzer defines the sparse id space, so picking one
    # per document would make matching depend on an ingest-time language guess.
    # Each field keeps its own document-frequency stats, hence its own IDF.
    sparse_vectors: dict[str, str] = field(
        default_factory=lambda: {
            "sparse_bm25_de": "german",
            "sparse_bm25_en": "english",
        }
    )
    sparse_model_id: str = "Qdrant/bm25"
    # BM25's length-normalization reference in tokens. It only shapes the
    # document side, but the embedder takes it either way, so it is kept
    # identical on both sides rather than left to drift. The fastembed version
    # is the third part of this contract and is pinned in each pyproject.
    sparse_avg_len: float = 256.0

    def sparse_vector_for(self, analyzer: str) -> str:
        """The sparse field name using this BM25 analyzer.

        Both stages name their German and English legs this way rather than by
        position, so renaming a field in parameters.yaml cannot silently swap
        which analyzer a leg queries.
        """
        for name, lang in self.sparse_vectors.items():
            if lang == analyzer:
                return name
        raise KeyError(
            f"no sparse vector uses the {analyzer!r} analyzer — "
            f"have {sorted(self.sparse_vectors.values())}"
        )


@dataclass(frozen=True)
class FinalizeParams:
    """Stage 6."""

    # Above this share of failed documents the run is not finalized.
    max_parse_error_rate: float = 0.2


@dataclass(frozen=True)
class AgentParams:
    """Serving. All query-time: changing these needs no re-ingest."""

    # Retrieval shape (block 7): each of the three legs prefetches this many
    # candidates, RRF fuses them down to top_k, the reranker cuts that to top_n.
    prefetch_limit: int = 30
    top_k: int = 30
    rerank_top_n: int = 5
    # Qdrant's own default. Small k keeps the ranking signal: rank 1 scores 15x
    # rank 30, where the literature's k=60 (calibrated for rankings thousands
    # deep, not a 30-candidate prefetch) puts them within 1.5x and fusion
    # degenerates into counting how many legs returned a document. That matters
    # here because the two sparse legs agree almost by construction, and because
    # with reranking off the fused order is the final order.
    rrf_k: int = 2
    # RRF weight per leg. Qdrant scores a leg 1/((k-1) + rank/weight) with k=2,
    # so a leg's top hit is worth 1/(1 + 1/weight): 0.5 at weight 1, 0.25 at 1/3.
    # The two sparse legs are the same BM25 query over the same text under two
    # analyzers, so at equal weights lexical evidence contributes up to 1.0
    # against dense's 0.5 — twice the say, for being split in two rather than for
    # being better. 1/3 each caps the pair at dense's 0.5. Revisit if the leg
    # count changes (more languages, or a language filter that makes one
    # analyzer per query enough).
    rrf_dense_weight: float = 1.0
    rrf_sparse_weight: float = 1 / 3
    # Cutting a search excerpt short, off by default. The prompt is already
    # bounded by rerank_top_n x stage 4's chunk_size (5 x 700 words, ~6k tokens
    # worst case on this corpus), so a second cap is not needed to protect the
    # context window — and it cuts on character index, which carries no meaning:
    # measured on the corpus it truncated 19 % of chunks holding 65 % of the
    # text, mid-table and mid-word, after the ranking had already scored them
    # whole. Turn it on only to compare prompt sizes.
    truncate_chunks: bool = False
    max_excerpt_chars: int = 1500
    llm_model: str = "anthropic/claude-haiku-4.5"
    llm_temperature: float = 0.1  # regulatory answers: follow the sources, don't improvise
    llm_max_tokens: int = 2048


@dataclass(frozen=True)
class Params:
    # Identifies the parameter set. Whatever the evaluation module stamps on the
    # set it produced; "v0-baseline" is the values this pipeline ran on before
    # any optimization.
    version: str = "v0-baseline"
    plan: PlanParams = field(default_factory=PlanParams)
    parse_chunk: ParseChunkParams = field(default_factory=ParseChunkParams)
    embed: EmbedParams = field(default_factory=EmbedParams)
    finalize: FinalizeParams = field(default_factory=FinalizeParams)
    agent: AgentParams = field(default_factory=AgentParams)


def params_path(path: str | Path | None = None) -> Path:
    """Where parameters.yaml is read from: the argument, then
    $RAG_PARAMETERS_PATH, then the copy shipped inside this package."""
    if path is not None:
        return Path(path)
    env = os.environ.get("RAG_PARAMETERS_PATH")
    if env:
        return Path(env)
    return resource_path("parameters.yaml")


def _build(cls: type, raw: Any, where: str):
    """One dataclass from its yaml mapping, defaults filling every absent key."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        # Loud, because this file is generated: a renamed key that is silently
        # ignored looks exactly like an optimization that did nothing.
        raise ValueError(
            f"{where}: unknown parameter(s) {', '.join(sorted(unknown))} — "
            f"known: {', '.join(sorted(known))}"
        )
    return cls(**{k: v for k, v in raw.items()})


def _load(path: Path) -> Params:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    known = {f.name for f in fields(Params)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{path}: unknown section(s) {', '.join(sorted(unknown))} — "
            f"known: {', '.join(sorted(known))}"
        )
    return Params(
        version=raw.get("version", Params.version),
        plan=_build(PlanParams, raw.get("plan"), f"{path}: plan"),
        parse_chunk=_build(
            ParseChunkParams, raw.get("parse_chunk"), f"{path}: parse_chunk"
        ),
        embed=_build(EmbedParams, raw.get("embed"), f"{path}: embed"),
        finalize=_build(FinalizeParams, raw.get("finalize"), f"{path}: finalize"),
        agent=_build(AgentParams, raw.get("agent"), f"{path}: agent"),
    )


@lru_cache(maxsize=None)
def _load_cached(resolved: Path) -> Params:
    return _load(resolved)


def load_params(path: str | Path | None = None) -> Params:
    """The parameter set, or the in-code defaults when no file is present.

    A missing file is not an error: it is what tests and a bare ``python -m``
    run see, and the dataclass defaults are the same values the yaml ships.
    """
    resolved = params_path(path)
    if not resolved.is_file():
        return Params()
    return _load_cached(resolved)
