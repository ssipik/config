"""The two things every stage reads, kept apart on purpose.

``Params`` — **model parameters**. Chunking, embedding, retrieval, the LLM.
Changing one changes what the pipeline retrieves or answers, so these are what a
sweep varies, what ``parameters.yaml`` versions, and what an evaluation run
records against its scores.

``Settings`` — **config variables**. Where the shared services are, how long to
wait for them, how work is batched, when a run aborts. Changing one changes
throughput or operations, never an answer.

Nothing appears in both. A function that needs both takes both, rather than one
flattened object where a reader cannot tell which kind of value it is holding —
that conflation is what this split exists to end.

One file each — ``parameters.yaml`` and ``settings.yaml`` — beside
``sources.yaml``, which this package already ships the same way. This module
holds the schema and the in-code fallbacks. Two layers here, and the stages add
a third on top:

    env var  >  parameters.yaml  >  the dataclass defaults below

``parameters.yaml`` is meant to be *generated* — the evaluation module emits an
optimized set and it gets blessed into this package under a version. So the yaml
stays values-only and every explanation of *why* a value is what it is stays
here, where a regenerated file cannot destroy it.

Credentials are deliberately absent from both: DSNs, API tokens and keys stay
environment variables, because this file is committed and packaged. So is
``user_agent``, which is per-component identity rather than a shared value —
stage 5 calls itself `rag-pipeline-ingest`, the agent `rag-pipeline-agent`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from rag_config.resources import resource_path

__all__ = [
    "INGEST_FINGERPRINT_EXCLUDE",
    "INGEST_FINGERPRINT_SECTIONS",
    "AgentParams",
    "EmbedParams",
    "Params",
    "ParseChunkParams",
    "Settings",
    "SparseLeg",
    "ingest_fingerprint",
    "load_params",
    "load_settings",
    "set_param",
    "params_path",
    "settings_path",
    "sparse_legs",
]


# --- model parameters ------------------------------------------------------


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
    """Stage 5's model half, and the part of it the agent must match.

    Model only: what the sparse legs are and how they embed. The Qdrant field
    names those vectors are *stored under* are config and live in ``Settings`` —
    renaming a field moves a vector, it does not change one, which is the same
    reason ``qdrant_collection`` and ``embed_batch_size`` sit there. The two
    halves are joined by ``sparse_legs()``.

    Changing anything here invalidates the vectors already in Qdrant, so it is a
    re-embed, not a knob.
    """

    # What TEI must be serving on ``tei_url``. Declarative, unlike sparse_models
    # above, which fastembed actually resolves: TEI is told its model
    # at container start (../tei/docker-compose.yml) and this stage cannot
    # choose it per request, so nothing sends this value anywhere. It is here to
    # be *hashed* — without it a TEI model swap produces different vectors in the
    # same collection and every score before it silently becomes incomparable.
    # Known gap: nothing verifies TEI is actually serving this. Stage 5 could
    # check it against TEI's /info at startup; it does not yet.
    dense_model: str = "BAAI/bge-m3"

    # Analyzer language -> the sparse model that embeds that leg.
    #
    # Unlike dense_model and agent.rerank_model below, these values are *sent*:
    # they go straight into fastembed's SparseTextEmbedding(model_name=...),
    # which validates them against list_supported_models(). So the spelling is
    # fastembed's, not ours — `Qdrant/bm25` is the HuggingFace repo publishing
    # it, and anything else fails at the first embed call.
    #
    # Every chunk gets every leg, whatever language it is in: the analyzer
    # defines the leg's sparse id space (fastembed derives stemmer and stopwords
    # from the language), so picking one per document would make matching depend
    # on an ingest-time language guess. Each leg keeps its own document-frequency
    # stats, hence its own IDF.
    #
    # Per language rather than one model for all of them because they are
    # expected to diverge: German needs decompounding ("Bankenaufsicht" ->
    # "Banken" + "Aufsicht"), which the Snowball stemmer behind `Qdrant/bm25`
    # does not do. When that lands it changes this one value.
    sparse_models: dict[str, str] = field(
        default_factory=lambda: {
            "german": "Qdrant/bm25",
            "english": "Qdrant/bm25",
        }
    )
    # BM25's length-normalization reference in tokens. It only shapes the
    # document side, but the embedder takes it either way, so it is kept
    # identical on both sides rather than left to drift. The fastembed version
    # is the third part of this contract and is pinned in each pyproject.
    sparse_avg_len: float = 256.0


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
    # Off until the evaluation says it earns its second container. It lives here
    # rather than only in RAG_RERANK_ENABLED so a tuned value has a home in the
    # file like every other knob; on this box the reranker falls back to Candle
    # and costs minutes per query (../tei_rerank/README.md).
    rerank_enabled: bool = False
    # What tei_rerank_url must be serving. Declarative for the same reason as
    # embed.dense_model — one model per container, chosen at start — but *not*
    # fingerprinted: reranking reorders what was already retrieved and touches
    # no stored vector, so changing it is a redeploy, not a re-embed.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Cutting a search excerpt short, off by default. The prompt is already
    # bounded by rerank_top_n x stage 4's chunk_size (5 x 700 words, ~6k tokens
    # worst case on this corpus), so a second cap is not needed to protect the
    # context window — and it cuts on character index, which carries no meaning:
    # measured on the corpus it truncated 19 % of chunks holding 65 % of the
    # text, mid-table and mid-word, after the ranking had already scored them
    # whole. Turn it on only to compare prompt sizes.
    truncate_chunks: bool = False
    max_excerpt_chars: int = 1500

    # fetch_document's two ceilings. Safeguards rather than tuning knobs — the
    # tool reassembles a whole document from its chunks, and both bound what one
    # tool call can put in front of the model. They live here because they shape
    # the answer path like max_excerpt_chars does, and because a limit nobody can
    # see in parameters.yaml is a limit nobody knows they hit.
    #
    # Characters of reassembled text kept. The tail is dropped.
    fetch_max_chars: int = 200_000
    # Chunks pulled from Qdrant for one document. At stage 4's chunk size this is
    # far beyond any real document; it exists so a pathological one cannot pull
    # an unbounded scroll.
    fetch_max_chunks: int = 1024
    llm_model: str = "anthropic/claude-haiku-4.5"
    llm_temperature: float = 0.1  # regulatory answers: follow the sources, don't improvise
    llm_max_tokens: int = 2048


@dataclass(frozen=True)
class Params:
    """The model parameters, and nothing else."""

    # Identifies the parameter set. Whatever the evaluation module stamps on the
    # set it produced; "v0-baseline" is the values this pipeline ran on before
    # any optimization.
    version: str = "v0-baseline"
    parse_chunk: ParseChunkParams = field(default_factory=ParseChunkParams)
    embed: EmbedParams = field(default_factory=EmbedParams)
    agent: AgentParams = field(default_factory=AgentParams)


# --- config variables ------------------------------------------------------


# Every Nomad MCP tool that only reads. Not here: the writes (run/stop/scale,
# drain, create/delete), Nomad Variables and anything ACL or Sentinel — those
# can print secrets. The server's own read-only token enforces the same line.
NOMAD_READ_TOOLS = (
    "list_jobs", "get_job", "get_job_summary", "get_job_allocations",
    "get_job_deployments", "get_job_evaluations", "get_job_services",
    "list_allocations", "get_allocation", "get_allocation_logs",
    "list_deployments", "get_deployment", "list_nodes", "get_node",
    "list_namespaces", "list_volumes", "get_volume",
    "get_cluster_leader", "list_cluster_peers", "list_regions",
)


@dataclass(frozen=True)
class McpServer:
    """One MCP server the agent reads live data from (ARCHITECTURE.md block 8).

    - `description` is prepended to every tool description of this server: the
      model sees tools, never this file, so it is the only place it learns what
      the server knows about.
    - `allowed_tools` is an allow-list, not a deny-list: a server upgrade that
      adds a write tool must not reach the agent unreviewed.
    - `max_result_chars` caps one tool result; an unfiltered API listing can run
      to megabytes.
    - `web_url` is where a person opens what the server reads (links in the
      source list); empty = entries carry no link.
    """

    name: str
    url: str
    description: str = ""
    allowed_tools: tuple[str, ...] = ()
    max_result_chars: int = 20_000
    web_url: str = ""

    def __post_init__(self):
        object.__setattr__(self, "allowed_tools", tuple(self.allowed_tools))


@dataclass(frozen=True)
class Settings:
    """Config shared across stages: services, batching, thresholds.

    Here rather than in each stage's own settings.py for one of two reasons:

    - the stages must **agree**, or the pipeline silently breaks. If stage 5
      writes to `rag_chunks` and the agent queries `rag_chunks_v2`, every search
      returns nothing and looks like a retrieval bug.
    - the same value was copy-pasted into every stage. `http_timeout` was
      declared seven times with the same 30.

    Each stage still has its own Settings for what is genuinely local — its DSN,
    its credentials, its user agent — and takes these as the default a `RAG_*`
    variable overrides.
    """

    # Shared services. One default, not one per stage; the environment moves
    # them per deployment the way it always did.
    qdrant_url: str = "http://localhost:6333"
    tei_url: str = "http://localhost:8080"
    tei_rerank_url: str = "http://localhost:8081"
    http_timeout: float = 30.0
    # Embedding a batch and reranking a candidate set are both far slower than a
    # normal call, so they get their own budgets rather than one shared number.
    tei_timeout: float = 300.0
    rerank_timeout: float = 600.0

    # What stages 5, 6 and the agent must agree on.
    qdrant_collection: str = "rag_chunks"
    # Chunks per TEI request: throughput only, same vectors either way.
    embed_batch_size: int = 32
    # The reranker's equivalent. TEI refuses a client batch above its own
    # --max-client-batch-size with a 422, so without chunking any top_k over 32
    # fails the request outright instead of being slower. Safe to split: a
    # cross-encoder scores each (query, passage) pair independently, so batching
    # changes nothing about the ranking.
    rerank_batch_size: int = 32

    # The Qdrant named-vector ids. Config, not model: a rename moves a vector
    # rather than changing one, the same reason qdrant_collection is here — and
    # it keeps them out of ingest_fingerprint by construction, where a rename
    # would otherwise look like a re-embed.
    #
    # They are named per model so a second one can be added without ambiguity
    # (dense_e5, sparse_splade), and stage 5 writes exactly these names while the
    # agent queries them: a mismatch is an empty result, not an error.
    dense_vector: str = "dense_bge_m3"
    # Analyzer language -> its Qdrant field. The keys must be the same set as
    # Params.embed.sparse_models, which is what sparse_legs() checks.
    sparse_vectors: dict[str, str] = field(
        default_factory=lambda: {
            "german": "sparse_bm25_de",
            "english": "sparse_bm25_en",
        }
    )

    # Stage 3's batching. Bytes of source per batch, and the ceiling on
    # documents in one batch.
    batch_target_cost: int = 15_000_000  # ~100 docs at the 150 KB default proxy
    batch_max_docs: int = 200
    # Stand-in size for a document whose byte count discovery could not read.
    default_doc_cost: int = 150_000
    default_cost_weight: float = 1.0

    # Stage 6's gate: above this share of failed documents the run is not
    # finalized.
    max_parse_error_rate: float = 0.2

    # The agent's live knowledge sources. Empty = corpus only. The agent reads
    # RAG_MCP_<NAME>_URL over `url`, so a deployment moves a server without
    # editing this list.
    mcp_servers: tuple[McpServer, ...] = (
        McpServer(
            name="bitbucket",
            url="http://localhost:3000/mcp",
            description=(
                "Bitbucket workspace get-capital: source code, commits, branches "
                "and pull requests of the internal systems, including this RAG "
                "pipeline (repositories discover, delta, plan, parse_chunk, "
                "embed, finalize, agent, config)."
            ),
            allowed_tools=("bb_get",),
            web_url="https://bitbucket.org",
        ),
        McpServer(
            name="nomad",
            url="http://localhost:8080/mcp",
            description=(
                "Nomad cluster: the live state of jobs, allocations, deployments, "
                "services and nodes, and the logs of running tasks — including this "
                "RAG pipeline's own services (rag-agent, qdrant, rag-postgres, "
                "tei-embed, tei-rerank, bitbucket-mcp, nomad-mcp)."
            ),
            allowed_tools=NOMAD_READ_TOOLS,
            web_url="https://nomad-ui.service.get",
        ),
    )

    def __post_init__(self):
        # yaml gives a list of mappings; checked like any other section, so a
        # misspelt key fails loudly instead of being dropped.
        servers = tuple(
            s if isinstance(s, McpServer) else _build(McpServer, s, "mcp_servers")
            for s in self.mcp_servers or ()
        )
        object.__setattr__(self, "mcp_servers", servers)


# --- loading ---------------------------------------------------------------

# What a top-level key in parameters.yaml may be. settings.yaml is a flat
# mapping in its own file, so a generated parameter set can replace one without
# touching the other.
_PARAM_SECTIONS = ("parse_chunk", "embed", "agent")
_SECTIONS = ("version", *_PARAM_SECTIONS)


def params_path(path: str | Path | None = None) -> Path:
    """Where parameters.yaml is read from: the argument, then
    $RAG_PARAMETERS_PATH, then the copy shipped inside this package."""
    if path is not None:
        return Path(path)
    env = os.environ.get("RAG_PARAMETERS_PATH")
    if env:
        return Path(env)
    return resource_path("parameters.yaml")


def settings_path(path: str | Path | None = None) -> Path:
    """Where settings.yaml is read from: the argument, then $RAG_SETTINGS_PATH,
    then the copy shipped inside this package."""
    if path is not None:
        return Path(path)
    env = os.environ.get("RAG_SETTINGS_PATH")
    if env:
        return Path(env)
    return resource_path("settings.yaml")


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


@lru_cache(maxsize=None)
def _load_raw(resolved: Path, sections: tuple[str, ...] | None) -> dict:
    """One yaml file as a mapping. Cached per path, so pointing
    RAG_PARAMETERS_PATH at a sweep candidate reads that candidate.

    `sections` validates the top level for parameters.yaml; settings.yaml is
    flat and its keys are checked against the dataclass instead.
    """
    raw = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved}: expected a mapping at the top level")
    if sections is not None:
        unknown = set(raw) - set(sections)
        if unknown:
            raise ValueError(
                f"{resolved}: unknown section(s) {', '.join(sorted(unknown))} — "
                f"known: {', '.join(sorted(sections))}"
            )
    return raw


def load_params(path: str | Path | None = None) -> Params:
    """The model parameters, or the in-code defaults when no file is present.

    A missing file is not an error: it is what tests and a bare ``python -m``
    run see, and the dataclass defaults are the same values the yaml ships.
    """
    resolved = params_path(path)
    if not resolved.is_file():
        return Params()
    raw = _load_raw(resolved, _SECTIONS)
    return Params(
        version=raw.get("version", Params.version),
        parse_chunk=_build(
            ParseChunkParams, raw.get("parse_chunk"), f"{resolved}: parse_chunk"
        ),
        embed=_build(EmbedParams, raw.get("embed"), f"{resolved}: embed"),
        agent=_build(AgentParams, raw.get("agent"), f"{resolved}: agent"),
    )


def load_settings(path: str | Path | None = None) -> Settings:
    """The shared config, or the in-code defaults when no file is present."""
    resolved = settings_path(path)
    if not resolved.is_file():
        return Settings()
    return _build(Settings, _load_raw(resolved, None), str(resolved))


@dataclass(frozen=True)
class SparseLeg:
    """One sparse retrieval leg: where it is stored, and what embeds it."""

    language: str
    vector: str
    model: str


def sparse_legs(
    params: Params | None = None, settings: Settings | None = None
) -> dict[str, SparseLeg]:
    """The sparse legs, keyed by analyzer language.

    The model is a parameter and the field name is config, so the two halves
    live in different files. This is the one place they are put back together,
    and so the only place a disagreement between them can be caught — which it
    must be, because a leg whose field name is missing from Qdrant returns no
    hits rather than an error.
    """
    params = params if params is not None else load_params()
    settings = settings if settings is not None else load_settings()
    models_, vectors = params.embed.sparse_models, settings.sparse_vectors
    if set(models_) != set(vectors):
        raise ValueError(
            "sparse legs disagree between the two files — "
            f"parameters.yaml embed.sparse_models has {sorted(models_)}, "
            f"settings.yaml sparse_vectors has {sorted(vectors)}"
        )
    if not models_:
        raise ValueError("no sparse legs configured — hybrid retrieval needs at least one")
    return {
        language: SparseLeg(language, vectors[language], model)
        for language, model in models_.items()
    }


def set_param(params: Params, dotted: str, value: Any) -> Params:
    """`agent.top_k` -> a new Params with that one field changed.

    One dotted vocabulary for everything that moves a single knob — a sweep
    candidate, a curve step, a baseline loaded from MLflow — so the three do not
    each invent their own spelling.
    """
    section, _, name = dotted.partition(".")
    if not name or section not in _PARAM_SECTIONS:
        raise ValueError(
            f"unknown parameter {dotted!r} — expected one of "
            f"{', '.join(f'{s}.<field>' for s in _PARAM_SECTIONS)}"
        )
    inner = getattr(params, section)
    if not any(f.name == name for f in fields(inner)):
        raise ValueError(
            f"unknown parameter {dotted!r} — {section} has "
            f"{', '.join(f.name for f in fields(inner))}"
        )
    return replace(params, **{section: replace(inner, **{name: value})})


# --- which parameters own the shape of the index ---------------------------
#
# Changing one of these invalidates the vectors already in Qdrant, so a sweep
# has to build a separate collection for it; changing anything else is a
# query-time knob that reuses the collection it already has. The split is the
# one the dataclass docstrings already state: stage 4 "needs stages 4-5 re-run",
# stage 5 "invalidates the vectors already in Qdrant", agent "needs no
# re-ingest".
#
# Whole sections are included and exceptions are named, rather than the other
# way round, because the two mistakes do not cost the same: a parameter wrongly
# included buys an unnecessary re-ingest and is obvious, while one wrongly left
# out silently scores the previous index. So a parameter added later is covered
# without anyone remembering to add it here.
INGEST_FINGERPRINT_SECTIONS = ("parse_chunk", "embed")
# Empty, and deliberately kept: the two entries it used to hold —
# embed.batch_size (throughput only) and embed.qdrant_collection (the name being
# derived, which cannot take part in its own hash) — are config, and moving them
# to Settings put them out of reach of the hash by construction rather than by
# exception. A model parameter added to either section is covered automatically;
# one that turns out not to belong goes here.
INGEST_FINGERPRINT_EXCLUDE: frozenset[str] = frozenset()


def ingest_fingerprint(params: Params | None = None, length: int = 6) -> str:
    """Short hash of the parameters that decide what is stored in Qdrant.

    Names one collection per ingest configuration, so an evaluation sweep over
    query-time knobs reuses the index it already built and only a re-chunk or
    re-embed pays for a new one.
    """
    params = params or load_params()
    payload: dict[str, Any] = {}
    for section in INGEST_FINGERPRINT_SECTIONS:
        for key, value in asdict(getattr(params, section)).items():
            name = f"{section}.{key}"
            if name not in INGEST_FINGERPRINT_EXCLUDE:
                payload[name] = value
    # sort_keys so the hash follows the values and not the key order they
    # happened to be written in.
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return digest[:length]
