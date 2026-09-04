from __future__ import annotations

from dataclasses import fields, replace

import pytest

from rag_config import (
    EmbedParams,
    Params,
    Settings,
    load_params,
    load_settings,
    params_path,
    settings_path,
)
from rag_config.params import INGEST_FINGERPRINT_SECTIONS, ingest_fingerprint


def test_packaged_yaml_matches_the_code_defaults():
    """The shipped v0 set is exactly the values the dataclasses fall back to.

    If these diverge, a stage running without the file behaves differently from
    one running with it — the failure this precedence order exists to avoid.
    """
    assert load_params(params_path()) == Params()
    assert load_settings(settings_path()) == Settings()


def test_packaged_yaml_is_versioned():
    assert load_params().version == "v0-baseline"


def test_missing_file_falls_back_to_code_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_PARAMETERS_PATH", str(tmp_path / "absent.yaml"))
    assert load_params() == Params()


def test_partial_file_fills_the_rest_from_defaults(tmp_path):
    p = tmp_path / "parameters.yaml"
    p.write_text("version: exp-1\nagent:\n  top_k: 50\n")
    params = load_params(p)
    assert params.version == "exp-1"
    assert params.agent.top_k == 50
    assert params.agent.rerank_top_n == Params().agent.rerank_top_n
    assert params.parse_chunk.chunk_size == Params().parse_chunk.chunk_size


def test_unknown_parameter_is_loud(tmp_path):
    p = tmp_path / "parameters.yaml"
    p.write_text("agent:\n  top_kk: 50\n")
    with pytest.raises(ValueError, match="unknown parameter"):
        load_params(p)


def test_unknown_section_is_loud(tmp_path):
    p = tmp_path / "parameters.yaml"
    p.write_text("retriever:\n  top_k: 50\n")
    with pytest.raises(ValueError, match="unknown section"):
        load_params(p)


def test_env_var_selects_the_file(tmp_path, monkeypatch):
    p = tmp_path / "parameters.yaml"
    p.write_text("agent:\n  top_k: 7\n")
    monkeypatch.setenv("RAG_PARAMETERS_PATH", str(p))
    assert load_params().agent.top_k == 7


def test_sparse_vector_names_and_analyzers_are_the_cross_stage_pair():
    """Stage 5 writes these named vectors; the agent queries them by name."""
    embed = load_params().embed
    assert embed.dense_vector == "dense_bge_m3"
    assert embed.sparse_vectors == {
        "sparse_bm25_de": "german",
        "sparse_bm25_en": "english",
    }


def test_embed_params_defaults_are_not_shared_between_instances():
    a, b = EmbedParams(), EmbedParams()
    assert a.sparse_vectors == b.sparse_vectors
    assert a.sparse_vectors is not b.sparse_vectors


# --- ingest fingerprint ----------------------------------------------------


def test_query_time_knobs_keep_the_fingerprint():
    """A sweep over retrieval shape must reuse the collection it already built."""
    p = load_params()
    for changed in (
        replace(p, agent=replace(p.agent, top_k=99)),
        replace(p, agent=replace(p.agent, rrf_dense_weight=0.5)),
        replace(p, agent=replace(p.agent, rerank_top_n=1)),
    ):
        assert ingest_fingerprint(changed) == ingest_fingerprint(p)


def test_reindexing_knobs_change_the_fingerprint():
    """Anything that changes what is stored has to get its own collection."""
    p = load_params()
    for changed in (
        replace(p, parse_chunk=replace(p.parse_chunk, chunk_size=384)),
        replace(p, parse_chunk=replace(p.parse_chunk, chunk_overlap=0)),
        replace(p, embed=replace(p.embed, dense_vector="dense_e5")),
        replace(p, embed=replace(p.embed, sparse_model_id="other/bm25")),
        replace(p, embed=replace(p.embed, sparse_avg_len=128.0)),
        replace(p, embed=replace(p.embed, sparse_vectors={"sparse_bm25_en": "english"})),
    ):
        assert ingest_fingerprint(changed) != ingest_fingerprint(p)


def test_config_cannot_reach_the_fingerprint():
    """The two fields that used to need an exception are now out by construction.

    embed_batch_size is throughput only and qdrant_collection is the name being
    derived; both are Settings now, and ingest_fingerprint only ever sees Params.
    """
    assert not {f.name for f in fields(Settings)} & {
        f.name for f in fields(EmbedParams)
    }
    assert "params" in ingest_fingerprint.__code__.co_varnames


def test_fingerprint_is_stable_across_key_order():
    """The hash follows the values, not the order they were written in."""
    a = EmbedParams(sparse_vectors={"sparse_bm25_de": "german", "sparse_bm25_en": "english"})
    b = EmbedParams(sparse_vectors={"sparse_bm25_en": "english", "sparse_bm25_de": "german"})
    p = load_params()
    assert ingest_fingerprint(replace(p, embed=a)) == ingest_fingerprint(replace(p, embed=b))


def test_every_section_is_classified():
    """A new params section must be a deliberate in-or-out decision, not a default."""
    sections = {f.name for f in fields(Params)} - {"version"}
    unclassified = sections - set(INGEST_FINGERPRINT_SECTIONS) - {"agent"}
    assert not unclassified, (
        f"new parameter section(s) {sorted(unclassified)}: decide whether changing them "
        "invalidates the Qdrant collection, then add to INGEST_FINGERPRINT_SECTIONS or here"
    )


def test_the_shipped_fingerprint_has_not_moved():
    """Collection names and evaluation/runs/<fingerprint>/ directories are keyed
    on this. Moving it orphans every index and every recorded sweep."""
    assert ingest_fingerprint() == "9b8010"


def test_params_and_settings_share_no_field_name():
    """The whole point of the split: one value, one home, one name."""
    param_names = {
        f.name
        for section in ("parse_chunk", "embed", "agent")
        for f in fields(type(getattr(load_params(), section)))
    }
    assert not param_names & {f.name for f in fields(Settings)}


def test_settings_are_read_from_their_own_file(tmp_path, monkeypatch):
    written = tmp_path / "s.yaml"
    written.write_text("qdrant_collection: rag_chunks_test\n")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(written))
    assert load_settings().qdrant_collection == "rag_chunks_test"
    assert load_settings().http_timeout == Settings().http_timeout
    # Pointing one file somewhere else must not move the other.
    assert load_params() == Params()


def test_an_unknown_settings_key_is_loud(tmp_path, monkeypatch):
    written = tmp_path / "s.yaml"
    written.write_text("qdrant_colection: typo\n")
    monkeypatch.setenv("RAG_SETTINGS_PATH", str(written))
    with pytest.raises(ValueError, match="unknown parameter"):
        load_settings()


def test_a_settings_key_in_parameters_yaml_is_loud(tmp_path, monkeypatch):
    """The two files are not interchangeable; putting one in the other says so."""
    written = tmp_path / "p.yaml"
    written.write_text("settings:\n  qdrant_collection: rag_chunks\n")
    monkeypatch.setenv("RAG_PARAMETERS_PATH", str(written))
    with pytest.raises(ValueError, match="unknown section"):
        load_params()
