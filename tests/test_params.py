from __future__ import annotations

import pytest

from rag_config import EmbedParams, Params, load_params, params_path
from rag_config.params import _load


def test_packaged_yaml_matches_the_code_defaults():
    """The shipped v0 set is exactly the values the dataclasses fall back to.

    If these diverge, a stage running without the file behaves differently from
    one running with it — the failure this precedence order exists to avoid.
    """
    assert _load(params_path()) == Params()


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
