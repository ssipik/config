from __future__ import annotations

import pytest

from rag_config import load_sources, sources_path


def test_packaged_copy_loads_without_env_or_cwd():
    """No argument, no env var: the copy shipped in the package."""
    sources = load_sources()
    assert "ecb_press" in sources
    assert sources["ecb_press"].source_system == "web_rss"


def test_every_source_declares_a_reader_type():
    for sid, cfg in load_sources().items():
        assert cfg.type, sid
        assert cfg.id == sid


def test_rolling_window_feeds_opt_out_of_deletion_detection():
    sources = load_sources()
    assert sources["ecb_press"].detect_deletions is False
    assert sources["local_pdfs"].detect_deletions is True


def test_env_var_overrides_the_packaged_copy(tmp_path, monkeypatch):
    other = tmp_path / "sources.yaml"
    other.write_text("sources:\n  only:\n    type: rss\n    source_system: web_rss\n")
    monkeypatch.setenv("RAG_SOURCES_PATH", str(other))
    assert set(load_sources()) == {"only"}
    assert sources_path() == other


def test_explicit_path_beats_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_SOURCES_PATH", "/nonexistent/sources.yaml")
    explicit = tmp_path / "sources.yaml"
    explicit.write_text("sources:\n  arg:\n    type: rss\n    source_system: web_rss\n")
    assert set(load_sources(explicit)) == {"arg"}


def test_missing_sources_section_is_an_error(tmp_path):
    empty = tmp_path / "sources.yaml"
    empty.write_text("{}\n")
    with pytest.raises(ValueError, match="no 'sources' section"):
        load_sources(empty)


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "sources.yaml"
    bad.write_text(
        "sources:\n  x:\n    type: rss\n    source_system: web_rss\n    typo: 1\n"
    )
    with pytest.raises(Exception):
        load_sources(bad)
