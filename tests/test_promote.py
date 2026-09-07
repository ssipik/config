"""Editing parameters.yaml in place.

The file is committed and its comments say what changing a value costs, so what
is tested here is that a promoted value lands and *nothing else in the file
moves* — a line edit that goes astray has to fail, not ship.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_config import (
    PromoteError,
    cast_section,
    load_params,
    plan,
    rewrite,
    run_promote,
    section_values,
)
from rag_config.baseline import Baseline
from rag_config.params import _PARAM_SECTIONS, params_path
from rag_config.promote import SECTION_PREFIX, SECTION_TYPES

FILE = """\
# The model parameters every stage reads.
#
# Values only, by design.

version: v0-baseline

parse_chunk:
  chunk_size: 700                 # rerun: stages 4-5, words not tokens
  chunk_overlap: 100              # rerun: stages 4-5

embed:
  dense_vector: dense_bge_m3      # rerun: stages 4-5 + agent redeploy
  sparse_vectors:                 # rerun: stages 4-5 + agent redeploy
    sparse_bm25_de: german
    sparse_bm25_en: english
  sparse_avg_len: 256.0           # rerun: stages 4-5 + agent redeploy

agent:
  top_k: 30                       # rerun: none
  rerank_enabled: false           # rerun: none
  rrf_sparse_weight: 0.3333333333333333   # rerun: none
  llm_model: anthropic/claude-haiku-4.5   # rerun: none
"""


def loaded(text: str):
    """The rewritten text through the real loader."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
    path = Path(handle.name)
    try:
        return load_params(path)
    finally:
        path.unlink(missing_ok=True)


class TestSections:
    def test_the_sections_are_the_ones_params_loads(self):
        # or promote would silently skip a section load_params reads
        assert tuple(SECTION_TYPES) == _PARAM_SECTIONS


class TestCasting:
    def test_text_becomes_the_type_its_field_declares(self):
        assert cast_section(
            "agent",
            {
                "top_k": "12",
                "rerank_enabled": "True",
                "rrf_sparse_weight": "0.5",
                "llm_model": "anthropic/claude-haiku-4.5",
            },
        ) == {
            "top_k": 12,
            "rerank_enabled": True,
            "rrf_sparse_weight": 0.5,
            "llm_model": "anthropic/claude-haiku-4.5",
        }

    def test_a_mapping_survives_the_repr_it_was_stored_as(self):
        assert cast_section("embed", {"sparse_vectors": "{'sparse_bm25_de': 'german'}"}) == {
            "sparse_vectors": {"sparse_bm25_de": "german"}
        }

    def test_what_is_not_a_knob_is_dropped(self):
        # an MLflow run also logs the collection and the parameters file it read
        assert cast_section("agent", {"top_k": "12", "qdrant_collection": "rag_chunks"}) == {
            "top_k": 12
        }

    def test_a_value_of_the_wrong_type_is_an_error_not_a_written_file(self):
        with pytest.raises(PromoteError, match="not a int"):
            cast_section("agent", {"top_k": "twelve"})

    def test_an_unknown_section_is_refused(self):
        with pytest.raises(PromoteError, match="unknown section"):
            cast_section("retrieval", {"top_k": "12"})


class TestRewrite:
    def test_the_header_and_the_rerun_notes_survive(self):
        new, _ = rewrite(FILE, {"agent": {"top_k": 12}})
        assert "# The model parameters every stage reads." in new
        assert "  top_k: 12                       # rerun: none" in new

    def test_only_the_named_values_move(self):
        new, changes = rewrite(FILE, {"agent": {"top_k": 12}})
        assert changes == ["agent.top_k: 30 -> 12"]
        assert loaded(new).agent.rerank_enabled is False
        assert loaded(new).parse_chunk.chunk_size == 700

    def test_a_bool_is_written_as_yaml_spells_it(self):
        new, _ = rewrite(FILE, {"agent": {"rerank_enabled": True}})
        assert "  rerank_enabled: true" in new
        assert loaded(new).agent.rerank_enabled is True

    def test_a_nested_mapping_is_replaced_whole(self):
        new, changes = rewrite(
            FILE, {"embed": {"sparse_vectors": {"sparse_bm25_fr": "french"}}}
        )
        assert loaded(new).embed.sparse_vectors == {"sparse_bm25_fr": "french"}
        # the key line keeps its comment, and the key after the block is intact
        assert "  sparse_vectors:                 # rerun:" in new
        assert loaded(new).embed.sparse_avg_len == 256.0
        assert changes

    def test_a_key_the_file_does_not_have_is_appended_to_its_section(self):
        new, changes = rewrite(FILE, {"agent": {"rrf_k": 60}})
        assert loaded(new).agent.rrf_k == 60
        assert "(absent) -> 60" in changes[0]
        # inside the agent block, not after the file's last line
        assert new.index("rrf_k") > new.index("llm_model")

    def test_a_section_the_file_does_not_have_is_appended_whole(self):
        without = FILE[: FILE.index("agent:")].rstrip() + "\n"
        new, changes = rewrite(without, {"agent": {"top_k": 12}})
        assert loaded(new).agent.top_k == 12
        assert "absent section" in changes[0]

    def test_the_version_is_stamped_when_one_is_given(self):
        new, changes = rewrite(FILE, {}, "m/7")
        assert loaded(new).version == "m/7"
        assert changes == ["version: v0-baseline -> m/7"]

    def test_an_unchanged_value_is_not_reported_as_a_change(self):
        new, changes = rewrite(FILE, {"agent": {"top_k": 30}}, "v0-baseline")
        assert changes == []
        assert new == FILE

    def test_an_unknown_section_is_refused(self):
        with pytest.raises(PromoteError, match="unknown section"):
            rewrite(FILE, {"retrieval": {"top_k": 12}})

    def test_the_shipped_file_survives_a_round_trip(self):
        # the real thing, not the fixture: its layout is what this has to keep
        shipped = params_path().read_text(encoding="utf-8")
        new, changes = rewrite(shipped, {"agent": {"top_k": 12}})
        assert changes == ["agent.top_k: 30 -> 12"]
        assert new.replace("top_k: 12", "top_k: 30", 1) == shipped


class TestPlan:
    def test_it_prepares_the_edit_without_touching_the_file(self, tmp_path):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")

        edit = plan({"agent": {"top_k": 12}}, version="m/7", path=target)
        assert edit.changes == ["version: v0-baseline -> m/7", "agent.top_k: 30 -> 12"]
        assert target.read_text() == FILE
        assert "-  top_k: 30" in edit.diff()

        edit.write()
        assert load_params(target).agent.top_k == 12

    def test_a_query_side_change_needs_no_re_ingest(self, tmp_path):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        assert not plan({"agent": {"top_k": 12}}, path=target).needs_reingest

    def test_an_index_shape_change_does(self, tmp_path):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        edit = plan({"parse_chunk": {"chunk_size": 900}}, path=target)
        assert edit.needs_reingest
        assert len(set(edit.fingerprints)) == 2

    def test_a_missing_target_is_an_error_not_a_new_file(self, tmp_path):
        with pytest.raises(PromoteError, match="no parameters.yaml"):
            plan({"agent": {"top_k": 12}}, path=tmp_path / "absent.yaml")


def baseline(**params: str) -> Baseline:
    return Baseline(
        ref="m@champion",
        name="m",
        version="7",
        run_id="abc123",
        summary="baseline m alias 'champion' -> run curve-top_k",
        params=params,
    )


def _resolves_to(monkeypatch, result: Baseline) -> None:
    """Stand in for the registry call — everything below it is about the file."""
    monkeypatch.setattr("rag_config.baseline.resolve_baseline", lambda ref: result)


class TestSectionValues:
    def test_the_agent_section_comes_from_the_run_prefix(self):
        # an evaluation run logs AgentParams as `run.*`, not `agent.*`
        assert SECTION_PREFIX["agent"] == "run."

    def test_the_three_sections_come_from_their_own_prefixes(self):
        values = section_values(
            baseline(
                **{
                    "parse_chunk.chunk_size": "900",
                    "embed.sparse_avg_len": "300.0",
                    "run.top_k": "12",
                }
            )
        )
        assert values["parse_chunk"] == {"chunk_size": 900}
        assert values["embed"] == {"sparse_avg_len": 300.0}
        assert values["agent"] == {"top_k": 12}

    def test_what_the_run_records_but_cannot_be_set_is_dropped(self):
        values = section_values(
            baseline(**{"run.top_k": "12", "ingest_fingerprint": "abc123"})
        )
        assert values["agent"] == {"top_k": 12}

    def test_a_run_the_evaluation_module_did_not_log_is_refused(self):
        with pytest.raises(PromoteError, match="not logged by the evaluation module"):
            section_values(baseline(some_other_param="1"))


class TestRunPromote:
    def test_it_writes_the_file_and_stamps_the_registry_address(
        self, tmp_path, monkeypatch, capsys
    ):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        _resolves_to(monkeypatch, baseline(**{"run.top_k": "12"}))

        assert run_promote("m@champion", out=target) == 0
        written = load_params(target)
        assert written.agent.top_k == 12
        assert written.version == "m/7"
        assert "# rerun: none" in target.read_text()
        assert "written" in capsys.readouterr().out

    def test_set_version_wins_over_the_registry_address(self, tmp_path, monkeypatch):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        _resolves_to(monkeypatch, baseline(**{"run.top_k": "12"}))

        run_promote("m@champion", out=target, version="v2-tuned")
        assert load_params(target).version == "v2-tuned"

    def test_a_dry_run_prints_the_diff_and_writes_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        _resolves_to(monkeypatch, baseline(**{"run.top_k": "12"}))

        assert run_promote("m@champion", out=target, dry_run=True) == 0
        assert target.read_text() == FILE
        out = capsys.readouterr().out
        assert "-  top_k: 30" in out and "nothing written" in out

    def test_a_re_ingest_is_called_out_when_the_index_shape_moved(
        self, tmp_path, monkeypatch, capsys
    ):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        _resolves_to(monkeypatch, baseline(**{"parse_chunk.chunk_size": "900"}))

        run_promote("m@champion", out=target)
        assert "stages 4-5 must be re-run" in capsys.readouterr().out

    def test_a_baseline_the_file_already_holds_changes_nothing(
        self, tmp_path, monkeypatch, capsys
    ):
        target = tmp_path / "parameters.yaml"
        target.write_text(FILE, encoding="utf-8")
        _resolves_to(monkeypatch, baseline(**{"run.top_k": "30"}))

        run_promote("m@champion", out=target, version="v0-baseline")
        assert target.read_text() == FILE
        assert "nothing to change" in capsys.readouterr().out
