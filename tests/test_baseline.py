"""Resolving a registry address, offline — no tracking server, no MLflow needed.

What is checked here is everything that happens before the client call: the two
address forms, and that the optional dependency stays optional.
"""

from __future__ import annotations

import pytest

from rag_config.baseline import Baseline, BaselineError, resolve_baseline, tracking_uri


class TestOptionalDependency:
    def test_importing_rag_config_does_not_import_mlflow(self):
        # the whole point of the extra: six stages install this package and none
        # of them talk to MLflow
        import subprocess
        import sys

        code = "import rag_config, sys; print('mlflow' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


class TestAddress:
    def test_without_a_tracking_uri_there_is_nothing_to_read(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert tracking_uri() == ""
        with pytest.raises(BaselineError, match="no MLFLOW_TRACKING_URI"):
            resolve_baseline("m@champion")

    @pytest.mark.parametrize("ref", ["m", "@champion", "m@", "/7"])
    def test_an_address_that_names_no_single_version_is_refused(self, ref, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
        with pytest.raises(BaselineError, match="expected name@alias or name/version"):
            resolve_baseline(ref)

    @pytest.mark.parametrize("ref", ["m/champion", "m/"])
    def test_a_version_must_be_a_number(self, ref, monkeypatch):
        # or `m/champion` would look like a version and resolve to nothing
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
        with pytest.raises(BaselineError, match="a version must be a number"):
            resolve_baseline(ref)


class TestBaseline:
    def test_a_section_comes_back_without_its_prefix(self):
        baseline = Baseline(
            ref="m@a",
            name="m",
            version="7",
            run_id="x",
            summary="s",
            params={"run.top_k": "12", "embed.sparse_avg_len": "256.0"},
        )
        assert baseline.section("run.") == {"top_k": "12"}
        assert baseline.section("embed.") == {"sparse_avg_len": "256.0"}

    def test_the_fingerprint_is_absent_rather_than_wrong_when_unlogged(self):
        assert Baseline("m@a", "m", "7", "x", "s", {}).fingerprint is None
