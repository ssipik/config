"""Read a parameter set back out of the MLflow model registry.

The registry is how a tuned configuration gets a version: an evaluation run logs
a placeholder model carrying that run's own params, and registering it under a
name and an alias makes `name@alias` the address of that configuration. This
module resolves such an address to the params behind it; `promote` writes them
into parameters.yaml.

**MLflow is an optional dependency of this package** — `rag-config[promote]`,
and `mlflow-skinny`, a client with no server or model serving, the same pin the
agent uses. Every stage installs `rag-config` without the extra, so nothing in
the pipeline grows an MLflow dependency from this. The import is therefore
function-local and this module is never imported by `rag_config/__init__.py`:
`import rag_config` must not touch MLflow.

The contact settings below are duplicated from `rag_agent.tracing` and
`rag_evaluation.mlflow_log` on purpose — the same rule the stages follow for
their infrastructure: each module owns its own MLflow contact rather than
importing another package's.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

__all__ = ["Baseline", "BaselineError", "resolve_baseline", "tracking_uri"]

log = logging.getLogger(__name__)

# Where the evaluation runs live. Same default and same variable as
# rag_evaluation.mlflow_log, or an address registered there would not resolve here.
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_EVALUATION", "RAG-Evaluation")


class BaselineError(RuntimeError):
    """The named baseline could not be resolved."""


def tracking_uri() -> str:
    """Where to read from, or "" for nowhere.

    `MLFLOW_TRACKING_URI` is MLflow's own variable, so the library picks up the
    same value without being told.
    """
    return os.environ.get("MLFLOW_TRACKING_URI", "")


@dataclass(frozen=True)
class Baseline:
    """A registered model version, and the run it was registered from.

    `params` is every param on that run, raw and unfiltered. The caller takes
    the half it needs — `promote` maps the three parameter sections out of it by
    prefix.
    """

    ref: str
    name: str
    version: str
    run_id: str
    summary: str
    params: dict[str, str]

    @property
    def fingerprint(self) -> str | None:
        """The ingest fingerprint the run was measured against, if it logged one."""
        return self.params.get("ingest_fingerprint")

    def section(self, prefix: str) -> dict[str, str]:
        """The params under one prefix, with the prefix removed."""
        return {
            key.removeprefix(prefix): value
            for key, value in self.params.items()
            if key.startswith(prefix)
        }


def _configure() -> None:
    """Point MLflow at the server and quiet it down.

    - warnings off: the tracking server's certificate is self-signed
    - insecure TLS: what that server needs to be reachable at all
    """
    import mlflow
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)


def resolve_baseline(ref: str) -> Baseline:
    """One registered model version, given `name@alias` or `name/version`.

    That is the same split MLflow's own model URIs use, and the two things that
    identify exactly one version. Tags are deliberately not accepted: MLflow
    puts no uniqueness constraint on them, so a tag can name several versions
    and picking one would be this module inventing a rule.

    The version is only a pointer. The numbers come from the run it was
    registered from, whose params record what was actually in force — so a
    baseline cannot drift from the run that justified it.
    """
    if not tracking_uri():
        raise BaselineError("no MLFLOW_TRACKING_URI, so no registry to read")

    if "@" in ref:
        name, _, want = ref.partition("@")
        by_alias = True
    elif "/" in ref:
        name, _, want = ref.partition("/")
        by_alias = False
        if not want.isdigit():
            raise BaselineError(f"{ref!r}: a version must be a number")
    else:
        raise BaselineError(f"{ref!r}: expected name@alias or name/version")
    if not name or not want:
        raise BaselineError(f"{ref!r}: expected name@alias or name/version")

    try:
        import mlflow
    except ImportError:
        raise BaselineError(
            "reading a baseline needs MLflow — install rag-config[promote]"
        ) from None

    _configure()
    client = mlflow.MlflowClient()
    try:
        version = (
            client.get_model_version_by_alias(name, want)
            if by_alias
            else client.get_model_version(name, want)
        )
    except Exception as exc:
        kind = "alias" if by_alias else "version"
        raise BaselineError(
            f"{ref!r}: no {kind} {want!r} on model {name!r} ({exc})"
        ) from None
    how = f"alias {want!r}" if by_alias else f"version {want}"

    if not version.run_id:
        raise BaselineError(f"{ref!r}: version {version.version} has no source run")
    run = client.get_run(version.run_id)
    run_name = run.data.tags.get("mlflow.runName", version.run_id[:8])
    return Baseline(
        ref=ref,
        name=name,
        version=str(version.version),
        run_id=version.run_id,
        summary=f"baseline {name} {how} -> run {run_name}",
        params=dict(run.data.params),
    )
