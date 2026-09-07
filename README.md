# rag-config

The one place the pipeline's configuration lives. Every stage depends on this
package instead of carrying its own copy.

- **`sources.yaml`** — what to ingest. One entry per feed/space/directory; the
  reader `type` picks the discover-stage reader, and `detect_deletions` says
  whether absence from a listing is real evidence of deletion.
- **`parameters.yaml`** — how to ingest and retrieve it. Chunk size, batch
  sizing, retrieval shape, RRF weights, the model and sampling settings, and the
  Qdrant vector names stage 5 writes and the agent queries.

## Reading a value

```python
from rag_config import load_sources, load_params

sources = load_sources()          # the packaged sources.yaml
top_k = load_params().agent.top_k
```

Both take an optional path, then fall back to `$RAG_SOURCES_PATH` /
`$RAG_PARAMETERS_PATH`, then to the copy shipped in this package. Stages layer
their environment variables on top, so the precedence a running stage sees is:

    RAG_* env var  >  parameters.yaml  >  the dataclass defaults in params.py

The last layer means a missing yaml is not an error — it is what tests and a
bare `python -m` run see, and it holds the same values the yaml ships.
`tests/test_params.py` asserts the two cannot drift.

## Editing

`sources.yaml` is hand-written. `parameters.yaml` is **generated** — the
evaluation module emits an optimized set and it is blessed in here under a new
`version`. That is why the file carries values only: every explanation of why a
value is what it is lives in `params.py`, where regenerating cannot destroy it.

That blessing is one command, and it lives here:

    uv sync --frozen --extra promote
    python -m rag_config promote --mlflow-model rag-retrieval-params@champion --dry-run
    python -m rag_config promote --mlflow-model rag-retrieval-params/7

It reads a registered MLflow model version — a placeholder carrying the params
of the run it was registered from, not weights — writes all three sections into
`parameters.yaml`, and stamps that address in `version:`, so the shipped file
points back at the run that justified it. `--set-version` overrides the stamp,
`--out` the target. Committing the result is the manual step.

Run from this repo the target defaults to the checked-out `parameters.yaml`,
which is the copy git will see; `MLFLOW_TRACKING_URI` comes from `.env` here the
way it does for the agent.

The version it reads comes from the evaluation module: a `curve` or `sweep` run
logs a `rag-retrieval-params` carrier, and the winner is registered from the
MLflow UI under a name and an alias — see `evaluation/README.md`.

Two modules behind it:

- `baseline.py` resolves `name@alias` or `name/version` to the params of the run
  behind it. **Its MLflow dependency is optional** — `rag-config[promote]`, and
  `mlflow-skinny`, a client with no server or model serving. Every stage installs
  this package *without* the extra, the import is function-local, and
  `__init__.py` does not import this module, so `import rag_config` never touches
  MLflow. `tests/test_baseline.py` asserts that in a subprocess.
- `promote.py` makes the edit. `plan(values, version, path)` returns it — the new
  text, a line per change, whether it needs a re-ingest — and `.write()` is the
  only part that touches disk. It edits values line by line instead of re-dumping
  the file, so the header and the `rerun:` notes survive, and it reads the result
  back through `load_params` and compares field by field before returning: a
  promoted value must land, and nothing else in the file may move.

A `rerun:` note on each entry says what changing it costs, which is what an
optimizer needs to plan a sweep — `top_k` is 20 cheap query runs, `chunk_size`
re-ingests the corpus per candidate.

Deployment values (DSNs, service URLs, credentials, MLflow URIs) are
deliberately absent: this file is committed, and those differ per environment.
They stay environment variables.

## The shared vector names

`embed.dense_vector` and `embed.sparse_vectors` are read by both stage 5, which
writes those named vectors into Qdrant, and the agent, which queries them. They
were two copies of the same constants in `embed/qdrant.py` and
`agent/qdrant.py`; a drift returned no hits rather than an error, which is the
main reason this package exists. Changing them invalidates every vector already
stored, so it is a re-embed, not a knob.

## Consuming it

```toml
[tool.uv.sources]
rag-config = { path = "../config", editable = true }   # now
rag-config = { git = "ssh://…/rag-config", tag = "v0.1.0" }   # once pushed
```

The path source is a stopgap: `uv export` writes it as `-e ../config`, which a
stage's Docker build cannot reach, because each build context is its own stage
directory. Host venvs work either way. Switch to the git source and the build
installs it from `uv.lock` like any other dependency.

## Tests

```bash
uv sync --frozen --extra dev
.venv/bin/python -m pytest
```
