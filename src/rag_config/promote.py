"""Write a tuned parameter set into parameters.yaml, in place.

parameters.yaml is generated territory (see params.py) but it is also committed,
and it carries a header plus a `rerun:` note per key saying what changing that
value costs. `yaml.safe_dump` of a `Params` would delete every one of them — so
this replaces values line by line and leaves the rest of the file byte for byte.

Here rather than in the evaluation module because this is knowledge about *this
file's format*, next to the loader that reads it and the tests that guard it.
Where the values come from is `baseline.py`, whose MLflow dependency is optional
(`rag-config[promote]`) so no stage grows one.

Not to be confused with the candidate files a sweep writes under `runs/` — those
are machine-read scratch output with no comments to keep, and `safe_dump` is
right for them.
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from rag_config.params import (
    AgentParams,
    EmbedParams,
    Params,
    ParseChunkParams,
    ingest_fingerprint,
    load_params,
    params_path,
)

__all__ = [
    "SECTION_PREFIX",
    "PromoteError",
    "Rewrite",
    "cast_section",
    "plan",
    "rewrite",
    "run_promote",
    "section_values",
]

# The same three sections params.py loads, with the dataclass each one builds.
# `tests/test_params.py` asserts the two lists cannot drift.
SECTION_TYPES: dict[str, type] = {
    "parse_chunk": ParseChunkParams,
    "embed": EmbedParams,
    "agent": AgentParams,
}


class PromoteError(ValueError):
    """The values could not be written into the file."""


# --- typing values that arrive as text --------------------------------------


def _field_types(cls: type) -> dict[str, type]:
    """Field name -> type, taken from an instance rather than the annotations.

    This module's `from __future__ import annotations` makes `field.type` a
    string, and `sparse_vectors` has a default_factory rather than a default —
    an instance answers both cases.
    """
    instance = cls()
    return {f.name: type(getattr(instance, f.name)) for f in fields(cls)}


def _cast(raw: str, kind: type, where: str) -> Any:
    """One text value back to the type its field declares."""
    text = raw.strip()
    if kind is bool:
        if text.lower() not in {"true", "false"}:
            raise PromoteError(f"{where}: {raw!r} is not true or false")
        return text.lower() == "true"
    if kind is dict:
        # A caller that got its values from MLflow has a mapping stored as its
        # repr, since mlflow.log_params stringifies; this reads that back.
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise PromoteError(f"{where}: {raw!r} is not a mapping ({exc})") from None
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in value.items()
        ):
            raise PromoteError(f"{where}: {raw!r} is not a mapping of strings")
        return value
    if kind is str:
        return text
    try:
        return kind(text)
    except ValueError:
        raise PromoteError(f"{where}: {raw!r} is not a {kind.__name__}") from None


def cast_section(section: str, raw: dict[str, str]) -> dict[str, Any]:
    """The values of one section, typed against its dataclass.

    Keys the dataclass does not have are dropped — a caller reading an MLflow
    run also gets the collection and the parameters file it ran against, and
    neither is a knob. Keys the dataclass has and the caller does not are simply
    absent, and the file keeps its own value for them.
    """
    if section not in SECTION_TYPES:
        raise PromoteError(
            f"unknown section {section!r} — one of {', '.join(SECTION_TYPES)}"
        )
    types = _field_types(SECTION_TYPES[section])
    return {
        key: _cast(value, types[key], f"{section}.{key}")
        for key, value in raw.items()
        if key in types
    }


# --- editing the file -------------------------------------------------------

_KEY = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<rest>.*)$")


def _render(value: Any) -> str:
    """One scalar as yaml would write it, quoting only when it has to."""
    dumped = yaml.safe_dump(
        {"v": value}, default_flow_style=True, allow_unicode=True
    ).strip()
    return dumped.removeprefix("{v: ").removesuffix("}")


def _comment_start(line: str) -> int:
    """Column of the inline comment, or -1. A quoted `#` does not count."""
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "#" and (index == 0 or line[index - 1] in " \t"):
            return index
    return -1


def _split(line: str) -> tuple[str, str, int]:
    """A `key: value  # comment` line as (value, comment, comment column)."""
    cut = _comment_start(line)
    body = line if cut < 0 else line[:cut]
    return body.split(":", 1)[1].strip(), ("" if cut < 0 else line[cut:]), cut


def _set_value(line: str, indent: str, key: str, value: str) -> str:
    """The same line with a new value, the comment kept at its own column."""
    _, comment, column = _split(line)
    head = f"{indent}{key}: {value}"
    if not comment:
        return head
    return head + " " * max(column - len(head), 2) + comment


def _mapping_lines(value: dict[str, str], indent: int) -> list[str]:
    dumped = yaml.safe_dump(value, default_flow_style=False, allow_unicode=True)
    return [" " * indent + line for line in dumped.strip().splitlines()]


def _blocks(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """The file split at its top-level keys; the leading `None` block is the header.

    Blank lines and the comments before the next key stay with the block above,
    so a key appended to a section lands before them.
    """
    blocks: list[tuple[str | None, list[str]]] = []
    key: str | None = None
    body: list[str] = []
    for line in lines:
        match = _KEY.match(line)
        if match and not match["indent"]:
            blocks.append((key, body))
            key, body = match["key"], [line]
        else:
            body.append(line)
    blocks.append((key, body))
    return blocks


def _append(body: list[str], new_lines: list[str]) -> list[str]:
    """Put lines after the block's last line with content, before its blank tail."""
    last = max((i for i, line in enumerate(body) if line.strip()), default=len(body) - 1)
    return body[: last + 1] + new_lines + body[last + 1 :]


def _edit_section(
    body: list[str], section: str, wanted: dict[str, Any], changes: list[str]
) -> list[str]:
    """One section block with its values replaced, comments and layout kept."""
    out: list[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(body):
        line = body[index]
        match = _KEY.match(line)
        if match is None or len(match["indent"]) != 2 or match["key"] not in wanted:
            out.append(line)
            index += 1
            continue

        key = match["key"]
        value = wanted[key]
        seen.add(key)
        index += 1

        if isinstance(value, dict):
            nested: list[str] = []
            while index < len(body):
                inner = _KEY.match(body[index])
                if inner is None or len(inner["indent"]) <= 2:
                    break
                nested.append(body[index])
                index += 1
            replacement = _mapping_lines(value, 4)
            if nested != replacement:
                changes.append(f"{section}.{key}: {_render(value)}")
            out.append(line)
            out.extend(replacement)
            continue

        old = _split(line)[0]
        new = _render(value)
        if old != new:
            changes.append(f"{section}.{key}: {old} -> {new}")
        out.append(_set_value(line, "  ", key, new))

    missing = [key for key in wanted if key not in seen]
    if missing:
        for key in missing:
            changes.append(f"{section}.{key}: (absent) -> {_render(wanted[key])}")
        out = _append(out, [f"  {key}: {_render(wanted[key])}" for key in missing])
    return out


def rewrite(
    text: str, values: dict[str, dict[str, Any]], version: str | None = None
) -> tuple[str, list[str]]:
    """parameters.yaml with those values in it, and a line per change.

    A section the file does not have is appended whole; so is a key its section
    is missing. Everything else — the header, the `rerun:` notes, the blank
    lines — is untouched.
    """
    unknown = set(values) - set(SECTION_TYPES)
    if unknown:
        raise PromoteError(
            f"unknown section(s) {', '.join(sorted(unknown))} — "
            f"known: {', '.join(SECTION_TYPES)}"
        )
    blocks = _blocks(text.splitlines())
    changes: list[str] = []
    out: list[str] = []
    written: set[str] = set()

    for key, body in blocks:
        if key in values:
            written.add(key)
            out.extend(_edit_section(body, key, values[key], changes))
        elif key == "version" and version is not None:
            old = _split(body[0])[0]
            if old != version:
                changes.append(f"version: {old} -> {version}")
            out.append(_set_value(body[0], "", "version", _render(version)))
            out.extend(body[1:])
        else:
            out.extend(body)

    for section, wanted in values.items():
        if section in written or not wanted:
            continue
        changes.append(f"{section}: (absent section) -> {len(wanted)} values")
        out.extend(
            ["", f"{section}:"]
            + [f"  {key}: {_render(value)}" for key, value in wanted.items()]
        )

    return "\n".join(out) + "\n", changes


# --- checking the edit before it is written ---------------------------------


def _parse(text: str) -> Params:
    """The text back through the real loader, so a bad edit fails here.

    A temp file rather than the target: `load_params` caches per path, and this
    has to read this text and not whatever that path held before.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
        temp = Path(handle.name)
    try:
        return load_params(temp)
    except (ValueError, yaml.YAMLError) as exc:
        raise PromoteError(f"the rewritten file does not load: {exc}") from None
    finally:
        temp.unlink(missing_ok=True)


def _verify(
    before: str, after: str, values: dict[str, dict[str, Any]], version: str | None
) -> tuple[Params, Params]:
    """Every named field holds its new value, and nothing else moved.

    This is what makes the line editing safe to ship: an edit that landed on the
    wrong line raises here rather than being committed.
    """
    old, new = _parse(before), _parse(after)
    for section, wanted in values.items():
        for name, value in wanted.items():
            got = getattr(getattr(new, section), name)
            if got != value:
                raise PromoteError(
                    f"{section}.{name} was written as {got!r}, not {value!r}"
                )
    for section, cls in SECTION_TYPES.items():
        for field in fields(cls):
            if field.name in values.get(section, {}):
                continue
            was = getattr(getattr(old, section), field.name)
            now = getattr(getattr(new, section), field.name)
            if was != now:
                raise PromoteError(
                    f"{section}.{field.name} changed to {now!r} but was not promoted"
                )
    if version is not None and new.version != version:
        raise PromoteError(f"version was written as {new.version!r}, not {version!r}")
    return old, new


@dataclass(frozen=True)
class Rewrite:
    """What promoting a parameter set into a file would do, before it is done.

    Built and checked by `plan`; `write` is the only thing that touches disk, so
    a caller can print it and stop.
    """

    path: Path
    before: str
    after: str
    changes: list[str]
    was: Params
    now: Params

    @property
    def needs_reingest(self) -> bool:
        """The promoted values change what is stored in Qdrant, so stages 4-5
        have to run again before the scores behind them hold."""
        return ingest_fingerprint(self.was) != ingest_fingerprint(self.now)

    @property
    def fingerprints(self) -> tuple[str, str]:
        return ingest_fingerprint(self.was), ingest_fingerprint(self.now)

    def diff(self) -> str:
        return "".join(
            difflib.unified_diff(
                self.before.splitlines(keepends=True),
                self.after.splitlines(keepends=True),
                fromfile=str(self.path),
                tofile=f"{self.path} (promoted)",
            )
        )

    def write(self) -> None:
        self.path.write_text(self.after, encoding="utf-8")


def plan(
    values: dict[str, dict[str, Any]],
    version: str | None = None,
    path: str | Path | None = None,
) -> Rewrite:
    """Prepare the edit to parameters.yaml, without writing it.

    The target is `path`, else `$RAG_PARAMETERS_PATH`, else the copy inside the
    installed package — which in a venv is site-packages, a copy git never sees,
    so a caller that means to commit the result passes `path`.
    """
    target = params_path(path)
    if not target.is_file():
        raise PromoteError(f"{target}: no parameters.yaml to write into")
    before = target.read_text(encoding="utf-8")
    after, changes = rewrite(before, values, version)
    was, now = _verify(before, after, values, version)
    return Rewrite(
        path=target, before=before, after=after, changes=changes, was=was, now=now
    )


# --- from a registered baseline ---------------------------------------------

# Section -> the param prefix an evaluation run logs it under. `agent` is logged
# as `run.` there — the prefix says "what was in force for this run", and
# renaming it would orphan every registered baseline.
SECTION_PREFIX = {
    "parse_chunk": "parse_chunk.",
    "embed": "embed.",
    "agent": "run.",
}


def section_values(baseline) -> dict[str, dict[str, Any]]:
    """Every section of a `Baseline`, typed against its dataclass.

    All three, not only the query knobs: a parameter set is what was measured,
    and the chunking and embedding half says which index it was measured on.
    """
    values = {
        section: cast_section(section, baseline.section(prefix))
        for section, prefix in SECTION_PREFIX.items()
    }
    if not any(values.values()):
        raise PromoteError(
            f"{baseline.ref}: its run logged no parameters under "
            f"{', '.join(SECTION_PREFIX.values())} — "
            "it was not logged by the evaluation module"
        )
    return values


def run_promote(
    ref: str,
    out: str | Path | None = None,
    version: str | None = None,
    dry_run: bool = False,
) -> int:
    """Read a registered baseline and write its values into parameters.yaml.

    `version:` is stamped with the registry address the values came from, so the
    shipped file points back at the run that justified it. Writing the file is
    the whole effect — committing it is the user's step.
    """
    from rag_config.baseline import resolve_baseline

    baseline = resolve_baseline(ref)
    values = section_values(baseline)
    version = version if version is not None else f"{baseline.name}/{baseline.version}"
    edit = plan(values, version=version, path=out)

    print(baseline.summary)
    print(f"  target  {edit.path}")
    if "site-packages" in edit.path.parts:
        print(
            "  warning: that is the installed copy, not a checkout — "
            "point --out at the config repo to commit it",
            file=sys.stderr,
        )
    if not edit.changes:
        print("  nothing to change: the file already holds this baseline")
        return 0
    for line in edit.changes:
        print(f"  {line}")
    if edit.needs_reingest:
        was, now = edit.fingerprints
        print(
            f"  ingest fingerprint {was} -> {now}: "
            "stages 4-5 must be re-run before these scores hold"
        )

    if dry_run:
        print(edit.diff(), end="")
        print("  dry run: nothing written")
        return 0

    edit.write()
    print(f"  written — commit {edit.path.name} to version it")
    return 0
