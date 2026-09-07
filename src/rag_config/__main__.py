"""Argparse only — the logic lives in the library modules.

    python -m rag_config promote --mlflow-model rag-retrieval-params@champion

One command so far. Run from this repo, `--out` defaults to the checked-out
parameters.yaml, which is the copy git will see.
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_config", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    promote = sub.add_parser(
        "promote",
        help="write a registered baseline's parameters into parameters.yaml",
    )
    promote.add_argument(
        "--mlflow-model",
        required=True,
        metavar="NAME@ALIAS|NAME/VERSION",
        help="the registered baseline to promote, as MLflow addresses one: name@alias or name/version",
    )
    promote.add_argument(
        "--out",
        help="the parameters.yaml to write (default RAG_PARAMETERS_PATH, then the packaged copy)",
    )
    promote.add_argument(
        "--set-version",
        help="what to stamp in `version:` (default: the registry address, e.g. rag-retrieval-params/7)",
    )
    promote.add_argument(
        "--dry-run", action="store_true", help="print the diff, write nothing"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "promote":
        from rag_config.baseline import BaselineError
        from rag_config.promote import PromoteError, run_promote

        try:
            return run_promote(
                args.mlflow_model,
                out=args.out,
                version=args.set_version,
                dry_run=args.dry_run,
            )
        except (BaselineError, PromoteError) as exc:
            print(f"promote: {exc}", file=sys.stderr)
            return 2

    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    # MLFLOW_TRACKING_URI from this repo's .env; absent without the promote
    # extra, and then the env has to carry it.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    raise SystemExit(main())
