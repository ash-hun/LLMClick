"""CLI: `llmclick run CONFIG`, `llmclick validate CONFIG`, `llmclick recipes`, `llmclick serve`."""

import sys
import json
import argparse

from tqdm.contrib.logging import logging_redirect_tqdm

from core.settings import prepare_process
from core.progress import BarProgress
from core.registry import catalogue
from core import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmclick", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in [("run", "Run the stages a YAML config asks for"),
                       ("validate", "Check a config and print its experiment key and stage plan")]:
        commands.add_parser(name, help=text).add_argument("config")
    commands.add_parser("recipes", help="List the recipes, their stages and the keys their registries accept")
    serve = commands.add_parser("serve", help="Run the API server (the same as `uvicorn core.api.app:app`)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    prepare_process()

    if args.command == "serve":
        import uvicorn
        uvicorn.run("core.api.app:app", host=args.host, port=args.port, log_config=None)  # uvicorn logs through the root handler
        return 0
    if args.command == "recipes":
        print(json.dumps(catalogue(), indent=2))
        return 0
    if args.command == "validate":
        built = pipeline.load(args.config)
        print(json.dumps({"valid": True, "recipe": built.kind, "experiment": built.experiment.key,
                          "directory": str(built.experiment.root), "stages": built.plan()}, indent=2))
        return 0
    with logging_redirect_tqdm():
        result = pipeline.load(args.config, BarProgress()).run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
