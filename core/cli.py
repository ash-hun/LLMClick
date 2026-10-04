"""CLI: `llmclick run CONFIG`, `llmclick validate CONFIG`, `llmclick recipes`."""

import argparse
import json
import logging
import sys

from dotenv import load_dotenv
from tqdm.contrib.logging import logging_redirect_tqdm

from core.config.experiment import LOG_FORMAT
from core.progress import BarProgress
from core.registry import catalogue
from core import pipeline

ENV_FILE = "environment/.env"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmclick", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in [("run", "Run the stages a YAML config asks for"),
                       ("validate", "Check a config and print its experiment key and stage plan")]:
        commands.add_parser(name, help=text).add_argument("config")
    commands.add_parser("recipes", help="List the recipes, their stages and the keys their registries accept")
    args = parser.parse_args(argv)
    load_dotenv(ENV_FILE)  # child processes (jeff, Hugging Face downloads) read HF_TOKEN from the environment
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)

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
