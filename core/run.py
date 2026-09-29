"""CLI: `llmclick run CONFIG [--stages data,mix]`, `llmclick validate CONFIG`, `llmclick registries`."""

import argparse
import json
import logging
import sys

from core.config.experiment import Experiment, load_config
from core.config.schema import STAGES
from core.registry import catalogue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmclick", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the pipeline described by a YAML config")
    run.add_argument("config")
    run.add_argument("--stages", help=f"Comma-separated subset of {','.join(STAGES)}; default all")
    validate = commands.add_parser("validate", help="Validate a config and print its experiment key")
    validate.add_argument("config")
    commands.add_parser("registries", help="List registered builders, converters, backbones, teachers and benchmarks")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stderr)

    if args.command == "registries":
        print(json.dumps(catalogue(), indent=2))
        return 0
    config = load_config(args.config)
    experiment = Experiment(config)
    if args.command == "validate":
        print(json.dumps({"valid": True, "experiment": experiment.key, "directory": str(experiment.root)}, indent=2))
        return 0
    from core import pipeline
    stages = args.stages.split(",") if args.stages else None
    print(json.dumps(pipeline.run(config, stages), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
