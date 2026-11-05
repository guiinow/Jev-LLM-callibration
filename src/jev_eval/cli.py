import argparse
from pathlib import Path

from jev_eval.config import load_config
from jev_eval.data.io import read_jsonl
from jev_eval.eval.plots import reliability_diagram
from jev_eval.pipeline.analyze import analyze
from jev_eval.pipeline.run_inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev-eval")
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mock", action="store_true")
    subparsers.add_parser("analyze")
    subparsers.add_parser("plot")
    subparsers.add_parser("smoke")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "smoke":
        records = run_inference(config, mock=True)
        analyze(records, config.ece_bins, config.paths["metrics"])
        reliability_diagram(config.paths["reliability_plot"], records, config.ece_bins)
    elif args.command == "run":
        run_inference(config, mock=args.mock)
    elif args.command == "analyze":
        records = read_jsonl(config.paths["raw_predictions"])
        analyze(records, config.ece_bins, config.paths["metrics"])
    elif args.command == "plot":
        records = read_jsonl(config.paths["raw_predictions"])
        reliability_diagram(config.paths["reliability_plot"], records, config.ece_bins)


if __name__ == "__main__":
    main()
