import argparse
from collections import defaultdict

from jev_eval.config import load_config
from jev_eval.data.datasets import build_adapter
from jev_eval.data.io import read_jsonl
from jev_eval.env import load_env
from jev_eval.eval.figures import make_report_outputs
from jev_eval.pipeline.analyze import analyze
from jev_eval.pipeline.run_inference import run_inference


def _by_dataset(config, records) -> dict[str, list[dict]]:
    """Metrics are never pooled across datasets: split the records and report each on its own."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["dataset"]].append(row)
    return {spec.name: grouped[spec.name] for spec in config.datasets if grouped[spec.name]}


def _analyze(config, records) -> None:
    for name, rows in _by_dataset(config, records).items():
        analyze(rows, config.ece_bins, config.output_dir / name / "metrics.json")


def _plot(config, records) -> None:
    specs = {spec.name: spec for spec in config.datasets}
    for name, rows in _by_dataset(config, records).items():
        examples = build_adapter(specs[name]).load_examples(config.sample_size)
        written = make_report_outputs(config.output_dir / name, rows, config.ece_bins,
                                      examples=examples, seed=config.seed)
        for path in written:
            print(path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev-eval")
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mock", action="store_true")
    run_parser.add_argument("--fresh", action="store_true",
                            help="discard existing raw predictions instead of resuming")
    subparsers.add_parser("analyze")
    subparsers.add_parser("plot")
    subparsers.add_parser("smoke")
    args = parser.parse_args()
    load_env()
    config = load_config(args.config)
    if args.command == "smoke":
        records = run_inference(config, mock=True, fresh=True)
        _analyze(config, records)
        _plot(config, records)
    elif args.command == "run":
        run_inference(config, mock=args.mock, fresh=args.fresh)
    elif args.command == "analyze":
        _analyze(config, read_jsonl(config.paths["raw_predictions"]))
    elif args.command == "plot":
        _plot(config, read_jsonl(config.paths["raw_predictions"]))


if __name__ == "__main__":
    main()
