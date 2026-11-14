import argparse
from pathlib import Path

from jev_eval.config import load_config
from jev_eval.data.io import read_jsonl
from jev_eval.env import load_env
from jev_eval.eval.figures import _write_csv, make_report_outputs
from jev_eval.eval.metrics import save_metrics
from jev_eval.pipeline.ablation import ablation, ablation_tables, load_baseline, with_baseline_jev
from jev_eval.pipeline.analyze import analyze, with_chosen_probability
from jev_eval.pipeline.run_inference import run_inference


def _records(config) -> tuple[list[dict], list[dict] | None]:
    """This run's records; with paths.baseline (E1 ablation), plus the baseline's Jev rows."""
    records = read_jsonl(config.paths["raw_predictions"])
    if "baseline" not in config.paths:
        return records, None
    baseline = with_chosen_probability(load_baseline(config))
    return with_baseline_jev(records, baseline), baseline


def _plot(config, records, output_dir: Path | None = None, baseline=None) -> None:
    examples = read_jsonl(config.paths["sample"])
    output_dir = output_dir or config.paths["raw_predictions"].parent
    written = make_report_outputs(output_dir, with_chosen_probability(records), config.ece_bins,
                                  examples=examples, seed=config.seed)
    if baseline is not None:
        header, rows, delta_header, deltas = ablation_tables(ablation(records, baseline, config.ece_bins))
        _write_csv(output_dir / "tables" / "ablation.csv", header, rows)
        _write_csv(output_dir / "tables" / "ablation_deltas.csv", delta_header, deltas)
        written += [output_dir / "tables" / "ablation.csv", output_dir / "tables" / "ablation_deltas.csv"]
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
    for name in ("analyze", "plot"):
        subparsers.add_parser(name).add_argument(
            "--output-dir", type=Path,
            help="write metrics/figures/tables here instead of next to raw_predictions.jsonl "
                 "(re-analysis without overwriting a run's saved outputs)")
    subparsers.add_parser("smoke")
    probe_parser = subparsers.add_parser("probe", help="pre-flight: prompt token count (Ollama) and one Jev call")
    probe_parser.add_argument("--skip-jev", action="store_true")
    args = parser.parse_args()
    load_env()
    config = load_config(args.config)
    if args.command == "smoke":
        records = run_inference(config, mock=True, fresh=True)
        analyze(records, config.ece_bins, config.paths["metrics"])
        _plot(config, records)
    elif args.command == "run":
        run_inference(config, mock=args.mock, fresh=args.fresh)
    elif args.command == "analyze":
        records, baseline = _records(config)
        metrics_path = args.output_dir / "metrics.json" if args.output_dir else config.paths["metrics"]
        analyze(records, config.ece_bins, metrics_path)
        print(metrics_path)
        if baseline is not None:
            ablation_path = metrics_path.with_name("ablation.json")
            save_metrics(ablation_path, ablation(records, baseline, config.ece_bins))
            print(ablation_path)
    elif args.command == "probe":
        from jev_eval.pipeline.probe import run_probe
        run_probe(config, skip_jev=args.skip_jev)
    elif args.command == "plot":
        records, baseline = _records(config)
        _plot(config, records, args.output_dir, baseline)


if __name__ == "__main__":
    main()
