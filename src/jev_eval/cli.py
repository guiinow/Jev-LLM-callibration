import argparse
from collections import defaultdict
from pathlib import Path

from jev_eval.config import load_config
from jev_eval.data.datasets import build_adapter
from jev_eval.data.io import read_jsonl
from jev_eval.env import load_env
from jev_eval.eval.figures import _write_csv, make_report_outputs
from jev_eval.eval.metrics import save_metrics
from jev_eval.pipeline.ablation import ablation, ablation_tables, load_baseline, with_baseline_jev
from jev_eval.pipeline.analyze import analyze, with_chosen_probability, with_conditions
from jev_eval.pipeline.run_inference import run_inference


def _records(config) -> tuple[list[dict], list[dict] | None]:
    """This run's records (pre-conditions runs mapped to conditions); with paths.baseline (E1
    ablation), plus the baseline's Jev rows."""
    records = with_conditions(read_jsonl(config.paths["raw_predictions"]))
    if "baseline" not in config.paths:
        return records, None
    baseline = with_chosen_probability(load_baseline(config))
    return with_baseline_jev(records, baseline), baseline


def _labels(config, name: str) -> set[str]:
    return set(build_adapter(next(s for s in config.datasets if s.name == name)).options("en"))


def _by_dataset(config, records) -> dict[str, list[dict]]:
    """Metrics are never pooled across datasets: split the records and report each on its own."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["dataset"]].append(row)
    return {spec.name: grouped[spec.name] for spec in config.datasets if grouped[spec.name]}


def _analyze(config, records, baseline=None, output_dir: Path | None = None) -> None:
    base_by_dataset = _by_dataset(config, baseline) if baseline is not None else {}
    for name, rows in _by_dataset(config, records).items():
        out = (output_dir or config.output_dir) / name
        analyze(rows, config.ece_bins, out / "metrics.json")
        print(out / "metrics.json")
        if baseline is not None:
            save_metrics(out / "ablation.json", ablation(rows, base_by_dataset.get(name, []), config.ece_bins,
                                                         labels=_labels(config, name)))
            print(out / "ablation.json")


def _plot(config, records, baseline=None, output_dir: Path | None = None) -> None:
    specs = {spec.name: spec for spec in config.datasets}
    base_by_dataset = _by_dataset(config, baseline) if baseline is not None else {}
    for name, rows in _by_dataset(config, records).items():
        out = (output_dir or config.output_dir) / name
        examples = build_adapter(specs[name]).load_examples(config.sample_size)
        written = make_report_outputs(out, with_chosen_probability(rows), config.ece_bins,
                                      examples=examples, seed=config.seed)
        if baseline is not None:
            header, table, delta_header, deltas = ablation_tables(
                ablation(rows, base_by_dataset.get(name, []), config.ece_bins, labels=_labels(config, name)))
            _write_csv(out / "tables" / "ablation.csv", header, table)
            _write_csv(out / "tables" / "ablation_deltas.csv", delta_header, deltas)
            written += [out / "tables" / "ablation.csv", out / "tables" / "ablation_deltas.csv"]
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
        _analyze(config, records)
        _plot(config, records)
    elif args.command == "run":
        run_inference(config, mock=args.mock, fresh=args.fresh)
    elif args.command == "analyze":
        records, baseline = _records(config)
        _analyze(config, records, baseline, args.output_dir)
    elif args.command == "plot":
        records, baseline = _records(config)
        _plot(config, records, baseline, args.output_dir)
    elif args.command == "probe":
        from jev_eval.pipeline.probe import run_probe
        run_probe(config, skip_jev=args.skip_jev)


if __name__ == "__main__":
    main()
