from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from ...infrastructure.experiments.loader import load_experiment_package
from ...infrastructure.experiments.payload import apply_serialized_experiment
from ...infrastructure.io.console import configure_logging, logger
from ...infrastructure.io.files import write_json
from .trial import execute_search_trial


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a single taac-search trial worker")
    parser.add_argument("--experiment", required=True, help="Experiment package path or module path")
    parser.add_argument("--config-path", required=True, help="Serialized trial experiment JSON")
    parser.add_argument("--result-path", required=True, help="Worker result JSON path")
    parser.add_argument("--device", help="Device passed to the worker, e.g. cuda:0")
    return parser.parse_args(argv)


def worker_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(log_path=Path(args.result_path).resolve().parent / "worker.log")
    try:
        with Path(args.config_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        experiment = apply_serialized_experiment(
            load_experiment_package(args.experiment),
            payload,
        )
        if args.device:
            experiment.train.device = args.device
        logger.info(
            "search worker start: experiment={} output_dir={} device={}",
            experiment.name,
            experiment.train.output_dir,
            experiment.train.device,
        )
        result = execute_search_trial(experiment)
        write_json(args.result_path, result)
        logger.info("search worker complete: status={}", result.get("status"))
        return 0
    except Exception as exc:
        logger.exception("search worker failed: {}", exc)
        write_json(
            args.result_path,
            {
                "status": "fail",
                "trial_error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


__all__ = ["parse_args", "worker_main"]
