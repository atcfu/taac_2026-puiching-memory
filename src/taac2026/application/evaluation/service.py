from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...domain.metrics import compute_classification_metrics
from ...infrastructure.experiments.loader import load_experiment_package
from ...infrastructure.io.console import logger
from ...infrastructure.io.files import write_json
from ..training.profiling import collect_loader_outputs, measure_latency, select_device


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def record_bucket(record: dict[str, Any]) -> int:
        if bool(record.get("latency_budget_met")) and float(record.get("latency_budget_ms_per_sample", 0.0)) > 0.0:
            return 0
        if bool(record.get("latency_budget_met")):
            return 1
        return 2

    return sorted(
        records,
        key=lambda record: (
            record_bucket(record),
            -float(record.get("auc", 0.0)),
            -float(record.get("pr_auc", 0.0)),
            float(record.get("mean_latency_ms_per_sample", float("inf"))),
            str(record.get("experiment_id", "")),
        ),
    )


def evaluate_checkpoint(
    experiment_path: str | Path,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    experiment = load_experiment_package(experiment_path)
    device = select_device(experiment.train.device)
    logger.info("evaluate start: experiment={} device={}", experiment_path, device)
    train_loader, val_loader, data_stats = experiment.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    del train_loader
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model = model.to(device)
    loss_fn, _ = experiment.build_loss_stack(
        experiment.data,
        experiment.model,
        experiment.train,
        data_stats,
        device,
    )

    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else Path(experiment.train.output_dir) / "best.pt"
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

    payload = torch.load(resolved_checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"incompatible checkpoint: {exc}") from exc

    logits, labels, groups, loss = collect_loader_outputs(model, val_loader, device, loss_fn)
    metrics = compute_classification_metrics(labels, logits, groups)
    latency = measure_latency(
        model,
        val_loader,
        device,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
    )
    report = {
        "experiment": experiment.name,
        "experiment_path": str(experiment_path),
        "model_name": experiment.model.name,
        "device": str(device),
        "checkpoint_path": str(resolved_checkpoint),
        "loss": loss,
        "auc": float(metrics.get("auc", 0.0)),
        "pr_auc": float(metrics.get("pr_auc", 0.0)),
        "metrics": metrics,
        **latency,
    }
    if output_path is not None:
        write_json(output_path, report)
    logger.info(
        "evaluate complete: experiment={} auc={:.6f} pr_auc={:.6f} latency_ms={:.4f}",
        experiment_path,
        float(metrics.get("auc", 0.0)),
        float(metrics.get("pr_auc", 0.0)),
        float(report["mean_latency_ms_per_sample"]),
    )
    return report


__all__ = ["_sort_records", "evaluate_checkpoint"]
