from __future__ import annotations

from typing import Any

import torch

from ...domain.experiment import ExperimentSpec
from ...domain.metrics import compute_classification_metrics, safe_mean
from ...infrastructure.io.console import create_progress_bar, logger
from ...infrastructure.io.files import ensure_dir, write_json
from .artifacts import write_training_curve_artifacts
from .profiling import (
    collect_compute_profile,
    collect_inference_profile,
    collect_loader_outputs,
    collect_model_profile,
    measure_latency,
    select_device,
    set_random_seed,
)


def run_training(
    experiment: ExperimentSpec,
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(experiment.train.output_dir)
    set_random_seed(experiment.train.seed)
    device = select_device(experiment.train.device)
    logger.info(
        "training start: experiment={} output_dir={} device={} epochs={} batch_size={}",
        experiment.name,
        output_dir,
        device,
        experiment.train.epochs,
        experiment.train.batch_size,
    )

    train_loader, val_loader, data_stats = experiment.build_data_pipeline(
        experiment.data,
        experiment.model,
        experiment.train,
    )
    model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
    model = model.to(device)
    loss_fn, auxiliary_loss = experiment.build_loss_stack(
        experiment.data,
        experiment.model,
        experiment.train,
        data_stats,
        device,
    )
    optimizer = experiment.build_optimizer_component(model, experiment.train)
    model_profile = collect_model_profile(model, val_loader, device)
    compute_profile = collect_compute_profile(
        experiment=experiment,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        data_stats=data_stats,
        device=device,
        model_profile=model_profile,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aucs: list[float] = []
    best_auc = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    progress_bar = None
    if show_progress:
        progress_bar = create_progress_bar(
            total=experiment.train.epochs,
            description=f"taac-train[{experiment.name}]",
        )

    try:
        for epoch in range(1, experiment.train.epochs + 1):
            model.train()
            batch_losses: list[float] = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = loss_fn(logits, batch.labels)
                if getattr(auxiliary_loss, "enabled", False) and getattr(auxiliary_loss, "requires_aux", False):
                    raise RuntimeError("Auxiliary losses requiring extra tensors are not implemented")
                loss.backward()
                if experiment.train.grad_clip_norm and experiment.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.train.grad_clip_norm)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = safe_mean(batch_losses)
            val_logits, val_labels, val_groups, val_loss = collect_loader_outputs(model, val_loader, device, loss_fn)
            val_metrics = compute_classification_metrics(val_labels, val_logits, val_groups)
            val_auc = float(val_metrics["auc"])

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)

            improved = val_auc >= best_auc
            if improved:
                best_auc = val_auc
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": val_metrics,
                    },
                    output_dir / "best.pt",
                )

            write_training_curve_artifacts(
                output_dir=output_dir,
                train_losses=train_losses,
                val_losses=val_losses,
                val_aucs=val_aucs,
                best_epoch=best_epoch,
            )

            epoch_status = {
                "epoch": f"{epoch}/{experiment.train.epochs}",
                "train_loss": f"{train_loss:.4f}",
                "val_auc": f"{val_auc:.4f}",
                "best": f"{best_auc:.4f}",
            }
            if progress_bar is not None:
                progress_bar.update()
                progress_bar.set_postfix(epoch_status, refresh=False)
            else:
                logger.info(
                    "epoch {}/{} train_loss={:.4f} val_loss={:.4f} val_auc={:.4f} best_auc={:.4f} improved={}",
                    epoch,
                    experiment.train.epochs,
                    train_loss,
                    val_loss,
                    val_auc,
                    best_auc,
                    improved,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    latency = measure_latency(
        model,
        val_loader,
        device,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
    )
    inference_profile = collect_inference_profile(experiment, val_loader, latency)

    summary = {
        "model_name": experiment.model.name,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "metrics": best_metrics,
        "model_profile": model_profile,
        "compute_profile": compute_profile,
        "inference_profile": inference_profile,
        **latency,
    }

    write_json(output_dir / "summary.json", summary)
    logger.info(
        "training complete: experiment={} best_epoch={} best_val_auc={:.6f} latency_ms={:.4f}",
        experiment.name,
        best_epoch,
        best_auc,
        float(summary["mean_latency_ms_per_sample"]),
    )
    return summary


__all__ = ["run_training"]
