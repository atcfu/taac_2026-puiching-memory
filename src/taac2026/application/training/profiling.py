from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from ...domain.experiment import ExperimentSpec
from ...domain.metrics import percentile, safe_mean


def select_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_loader_outputs(model, loader, device, loss_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    group_list: list[np.ndarray] = []
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, batch.labels).detach().cpu().item()))
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(batch.labels.detach().cpu().numpy())
            group_list.append(batch.user_indices.detach().cpu().numpy())
    if not logits_list:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, 0.0
    return (
        np.concatenate(logits_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(group_list, axis=0),
        safe_mean(losses),
    )


def measure_latency(model, loader, device, warmup_steps: int, measure_steps: int) -> dict[str, float]:
    durations: list[float] = []
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if step < warmup_steps:
            with torch.no_grad():
                _ = model(batch)
            continue
        if measure_steps > 0 and len(durations) >= measure_steps:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        durations.append((elapsed * 1000.0) / max(batch.batch_size, 1))
    return {
        "mean_latency_ms_per_sample": safe_mean(durations),
        "p95_latency_ms_per_sample": percentile(durations, 95.0),
    }


def _profiler_activities_for(device: torch.device) -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _loader_num_batches(loader) -> int:
    try:
        return int(len(loader))
    except TypeError:
        return 0


def _loader_num_samples(loader, max_batches: int | None = None) -> int:
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        sample_count += int(batch.batch_size)
    return sample_count


def _count_latency_probe_batches(total_batches: int, warmup_steps: int, measure_steps: int) -> int:
    if total_batches <= 0:
        return 0
    warmup_batches = min(total_batches, max(warmup_steps, 0))
    remaining_batches = max(total_batches - warmup_batches, 0)
    measured_batches = remaining_batches if measure_steps <= 0 else min(remaining_batches, measure_steps)
    return warmup_batches + measured_batches


def collect_inference_profile(
    experiment: ExperimentSpec,
    val_loader,
    latency: dict[str, float],
) -> dict[str, float | int]:
    val_sample_count = _loader_num_samples(val_loader)
    mean_latency_ms_per_sample = float(latency.get("mean_latency_ms_per_sample", 0.0))
    p95_latency_ms_per_sample = float(latency.get("p95_latency_ms_per_sample", 0.0))
    estimated_end_to_end_inference_seconds = (mean_latency_ms_per_sample * float(val_sample_count)) / 1000.0
    estimated_end_to_end_inference_seconds_p95 = (p95_latency_ms_per_sample * float(val_sample_count)) / 1000.0
    return {
        "val_sample_count": int(val_sample_count),
        "latency_warmup_steps": int(experiment.train.latency_warmup_steps),
        "latency_measure_steps": int(experiment.train.latency_measure_steps),
        "estimated_end_to_end_inference_seconds": estimated_end_to_end_inference_seconds,
        "estimated_end_to_end_inference_minutes": estimated_end_to_end_inference_seconds / 60.0,
        "estimated_end_to_end_inference_seconds_p95": estimated_end_to_end_inference_seconds_p95,
        "estimated_end_to_end_inference_minutes_p95": estimated_end_to_end_inference_seconds_p95 / 60.0,
    }


def collect_model_profile(model, loader, device) -> dict[str, float | int | str]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    profile_batch = next(iter(loader), None)
    if profile_batch is None:
        return {
            "profile_scope": "single_eval_forward",
            "profile_batch_size": 0,
            "total_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
            "flops_per_batch": 0.0,
            "tflops_per_batch": 0.0,
            "flops_per_sample": 0.0,
        }

    profile_batch = profile_batch.to(device)
    batch_size = max(profile_batch.batch_size, 1)
    was_training = model.training
    model.eval()

    with profile(activities=_profiler_activities_for(device), with_flops=True, record_shapes=False, acc_events=True) as profiler:
        with torch.no_grad():
            _ = model(profile_batch)

    if was_training:
        model.train()

    total_flops = float(profiler.key_averages().total_average().flops or 0.0)
    return {
        "profile_scope": "single_eval_forward",
        "profile_batch_size": int(profile_batch.batch_size),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "parameter_size_mb": parameter_bytes / (1024.0 * 1024.0),
        "flops_per_batch": total_flops,
        "tflops_per_batch": total_flops / 1.0e12,
        "flops_per_sample": total_flops / float(batch_size),
    }


def collect_compute_profile(
    experiment: ExperimentSpec,
    model,
    loss_fn,
    train_loader,
    val_loader,
    data_stats,
    device,
    model_profile: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    train_batches_per_epoch = _loader_num_batches(train_loader)
    val_batches_per_epoch = _loader_num_batches(val_loader)
    latency_probe_batches = _count_latency_probe_batches(
        total_batches=val_batches_per_epoch,
        warmup_steps=experiment.train.latency_warmup_steps,
        measure_steps=experiment.train.latency_measure_steps,
    )
    latency_probe_samples = _loader_num_samples(val_loader, max_batches=latency_probe_batches)

    train_profile_batch = next(iter(train_loader), None)
    if train_profile_batch is None:
        train_step_flops = 0.0
        train_profile_batch_size = 0
        train_step_flops_per_sample = 0.0
    else:
        train_profile_batch = train_profile_batch.to(device)
        train_profile_batch_size = int(train_profile_batch.batch_size)
        profile_model = experiment.build_model_component(experiment.data, experiment.model, data_stats.dense_dim)
        profile_model = profile_model.to(device)
        profile_optimizer = experiment.build_optimizer_component(profile_model, experiment.train)
        profile_model.train()

        with profile(
            activities=_profiler_activities_for(device),
            with_flops=True,
            record_shapes=False,
            acc_events=True,
        ) as profiler:
            profile_optimizer.zero_grad(set_to_none=True)
            logits = profile_model(train_profile_batch)
            loss = loss_fn(logits, train_profile_batch.labels)
            loss.backward()
            if experiment.train.grad_clip_norm and experiment.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(profile_model.parameters(), experiment.train.grad_clip_norm)
            profile_optimizer.step()

        train_step_flops = float(profiler.key_averages().total_average().flops or 0.0)
        train_step_flops_per_sample = train_step_flops / float(max(train_profile_batch_size, 1))

        del profile_optimizer
        del profile_model
        del train_profile_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eval_flops_per_sample = float(model_profile.get("flops_per_sample", 0.0))
    train_samples_per_epoch = int(data_stats.train_size)
    val_samples_per_epoch = int(data_stats.val_size)

    estimated_train_flops_total = train_step_flops_per_sample * float(train_samples_per_epoch) * float(experiment.train.epochs)
    estimated_eval_flops_total = eval_flops_per_sample * float(val_samples_per_epoch) * float(experiment.train.epochs)
    estimated_latency_probe_flops_total = eval_flops_per_sample * float(latency_probe_samples)
    estimated_end_to_end_flops_total = (
        estimated_train_flops_total
        + estimated_eval_flops_total
        + estimated_latency_probe_flops_total
    )

    return {
        "estimation_method": "profiled_single_step_scaled_by_observed_sample_counts",
        "epochs": int(experiment.train.epochs),
        "train_batches_per_epoch": train_batches_per_epoch,
        "val_batches_per_epoch": val_batches_per_epoch,
        "train_samples_per_epoch": train_samples_per_epoch,
        "val_samples_per_epoch": val_samples_per_epoch,
        "latency_probe_batches": latency_probe_batches,
        "latency_probe_samples": latency_probe_samples,
        "train_profile_scope": "single_train_step_forward_backward_optimizer",
        "train_profile_batch_size": train_profile_batch_size,
        "train_step_flops": train_step_flops,
        "train_step_tflops": train_step_flops / 1.0e12,
        "train_step_flops_per_sample": train_step_flops_per_sample,
        "estimated_train_flops_total": estimated_train_flops_total,
        "estimated_train_tflops_total": estimated_train_flops_total / 1.0e12,
        "estimated_eval_flops_total": estimated_eval_flops_total,
        "estimated_eval_tflops_total": estimated_eval_flops_total / 1.0e12,
        "estimated_latency_probe_flops_total": estimated_latency_probe_flops_total,
        "estimated_latency_probe_tflops_total": estimated_latency_probe_flops_total / 1.0e12,
        "estimated_end_to_end_flops_total": estimated_end_to_end_flops_total,
        "estimated_end_to_end_tflops_total": estimated_end_to_end_flops_total / 1.0e12,
    }


__all__ = [
    "collect_compute_profile",
    "collect_inference_profile",
    "collect_loader_outputs",
    "collect_model_profile",
    "measure_latency",
    "select_device",
    "set_random_seed",
]
