from __future__ import annotations

from pathlib import Path

from ...infrastructure.io.files import write_json


def render_training_curves_plot(
    output_path: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    best_epoch: int,
) -> None:
    if not train_losses:
        return

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render training curves; run uv sync --locked") from exc

    epochs = list(range(1, len(train_losses) + 1))
    figure, (loss_axis, auc_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    loss_axis.plot(epochs, train_losses, marker="o", linewidth=2.0, label="train_loss")
    loss_axis.plot(epochs, val_losses, marker="s", linewidth=2.0, label="val_loss")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    auc_axis.plot(epochs, val_aucs, marker="o", linewidth=2.0, color="#2ca02c", label="val_auc")
    if 0 < best_epoch <= len(val_aucs):
        best_auc = val_aucs[best_epoch - 1]
        auc_axis.axvline(best_epoch, color="#7f7f7f", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
        auc_axis.scatter([best_epoch], [best_auc], color="#d62728", s=60, zorder=3, label=f"best_auc={best_auc:.4f}")
    auc_axis.set_xlabel("Epoch")
    auc_axis.set_ylabel("AUC")
    auc_axis.grid(True, alpha=0.3)
    auc_axis.legend()

    figure.suptitle("Training Curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_training_curve_artifacts(
    output_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_aucs: list[float],
    best_epoch: int,
) -> None:
    curves = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_auc": val_aucs,
        "best_epoch": best_epoch,
    }
    write_json(output_dir / "training_curves.json", curves)
    render_training_curves_plot(
        output_path=output_dir / "training_curves.png",
        train_losses=train_losses,
        val_losses=val_losses,
        val_aucs=val_aucs,
        best_epoch=best_epoch,
    )


__all__ = ["render_training_curves_plot", "write_training_curve_artifacts"]
