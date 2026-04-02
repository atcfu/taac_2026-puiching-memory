from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..utils import ensure_dir
from .dataset_analysis import build_row_feature_frame


def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


_configure_matplotlib()


def _normalize_formats(formats: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not formats:
        return ("png", "svg")

    normalized: list[str] = []
    for value in formats:
        format_name = value.strip().lower()
        if format_name not in {"png", "svg"}:
            raise ValueError(f"不支持的导出格式: {value}。当前仅支持 png/svg。")
        if format_name not in normalized:
            normalized.append(format_name)
    return tuple(normalized) or ("png", "svg")


def _save_figure(figure: plt.Figure, base_path: str | Path, formats: list[str] | tuple[str, ...] | None = None) -> list[Path]:
    output_base = Path(base_path)
    if output_base.suffix:
        output_base = output_base.with_suffix("")
    ensure_dir(output_base.parent)

    written_paths: list[Path] = []
    for format_name in _normalize_formats(formats):
        output_path = output_base.with_suffix(f".{format_name}")
        figure.savefig(output_path, format=format_name, dpi=180, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths


def _safe_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_safe_json(inner) for inner in value]
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(_safe_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    derived = frame.copy()
    timestamp_min = float(derived["timestamp"].min())
    timestamp_max = float(derived["timestamp"].max())
    timestamp_span = max(timestamp_max - timestamp_min, 1.0)

    derived["timestamp_rank"] = (derived["timestamp"] - timestamp_min) / timestamp_span
    derived["log_total_event_count"] = np.log1p(derived["total_event_count"])
    derived["log_selected_event_count"] = np.log1p(derived["selected_event_count"])
    derived["log_active_span_hours"] = np.log1p(derived["active_span_hours"])
    derived["log_behavior_density"] = np.log1p(derived["behavior_density"])
    derived["log_user_frequency"] = np.log1p(derived["user_frequency"])
    derived["log_item_frequency"] = np.log1p(derived["item_frequency"])
    derived["hour_sin"] = np.sin(2.0 * np.pi * derived["hour_of_day"] / 24.0)
    derived["hour_cos"] = np.cos(2.0 * np.pi * derived["hour_of_day"] / 24.0)

    feature_columns = [
        "log_total_event_count",
        "log_selected_event_count",
        "log_active_span_hours",
        "log_behavior_density",
        "non_empty_group_count",
        "user_feature_count",
        "item_feature_count",
        "log_user_frequency",
        "log_item_frequency",
        "action_seq_share",
        "content_seq_share",
        "item_seq_share",
        "truncation_flag",
        "timestamp_rank",
        "hour_sin",
        "hour_cos",
    ]
    return derived, feature_columns


def _candidate_k_values(row_count: int, k_min: int, k_max: int) -> list[int]:
    upper_bound = min(int(k_max), max(row_count - 1, 2), 12)
    lower_bound = min(max(int(k_min), 2), upper_bound)
    return list(range(lower_bound, upper_bound + 1))


def _evaluate_candidate_k(X: np.ndarray, candidate_k: list[int], random_state: int) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for k in candidate_k:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        cluster_sizes = np.bincount(labels, minlength=k)
        records.append(
            {
                "k": float(k),
                "silhouette": float(silhouette_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "inertia": float(model.inertia_),
                "min_cluster_size": float(cluster_sizes.min()),
                "max_cluster_size": float(cluster_sizes.max()),
                "min_cluster_share": float(cluster_sizes.min() / max(X.shape[0], 1)),
                "max_cluster_share": float(cluster_sizes.max() / max(X.shape[0], 1)),
                "cluster_sizes": [float(value) for value in sorted(cluster_sizes.tolist())],
            }
        )
    return records


def _select_best_k(records: list[dict[str, float]], min_cluster_size: int, min_cluster_share: float) -> dict[str, float]:
    if not records:
        raise ValueError("没有可用的候选 k 评估结果。")
    stable_records = [
        record
        for record in records
        if record["min_cluster_size"] >= float(min_cluster_size) and record["min_cluster_share"] >= float(min_cluster_share)
    ]
    candidate_pool = stable_records if stable_records else records
    return min(
        candidate_pool,
        key=lambda record: (-record["silhouette"], record["davies_bouldin"], -record["calinski_harabasz"], record["inertia"]),
    )


def _series_summary(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=np.float64)
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "p50": float(np.quantile(values, 0.5)) if values.size else 0.0,
        "p90": float(np.quantile(values, 0.9)) if values.size else 0.0,
    }


def _cluster_profiles(frame: pd.DataFrame, feature_columns: list[str], scaler: StandardScaler) -> list[dict[str, Any]]:
    scaled = pd.DataFrame(scaler.transform(frame[feature_columns]), columns=feature_columns, index=frame.index)
    profiles: list[dict[str, Any]] = []

    display_columns = [
        "total_event_count",
        "selected_event_count",
        "active_span_hours",
        "behavior_density",
        "user_feature_count",
        "item_feature_count",
        "user_frequency",
        "item_frequency",
        "action_seq_length",
        "content_seq_length",
        "item_seq_length",
    ]

    for cluster_id in sorted(frame["cluster_id"].unique()):
        cluster_frame = frame[frame["cluster_id"] == cluster_id]
        cluster_scaled = scaled.loc[cluster_frame.index]
        z_scores = cluster_scaled.mean().sort_values(ascending=False)
        profiles.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(cluster_frame.shape[0]),
                "size_rate": float(cluster_frame.shape[0] / max(frame.shape[0], 1)),
                "positive_rate": float(cluster_frame["label"].mean()),
                "summary": {column: _series_summary(cluster_frame[column]) for column in display_columns},
                "top_positive_deviations": [
                    {"feature": str(feature), "zscore": float(value)}
                    for feature, value in z_scores.head(5).items()
                ],
                "top_negative_deviations": [
                    {"feature": str(feature), "zscore": float(value)}
                    for feature, value in z_scores.sort_values().head(5).items()
                ],
            }
        )
    return profiles


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = ["# 聚类分析报告", ""]
    selection = payload["model_selection"]
    outlier = payload["outlier_detection"]
    lines.append(
        f"先用 {outlier['algorithm']} 标记 {outlier['outlier_count']} 个 outlier ({outlier['outlier_rate']:.2%})，再在 {outlier['inlier_count']} 个 inlier 上做 KMeans。"
    )
    lines.append("")
    lines.append(
        f"最终选用 KMeans，k={selection['selected_k']}，silhouette={selection['selected_metrics']['silhouette']:.4f}，davies_bouldin={selection['selected_metrics']['davies_bouldin']:.4f}。"
    )
    lines.append("")
    lines.append("## 候选 k 评分")
    lines.append("")
    lines.append("| k | silhouette | calinski_harabasz | davies_bouldin | inertia | min_cluster_share | max_cluster_share |")
    lines.append("| -: | ---------: | ----------------: | -------------: | ------: | ----------------: | ----------------: |")
    for record in selection["candidate_metrics"]:
        lines.append(
            f"| {int(record['k'])} | {record['silhouette']:.4f} | {record['calinski_harabasz']:.2f} | {record['davies_bouldin']:.4f} | {record['inertia']:.2f} | {record['min_cluster_share']:.2%} | {record['max_cluster_share']:.2%} |"
        )

    lines.append("")
    lines.append("## Cluster Profiles")
    lines.append("")
    for profile in payload["clusters"]:
        lines.append(f"### Cluster {profile['cluster_id']}")
        lines.append("")
        lines.append(
            f"- 样本数: {profile['size']} ({profile['size_rate']:.2%})\n- 正样本率: {profile['positive_rate']:.4f}"
        )
        lines.append(f"- 正向偏离最明显: {', '.join(item['feature'] for item in profile['top_positive_deviations'])}")
        lines.append(f"- 负向偏离最明显: {', '.join(item['feature'] for item in profile['top_negative_deviations'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _plot_k_selection(candidate_metrics: list[dict[str, float]], output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    k_values = [int(record["k"]) for record in candidate_metrics]
    silhouette = [float(record["silhouette"]) for record in candidate_metrics]
    ch = [float(record["calinski_harabasz"]) for record in candidate_metrics]
    dbi = [float(record["davies_bouldin"]) for record in candidate_metrics]
    inertia = [float(record["inertia"]) for record in candidate_metrics]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    axes[0, 0].plot(k_values, silhouette, marker="o", color="#0F766E")
    axes[0, 0].set_title("Silhouette Score")
    axes[0, 1].plot(k_values, ch, marker="o", color="#2563EB")
    axes[0, 1].set_title("Calinski-Harabasz")
    axes[1, 0].plot(k_values, dbi, marker="o", color="#BE123C")
    axes[1, 0].set_title("Davies-Bouldin")
    axes[1, 1].plot(k_values, inertia, marker="o", color="#7C3AED")
    axes[1, 1].set_title("Inertia")
    for axis in axes.flat:
        axis.set_xlabel("k")
        axis.grid(alpha=0.25)
    return _save_figure(figure, output_dir / "cluster_k_selection", formats)


def _plot_pca_scatter(frame: pd.DataFrame, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    cluster_ids = sorted(int(value) for value in frame["cluster_id"].unique())
    color_map = plt.get_cmap("tab10", max(len(cluster_ids), 1))
    for color_index, cluster_id in enumerate(cluster_ids):
        cluster_frame = frame[frame["cluster_id"] == cluster_id]
        marker = "x" if cluster_id == -1 else "o"
        scatter_kwargs = {
            "color": color_map(color_index),
            "s": 55 if cluster_id == -1 else 45,
            "alpha": 0.85,
            "marker": marker,
            "label": "outlier" if cluster_id == -1 else f"cluster {cluster_id}",
        }
        if cluster_id != -1:
            scatter_kwargs["edgecolors"] = "white"
            scatter_kwargs["linewidths"] = 0.3
        axis.scatter(
            cluster_frame["pca_x"],
            cluster_frame["pca_y"],
            **scatter_kwargs,
        )
    axis.set_title("PCA Projection by Cluster")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    return _save_figure(figure, output_dir / "cluster_pca_scatter", formats)


def _plot_cluster_size_label_rate(frame: pd.DataFrame, output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    summary = frame.groupby("cluster_id").agg(size=("cluster_id", "size"), positive_rate=("label", "mean")).reset_index()
    labels = [str(int(value)) for value in summary["cluster_id"]]

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.bar(labels, summary["size"], color="#2563EB", label="Cluster size")
    rate_axis = axis.twinx()
    rate_axis.plot(labels, summary["positive_rate"], marker="o", color="#F59E0B", label="Positive rate")
    axis.set_title("Cluster Size and Positive Rate")
    axis.set_xlabel("Cluster ID")
    axis.set_ylabel("Size")
    rate_axis.set_ylabel("Positive rate")
    lines, axis_labels = axis.get_legend_handles_labels()
    lines2, axis_labels2 = rate_axis.get_legend_handles_labels()
    axis.legend(lines + lines2, axis_labels + axis_labels2, loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, output_dir / "cluster_size_label_rate", formats)


def _plot_cluster_profile_heatmap(frame: pd.DataFrame, feature_columns: list[str], output_dir: Path, formats: tuple[str, ...]) -> list[Path]:
    heatmap = frame.groupby("cluster_id")[feature_columns].mean()
    standardized = (heatmap - heatmap.mean(axis=0)) / heatmap.std(axis=0).replace(0.0, 1.0)

    figure, axis = plt.subplots(figsize=(14, 6), constrained_layout=True)
    image = axis.imshow(standardized.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm")
    axis.set_title("Cluster Profile Heatmap (column z-score)")
    axis.set_xticks(np.arange(len(feature_columns)), feature_columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(standardized.index)), [str(int(cluster_id)) for cluster_id in standardized.index])
    axis.set_xlabel("Feature")
    axis.set_ylabel("Cluster ID")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("z-score")
    return _save_figure(figure, output_dir / "cluster_profile_heatmap", formats)


def build_clustering_artifacts(
    dataset_path: str | Path,
    label_action_type: int = 2,
    max_seq_len: int = 256,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    outlier_fraction: float = 0.01,
    min_cluster_size: int = 20,
    min_cluster_share: float = 0.03,
) -> dict[str, Any]:
    row_frame = build_row_feature_frame(dataset_path, label_action_type=label_action_type, max_seq_len=max_seq_len)
    feature_frame, feature_columns = _build_feature_matrix(row_frame)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_frame[feature_columns])

    if outlier_fraction > 0.0:
        detector = IsolationForest(contamination=outlier_fraction, random_state=random_state)
        inlier_mask = detector.fit_predict(X) == 1
    else:
        inlier_mask = np.ones(X.shape[0], dtype=bool)

    X_inlier = X[inlier_mask]
    if X_inlier.shape[0] < 2:
        raise ValueError("去除异常点后剩余样本不足，无法继续聚类。")

    candidate_k = _candidate_k_values(X_inlier.shape[0], k_min, k_max)
    candidate_metrics = _evaluate_candidate_k(X_inlier, candidate_k, random_state=random_state)
    selected_metrics = _select_best_k(candidate_metrics, min_cluster_size=min_cluster_size, min_cluster_share=min_cluster_share)
    selected_k = int(selected_metrics["k"])

    final_model = KMeans(n_clusters=selected_k, n_init=30, random_state=random_state)
    inlier_labels = final_model.fit_predict(X_inlier)
    cluster_labels = np.full(feature_frame.shape[0], -1, dtype=np.int64)
    cluster_labels[inlier_mask] = inlier_labels

    pca = PCA(n_components=2, random_state=random_state)
    pca_embedding = pca.fit_transform(X)

    feature_frame = feature_frame.copy()
    feature_frame["cluster_id"] = cluster_labels.astype(np.int64)
    feature_frame["pca_x"] = pca_embedding[:, 0]
    feature_frame["pca_y"] = pca_embedding[:, 1]

    assignments = feature_frame[
        [
            "user_id",
            "item_id",
            "timestamp",
            "label",
            "cluster_id",
            "pca_x",
            "pca_y",
            "total_event_count",
            "behavior_density",
            "user_frequency",
            "item_frequency",
        ]
    ].copy()

    payload = {
        "dataset": {
            "path": str(dataset_path),
            "rows": int(feature_frame.shape[0]),
            "label_action_type": int(label_action_type),
            "max_seq_len": int(max_seq_len),
        },
        "features": {
            "feature_columns": feature_columns,
            "row_feature_columns": list(feature_frame.columns),
        },
        "model_selection": {
            "algorithm": "kmeans",
            "candidate_metrics": candidate_metrics,
            "selected_k": selected_k,
            "selected_metrics": selected_metrics,
            "min_cluster_size_constraint": int(min_cluster_size),
            "min_cluster_share_constraint": float(min_cluster_share),
        },
        "outlier_detection": {
            "algorithm": "isolation_forest" if outlier_fraction > 0.0 else "disabled",
            "outlier_fraction": float(outlier_fraction),
            "outlier_count": int((~inlier_mask).sum()),
            "outlier_rate": float((~inlier_mask).mean()),
            "inlier_count": int(inlier_mask.sum()),
        },
        "pca": {
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
            "component_count": 2,
        },
        "clusters": _cluster_profiles(feature_frame, feature_columns, scaler),
        "assignments_preview": assignments.head(20).to_dict(orient="records"),
        "artifacts": {},
        "tables": {
            "cluster_counts": assignments.groupby("cluster_id").size().to_dict(),
        },
        "_row_frame": feature_frame,
        "_assignments": assignments,
    }
    return payload


def export_clustering_artifacts(
    payload: dict[str, Any],
    output_dir: str | Path,
    formats: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    output_root = ensure_dir(output_dir)
    plots_dir = ensure_dir(output_root / "plots")
    normalized_formats = _normalize_formats(formats)

    row_frame = payload.pop("_row_frame")
    assignments = payload.pop("_assignments")

    assignments_path = output_root / "cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    report_json_path = output_root / "cluster_report.json"
    report_md_path = output_root / "cluster_report.md"

    plot_paths: list[Path] = []
    plot_paths.extend(_plot_k_selection(payload["model_selection"]["candidate_metrics"], plots_dir, normalized_formats))
    plot_paths.extend(_plot_pca_scatter(row_frame, plots_dir, normalized_formats))
    plot_paths.extend(_plot_cluster_size_label_rate(row_frame, plots_dir, normalized_formats))
    plot_paths.extend(_plot_cluster_profile_heatmap(row_frame, payload["features"]["feature_columns"], plots_dir, normalized_formats))

    payload["artifacts"] = {
        "output_dir": str(output_root),
        "cluster_assignments": str(assignments_path),
        "report_json": str(report_json_path),
        "report_markdown": str(report_md_path),
        "plots": [str(path) for path in plot_paths],
    }

    _write_json(report_json_path, payload)
    report_md_path.write_text(_build_markdown(payload), encoding="utf-8")
    return payload


def print_clustering_summary(payload: dict[str, Any]) -> None:
    selection = payload["model_selection"]
    outlier = payload["outlier_detection"]
    print(
        f"rows={payload['dataset']['rows']} outliers={outlier['outlier_count']} selected_k={selection['selected_k']} silhouette={selection['selected_metrics']['silhouette']:.4f} davies_bouldin={selection['selected_metrics']['davies_bouldin']:.4f}"
    )
    for profile in payload["clusters"]:
        print(
            f"cluster={profile['cluster_id']} size={profile['size']} size_rate={profile['size_rate']:.4f} positive_rate={profile['positive_rate']:.4f}"
        )


__all__ = ["build_clustering_artifacts", "export_clustering_artifacts", "print_clustering_summary"]