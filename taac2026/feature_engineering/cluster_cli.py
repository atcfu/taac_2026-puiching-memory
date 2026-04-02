from __future__ import annotations

import argparse

from .clustering import build_clustering_artifacts, export_clustering_artifacts, print_clustering_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出 TAAC 2026 数据集的专业级聚类分析报告、cluster assignment 与图表。")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/datasets--TAAC2026--data_sample_1000/snapshots/2f0ddba721a8323495e73d5229c836df5d603b39/sample_data.parquet",
        help="Parquet 数据集路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/feature_engineering/clustering",
        help="导出目录，默认输出 cluster_report.json / .md / assignments.csv 和 plots/。",
    )
    parser.add_argument(
        "--label-action-type",
        type=int,
        default=2,
        help="正样本 action_type，默认与训练配置保持一致。",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="用于衍生截断相关特征的 max_seq_len。",
    )
    parser.add_argument("--k-min", type=int, default=2, help="候选聚类数最小值。")
    parser.add_argument("--k-max", type=int, default=8, help="候选聚类数最大值。")
    parser.add_argument("--random-state", type=int, default=42, help="KMeans / PCA 的随机种子。")
    parser.add_argument("--outlier-fraction", type=float, default=0.01, help="先用 IsolationForest 剔除异常点的比例。设为 0 可关闭。")
    parser.add_argument("--min-cluster-size", type=int, default=20, help="选 k 时要求每个主体簇至少包含的样本数。")
    parser.add_argument("--min-cluster-share", type=float, default=0.03, help="选 k 时要求每个主体簇至少占 inlier 样本的比例。")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        help="图像导出格式列表，支持 png 和 svg。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_clustering_artifacts(
        dataset_path=args.dataset_path,
        label_action_type=args.label_action_type,
        max_seq_len=args.max_seq_len,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        outlier_fraction=args.outlier_fraction,
        min_cluster_size=args.min_cluster_size,
        min_cluster_share=args.min_cluster_share,
    )
    payload = export_clustering_artifacts(payload, output_dir=args.output_dir, formats=args.formats)
    print_clustering_summary(payload)
    for path in payload["artifacts"]["plots"]:
        print(f"plot_written_to={path}")
    print(f"cluster_report_written_to={payload['artifacts']['report_json']}")
    print(f"cluster_assignments_written_to={payload['artifacts']['cluster_assignments']}")


if __name__ == "__main__":
    main()