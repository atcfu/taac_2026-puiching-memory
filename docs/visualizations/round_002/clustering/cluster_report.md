# 聚类分析报告

先用 isolation_forest 标记 10 个 outlier (1.00%)，再在 990 个 inlier 上做 KMeans。

最终选用 KMeans，k=3，silhouette=0.2272，davies_bouldin=1.6505。

## 候选 k 评分

| k | silhouette | calinski_harabasz | davies_bouldin | inertia | min_cluster_share | max_cluster_share |
| -: | ---------: | ----------------: | -------------: | ------: | ----------------: | ----------------: |
| 2 | 0.2187 | 173.14 | 1.9911 | 9485.79 | 27.98% | 72.02% |
| 3 | 0.2272 | 148.11 | 1.6505 | 8574.71 | 3.74% | 69.39% |
| 4 | 0.1508 | 141.26 | 1.7906 | 7796.97 | 3.74% | 43.74% |
| 5 | 0.1607 | 144.96 | 1.5934 | 7017.24 | 3.74% | 39.90% |
| 6 | 0.1808 | 144.18 | 1.4804 | 6434.26 | 3.74% | 33.74% |
| 7 | 0.1808 | 138.69 | 1.5254 | 6037.44 | 3.74% | 30.71% |
| 8 | 0.1919 | 138.32 | 1.2762 | 5613.32 | 0.20% | 31.62% |

## Cluster Profiles

### Cluster -1

- 样本数: 10 (1.00%)
- 正样本率: 0.1000
- 正向偏离最明显: item_seq_share, log_item_frequency, log_behavior_density, content_seq_share, log_user_frequency
- 负向偏离最明显: truncation_flag, log_selected_event_count, log_active_span_hours, non_empty_group_count, log_total_event_count

### Cluster 0

- 样本数: 687 (68.70%)
- 正样本率: 0.0975
- 正向偏离最明显: action_seq_share, log_total_event_count, log_behavior_density, hour_sin, hour_cos
- 负向偏离最明显: content_seq_share, item_seq_share, log_item_frequency, log_user_frequency, user_feature_count

### Cluster 1

- 样本数: 266 (26.60%)
- 正样本率: 0.1165
- 正向偏离最明显: item_seq_share, content_seq_share, hour_cos, hour_sin, non_empty_group_count
- 负向偏离最明显: action_seq_share, log_total_event_count, log_behavior_density, item_feature_count, log_user_frequency

### Cluster 2

- 样本数: 37 (3.70%)
- 正样本率: 0.1081
- 正向偏离最明显: content_seq_share, non_empty_group_count, truncation_flag, log_selected_event_count, user_feature_count
- 负向偏离最明显: hour_cos, hour_sin, timestamp_rank, action_seq_share, log_total_event_count

