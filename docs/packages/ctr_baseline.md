# CTR Baseline

这个实验包把 creatorwyx 的 TAAC2026-CTR-Baseline 迁移到当前仓库的 parquet 实验契约里，保留它最核心的 DIN 思路：候选目标向量对历史行为序列做 target-aware attention，再与用户侧和稠密特征拼接做 CTR 打分。

## 来源

1. [creatorwyx/TAAC2026-CTR-Baseline](https://github.com/creatorwyx/TAAC2026-CTR-Baseline)

## 当前适配方式

1. 没有沿用外部仓库的 JSONL 分卷和 byte-offset 地址簿，而是直接复用本仓库现有的 parquet data pipeline。
2. 候选 item 表示映射为 candidate_post_tokens、candidate_author_tokens 和 candidate_tokens 的联合编码。
3. 历史兴趣序列映射为 history_post_tokens、history_author_tokens 和 history_action_tokens 的联合编码，再走 DIN 风格 attention。
4. 用户侧画像映射为 user_tokens 和 context_tokens 的池化表示。
5. 外部仓库里的 cross_dense 映射为本仓库已有的 dense_features。

## 运行

```bash
uv run taac-train --experiment config/gen/ctr_baseline
uv run taac-evaluate single --experiment config/gen/ctr_baseline
```

## 当前验证状态

1. 已接入 experiment package forward regression。
2. 当前实现优先保证与现有 parquet batch 契约兼容，不追求逐字段复刻外部 Tenrec 数据清洗流程。