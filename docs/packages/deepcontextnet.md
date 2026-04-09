# DeepContextNet

这个实验包把 suyanli220 的 TAAC-2026-Baseline-Tencent-Advertisement-Contest 迁移到当前仓库的 parquet 实验契约里，保留它最关键的三件事：全局 CLS 聚合、统一的 user/item/sequence token 序列建模，以及 Muon 风格的矩阵正交优化器。

## 来源

1. [suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest](https://github.com/suyanli220/TAAC-2026-Baseline-Tencent-Advertisement-Contest)

## 当前适配方式

1. 没有照搬外部仓库把数据扫描、特征映射、模型和训练循环都写在单个 main.py 里的形式，而是拆成当前仓库的目录式实验包。
2. 数据层直接复用现有 parquet pipeline，不重新实现外部仓库的 pandas 扫描和临时特征字典。
3. user_feature 和 context_feature 被汇总成一个 user-context token，candidate 相关张量被汇总成一个 item token。
4. 历史序列用 history_post_tokens、history_action_tokens、history_time_gap、history_group_ids 做加性融合，映射外部仓库的 item_seq + action_seq + temporal bucket 主线。
5. 公开 README 把该方案描述成 HSTU / Rotary 风格基线，但当前公开 main.py 里最直接可复用的主干实际上是 pre-norm 多头注意力块加 MLP，因此本集成按可见代码实现复现，而不是按 README 里的术语额外补一套未出现的骨干。
6. 优化器层保留了外部仓库最有辨识度的 MatrixUnitaryOptimizer 思路，用 SVD 近似正交化矩阵更新。

## 运行

```bash
uv run taac-train --experiment config/gen/deepcontextnet
uv run taac-evaluate single --experiment config/gen/deepcontextnet
```

## 当前验证状态

1. 已接入 experiment package forward regression。
2. 当前实现优先保证和现有 parquet batch 契约兼容，暂未复刻外部 main.py 里的 pandas 特征扫描与 AMP 训练循环。