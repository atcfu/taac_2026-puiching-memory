# UniRec

本文档记录两部分内容。

第一部分概括公开仓库 hojiahao/TAAC2026 的真实模型主线。该仓库虽然 repo 名字叫 TAAC2026，但 README 和代码里的模型身份实际上是 UniRec。

第二部分说明当前仓库里的 unirec 集成如何把这套结构映射到 TAAC_2026 现有的 parquet batch 契约和共享训练栈上。

## 当前入口与验证

当前仓库把 UniRec 作为目录式实验包维护：

```bash
uv run taac-train --experiment config/gen/unirec
uv run taac-evaluate single --experiment config/gen/unirec
```

默认数据集是 sample parquet，默认输出目录是 outputs/gen/unirec。

当前可复核状态：

1. tests/test_experiment_packages.py 覆盖了 test_experiment_package_builds_and_runs_forward，其中包含 config.gen.unirec。
2. 当前先提供 package-level forward regression。
3. 还没有单独的 sample smoke summary。

## 1. 原项目主线

公开仓库 hojiahao/TAAC2026 把模型称为 UniRec，目标是把多字段特征交互和多序列行为建模压到同一个统一骨干里，服务 CVR 预测。

它在 README 中强调的结构主线主要包括：

1. Unified tokenization，把 user/item/context 等非序列特征和 action/content/item 三条行为序列统一成同一个 token 空间。
2. Feature cross layer，先在非序列特征区做显式特征交叉，对应 InterFormer 风格的前置 cross。
3. Target-aware interest，用候选 item 表示去查询历史序列，对应 DIN 风格的目标感知兴趣提取。
4. MoT，也就是对三条异构序列分别建模后再融合，对应 HSTU 2.0 里强调的多路 transducer。
5. Hybrid attention mask，把 feature 区、sequence 区和 special token 区放进同一注意力图里，用半局部窗口替代完全全局注意力。
6. Attention truncation 和 block attention residuals，用于降低长序列主干的计算代价并保持深层信息回取。
7. AUC-oriented loss 与 Muon/AdamW 混合优化器。

## 2. 当前仓库实现

当前实现位于 config/gen/unirec，并保持 data.py、model.py、utils.py 私有，不依赖其它实验包的模型实现。

### 2.1 数据侧映射

原项目有自己的一套显式 feature_id vocab、三路 sequence tensor 和训练时 vocab 构建逻辑。当前仓库没有照搬那套 parquet 解析，而是复用了现有 baseline 的数据管线。

这意味着当前版本直接消费的仍然是统一 BatchTensors：

1. user_tokens、context_tokens、candidate_tokens、candidate_post_tokens、candidate_author_tokens 组成非序列侧。
2. history_tokens、history_post_tokens、history_author_tokens、history_action_tokens、history_time_gap、history_group_ids 提供事件级行为信息。
3. history_group_ids 会在模型内部重新拆回 action/content/item 三路序列。

这种做法的好处是：不改共享 parquet 契约，也能保留原项目真正有辨识度的统一组装过程。

### 2.2 当前保留的结构

当前集成版保留了 UniRec 最关键的几块：

1. 六个非序列 token 的独立投影，加 feature cross 层，保留 mixed parameterization 和前置特征交叉。
2. 从合并 history 中按 group_id 重建 action/content/item 三路序列。
3. 候选 item 对全历史的 target-aware interest 聚合。
4. 三路 BranchTransducer + gated fusion 的 MoT。
5. 统一序列组装：feature tokens + three sequence branches + optional MoT + interest + target。
6. Hybrid attention mask、SiLU gated attention、block attention residual、末端 truncation。
7. BCE + pairwise AUC 的组合损失，以及 Muon + AdamW 的混合优化器分组。

### 2.3 与原项目的主要差异

当前仓库里的版本不是原仓库逐脚本复刻，主要差异有：

1. 原项目的数据层是 feature_id 显式词表和独立 dataloader；当前版本改成复用本仓库 baseline parquet pipeline。
2. 原项目的训练入口支持 DDP、bf16、compile、checkpointing 和更多 loss/optimizer 开关；当前版本只保留了最关键的建模与优化思想，适配本仓库统一 train/evaluate runtime。
3. 原项目的 attention truncation 会同时保留所有 NS token 和 special token；当前版本遵循同一思路，但使用当前共享 ModelConfig 能承载的最小参数集实现。
4. 原项目的半局部注意力是论文化设计；当前版本为了在现有 runtime 下更稳地训练，把 SiLU gated attention 做了归一化处理，避免 sample parquet 上数值过度放大。

### 2.4 当前适合的定位

当前 unirec 包适合被看作：

1. 一个正式可训练的 UniRec-style 统一骨干实验包。
2. 一个把 hojiahao/TAAC2026 的核心结构压缩到当前共享 parquet 契约上的适配版本。
3. 后续继续补更完整 scaling 配置、更多 loss 选项和 smoke 产物的起点。