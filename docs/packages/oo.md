# O_o

本文档记录两部分内容。

第一部分概括外部项目 salmon1802/O_o 的真实公开实现主线。

第二部分说明当前仓库里的 oo 集成如何把它映射到 TAAC_2026 现有的 parquet 数据契约和共享训练栈上。

## 当前入口与验证

当前仓库把 O_o 作为目录式实验包维护：

```bash
uv run taac-train --experiment config/gen/oo
uv run taac-evaluate single --experiment config/gen/oo
```

默认数据集是 sample parquet，默认输出目录是 `outputs/gen/oo`。

当前可复核状态：

1. `tests/test_experiment_packages.py` 覆盖了 `test_experiment_package_builds_and_runs_forward`，其中包含 `config.gen.oo`。
2. 一轮 sample smoke 已写到 `outputs/smoke/oo/summary.json`。
3. 当前 10-epoch sample 结果的最佳 epoch 是 5，AUC 0.6851，PR-AUC 0.2669，平均时延 0.4663 ms / sample。

## 1. 原项目主线

O_o 是腾讯广告算法大赛 2025 的一套检索式序列建模方案。它的训练目标不是标准 pointwise BCE，而是更偏 next-item retrieval：

1. 用时间感知的序列编码器把用户历史编码成 query representation。
2. 用 item 侧特征编码器得到候选物品表示。
3. 用 InfoNCE 加 batch-all negatives 做训练，并结合 item popularity 的 -logQ 修正。

公开仓库的主要文件是：

1. model.py：核心模型。
2. dataset.py：基于 seq.jsonl、item_feat_dict.json、indexer.pkl 的数据集与特征交叉构造。
3. main.py：训练入口，包含 InfoNCE、自监督开关、AMP、feature crossing 配置。
4. model_rqkmeans.py：面向压缩和部署的残差 kmeans / RQ 风格量化。

## 2. 原项目最核心的模型设计

### 2.1 HSTU

原项目的主干不是标准 TransformerEncoder，而是 HSTU。每层把 QK 注意力拆成三路：

1. plain dot-product attention。
2. RoPE 旋转后的 attention。
3. relative time bias 对 V 的时间加权读出。

三路结果拼接后再乘一个 U 门控，再线性映射回隐藏维并做 RMSNorm 残差。

### 2.2 时间建模

原项目同时用了两类时间信息：

1. FourierTimeEncoding，把绝对时间戳映射成多频正余弦后再投到隐藏维。
2. RelativeTimeBias，把事件间时间差分桶后直接作为 attention bias。

这也是它和普通 DIN / SASRec 风格序列编码器相比最有辨识度的地方。

### 2.3 特征交互编码

原项目在 item 侧和 user-item 侧都用了一个 FeatureInteractionEncoder。它先把高维拼接特征 down-project 到隐藏维，再做一个 expansion + gated mixing，形成轻量的 feature interaction block。

### 2.4 原始训练目标

原项目训练时输出的不是单个 CTR logit，而是：

1. user sequence hidden states。
2. positive item embeddings。
3. negative item embeddings。

随后 main.py 里的 InfoNCE 用这些表示进行 batch 内对比学习，并支持 popularity-aware 的 -logQ 修正以及自监督增强。

## 3. 当前仓库实现

当前实现位于 config/gen/oo，并保持 data.py、model.py、utils.py 私有，不依赖其它实验包。

### 3.1 已直接移植的结构

1. RotaryEmbedding。
2. RelativeTimeBias。
3. FourierTimeEncoding。
4. HSTU 三路时间感知注意力块。
5. FeatureInteractionEncoder。

这些结构现在都在本仓库的 `config/gen/oo/model.py` 中以 TAAC 当前 batch 契约重写。

### 3.2 当前如何映射到 TAAC batch 契约

由于当前共享数据栈输出的是 parquet batch，而不是 O_o 原仓库的 seq.jsonl + item_feat_dict.json，因此当前映射方式是：

1. user_tokens / context_tokens / dense_features 作为全局 user-context 条件。
2. history_tokens / history_post_tokens / history_author_tokens / history_action_tokens / history_time_gap / history_group_ids 共同构成历史事件表示。
3. candidate_tokens / candidate_post_tokens / candidate_author_tokens 构成候选 item 表示。

序列侧用 merged history 做单流 HSTU 编码，保留 O_o 的 retrieval-style “history query 表征 + candidate item 表征” 主线；最后再加一个轻量 MLP score head，把它适配到当前仓库统一的 CTR logit 输出接口。

### 3.3 与原项目的主要差异

1. 原项目是检索式 InfoNCE 训练；当前仓库集成先适配为 pointwise BCE 训练，以便直接接入共享 train/evaluate runtime。
2. 原项目依赖 seq.jsonl、indexer.pkl、item_feat_dict.json、feature_cross_manifest.json；当前版本改为复用本仓库私有 parquet 数据管道。
3. 原项目的 feature crossing、-logQ correction、自监督增强、RQ-kmeans 量化暂未接入。
4. 原项目更像 2025 检索方案；当前版本保留的是它最核心、最可迁移的序列建模主干，而不是逐脚本复现它的整套离线检索训练系统。

### 3.4 当前适合的定位

当前 oo 包适合被看作：

1. 一个正式可训练的 O_o-style HSTU 基线。
2. 一个把 2025 检索架构迁移到 2026 CTR 统一运行时下的适配版本。
3. 后续继续补 InfoNCE、feature crosses、popularity correction 的起点。
