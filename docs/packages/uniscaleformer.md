# UniScaleFormer

本文档记录两部分内容。

第一部分概括公开仓库 twx145/Unirec 的真实主线。虽然仓库名字叫 Unirec，但它真正的新模型不是再做一个 UniRec 复刻，而是一个统一模板里的核心模型 UniScaleFormer。

第二部分说明当前仓库里的 uniscaleformer 集成如何把这套结构映射到 TAAC_2026 现有的 parquet batch 契约和共享训练栈上。

## 当前入口与验证

当前仓库把 UniScaleFormer 作为目录式实验包维护：

```bash
uv run taac-train --experiment config/gen/uniscaleformer
uv run taac-evaluate single --experiment config/gen/uniscaleformer
```

默认数据集是 sample parquet，默认输出目录是 outputs/gen/uniscaleformer。

当前可复核状态：

1. tests/test_experiment_packages.py 覆盖了 test_experiment_package_builds_and_runs_forward，其中包含 config.gen.uniscaleformer。
2. 当前先提供 package-level forward regression。
3. 还没有单独的 sample smoke summary。

## 1. 原项目主线

twx145/Unirec 实际上是一个面向 TAAC2026 的统一训练模板。它把四个模型放在同一个数据栈和训练入口里：

1. UniScaleFormer，仓库自己的主模型。
2. InterFormer 基线。
3. OneTrans 基线。
4. HyFormer 基线。

其中真正新增、且当前仓库尚未独立接入的，是 UniScaleFormer。它的 README 给出的主线包括：

1. Unified tokenizer，把静态多字段特征和多序列事件统一转成 token。
2. 分类型 value adapter，整数值走哈希 embedding，浮点值走 MLP。
3. 多序列 memory 压缩，每条行为序列先编码，再压成 memory token，降低长序列代价。
4. Interleaved Query Block，用目标感知 queries 去读 sequence memory，再和 static tokens 交替混合。
5. Hybrid Head，用 pooled Transformer 表征和 FM 显式交叉共同做打分。
6. 可选 auxiliary contrastive loss，用 query pool 和 item token 相似度做额外约束。

## 2. 当前仓库实现

当前实现位于 config/gen/uniscaleformer，并保持 data.py、model.py、utils.py 私有，不依赖其它实验包的模型实现。

### 2.1 为什么只接入 UniScaleFormer

外部仓库虽然还能跑 interformer、onetrans、hyformer 三个模型，但这三类结构当前仓库已经分别有独立实验包：

1. config/gen/interformer
2. config/gen/onetrans
3. config/gen/hyformer

因此这次集成不再重复接外部模板里的三个基线，而是只把 UniScaleFormer 这一条主模型接进来。

### 2.2 数据侧映射

原项目自己的数据栈会构造下面这些张量：

1. static_token_ids / static_feature_ids / static_type_ids / static_float_values / static_mask
2. seq_token_ids / seq_feature_ids / seq_type_ids / seq_pos_ids / seq_mask / seq_name_ids

当前仓库没有照搬那套 collator，而是复用了现有 baseline parquet 数据管线，再在模型内部完成映射：

1. user_indices 与 item_indices 映射成 special static tokens，对齐原项目里的 user_id / item_id 特征。
2. user_tokens、context_tokens、candidate_tokens、candidate_post_tokens、candidate_author_tokens、dense_features 组成静态 token 区。
3. history_tokens、history_post_tokens、history_author_tokens、history_action_tokens、history_time_gap、history_group_ids 组成事件级表示。
4. history_group_ids 会在模型内部重新拆回 action/content/item 三条序列，再送入每条序列自己的 self-attn encoder 和 memory compressor。

这种适配方式保留了原项目最关键的统一建模路径，同时不破坏当前仓库共享的 BatchTensors 契约。

### 2.3 当前保留的结构

当前集成版保留了 UniScaleFormer 最核心的几块：

1. 静态 token 区和多序列事件区分开编码。
2. 每条序列单独 self-attn，再用 learned memory tokens 压缩成 memory。
3. query_seed 加 static_pool / sequence_pool 生成目标感知 query。
4. 多层 UniScaleLayer：query 读 memory，然后和 static tokens 一起做 token mixing 与 FFN。
5. 最后把 queries 与 static tokens 再做一次 self-attn 融合。
6. Hybrid Head：query_pool + static_pool + fused_pool + FM 显式交叉。

### 2.4 与原项目的主要差异

当前仓库里的版本不是原仓库逐脚本复刻，主要差异有：

1. 原项目的 tokenizer 依赖自己的 collate 输出 feature_ids、type_ids 和 float_values；当前版本改成在现有 BatchTensors 上做语义对齐，不重写共享数据栈。
2. 原项目用 auxiliary contrastive loss；当前共享 runtime 只接 pointwise logits，因此当前版本先保留模型里的相似度表征，但训练损失适配为 BCEWithLogitsLoss。
3. 原项目的 static token 数量可以非常大；当前版本为了适配现有 sample parquet 和共享运行时，用较小、固定数目的静态 token 近似这部分结构。
4. 原项目是四模型统一模板；当前版本只把 UniScaleFormer 作为独立实验包落地。

### 2.5 当前适合的定位

当前 uniscaleformer 包适合被看作：

1. 一个正式可训练的 UniScaleFormer-style 统一模板主模型。
2. 一个把 twx145/Unirec 里真正新增的 memory-compressed query-static 交替结构迁移到当前 parquet runtime 的适配版本。
3. 后续继续补更完整 auxiliary loss、more.md 里的对比脚本语义和 smoke 产物的起点。