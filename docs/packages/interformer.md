# InterFormer

本文档分成两部分。

第一部分是根据 InterFormer 论文整理出的 Markdown 化说明，目的是把论文的叙事、符号和模块关系压成一份便于工程实现的文字版本。这里是基于论文内容的中文转述与整理，不是逐字翻译，也不是原文转录。

第二部分说明当前仓库里的实现方式、已经做出的集成选择，以及与论文仍然存在的差距，方便后续继续向论文靠拢。

如果想直接看完整论文整理版，可继续看 [InterFormer 论文拆解](../papers/interformer.md)。

## 当前入口与验证

当前仓库把 InterFormer 作为目录式实验包维护：

```bash
uv run taac-train --experiment config/gen/interformer
uv run taac-evaluate single --experiment config/gen/interformer
```

默认数据集是 sample parquet，默认输出目录是 `outputs/gen/interformer`。

当前可复核状态：

1. `tests/test_experiment_packages.py` 覆盖了 `test_experiment_package_builds_and_runs_forward`，其中包含 `config.gen.interformer`。
2. 一轮 sample smoke 已写到 `outputs/smoke/interformer/summary.json`。
3. 当前 10-epoch sample 结果的最佳 epoch 是 8，AUC 0.6707，PR-AUC 0.3006，平均时延 19.2891 ms / sample。

## 1. 论文问题定义

InterFormer 面向的是推荐系统中的 CTR / CVR 一类排序任务。输入通常同时包含两类信息。

一类是 non-sequence 特征，也就是用户静态画像、上下文信息、候选物品特征等非行为序列信号。

另一类是 multi-sequence 特征，也就是用户过去在不同语义空间上的多条行为序列。例如点击序列、内容消费序列、商品交互序列、动作类型序列等。它们既有时间顺序，又来自不同的行为域。

论文要解决的核心问题不是单纯把这些特征拼在一起，而是要同时处理三种结构性困难。

第一，不同来源的非序列特征和序列特征处在不同的语义空间里。

第二，用户的行为并不只来自一条序列，而是来自多条并行序列；这些序列之间既有局部依赖，也有跨序列关系。

第三，推荐打分需要同时建模用户的稳定偏好、近期兴趣和与候选物品相关的上下文依赖。

InterFormer 的主张是，不要把序列侧和非序列侧完全割裂地独立建模再晚融合，而是通过 interleaving learning，让两侧在层间交替交换摘要信息。

## 2. 论文整体结构

InterFormer 可以理解为三块结构共同组成的层叠模型。

一块是 Cross Arch，用来把非序列侧和多序列侧分别压成可交换的摘要表示。

一块是 Interaction Arch，用来在非序列侧做行为感知的交互学习。

一块是 Sequence Arch，用来在序列侧做上下文感知的序列建模。

每一层里，Cross Arch 先从当前层的非序列表示和序列表示中提取摘要；然后 Interaction Arch 用序列摘要更新非序列表示，Sequence Arch 用非序列摘要更新序列表示。这样层层交替，逐渐把用户稳定偏好与行为动态整合起来。

可以把一层的输入记作两部分。

非序列侧记为 X。

序列侧记为 S。

Cross Arch 生成两个摘要。

非序列摘要记为 X_sum。

序列摘要记为 S_sum。

随后：

Interaction Arch 接收 X 和 S_sum，输出新的 X。

Sequence Arch 接收 S 和 X_sum，输出新的 S。

最终分类头基于两边的摘要做预测。

## 3. Cross Arch

Cross Arch 的职责是为“非序列更新”和“序列更新”提供交换信号。论文里这一步不是简单均值池化，而是带结构的摘要提取。

### 3.1 非序列摘要

非序列侧包含用户特征、上下文特征、候选物品特征等。论文的做法是把这些特征编码成一组 token，然后通过压缩和门控，形成固定数量的 summary token，作为 X_sum。

这一步的目标不是还原所有非序列细节，而是把当前排序决策最有用的非序列上下文压缩出来，提供给序列侧作为条件信息。

### 3.2 序列摘要

序列侧不是只保留一个全局池化向量，而是组合多种摘要视角。

论文强调了几类信息来源。

一类是由非序列摘要引入的 cls-like summary token。它们在第一层被 prepend 到序列前面，后续层里继续参与更新。

一类是 PMA 风格的 pooled tokens，用来从整段序列中抽取全局兴趣概括。

一类是 recent tokens，用来直接保留最新行为，以防止全局压缩丢掉短期兴趣。

这几部分拼起来形成 S_sum。然后 S_sum 反馈给非序列侧，用于行为感知的交互学习。

### 3.3 多序列统一

论文不是把多条行为序列完全独立编码到底，再在最后做简单拼接，而是希望在 Cross Arch 中就开始做多序列统一。这个统一过程包含两个关键词。

一个是 LCE，也就是线性压缩式的序列统一。

一个是 self-gating，也就是对压缩后的表示再做门控筛选。

直觉上，LCE 负责把冗余长度和跨序列维度压成一个更适合后续层消费的表示，自门控负责控制不同来源信息的保留比例。

## 4. Interaction Arch

Interaction Arch 位于非序列分支，它的目标不是做一般意义上的 feature interaction，而是做 behavior-aware interaction。

这里的“behavior-aware”意思是：非序列特征之间的交互强度，应该受到当前序列摘要的调制。

例如，用户画像中的长期偏好与候选物品特征之间是否应该产生强交互，不应只由两者本身决定，还应受到最近行为摘要的影响。

因此，Interaction Arch 的核心逻辑可以理解成两步。

第一步，在非序列 token 内部做自交互，得到当前非序列上下文。

第二步，引入序列摘要，让非序列 token 在序列语义的条件下重新校准交互结果。

最终输出新的非序列表示，传给下一层。

## 5. Sequence Arch

Sequence Arch 位于序列分支，它的目标是在上下文条件下建模多条行为序列。

论文强调两点。

第一，序列并不是孤立处理的，非序列摘要 X_sum 会作为条件信息注入到序列建模过程中。

第二，序列建模并不只靠一个标准 Transformer block，而是结合了个性化前馈与多头注意力。

论文里可以把这一层理解成：

先用 PFFN 根据 X_sum 生成面向当前样本的个性化调制，再用 MHA 建模序列内部依赖。

这样得到的序列表示既保留了行为顺序关系，也带上了非序列侧提供的上下文条件。

### 5.1 PFFN

PFFN 的本质是条件化前馈层。它根据当前样本的非序列摘要，决定序列特征应该如何被投影和放大。

直观上，它不是给所有样本一套完全共享的前馈变换，而是让变换对当前用户和当前请求上下文具有一定的个性化。

### 5.2 MHA

在完成条件化投影之后，Sequence Arch 再用多头注意力学习序列内部依赖。这样可以同时覆盖长期依赖、短期依赖和跨行为域的局部关系。

## 6. 层间交替更新

InterFormer 的关键不只是拥有三块模块，而是这三块模块的执行顺序。

每一层都重复如下逻辑：

1. 从当前 X 和 S 中提取 X_sum 与 S_sum。
2. 用 S_sum 更新 X。
3. 用 X_sum 更新 S。

这样做的好处是，非序列侧不会只在输入阶段影响序列一次，序列侧也不会只在末端影响非序列一次。两侧在每一层都发生信息交换，这就是论文强调的 interleaving learning。

## 7. 输出头

论文最终的排序预测不是直接用最后一个 token，而是基于两边的摘要表示做打分。

可以把输出理解为：

先得到最终层的 X_sum 和 S_sum。

再把两者拼接后送入分类头，输出点击或转化概率对应的 logit。

这样做的动机是，最终打分既依赖非序列侧的稳定上下文，也依赖序列侧的动态兴趣摘要。

## 8. 论文侧的实现要点

如果只保留对工程最重要的要点，可以把论文概括成下面几条。

1. 非序列与序列不是早期拼接，也不是末端松散融合，而是在每层交替交换摘要。
2. 序列不是单序列，而是多序列统一建模。
3. Cross Arch 负责摘要和统一，Interaction Arch 负责行为感知的非序列交互，Sequence Arch 负责上下文感知的序列建模。
4. 序列摘要同时保留全局兴趣和近期兴趣。
5. 最终预测基于非序列摘要与序列摘要的联合表示。

## 9. 当前仓库实现映射

当前仓库把 InterFormer 放在 config/gen/interformer 下，保持实验私有代码自包含。

文件划分如下：

- data.py: 私有数据管道，负责把 parquet 样本编码成 batch。
- model.py: InterFormer 主体结构。
- utils.py: 损失和优化器定义。
- __init__.py: 默认配置与 `ExperimentSpec` 装配。

### 9.1 当前数据侧实现

当前数据实现不是论文原始工业特征流水线，而是为了在 TAAC_2026 当前仓库中先把实验跑通，做了一套私有 token 化编码。

主要特点如下。

1. 非序列侧把 user、context、candidate 特征编码成 token 序列。
2. 序列侧除了保留 sequence_tokens，还额外构造了 history_post、history_author、history_action、time_gap、group_id 等显式事件通道。
3. 多条行为序列会先按时间合并成一条 interleaved history，再用 group_id 保留原始来源序列。

这套数据流不是论文数据处理的逐项复刻，但它能把论文真正需要的几个语义维度落到 batch contract 上：内容实体、作者实体、动作类型、时间间隔、来源序列。

### 9.2 当前模型实现

当前模型保留了论文的三段式骨架。

1. NonSequenceSummary 对非序列 token 做 LCE 风格压缩，形成 X_sum。
2. SequenceSummary 组合 cls token、PMA token 与 recent token，形成 S_sum。
3. BehaviorAwareInteractionArch 用 S_sum 更新非序列表示。
4. ContextAwareSequenceArch 用 X_sum 更新序列表示。
5. 最终分类头基于最终层的 X_sum 与 S_sum 做打分。

### 9.3 本次更新后的关键改动

这次集成里做了两项重要收敛。

第一，序列分支现在真正吃进了显式事件通道，而不是只看粗粒度的 sequence_tokens。当前 sequence branch 直接使用下列输入构造每个事件位置的表示：

- history token
- history_post token
- history_author token
- history_action token
- time_gap embedding
- group_id embedding

第二，序列自注意力已经从之前的 rotary 版本退回标准 MHA。这样做的原因不是 rotary 一定更差，而是论文文本更接近“条件化前馈 + 标准多头注意力”的表述；先收回到保守实现，更便于后续做论文对照和 ablation。

## 10. 当前的集成取舍

这部分记录的是“为什么现在这样实现”。

### 10.1 为什么使用显式 history 通道

如果只保留 sequence_tokens，那么模型知道“这里发生过一个事件”，但不知道这个事件的实体、作者、动作和时间间隔分别是什么。

论文的建模重点之一就是行为语义和跨序列关系，因此当前实现明确把 post、author、action、time gap、group id 接进 sequence branch。这样做比只用一个合成 token 更接近论文意图。

### 10.2 为什么使用按时间合并后的 interleaved history

论文强调 interleaving learning。工程上，如果仍然严格保留三维张量形状 [sequence_type, position, feature]，后续在序列层里真正表达跨序列时间关系会比较别扭。

所以当前代码选择先把多条行为序列按时间合并，再用 group_id 标记来源序列。这样序列主干天然看到的是一条按时间排序的混合行为流，更符合“交错建模”的直觉。

### 10.3 为什么先用 BCE-only

当前 utils.py 只保留了 BCEWithLogitsLoss，没有额外叠加 pairwise ranking loss。理由很简单：先把论文主干结构复原，再决定是否需要额外训练技巧。否则结构差异和训练差异会混在一起，难以分析。

### 10.4 为什么默认配置仍然偏小

当前默认配置优先服务于仓库内的 smoke 训练和回归测试，而不是一次性追到论文实验规模。这样做的好处是可以更快验证接口、收敛路径和结果文件产出。等结构稳定后，再扩到论文级配置更稳妥。

## 11. 当前实现与论文的差距

这部分专门列出还没有完全对齐论文的地方。

1. 当前数据编码仍然是仓库私有的哈希 token 方案，不是论文原始工业特征系统。
2. 当前 PFFN 的参数化是工程实现版，不保证与论文内部公式逐项同构。
3. 当前 Cross Arch 的 LCE 与 self-gating 是按论文思想重建的近似实现，而不是官方代码复刻。
4. 当前默认超参数和训练预算远小于论文正式实验。
5. 当前验证重点仍然是“链路正确与可训练”，而不是“论文指标完全对齐”。

## 12. 后续建议

如果目标是继续往论文复现推进，建议按下面顺序做。

1. 固定当前结构，补一组 ablation：去掉 time gap、去掉 group_id、去掉 recent token、去掉 PFFN。
2. 把数据编码进一步往论文语义靠拢，减少哈希签名带来的信息损失。
3. 放大模型规模和训练预算，再看指标是否开始逼近论文趋势。
4. 最后再决定是否需要加入更多训练技巧，而不是过早把结构差异和训练技巧混在一起。
