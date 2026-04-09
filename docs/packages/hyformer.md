# HyFormer

本文档分成两部分。

第一部分把论文 HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction 压成便于工程实现的 Markdown 说明。

第二部分记录当前仓库里 HyFormer 的实现映射与取舍，明确哪些已经按论文主线落下，哪些仍然是部署侧或工业特征侧的差距。

如果想直接看完整论文整理版，可继续看 [HyFormer 论文拆解](../papers/hyformer.md)。

## 当前入口与验证

当前仓库把 HyFormer 作为目录式实验包维护：

```bash
uv run taac-train --experiment config/gen/hyformer
uv run taac-evaluate single --experiment config/gen/hyformer
```

默认数据集是 sample parquet，默认输出目录是 `outputs/gen/hyformer`。

当前可复核状态：

1. `tests/test_experiment_packages.py` 覆盖了 `test_experiment_package_builds_and_runs_forward`，其中包含 `config.gen.hyformer`。
2. 当前工作区已有一轮 10-epoch sample smoke，见 `outputs/smoke/hyformer/summary.json`。
3. 当前 10-epoch sample 结果的最佳 epoch 是 10，AUC 0.7615，PR-AUC 0.3338，平均时延 19.6337 ms / sample。

## 1. 论文主线

HyFormer 要解决的问题，是工业 CTR/LRM 里长期存在的两段式范式：先用 LONGER 之类的 sequence encoder 压缩超长行为序列，再把压缩后的少量 query token 和 heterogeneous non-sequential features 送进 RankMixer 一类的 feature interaction 模块。

论文认为这个范式有三个核心限制。

1. 序列压缩使用的 query token 往往只来自有限的 target/global 特征，query 容量不足。
2. sequence modeling 和 heterogeneous interaction 之间是晚融合，交互太浅。
3. 扩大计算预算时，新增容量更多花在孤立的 sequence encoder 或 token mixer 上，而不是联合表示上。

HyFormer 的基本做法，是引入一组 Global Tokens 作为序列和异构特征之间的共享语义接口，并把建模过程写成每层交替执行的两个模块。

1. Query Decoding：用 Global Tokens 去各条行为序列上做 cross-attention 解码。
2. Query Boosting：把解码后的 Global Tokens 与 NS tokens 一起做 mixer 式交互，再得到更强的下一层 Global Tokens。

因此，HyFormer 不是把 sequence 与 feature interaction 合成一个单一 self-attention 块，而是把两者做成一个交替迭代的 hybrid stack。

## 2. Input Tokenization 与 Query Generation

### 2.1 Non-Sequential Tokens

论文在 Query Generation 之前先把非序列特征组织成语义明确的 NS tokens。与 OneTrans 默认偏好的 Auto-Split 不同，HyFormer 明确写到在它的场景里采用 semantic grouping，因为用户、上下文、候选物品、交叉特征等语义边界清晰，语义分组有助于 inductive bias 和可解释性。

### 2.2 Initial Query Generation

初始查询不是只靠 target item 生成。论文定义的 Global Info 由两部分拼接而成。

1. 全部 NS feature vectors。
2. 各条行为序列的池化 summary。

公式可以概括成：

Q = [FFN_1(GlobalInfo), ..., FFN_N(GlobalInfo)]

其中 N 是 query token 数量。多序列场景下，每条序列有自己的一组 query tokens。深层不再重新从头用 MLP 生成 query，而是直接复用上一层 boosting 后得到的 query 表示。

## 3. Query Decoding

Query Decoding 负责让全局异构特征直接去读取长行为序列的 layer-wise 表示。论文允许多种 sequence encoding 方式：

1. Full Transformer Encoding。
2. LONGER-style Efficient Encoding。
3. Decoder-style Lightweight Encoding。

无论采用哪种序列编码方式，每一层都会得到对应的 K/V 表示，然后用 sequence-specific queries 做 cross-attention：

Q_decoded^(l) = CrossAttn(Q^(l-1), K^(l), V^(l))

这样 Global Tokens 在每层都能直接吸收新的 sequence-aware 信息，而不是只在最顶层读一次序列压缩结果。

## 4. Query Boosting

解码后的 queries 已经有 sequence-aware 语义，但论文认为它们和 heterogeneous NS tokens 的交互仍然不够，因此需要 Query Boosting。

Boosting 的输入是：

Q_union = [Q_decoded^(l), NS Tokens]

然后应用 RankMixer/MLP-Mixer 风格的 token mixing。论文的写法是把每个 token 沿 channel 维拆成 T 个子空间，再按子空间编号收集所有 token 的对应子向量，拼成新的 mixed tokens，之后接一个 per-token FFN，并加残差：

Q_boost = Q_union + PerTokenFFN(TokenMix(Q_union))

它的作用不是替代 sequence decoding，而是在序列解码之后显式加强：

1. query 与 query 的交互。
2. query 与 NS token 的交互。
3. 多序列查询之间的跨序列信息交换。

## 5. Multi-Sequence Modeling

HyFormer 在多序列问题上的立场很明确：不要像 MTGR / OneTrans 那样把不同序列粗暴 merge 成一个统一流。论文认为不同序列的 feature space 和语义往往不同，merge 会损伤表现。

因此它采用：

1. 每条序列独立编码。
2. 每条序列独立 query decoding。
3. 跨序列交互延后到 query boosting 阶段，通过 query-level token mixing 完成。

这也是 HyFormer 和 OneTrans 最本质的结构差异之一。

## 6. 训练与部署优化

论文还包含两类系统优化。

1. GPU Pooling for Long Sequence：在 GPU 侧重建压缩存储的长序列特征，减少传输和 host memory 压力。
2. Asynchronous AllReduce：把梯度同步与下一步计算重叠。

这些属于工业训练/部署侧优化，不是 HyFormer block 的最小结构定义。

## 7. 当前仓库实现

当前实现位于 config/gen/hyformer，并保持数据、模型、损失私有放在实验包内。

### 7.1 已实现的论文主线

1. Semantic-grouped NS tokens：当前代码显式构造 user、context、candidate、candidate-post、candidate-author，再补 dense-group tokens，而不是走 Auto-Split。
2. Multi-sequence 独立建模：序列侧不是 merge 成一条流，而是根据 history_group_ids 把各条行为序列拆开分别编码。
3. Query Generation：初始 queries 由全部 NS tokens 和各序列池化 summary 共同生成，并采用 sequence-specific 的 query MLP。
4. Query Decoding：每层对每条序列独立做 cross-attention 解码。
5. Query Boosting：把 decoded queries 与 NS tokens 拼起来，使用固定 token-count 的 mixer 式 boosting，再把前半部分切回下一层 global queries。
6. 论文里的 13 NS tokens + 3 global tokens 设计已被当前默认配置复现，总 mixer token 数为 16。

### 7.2 当前实现的关键取舍

1. 当前 sequence encoding 默认落的是 Full Transformer 风格的 per-sequence encoder，没有再实现 LONGER-style 和 decoder-style 两个变体。
2. 当前序列事件表示使用仓库现有的 history、post、author、action、time-gap 私有哈希特征，而不是论文工业系统里的原始特征 schema。
3. Query Boosting 采用论文同构的子空间重排思路；当 hidden dim 不能被 token 数整除时，代码会先投影到最近的可整除维度再做 mixing，以保证实现稳定。

### 7.3 当前与论文仍有差距的部分

1. 没有接入 GPU pooling 的长序列重建与稀疏去重算子。
2. 没有接入 Asynchronous AllReduce。
3. 没有实现 LONGER-style Efficient Encoding 和 Decoder-style Lightweight Encoding 两种 sequence encoder 分支。
4. 当前默认规模是仓库内 smoke / regression 友好的小规模配置，不是论文线上亿级参数与 64 GPU 训练设置。
